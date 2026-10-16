"""Request-scoped reuse for complete Qlib backtests, never shared account state."""

from __future__ import annotations

import gc
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

from qlib_yahoo.dataset import DatasetError


def check_cancel(cancel_file):
    if cancel_file is not None and Path(cancel_file).is_file():
        raise KeyboardInterrupt("cancelled by the user")


class BacktestPreparation:
    """Keep one verified window's quote objects and bounded immutable feature data.

    The runner still hashes the actual dataset and universe bytes for EACH run.
    A changed identity invalidates everything. There is no persistent disk cache,
    no mtime trust, and no position/order/strategy/executor reuse.
    """

    def __init__(self, cache_bytes=128 * 1024**2):
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise DatasetError("preparation cache_bytes must be a nonnegative integer")
        self.cache_bytes = cache_bytes
        self.identity = None
        self.initialized = self.factor_validated = False
        self.exchange = None
        self.eligible = None
        self._frames = OrderedDict()
        self._bytes = 0
        self.hits = self.misses = 0

    def bind(self, identity):
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        if digest != self.identity:
            self.clear()
            self.identity = digest

    def clear(self):
        self._frames.clear()
        self._bytes = 0
        self.initialized = self.factor_validated = False
        self.exchange = self.eligible = None
        self.identity = None

    def initialize(self, performance, initializer, **options):
        if self.initialized:
            self.hits += 1
            print("Reused verified Qlib initialization", flush=True)
            return
        performance.call("Qlib initialization", initializer, **options)
        self.initialized = True

    def features(self, performance, stage, provider, symbols, fields, **options):
        key = (tuple(symbols), tuple(fields), tuple(sorted((name, str(value)) for name, value in options.items())))
        if key in self._frames:
            self.hits += 1
            self._frames.move_to_end(key)
            print(f"Reused verified {stage}", flush=True)
            return self._frames[key].copy(deep=True)
        self.misses += 1
        value = performance.call(stage, provider, symbols, fields, **options)
        size = int(value.memory_usage(index=True, deep=True).sum())
        if size <= self.cache_bytes:
            while self._frames and self._bytes + size > self.cache_bytes:
                _, removed = self._frames.popitem(last=False)
                self._bytes -= int(removed.memory_usage(index=True, deep=True).sum())
            self._frames[key] = value.copy(deep=True)
            self._bytes += size
        return value

    def quote_exchange(self, performance, factory, **options):
        if self.exchange is not None:
            self.hits += 1
            print("Reused verified exchange quote preparation (fresh account)", flush=True)
            return self.exchange
        self.exchange = performance.call("exchange quote preparation", factory, **options)
        return self.exchange

    def snapshot(self):
        return {"scope": "one sequential batch; immutable preparation only",
                "identity_sha256": self.identity, "dataset_byte_verification": "every run",
                "hits": self.hits, "misses": self.misses, "cached_frame_bytes": self._bytes,
                "frame_cache_limit_bytes": self.cache_bytes, "fresh_account_per_run": True}


def run_backtest_batch(dataset: Path, alphas, *, data_root: Path | None = None, **settings) -> Path:
    """Run independent alphas in one worker, then write a normal workbench receipt."""
    from datetime import datetime, timezone
    from qlib_yahoo.backtest import _atomic_json, run_backtest
    from qlib_yahoo.expressions import check_alpha_syntax, sec_selection

    if not isinstance(alphas, (list, tuple)) or not 1 <= len(alphas) <= 200:
        raise DatasetError("batch requires between 1 and 200 expressions")
    for alpha in alphas:
        if not isinstance(alpha, str) or not alpha.strip():
            raise DatasetError("batch expressions must be nonempty strings")
        _catalog, bindings = sec_selection([alpha], settings.get("sec"))
        if bindings:
            check_alpha_syntax(alpha, extra_features=bindings)
        else:
            check_alpha_syntax(alpha)
    if "alpha" in settings or "_preparation" in settings:
        raise DatasetError("batch settings must not contain alpha or private preparation")
    dataset = Path(dataset).resolve()
    data_root = Path(data_root).resolve() if data_root else dataset.parent.parent
    folder = data_root / "workbench" / "batches"
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / f"backtests-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": str(dataset),
              "alphas": list(alphas), "runs": [], "errors": [],
              "settings": {k: v for k, v in settings.items() if not k.startswith("_")}, "status": "running"}
    preparation = BacktestPreparation()
    try:
        _atomic_json(receipt, record)
        print("__QUANT_JOB_ACTIVE_BATCH__=" + json.dumps(str(receipt)), flush=True)
        for index, alpha in enumerate(alphas, 1):
            check_cancel(settings.get("_cancel_file"))
            print(f"[{index}/{len(alphas)}] {alpha}", flush=True)
            try:
                run = run_backtest(dataset, alpha, _preparation=preparation, **settings)
                record["runs"].append(str(run))
            except Exception as error:
                record["errors"].append({"alpha": alpha, "error": f"{type(error).__name__}: {error}"})
                print(f"Error in [{index}/{len(alphas)}]: {error}", flush=True)
            finally:
                # Account/strategy infrastructure contains cycles; release each complete
                # independent trial before allocating its successor's weights/history.
                gc.collect()
            record["preparation"] = preparation.snapshot()
            _atomic_json(receipt, record)
        record["status"] = "completed" if record["runs"] else "failed"
        _atomic_json(receipt, record)
        if not record["runs"]:
            raise DatasetError(f"all {len(alphas)} backtests failed; first error: {record['errors'][0]['error']}")
        return receipt
    except BaseException as error:
        record["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        _atomic_json(receipt, record)
        raise
    finally:
        preparation.clear()
        gc.collect()
