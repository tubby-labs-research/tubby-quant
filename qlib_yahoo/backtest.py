"""Run a bounded Qlib daily simulation and seal its replayable net ledger."""

from __future__ import annotations

import json
import gzip
import io
import math
import os
import sys
import tempfile
import uuid
import webbrowser
import warnings
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

from qlib_yahoo import QLIB_COMMIT, QLIB_RELEASE
from qlib_yahoo.dataset import DatasetError, sha256
from qlib_yahoo.dataset_integrity import verify_dataset
from qlib_yahoo.expressions import validate_alpha


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        raise DatasetError("cannot write a redirected JSON record")
    temporary = None
    try:
        # Exclusive unique creation does not follow a pre-existing .tmp symlink.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=f".{path.name}.",
                                         suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _versions() -> dict[str, str]:
    result = {}
    for name in ("pyqlib", "numpy", "pandas", "yahooquery", "plotly", "statsmodels"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def _positions_frame(positions: Any) -> Any:
    import pandas as pd

    rows = []
    for timestamp, position in positions.items():
        cash, account = float(position.get_cash()), float(position.calculate_value())
        stocks = sorted(position.get_stock_list())
        for instrument in stocks:
            amount = float(position.get_stock_amount(instrument))
            price = float(position.get_stock_price(instrument))
            value = amount * price
            rows.append((timestamp, instrument, amount, price, value, value / account, cash, account))
        if not stocks:
            rows.append((timestamp, "", 0.0, None, 0.0, 0.0, cash, account))
    return pd.DataFrame(rows, columns=("datetime", "instrument", "amount", "price", "value", "weight", "cash", "account"))


def _check_ledger(report: Any, initial_cash: float) -> Any:
    import numpy as np

    required = ("account", "return", "cost", "turnover", "cash", "value", "total_cost")
    if report.empty or any(name not in report for name in required):
        raise DatasetError("Qlib returned an empty or incomplete daily ledger")
    if not np.isfinite(report[list(required)].to_numpy(dtype=float)).all():
        raise DatasetError("Qlib returned non-finite daily ledger values")
    if (report["account"] <= 0).any() or float(report["turnover"].sum()) <= 0:
        raise DatasetError("Qlib account invalid or no orders executed")
    net = report["return"] - report["cost"]
    previous = report["account"].shift(1)
    previous.iloc[0] = initial_cash
    if not np.allclose(net, report["account"] / previous - 1, rtol=1e-7, atol=1e-9):
        raise DatasetError("Qlib return/cost does not reconcile with account changes")
    if not np.allclose(report["account"], report["cash"] + report["value"], rtol=1e-7, atol=1e-6):
        raise DatasetError("Qlib cash and holdings do not reconcile with account")
    return net


def _write_table(frame: Any, path: Path) -> None:
    """Per-instrument tables are Parquet (zstd): read about 100x faster than gzip CSV at TOP3000 scale."""
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)


def run_backtest(
    dataset: Path, alpha: str, start: str | None = None, end: str | None = None,
    topk: int = 5, n_drop: int = 1, initial_cash: float = 100_000,
    cost_per_side: float = 0.0005, open_report: bool = False,
    *, annual_cash_rate: float = 0.0, warmup_sessions: int = 252,
    risk_degree: float = 0.95, report_level: str = "summary", holdings: str = "auto",
    threshold: float = 1.3, min_observations: int = 252,
    strategy_mode: str = "long-only", gross_exposure: float = 1.0,
    annual_borrow_rate: float = 0.004, universe: str | None = None, signal_transform: str = "raw",
    execution_profile: str = "fast",
    history_mode: str | None = None,
    runtime_profile: str = "auto",
    sec: dict | None = None,
    _preparation=None,
    _cancel_file: str | None = None,
    _signed_pin_path: str | None = None,
    _ordered_session: bool = False,
    _compiler_path: str | None = None,
    _compiled_targets: bool = False,
    _columnar_history: bool = False,
    _flat_indicators: bool = False,
    _buffered_history: bool = False,
    _decision_gzip_level: int | None = None,
    _gzip_path: str | None = None,
    _isal_level: int | None = None,
    _index_reuse: bool = False,
    _numeric_projection: bool = False,
    _row_range_manifest: str | None = None,
    _installed_range_root: str | None = None,
) -> Path:
    # Retain only a rejection boundary for archived V4 requests.
    if _numeric_projection is not False:
        raise DatasetError("numeric projection (V4) is unsupported: the experiment was retired")
    from qlib_yahoo.signed import SIGNAL_TRANSFORMS, WEIGHT_FORMULAS
    from qlib_yahoo.performance import RunPerformance

    performance = RunPerformance()
    from qlib_yahoo.expressions import check_alpha_syntax
    from qlib_yahoo.sec_resolution import resolve_config

    try:
        sec = (performance.call("SEC resolution and selected verification", resolve_config,
                                [alpha], sec, dialect="qlib", verify=True) if sec else
               resolve_config([alpha], sec, dialect="qlib", verify=True))
    except (ValueError, TypeError, KeyError) as error:
        raise DatasetError(str(error)) from error
    sec_catalog = Path(sec["catalog"]) if sec else None
    sec_bindings = sec["bindings"] if sec else {}
    registry = sec.get("registry") if sec and sec.get("mode") == "dictionary" else None
    if sec_bindings:
        check_alpha_syntax(alpha, extra_features=sec_bindings)
    sec_handle = sec_identity = None
    print(f"Backtest worker PID {os.getpid()} (memory below refers to this worker, not the window)", flush=True)

    if strategy_mode not in ("long-only", "signed"):
        raise DatasetError("strategy_mode must be long-only or signed")
    if (type(_ordered_session) is not bool or type(_compiled_targets) is not bool or type(_columnar_history) is not bool
            or type(_flat_indicators) is not bool
            or type(_buffered_history) is not bool
            or type(_index_reuse) is not bool
            or (_compiled_targets and not _ordered_session)
            or (_ordered_session and (strategy_mode != "signed" or execution_profile != "fast" or not _compiler_path))
            or (_compiler_path is not None and not _ordered_session)
            or (_signed_pin_path is not None and strategy_mode != "signed")):
        raise DatasetError("experimental ordered sessions require signed fast mode and an explicit compiler path")
    if execution_profile not in ("fast", "reference"):
        raise DatasetError("execution_profile must be fast or reference")
    if runtime_profile not in ("auto", "native"):
        raise DatasetError("runtime_profile must be auto or native")
    history_mode = history_mode or ("stream" if execution_profile == "fast" and strategy_mode == "signed" else "memory")
    if history_mode not in ("stream", "memory") or (history_mode == "stream" and strategy_mode != "signed"):
        raise DatasetError("history_mode must be memory, or stream for signed runs")
    if _columnar_history and (strategy_mode != "signed" or history_mode != "stream"):
        raise DatasetError("experimental columnar history requires signed stream mode")
    if _flat_indicators and (strategy_mode != "signed" or history_mode != "stream" or execution_profile != "fast"):
        raise DatasetError("experimental flat indicators require signed fast stream mode")
    if _buffered_history and (strategy_mode != "signed" or history_mode != "stream"):
        raise DatasetError("experimental buffered history requires signed stream mode")
    if _index_reuse and (strategy_mode != "signed" or execution_profile != "fast"):
        raise DatasetError("experimental date-index reuse requires signed fast mode")
    if _row_range_manifest is not None and (
            type(_row_range_manifest) is not str or not _row_range_manifest
            or strategy_mode != "signed" or execution_profile != "fast"):
        raise DatasetError("experimental date-row comparison requires signed fast mode and an explicit build receipt path")
    if _installed_range_root is not None and (
            type(_installed_range_root) is not str or not _installed_range_root
            or strategy_mode != "signed" or execution_profile != "fast"
            or _row_range_manifest is not None):
        raise DatasetError("installed date-range helper requires signed fast mode and cannot combine with a prototype receipt")
    if _decision_gzip_level is not None and (
            type(_decision_gzip_level) is not int or not 1 <= _decision_gzip_level <= 9
            or strategy_mode != "signed" or history_mode != "stream"):
        raise DatasetError("experimental decision gzip level requires signed stream mode and integer 1 through 9")
    if (_gzip_path is not None and (strategy_mode != "signed" or history_mode != "stream"
                                   or _decision_gzip_level is not None)
            or _isal_level is not None and (type(_isal_level) is not int or not 0 <= _isal_level <= 3
                                           or _gzip_path is None)):
        raise DatasetError("experimental ISA-L requires signed stream mode, an isolated path, optional level 0 through 3, and no zlib level")
    if signal_transform not in SIGNAL_TRANSFORMS:
        raise DatasetError(f"signal_transform must be one of {', '.join(SIGNAL_TRANSFORMS)}")
    if strategy_mode == "long-only" and signal_transform != "raw":
        raise DatasetError("signal transforms apply to signed weights; long-only Top-k already ranks the scores")
    if universe is not None and (not isinstance(universe, str) or not universe or universe == "all"):
        raise DatasetError("universe must name a dated universe recorded by the dataset, or be omitted")
    if (isinstance(gross_exposure, (bool, str, bytes, complex))
            or not isinstance(gross_exposure, (int, float)) or not math.isfinite(gross_exposure)
            or not 0 <= gross_exposure <= 1
            or isinstance(annual_borrow_rate, (bool, str, bytes, complex))
            or not isinstance(annual_borrow_rate, (int, float)) or not math.isfinite(annual_borrow_rate)
            or annual_borrow_rate < 0):
        raise DatasetError("invalid signed exposure or borrow rate")
    if sys.version_info >= (3, 13):
        raise DatasetError("Qlib 0.9.7 needs Python 3.8-3.12")
    if ((strategy_mode == "long-only" and (type(topk) is not int or type(n_drop) is not int or topk < 1 or not 0 <= n_drop <= topk))
            or not math.isfinite(initial_cash) or initial_cash <= 0
            or not math.isfinite(cost_per_side) or cost_per_side < 0
            or not math.isfinite(annual_cash_rate) or annual_cash_rate <= -1
            or type(warmup_sessions) is not int or warmup_sessions < 1
            or not math.isfinite(risk_degree) or not 0 < risk_degree <= 1):
        raise DatasetError("invalid portfolio, cost, cash-rate or warmup setting")
    if (report_level not in ("summary", "overview", "detailed")
            or holdings not in ("auto", "always", "never")
            or isinstance(threshold, (str, bytes, bool, complex))
            or not isinstance(threshold, (int, float)) or not math.isfinite(threshold)
            or type(min_observations) is not int or min_observations < 1):
        raise DatasetError("invalid report level, holdings gate, threshold or observation count")
    signed_identity = None
    profile_identity = {"requested": runtime_profile, "selection": "native",
                        "reason": "long_only_or_reference_or_memory"}
    # Archived probes retain their explicit option sets. Normal console, CLI,
    # notebook and batch callers select the accepted platform recipe here,
    # before importing Qlib. Missing/corrupt Windows dependencies fail clearly;
    # they never silently turn a signed trial into long-only or a slower recipe.
    explicit_options = any((
        _signed_pin_path is not None, _ordered_session, _compiler_path is not None,
        _compiled_targets, _columnar_history, _flat_indicators, _buffered_history,
        _decision_gzip_level is not None, _gzip_path is not None, _isal_level is not None,
        _index_reuse, _row_range_manifest is not None,
        _installed_range_root is not None,
    ))
    if runtime_profile == "native":
        profile_identity["reason"] = "explicit_native_profile"
    elif explicit_options:
        profile_identity.update(selection="explicit_options", reason="preserve_explicit_experiment")
    elif strategy_mode == "signed" and execution_profile == "fast" and history_mode == "stream":
        # Optional packages are verified against wheel source bytes, so do not
        # create (or later accept) an unrecorded bytecode cache in their trees.
        # The console executes this in a fresh isolated worker.
        sys.dont_write_bytecode = True
        from qlib_yahoo.fast_runtime import resolve_fast_profile
        accepted_options, profile_identity = performance.call(
            "accepted runtime selection", resolve_fast_profile)
        _ordered_session = accepted_options.get("_ordered_session", False)
        _compiler_path = accepted_options.get("_compiler_path")
        _columnar_history = accepted_options.get("_columnar_history", False)
        _flat_indicators = accepted_options.get("_flat_indicators", False)
        _buffered_history = accepted_options.get("_buffered_history", False)
        _index_reuse = accepted_options.get("_index_reuse", False)
        _gzip_path = accepted_options.get("_gzip_path")
        _isal_level = accepted_options.get("_isal_level")
        _installed_range_root = accepted_options.get("_installed_range_root")
    if strategy_mode == "signed":
        from qlib_yahoo.runtime import ensure_signed_runtime
        signed_identity = performance.call("runtime verification", ensure_signed_runtime) if _signed_pin_path is None else (
            performance.call("runtime verification", ensure_signed_runtime, pin_path=Path(_signed_pin_path)))
    required_history = (validate_alpha(alpha, extra_features=sec_bindings) if sec_bindings else validate_alpha(alpha))
    if warmup_sessions < required_history + 1:
        raise DatasetError(f"warmup needs at least {required_history + 1} sessions for alpha history and prior-close signal")

    import numpy as np
    import pandas as pd
    import qlib
    from qlib import config as qlib_config
    from qlib.backtest import backtest as qlib_backtest
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.backtest.position import BasePosition
    from qlib.constant import REG_US
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.data import D
    from qlib_yahoo.metrics import STRATEGY_METRIC_VERSION, portfolio_metrics
    compiler_identity = None
    gzip_identity = gzip_factory = None
    if _gzip_path is not None:
        from qlib_yahoo.fast_gzip import activate_gzip
        gzip_factory, gzip_identity = performance.call(
            "decision gzip activation", activate_gzip, _gzip_path, 2 if _isal_level is None else _isal_level)
    order_batch = None
    range_validator = range_identity = None
    if _row_range_manifest is not None or _installed_range_root is not None:
        from qlib_yahoo.fast_signed import SUPPORTED_COMMITS
        if signed_identity["commit"] not in SUPPORTED_COMMITS:
            raise DatasetError("date-row comparison experiment requires a reviewed signed engine")
        if _installed_range_root is not None:
            from qlib_yahoo.range_runtime import activate_installed_range
            range_validator, range_identity = performance.call(
                "date-row compiler activation", activate_installed_range, _installed_range_root)
        else:
            from qlib_yahoo.range_runtime import activate_range
            range_validator, range_identity = performance.call(
                "date-row compiler activation", activate_range, _row_range_manifest)
    if _flat_indicators:
        from qlib_yahoo.fast_signed import SUPPORTED_COMMITS
        if signed_identity["commit"] not in SUPPORTED_COMMITS:
            raise DatasetError("flat-indicator experiment requires a reviewed signed engine")
    if _ordered_session:
        if signed_identity["commit"] != "8ab6249aa6143cb0675259d5ff986d294e918ef4":
            raise DatasetError("ordered-session experiment requires the explicitly reviewed tubby.4 hook build")
        from qlib_yahoo.ordered_signed import activate_compiler
        compiler_identity = performance.call("ordered compiler activation", activate_compiler, _compiler_path)
    quote_options = {}
    if execution_profile == "fast":
        from qlib_yahoo.daily_quote import DailyQuote
        if _index_reuse:
            from functools import partial
            construction_options = {"_index_reuse": True}
            quote_options["quote_cls"] = partial(DailyQuote, **construction_options)
        else:
            quote_options["quote_cls"] = DailyQuote

    dataset = Path(dataset).absolute()
    dataset_manifest_path = dataset / "manifest.json"
    try:
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetError(f"cannot read dataset manifest {dataset_manifest_path}: {error}") from error
    if not isinstance(dataset_manifest, dict):
        raise DatasetError("dataset manifest must be a JSON object")
    if "storage" in dataset_manifest:
        from qlib_yahoo.packed import _reject_redirects
        _reject_redirects(dataset_manifest_path)
    dataset = dataset.resolve()
    dataset_manifest_path = dataset / "manifest.json"
    symbols = dataset_manifest.get("symbols")
    if not isinstance(symbols, list) or not symbols or not all(isinstance(s, str) for s in symbols):
        raise DatasetError("dataset manifest needs a nonempty symbols list")
    if strategy_mode == "long-only" and topk >= len(symbols):
        raise DatasetError("topk must be smaller than the trade universe")
    print(f"Verifying dataset {dataset.name} ...", flush=True)
    integrity = performance.call("dataset verification", verify_dataset, dataset, dataset_manifest)
    spans = universe_record = None
    if universe is not None:
        from qlib_yahoo.universe import load_dataset_universe, mask_to_members, members_between
        spans, universe_record = load_dataset_universe(dataset, dataset_manifest, universe)
        if int(universe_record.get("min_history_sessions", 0)) < required_history + 1:
            raise DatasetError(f"universe {universe} requires {universe_record.get('min_history_sessions')} "
                               f"history sessions; the alpha needs {required_history + 1}")
    qlib_tree_hash = integrity["qlib_binary_tree_sha256"]
    qlib_file_count = integrity["qlib_binary_file_count"]
    if sec_bindings:
        from qlib_yahoo.sec_features import _verify_catalog, feature_identity, materialize

        performance.call("SEC source verification", _verify_catalog, sec_catalog, force=True)
        sec_identity = feature_identity(sec_catalog, sec_bindings)
        sec_dates = np.asarray((dataset / "qlib_data/calendars/day.txt").read_text(encoding="utf-8").split(),
                               dtype="datetime64[D]")
        # Selected-panel cache keys include the complete binding set, dates and
        # symbol order; subsets/supersets do not reuse individual cached fields.
        sec_handle = performance.call("SEC selected fields", materialize, sec_catalog, sec_bindings,
                                      sec_dates, symbols, dataset.parent.parent / "cache" / "sec",
                                      progress=lambda message: print(message, flush=True),
                                      force_verify=False)
        if feature_identity(sec_catalog, sec_bindings) != sec_identity:
            raise DatasetError("SEC source changed while materializing selected fields")
    if _preparation is not None:
        _preparation.bind({
            "dataset": str(dataset), "manifest_sha256": sha256(dataset_manifest_path), "integrity": integrity,
            "runtime": signed_identity if strategy_mode == "signed" else _versions().get("pyqlib"),
            "source": {name: sha256(Path(__file__).with_name(name)) for name in (
                "backtest.py", "preparation.py", "expressions.py", "daily_quote.py", "fast_signed.py",
                "streaming.py", "packed_provider.py", "array_dataset_provider.py", "quote_construction.py")
                + (("ordered_signed.py", "ordered_signed_kernel.py") if _ordered_session else ())
                + (("target_session.py", "target_session_kernel.py") if _compiled_targets else ())
                + (("streaming_columns.py",) if _columnar_history else ())
                + (("flat_order_indicator.py",) if _flat_indicators else ())
                + (("parquet_buffer.py",) if _buffered_history else ())
                + (("fast_gzip.py",) if _gzip_path is not None else ())
                + (("range_runtime.py", "session_loops.pyx", "session_range_buffer.pyx")
                   if _row_range_manifest is not None or _installed_range_root is not None else ())
                + (("fast_runtime.py",) if profile_identity.get("name") == "accepted-v1" else ())
                + (("quote_construction_v2.py", "quote_construction_v3.py") if _index_reuse else ())
                + (("sec_catalog.py", "sec_features.py", "sec_provider.py") if sec_handle is not None else ())
                + (("sec_resolution.py", "sec_registry.py") if registry else ())},
            "window": [start, end, warmup_sessions, universe], "mode": strategy_mode,
            "exchange": [execution_profile, cost_per_side],
            "ordered_session_compiler": compiler_identity,
            "compiled_targets": _compiled_targets, "columnar_history": _columnar_history,
            "flat_indicators": _flat_indicators,
            "buffered_history": _buffered_history,
            "decision_gzip_level": _decision_gzip_level,
            "decision_gzip_backend": gzip_identity,
            "date_index_reuse": _index_reuse,
            "date_row_comparison": range_identity,
            "runtime_profile": profile_identity,
            **({"sec": {"source": sec_identity, "materialized": sec_handle["identity"],
                        **({"registry": registry} if registry else {})}}
               if sec_handle is not None else {}),
        })
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    provider_options = {}
    array_preparation = False
    if strategy_mode == "signed" and execution_profile == "fast":
        from qlib_yahoo.fast_signed import SUPPORTED_COMMITS
        if signed_identity["commit"] in SUPPORTED_COMMITS:
            provider_options["dataset_provider"] = {
                "class": "ArrayDatasetProvider", "module_path": "qlib_yahoo.array_dataset_provider",
            }
            array_preparation = True
    if dataset_manifest.get("storage"):
        from qlib_yahoo.packed import feature_provider_config
        provider_options["feature_provider"] = feature_provider_config(dataset, dataset_manifest)
    if sec_handle is not None:
        provider_options["feature_provider"] = {
            "class": "SECFeatureProvider", "module_path": "qlib_yahoo.sec_provider",
            "kwargs": {"handle": sec_handle, "price_provider": provider_options.get("feature_provider")},
        }
    initialize_options = dict(provider_uri=str(dataset / "qlib_data"), region=REG_US, kernels=1,
                              dataset_cache=None, expression_cache=None, **provider_options)
    if _preparation is None:
        performance.call("Qlib initialization", qlib.init, **initialize_options)
    else:
        _preparation.initialize(performance, qlib.init, **initialize_options)
    effective_limit = qlib_config.C.limit_threshold
    if effective_limit is not None:
        raise DatasetError(f"REG_US active limit_threshold is {effective_limit!r}; this smoke contract requires None")
    sessions = pd.DatetimeIndex(D.calendar(freq="day"))
    if len(sessions) < warmup_sessions + 3:
        raise DatasetError("dataset has too few calendar sessions")
    first_member_index = None
    if spans is not None:
        first_member_day = min(begin for intervals in spans.values() for begin, _ in intervals)
        first_member_index = sessions.get_indexer([first_member_day])[0]
        if first_member_index < 0 or first_member_index + 1 >= len(sessions) - 2:
            raise DatasetError(f"universe {universe} has no membership early enough for a backtest")
    # The first execution session consumes the previous close's score, so a
    # membership effective on session i can first be used at session i + 1.
    default_start_index = (max(warmup_sessions, first_member_index + 1)
                           if first_member_index is not None else warmup_sessions)
    start_day = pd.Timestamp(start) if start else sessions[default_start_index]
    end_day = pd.Timestamp(end) if end else sessions[-2]
    if start_day not in sessions or end_day not in sessions or start_day >= end_day or end_day >= sessions[-1]:
        raise DatasetError("start/end must be ordered sessions with a later boundary session")
    start_index = sessions.get_loc(start_day)
    if start_index < warmup_sessions:
        raise DatasetError(f"start requires {warmup_sessions} preceding sessions")
    if first_member_index is not None and start_index <= first_member_index:
        raise DatasetError(f"universe {universe} first has members on {sessions[first_member_index]:%Y-%m-%d}; "
                           f"earliest backtest start is {sessions[first_member_index + 1]:%Y-%m-%d} "
                           "because signals use the prior session")
    warmup_start = sessions[start_index - warmup_sessions]
    signal_start = sessions[start_index - 1]
    signal_end = sessions[sessions.get_loc(end_day) - 1]
    if spans is None:
        if _preparation is not None and _preparation.eligible is not None:
            eligible = list(_preparation.eligible)
        else:
            warmup_close = performance.call("warm-up eligibility", D.features, symbols, ["$close"], start_time=warmup_start,
                                      end_time=signal_start, freq="day", disk_cache=0)
            counts = warmup_close["$close"].groupby(level="instrument").count()
            eligible = [symbol for symbol in symbols if int(counts.get(symbol, 0)) >= warmup_sessions]
            if _preparation is not None:
                _preparation.eligible = tuple(eligible)
            del warmup_close, counts
    else:
        # Dated membership replaces the fixed warm-up universe: an instrument is
        # tradable only while a ranking made before the signal session includes it.
        eligible = members_between(spans, signal_start, signal_end)
    if not eligible or (strategy_mode == "long-only" and len(eligible) <= topk):
        raise DatasetError("too few symbols have complete warmup close history" if spans is None
                           else f"universe {universe} has too few members between {signal_start:%Y-%m-%d} and {signal_end:%Y-%m-%d}")
    if strategy_mode == "signed" and not (_preparation is not None and _preparation.factor_validated):
        factor = performance.call("factor validation data", D.features, eligible, ["$factor"], start_time=signal_start,
                            end_time=end_day, freq="day", disk_cache=0)
        expected_factor = pd.MultiIndex.from_product(
            [sessions[(sessions >= signal_start) & (sessions <= end_day)], eligible],
            names=["datetime", "instrument"],
        )
        observed_factor = factor["$factor"].reorder_levels(["datetime", "instrument"]).reindex(expected_factor)
        if spans is not None:
            # Dated members may lack bars outside their membership; every supplied factor must still be identity.
            present = observed_factor.dropna()
            if not np.isfinite(present.to_numpy(dtype=float)).all() or not (present == 1).all():
                raise DatasetError("signed mode requires factor=1 wherever a universe bar exists")
            del present
        elif not np.isfinite(observed_factor.to_numpy(dtype=float)).all() or not (observed_factor == 1).all():
            raise DatasetError("signed mode requires factor=1 for every eligible symbol and session")
        del factor, expected_factor, observed_factor
        if _preparation is not None:
            _preparation.factor_validated = True

    run_dir = dataset.parent.parent / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    print("__QUANT_JOB_ACTIVE_RUN__=" + json.dumps(str(run_dir)), flush=True)
    execution = {
        "score_timing": "session close; Qlib TopkDropoutStrategy shift=1",
        "execution": "next session open", "mark": "session close", "end": "unliquidated mark at close",
        "benchmark": None,
        "topk": topk, "n_drop": n_drop, "risk_degree": risk_degree,
        "zero_dropout_policy": "no sales; vacant topk slots may be filled" if n_drop == 0 else None,
        "method_sell": "random" if n_drop == 0 else "bottom", "method_buy": "top", "hold_thresh": 1,
        "only_tradable": True, "forbid_all_trade_at_limit": True,
        "settle_type": "ST_NO", "deal_price": "open", "trade_unit": 1,
        "limit_threshold": effective_limit, "volume_threshold": None,
        "open_cost": cost_per_side, "close_cost": cost_per_side,
        "min_cost": 0.0, "impact_cost": 0.0, "cash_interest": 0.0,
    }
    backtest_contract = {
        "start": start_day.strftime("%Y-%m-%d"), "end": end_day.strftime("%Y-%m-%d"),
        "cash": initial_cash, "warmup_sessions": warmup_sessions,
        "warmup_start": warmup_start.strftime("%Y-%m-%d"),
        "signal_start": signal_start.strftime("%Y-%m-%d"),
        "signal_end": signal_end.strftime("%Y-%m-%d"),
        "required_alpha_history": required_history, "trade_symbols": eligible,
        "strategy_mode": strategy_mode,
        "execution_profile": execution_profile,
        "execution_contract": execution,
    }
    if spans is not None:
        backtest_contract["universe"] = {
            "name": universe, "instruments_sha256": universe_record.get("instruments_sha256"),
            "rule": {key: universe_record.get(key) for key in (
                "size", "adv_sessions", "min_history_sessions", "ranking", "eligibility",
                "reconstitution", "effective", "proxy_for")},
            "members_in_window": len(eligible),
            "masking": "scores kept only for (session, instrument) inside a dated membership span",
            "exit_policy": ("long-only: held non-members have no score, rank last and are sold first within n_drop"
                            if strategy_mode == "long-only" else
                            "signed: instruments without a member weight get target zero and are closed"),
        }
    if strategy_mode == "signed":
        execution = {
            "score_timing": "session close; preceding session weights", "execution": "next session open",
            "mark": "session close", "end": "unliquidated mark at close",
            "benchmark": None,
            "deal_price": "open", "trade_unit": 1, "limit_threshold": effective_limit,
            "volume_threshold": None, "open_cost": cost_per_side, "close_cost": cost_per_side,
            "min_cost": 0.0, "impact_cost": 0.0, "cash_interest": 0.0,
            "gross_exposure": float(gross_exposure), "annual_borrow_rate": float(annual_borrow_rate),
            "borrow_day_count": "ACT/365", "restricted_short_proceeds": True,
            "signed_contract_version": 2,
            "signal_transform": signal_transform,
            "weight_formula": WEIGHT_FORMULAS[signal_transform],
            "weight_precision": "float32; conservative rounding at gross cap",
            "sizing": "previous-close equity after borrow fees / current open; whole shares toward zero",
            "neutralization": None if signal_transform == "raw" else "cross-sectional: weights sum to zero before gross scaling",
            "cash_borrowing": False,
            "short_borrow_availability": "all selected instruments; no recalls",
            "borrow_valuation": "previous-close short liability before current open fills",
            "margin_calls": False, "forced_liquidation": False,
            "settle_type": "ST_NO", "factor_contract": "identity only; no corporate-action ledger",
            "eligibility_exclusion_policy": (
                "exclude symbols with fewer than warmup_sessions valid close observations; then require every eligible symbol to have a finite score each signal session"
                if spans is None else
                "dated universe membership; members without a finite score that session get zero weight"),
            "excluded_symbols": sorted(set(symbols) - set(eligible)) if spans is None else [],
        }
        backtest_contract["execution_contract"] = execution
        backtest_contract["strategy_mode"] = "signed"
    source_dir = Path(__file__).resolve().parent
    manifest: dict[str, Any] = {
        "schema_version": 6, "strategy_mode": strategy_mode,
        "artifact_format": {"tables": "parquet (zstd)", "ledger": "daily_report.csv", "signed_decisions": "json.gz"},
        "status": "running", "stage": "initialized",
        "created_at_utc": _now(), "dataset": str(dataset),
        "dataset_manifest_sha256": sha256(dataset_manifest_path),
        "qlib_binary_tree_sha256": qlib_tree_hash,
        "qlib_binary_file_count": qlib_file_count,
        "dataset_integrity": integrity,
        "runner_source_sha256": {
            name: sha256(source_dir / name)
            for name in ("backtest.py", "dataset_integrity.py", "expressions.py", "analysis.py", "holding_series.py", "metrics.py", "dataset.py", "__main__.py", "performance.py", "packed.py", "packed_provider.py", "daily_quote.py", "quote_construction.py", "fast_account.py", "fast_signed.py", "streaming.py", "preparation.py", "array_dataset_provider.py")
            if (source_dir / name).is_file()
        },
        "python": sys.version.split()[0], "packages": _versions(),
        "qlib_reviewed_source": {"release": QLIB_RELEASE, "commit": QLIB_COMMIT,
                                  "installed_vcs_commit_verified": False},
        "alpha": alpha, "backtest": backtest_contract,
        "history_storage": {"mode": history_mode, "account_reset_at_boundaries": False,
                            "positions_order": "datetime,instrument"},
        "runtime_profile": profile_identity,
    }
    if sec_handle is not None:
        # Additive provenance: schema 6 tables and execution semantics are unchanged.
        manifest["sec"] = {"catalog": str(sec_catalog), "bindings": sec_bindings,
                           "source_identity": sec_identity, "materialized": sec_handle,
                           **({"mode": "dictionary", "registry": registry} if registry else {})}
        for name in (("sec_catalog.py", "sec_features.py", "sec_provider.py")
                     + (("sec_resolution.py", "sec_registry.py") if registry else ())):
            manifest["runner_source_sha256"][name] = sha256(source_dir / name)
    if strategy_mode == "signed":
        manifest["strategy_mode"] = "signed"
        manifest["signed_engine_identity"] = signed_identity
        manifest["data_preparation"] = {
            "dataset_provider": "ArrayDatasetProvider" if array_preparation else "native LocalDatasetProvider",
            "expression_evaluation": "unchanged native per-instrument ExpressionD.expression",
            "unsupported_layouts": "native assembly fallback; no repeated expression evaluation",
        }
        for name in ("signed.py", "runtime.py"):
            manifest["runner_source_sha256"][name] = sha256(source_dir / name)
        if profile_identity.get("name") == "accepted-v1":
            manifest["runner_source_sha256"]["fast_runtime.py"] = sha256(source_dir / "fast_runtime.py")
        if _ordered_session:
            for name in ("ordered_signed.py", "ordered_signed_kernel.py"):
                manifest["runner_source_sha256"][name] = sha256(source_dir / name)
            manifest["ordered_session_compiler"] = compiler_identity
        if _compiled_targets:
            for name in ("target_session.py", "target_session_kernel.py"):
                manifest["runner_source_sha256"][name] = sha256(source_dir / name)
        if _columnar_history:
            manifest["runner_source_sha256"]["streaming_columns.py"] = sha256(source_dir / "streaming_columns.py")
        if _flat_indicators:
            manifest["runner_source_sha256"]["flat_order_indicator.py"] = sha256(source_dir / "flat_order_indicator.py")
        if _buffered_history:
            manifest["runner_source_sha256"]["parquet_buffer.py"] = sha256(source_dir / "parquet_buffer.py")
        if gzip_identity is not None:
            manifest["runner_source_sha256"]["fast_gzip.py"] = sha256(source_dir / "fast_gzip.py")
            manifest["decision_gzip_backend"] = gzip_identity
        if _index_reuse:
            for name in ("quote_construction_v2.py", "quote_construction_v3.py"):
                manifest["runner_source_sha256"][name] = sha256(source_dir / name)
        if range_identity is not None:
            for name in ("range_runtime.py", "session_loops.pyx", "session_range_buffer.pyx"):
                manifest["runner_source_sha256"][name] = sha256(source_dir / name)
            manifest["date_row_comparison_runtime"] = range_identity
    if spans is not None:
        manifest["runner_source_sha256"]["universe.py"] = sha256(source_dir / "universe.py")
    manifest_path = run_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    history_writer = None
    try:
        manifest["stage"] = "scoring"
        _atomic_json(manifest_path, manifest)
        print(f"Scoring the alpha for {len(eligible):,} instruments from {warmup_start:%Y-%m-%d} ...", flush=True)
        feature_call = performance.call if _preparation is None else (
            lambda stage, provider, *args, **kwargs: _preparation.features(performance, stage, provider, *args, **kwargs))
        feature = feature_call("alpha evaluation", D.features, eligible, [alpha], start_time=warmup_start, end_time=signal_end,
                             freq="day", disk_cache=0)
        signal_days = sessions[(sessions >= signal_start) & (sessions <= signal_end)]
        if spans is not None:
            in_window = feature[alpha].reorder_levels(["datetime", "instrument"]).sort_index()
            in_window = mask_to_members(in_window.loc[in_window.index.get_level_values("datetime").isin(signal_days)], spans)
            scored = np.isfinite(in_window.to_numpy(dtype=float))
            per_day = pd.Series(scored, index=in_window.index).groupby(level="datetime").sum()
            per_day = per_day.reindex(signal_days, fill_value=0)
            manifest["universe_scoring"] = {
                "member_observations": int(len(in_window)), "unscored_member_observations": int((~scored).sum()),
                "min_scored_members": int(per_day.min()), "median_scored_members": float(per_day.median()),
                "max_scored_members": int(per_day.max()),
            }
            # Fewer scored members than top-k only means fewer holdings that day (recorded above);
            # a session with none cannot be traded at all.
            empty = per_day[per_day == 0]
            if len(empty):
                raise DatasetError(f"universe {universe} has no scored member on {empty.index[0]:%Y-%m-%d}")
            del in_window, scored, per_day, empty
        if strategy_mode == "signed":
            from qlib_yahoo.signed import normalize_scores
            scores, weights = performance.call("signed target weights", normalize_scores, feature, alpha, eligible, signal_days, gross_exposure, spans=spans,
                                               transform=signal_transform)
            daily_gross = weights.abs().groupby(level="datetime").sum()
            if (daily_gross > 1 + 1e-12).any():
                raise DatasetError(f"signed target normalization exceeds account exposure: {daily_gross.max()}")
            _write_table(weights.rename("weight").reset_index(), run_dir / "signed_weights.parquet")
        else:
            scores = (feature[[alpha]].replace([np.inf, -np.inf], np.nan).reset_index()
                      .dropna(subset=[alpha]).set_index(["datetime", "instrument"])[alpha].sort_index())
            if spans is not None:
                scores = mask_to_members(scores, spans)
            missing_days = signal_days.difference(scores.index.get_level_values("datetime").unique())
            if len(missing_days):
                raise DatasetError(f"alpha has no finite score for required session {missing_days[0]}")
            scores = scores.loc[scores.index.get_level_values("datetime").isin(signal_days)]
        _write_table(scores.rename("score").reset_index(), run_dir / "alpha_scores.parquet")
        del feature
        # Forward labels are diagnostics only; extraction/serialization cannot fail a simulation.
        label_path = run_dir / "signal_forward_labels.parquet"
        label_temp = run_dir / "signal_forward_labels.parquet.tmp"
        try:
            label = feature_call("diagnostic labels", D.features, eligible, ["Ref($open, -2)/Ref($open, -1)-1"],
                               start_time=signal_start, end_time=signal_end, freq="day", disk_cache=0)
            label.columns = ["label"]
            label = label.reorder_levels(["datetime", "instrument"]).sort_index()
            paired = pd.concat((scores.rename("score"), label), axis=1)
            _write_table(paired.reset_index(), label_temp)
            os.replace(label_temp, label_path)
            manifest["diagnostics"] = {"forward_labels": "available"}
            del label, paired
        except Exception as error:
            label_temp.unlink(missing_ok=True)
            manifest["diagnostics"] = {"forward_labels": "unavailable",
                                       "reason": f"{type(error).__name__}: {error}"}

        manifest["stage"] = "simulation"
        _atomic_json(manifest_path, manifest)
        print(f"Simulating {backtest_contract['start']} to {backtest_contract['end']} ({strategy_mode}) ...", flush=True)
        # Qlib 0.9.7 TopkDropoutStrategy.generate_trade_decision's bottom
        # selector does not honor zero: only_tradable=True's get_first_n
        # appends before testing n, while the other branch uses [-0:].
        # Its supported random seller selects zero stocks when n_drop=0;
        # top buying remains deterministic and fills any vacant slots.
        if strategy_mode == "signed":
            from qlib.backtest.signed_account import SignedAccount
            from qlib.backtest.signed_exchange import SignedExchange
            from qlib.contrib.strategy.signed_weight import SignedWeightStrategy
            if execution_profile == "fast":
                from qlib_yahoo.fast_account import account_class_for
                from qlib_yahoo.fast_signed import classes_for
                SignedAccount = account_class_for(signed_identity["commit"])
                SignedExchange, SignedWeightStrategy = classes_for(signed_identity["commit"])
            if _compiled_targets:
                # Load the optional adapter inside the ordinary worker boundary,
                # before simulation observers select its functions. First JIT
                # compilation still occurs inside the first target session.
                performance.call("compiled target adapter imports", import_module, "qlib_yahoo.target_session")
            strategy = SignedWeightStrategy(signal=weights, **({"compiled_targets": True} if _compiled_targets else {}))
            account_options = {}
            if history_mode == "stream":
                from qlib_yahoo.streaming import DECISION_GZIP_LEVEL, SignedHistoryWriter, streaming_account_class
                history_options = {}
                if _columnar_history:
                    history_options["columnar"] = True
                if _buffered_history:
                    history_options["buffered"] = True
                if _decision_gzip_level is not None:
                    history_options["decision_gzip_level"] = _decision_gzip_level
                if gzip_factory is not None:
                    history_options.update(_gzip_factory=gzip_factory,
                                           decision_gzip_level=gzip_identity["actual_level"])
                history_writer = SignedHistoryWriter(run_dir, strategy, **history_options)
                manifest["history_storage"]["decision_compression"] = {"codec": "gzip", "level": history_writer.decision_gzip_level}
                if gzip_identity is not None:
                    manifest["history_storage"]["decision_compression"].update(
                        backend=gzip_identity["backend"], level_equivalent_to_zlib=False)
                SignedAccount = streaming_account_class(SignedAccount, **({"flat_indicators": True} if _flat_indicators else {}))
                account_options["history_writer"] = history_writer
            account = SignedAccount(init_cash=initial_cash, annual_borrow_rate=annual_borrow_rate,
                                    benchmark_config={"benchmark": None}, **account_options)
            if _ordered_session:
                from qlib_yahoo.ordered_signed import OrderedSessionExecutor
                order_batch = OrderedSessionExecutor(account_type=SignedAccount)
            exchange_call = (lambda factory, **options: performance.call("exchange quote preparation", factory, **options)) if _preparation is None else (
                lambda factory, **options: _preparation.quote_exchange(performance, factory, **options))
            exchange = exchange_call(SignedExchange, start_time=start_day, end_time=end_day, codes=eligible,
                                      open_cost=cost_per_side, close_cost=cost_per_side,
                                      limit_threshold=effective_limit, trade_unit=1, **quote_options,
                                      **({"_range_validator": range_validator} if range_validator is not None else {}))
            if _index_reuse:
                manifest["data_preparation"]["date_index_reuse"] = exchange.quote.index_reuse_stats
            if execution_profile == "fast" and SignedExchange.__module__ == "qlib_yahoo.fast_signed":
                # A prepared exchange may be reused, but no quote/check state
                # from the preceding account is allowed to leak into a new run.
                exchange.reset_session_cache()
                manifest["execution_acceleration"] = {
                    "quote_checks": "guarded one-session native panel row cache",
                    "independent_target_sizing": "guarded fused native target quote queries; unchanged native scalar arithmetic and order construction",
                    "fills": "unchanged pinned native sequential SignedExchange",
                    "executor": "unchanged exact native daily serial SimulatorExecutor",
                    "fastmath": False,
                }
        else:
            strategy = TopkDropoutStrategy(
                signal=scores, topk=topk, n_drop=n_drop, risk_degree=risk_degree,
                method_sell=execution["method_sell"], method_buy="top", hold_thresh=1,
                only_tradable=True, forbid_all_trade_at_limit=True,
            )
            if _preparation is not None:
                from qlib.backtest import get_exchange
                exchange = _preparation.quote_exchange(
                    performance, get_exchange, start_time=start_day, end_time=end_day,
                    codes=eligible, freq="day", deal_price="open", trade_unit=1,
                    limit_threshold=effective_limit, volume_threshold=None,
                    open_cost=cost_per_side, close_cost=cost_per_side,
                    min_cost=0.0, impact_cost=0.0, **quote_options)
        if _cancel_file is not None:
            from qlib_yahoo.preparation import check_cancel
            original_decision = strategy.generate_trade_decision
            # An instance supplied through Qlib's strategy hook; native sizing,
            # order ordering and execution remain unchanged after this guard.
            def cancellable_decision(*args, **kwargs):
                check_cancel(_cancel_file)
                return original_decision(*args, **kwargs)
            strategy.generate_trade_decision = cancellable_decision
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning,
                                    module=r"qlib\.utils\.index_data")
            portfolio, indicators = performance.call("daily simulation", qlib_backtest,
                start_time=backtest_contract["start"], end_time=backtest_contract["end"],
                strategy=strategy,
                executor=SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True,
                                           settle_type=BasePosition.ST_NO,
                                           **({"order_batch_executor": order_batch} if order_batch is not None else {})),
                account=account if strategy_mode == "signed" else initial_cash,
                # Original 0.9.7 turns benchmark=None into {}, which restores
                # its default CSI300 lookup. An empty saved-series input is
                # the supported no-data sentinel; its native bench is ignored.
                # The derivative's supplied account already disables it fully.
                benchmark=None if strategy_mode == "signed" else pd.Series(dtype=float),
                exchange_kwargs={"exchange": exchange} if strategy_mode == "signed" or _preparation is not None else
                    {"codes": eligible, "freq": "day", "deal_price": "open", "trade_unit": 1,
                     "limit_threshold": effective_limit, "volume_threshold": None,
                     "open_cost": cost_per_side, "close_cost": cost_per_side,
                     "min_cost": 0.0, "impact_cost": 0.0, **quote_options},
            )
        report, positions = portfolio["1day"]
        if order_batch is not None:
            manifest["execution_acceleration"]["ordered_session"] = order_batch.snapshot()
            manifest["execution_acceleration"]["fills"] = "guarded complete ordered-session kernel; native per-order fallback"
            manifest["execution_acceleration"]["executor"] = "exact native SimulatorExecutor with opt-in generic batch hook"
        if range_identity is not None:
            manifest["execution_acceleration"]["date_row_comparison"] = {
                "selection": range_identity["selection"], "fallback": range_identity["fallback"],
                "counts": exchange.range_counts.copy() if range_validator is not None else None,
                "financial_arithmetic": "unchanged native NumPy scalars",
            }
        if _compiled_targets:
            manifest["execution_acceleration"]["independent_target_sizing"] = "guarded complete compiled sizing/classification; native signal/weights/equity and Order objects"
            manifest["execution_acceleration"]["target_session"] = strategy.target_session_counts.copy()
        if _columnar_history:
            manifest["history_storage"]["columnar_serialization"] = {
                "position_columns": True, "columnar_indicator_sessions": history_writer.columnar_sessions,
                "native_indicator_fallback_sessions": history_writer.columnar_fallback_sessions}
        if _flat_indicators:
            manifest["history_storage"]["flat_order_indicators"] = account.indicator.flat_update_counts.copy()
        if history_writer is not None:
            performance.call("history finalization", history_writer.close, completed=True)
            if _buffered_history:
                manifest["history_storage"]["buffered_parquet"] = history_writer.buffered_stats()
        indicator_frame, _ = indicators["1day"]
        indicator_frame.to_csv(run_dir / "qlib_trade_indicators.csv", index_label="datetime", lineterminator="\n")
        if strategy_mode == "signed":
            from qlib_yahoo.signed import signed_daily, signed_decisions, signed_positions_frame
            daily = signed_daily(report, positions, initial_cash)
            net = daily["net_return"]
            if history_writer is None:
                # No timestamp or file name in the gzip header, so the same run writes the same bytes.
                with open(run_dir / "signed_decisions.json.gz", "wb") as raw, \
                        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as packed, \
                        io.TextIOWrapper(packed, encoding="utf-8") as handle:
                    json.dump(signed_decisions(strategy), handle, allow_nan=False)
                positions_frame = performance.call("position table", signed_positions_frame, positions, daily)
        else:
            net = _check_ledger(report, initial_cash)
            daily = pd.DataFrame({
                "account": report["account"], "net_return": net,
                "cost": report["cost"],
                "cost_dollars": report["total_cost"].diff().fillna(report["total_cost"].iloc[0]),
                "turnover": report["turnover"], "cash": report["cash"],
                "holdings_value": report["value"],
            })
        daily.index.name = "datetime"
        daily.to_csv(run_dir / "daily_report.csv", lineterminator="\n")
        if strategy_mode != "signed":
            positions_frame = _positions_frame(positions)
        if history_writer is None:
            _write_table(positions_frame, run_dir / "positions.parquet")
        position_rows = len(positions_frame) if history_writer is None else history_writer.rows
        if position_rows == 0 or len(daily) != len(positions):
            raise DatasetError("position history does not cover daily ledger")
        undefined = int(net.isna().sum()) if strategy_mode == "signed" else 0
        if undefined:
            metrics = portfolio_metrics([0.0] * len(net),
                                        annual_cash_rate=annual_cash_rate, periods_per_year=252)
            for key in ("strategy_net",):
                metrics[key] = {name: len(net) if name == "observations" else None
                                for name in metrics[key]}
        else:
            metrics = portfolio_metrics(net.tolist(),
                                        annual_cash_rate=annual_cash_rate, periods_per_year=252)
        summary = {
            "metric_contract": {"version": STRATEGY_METRIC_VERSION, "periods_per_year": 252,
                                "annual_cash_rate": annual_cash_rate,
                                "risk_free_convention": "effective annual rate converted per trading session",
                                "brain_equivalent": False},
            "metrics": metrics, "backtest": backtest_contract,
        }
        if strategy_mode == "signed":
            summary["metrics_status"] = {"strategy": "unavailable" if undefined else "available",
                                         "reason": "nonpositive_equity" if undefined else None,
                                         "undefined_observations": undefined}
            summary["funding_state"] = {"feasible_all_sessions": bool(daily["funding_feasible"].all()),
                                        "unfunded_sessions": int((~daily["funding_feasible"].astype(bool)).sum()),
                                        "max_unfunded_cash_shortfall": float(daily["unfunded_cash_shortfall"].max()),
                                        "stale_mark_sessions": int(daily["stale_marks"].sum())}
        _atomic_json(run_dir / "summary.json", summary)
        manifest["stage"] = "sealed"
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = _now()
        manifest["performance"] = performance.snapshot()
        if _preparation is not None:
            manifest["preparation_reuse"] = _preparation.snapshot()
        manifest["artifacts_sha256"] = {
            path.name: sha256(path) for path in run_dir.iterdir()
            if path.is_file() and path.name != "manifest.json"
        }
        _atomic_json(manifest_path, manifest)
    except BaseException as error:  # also a cancel (SIGTERM becomes KeyboardInterrupt in the worker)
        if history_writer is not None:
            try:
                history_writer.close(completed=False)
            except BaseException as cleanup_error:
                print(f"History cleanup warning: {cleanup_error}", file=sys.stderr)
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = _now()
        manifest["error"] = {"type": type(error).__name__, "message": str(error) or type(error).__name__}
        manifest["performance"] = performance.snapshot()
        _atomic_json(manifest_path, manifest)
        raise

    print(f"Backtest completed: {run_dir}")
    print(f"Evaluation: {backtest_contract['start']} to {backtest_contract['end']}; annual cash-rate assumption: {annual_cash_rate:g}")
    try:
        from qlib_yahoo.analysis import render_report

        effective_level = "overview" if open_report and report_level == "summary" else report_level
        print(f"Rendering the {effective_level} report ...", flush=True)
        report_path = performance.call("report rendering", render_report, run_dir, level=effective_level, holdings=holdings,
                                    threshold=threshold, min_observations=min_observations)
        print(f"Report: {report_path}")
        if open_report:
            webbrowser.open(report_path.as_uri())
    except Exception as error:
        print(f"Report warning (simulation remains completed): {error}", file=sys.stderr)
    return run_dir
