"""Offline adapter to the current Alpha Lab engine, with sealed research splits."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _finite(value):
    """JSON numbers must be finite; undefined statistics stay explicitly null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def _date(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{label} must be an ISO YYYY-MM-DD date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a valid ISO date") from error


class AlphaLabBackend:
    """Run development and validation separately; never score a holdout.

    ``config`` is the backend subdictionary. Dataset paths are local configuration,
    while durable identities use verified content hashes and effective settings.
    """

    def __init__(self, config, runtime_root):
        from qlib_yahoo.alphalab.settings import Settings

        if not isinstance(config, dict):
            raise ValueError("backend config must be an object")
        required = {"dataset", "universe", "development_start", "development_end",
                    "validation_start", "validation_end", "cost_bps"}
        optional = {"holdout_start", "delay", "warmup", "decay", "neutralization", "truncation",
                    "pasteurization", "booksize"}
        missing = sorted(required - config.keys())
        if missing:
            raise ValueError("backend config requires " + ", ".join(missing))
        unknown = sorted(config.keys() - required - optional)
        if unknown:
            raise ValueError("unknown backend settings: " + ", ".join(unknown))
        if not isinstance(config["dataset"], (str, Path)) or not str(config["dataset"]):
            raise ValueError("dataset must be an explicit local path")
        if not isinstance(config["universe"], str) or not config["universe"].strip():
            raise ValueError("universe must be explicit")
        self.dataset = Path(config["dataset"]).resolve()
        self.runtime_root = Path(runtime_root).resolve()
        self.bounds = {key: config[key] for key in
                       ("development_start", "development_end", "validation_start", "validation_end")}
        ds, de, vs, ve = [_date(value, key) for key, value in self.bounds.items()]
        if not ds <= de < vs <= ve:
            raise ValueError("development and validation must be chronological and nonoverlapping")
        if "holdout_start" in config and _date(config["holdout_start"], "holdout_start") <= ve:
            raise ValueError("holdout_start must be after validation_end")
        if type(config.get("delay", 1)) is not int or config.get("delay", 1) != 1:
            raise ValueError("daily autoresearch requires delay=1")
        for key in ("cost_bps", "truncation", "booksize"):
            if key in config and (type(config[key]) not in (int, float) or not math.isfinite(config[key])):
                raise ValueError(f"{key} must be a finite number")
        if "pasteurization" in config and type(config["pasteurization"]) is not bool:
            raise ValueError("pasteurization must be true or false")
        if "neutralization" in config and not isinstance(config["neutralization"], str):
            raise ValueError("neutralization must be text")
        self.settings = Settings(**{key: config[key] for key in
                                   ("universe", "cost_bps", "warmup", "decay", "neutralization", "truncation",
                                    "pasteurization", "booksize") if key in config}, delay=1).validate()
        if self.settings.warmup < 2:
            raise ValueError("warmup must be at least 2 sessions for delay=1")

    def _effective_settings(self):
        settings = asdict(self.settings)
        for key in ("start", "end", "oos_start"):
            settings.pop(key)
        return {**settings, **self.bounds,
                "period_bounds": "inclusive session dates",
                "timing": "close information t-1 sets weights held from close t to close t+1",
                "cost_semantics": "net PnL subtracts cost_bps/10000 times daily traded notional",
                "holdout_evaluated": False}

    def identity(self):
        from qlib_yahoo.alphalab.panel import source_identity
        from qlib_yahoo.alphalab.simulate import ENGINE_VERSION
        import qlib_yahoo

        source = source_identity(self.dataset)
        # Native source_sha256 includes a machine-local dataset path. Rehash the
        # verified source content to preserve portability across dataset moves.
        portable = {key: value for key, value in source.items() if key not in ("dataset", "source_sha256")}
        package = Path(qlib_yahoo.__file__).parent
        code = {"autoresearch/backend.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        code.update({"qlib_yahoo/" + path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(package.rglob("*.py"))})
        versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas")}
        return {"engine": ENGINE_VERSION, "data_identity": _digest(portable), "source": portable,
                "code_sha256": _digest(code), "code": code, "dependencies": versions,
                "settings": self._effective_settings()}

    def validate(self, expression):
        from qlib_yahoo.alphalab.engine import ADV, GROUPS, check_expression
        from qlib_yahoo.alphalab.fastexpr import names, parse
        from qlib_yahoo.alphalab.panel import source_identity
        from qlib_yahoo.reference import dataset_groups

        if not isinstance(expression, str) or not expression.strip():
            return ["expression must be nonempty text"]
        if len(expression) > 4096:
            return ["expression exceeds 4096 characters"]
        problems = check_expression(expression)
        if problems:
            return problems
        try:
            source = source_identity(self.dataset)
            fields = set(source["fields"])
            manifest = json.loads((self.dataset / "manifest.json").read_text())
            rows = dataset_groups(self.dataset, manifest)
            groups = {group for group in GROUPS if group == "market" or any(row.get(group) for row in rows.values())}
            program = parse(expression)
            used = names(program) - {name for name, _ in program.assignments} - {"true", "false"}
            for name in sorted(used):
                if name in GROUPS:
                    if name not in groups:
                        problems.append(f"group {name!r} is unavailable in dataset")
                    continue
                needed = {name}
                if name == "returns":
                    needed = {"close"}
                elif name == "vwap" and name not in fields:
                    needed = {"high", "low", "close"}
                elif name == "dollar_volume" or ADV.fullmatch(name):
                    needed = {"close", "volume"}
                for missing in sorted(needed - fields):
                    problems.append(f"field {missing!r} required by {name!r} is unavailable in dataset")
            group = self.settings.neutralization.lower()
            if group != "none" and group not in groups:
                problems.append(f"neutralization group {group!r} is unavailable in dataset")
            if not problems:
                # Alpha Lab's parser check does not validate operator argument
                # types/windows. A tiny artificial context checks those runtime
                # contracts without scoring or reading any holdout prices.
                import numpy as np
                from qlib_yahoo.alphalab.engine import Context, evaluate
                from qlib_yahoo.alphalab.fastexpr import ExpressionError
                probe = Context(dates=np.arange("2000-01-01", "2000-01-09", dtype="datetime64[D]"),
                                symbols=["PROBE_A", "PROBE_B"],
                                fields={field: np.ones((8, 2), dtype=float) for field in fields},
                                members=np.ones((8, 2), dtype=bool),
                                groups={name: np.zeros(2) for name in groups if name != "market"})
                try:
                    evaluate(expression, probe)
                except ExpressionError as error:
                    problems.append(self._error(error))
                except Exception:
                    # Insufficient artificial history or numerical engine bugs
                    # are not proof that an expression is invalid. The real run
                    # reports those separately as operational failures.
                    pass
        except Exception as error:
            problems.append(self._error(error))
        return problems

    def _error(self, error):
        # Engine errors occasionally interpolate local paths; durable errors do
        # not expose local machine locations, including unexpected OS errors.
        message = str(error).replace(str(self.dataset), "<dataset>").replace(str(self.runtime_root), "<runtime>")
        return re.sub(r"(?<![\w])/(?:[^\s'\"<>]+/)*[^\s'\"<>]*", "<local-path>", message)[:2000]

    def evaluate(self, expression):
        from qlib_yahoo.alphalab.batch import run_batch

        result = {"status": "error", "metrics": {}, "periods": {}, "settings": self._effective_settings()}
        try:
            identity = self.identity()
            result.update(engine=identity["engine"], data_identity=identity["data_identity"])
            problems = self.validate(expression)
            if problems:
                result["error"] = "; ".join(problems)
                result["error_kind"] = "expression"
                return result
            # Isolate cached simulation records by full code/settings/data identity;
            # Alpha Lab's native cache keys alone do not hash adapter source code.
            isolated = self.runtime_root / "backend" / _digest(identity)
            periods, artifacts, cached = {}, {}, True
            for period in ("development", "validation"):
                start, end = self.bounds[period + "_start"], self.bounds[period + "_end"]
                settings = replace(self.settings, start=start, end=end)
                raw = run_batch(self.dataset, [expression], settings, data_root=isolated,
                                library=isolated / "library.sqlite", workers=1, worker_ceiling=1)[0]
                if "error" in raw and raw.get("cached"):
                    # The native library retains failures too. A controller retry
                    # must reattempt a failed period, while successful periods
                    # continue to reuse their normal native cache entries.
                    raw = run_batch(self.dataset, [expression], settings, data_root=isolated,
                                    library=isolated / "library.sqlite", workers=1,
                                    worker_ceiling=1, force=True)[0]
                if "error" in raw:
                    raise ValueError(self._error(raw["error"]))
                dates = raw["daily"]["dates"]
                if not dates or any(not start <= day <= end for day in dates):
                    raise ValueError(f"engine returned dates outside {period} bounds")
                periods[period] = _finite(raw["summary"])
                artifacts[period] = {"summary": periods[period], "daily": _finite(raw["daily"])}
                cached = cached and bool(raw.get("cached"))
            if self.identity() != identity:
                raise ValueError("data or engine identity changed during evaluation")
            artifact = isolated / "evaluations" / (_digest(expression) + ".json")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(json.dumps(artifacts, sort_keys=True, allow_nan=False) + "\n")
            result.update(status="ok", metrics=periods, periods=periods, cached=cached,
                          artifacts=[artifact.relative_to(self.runtime_root).as_posix()])
        except Exception as error:
            result["error"] = self._error(error)
        return _finite(result)


def create_demo_dataset(directory):
    """Create deterministic artificial OHLCV; this is no economic evidence.

    Eight static synthetic instruments, weekdays 2023-01-02..2024-12-31,
    universe TOP8. No network or writes outside ``directory``.
    """
    import gzip
    import numpy as np
    from qlib_yahoo.dataset import tree_sha256
    from qlib_yahoo.alphalab.panel import source_identity

    directory = Path(directory).resolve()
    if (directory / "manifest.json").exists():
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("provider") != "autoresearch-deterministic-artificial-v1":
            raise ValueError("demo refuses to replace an existing dataset")
        source_identity(directory)
        return directory
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("demo dataset directory must be empty")
    days = np.arange("2023-01-02", "2025-01-01", dtype="datetime64[D]")
    days = days[np.is_busday(days)]
    symbols = [f"SYN{i:02d}" for i in range(8)]
    qlib = directory / "qlib_data"
    (qlib / "calendars").mkdir(parents=True)
    (qlib / "instruments").mkdir()
    (qlib / "calendars/day.txt").write_text("\n".join(map(str, days)) + "\n")
    (qlib / "instruments/all.txt").write_text("".join(f"{s}\t{days[0]}\t{days[-1]}\n" for s in symbols))
    rng = np.random.default_rng(7403)
    for i, symbol in enumerate(symbols):
        returns = 0.0002 + 0.002 * np.sin(np.arange(len(days)) / (7 + i)) + rng.normal(0, 0.008, len(days))
        close = (20 + i * 5) * np.exp(np.cumsum(returns))
        open_ = close * np.exp(rng.normal(0, 0.002, len(days)))
        fields = {"close": close, "open": open_, "high": np.maximum(close, open_) * 1.005,
                  "low": np.minimum(close, open_) * 0.995,
                  "volume": rng.integers(100_000, 1_000_000, len(days)).astype(float)}
        folder = qlib / "features" / symbol.lower()
        folder.mkdir(parents=True)
        for field, values in fields.items():
            np.concatenate(([0], values)).astype("<f4").tofile(folder / f"{field}.day.bin")
    universe = directory / "universe"
    universe.mkdir()
    membership = universe / "membership.csv.gz"
    membership.write_bytes(gzip.compress(("instrument,effective_start,effective_end,rank\n" +
                                          "".join(f"{s},{days[0]},{days[-1]},{i+1}\n" for i, s in enumerate(symbols))).encode(), mtime=0))
    manifest = {"status": "ready", "provider": "autoresearch-deterministic-artificial-v1",
                "note": "Artificial fixture only; no investment or economic evidence.",
                "fields": ["open", "high", "low", "close", "volume"],
                "qlib_binary_tree_sha256": tree_sha256(qlib)[0],
                "universes": {"top8": {"size": 8, "membership_csv": "universe/membership.csv.gz",
                                       "membership_sha256": hashlib.sha256(membership.read_bytes()).hexdigest()}}}
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    source_identity(directory)
    return directory
