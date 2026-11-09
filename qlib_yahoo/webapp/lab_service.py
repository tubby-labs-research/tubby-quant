"""Console service for the research-domain lab: autonomous alpha discovery and machine learning.

The console turns a short form (dataset, universe, preset) into a complete
``tubby-lab`` campaign file, runs the campaign in a separate worker process
(the same command-line program a researcher would use), reads progress from
that process and reads results from the record store and the trial ledger.

Nothing here evaluates a candidate itself, and nothing bypasses the harness:
budgets, the sealed development/validation/holdout periods, the hash-chained
ledger and the single-use holdout are the campaign's own rules. Finalize is the
same command the CLI exposes, started only by an explicit request.

Layout under ``<data root>/autoresearch/lab/<domain>/``::

    campaigns/<slug>.json          the campaign file (what tubby-lab runs)
    campaigns/<slug>.console.json  console bookkeeping: preset, dataset, campaign_id once known
    records/                       the shared record store and trial ledgers of this domain
    runtime/                       checkpoints, inboxes and engine caches
    exports/                       batch files written on request

One record store per domain means a campaign started by nudging a threshold or
reusing a dataset shares its validation-trial count and holdout accesses with
every earlier campaign on the same data, as the harness intends.
"""

from __future__ import annotations

import copy
import json
import math
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autoresearch.records import ResearchStore

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ("alpha", "ml")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
_TOP = re.compile(r"^TOP(\d{1,5})$")
WARMUP = 130
MIN_USABLE_SESSIONS = 300

PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "alpha": {
        "quick": {"label": "Quick", "note": "6 rounds of 24 ideas, 2 validations each. Minutes on demonstration data.",
                  "limits": {"max_rounds": 6, "proposals_per_round": 24, "validations_per_round": 2,
                             "max_validation_trials": 12, "max_promotions": 6, "population_size": 30}},
        "standard": {"label": "Standard", "note": "15 rounds of 48 ideas, 3 validations each. Tens of minutes to an hour.",
                     "limits": {"max_rounds": 15, "proposals_per_round": 48, "validations_per_round": 3,
                                "max_validation_trials": 45, "max_promotions": 12, "population_size": 40}},
        "thorough": {"label": "Thorough", "note": "40 rounds of 64 ideas, 3 validations each. Hours; run it while you work.",
                     "limits": {"max_rounds": 40, "proposals_per_round": 64, "validations_per_round": 3,
                                "max_validation_trials": 100, "max_promotions": 20, "population_size": 60}},
    },
    "ml": {
        "quick": {"label": "Quick", "note": "5 rounds of 8 recipes, 2 validations each. A few minutes on a built-in task.",
                  "limits": {"max_rounds": 5, "proposals_per_round": 8, "validations_per_round": 2,
                             "max_validation_trials": 10, "max_promotions": 5, "population_size": 20}},
        "standard": {"label": "Standard", "note": "10 rounds of 10 recipes, 2 validations each.",
                     "limits": {"max_rounds": 10, "proposals_per_round": 10, "validations_per_round": 2,
                                "max_validation_trials": 24, "max_promotions": 8, "population_size": 30}},
        "thorough": {"label": "Thorough", "note": "25 rounds of 16 recipes, 3 validations each. Longer training runs.",
                     "limits": {"max_rounds": 25, "proposals_per_round": 16, "validations_per_round": 3,
                                "max_validation_trials": 60, "max_promotions": 12, "population_size": 40}},
    },
}
ML_TASKS = [
    {"id": "builtin:friedman1", "title": "Friedman #1 · regression with a known answer",
     "note": "Ten features, five informative, noise floor RMSE 1.0. A good search beats the constant baseline clearly."},
    {"id": "builtin:interaction", "title": "Interaction · classification with a known answer",
     "note": "Eight features; the label depends on products and curves of four of them."},
    {"id": "builtin:null_regression", "title": "Null · nothing to find",
     "note": "Pure noise. A search that promotes anything here is fooling itself."},
    {"id": "csv", "title": "Your own CSV table", "note": "Numeric columns; name the target column. Rows may carry an order column for chronological splits."},
    {"id": "panel", "title": "Cross-sectional return panel on a price dataset",
     "note": "Library price/volume features predict the next session's cross-sectional return rank; predictions are scored by Alpha Lab."},
]
ML_FAMILIES = [
    {"id": "ridge", "title": "Ridge regression", "problems": ["regression"]},
    {"id": "logistic", "title": "Logistic regression", "problems": ["classification"]},
    {"id": "gbdt", "title": "Gradient-boosted trees", "problems": ["regression", "classification"]},
    {"id": "random_forest", "title": "Random forest", "problems": ["regression", "classification"]},
    {"id": "mlp", "title": "Neural network (MLP)", "problems": ["regression", "classification"]},
]
STAGES = ["propose", "screen", "shortlist", "validate", "review", "promote", "finalize"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LabService:
    def __init__(self, data_root: str | Path, source_root: str | Path = ROOT, *,
                 spawn: Callable[[list[str]], subprocess.Popen[str]] | None = None) -> None:
        self.data_root = Path(data_root).resolve()
        self.source_root = Path(source_root).resolve()
        self.root = self.data_root / "autoresearch" / "lab"
        self._lock = threading.RLock()
        self._spawn: Callable[[list[str]], subprocess.Popen[str]] = spawn or self._default_spawn
        self._process: subprocess.Popen[str] | None = None
        self._drained = threading.Event()
        self._drained.set()
        self._closed = False
        self._stop_requested = False
        self._worker = self._idle()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _idle() -> dict[str, Any]:
        return {"running": False, "action": None, "id": None, "domain": None, "campaign_id": None, "status": "idle",
                "round": None, "rounds_requested": None, "screened": 0, "validated": 0, "promoted": 0,
                "best_fitness": None, "last_error": None, "result": None, "log": [],
                "started_at": None, "finished_at": None}

    def _default_spawn(self, command: list[str]) -> subprocess.Popen[str]:
        return subprocess.Popen(command, cwd=str(self.source_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)

    @staticmethod
    def _split_id(value: Any) -> tuple[str, str]:
        if not isinstance(value, str) or value.count("/") != 1:
            raise ValueError("Choose a campaign.")
        domain, slug = value.split("/")
        if domain not in DOMAINS or not _SLUG.fullmatch(slug):
            raise ValueError("Unknown campaign identifier.")
        return domain, slug

    def _dir(self, domain: str, part: str) -> Path:
        if domain not in DOMAINS:
            raise ValueError("Unknown research domain.")
        path = self.root / domain / part
        if path.is_symlink():
            raise ValueError("Lab folders must not be symbolic links.")
        return path

    def _config_path(self, domain: str, slug: str) -> Path:
        path = self._dir(domain, "campaigns") / f"{slug}.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError("That campaign no longer exists.")
        return path

    def _sidecar_path(self, domain: str, slug: str) -> Path:
        return self._dir(domain, "campaigns") / f"{slug}.console.json"

    def _sidecar(self, domain: str, slug: str) -> dict[str, Any]:
        try:
            value = json.loads(self._sidecar_path(domain, slug).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write_sidecar(self, domain: str, slug: str, value: dict[str, Any]) -> None:
        from autoresearch.lab.campaign import atomic_json
        atomic_json(self._sidecar_path(domain, slug), value)

    def _clean(self, text: str) -> str:
        text = str(text)
        for root, label in ((str(self.data_root), "~data"), (str(self.source_root), "~app"), (str(Path.home()), "~")):
            text = text.replace(root, label)
        return re.sub(r"(?:/Users/|/home/|/private/|/tmp/|/var/folders/)[^\s'\"<>]*", "[local path]", text)[:2000]

    def _dataset(self, value: Any) -> tuple[Path, dict[str, Any]]:
        """A ready price dataset under the data folder (the Lab and Qlib datasets)."""
        from qlib_yahoo.workspace import dataset_summary, latest_dataset

        if value in (None, ""):
            chosen = latest_dataset(self.data_root)
            if chosen is None:
                raise ValueError("Choose a dataset: no default prices are configured yet.")
        elif isinstance(value, str):
            chosen = Path(value).expanduser()
            if not chosen.is_absolute():
                chosen = self.data_root / "datasets" / chosen
        else:
            raise ValueError("Choose a dataset.")
        resolved = chosen.resolve()
        datasets = (self.data_root / "datasets").resolve()
        if resolved.parent != datasets:
            raise ValueError("Datasets must sit directly under the data folder's datasets directory.")
        if not resolved.is_dir():
            raise ValueError(f"Dataset {resolved.name!r} was not found under {datasets}. Open Data to see what is available.")
        try:
            summary = dataset_summary(self.data_root, resolved)
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read that dataset: {error}") from error
        if not summary.get("price_eligible"):
            raise ValueError(summary.get("compatibility_note") or "Only a ready ordinary price dataset can be searched.")
        return resolved, summary

    @staticmethod
    def _calendar(dataset: Path) -> list[str]:
        plain = dataset / "qlib_data" / "calendars" / "day.txt"
        if plain.is_file():
            return [line.strip() for line in plain.read_text(encoding="utf-8").splitlines() if line.strip()]
        import zipfile
        for archive in sorted(dataset.glob("*.zip")):
            try:
                with zipfile.ZipFile(archive) as handle:
                    member = next((n for n in handle.namelist() if n.endswith("calendars/day.txt")), None)
                    if member:
                        return [line.strip() for line in handle.read(member).decode("utf-8").splitlines() if line.strip()]
            except (OSError, zipfile.BadZipFile):
                continue
        raise ValueError("Cannot read the dataset calendar (qlib_data/calendars/day.txt).")

    @staticmethod
    def _universe(summary: dict[str, Any], value: Any) -> str:
        names = {str(name).upper(): record for name, record in (summary.get("universes") or {}).items()}
        if not names:
            raise ValueError("This dataset has no dated universe; import it with a TOP-N universe first.")
        if not isinstance(value, str) or not value.strip():
            return max(names, key=lambda n: int((names[n] or {}).get("size") or 0))
        wanted = value.strip().upper()
        if wanted in names:
            return wanted
        match = _TOP.fullmatch(wanted)
        largest = max(int((record or {}).get("size") or 0) for record in names.values())
        if match and 0 < int(match.group(1)) <= largest:
            return wanted
        raise ValueError(f"Universe {wanted} is not available; the dataset offers {', '.join(sorted(names))} "
                         f"(or TOP-N up to {largest}).")

    @staticmethod
    def _number(data: dict[str, Any], key: str, default: float, low: float, high: float) -> float:
        raw = data.get(key, default)
        if raw in (None, ""):
            raw = default
        try:
            number = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be a number") from error
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"{key} must be between {low:g} and {high:g}")
        return int(number) if number.is_integer() and key in ("rows", "seed") else number

    @staticmethod
    def _unavailable_names(expression: str, fields: set[str]) -> list[str]:
        """Names an expression references that the selected dataset cannot supply.

        Mirrors the Alpha Lab editor's check: dataset fields plus the derived
        returns / vwap / dollar_volume and adv<N>; classification groups are not
        available to Discover campaigns, which run without a classification source.
        """
        from qlib_yahoo.alphalab.fastexpr import names, parse

        program = parse(expression)
        assigned = {"true", "false"}
        referenced: set[str] = set()
        for binding, value in program.assignments:
            referenced.update(names(value) - assigned)
            assigned.add(binding)
        referenced.update(names(program.result) - assigned)
        available = fields | {"returns", "vwap", "dollar_volume"}
        problems = []
        for name in sorted(referenced):
            if name in ("market", "sector", "industry", "country", "subindustry"):
                problems.append(f"group '{name}' is not available in Discover campaigns (they run without a classification source)")
            elif name not in available and not re.fullmatch(r"adv\d{1,3}", name):
                problems.append(f"field '{name}' is not in this dataset (it has {', '.join(sorted(fields))})")
        return problems

    def _panel_features(self, data: dict[str, Any], fields: set[str] | None = None) -> tuple[Any, list[str]]:
        """Library feature names (or "default" for all) and checked extra FASTEXPR factors."""
        library = [f["name"] for f in self.catalog()["ml_features"]]
        chosen = data.get("features", "default")
        if chosen in (None, "", "default"):
            features: Any = "default"
        else:
            if not isinstance(chosen, list) or not chosen or any(name not in library for name in chosen):
                raise ValueError("Choose factors from the library (or all of them).")
            features = "default" if len(set(chosen)) == len(library) else list(dict.fromkeys(chosen))
        extra = data.get("extra_features") or []
        if isinstance(extra, str):
            extra = [line.strip() for line in extra.splitlines() if line.strip() and not line.strip().startswith("#")]
        if not isinstance(extra, list) or any(not isinstance(e, str) for e in extra):
            raise ValueError("Extra factors must be FASTEXPR expressions, one per line.")
        if len(extra) > 30:
            raise ValueError("Keep extra factors to 30 expressions or fewer.")
        check_expression = self._expression_checker()
        for expression in extra:
            problems = check_expression(expression) or (self._unavailable_names(expression, fields) if fields is not None else [])
            if problems:
                raise ValueError(f"Extra factor {expression!r}: {'; '.join(problems)[:300]}")
        return features, extra

    @staticmethod
    def _expression_checker() -> Callable[[str], list[str]]:
        """Alpha Lab's own expression checker; a missing engine is an error, never a silent pass."""
        try:
            from qlib_yahoo.alphalab.engine import check_expression
        except ImportError as error:
            raise ValueError(f"The expression checker is unavailable (Alpha Lab engine import failed: {error}). "
                             "Expressions cannot be validated, so nothing was created.") from error
        return check_expression

    @staticmethod
    def _budget(counts: dict[str, Any], limits: dict[str, Any]) -> str | None:
        """The campaign's own budget rule, applied to the current limits (which may have been raised)."""
        if not limits:
            return None
        if counts.get("round", 0) >= limits.get("max_rounds", 1e9):
            return "round budget reached"
        if counts.get("validation_trials", 0) >= limits.get("max_validation_trials", 1e9):
            return "validation-trial budget reached"
        if counts.get("screen_trials", 0) >= limits.get("max_screen_trials", 1e9):
            return "screen-trial budget reached"
        if counts.get("promotions", 0) >= limits.get("max_promotions", 1e9):
            return "promotion budget reached"
        return None

    @staticmethod
    def _split(calendar: list[str], warmup: int) -> dict[str, Any]:
        usable = calendar[warmup:]
        if len(usable) < MIN_USABLE_SESSIONS:
            raise ValueError(f"The dataset has {len(calendar)} sessions; at least {MIN_USABLE_SESSIONS + warmup} are "
                             "needed to split development, validation and a reserved holdout.")
        n = len(usable)
        dev_end = int(n * 0.55) - 1
        val_end = int(n * 0.80) - 1
        periods = {"development": (0, dev_end), "validation": (dev_end + 1, val_end), "holdout": (val_end + 1, n - 1)}
        return {name: {"start": usable[a], "end": usable[b], "sessions": b - a + 1} for name, (a, b) in periods.items()}

    # --------------------------------------------------------------- catalogue
    def catalog(self) -> dict[str, Any]:
        try:
            from autoresearch.lab.domain import catalog
            domains = catalog()
            available = True
        except Exception as error:  # the research-domain package is optional in a framework export
            domains, available = [], False
            note = f"Research domains unavailable: {type(error).__name__}: {error}"
        try:
            from autoresearch.domains.ml.tasks import FEATURE_LIBRARY
            features = [{"name": name, "expression": expression} for name, expression in FEATURE_LIBRARY.items()]
        except Exception:
            features = []
        return {"available": available, "domains": domains, "presets": PRESETS, "ml_tasks": ML_TASKS,
                "ml_families": ML_FAMILIES, "ml_features": features, "stages": STAGES,
                "defaults": {"cost_bps": 5, "min_sharpe": 1.0, "warmup": WARMUP},
                **({} if available else {"note": note})}

    # -------------------------------------------------------------------- plan
    def plan(self, data: dict[str, Any]) -> dict[str, Any]:
        """The complete campaign file the form describes, plus a readable summary. Writes nothing."""
        from autoresearch.lab.campaign import validate_lab_config

        if not isinstance(data, dict):
            raise ValueError("Campaign setup must be an object.")
        domain = data.get("domain")
        if domain not in DOMAINS:
            raise ValueError("Choose a research domain: alpha or ml.")
        name = str(data.get("name") or "").strip()
        if not name:
            name = "Alpha discovery" if domain == "alpha" else "Model search"
        if len(name) > 120:
            raise ValueError("Keep the campaign name under 120 characters.")
        preset = data.get("preset") or "quick"
        if preset not in PRESETS[domain]:
            raise ValueError("Choose a preset: quick, standard or thorough.")
        seed = int(self._number(data, "seed", 7, 0, 10**9))
        limits = dict(PRESETS[domain][preset]["limits"])
        summary: dict[str, Any] = {"domain": domain, "name": name, "preset": preset, "limits": limits,
                                   "estimated_screens": limits["max_rounds"] * limits["proposals_per_round"]}
        config: dict[str, Any] = {"name": name, "domain": domain, "seed": seed, "limits": limits,
                                  "research_root": str(self._dir(domain, "records")),
                                  "runtime_root": str(self._dir(domain, "runtime"))}
        task = data.get("task") or "builtin:friedman1"
        if domain == "alpha" or task == "panel":
            dataset, info = self._dataset(data.get("dataset"))
            universe = self._universe(info, data.get("universe"))
            cost = self._number(data, "cost_bps", 5, 0, 200)
            calendar = self._calendar(dataset)
            split = self._split(calendar, WARMUP)
            settings = {"dataset": str(dataset), "universe": universe, "cost_bps": cost, "warmup": WARMUP,
                        "truncation": 0.05,
                        "development_start": split["development"]["start"], "development_end": split["development"]["end"],
                        "validation_start": split["validation"]["start"], "validation_end": split["validation"]["end"],
                        "holdout_start": split["holdout"]["start"]}
            summary.update(dataset=info.get("name"), dataset_path=str(dataset), universe=universe, cost_bps=cost,
                           sessions=len(calendar), warmup=WARMUP, periods=split)
            if domain == "alpha":
                seeds = data.get("seeds") or ""
                if isinstance(seeds, str):
                    seeds = [line.strip() for line in seeds.splitlines() if line.strip() and not line.strip().startswith("#")]
                if not isinstance(seeds, list) or any(not isinstance(s, str) for s in seeds):
                    raise ValueError("Seed ideas must be FASTEXPR expressions, one per line.")
                if len(seeds) > 50:
                    raise ValueError("Keep seed ideas to 50 expressions or fewer.")
                if seeds:
                    check_expression = self._expression_checker()
                    fields = set(info.get("fields") or [])
                    for number, seed in enumerate(seeds, 1):
                        problems = check_expression(seed) or self._unavailable_names(seed, fields)
                        if problems:
                            raise ValueError(f"Seed idea {number} ({seed[:60]!r}): {'; '.join(problems)[:300]}")
                min_sharpe = self._number(data, "min_sharpe", 1.0, 0, 10)
                if seeds:
                    settings["search"] = {"seeds": seeds}
                config["settings"] = settings
                config["policy"] = {"min_sharpe": min_sharpe}
                summary.update(min_sharpe=min_sharpe, seeds=seeds)
            else:
                features, extra = self._panel_features(data, set(info.get("fields") or []))
                config["settings"] = {"task": "panel", "features": features, **({"extra_features": extra} if extra else {}), **settings}
                library = len(features) if isinstance(features, list) else len(self.catalog()["ml_features"])
                summary.update(task="panel", task_title=next(t["title"] for t in ML_TASKS if t["id"] == "panel"),
                               features=library, extra_features=extra)
        else:
            if task not in {t["id"] for t in ML_TASKS}:
                raise ValueError("Choose a machine-learning task.")
            families = data.get("families") or [f["id"] for f in ML_FAMILIES]
            if not isinstance(families, list) or not families or any(f not in {x["id"] for x in ML_FAMILIES} for f in families):
                raise ValueError("Choose at least one model family.")
            split = {"development": 0.6, "validation": 0.2, "holdout": 0.2}
            settings = {"task": "tabular", "split": split, "search": {"families": list(families)}}
            if task.startswith("builtin:"):
                rows = int(self._number(data, "rows", 4000, 300, 200000))
                settings.update(table=task, rows=rows)
                summary.update(rows=rows)
            else:
                table = str(data.get("table") or "").strip()
                target = str(data.get("target") or "").strip()
                path = Path(table).expanduser()
                if not table or not path.is_absolute() or path.suffix.lower() != ".csv" or not path.is_file() or path.is_symlink():
                    raise ValueError("Give the full path of an existing .csv file.")
                if not target:
                    raise ValueError("Name the target column of your table.")
                settings.update(table=str(path.resolve()), target=target)
                order_by = str(data.get("order_by") or "").strip()
                if order_by:
                    settings["order_by"] = order_by
                summary.update(table=path.name, target=target, order_by=order_by or None)
            config["settings"] = settings
            summary.update(task=task, task_title=next(t["title"] for t in ML_TASKS if t["id"] == task),
                           families=list(families), split=split)
        validate_lab_config(config)
        return {"config": config, "summary": summary}

    # ------------------------------------------------------------------ create
    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        from autoresearch.lab.campaign import atomic_json

        planned = self.plan(data)
        config, summary = planned["config"], planned["summary"]
        domain = config["domain"]
        base = re.sub(r"[^a-z0-9]+", "-", config["name"].lower()).strip("-")[:40] or domain
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder = self._dir(domain, "campaigns")
        folder.mkdir(parents=True, exist_ok=True)
        slug, counter = f"{base}-{stamp}", 1
        while (folder / f"{slug}.json").exists():
            counter += 1
            slug = f"{base}-{stamp}-{counter}"
        if not _SLUG.fullmatch(slug):
            raise ValueError("The campaign name produced an invalid identifier; use letters and digits.")
        atomic_json(folder / f"{slug}.json", config)
        self._write_sidecar(domain, slug, {"created_at": utc_now(), "summary": summary, "campaign_id": None,
                                           "runs": []})
        return self.campaign(f"{domain}/{slug}")

    # ------------------------------------------------------------------- read
    def _checkpoint(self, domain: str, campaign_id: str | None) -> dict[str, Any] | None:
        if not campaign_id or not re.fullmatch(r"[A-Za-z0-9_-]{4,80}", campaign_id):
            return None
        path = self._dir(domain, "runtime") / f"{campaign_id}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def _campaign_fields(self, domain: str, slug: str) -> set[str] | None:
        """The fields of the dataset an alpha campaign was created on, or None when they cannot be read."""
        try:
            config = json.loads(self._config_path(domain, slug).read_text(encoding="utf-8"))
            settings = config.get("settings") if isinstance(config.get("settings"), dict) else {}
            _, info = self._dataset(settings.get("dataset"))
        except (OSError, ValueError, TypeError, KeyError):
            return None
        fields = info.get("fields") if isinstance(info, dict) else None
        return set(fields) if fields else None

    def _row(self, domain: str, slug: str) -> dict[str, Any]:
        path = self._config_path(domain, slug)
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
        side = self._sidecar(domain, slug)
        summary_value, campaign_value = side.get("summary"), side.get("campaign_id")
        summary: dict[str, Any] = summary_value if isinstance(summary_value, dict) else {}
        campaign_id = campaign_value if isinstance(campaign_value, str) else None
        limits = config.get("limits") if isinstance(config.get("limits"), dict) else {}
        row = {"id": f"{domain}/{slug}", "domain": domain, "name": config.get("name") or slug,
               "preset": summary.get("preset"), "created_at": side.get("created_at"), "campaign_id": campaign_id,
               "dataset": summary.get("dataset"), "universe": summary.get("universe"), "task": summary.get("task"),
               "task_title": summary.get("task_title"), "limits": limits, "status": "not started", "round": 0,
               "screen_trials": 0, "validation_trials": 0, "promotions": 0, "holdout_accessed": False,
               "best_fitness": None, "budget": None, "running": False}
        checkpoint = self._checkpoint(domain, campaign_id)
        if checkpoint:
            population = checkpoint.get("population") or []
            # The checkpoint's own status: "running" means resumable work in progress, not a live process.
            labels = {"running": "in progress", "ready": "not started", "complete": "complete"}
            checkpoint_status = checkpoint.get("status") or "in progress"
            row.update(status=labels.get(checkpoint_status, str(checkpoint_status)),
                       round=checkpoint.get("round", 0),
                       screen_trials=checkpoint.get("screen_trials", 0), validation_trials=checkpoint.get("validation_trials", 0),
                       promotions=checkpoint.get("promotions", 0), holdout_accessed=bool(checkpoint.get("holdout_accessed")),
                       best_fitness=population[0].get("fitness") if population else None)
            row["budget"] = self._budget(row, limits)
            row["status"] = "complete" if row["budget"] else ("in progress" if row["round"] else "not started")
        with self._lock:
            if self._worker["id"] == row["id"] and self.busy():
                row["running"] = True
                row["status"] = "running"
        return row

    def campaigns(self) -> list[dict[str, Any]]:
        rows = []
        for domain in DOMAINS:
            folder = self.root / domain / "campaigns"
            if not folder.is_dir() or folder.is_symlink():
                continue
            for path in sorted(folder.glob("*.json")):
                if path.name.endswith(".console.json") or path.is_symlink() or not _SLUG.fullmatch(path.stem):
                    continue
                try:
                    rows.append(self._row(domain, path.stem))
                except ValueError:
                    continue
        rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return rows

    def campaign(self, value: Any) -> dict[str, Any]:
        domain, slug = self._split_id(value)
        row = self._row(domain, slug)
        config = json.loads(self._config_path(domain, slug).read_text(encoding="utf-8"))
        public = {k: v for k, v in config.items() if k not in ("research_root", "runtime_root")}
        settings = dict(public.get("settings") or {})
        for key in ("dataset", "table"):
            if isinstance(settings.get(key), str) and not settings[key].startswith("builtin:"):
                settings[key] = Path(settings[key]).name
        public["settings"] = settings
        side = self._sidecar(domain, slug)
        return {**row, "config": public, "summary": side.get("summary") or {}, "runs": side.get("runs") or []}

    def report(self, value: Any) -> dict[str, Any]:
        from autoresearch.lab.report import store_report

        domain, slug = self._split_id(value)
        detail = self.campaign(value)
        if not detail["campaign_id"]:
            return {**detail, "report": None,
                    "note": "This campaign has not run yet. Run it to see candidates, trials and the ledger."}
        try:
            recorded = store_report(self._dir(domain, "records"), detail["campaign_id"])["campaign"]
        except ValueError as error:
            return {**detail, "report": None, "note": str(error)}
        if recorded is None:
            return {**detail, "report": None, "note": "no campaign with that identifier in this store yet; run it to record trials"}
        report: dict[str, Any] = dict(recorded)
        report["leaderboard"] = report.get("leaderboard", [])[:200]
        report.update(self._analysis(domain, detail["campaign_id"], report["leaderboard"]))
        report["lineage"] = self._lineage(domain, slug)
        return {**detail, "report": report}

    def _lineage(self, domain: str, slug: str) -> dict[str, Any] | None:
        """The campaign's lineage registry view (trial counts across record stores, holdout retirement), read-only."""
        from autoresearch.lab.campaign import LabCampaign, load_lab_config

        try:
            campaign = LabCampaign(load_lab_config(self._config_path(domain, slug)))
            summary = campaign.lineage_summary()
        except Exception as error:  # a campaign whose registry cannot be read still reports; the console says why
            return {"error": self._clean(str(error))}
        summary["registry"] = self._clean(str(summary.get("registry", "")))
        summary["default_registry"] = self._clean(str(summary.get("default_registry", "")))
        seeded = summary.get("seeded")
        if isinstance(seeded, dict) and seeded.get("research_root"):
            seeded["research_root"] = self._clean(str(seeded["research_root"]))
        return summary

    def _analysis(self, domain: str, campaign_id: str, leaderboard: list[dict[str, Any]]) -> dict[str, Any]:
        """Round-by-round progress and, for model recipes with importances, which factors mattered."""
        from autoresearch.records import ResearchStore

        store = ResearchStore(self._dir(domain, "records"))
        rounds = []
        for event in store.list("events"):
            if event.get("campaign_id") == campaign_id and event.get("kind") == "round":
                rounds.append({key: event.get(key) for key in ("round", "proposed", "screened", "screen_ok", "submitted",
                                                              "validated", "promoted", "best_fitness")})
        rounds.sort(key=lambda r: r.get("round") or 0)
        out: dict[str, Any] = {"rounds": rounds, "feature_names": self._feature_names(store, campaign_id)}
        if domain == "ml":
            out["factor_importance"] = self._factor_importance(store, campaign_id, leaderboard, out["feature_names"])
        return out

    @staticmethod
    def _feature_names(store: ResearchStore, campaign_id: str) -> list[str]:
        record = store.get("campaigns", campaign_id) or {}
        task = ((record.get("identity") or {}).get("task") or {})
        features = task.get("features")
        if isinstance(features, dict):
            return [str(name) for name in features]
        if isinstance(features, list):
            return [str(name) for name in features]
        return []

    @staticmethod
    def _factor_importance(store: ResearchStore, campaign_id: str, leaderboard: list[dict[str, Any]],
                           names: list[str]) -> dict[str, Any]:
        decisions = {row["hypothesis_id"]: row.get("decision") for row in leaderboard}
        totals: dict[str, list[float]] = {}
        models = {"promote": 0, "all": 0}
        for experiment in store.list("experiments"):
            if experiment.get("campaign_id") != campaign_id:
                continue
            model = (experiment.get("result") or {}).get("model") or {}
            weights = model.get("feature_importance")
            if not isinstance(weights, list) or not weights:
                continue
            total = sum(w for w in weights if isinstance(w, (int, float)) and w >= 0)
            if total <= 0:
                continue
            bucket = "promote" if decisions.get(experiment.get("hypothesis_id")) == "promote" else "other"
            models["all"] += 1
            if bucket == "promote":
                models["promote"] += 1
            for index, weight in enumerate(weights):
                name = names[index] if index < len(names) else f"feature_{index}"
                totals.setdefault(name, [0.0, 0.0])
                totals[name][0 if bucket == "promote" else 1] += float(weight) / total
        source = "promote" if models["promote"] else "all"
        count = models["promote"] if source == "promote" else models["all"]
        rows: list[dict[str, Any]] = []
        for name, (promoted, other) in totals.items():
            weight = promoted if source == "promote" else promoted + other
            if count:
                rows.append({"name": name, "weight": weight / count})
        rows.sort(key=lambda r: -r["weight"])
        return {"source": "promoted recipes" if source == "promote" else "every validated recipe", "models": count, "rows": rows[:40],
                "note": "Split-gain importance of tree models (gradient boosting, random forest), averaged per recipe; "
                        "linear models and networks carry no importance and are not counted."}

    def candidate(self, data: dict[str, Any]) -> dict[str, Any]:
        """One candidate's full evidence: hypothesis, latest experiment, review and final."""
        from autoresearch.records import ResearchStore

        domain, slug = self._split_id((data or {}).get("id"))
        hypothesis_id = (data or {}).get("hypothesis")
        if not isinstance(hypothesis_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{4,120}", hypothesis_id):
            raise ValueError("Choose a candidate.")
        row = self._row(domain, slug)
        if not row["campaign_id"]:
            raise ValueError("Run the campaign first.")
        store = ResearchStore(self._dir(domain, "records"))
        hypothesis = store.get("hypotheses", hypothesis_id)
        if not hypothesis or hypothesis.get("campaign_id") != row["campaign_id"]:
            raise ValueError("That candidate does not belong to this campaign.")
        latest = lambda collection: max((r for r in store.list(collection) if r.get("hypothesis_id") == hypothesis_id),
                                        key=lambda r: r.get("created_at", ""), default=None)
        experiment, review, final = latest("experiments"), latest("reviews"), latest("finals")
        result = (experiment or {}).get("result") or {}
        keep = {key: result.get(key) for key in ("status", "metrics", "periods", "robustness", "model", "error", "error_kind", "cached")}
        settings = result.get("settings") or {}
        keep["settings"] = {key: settings.get(key) for key in ("universe", "cost_bps", "neutralization", "decay", "truncation", "warmup")
                            if key in settings}
        return {"id": row["id"], "campaign_id": row["campaign_id"], "feature_names": self._feature_names(store, row["campaign_id"]),
                "hypothesis": {key: hypothesis.get(key) for key in ("id", "title", "family", "generator", "round", "expression", "recipe",
                                                                     "parents", "screen", "screen_fitness", "post_holdout", "created_at")},
                "experiment": keep if experiment else None, "review": review and {key: review.get(key) for key in ("decision", "reason", "checks", "created_at")},
                "final": final and {key: final.get(key) for key in ("decision", "reason", "reused", "post_holdout", "access_sequence", "created_at")}
                | ({"holdout": ((final.get("result") or {}).get("metrics") or {}).get("holdout")} if final else {})}

    def extend(self, data: dict[str, Any]) -> dict[str, Any]:
        """Raise a finished campaign's operational budget by one preset step; gates and identity stay as they are."""
        from autoresearch.lab.campaign import atomic_json, validate_lab_config

        domain, slug = self._split_id((data or {}).get("id"))
        if self.busy() and self._worker.get("id") == f"{domain}/{slug}":
            raise ValueError("Stop the campaign before changing its budget.")
        path = self._config_path(domain, slug)
        config = json.loads(path.read_text(encoding="utf-8"))
        side = self._sidecar(domain, slug)
        preset = ((side.get("summary") or {}).get("preset")) or "quick"
        step = PRESETS[domain].get(preset, PRESETS[domain]["quick"])["limits"]
        limits = dict(config.get("limits") or {})
        for key in ("max_rounds", "max_validation_trials", "max_promotions"):
            limits[key] = int(limits.get(key, step[key])) + int(step[key])
        limits["max_screen_trials"] = int(limits.get("max_screen_trials", 2000)) + int(step["max_rounds"]) * int(step["proposals_per_round"])
        config["limits"] = limits
        validate_lab_config(config)
        atomic_json(path, config)
        side.setdefault("extensions", []).append({"at": utc_now(), "limits": limits})
        self._write_sidecar(domain, slug, side)
        return self.campaign(f"{domain}/{slug}")

    def verify(self, value: Any) -> dict[str, Any]:
        from autoresearch.lab.ledger import TrialLedger

        domain, slug = self._split_id(value)
        row = self._row(domain, slug)
        if not row["campaign_id"]:
            raise ValueError("Run the campaign first; a ledger is written from its first round.")
        ledger = TrialLedger(self._dir(domain, "records") / "ledgers" / (row["campaign_id"] + ".jsonl")).verify()
        stages = ledger.get("stages") or {}
        consistent = bool(ledger.get("ok")) and stages.get("validate", 0) == row["validation_trials"] \
            and stages.get("screen", 0) == row["screen_trials"]
        registry = self._registry_check(domain, slug)
        registry_ok = registry.get("status") not in ("registry_reset", "registry_shrunk")
        return {"id": row["id"], "campaign_id": row["campaign_id"], "ledger": ledger,
                "checkpoint_matches_ledger": consistent, "registry": registry, "ok": bool(consistent and registry_ok),
                "checkpoint": {"screen_trials": row["screen_trials"], "validation_trials": row["validation_trials"]}}

    def _registry_check(self, domain: str, slug: str) -> dict[str, Any]:
        """The lineage registry against the ledger's latest snapshot (registry_reset / registry_shrunk / no_snapshot / ok)."""
        from autoresearch.lab.campaign import LabCampaign, load_lab_config

        try:
            result = LabCampaign(load_lab_config(self._config_path(domain, slug))).verify()
        except Exception as error:  # the ledger verdict above stands on its own; say why the registry could not be read
            return {"status": "unavailable", "problems": [self._clean(str(error))]}
        registry = dict(result.get("registry") or {})
        if "path" in registry:
            registry["path"] = self._clean(str(registry["path"]))
        registry.setdefault("status", "ok" if result.get("ok") else "unknown")
        return registry

    # ------------------------------------------------------------------ worker
    def busy(self) -> bool:
        """Running until the worker process has exited and its last output has been recorded."""
        with self._lock:
            return self._process is not None and (self._process.poll() is None or not self._drained.is_set())

    def worker(self) -> dict[str, Any]:
        with self._lock:
            value = copy.deepcopy(self._worker)
            value["running"] = self.busy()
            return value

    def _launch(self, action: str, campaign_id_hint: str | None, identifier: str, domain: str, command: list[str],
                rounds: int | None = None) -> dict[str, Any]:
        with self._lock:
            if self.busy() or self._closed:
                raise ValueError("A lab campaign is already running. Stop it or wait for it to finish.")
            process = self._spawn([sys.executable, "-m", "autoresearch.lab.cli", *command])
            self._stop_requested = False
            self._process = process
            self._drained = threading.Event()
            self._worker = {**self._idle(), "running": True, "action": action, "id": identifier, "domain": domain,
                            "campaign_id": campaign_id_hint, "status": "starting", "rounds_requested": rounds,
                            "started_at": utc_now()}
            threading.Thread(target=self._drain, args=(process, action, identifier, domain), daemon=True).start()
            return copy.deepcopy(self._worker)

    def _record_campaign_id(self, domain: str, slug: str, campaign_id: str) -> None:
        side = self._sidecar(domain, slug)
        if side.get("campaign_id") != campaign_id:
            side["campaign_id"] = campaign_id
            self._write_sidecar(domain, slug, side)

    def _drain(self, process: subprocess.Popen[str], action: str, identifier: str, domain: str) -> None:
        domain, slug = self._split_id(identifier)
        buffer: list[str] = []
        stream = process.stdout
        assert stream is not None, "the lab worker is started with stdout=PIPE"
        try:
            for raw in stream:
                line = raw.rstrip("\n")[:8000]
                buffer.append(line)
                with self._lock:
                    if self._process is not process:
                        return  # superseded; the finally block leaves the new process's event alone
                    worker = self._worker
                    try:
                        update = json.loads(line)
                    except ValueError:
                        update = None
                    if not isinstance(update, dict):
                        text = self._clean(line)
                        if text.strip():
                            worker["log"] = (worker["log"] + [text])[-80:]
                            if text.startswith("tubby-lab:") or text.startswith("Traceback"):
                                worker["last_error"] = text[:1000]
                        continue
                    if isinstance(update.get("campaign_id"), str):
                        worker["campaign_id"] = update["campaign_id"]
                        try:
                            self._record_campaign_id(domain, slug, update["campaign_id"])
                        except (OSError, ValueError):
                            pass
                    status = update.get("status")
                    if status == "round":
                        worker["status"] = "running"
                        worker["round"] = update.get("round")
                        worker["screened"] += int(update.get("screened") or 0)
                        worker["validated"] += int(update.get("validated") or 0)
                        worker["promoted"] += int(update.get("promoted") or 0)
                        worker["best_fitness"] = update.get("best_fitness")
                        decisions = update.get("decisions") or []
                        verdicts = ", ".join(f"{d.get('decision')} ({d.get('score'):.2f})" if isinstance(d.get("score"), (int, float))
                                             else str(d.get("decision")) for d in decisions) or "no validation this round"
                        text = (f"Round {update.get('round')}: {update.get('proposed', 0)} proposed, {update.get('screened', 0)} screened"
                                f" ({update.get('screen_ok', 0)} scored), {update.get('validated', 0)} validated · {verdicts}")
                        worker["log"] = (worker["log"] + [text])[-80:]
                    elif status == "complete":
                        worker["status"] = "complete"
                        worker["log"] = (worker["log"] + [f"Campaign complete: {update.get('reason', 'budget reached')}"])[-80:]
                    elif status == "stopped":
                        worker["round"] = update.get("round", worker["round"])
                        worker["result"] = {"screen_trials": update.get("screen_trials"), "validation_trials": update.get("validation_trials"),
                                            "promotions": update.get("promotions"), "budget": update.get("budget")}
                        worker["log"] = (worker["log"] + [
                            f"Stopped after round {update.get('round')}: {update.get('screen_trials')} screened, "
                            f"{update.get('validation_trials')} validated, {update.get('promotions')} promoted"
                            + (f" · {update['budget']}" if update.get("budget") else "")])[-80:]
                    elif status == "finalized":
                        worker["result"] = {"finals": update.get("finals")}
                    else:
                        worker["log"] = (worker["log"] + [self._clean(line)])[-80:]
            code = process.wait()
            with self._lock:
                if self._process is not process:
                    return
                worker = self._worker
                if action == "finalize" and worker["result"] is None:
                    try:
                        whole = json.loads("\n".join(buffer))
                        if isinstance(whole, dict) and whole.get("status") == "finalized":
                            worker["result"] = {"finals": whole.get("finals")}
                            if isinstance(whole.get("campaign_id"), str):
                                worker["campaign_id"] = whole["campaign_id"]
                    except ValueError:
                        pass
                if worker["result"] and worker["result"].get("finals") is not None:
                    finals = worker["result"]["finals"]
                    confirmed = sum(1 for f in finals if f.get("decision") == "confirmed")
                    worker["log"] = (worker["log"] + [f"Finalized {len(finals)} candidate(s) on the holdout: {confirmed} confirmed"])[-80:]
                worker["running"] = False
                worker["finished_at"] = utc_now()
                if code and not self._stop_requested:
                    worker["status"] = "error"
                    if not worker["last_error"]:
                        worker["last_error"] = f"The lab worker exited with code {code}"
                elif self._stop_requested:
                    worker["status"] = "stopped"
                elif worker["status"] not in ("complete",):
                    worker["status"] = "finished"
        finally:
            if process.stdout:
                process.stdout.close()
            with self._lock:
                if self._process is process:
                    self._drained.set()

    def start(self, data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("Choose a campaign to run.")
        domain, slug = self._split_id(data.get("id"))
        row = self._row(domain, slug)
        if row["budget"]:
            raise ValueError(f"This campaign has used its budget ({row['budget']}). Add budget to continue, or create a new campaign.")
        rounds = data.get("rounds")
        command = ["run", str(self._config_path(domain, slug))]
        if rounds not in (None, ""):
            rounds = int(self._number({"rounds": rounds}, "rounds", 1, 1, 1000))
            command += ["--rounds", str(rounds)]
        else:
            rounds = None
        return self._launch("run", row["campaign_id"], row["id"], domain, command, rounds)

    def finalize(self, data: dict[str, Any]) -> dict[str, Any]:
        """Score promoted candidates once on the reserved holdout: explicit, never automatic, never reused here."""
        if not isinstance(data, dict) or data.get("confirm") is not True:
            raise ValueError("Finalize needs an explicit confirmation: it reads the reserved holdout once per candidate.")
        domain, slug = self._split_id(data.get("id"))
        row = self._row(domain, slug)
        if not row["campaign_id"]:
            raise ValueError("Run the campaign before finalizing.")
        if row["promotions"] < 1:
            raise ValueError("Nothing was promoted, so there is nothing to score on the holdout.")
        command = ["finalize", str(self._config_path(domain, slug))]
        if data.get("non_confirmatory") is True:
            # The lineage's holdout has already been read (or another record store used it): the read is evidence only,
            # recorded as non-confirmatory by the harness and never labelled confirmed.
            command.append("--non-confirmatory")
        elif "non_confirmatory" in data and data["non_confirmatory"] not in (False, None):
            raise ValueError("non_confirmatory must be true or false.")
        keys = data.get("keys")
        if keys:
            if not isinstance(keys, list) or any(not isinstance(k, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", k) for k in keys):
                raise ValueError("Candidate keys must be identifiers.")
        # A candidate whose holdout was already read (here or in another campaign on the same data) cannot be read again
        # from the console: the harness refuses a repeat unless --allow-reuse, which only the command line may pass. Leave
        # such candidates out of the request and say so, rather than letting the worker fail on them.
        skipped: list[str] = []
        promoted, accessed = self._promoted_and_read(domain, slug)
        if promoted is not None:
            wanted = [k for k in (keys or promoted) if k in promoted] if keys else list(promoted)
            skipped = [k for k in wanted if k in accessed]
            remaining = [k for k in wanted if k not in accessed]
            if wanted and not remaining:
                raise ValueError("Every promoted candidate's holdout was already read, in this or another campaign on the same "
                                 "data (" + ", ".join(skipped[:6]) + ("…" if len(skipped) > 6 else "") + "). The console never "
                                 "repeats a read; a repeated read is requested from the command line with --allow-reuse and "
                                 "is labelled reused.")
            if skipped:
                keys = remaining
        if keys:
            for key in keys[:50]:
                command += ["--key", key]
        worker = self._launch("finalize", row["campaign_id"], row["id"], domain, command)
        if skipped:
            worker = {**worker, "skipped_already_read": skipped}
        return worker

    def _promoted_and_read(self, domain: str, slug: str) -> tuple[list[str] | None, set[str]]:
        """The campaign's promoted keys and the lineage's holdout keys already read; (None, set()) when it cannot be inspected."""
        from autoresearch.lab.campaign import LabCampaign, load_lab_config

        try:
            campaign = LabCampaign(load_lab_config(self._config_path(domain, slug)))
            campaign.status()
            promoted = [str(entry["key"]) for entry in campaign.checkpoint["accepted"]]
            accessed = set(campaign.lineage_summary().get("holdout_accesses") or {})
        except Exception:  # the worker will say what is wrong; the pre-check is a courtesy, not the gate
            return None, set()
        return promoted, accessed

    def stop(self, *, _grace: float = 65) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return self.worker()
            self._stop_requested = True
            self._worker["status"] = "stopping"
            process.terminate()
        try:
            process.wait(timeout=_grace)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        deadline = time.time() + 5
        while time.time() < deadline:  # the drain thread records the final state
            with self._lock:
                if not self._worker["running"]:
                    break
            time.sleep(0.05)
        return self.worker()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self.stop(_grace=5)

    # ------------------------------------------------------------- operations
    def submit(self, data: dict[str, Any]) -> dict[str, Any]:
        """Queue one idea of your own for the campaign's next round; it is screened like every other candidate."""
        from autoresearch.lab.campaign import append_inbox

        if not isinstance(data, dict):
            raise ValueError("Choose a campaign.")
        domain, slug = self._split_id(data.get("id"))
        row = self._row(domain, slug)
        if not row["campaign_id"]:
            raise ValueError("Run the campaign once before submitting ideas; before the first round, add them as seed ideas in the setup.")
        candidate: dict[str, Any]
        if domain == "alpha":
            expression = str(data.get("expression") or "").strip()
            if not expression or len(expression) > 2000:
                raise ValueError("Write one FASTEXPR expression.")
            # Same check as a seed idea: a syntax error or a field this campaign's dataset cannot supply is named
            # here, before anything is written, instead of failing silently in the next round's screen.
            problems = self._expression_checker()(expression)
            if not problems:
                fields = self._campaign_fields(domain, slug)
                if fields is not None:
                    problems = self._unavailable_names(expression, fields)
            if problems:
                raise ValueError(f"Idea ({expression[:60]!r}): {'; '.join(problems)[:300]}")
            candidate = {"expression": expression, "rationale": str(data.get("rationale") or "")[:500]}
        else:
            recipe = data.get("recipe")
            if isinstance(recipe, str):
                try:
                    recipe = json.loads(recipe)
                except ValueError as error:
                    raise ValueError("A recipe is JSON such as {\"family\": \"gbdt\", \"params\": {\"depth\": 4}}.") from error
            if not isinstance(recipe, dict) or not isinstance(recipe.get("family"), str):
                raise ValueError("A recipe names a model family and its params.")
            candidate = {"family": recipe["family"], "params": recipe.get("params") or {}}
        inbox = self._dir(domain, "runtime") / f"{row['campaign_id']}.inbox.jsonl"
        append_inbox(inbox, candidate, "console")
        return {"queued": True, "id": row["id"], "note": "Screened at the start of the next round. Run the campaign to process it."}

    def export(self, data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("Choose a campaign.")
        domain, slug = self._split_id(data.get("id"))
        decision = data.get("decision") or "promote"
        if decision not in ("promote", "confirmed", "all"):
            raise ValueError("Export promoted, confirmed or all candidates.")
        row = self._row(domain, slug)
        if not row["campaign_id"]:
            raise ValueError("Run the campaign before exporting.")
        folder = self._dir(domain, "exports")
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if domain == "alpha":
            from autoresearch.handoff import campaign_report, export_batch
            from autoresearch.records import ResearchStore

            report = campaign_report(ResearchStore(self._dir(domain, "records")), row["campaign_id"])
            destination = folder / f"{slug}-{decision}-{stamp}.txt"
            path = export_batch(report, destination, decision)
            count = sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#"))
            note = "An Alpha Lab batch: open it with the Lab's batch loader or paste the expressions into Alpha Lab."
        else:
            from autoresearch.lab.report import store_report

            recorded = store_report(self._dir(domain, "records"), row["campaign_id"])["campaign"]
            if recorded is None:
                raise ValueError("This campaign has no records yet; run it before exporting.")
            rows = recorded["leaderboard"]
            if decision == "all":
                chosen = [r for r in rows if r.get("outcome") == "ok"]
            elif decision == "confirmed":
                chosen = [r for r in rows if (final := r["final"]) is not None and final["decision"] == "confirmed"]
            else:
                chosen = [r for r in rows if r.get("decision") == "promote"]
            path = folder / f"{slug}-{decision}-{stamp}.json"
            path.write_text(json.dumps({"campaign_id": row["campaign_id"], "recipes": chosen}, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
            count = len(chosen)
            note = "Recipes as JSON (family and params); each round-trips exactly through the numpy model library."
        return {"file": str(path), "count": count, "decision": decision, "note": note}

    def demo(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create and start a short campaign on artificial data with a known answer."""
        domain = (data or {}).get("domain") if isinstance(data, dict) else None
        if domain not in DOMAINS:
            raise ValueError("Choose a demonstration: alpha or ml.")
        if self.busy():
            raise ValueError("A lab campaign is already running. Stop it or wait for it to finish.")
        if domain == "alpha":
            from qlib_yahoo.demo_market import DEFAULT_NAME, create_demo_market
            from qlib_yahoo.workspace import latest_dataset, set_default_dataset

            dataset = create_demo_market(self.data_root / "datasets" / DEFAULT_NAME)
            if latest_dataset(self.data_root) is None:
                set_default_dataset(self.data_root, dataset)
            row = self.create({"domain": "alpha", "name": "Demo · alpha discovery on planted data", "preset": "quick",
                               "dataset": str(dataset), "universe": "", "cost_bps": 5, "min_sharpe": 1.0})
        else:
            # Factor analysis on the planted panel: the three planted signals sit in the factor library
            # (ret_5, log_volume_shock, vol_20), so the importance chart has a known answer too.
            from qlib_yahoo.demo_market import DEFAULT_NAME, create_demo_market
            from qlib_yahoo.workspace import latest_dataset, set_default_dataset

            dataset = create_demo_market(self.data_root / "datasets" / DEFAULT_NAME)
            if latest_dataset(self.data_root) is None:
                set_default_dataset(self.data_root, dataset)
            row = self.create({"domain": "ml", "name": "Demo · factor analysis on planted data", "preset": "quick",
                               "task": "panel", "dataset": str(dataset), "universe": "", "cost_bps": 5,
                               "families": ["ridge", "gbdt", "random_forest"]})
        worker = self.start({"id": row["id"], **({"rounds": 3} if domain == "ml" else {})})
        return {"campaign": row, "worker": worker}
