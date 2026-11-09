"""Supervised-learning tasks with structurally sealed splits.

Two task kinds share one contract — ``development()``, ``validation()``,
``holdout()`` and ``identity()`` — so the machine-learning domain can search
training recipes on either without knowing which:

* :class:`TabularTask` — any numeric table (CSV) or a built-in synthetic task.
  Rows are split once, chronologically by an ``order_by`` column or by a seeded
  permutation; the holdout rows are not kept in memory and are re-read only by
  ``holdout()``.
* :class:`PanelTask` — cross-sectional return prediction on an Alpha Lab
  dataset. Features are FASTEXPR expressions ranked across the universe each
  day; the target is the next tradable close-to-close return under delay 1
  (information through the close of t, return from t+1 to t+2). Training rows
  never include a target realized after the end of their split (purged), and
  the feature panel for development and validation is computed on a workspace
  whose prices end at the validation end, bounded there by the loader itself
  (``sealed_panel``): no later session is deserialized by search-time code.
  The reserved sessions are read by ``PanelTask.evaluate_holdout`` alone,
  through ``open_holdout_panel``, on a panel that is dropped when it returns.
  Predictions are scored by the unmodified Alpha Lab simulator, injected as a
  data field, so an ML signal and a FASTEXPR alpha are measured identically.

Built-in tabular tasks (deterministic, offline) have known answers:
``friedman1`` (regression; irreducible RMSE 1.0), ``interaction``
(classification; Bayes-optimal log loss reported from the true probabilities)
and ``null_regression`` (no relation at all; any promotion is false).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from .models import Model

BUILTINS = ("friedman1", "interaction", "null_regression")
DEFAULT_SPLIT = {"development": 0.6, "validation": 0.2, "holdout": 0.2}

FEATURE_LIBRARY = {
    "ret_1": "returns",
    "ret_5": "ts_delta(close, 5) / ts_delay(close, 5)",
    "ret_20": "ts_delta(close, 20) / ts_delay(close, 20)",
    "ret_60": "ts_delta(close, 60) / ts_delay(close, 60)",
    "decay_ret_10": "ts_decay_linear(returns, 10)",
    "sharpe_20": "ts_mean(returns, 20) / ts_std_dev(returns, 20)",
    "vol_5": "ts_std_dev(returns, 5)",
    "vol_20": "ts_std_dev(returns, 20)",
    "vol_60": "ts_std_dev(returns, 60)",
    "max_ret_20": "ts_max(returns, 20)",
    "min_ret_20": "ts_min(returns, 20)",
    "volume_ratio_5": "volume / ts_mean(volume, 5)",
    "volume_ratio_20": "volume / ts_mean(volume, 20)",
    "log_volume_shock": "log(volume) - ts_mean(log(volume), 20)",
    "volume_trend": "ts_mean(volume, 5) / ts_mean(volume, 60)",
    "liquidity": "log(adv20)",
    "range_5": "ts_mean((high - low) / close, 5)",
    "close_location": "(close - low) / (high - low)",
    "gap": "open / ts_delay(close, 1) - 1",
    "vwap_deviation": "(close - vwap) / vwap",
    "price_volume_corr_10": "ts_corr(rank(close), rank(volume), 10)",
    "ts_rank_close_20": "ts_rank(close, 20)",
}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------- tabular
def builtin_table(name: str, rows: int = 4000, seed: int = 7) -> dict[str, Any]:
    """Deterministic synthetic tables with known structure."""
    rng = np.random.default_rng(seed)
    if name == "friedman1":
        x = rng.uniform(0, 1, (rows, 10))
        signal = 10 * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20 * (x[:, 2] - 0.5) ** 2 + 10 * x[:, 3] + 5 * x[:, 4]
        y = signal + rng.normal(0, 1.0, rows)
        return {"X": x, "y": y, "features": [f"x{i}" for i in range(10)], "problem": "regression",
                "known": {"noise_floor_rmse": 1.0, "informative": ["x0", "x1", "x2", "x3", "x4"]}}
    if name == "interaction":
        x = rng.normal(0, 1, (rows, 8))
        logit = 1.5 * x[:, 0] * x[:, 1] + 1.2 * np.sin(2 * x[:, 2]) - 0.8 * x[:, 3] ** 2 + 0.6
        probability = 1 / (1 + np.exp(-logit))
        y = (rng.uniform(0, 1, rows) < probability).astype(float)
        return {"X": x, "y": y, "features": [f"x{i}" for i in range(8)], "problem": "classification",
                "probability": probability, "known": {"informative": ["x0", "x1", "x2", "x3"]}}
    if name == "null_regression":
        x = rng.normal(0, 1, (rows, 10))
        y = rng.normal(0, 1, rows)
        return {"X": x, "y": y, "features": [f"x{i}" for i in range(10)], "problem": "regression",
                "known": {"noise_floor_rmse": 1.0, "informative": []}}
    raise ValueError(f"unknown built-in table {name!r}; available: {', '.join(BUILTINS)}")


def read_csv(path: Path, target: str, order_by: str | None) -> dict[str, Any]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise ValueError("table is empty")
        rows = [row for row in reader if row]
    if target not in header:
        raise ValueError(f"target column {target!r} is not in the table")
    if order_by is not None and order_by not in header:
        raise ValueError(f"order_by column {order_by!r} is not in the table")

    def number(text: str) -> float:
        text = text.strip()
        if text == "" or text.lower() in {"na", "nan", "null"}:
            return math.nan
        return float(text)

    columns = [name for name in header if name not in (target, order_by)]
    index = {name: header.index(name) for name in header}
    try:
        X = np.array([[number(row[index[name]]) for name in columns] for row in rows], dtype=float)
        y = np.array([number(row[index[target]]) for row in rows], dtype=float)
    except (ValueError, IndexError) as error:
        raise ValueError("every feature and target value must be numeric (empty means missing)") from error
    order = None
    if order_by is not None:
        order = [row[index[order_by]] for row in rows]
    if np.isnan(y).any():
        raise ValueError("the target column has missing values")
    return {"X": X, "y": y, "features": columns, "order": order}


class TabularTask:
    kind: Literal["tabular"] = "tabular"

    def __init__(self, settings: dict[str, Any], seed: int = 0) -> None:
        allowed = {"task", "table", "target", "problem", "split", "order_by", "rows", "metric", "table_seed"}
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError("unknown tabular settings: " + ", ".join(sorted(unknown)))
        self.settings = dict(settings)
        self.seed = int(seed)
        table = settings.get("table")
        if not isinstance(table, str) or not table:
            raise ValueError("tabular task needs table: builtin:<name> or a CSV path")
        split = {**DEFAULT_SPLIT, **settings.get("split", {})}
        if set(split) != set(DEFAULT_SPLIT) or any(not 0 < v < 1 for v in split.values()) \
                or abs(sum(split.values()) - 1) > 1e-9:
            raise ValueError("split must give development, validation and holdout fractions summing to 1")
        self.split = split
        data = self._load()
        self.problem = data["problem"]
        if self.problem not in ("regression", "classification"):
            raise ValueError("problem must be regression or classification")
        if self.problem == "classification" and not np.isin(data["y"], (0.0, 1.0)).all():
            raise ValueError("classification targets must be 0 or 1")
        self.metric = settings.get("metric") or ("rmse" if self.problem == "regression" else "log_loss")
        if self.metric not in ("rmse", "mae", "log_loss"):
            raise ValueError("metric must be rmse, mae (regression) or log_loss (classification)")
        self.features = data["features"]
        self.known = data.get("known", {})
        n = len(data["y"])
        if n < 50:
            raise ValueError("a tabular task needs at least 50 rows")
        if data.get("order") is not None:
            # Numbers sort as numbers (day 9 before day 1000); anything else must sort as text,
            # which is chronological for ISO dates (YYYY-MM-DD) and times.
            order = data["order"]
            try:
                keys = [float(value) for value in order]
            except ValueError:
                keys = list(order)
            ordering = np.array(sorted(range(n), key=lambda i: (keys[i], i)))
            self.split_mode = "chronological by " + settings["order_by"]
            self.ordered = True   # rows are a time series: inference must resample blocks, not rows
        else:
            ordering = np.random.default_rng(self.seed).permutation(n)
            self.split_mode = "seeded permutation"
            self.ordered = False
        dev_end = int(round(n * split["development"]))
        val_end = dev_end + int(round(n * split["validation"]))
        self._rows = {"development": ordering[:dev_end], "validation": ordering[dev_end:val_end],
                      "holdout": ordering[val_end:]}
        self._content = data["content_sha256"]
        # Keep development and validation only; the holdout is re-read on demand.
        self._arrays = {period: (data["X"][self._rows[period]], data["y"][self._rows[period]])
                        for period in ("development", "validation")}

    def _load(self) -> dict[str, Any]:
        table = self.settings["table"]
        if table.startswith("builtin:"):
            name = table.split(":", 1)[1]
            rows = int(self.settings.get("rows", 4000))
            data = builtin_table(name, rows=rows, seed=int(self.settings.get("table_seed", 7)))
            if self.settings.get("problem") not in (None, data["problem"]):
                raise ValueError(f"built-in {name} is a {data['problem']} task")
            data["content_sha256"] = _sha({"builtin": name, "rows": rows})
            return data
        path = Path(table)
        if not path.is_file():
            raise ValueError("table must be an existing CSV file")
        target = self.settings.get("target")
        if not isinstance(target, str):
            raise ValueError("a CSV task needs the target column name")
        data = read_csv(path, target, self.settings.get("order_by"))
        data["problem"] = self.settings.get("problem", "regression")
        data["content_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return data

    def identity(self) -> dict[str, Any]:
        return {"kind": self.kind, "content_sha256": self._content, "rows": {k: int(len(v)) for k, v in self._rows.items()},
                "features": self.features, "problem": self.problem, "metric": self.metric,
                "split": self.split, "split_mode": self.split_mode, "split_seed": self.seed}

    def development(self) -> tuple[np.ndarray, np.ndarray]:
        return self._arrays["development"]

    def validation(self) -> tuple[np.ndarray, np.ndarray]:
        return self._arrays["validation"]

    def holdout(self) -> dict[str, np.ndarray]:
        data = self._load()
        rows = self._rows["holdout"]
        out = {"X": data["X"][rows], "y": data["y"][rows]}
        if "probability" in data:
            out["probability"] = data["probability"][rows]
        return out


# ----------------------------------------------------------------------- panel
def rank_normalize(values: np.ndarray, members: np.ndarray) -> np.ndarray:
    """Per-day cross-sectional ranks scaled to [-0.5, 0.5] among members with finite values.

    Ties take their average rank. Vectorized over days: invalid cells sort last
    and are masked out, so a day's scaling uses only its own valid count.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = members & np.isfinite(values)
    keyed = np.where(valid, values, np.inf)
    order = np.argsort(keyed, axis=1, kind="mergesort")
    ranks = np.empty_like(keyed)
    rows = np.arange(keyed.shape[0])[:, None]
    ranks[rows, order] = np.arange(keyed.shape[1])[None, :]
    # Average tied ranks: equal sorted neighbours share the mean of their positions.
    sorted_values = np.take_along_axis(keyed, order, axis=1)
    out_sorted = np.take_along_axis(ranks, order, axis=1).astype(np.float64)
    with np.errstate(invalid="ignore"):
        tied_rows = np.flatnonzero((np.diff(sorted_values, axis=1) == 0).any(axis=1))
    for row in tied_rows:
        line = sorted_values[row]
        start = 0
        while start < len(line):
            stop = start
            while stop + 1 < len(line) and line[stop + 1] == line[start] and np.isfinite(line[start]):
                stop += 1
            if stop > start:
                out_sorted[row, start:stop + 1] = (start + stop) / 2
            start = stop + 1
    np.put_along_axis(ranks, order, out_sorted, axis=1)
    count = valid.sum(axis=1, keepdims=True).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = ranks / (count - 1) - 0.5
    return np.where(valid & (count >= 2), scaled, np.nan)


class PanelTask:
    kind: Literal["panel"] = "panel"
    PRED_FIELD = "lab_ml_prediction"

    def __init__(self, settings: dict[str, Any], seed: int = 0, runtime: Path | None = None) -> None:
        from autoresearch.backend import AlphaLabBackend

        allowed = {"task", "dataset", "universe", "development_start", "development_end", "validation_start",
                   "validation_end", "holdout_start", "cost_bps", "warmup", "decay", "neutralization", "truncation",
                   "pasteurization", "booksize", "features", "extra_features", "max_train_rows"}
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError("unknown panel settings: " + ", ".join(sorted(unknown)))
        self.settings = dict(settings)
        self.seed = int(seed)
        backend_keys = {k: v for k, v in settings.items()
                        if k not in ("task", "features", "extra_features", "max_train_rows")}
        from autoresearch.domains.alpha.evaluator import require_plain_snapshot

        # Before anything verifies the dataset (verifying a packed snapshot caches the whole archive in the process).
        require_plain_snapshot(Path(backend_keys["dataset"]))
        self.backend = AlphaLabBackend(backend_keys, Path(runtime or ".") / "engine")
        names = settings.get("features", "default")
        if names == "default":
            features = dict(FEATURE_LIBRARY)
        elif isinstance(names, list) and names and all(n in FEATURE_LIBRARY for n in names):
            features = {n: FEATURE_LIBRARY[n] for n in names}
        else:
            raise ValueError("features must be 'default' or a list of library feature names")
        for index, expression in enumerate(settings.get("extra_features", []) or []):
            if not isinstance(expression, str) or not expression.strip():
                raise ValueError("extra_features must be FASTEXPR expressions")
            features[f"extra_{index}"] = expression
        self.feature_expressions = features
        self.features = list(features)
        self.max_train_rows = int(settings.get("max_train_rows", 200_000))
        self.problem = "regression"
        self.metric = "ic"
        self._cache: dict[str, Any] = {}

    def identity(self) -> dict[str, Any]:
        return {"kind": self.kind, "alpha_lab": self.backend.identity(), "features": self.feature_expressions,
                "target": "cross-sectional rank of the delay-1 next-session return", "max_train_rows": self.max_train_rows}

    # ------------------------------------------------------------ sealed data
    # Everything the search touches is cached here, and all of it ends at the validation end: the
    # loader is bounded there (sealed_panel), so a holdout price is never deserialized into this
    # process by search-time code. There is no "full" workspace; the holdout path opens its own
    # panel in evaluate_holdout() and drops it.
    def _fields(self) -> list[str]:
        from autoresearch.domains.alpha.evaluator import workspace_fields

        return workspace_fields(self.backend.dataset, ("open", "high", "low", "close", "volume", "vwap"))

    def _workspace(self) -> Any:
        """The validation-sealed workspace: prices through ``validation_end``, bounded at load."""
        if "sealed" in self._cache:
            return self._cache["sealed"]
        from autoresearch.domains.alpha.evaluator import sealed_panel
        from qlib_yahoo.alphalab.simulate import Workspace

        panel = sealed_panel(self.backend.dataset, self.backend.bounds["validation_end"], self._fields())
        workspace = Workspace(self.backend.dataset, panel=panel)
        self._cache["sealed"] = workspace
        return workspace

    def _features(self, workspace: Any) -> dict[str, Any]:
        """Ranked features, purged next-session target and membership on ``workspace``'s sessions.

        The context starts ``warmup`` sessions before ``development_start`` (clipped to the
        workspace's first session) and runs to its last session.
        """
        from qlib_yahoo.alphalab.engine import evaluate

        settings = self.backend.settings
        members = workspace.universe(settings.universe)
        dates = workspace.dates
        first_row = int(np.searchsorted(dates, np.datetime64(self.backend.bounds["development_start"], "D")))
        first = max(0, first_row - settings.warmup)
        last = len(dates) - 1
        context = workspace.context(first, last, members)
        stack = []
        for name in self.features:
            values = np.asarray(evaluate(self.feature_expressions[name], context), dtype=np.float64)
            stack.append(rank_normalize(values, context.members))
        close = context.fields["close"].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            forward = np.full(close.shape, np.nan)
            forward[:-2] = close[2:] / close[1:-1] - 1
        tradable = np.isfinite(close)
        target = rank_normalize(forward, context.members & tradable)
        return {"X": np.stack(stack, axis=-1), "y": target, "members": context.members & tradable,
                "dates": context.dates, "offset": first, "rows": len(dates)}

    def _matrices(self) -> dict[str, Any]:
        """The sealed feature matrices (cached); their last session is the validation end."""
        if "matrices" not in self._cache:
            self._cache["matrices"] = self._features(self._workspace())
        return self._cache["matrices"]

    def _row_index(self, dates: np.ndarray, day: str, side: Literal["left", "right"] = "left") -> int:
        return int(np.searchsorted(dates, np.datetime64(day, "D"), side=side))

    def rows(self, period: str, *, inner: str | None = None) -> np.ndarray:
        """Flat (day, name) indices of usable rows; targets never cross the end of their split."""
        data = self._matrices()
        dates, bounds = data["dates"], self.backend.bounds
        if period == "development":
            start = self._row_index(dates, bounds["development_start"])
            end = self._row_index(dates, bounds["development_end"], "right") - 1 - 2  # purge: t + 2 <= split end
        elif period == "validation":
            start = self._row_index(dates, bounds["validation_start"])
            end = self._row_index(dates, bounds["validation_end"], "right") - 1 - 2
        elif period == "train_final":
            start = self._row_index(dates, bounds["development_start"])
            end = self._row_index(dates, bounds["validation_end"], "right") - 1 - 2
        else:
            raise ValueError("unknown panel period")
        if inner is not None:
            cut = start + int((end - start) * 0.75)
            start, end = (start, cut - 2) if inner == "train" else (cut, end)
        days = np.arange(start, end + 1)
        mask = np.zeros(data["y"].shape, dtype=bool)
        mask[days] = True
        mask &= data["members"] & np.isfinite(data["y"]) & np.isfinite(data["X"]).any(axis=-1)
        return np.flatnonzero(mask.ravel())

    def arrays(self, flat_rows: np.ndarray, *, limit: int | None = None,
               rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = self._matrices()
        k = data["X"].shape[-1]
        if limit is not None and len(flat_rows) > limit:
            rng = rng or np.random.default_rng(self.seed)
            flat_rows = np.sort(rng.choice(flat_rows, size=limit, replace=False))
        return data["X"].reshape(-1, k)[flat_rows], data["y"].ravel()[flat_rows], flat_rows

    def predict_matrix(self, model: Model, data: dict[str, Any] | None = None) -> np.ndarray:
        """``model``'s prediction for every usable (session, name) cell of ``data`` (the sealed matrices by default)."""
        if data is None:
            data = self._matrices()
        k = data["X"].shape[-1]
        flat = data["X"].reshape(-1, k)
        usable = data["members"].ravel() & np.isfinite(flat).any(axis=1)
        out = np.full(flat.shape[0], np.nan)
        if usable.any():
            out[usable] = model.predict(flat[usable])
        return out.reshape(data["y"].shape)

    def daily_ic(self, predictions: np.ndarray, flat_rows: np.ndarray) -> list[float]:
        data = self._matrices()
        mask = np.zeros(data["y"].size, dtype=bool)
        mask[flat_rows] = True
        mask = mask.reshape(data["y"].shape)
        values = []
        for row in np.flatnonzero(mask.any(axis=1)):
            ok = mask[row] & np.isfinite(predictions[row])
            if ok.sum() < 5:
                continue
            a = np.argsort(np.argsort(predictions[row, ok])).astype(float)
            b = np.argsort(np.argsort(data["y"][row, ok])).astype(float)
            if a.std() > 0 and b.std() > 0:
                values.append(float(np.corrcoef(a, b)[0, 1]))
        return values

    def simulate(self, predictions: np.ndarray, period: str, *, cost_multiplier: float = 1.0) -> dict[str, Any]:
        """Score a prediction matrix on ``period`` ("development" or "validation") with the unmodified
        Alpha Lab simulator, injected as a data field of the sealed workspace."""
        if period not in ("development", "validation"):
            raise ValueError("the sealed panel task scores development or validation; the reserved period "
                             "is scored by evaluate_holdout only")
        bounds = self.backend.bounds
        return self._simulate_on(self._workspace(), self._matrices(), predictions,
                                 bounds[period + "_start"], bounds[period + "_end"], cost_multiplier)

    def _simulate_on(self, workspace: Any, data: dict[str, Any], predictions: np.ndarray, start: str,
                     end: str | None, cost_multiplier: float) -> dict[str, Any]:
        from qlib_yahoo.alphalab.simulate import simulate

        field = np.full((len(workspace.dates), len(workspace.symbols)), np.nan, dtype=np.float32)
        field[data["offset"]:data["offset"] + predictions.shape[0]] = predictions.astype(np.float32)
        workspace.fields[self.PRED_FIELD] = field
        try:
            settings = replace(self.backend.settings, start=start, end=end,
                               cost_bps=self.backend.settings.cost_bps * cost_multiplier)
            return simulate(self.PRED_FIELD, settings, workspace)
        finally:
            workspace.fields.pop(self.PRED_FIELD, None)

    # ----------------------------------------------------------------- holdout
    def evaluate_holdout(self, model: Model) -> dict[str, Any]:
        """Predict and score a fitted ``model`` on the reserved sessions: the one method that reads a holdout price.

        The panel comes from :func:`autoresearch.domains.alpha.holdout.open_holdout_panel`, is used
        for this call and dropped: ``_cache`` never holds a session after the validation end. The
        window starts where the sealed features start (``development_start`` less the warm-up), not
        merely a warm-up before the holdout: a rolling feature at a holdout session is then the same
        floating-point number the whole snapshot gives (pandas' rolling sums carry rounding from the
        start of a series), so the bounded load leaves the recorded holdout numbers unchanged.
        """
        from autoresearch.domains.alpha.holdout import open_holdout_panel
        from qlib_yahoo.alphalab.simulate import Workspace

        start, end = self.backend.holdout_bounds()
        panel = open_holdout_panel(self.backend.dataset, self._fields(), holdout_start=start,
                                   warmup=self.backend.settings.warmup,
                                   history_from=self.backend.bounds["development_start"])
        workspace = Workspace(self.backend.dataset, panel=panel)
        data = self._features(workspace)
        return self._simulate_on(workspace, data, self.predict_matrix(model, data), start, end, 1.0)

    def release(self) -> None:
        self._cache.clear()
