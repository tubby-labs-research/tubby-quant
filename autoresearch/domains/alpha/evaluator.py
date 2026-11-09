"""In-process Alpha Lab evaluation on period-sealed copies of a dataset.

The search screens thousands of expressions, so it cannot afford a disk-cached
batch run per candidate. :class:`SealedEvaluator` loads the snapshot once,
**through the validation end only**, and builds two in-memory workspaces whose
price arrays physically end at the development end and at the validation end.
The screen runs on the first, the validation probes (cost stress, sub-periods,
window perturbations) on the second, and neither can read a holdout price
because none is in memory: the loader itself stops at the validation end
(:func:`sealed_panel`), so a later session is never deserialized in a search
process, whatever code runs in it. The reserved period is opened only by
:func:`autoresearch.domains.alpha.holdout.open_holdout_panel`, which nothing
in this module or on the search path calls.

Both run the unmodified :func:`qlib_yahoo.alphalab.simulate.simulate`, so the
numbers equal the canonical backend's for the same expression and settings;
``tests/test_lab_alpha.py`` checks that agreement.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from autoresearch.lab.selection import moments

if TYPE_CHECKING:
    from qlib_yahoo.alphalab.settings import Settings
    from qlib_yahoo.alphalab.simulate import Workspace
    from qlib_yahoo.panel import Panel

#: The price fields a workspace carries when the snapshot has them (the simulator's own default list).
WORKSPACE_FIELDS = ("open", "high", "low", "close", "volume", "close_raw", "dividend", "cap", "vwap")
SEARCH_PERIODS = ("development", "validation")


def workspace_fields(dataset: Path, wanted: Sequence[str] = WORKSPACE_FIELDS) -> list[str]:
    """The fields of ``wanted`` that every instrument of the snapshot carries."""
    from qlib_yahoo.panel import available_fields

    manifest = json.loads((Path(dataset) / "manifest.json").read_text(encoding="utf-8"))
    return [name for name in wanted if name in available_fields(manifest)]


def require_plain_snapshot(dataset: Path) -> None:
    """Refuse a packed snapshot: sealing is structural for plain snapshots only.

    A packed snapshot (``storage`` in its manifest) is verified by reading every archive entry into
    a module-global cache (``qlib_yahoo.packed.verified_entries``), and ``load_panel`` then slices a
    view of those bytes: the window would bound what a panel shows, not what the process holds, so
    every session of the snapshot, the reserved ones included, would sit in the search process for
    its lifetime. Until the packed reader verifies by streaming and reads row ranges, the lab works
    on the plain snapshot a packed one was made from.
    """
    manifest = json.loads((Path(dataset) / "manifest.json").read_text(encoding="utf-8"))
    if "storage" in manifest:
        raise ValueError(f"{dataset}: sealed search needs a plain snapshot; a packed snapshot is verified into memory "
                         "whole (qlib_yahoo/packed.py), which would keep the reserved sessions in the search process. "
                         "Use the plain snapshot the packed one was made from.")


def sealed_panel(dataset: Path, end_date: str, fields: Sequence[str] | None = None) -> Panel:
    """The snapshot's price panel through ``end_date`` inclusive, bounded by the loader itself.

    This is the only way search-time code (screens, validations, their caches) loads prices: the
    rows after ``end_date`` are never read, so no object in the search process holds them. The
    reserved period has its own, separately named path (``autoresearch.domains.alpha.holdout``).
    A packed snapshot is refused (:func:`require_plain_snapshot`).
    """
    from qlib_yahoo.panel import load_panel

    require_plain_snapshot(dataset)
    return load_panel(Path(dataset), list(fields) if fields is not None else workspace_fields(dataset),
                      end_date=end_date)


class SealedEvaluator:
    def __init__(self, dataset: Path, settings: Settings, bounds: dict[str, Any]) -> None:
        self.dataset = Path(dataset)
        self.settings = settings          # qlib_yahoo.alphalab.settings.Settings (delay 1, costs, neutralization)
        self.bounds = bounds              # development_start/_end, validation_start/_end (ISO dates)
        # The snapshot through the validation end, loaded once with the loader's own bound: no later
        # session is deserialized in this process. The development workspace is a slice of it.
        self._panel: Panel | None = None
        self._workspaces: dict[str, Workspace] = {}

    def _workspace(self, period: str) -> Workspace:
        if period in self._workspaces:
            return self._workspaces[period]
        if period not in SEARCH_PERIODS:
            raise ValueError("the sealed evaluator only holds development or validation workspaces")
        from qlib_yahoo.alphalab.simulate import Workspace

        if self._panel is None:
            self._panel = sealed_panel(self.dataset, self.bounds["validation_end"])
        panel = self._panel
        end = np.datetime64(self.bounds[period + "_end"], "D")
        rows = int(np.searchsorted(panel.dates, end, side="right"))
        sealed = panel if rows == len(panel.dates) else replace(
            panel, dates=panel.dates[:rows], fields={name: values[:rows] for name, values in panel.fields.items()})
        workspace = Workspace(self.dataset, panel=sealed)
        self._workspaces[period] = workspace
        return workspace

    def release(self) -> None:
        self._workspaces.clear()
        self._panel = None

    def simulate(self, expression: str, period: str, **overrides: Any) -> dict[str, Any]:
        """One Alpha Lab run on ``period`` ("development" or "validation") of its sealed workspace."""
        from qlib_yahoo.alphalab.simulate import simulate

        if period not in SEARCH_PERIODS:
            raise ValueError("the sealed evaluator only scores development or validation")
        settings = replace(self.settings, start=self.bounds[period + "_start"], end=self.bounds[period + "_end"],
                           **overrides)
        return simulate(expression, settings, self._workspace(period))

    def alpha_matrix(self, expression: str, period: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raw alpha values, membership and dates for ``period`` (no weights, no PnL)."""
        from qlib_yahoo.alphalab.engine import evaluate

        workspace = self._workspace(period)
        settings = replace(self.settings, start=self.bounds[period + "_start"], end=self.bounds[period + "_end"])
        members = workspace.universe(settings.universe)
        first, start, end = workspace.window(settings)
        context = workspace.context(first, end, members)
        alpha = evaluate(expression, context)
        offset = start - first
        return np.asarray(alpha[offset:], dtype=np.float64), context.members[offset:], context.dates[offset:]


def bucketed(series: Any, buckets: int = 128) -> list[float]:
    """Sum a daily series into at most ``buckets`` consecutive chunks (a compact diversity signature)."""
    values = np.asarray(series, dtype=np.float64)
    if len(values) <= buckets:
        return values.tolist()
    edges = np.linspace(0, len(values), buckets + 1).astype(int)
    return [float(values[a:b].sum()) for a, b in zip(edges[:-1], edges[1:])]


def daily_profile(pnl: Any) -> dict[str, Any]:
    """Per-day Sharpe, skewness and kurtosis of a daily PnL series (for the deflated Sharpe ratio)."""
    stats = moments(pnl)
    mean, std = stats["mean"], stats["std"]
    sharpe = mean / std if std and mean is not None else None
    return {"days": stats["n"], "daily_sharpe": sharpe, "skew": stats["skew"], "kurtosis": stats["kurtosis"]}


def sharpe_of(pnl: Any) -> float | None:
    stats = moments(pnl)
    mean, std = stats["mean"], stats["std"]
    if not std or mean is None:
        return None
    return float(mean / std * np.sqrt(252))
