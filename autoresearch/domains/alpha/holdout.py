"""The reserved period's data path: the only way holdout prices enter a lab process.

Search-time code (the sealed evaluator, the ML panel task's caches, every screen
and validation) loads prices through
:func:`autoresearch.domains.alpha.evaluator.sealed_panel`, which the loader
bounds at the validation end. The holdout is read through
:func:`open_holdout_panel` alone, from inside a domain's ``evaluate_holdout``,
and the panel it returns is a local of that call: nothing caches it, so the next
screen or validation in the same process finds no holdout price in memory.

Nothing on the search path imports this module. A campaign reaches it only
through ``Domain.evaluate_holdout`` (``LabCampaign.finalize``), which is also
where access is reserved and counted.

Packed snapshots are refused here as in ``sealed_panel``
(:func:`autoresearch.domains.alpha.evaluator.require_plain_snapshot`): their
reader verifies the archive by keeping every entry in a module-global cache, so
a bounded window would not bound what the process holds.

Next step, designed but not built: run ``evaluate_holdout`` in a child process
that opens the panel, scores, returns the result and exits, so that even the
finalizing process never holds holdout prices beside the search caches. The
in-process version here already gives the same window and the same numbers;
the child-process version would wrap :func:`open_holdout_panel` plus the
domain's scoring in ``multiprocessing`` with the result JSON as the only
channel back.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .evaluator import require_plain_snapshot, workspace_fields

if TYPE_CHECKING:
    from qlib_yahoo.panel import Panel


def holdout_window(dataset: Path, *, holdout_start: str, warmup: int, history_from: str | None = None) -> tuple[str, str]:
    """(first session, last session) the holdout evaluation reads.

    The window runs from ``warmup`` sessions before ``holdout_start`` (the simulator's warm-up: the
    weights held on the first holdout session come from the alpha of the session before it, and
    every time-series operator needs its lookback) through the snapshot's last session. A dated
    universe with a history rule (``min_history_sessions``) needs that many sessions more before
    the warm-up, so its eligibility on the warm-up sessions equals the full snapshot's.
    ``history_from`` moves the start further back, to ``warmup`` sessions before that session: the
    ML panel task uses it so a holdout feature is computed from the same first row as the sealed
    features (pandas' rolling sums carry rounding from the start of a series, so a different first
    row would change a holdout number at machine precision).
    """
    import json

    from qlib_yahoo.universe import dataset_universes

    dataset = Path(dataset)
    calendar = (dataset / "qlib_data" / "calendars" / "day.txt").read_text(encoding="utf-8").split()
    dates = np.asarray(calendar, dtype="datetime64[D]")
    if not len(dates):
        raise ValueError("the dataset has no sessions")
    anchor = holdout_start
    if history_from is not None:
        if history_from > holdout_start:
            raise ValueError("history_from must not be after holdout_start")
        anchor = history_from
    row = int(np.searchsorted(dates, np.datetime64(anchor, "D")))
    if row >= len(dates):
        raise ValueError(f"{anchor} is after the dataset's last session {dates[-1]}")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    history = max((int(record.get("min_history_sessions", 1)) for record in dataset_universes(manifest).values()
                   if record.get("mode") == "historical_index"), default=1)
    first = max(0, row - int(warmup) - (history - 1))
    return str(dates[first]), str(dates[-1])


def open_holdout_panel(dataset: Path, fields: Sequence[str] | None = None, *, holdout_start: str, warmup: int,
                       history_from: str | None = None) -> Panel:
    """Load exactly the sessions a holdout evaluation reads (see :func:`holdout_window`).

    The only loader of reserved sessions in the lab. Call it inside ``evaluate_holdout``, keep the
    panel local to that call, and never cache it on a domain or task: the structural guarantee that
    the search process holds no session after the validation end depends on that.
    """
    from qlib_yahoo.panel import load_panel

    dataset = Path(dataset)
    require_plain_snapshot(dataset)
    first, last = holdout_window(dataset, holdout_start=holdout_start, warmup=warmup, history_from=history_from)
    panel = load_panel(dataset, list(fields) if fields is not None else workspace_fields(dataset),
                       start_date=first, end_date=last)
    if str(panel.dates[0]) != first or str(panel.dates[-1]) != last:
        raise ValueError("the loader returned a different window than the holdout evaluation asked for")
    return panel
