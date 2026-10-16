"""One command to refresh the market data: tickers -> Yahoo charts -> table -> snapshot -> quality scan.

Everything goes into new dated folders under ``<data root>/sources`` and
``<data root>/datasets``; nothing earlier is overwritten. The ticker list is
every stock in today's Nasdaq screener (all US exchanges) plus the tickers of
earlier tables, so new listings enter without Stooq and names that have since
left the screener are still fetched while Yahoo serves them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from qlib_yahoo.dataset import DatasetError


def ticker_list(screener_path: Path, previous_tables: Iterable[Path] = ()) -> list[str]:
    """Screener stocks (Nasdaq's symbol style converted) plus tickers from earlier tables."""
    from qlib_yahoo.reference import Screener
    from qlib_yahoo.universe import exclusion_reason

    screener = Screener(screener_path)
    tickers = set(screener.rows)
    for table in previous_tables:
        table = Path(table)
        if table.suffix.lower() != ".parquet":
            raise DatasetError(f"previous tables must be Parquet: {table}")
        import pyarrow.parquet as pq

        tickers |= set(pq.read_table(table, columns=["ticker"]).column("ticker").unique().to_pylist())
    # The importer's own ticker rules (test symbols, series, reserved names) apply before fetching.
    return sorted(t for t in tickers if t and exclusion_reason(t, "stock") is None)


def refresh(data_root: Path, *, previous_tables: Iterable[Path] = (), chart_start: str = "2009-12-01",
            import_start: str = "2010-01-01", min_price: float = 1.0, universe_size: int = 3000,
            make_latest: bool = False, workers: int = 6, progress: Callable[[str], None] | None = None,
            fetch: Callable[..., Any] | None = None, opener: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Run every step; return the paths produced and the quality summary."""
    from qlib_yahoo.data_quality import scan, summary_lines
    from qlib_yahoo.reference import fetch_screener
    from qlib_yahoo.stooq_table import prepare_stooq_table
    from qlib_yahoo.universe import UniverseRule
    from qlib_yahoo.yahoo_chart import fetch_archive, fetch_chart
    from qlib_yahoo.yahoo_table import build_table

    say = progress or (lambda _message: None)
    data_root = Path(data_root)
    now = datetime.now(timezone.utc)
    day, stamp = f"{now:%Y%m%d}", f"{now:%Y%m%dT%H%M%SZ}"
    sources = data_root / "sources"
    sources.mkdir(parents=True, exist_ok=True)

    say("1/5 Nasdaq screener (security names, sector, industry)")
    screener = fetch_screener(sources / f"nasdaq-screener-{stamp}",
                              **({"opener": opener} if opener is not None else {}))
    tickers = ticker_list(screener, previous_tables)
    say(f"2/5 Yahoo charts for {len(tickers):,} tickers (resumable)")
    charts = sources / f"yahoo-chart-{day}"
    tomorrow = datetime.fromtimestamp(now.timestamp() + 86400, timezone.utc).date().isoformat()
    fetch_archive(tickers, charts, chart_start, tomorrow, workers=workers, progress=say, fetch=fetch or fetch_chart)
    say("3/5 Consolidated table")
    table_dir = sources / f"yahoo-table-{stamp}"
    table_dir.mkdir()
    table = table_dir / "table.parquet"
    built = build_table(charts, screener, table, progress=say)
    end = built["cutoff"]["dropped_from"]  # exclusive: the first possibly partial session
    say(f"4/5 Snapshot {import_start} to {end} (end exclusive), TOP{universe_size}, min price {min_price:g}")
    snapshot = prepare_stooq_table(table, data_root, import_start, end,
                                   rule=UniverseRule(name=f"top{universe_size}", size=universe_size, min_price=min_price),
                                   groups=screener, make_latest=make_latest, progress=say)
    say("5/5 Quality scan")
    report = scan(snapshot)
    for line in summary_lines(report):
        say(line)
    record = {"created_at_utc": now.isoformat(), "screener": str(screener), "charts": str(charts), "table": str(table),
              "snapshot": str(snapshot), "tickers": len(tickers), "quality": report["totals"]}
    (table_dir / "refresh.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record
