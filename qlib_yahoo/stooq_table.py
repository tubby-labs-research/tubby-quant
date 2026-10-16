"""Convert a consolidated Stooq US daily table into a Qlib snapshot with a dated universe.

The table is one long file (Parquet or CSV) with columns
``ticker, exchange, type, date, open, high, low, close, volume``, or Stooq's
official daily US ASCII download (``d_us_txt.zip``) read member by member
without extraction. It is read in streamed batches and never modified. Output
reuses the pinned Qlib ``dump_bin.py`` and is verified by reading every produced
binary back against the normalized values.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from qlib_yahoo import QLIB_COMMIT, QLIB_RELEASE
from qlib_yahoo.dataset import DatasetError, sha256, tree_sha256
from qlib_yahoo.stooq import HEADER as ASCII_HEADER, _dump_script, _inside, _write_json_atomically
from qlib_yahoo.universe import (UniverseRule, exclusion_reason, liquidity_membership,
                                 membership_spans, rank_sessions, write_instrument_file)


COLUMNS = ("ticker", "exchange", "type", "date", "open", "high", "low", "close", "volume")
PRICE_FIELDS = ("open", "high", "low", "close", "volume")
FIELDS = (*PRICE_FIELDS, "factor")
ACTION_FIELDS = ("close_raw", "dividend")  # added when Yahoo corporate actions are supplied
# Optional columns a consolidated table may carry (one built from Yahoo or Sharadar): the traded (unadjusted)
# close, the cash dividend on ex-dates and the market capitalization in dollars. Qlib's factor stays 1:
# the runner's contract requires it.
TABLE_EXTRAS = ("close_raw", "dividend", "cap", "vwap")
# Tables written by build-yahoo-table / build-sharadar-table carry a sidecar manifest naming their sources.
BUILDERS = {"yahoo_chart": "yahoo", "sharadar": "sharadar", "historicaldata": "historicaldata"}


def table_extras(table: Path) -> tuple[str, ...]:
    """The optional TABLE_EXTRAS columns a Parquet or CSV table provides (the ASCII zip has none)."""
    suffix = Path(table).suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        names = set(pq.ParquetFile(table).schema_arrow.names)
    elif suffix == ".csv":
        import pyarrow.csv as pacsv

        names = set(pacsv.open_csv(table).schema.names)
    else:
        return ()
    return tuple(name for name in TABLE_EXTRAS if name in names)
EXAMPLES = 20
SPARSE_LOOKBACK = 20
# Stooq's daily US ASCII zip: data/daily/us/<exchange> <stocks|etfs>/[<bucket>/]<ticker>.us.txt
ARCHIVE_EXCHANGES = {"nasdaq": "NASDAQ", "nyse": "NYSE", "nysemkt": "NYSEMKT"}
ARCHIVE_TYPES = {"stocks": "stock", "etfs": "etf"}
ARCHIVE_BATCH_ROWS = 1 << 20


def _log(progress: Callable[[str], None] | None, message: str) -> None:
    (progress or (lambda text: print(text, flush=True)))(message)


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"sha256": sha256(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _safe_output(table: Path, data_root: Path) -> tuple[Path, Path]:
    if table.is_symlink():
        raise DatasetError(f"Stooq table is a redirect: {table}")
    table, output = table.resolve(), data_root.resolve()
    if not table.is_file():
        raise DatasetError(f"Stooq table is missing: {table}")
    if table.suffix.lower() not in (".parquet", ".csv", ".zip"):
        raise DatasetError("Stooq table must be a .parquet or .csv table or Stooq's daily US ASCII .zip")
    # The importer writes <data root>/datasets and <data root>/latest.json only, so a table kept elsewhere
    # under the data root (e.g. <data root>/sources) is safe; a data root inside the table's folder is not.
    protected, written = table.parent, output / "datasets"
    if _inside(output, protected) or _inside(protected, written):
        raise DatasetError(f"output {output} overlaps read-only Stooq table folder {protected}")
    return table, output


def _check_interval(start: str, end: str) -> tuple[date, date]:
    try:
        begin, stop = date.fromisoformat(start), date.fromisoformat(end)
        if begin.isoformat() != start or stop.isoformat() != end or begin >= stop:
            raise ValueError("interval")
    except (TypeError, ValueError) as error:
        raise DatasetError("start/end require YYYY-MM-DD with inclusive start and exclusive later end") from error
    return begin, stop


def _strict_dates(values: Any, fmt: str) -> Any:
    """Parse dates as second timestamps; impossible days become null instead of rolling forward.

    Arrow's strptime turns 2023-02-30 into 2023-03-02, so each parse is checked by formatting it back.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    parsed = pc.strptime(values, format=fmt, unit="s", error_is_null=True)
    return pc.if_else(pc.equal(pc.strftime(parsed, format=fmt), values), parsed, pa.scalar(None, pa.timestamp("s")))


def _archive_batches(table: Path, empty: list[str] | None = None) -> Any:
    """Yield record batches with COLUMNS from Stooq's daily US ASCII zip, checking every member.

    Exchange and type come from the member's folder. Members without bars are
    skipped and listed in ``empty``; any other deviation from the layout fails.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv

    prices = ("<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>", "<OPENINT>")
    types = {**{name: pa.string() for name in ASCII_HEADER if name not in prices}, **{name: pa.float64() for name in prices}}
    pending, rows = [], 0
    with zipfile.ZipFile(table) as archive:
        for info in archive.infolist():
            name = info.filename
            if info.is_dir() or name.startswith("__MACOSX/") or not name.lower().endswith(".us.txt"):
                continue
            parts = name.split("/")
            group = parts[3].split(" ") if len(parts) in (5, 6) and parts[:3] == ["data", "daily", "us"] else []
            if len(group) != 2 or group[0] not in ARCHIVE_EXCHANGES or group[1] not in ARCHIVE_TYPES:
                raise DatasetError(f"unexpected Stooq archive member {name}; expected data/daily/us/<exchange> <stocks|etfs>/...")
            content = archive.read(info)
            frame = None
            if content.strip():
                try:
                    frame = pacsv.read_csv(io.BytesIO(content), convert_options=pacsv.ConvertOptions(column_types=types))
                except pa.ArrowInvalid as error:
                    raise DatasetError(f"cannot parse Stooq archive member {name}: {error}") from error
                if tuple(frame.column_names) != ASCII_HEADER:
                    raise DatasetError(f"Stooq archive member {name} has an unexpected header")
            if frame is None or frame.num_rows == 0:
                if empty is not None:
                    empty.append(name)
                continue
            symbol = parts[-1][:-len(".us.txt")].upper()
            for column, expected in (("<TICKER>", f"{symbol}.US"), ("<PER>", "D"), ("<TIME>", "000000")):
                if pc.all(pc.equal(frame[column], expected), skip_nulls=False).as_py() is not True:
                    raise DatasetError(f"Stooq archive member {name}: every {column} must be {expected}")
            count = frame.num_rows
            pending.append(pa.table({
                "ticker": pa.repeat(symbol, count), "exchange": pa.repeat(ARCHIVE_EXCHANGES[group[0]], count),
                "type": pa.repeat(ARCHIVE_TYPES[group[1]], count),
                "date": _strict_dates(frame["<DATE>"], "%Y%m%d"),
                **{field: frame[f"<{'VOL' if field == 'volume' else field.upper()}>"] for field in PRICE_FIELDS}}))
            rows += count
            if rows >= ARCHIVE_BATCH_ROWS:
                yield from pa.concat_tables(pending).combine_chunks().to_batches()
                pending, rows = [], 0
    if pending:
        yield from pa.concat_tables(pending).combine_chunks().to_batches()


def _batches(table: Path, columns: list[str], begin: date, stop: date, empty: list[str] | None = None) -> Any:
    """Stream record batches of ``columns`` inside [begin, stop); dates become second timestamps."""
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.csv as pacsv
        import pyarrow.parquet as pq
    except ImportError as error:
        raise DatasetError("reading the Stooq table needs pyarrow; install requirements-qlib-table.txt") from error
    low = pa.scalar(datetime(begin.year, begin.month, begin.day), pa.timestamp("s"))
    high = pa.scalar(datetime(stop.year, stop.month, stop.day), pa.timestamp("s"))
    try:
        if table.suffix.lower() == ".parquet":
            source = pq.ParquetFile(table)
            missing = set(COLUMNS) - set(source.schema_arrow.names)
            if missing:
                raise DatasetError(f"Stooq table lacks columns {sorted(missing)}")
            batches = source.iter_batches(batch_size=1 << 20, columns=columns)
        elif table.suffix.lower() == ".zip":
            batches = (batch.select(columns) for batch in _archive_batches(table, empty))
        else:
            header = pacsv.open_csv(table, read_options=pacsv.ReadOptions(block_size=1 << 20)).schema.names
            missing = set(COLUMNS) - set(header)
            if missing:
                raise DatasetError(f"Stooq table lacks columns {sorted(missing)}")
            types = {"ticker": pa.string(), "exchange": pa.string(), "type": pa.string(), "date": pa.string(),
                     **{name: pa.float64() for name in (*PRICE_FIELDS, *TABLE_EXTRAS)}}
            batches = pacsv.open_csv(table, read_options=pacsv.ReadOptions(block_size=16 << 20),
                                     convert_options=pacsv.ConvertOptions(
                                         include_columns=columns, column_types={c: types[c] for c in columns},
                                         strings_can_be_null=True, null_values=[""]))
        for batch in batches:
            day = batch.column("date")
            if pa.types.is_string(day.type) or pa.types.is_large_string(day.type):
                day = _strict_dates(day, "%Y-%m-%d")
            day = pc.cast(day, pa.timestamp("s"))
            for name in ("ticker", "type"):
                if name in columns and batch.column(name).null_count:
                    raise DatasetError(f"Stooq table has empty {name} values")
            if day.null_count:
                raise DatasetError("Stooq table has empty or unparseable dates")
            batch = batch.set_column(batch.schema.get_field_index("date"), "date", day)
            yield batch.filter(pc.and_(pc.greater_equal(day, low), pc.less(day, high)))
    except (pa.ArrowInvalid, pa.ArrowTypeError, OSError, zipfile.BadZipFile) as error:
        raise DatasetError(f"cannot read Stooq table {table.name}: {error}") from error


def _securities(table: Path, begin: date, stop: date, empty: list[str] | None = None) -> tuple[Any, int]:
    """First pass: distinct (ticker, type) pairs and the row count inside the interval."""
    import pandas as pd
    import pyarrow as pa

    pairs = []
    rows = 0
    for batch in _batches(table, ["ticker", "type", "date"], begin, stop, empty):
        rows += batch.num_rows
        if batch.num_rows:
            pairs.append(pa.Table.from_batches([batch]).group_by(["ticker", "type"]).aggregate([]))
    if not pairs:
        return pd.DataFrame(columns=["ticker", "type"]), 0
    return pa.concat_tables(pairs).group_by(["ticker", "type"]).aggregate([]).to_pandas(), rows


def common_stock_tickers(table: Path, start: str, end: str) -> list[str]:
    """Tickers the importer would treat as common stock in ``[start, end)``, e.g. to fetch their corporate actions."""
    begin, stop = _check_interval(start, end)
    securities, _ = _securities(Path(table), begin, stop)
    return sorted(ticker for ticker, kind in zip(securities["ticker"], securities["type"])
                  if exclusion_reason(ticker, kind) is None)


def _sparse_sessions(counts: Any, coverage: float, lookback: int = SPARSE_LOOKBACK) -> Any:
    """Sessions whose bar count is below ``coverage`` of the trailing median.

    The first sessions of a window lack five earlier sessions; they are compared with
    the median of the following sessions instead, so an opening holiday is also caught.
    """
    trailing = counts.shift(1).rolling(lookback, min_periods=5).median()
    leading = counts[::-1].shift(1).rolling(lookback, min_periods=5).median()[::-1]
    reference = trailing.fillna(leading)
    return counts.index[(counts < coverage * reference).to_numpy()]


def _stock_frame(table: Path, begin: date, stop: date, stocks: list[str], coverage: float,
                 extras: tuple[str, ...] = ()) -> tuple[Any, list[dict[str, Any]], int]:
    """Second pass: selected stocks without sparse sessions, compact and sorted by (ticker, date).

    Returns the frame, a report of dropped sparse sessions and the dropped row count.
    """
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc

    # The system allocator returns freed Arrow buffers to the OS; pooled allocators keep them resident.
    previous = pa.default_memory_pool()
    pa.set_memory_pool(pa.system_memory_pool())
    try:
        wanted = pa.array(stocks, pa.string())
        kept = [batch.filter(pc.is_in(batch.column("ticker"), value_set=wanted))
                for batch in _batches(table, ["ticker", "date", *PRICE_FIELDS, *extras], begin, stop)]
        arrow = pa.Table.from_batches([batch for batch in kept if batch.num_rows])
        del kept
        counted = arrow.group_by("date").aggregate([("ticker", "count")]).to_pandas()
        counts = counted.set_index("date")["ticker_count"].sort_index()
        counts.index = pd.DatetimeIndex(counts.index)
        sparse = _sparse_sessions(counts, coverage)
        report, dropped = [], 0
        if len(sparse):
            dropped_mask = pc.is_in(arrow["date"], value_set=pa.array(sparse.to_numpy().astype("datetime64[s]")))
            for day in sparse:
                on_day = arrow.filter(pc.equal(arrow["date"], pa.scalar(day.to_pydatetime(), pa.timestamp("s"))))
                report.append({"session": f"{day:%Y-%m-%d}", "bars": int(counts[day]),
                               "examples": sorted(on_day["ticker"].to_pylist())[:EXAMPLES]})
            dropped = int(pc.sum(pc.cast(dropped_mask, pa.int64())).as_py() or 0)
            arrow = arrow.filter(pc.invert(dropped_mask))
            del dropped_mask
        # Byte-wise ticker order equals Python's sorted() order for these ASCII symbols.
        arrow = arrow.sort_by([("ticker", "ascending"), ("date", "ascending")])
        frame = arrow.to_pandas(strings_to_categorical=True, date_as_object=False, coerce_temporal_nanoseconds=True,
                                self_destruct=True, split_blocks=True)
        del arrow
    finally:
        pa.set_memory_pool(previous)
    frame["ticker"] = frame["ticker"].cat.set_categories(sorted(stocks))
    for column in (*PRICE_FIELDS, *extras):
        if frame[column].dtype != np.float64:
            raise DatasetError(f"Stooq column {column} is not numeric")
    return frame, report, dropped


def _duplicate_example(frame: Any) -> str | None:
    """Rows are sorted by (ticker, date): a duplicate is an equal neighbour."""
    import numpy as np

    codes = frame["ticker"].cat.codes.to_numpy()
    days = frame["date"].to_numpy()
    same = (codes[1:] == codes[:-1]) & (days[1:] == days[:-1])
    if not same.any():
        return None
    row = int(np.flatnonzero(same)[0]) + 1
    return f"{frame['ticker'].iloc[row]} {frame['date'].iloc[row]:%Y-%m-%d}"


def _invalid_bars(frame: Any) -> tuple[Any, dict[str, Any]]:
    """Mark unusable bars column by column; zero volume is kept separately (as in the per-file importer).

    Optional table columns must be usable too: close_raw finite and positive, dividend
    finite and not negative.
    """
    import numpy as np

    op, high, low, close, volume = (frame[name].to_numpy() for name in PRICE_FIELDS)
    nonfinite = ~np.isfinite(op)
    for column in (high, low, close, volume):
        nonfinite |= ~np.isfinite(column)
    with np.errstate(invalid="ignore"):
        smallest = np.minimum(np.minimum(op, high), np.minimum(low, close))
        nonpositive = ~nonfinite & (smallest <= 0)
        del smallest
        negative_volume = ~nonfinite & (volume < 0)
        inconsistent = ~nonfinite & ~nonpositive & (
            (high < np.maximum(np.maximum(op, low), close)) | (low > np.minimum(np.minimum(op, high), close)))
    reasons = {"nonfinite": nonfinite, "nonpositive_price": nonpositive,
               "negative_volume": negative_volume, "inconsistent_ohlc": inconsistent}
    extra_bad = np.zeros(len(frame), dtype=bool)
    with np.errstate(invalid="ignore"):
        if "close_raw" in frame:
            values = frame["close_raw"].to_numpy()
            extra_bad |= ~(np.isfinite(values) & (values > 0))
        if "dividend" in frame:
            values = frame["dividend"].to_numpy()
            extra_bad |= ~(np.isfinite(values) & (values >= 0))
    if any(name in frame for name in TABLE_EXTRAS):
        reasons["invalid_table_extras"] = extra_bad & ~(nonfinite | nonpositive | negative_volume | inconsistent)
    report = {}
    for reason, mask in reasons.items():
        rows = frame.loc[mask, ["ticker", "date"]].head(EXAMPLES)
        report[reason] = {"rows": int(mask.sum()),
                          "examples": [f"{t} {d:%Y-%m-%d}" for t, d in zip(rows["ticker"], rows["date"])]}
    return nonfinite | nonpositive | negative_volume | inconsistent | extra_bad, report


def _panel(frame: Any, rows: Any, values: Any, fill: Any) -> Any:
    """Dates x tickers matrix filled by position; avoids pandas pivot's large temporaries."""
    import numpy as np

    values = np.asarray(values)
    matrix = np.full((len(rows), len(frame["ticker"].cat.categories)), fill, dtype=values.dtype)
    matrix[rows.get_indexer(frame["date"]), frame["ticker"].cat.codes.to_numpy()] = values
    return matrix


def _symbol_slices(frame: Any) -> list[tuple[str, int, int]]:
    """(symbol, start, stop) row ranges of a frame sorted by symbol then date."""
    import numpy as np

    codes = frame["symbol"].cat.codes.to_numpy()
    edges = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(codes)]))
    names = frame["symbol"].cat.categories
    return [(str(names[codes[a]]), int(a), int(b)) for a, b in zip(starts, stops)]


def _write_normalized(frame: Any, directory: Path, workers: int,
                      fields: tuple[str, ...] = FIELDS) -> dict[str, tuple[str, str, int]]:
    """One CSV per symbol; masked bars keep their date with NaN fields (no fill).

    Slices are written lazily so only ``workers`` symbols are materialized at once.
    """
    columns = ["date", "symbol", *fields]

    def write(item: tuple[str, int, int]) -> tuple[str, tuple[str, str, int]]:
        import numpy as np

        symbol, first, stop = item
        group = frame.iloc[first:stop]
        # Qlib stores float32. Rounding here and writing 9 significant digits makes the CSV round-trip
        # exactly; pandas' default float text is not exact for long mantissas (e.g. prices x split ratios).
        out = group[columns].astype({field: np.float32 for field in fields})
        out.to_csv(directory / f"{symbol}.csv", index=False, date_format="%Y-%m-%d", float_format="%.9g",
                   na_rep="nan", lineterminator="\n")
        days = group["date"]
        return symbol, (f"{days.iloc[0]:%Y-%m-%d}", f"{days.iloc[-1]:%Y-%m-%d}", stop - first)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(write, _symbol_slices(frame)))


def _verify_dump(qlib: Path, frame: Any, sessions: Any, ranges: dict[str, tuple[str, str, int]],
                 fields: tuple[str, ...] = FIELDS) -> None:
    """Independent readback: calendar, instrument ranges, inventory and float32 values/NaNs."""
    import numpy as np

    text = [f"{day:%Y-%m-%d}" for day in sessions]
    if (qlib / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines() != text:
        raise DatasetError("Qlib calendar differs from the retained Stooq sessions")
    entries: dict[str, tuple[str, str]] = {}
    for line in (qlib / "instruments" / "all.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or parts[0] in entries:
            raise DatasetError("Qlib instrument list is malformed")
        entries[parts[0]] = (parts[1], parts[2])
    if entries != {symbol: (first, last) for symbol, (first, last, _) in ranges.items()}:
        raise DatasetError("Qlib instrument list or ranges differ from normalized symbols")
    features = qlib / "features"
    expected = {f"{symbol.lower()}/{field}.day.bin" for symbol in ranges for field in fields}
    actual = {path.relative_to(features).as_posix() for path in features.rglob("*") if path.is_file()}
    if actual != expected:
        raise DatasetError(f"Qlib feature inventory mismatch; missing={len(expected - actual)}, extra={len(actual - expected)}")
    for symbol, first, stop in _symbol_slices(frame):
        group = frame.iloc[first:stop]
        slots = sessions.get_indexer(group["date"])
        if (slots < 0).any():
            raise DatasetError(f"normalized dates outside the calendar for {symbol}")
        begin, last = int(slots[0]), int(slots[-1])
        if (text[begin], text[last]) != ranges[symbol][:2]:
            raise DatasetError(f"normalized range differs for {symbol}")
        for field in fields:
            want = np.full(last - begin + 1, np.nan, dtype=np.float32)
            want[slots - begin] = group[field].to_numpy(dtype=np.float64).astype(np.float32)
            payload = np.fromfile(features / symbol.lower() / f"{field}.day.bin", dtype="<f4")
            if len(payload) != len(want) + 1 or payload[0] != begin:
                raise DatasetError(f"Qlib bin length or calendar offset mismatch: {symbol}/{field}")
            got = payload[1:]
            gaps = np.isnan(want)
            if not (np.array_equal(np.isnan(got), gaps)
                    and np.array_equal(got[~gaps].view(np.uint32), want[~gaps].view(np.uint32))):
                raise DatasetError(f"Qlib bin readback mismatch: {symbol}/{field}")


def _limitations(kind: str | None, repaired: bool) -> list[str]:
    universe = "The universe is a liquidity-ranked local proxy, not Brain TOP3000 or index membership."
    if kind == "sharadar":
        return [
            "Delisted companies are included (Sharadar estimates its coverage 99% survivorship-free); a position "
            "ends at the last listed close: no delisting return is modelled.",
            universe,
            "Security types come from Sharadar's category: common stock (domestic, ADR, Canadian, any share class) "
            "is kept; preferreds, funds and blank-check companies (SIC 6770) are excluded.",
            "Prices are Sharadar's split-, dividend- and spinoff-adjusted closes; factor=1 is only an identity marker.",
            "Licensed data (Sharadar personal use): do not share the snapshot or the table.",
        ]
    if kind == "historicaldata":
        return [
            "Delisted companies are included (one file per listing lifecycle since 2003-10); a position ends at the "
            "last traded close: no delisting return is modelled.",
            universe,
            "Security types come from the archive's symbols.csv: common stock (CS, ADRC, OS) is kept; ETFs, preferreds, "
            "warrants, units, rights, OTC listings and blank-check names (\"... Acquisition Corp\") are excluded.",
            "Prices are the archive's split- and cash-distribution-adjusted columns (split-only for securities that paid "
            "in a foreign currency, listed in the table manifest); vwap is real. factor=1 is only an identity marker.",
            "Licensed data (per-user licence): do not redistribute the table or snapshot beyond licensed users.",
        ]
    survivorship = "Survivorship bias: the table lists securities quoted in 2026; earlier delistings are absent."
    if kind == "yahoo":
        return [
            survivorship, universe,
            "Security types come from Yahoo and Nasdaq security names: ETFs, funds, preferreds, warrants, units, "
            "rights and SPACs are excluded; ADRs and partnership units remain.",
            "Prices are Yahoo's split- and dividend-adjusted closes (most spinoffs included) with unrecorded splits "
            "repaired at build (see the table manifest). factor=1 is only an identity marker.",
            "No publication-time, delisting-return or redistribution guarantee from the source table.",
        ]
    return [
        survivorship, universe,
        "Security type comes from the table; closed-end funds, ADRs and SPAC shares can remain 'stock'.",
        ("Stooq's split/dividend back-adjustment was checked against Yahoo split events; unadjusted recent "
         "splits were repaired (see corporate_actions). factor=1 is only an identity marker." if repaired else
         "Adjustment semantics and corporate actions are unverified; factor=1 is only an identity marker."),
        "No publication-time, delisting-return or redistribution guarantee from the source table.",
    ]


def prepare_stooq_table(table: Path, data_root: Path, start: str, end: str, *,
                        rule: UniverseRule | None = None, min_session_coverage: float = 0.5,
                        make_latest: bool = False, max_workers: int = 4, keep_normalized: bool = False,
                        corporate_actions: Path | None = None, groups: Path | None = None,
                        progress: Callable[[str], None] | None = None) -> Path:
    """Build a verified snapshot plus a dated liquidity universe from one consolidated table.

    With ``corporate_actions`` (a fetched Yahoo chart archive), splits Stooq has not
    adjusted yet are repaired, Yahoo's non-equity instruments are excluded, and the
    snapshot gains ``close_raw`` and ``dividend`` fields (see corporate_actions.py).
    With ``groups`` (a fetched Nasdaq screener), ``universe/groups.csv`` records each
    symbol's sector, industry and country for group neutralization.
    """
    import numpy as np
    import pandas as pd

    rule = (rule or UniverseRule()).validate()
    table, output = _safe_output(Path(table), Path(data_root))
    archive = None
    if corporate_actions is not None:
        from qlib_yahoo.yahoo_chart import ChartArchive

        archive = ChartArchive(Path(corporate_actions))
    extras = table_extras(table)
    # A table built by build-yahoo-table or build-sharadar-table carries a sidecar manifest naming its sources.
    builder = None
    sidecar = table.with_name(table.name + ".manifest.json")
    if sidecar.is_file():
        import hashlib
        import json

        builder_bytes = sidecar.read_bytes()
        builder_record = json.loads(builder_bytes)
        kinds = [BUILDERS[name] for name in (builder_record.get("sources") or {}) if name in BUILDERS]
        if builder_record.get("table_sha256") and kinds:
            builder = {"kind": kinds[0], "manifest": str(sidecar),
                       "manifest_sha256": hashlib.sha256(builder_bytes).hexdigest(),
                       "table_sha256": builder_record["table_sha256"], "sources": builder_record["sources"],
                       "classification": {k: v["count"] for k, v in builder_record.get("classification", {}).items()},
                       "isolated_spikes": builder_record.get("isolated_spikes", {}).get("rows"),
                       "cutoff": builder_record.get("cutoff"),
                       **({"delisted_stocks": builder_record.get("delisted_stocks", {}).get("count"),
                           "listing_filter_bars_dropped": builder_record.get("listing_filter", {}).get("bars_dropped")}
                          if kinds[0] in ("sharadar", "historicaldata") else {})}
    if archive is not None and extras:
        raise DatasetError("this table already carries raw-close/dividend columns; do not also apply corporate actions")
    fields = (*FIELDS, *ACTION_FIELDS) if archive is not None else (*FIELDS, *extras)
    zip_source = table.suffix.lower() == ".zip"
    begin, stop = _check_interval(start, end)
    if (isinstance(min_session_coverage, bool) or not isinstance(min_session_coverage, (int, float))
            or not 0 <= min_session_coverage < 1):
        raise DatasetError("min_session_coverage must be in [0, 1)")
    if type(max_workers) is not int or max_workers < 1:
        raise DatasetError("max_workers must be a positive integer")
    script = _dump_script(output)

    dataset_dir = output / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{builder['kind']}-table" if builder else "stooq-table"
    snapshot = dataset_dir / f"{prefix}-{rule.name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    snapshot.mkdir(exist_ok=False)
    record: dict[str, Any] = {"status": "building", "source_table": str(table), "stage": "fingerprinting",
                              "request": {"start": start, "end": end, "universe": rule.contract(),
                                          "min_session_coverage": min_session_coverage,
                                          "corporate_actions": str(corporate_actions) if corporate_actions else None}}
    timings: dict[str, float] = {}
    clock = [time.monotonic()]

    def stage(name: str, message: str) -> None:
        now = time.monotonic()
        timings[record["stage"]] = round(now - clock[0], 2)
        clock[0] = now
        record["stage"] = name
        _write_json_atomically(snapshot / "progress.json", record)
        _log(progress, message)

    try:
        _write_json_atomically(snapshot / "progress.json", record)
        _log(progress, f"Snapshot: {snapshot}")
        _log(progress, f"Fingerprinting read-only table {table.name} ...")
        before = _fingerprint(table)
        read_started = datetime.now(timezone.utc).isoformat()
        stage("reading", f"Scanning {start} to {end} (end exclusive) ...")
        empty_members: list[str] = []
        securities, row_count = _securities(table, begin, stop, empty_members)
        if row_count == 0:
            raise DatasetError("Stooq table has no rows in the requested interval")

        stage("classifying", f"Classifying {securities['ticker'].nunique():,} tickers from {row_count:,} rows ...")
        if securities["ticker"].duplicated().any():
            changed = securities.loc[securities["ticker"].duplicated(), "ticker"].iloc[0]
            raise DatasetError(f"ticker {changed} changes security type inside the table; an identifier map is required")
        reasons = {ticker: exclusion_reason(ticker, kind) for ticker, kind in zip(securities["ticker"], securities["type"])}
        excluded: dict[str, list[str]] = {}
        for ticker, reason in sorted(reasons.items()):
            if reason:
                excluded.setdefault(reason, []).append(ticker)
        stocks = sorted(ticker for ticker, reason in reasons.items() if reason is None)
        if archive is not None:
            from qlib_yahoo.corporate_actions import yahoo_type_exclusions

            for ticker, reason in sorted(yahoo_type_exclusions(archive, stocks).items()):
                excluded.setdefault(reason, []).append(ticker)
                stocks.remove(ticker)
        if not stocks:
            raise DatasetError("no common-stock tickers remain after the security filter")

        stage("loading", f"Loading {len(stocks):,} common stocks; screening sparse sessions ...")
        frame, sparse_report, sparse_rows = _stock_frame(table, begin, stop, stocks, float(min_session_coverage), extras)
        duplicate = _duplicate_example(frame)
        if duplicate:
            raise DatasetError(f"Stooq table repeats a ticker/date row, e.g. {duplicate}")
        invalid, invalid_report = _invalid_bars(frame)
        action_columns, action_report = None, None
        if archive is not None:
            from qlib_yahoo.corporate_actions import reconcile

            stage("corporate_actions", f"Reconciling {len(stocks):,} stocks with Yahoo split and dividend events ...")
            action_columns, action_report = reconcile(frame, invalid, archive, progress=progress)
            invalid = invalid | action_columns["masked"]
        zero_volume = ~invalid & (frame["volume"].to_numpy() == 0)
        sessions = pd.DatetimeIndex(np.sort(frame["date"].unique()))

        stage("universe", f"Ranking {len(stocks):,} common stocks over {len(sessions):,} sessions ...")
        usable = ~invalid & ~zero_volume
        with np.errstate(invalid="ignore"):
            dollars = np.where(usable, frame["close"].to_numpy() * frame["volume"].to_numpy(), 0.0)
        tickers = frame["ticker"].cat.categories
        dollar_volume = pd.DataFrame(_panel(frame, sessions, dollars, 0.0), index=sessions, columns=tickers)
        valid = pd.DataFrame(_panel(frame, sessions, usable, False), index=sessions, columns=tickers)
        del dollars
        price = None
        if rule.min_price > 0:
            on_rank = frame["date"].isin(rank_sessions(sessions)).to_numpy()
            if action_columns is not None:
                closes = action_columns["close_raw"]
            elif "close_raw" in extras:
                closes = frame["close_raw"].to_numpy()
            else:
                closes = frame["close"].to_numpy()
            closes = np.where(usable, closes, np.nan)[on_rank]
            price = pd.DataFrame(_panel(frame.loc[on_rank], rank_sessions(sessions), closes, np.nan),
                                 index=rank_sessions(sessions), columns=tickers)
        membership, rankings = liquidity_membership(dollar_volume, valid, rule, price)
        del price
        del dollar_volume, valid
        if membership.empty:
            raise DatasetError("no ranking session produced members; extend the interval or lower min history")
        spans = membership_spans(membership, rankings)
        members = sorted(spans)
        never_ranked = sorted(set(stocks) - set(members))
        full = rankings.loc[rankings["members"] == rule.size, "ranked_on"]
        _log(progress, f"{len(rankings)} monthly rankings; {len(members):,} unique members; "
                       + (f"full {rule.size} from {full.iloc[0]:%Y-%m-%d}" if len(full) else f"never reached {rule.size}"))

        stage("normalizing", f"Writing normalized CSVs for {len(members):,} members ...")
        rows = np.flatnonzero(frame["ticker"].isin(members).to_numpy())
        masked = (invalid | zero_volume)[rows]
        zero_volume_rows = int(zero_volume[rows].sum())
        del invalid, zero_volume, usable
        # Build the member frame column by column, then release the full stock frame.
        columns = {"date": frame["date"].to_numpy()[rows],
                   "symbol": pd.Categorical.from_codes(frame["ticker"].cat.codes.to_numpy()[rows],
                                                       categories=frame["ticker"].cat.categories)}
        for field in PRICE_FIELDS:
            values = frame[field].to_numpy()[rows]
            values[masked] = np.nan
            columns[field] = values
        columns["factor"] = np.where(masked, np.nan, 1.0)
        for field in extras:
            columns[field] = np.where(masked, np.nan, frame[field].to_numpy()[rows])
        if action_columns is not None:
            for field in ACTION_FIELDS:
                columns[field] = np.where(masked, np.nan, action_columns[field][rows])
            del action_columns
        del frame, rows
        frame = pd.DataFrame(columns)
        del columns
        frame["symbol"] = frame["symbol"].cat.remove_unused_categories()
        normalized = snapshot / "normalized"
        normalized.mkdir()
        ranges = _write_normalized(frame, normalized, max_workers, fields)

        stage("dumping", f"Converting {len(ranges):,} symbols with pinned Qlib dump_bin (workers={max_workers}) ...")
        qlib = snapshot / "qlib_data"
        qlib.mkdir()
        try:
            subprocess.run([sys.executable, str(script), "dump_all", "--data_path", str(normalized),
                            "--qlib_dir", str(qlib), "--freq", "day", "--date_field_name", "date",
                            "--symbol_field_name", "symbol", "--exclude_fields", "date,symbol",
                            "--file_suffix", ".csv", "--max_workers", str(max_workers)],
                           check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            tail = (error.stderr or error.stdout or "").strip().splitlines()[-3:]
            raise DatasetError(f"Qlib dump failed: {' | '.join(tail) or error}") from error
        except OSError as error:
            raise DatasetError(f"Qlib dump failed: {error}") from error

        stage("verifying", "Reading every binary back against the normalized values ...")
        dumped_sessions = pd.DatetimeIndex(np.sort(frame["date"].unique()))
        _verify_dump(qlib, frame, dumped_sessions, ranges, fields)
        instruments_path = qlib / "instruments" / f"{rule.name}.txt"
        write_instrument_file(instruments_path, spans)
        universe_dir = snapshot / "universe"
        universe_dir.mkdir()
        membership_path = universe_dir / f"{rule.name}_membership.csv.gz"
        membership.to_csv(membership_path, index=False, date_format="%Y-%m-%d", lineterminator="\n",
                          compression={"method": "gzip", "mtime": 0})
        rankings_path = universe_dir / f"{rule.name}_rankings.csv"
        rankings.to_csv(rankings_path, index=False, date_format="%Y-%m-%d", lineterminator="\n")
        groups_record = None
        if groups is not None:
            from qlib_yahoo.reference import group_source, groups_csv

            reference = group_source(Path(groups))
            group_rows, coverage = reference.groups(ranges)
            groups_path = universe_dir / "groups.csv"
            groups_path.write_bytes(groups_csv(group_rows))
            groups_record = {"file": "universe/groups.csv", "sha256": sha256(groups_path),
                             "source": reference.identity(), "coverage": coverage,
                             "fields": ["sector", "industry", "country"],
                             "limitation": reference.group_limitation}

        after = _fingerprint(table)
        if after != before:
            raise DatasetError("Stooq table changed while it was being converted")
        read_finished = datetime.now(timezone.utc).isoformat()
        stage("sealing", "Hashing the Qlib tree and sealing the manifest ...")
        binary_hash, file_count = tree_sha256(qlib)
        symbols = sorted(ranges)
        populated = rankings.loc[rankings["members"] > 0]
        manifest = {
            "schema_version": 1, "status": "ready", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "provider": ("Sharadar US stock prices incl. delisted via build-sharadar-table (local table, read-only)"
                         if builder and builder["kind"] == "sharadar" else
                         "HistoricalData.net US daily archive incl. delisted via build-historicaldata-table (local table, read-only)"
                         if builder and builder["kind"] == "historicaldata" else
                         "Yahoo Finance daily charts via build-yahoo-table (local table, read-only)" if builder
                         else "Stooq daily US ASCII archive (local, read-only)" if zip_source
                         else "Stooq US daily consolidated table (local, read-only)"),
            "table_builder": builder,
            "qlib_source": {"release": QLIB_RELEASE, "commit": QLIB_COMMIT},
            "date_range": {"start_inclusive": start, "end_exclusive": end},
            "symbols": symbols, "benchmark": None, "sessions": len(dumped_sessions),
            "row_counts": {symbol: count for symbol, (_, _, count) in ranges.items()},
            "zero_volume_rows": zero_volume_rows,
            "source_table": {"path": str(table), "before": before, "after": after, "columns": list(COLUMNS),
                             "format": "stooq-daily-us-ascii-zip" if zip_source else table.suffix.lower().lstrip("."),
                             **({"empty_members": {"count": len(empty_members), "examples": empty_members[:EXAMPLES]}}
                                if zip_source else {}),
                             "rows_in_interval": row_count,
                             "read_started_at_utc": read_started, "read_finished_at_utc": read_finished},
            "security_filter": {
                "rule": "type == 'stock'; ticker [A-Z][A-Z0-9]*(-[A-Z])? without W/U/R/V suffix; "
                        "no underscore series; no exchange test symbols (e.g. ZVZZT); no Windows-reserved names",
                "common_stocks": len(stocks),
                "excluded": {reason: {"count": len(names), "examples": names[:EXAMPLES]}
                             for reason, names in sorted(excluded.items())},
                "never_ranked_not_dumped": {"count": len(never_ranked), "examples": never_ranked[:EXAMPLES]},
            },
            "session_filter": {"rule": f"drop sessions with stock bars < {min_session_coverage} x median "
                                       f"of the previous {SPARSE_LOOKBACK} sessions (following sessions for the first five)",
                               "dropped_sessions": sparse_report, "dropped_rows": sparse_rows},
            "invalid_bars": {"policy": "date kept; all fields NaN; no fill", **invalid_report},
            "fields": list(fields),
            "corporate_actions": action_report,
            "groups": groups_record,
            "universes": {rule.name: {
                **rule.contract(),
                "instruments_file": f"instruments/{instruments_path.name}",
                "instruments_sha256": sha256(instruments_path),
                "membership_csv": f"universe/{membership_path.name}", "membership_sha256": sha256(membership_path),
                "rankings_csv": f"universe/{rankings_path.name}", "rankings_sha256": sha256(rankings_path),
                "ranking_sessions": int(len(rankings)), "unique_members": len(members),
                "first_effective": f"{populated['effective_start'].iloc[0]:%Y-%m-%d}",
                "last_effective": f"{rankings['effective_end'].iloc[-1]:%Y-%m-%d}",
                "full_size_from": f"{full.iloc[0]:%Y-%m-%d}" if len(full) else None,
                "shortfall_rankings": int((rankings["members"] < rule.size).sum()),
            }},
            "normalized_sha256": {f"{symbol}.csv": sha256(normalized / f"{symbol}.csv") for symbol in symbols},
            "qlib_calendar_sha256": sha256(qlib / "calendars" / "day.txt"),
            "qlib_instruments_sha256": sha256(qlib / "instruments" / "all.txt"),
            "qlib_binary_tree_sha256": binary_hash, "qlib_binary_file_count": file_count,
            "converter": {"module": "qlib_yahoo.stooq_table", "module_sha256": sha256(Path(__file__)),
                          "universe_sha256": sha256(Path(__file__).with_name("universe.py")),
                          "dump_script_sha256": sha256(script)},
            "normalization": (("OHLCV adjusted as supplied by the table, with its traded close (close_raw) and dividends; "
                               "factor=1 identity; zero-volume and invalid bars NaN")
                              if extras else
                              ("OHLCV in Stooq's adjusted units (unadjusted splits repaired when corporate actions are "
                               "supplied); factor=1 identity; zero-volume and invalid bars NaN")),
            "calendar": "union of retained member dates after the sparse-session screen; missing bars stay missing",
            "timings_seconds": timings,
            "limitations": _limitations(builder["kind"] if builder else None, archive is not None),
        }
        # The normalized CSVs only feed dump_bin; every binary was read back against the source values,
        # and the tree hash verifies the snapshot from now on. Their hashes stay in the manifest.
        manifest["normalized_csv"] = {
            "retained": bool(keep_normalized),
            "note": "input to dump_bin; hashes kept above; the verified Qlib binaries are the dataset",
        }
        if not keep_normalized:
            shutil.rmtree(normalized)
        timings["sealing"] = round(time.monotonic() - clock[0], 2)
        record["status"] = "ready"
        _write_json_atomically(snapshot / "manifest.json", manifest)
        _write_json_atomically(snapshot / "progress.json", record)
        if make_latest:
            _write_json_atomically(output / "latest.json", {"dataset": str(snapshot)})
    except BaseException as error:  # a cancel arrives as KeyboardInterrupt; record it before unwinding
        record["status"] = "failed"
        record["error"] = str(error) or type(error).__name__
        _write_json_atomically(snapshot / "progress.json", record)
        _write_json_atomically(snapshot / "failed_manifest.json", record)
        raise
    _log(progress, f"Ready: {snapshot}")
    return snapshot
