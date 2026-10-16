"""Sharadar US stock prices with delisted companies: bulk download and consolidated table.

Yahoo and Stooq serve only securities that still trade, so universes before about
2019 are built from today's survivors (see docs/MARKET_DATA.md). Sharadar sells
end-of-day prices for active and delisted US stocks from 1998 (about 99%
survivorship-free by its own estimate) with corporate actions and daily market
capitalization: sharadar.com under a personal-use licence, or the same tables as
Nasdaq Data Link's SHARADAR/SEP, TICKERS, ACTIONS and DAILY for professional use.
This module downloads the bulk tables with the user's own API key and writes the
long table build-yahoo-table writes, so import-stooq-table, the universe, Alpha Lab
and the Qlib runner use it unchanged.

Conventions, checked on Sharadar's public sample (30 stocks, five years) against
the Yahoo-built table, where the traded close matched on every bar:

* ``open high low close`` are split-adjusted, ``closeadj`` is split-, cash dividend-
  and spinoff-adjusted, ``closeunadj`` is the traded close. The table's OHLC are
  split-adjusted values times ``closeadj / close``.
* ``volume`` is split-adjusted, so ``close x volume`` is the traded dollar volume; it
  is rescaled so the adjusted close times volume keeps that meaning.
* A dividend ``value`` in the actions table is on today's share basis (Honeywell's
  doubled after its 2026 1:2 reverse split); ``value x closeunadj / close`` on the
  ex-date is the cash per share traded that day.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from qlib_yahoo.dataset import DatasetError, sha256

API = "https://api.sharadar.com/v1.0/data/{table}"
TABLES = ("tickers", "actions", "stocks", "daily")
REQUIRED_TABLES = ("tickers", "actions", "stocks")
# Nasdaq Data Link names the same tables SHARADAR_SEP, SHARADAR_TICKERS, ...
FILE_NAMES = {"stocks": ("stocks", "sep"), "tickers": ("tickers",), "actions": ("actions",), "daily": ("daily",)}
COLUMNS = {
    "stocks": ("ticker", "date", "open", "high", "low", "close", "volume", "closeadj", "closeunadj"),
    "tickers": ("table", "ticker", "exchange", "isdelisted", "category", "siccode", "sector", "industry", "location"),
    "actions": ("date", "action", "ticker", "value", "contraname"),
    "daily": ("ticker", "date", "marketcap"),
}
TICKER_TABLES = frozenset({"stocks", "SEP"})  # the tickers table's name for the stock price table
LISTED = frozenset({"NYSE", "NASDAQ", "NYSEMKT", "NYSEARCA", "BATS"})
BUCKETS = 32
EXAMPLES = 20
AGENT = "qlib-yahoo-sharadar/1"
KEY_PATTERN = re.compile(r"[A-Za-z0-9_\-]{8,128}\Z")
LICENCE = ("Sharadar Personal Use License: one natural person's own research and trading; the files and data "
           "derived from them must not be shared with other people or organizations")


def _log(progress: Callable[[str], None] | None, message: str) -> None:
    if progress:
        progress(message)


def api_key_from_environment(path: Path | None = None) -> str:
    """The key from ``path`` (first line) or the SHARADAR_API_KEY variable; never from the command line."""
    key = (Path(path).read_text(encoding="utf-8").strip().splitlines() or [""])[0].strip() if path else \
        os.environ.get("SHARADAR_API_KEY", "").strip()
    if not KEY_PATTERN.fullmatch(key):
        raise DatasetError("Sharadar API key missing or malformed: set SHARADAR_API_KEY or pass --api-key-file")
    return key


# --------------------------------------------------------------------------- download


def _write_manifest(destination: Path, manifest: dict[str, Any]) -> None:
    temporary = destination / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, destination / "manifest.json")


def _check_zip(path: Path, table: str) -> dict[str, Any]:
    """One CSV member whose header has the table's required columns."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(members) != 1:
                raise DatasetError(f"Sharadar {table} download should hold one CSV, found {len(members)}")
            with archive.open(members[0]) as handle:
                header = io.TextIOWrapper(handle, encoding="utf-8").readline().strip().split(",")
    except zipfile.BadZipFile as error:
        raise DatasetError(f"Sharadar {table} download is not a zip file") from error
    missing = [name for name in COLUMNS[table] if name not in header]
    if missing:
        raise DatasetError(f"Sharadar {table} file lacks columns {missing}")
    return {"member": members[0], "columns": header}


def fetch_bulk(destination: Path, api_key: str, *, tables: Iterable[str] = REQUIRED_TABLES, years: str = "full",
               opener: Callable[..., Any] = urllib.request.urlopen, progress: Callable[[str], None] | None = None,
               chunk: int = 1 << 20) -> Path:
    """Download Sharadar bulk tables into ``destination`` (new, or one earlier interrupted download).

    Each table is the vendor's zipped CSV, kept byte for byte with its SHA-256. The
    key is sent only in the request and never written or printed; failures report
    the HTTP status alone.
    """
    tables = tuple(dict.fromkeys(tables))
    unknown = [table for table in tables if table not in TABLES]
    if unknown or not tables:
        raise DatasetError(f"Sharadar tables must be among {', '.join(TABLES)}")
    if years not in ("5", "10", "full"):
        raise DatasetError("years must be 5, 10 or full")
    if not KEY_PATTERN.fullmatch(api_key or ""):
        raise DatasetError("Sharadar API key missing or malformed")
    destination = Path(destination)
    manifest_path = destination / "manifest.json"
    if destination.exists():
        if not manifest_path.is_file():
            raise DatasetError(f"{destination} exists and is not a Sharadar download; choose a new folder")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("years") != years:
            raise DatasetError(f"{destination} holds a years={manifest.get('years')} download")
    else:
        destination.mkdir(parents=True)
        manifest = {"schema": 1, "provider": "Sharadar bulk download (sharadar.com API)", "years": years,
                    "licence": LICENCE, "files": {}}
    for table in tables:
        target = destination / f"{table}.csv.zip"
        record = manifest["files"].get(table)
        if record and target.is_file() and sha256(target) == record["sha256"]:
            _log(progress, f"{table}: already downloaded")
            continue
        _log(progress, f"{table}: downloading (years={years})")
        query = urllib.parse.urlencode({"api_key": api_key, "years": years})
        request = urllib.request.Request(f"{API.format(table=table)}?{query}", headers={"User-Agent": AGENT})
        temporary = target.with_name(target.name + ".part")
        digest, size = hashlib.sha256(), 0
        try:
            with opener(request, timeout=600) as response, temporary.open("wb") as handle:
                while True:
                    block = response.read(chunk)
                    if not block:
                        break
                    handle.write(block)
                    digest.update(block)
                    size += len(block)
                    if progress and size // (100 << 20) != (size - len(block)) // (100 << 20):
                        progress(f"{table}: {size / (1 << 20):,.0f} MB")
        except urllib.error.HTTPError as error:
            temporary.unlink(missing_ok=True)
            hint = " (check the key and that the plan includes this table)" if error.code in (401, 403) else ""
            raise DatasetError(f"Sharadar {table} download failed: HTTP {error.code}{hint}") from None
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            temporary.unlink(missing_ok=True)
            reason = str(getattr(error, "reason", error)).replace(api_key, "<key>")
            raise DatasetError(f"Sharadar {table} download failed: {reason}") from None
        try:
            details = _check_zip(temporary, table)
        except DatasetError as error:
            head = temporary.read_bytes()[:600].decode("utf-8", "replace").replace(api_key, "<key>")
            temporary.unlink(missing_ok=True)
            vendor = f": {head.strip()}" if head.lstrip().startswith("{") else ""
            raise DatasetError(f"{error}{vendor}") from None
        os.replace(temporary, target)
        manifest["files"][table] = {"file": target.name, "sha256": digest.hexdigest(), "size_bytes": size,
                                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(), **details}
        _write_manifest(destination, manifest)
        _log(progress, f"{table}: {size / (1 << 20):,.1f} MB")
    _write_manifest(destination, manifest)
    return destination


# --------------------------------------------------------------------------- source folder


def symbol(ticker: str) -> str:
    """Sharadar writes share classes as BRK.B; the tables here use BRK-B."""
    return ticker.strip().upper().replace(".", "-")


def security_type(category: str, siccode: Any) -> str:
    """Common stock (domestic, ADR, Canadian; any share class) is 'stock', except blank-check companies."""
    text = (category or "").lower()
    if "common stock" in text:
        return "spac" if str(siccode).strip().split(".")[0] == "6770" else "stock"
    if "preferred" in text:
        return "preferred"
    if "warrant" in text:
        return "warrant"
    if "fund" in text or text in ("etf", "etn", "etd", "cef", "etmf"):
        return "fund"
    return "other"


def country(location: str) -> str:
    """'California; U.S.A' -> 'United States' (the Nasdaq screener's spelling); other places as written."""
    place = (location or "").rsplit(";", 1)[-1].strip()
    return "United States" if place.upper() in ("U.S.A", "U.S.A.", "USA") else place


class SharadarSource:
    """A folder of Sharadar tables: fetch-sharadar's download, or zips/CSVs saved by hand.

    Files are found by table name (``stocks.csv.zip``, ``SHARADAR_SEP_2026.zip``, ``tickers.csv`` ...).
    A download's manifest hashes are verified; hand-saved files are hashed and recorded as found.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.is_dir():
            raise DatasetError(f"Sharadar folder is missing: {self.path}")
        manifest_path = self.path / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        self.files: dict[str, Path] = {}
        for table, names in FILE_NAMES.items():
            found = [item for item in sorted(self.path.iterdir()) if item.is_file() and item.suffix.lower() in (".zip", ".csv")
                     and not item.name.endswith(".part") and _table_name(item.name) in names]
            if len(found) > 1:
                raise DatasetError(f"several {table} files in {self.path}: {[item.name for item in found]}")
            if found:
                self.files[table] = found[0]
        missing = [table for table in REQUIRED_TABLES if table not in self.files]
        if missing:
            raise DatasetError(f"Sharadar folder {self.path} lacks the {', '.join(missing)} table(s)")
        self.hashes = {table: sha256(path) for table, path in self.files.items()}
        for table, record in (self.manifest.get("files") or {}).items():
            if table in self.hashes and record.get("sha256") != self.hashes[table]:
                raise DatasetError(f"Sharadar {table} file differs from its download manifest")
        self._tickers: Any = None
        self.duplicate_symbols: list[str] = []

    @staticmethod
    def looks_like(path: Path) -> bool:
        path = Path(path)
        return path.is_dir() and any(_table_name(item.name) == "tickers" for item in path.iterdir() if item.is_file())

    def identity(self) -> dict[str, Any]:
        return {"path": str(self.path), "provider": self.manifest.get("provider", "Sharadar tables (saved by hand)"),
                "years": self.manifest.get("years"),
                "files": {table: {"file": path.name, "sha256": self.hashes[table]} for table, path in self.files.items()},
                "licence": LICENCE, "point_in_time": False}

    @contextmanager
    def reader(self, table: str, columns: Iterable[str], types: dict[str, Any]) -> Iterator[Any]:
        """A streaming Arrow CSV reader over the table (zip member or plain CSV), closed on exit."""
        import pyarrow.csv as pacsv

        path = self.files[table]
        archive = zipfile.ZipFile(path) if path.suffix.lower() == ".zip" else None
        try:
            if archive is not None:
                members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if len(members) != 1:
                    raise DatasetError(f"{path.name} should hold one CSV, found {len(members)}")
                handle = archive.open(members[0])
            else:
                handle = path.open("rb")
            with handle:
                columns = list(columns)
                yield pacsv.open_csv(handle, read_options=pacsv.ReadOptions(block_size=16 << 20),
                                     convert_options=pacsv.ConvertOptions(
                                         include_columns=columns, column_types={c: types[c] for c in columns if c in types},
                                         strings_can_be_null=True, null_values=[""]))
        finally:
            if archive is not None:
                archive.close()

    def tickers(self) -> Any:
        """The stock price table's securities, one row per converted symbol (pandas, indexed by symbol)."""
        if self._tickers is None:
            import pyarrow as pa

            names = COLUMNS["tickers"] + ("sicindustry", "lastpricedate", "permaticker")
            header = self._header("tickers")
            wanted = [name for name in names if name in header]
            with self.reader("tickers", wanted, {name: pa.string() for name in wanted}) as reader:
                frame = reader.read_all().to_pandas()
            frame = frame[frame["table"].isin(TICKER_TABLES)].copy()
            frame["symbol"] = frame["ticker"].map(symbol)
            frame["type"] = [security_type(c, s) for c, s in zip(frame["category"].fillna(""), frame["siccode"].fillna(""))]
            if "lastpricedate" in frame:
                frame = frame.sort_values("lastpricedate", na_position="first")
            self.duplicate_symbols = sorted(frame.loc[frame["symbol"].duplicated(keep=False), "ticker"].unique())
            self._tickers = frame.drop_duplicates("symbol", keep="last").set_index("symbol")
        return self._tickers

    def _header(self, table: str) -> list[str]:
        path = self.files[table]
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                member = [name for name in archive.namelist() if name.lower().endswith(".csv")][0]
                with archive.open(member) as handle:
                    line = io.TextIOWrapper(handle, encoding="utf-8").readline()
        else:
            with path.open(encoding="utf-8") as handle:
                line = handle.readline()
        header = line.strip().split(",")
        missing = [name for name in COLUMNS[table] if name not in header]
        if missing:
            raise DatasetError(f"Sharadar {table} file lacks columns {missing}")
        return header

    def groups(self, symbols: Iterable[str]) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Sector, industry and country per symbol from the tickers table (current, not point-in-time)."""
        tickers = self.tickers()
        table, missing, blank = [], [], 0
        for name in sorted(symbols):
            row = tickers.loc[name] if name in tickers.index else None
            if row is None:
                missing.append(name)
            values = {"sector": _text(None if row is None else row["sector"]),
                      "industry": _text(None if row is None else row["industry"]),
                      "country": country(_text(None if row is None else row["location"]))}
            blank += not values["sector"]
            table.append({"symbol": name, **values})
        coverage = {"symbols": len(table), "not_in_screener": len(missing), "missing_examples": missing[:EXAMPLES],
                    "without_sector": blank, "sectors": len({r["sector"] for r in table if r["sector"]}),
                    "industries": len({r["industry"] for r in table if r["industry"]})}
        return table, coverage

    group_limitation = "Sharadar's current sector/industry/location applied to the whole history"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _table_name(filename: str) -> str:
    stem = filename.lower()
    for suffix in (".zip", ".csv"):
        stem = stem[: -len(suffix)] if stem.endswith(suffix) else stem
    stem = re.sub(r"^sharadar_", "", stem)
    return re.sub(r"[_\-]\d.*$", "", stem)


# --------------------------------------------------------------------------- consolidated table


def _bucket(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) % BUCKETS


def _partition(source: SharadarSource, table: str, columns: tuple[str, ...], numeric: tuple[str, ...],
               directory: Path, start: Any, progress: Callable[[str], None] | None) -> dict[str, int]:
    """Stream one table into BUCKETS Parquet files keyed by symbol; return row counts."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from qlib_yahoo.stooq_table import _strict_dates

    types = {"ticker": pa.string(), "date": pa.string(), **{name: pa.float64() for name in numeric}}
    writers: dict[int, Any] = {}
    counts = {"rows": 0, "kept": 0, "bad_dates": 0}
    try:
        with source.reader(table, columns, types) as reader:
            for batch in reader:
                counts["rows"] += batch.num_rows
                day = _strict_dates(batch.column("date"), "%Y-%m-%d")
                counts["bad_dates"] += day.null_count
                # symbols and buckets are computed once per distinct ticker of the batch
                encoded = pc.dictionary_encode(pc.fill_null(batch.column("ticker"), ""))
                unique = [symbol(value) for value in encoded.dictionary.to_pylist()]
                names = pc.take(pa.array(unique, pa.string()), encoded.indices)
                codes = pc.take(pa.array([_bucket(name) for name in unique], pa.int16()), encoded.indices)
                mask = pc.and_(pc.is_valid(day), pc.not_equal(names, ""))
                if start is not None:
                    mask = pc.and_(mask, pc.greater_equal(day, pa.scalar(start, pa.timestamp("s"))))
                frame = pa.table({"symbol": names, "date": day, **{name: batch.column(name) for name in numeric}})
                frame, codes = frame.filter(mask), pc.filter(codes, mask).to_numpy()
                counts["kept"] += frame.num_rows
                for code in np.unique(codes):
                    part = frame.filter(pa.array(codes == code))
                    if code not in writers:
                        writers[code] = pq.ParquetWriter(directory / f"{table}-{code:02d}.parquet", part.schema)
                    writers[code].write_table(part)
                if progress and counts["rows"] // 5_000_000 != (counts["rows"] - batch.num_rows) // 5_000_000:
                    progress(f"{table}: read {counts['rows']:,} rows")
    finally:
        for writer in writers.values():
            writer.close()
    return counts


def _venue_timeline(actions: Any) -> dict[str, tuple[bool | None, list[tuple[Any, bool]]]]:
    """Per symbol: the listed state before the first event (None: take the tickers table's exchange) and
    dated switches of 'listed on a US exchange' from exchange moves, listings and delistings."""
    import numpy as np

    timeline: dict[str, tuple[bool | None, list[tuple[Any, bool]]]] = {}
    moves = actions[actions["action"].isin(["exchangefrom", "exchangeto", "delisted", "listed"])]
    day_after = np.timedelta64(1, "D")
    for name, group in moves.sort_values("date", kind="stable").groupby("symbol", sort=False):
        initially = None
        first_from = group[group["action"] == "exchangefrom"]
        if len(first_from):
            initially = _listed_venue(first_from["contraname"].iloc[0])
        elif group["action"].isin(["delisted", "listed"]).any():
            initially = True  # a security that was listed or delisted traded on an exchange before that
        events = []
        for day, action, venue in zip(group["date"].to_numpy().astype("datetime64[s]"), group["action"], group["contraname"]):
            if action == "exchangeto":
                events.append((day, _listed_venue(venue)))
            elif action == "delisted":
                events.append((day + day_after, False))  # the delisting day's own bar is kept
            else:
                events.append((day, True))
        timeline[name] = (initially, events)
    return timeline


def _listed_venue(venue: Any) -> bool:
    return _text(venue).upper().replace(" ", "") in LISTED


def _listed_mask(dates: Any, initially: bool, events: list[tuple[Any, bool]]) -> Any:
    import numpy as np

    state = np.full(len(dates), initially)
    for day, listed in events:
        state[dates >= day] = listed
    return state


def build_table(source_path: Path, destination: Path, *, start: str | None = None,
                progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Write ``destination`` (Parquet) and ``destination``.manifest.json from a Sharadar folder; return the manifest."""
    import tempfile

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from qlib_yahoo.yahoo_table import _level_breaks, _spikes

    source = SharadarSource(Path(source_path))
    destination = Path(destination)
    if destination.exists():
        raise DatasetError(f"{destination} exists; choose a new file")
    begin = None
    if start is not None:
        try:
            begin = datetime.strptime(start, "%Y-%m-%d")
        except ValueError as error:
            raise DatasetError("start must be YYYY-MM-DD") from error
    tickers = source.tickers()
    _log(progress, f"{len(tickers):,} securities in the tickers table "
                   f"({int((tickers['isdelisted'] == 'Y').sum()):,} delisted)")
    with source.reader("actions", COLUMNS["actions"], {name: pa.string() for name in COLUMNS["actions"]} |
                       {"value": pa.float64()}) as reader:
        actions = reader.read_all().to_pandas()
    actions["symbol"] = actions["ticker"].fillna("").map(symbol)
    actions["date"] = pd.to_datetime(actions["date"], format="%Y-%m-%d", errors="coerce").values.astype("datetime64[s]")
    actions = actions.dropna(subset=["date"])
    paid = actions[(actions["action"] == "dividend") & actions["value"].notna()]
    paid = paid.groupby(["symbol", "date"], sort=True)["value"].sum().reset_index()
    dividends = {name: (group["date"].to_numpy().astype("datetime64[s]"), group["value"].to_numpy(np.float64))
                 for name, group in paid.groupby("symbol", sort=False)}
    timeline = _venue_timeline(actions)

    schema_fields = [("ticker", pa.string()), ("exchange", pa.string()), ("type", pa.string()),
                     ("date", pa.timestamp("s"))] + [(name, pa.float64()) for name in
                                                     ("open", "high", "low", "close", "volume", "close_raw", "dividend")]
    with_cap = "daily" in source.files
    if with_cap:
        schema_fields.append(("cap", pa.float64()))
    schema = pa.schema(schema_fields)
    report: dict[str, Any] = {"not_in_tickers": [], "unlisted_bars": 0, "unlisted_examples": [],
                              "duplicate_bars": 0, "spike_rows": 0, "spikes": [], "breaks": [],
                              "dividends_matched": 0, "cap_rows": 0}
    classification: dict[str, list[str]] = {}
    delisted_written: list[str] = []
    rows_written = 0
    temporary = destination.with_name(destination.name + ".tmp")
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".sharadar-") as scratch:
        scratch = Path(scratch)
        counts = {"stocks": _partition(source, "stocks", COLUMNS["stocks"], COLUMNS["stocks"][2:], scratch, begin,
                                       progress)}
        if with_cap:
            counts["daily"] = _partition(source, "daily", COLUMNS["daily"], ("marketcap",), scratch, begin, progress)
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for code in range(BUCKETS):
                path = scratch / f"stocks-{code:02d}.parquet"
                if not path.is_file():
                    continue
                prices = pq.read_table(path).to_pandas()
                size = len(prices)
                prices = prices.drop_duplicates(["symbol", "date"], keep="last")
                report["duplicate_bars"] += size - len(prices)
                if with_cap and (scratch / f"daily-{code:02d}.parquet").is_file():
                    cap = pq.read_table(scratch / f"daily-{code:02d}.parquet").to_pandas()
                    cap = cap.drop_duplicates(["symbol", "date"], keep="last")
                    prices = prices.merge(cap, on=["symbol", "date"], how="left")
                elif with_cap:
                    prices["marketcap"] = np.nan
                prices = prices.sort_values(["symbol", "date"], kind="stable")
                batch = []
                for name, bars in prices.groupby("symbol", sort=True):
                    if name in tickers.index:
                        meta = tickers.loc[name]
                        kind, exchange = meta["type"], (meta["exchange"] or "OTHER")
                        delisted = meta["isdelisted"] == "Y"
                    else:
                        report["not_in_tickers"].append(name)
                        kind, exchange, delisted = "other", "OTHER", False
                    dates = bars["date"].to_numpy().astype("datetime64[s]")
                    initially, events = timeline.get(name, (None, []))
                    if initially is None:  # no recorded move: the current venue decides, and a delisted stock's
                        initially = _listed_venue(exchange) or delisted  # prices are its listed history
                    listed = _listed_mask(dates, initially, events)
                    if not listed.all():
                        report["unlisted_bars"] += int((~listed).sum())
                        if len(report["unlisted_examples"]) < EXAMPLES:
                            report["unlisted_examples"].append(f"{name} {int((~listed).sum())} bars")
                        bars, dates = bars[listed], dates[listed]
                    if not len(bars):
                        continue
                    close = bars["close"].to_numpy(np.float64)
                    closeadj = bars["closeadj"].to_numpy(np.float64)
                    raw = bars["closeunadj"].to_numpy(np.float64)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        factor = closeadj / close
                        series = {"open": bars["open"].to_numpy(np.float64) * factor,
                                  "high": bars["high"].to_numpy(np.float64) * factor,
                                  "low": bars["low"].to_numpy(np.float64) * factor,
                                  "close": closeadj.copy(),
                                  "volume": bars["volume"].to_numpy(np.float64) * close / closeadj,
                                  "close_raw": raw.copy()}
                        cash = np.zeros(len(bars))
                        if name in dividends:
                            days, values = dividends[name]
                            where = np.minimum(np.searchsorted(dates, days), len(dates) - 1)
                            hit = dates[where] == days  # an ex-date without a bar (e.g. off-exchange) is not moved
                            np.add.at(cash, where[hit], values[hit] * raw[where[hit]] / close[where[hit]])
                            report["dividends_matched"] += int(hit.sum())
                        series["dividend"] = np.where(np.isfinite(cash), cash, 0.0)
                    flags = _spikes(series["close"])
                    if flags.any():
                        report["spike_rows"] += int(flags.sum())
                        report["spikes"].extend(f"{name} {day}" for day in dates[flags].astype("datetime64[D]").astype(str))
                        for field in ("open", "high", "low", "close", "volume", "close_raw"):
                            series[field] = np.where(flags, np.nan, series[field])
                    for index in _level_breaks(series["close"], series["volume"]):
                        report["breaks"].append(f"{name} {str(dates[index])[:10]} "
                                                f"x{series['close'][index] / series['close'][index - 1]:.3g}")
                        for field in ("open", "high", "low", "close", "volume", "close_raw"):
                            series[field][:index] = np.nan
                    count = len(dates)
                    columns = {"ticker": pa.array([name] * count, pa.string()),
                               "exchange": pa.array([str(exchange)] * count, pa.string()),
                               "type": pa.array([kind] * count, pa.string()),
                               "date": pa.array(dates, pa.timestamp("s")),
                               **{field: pa.array(series[field], pa.float64()) for field in
                                  ("open", "high", "low", "close", "volume", "close_raw", "dividend")}}
                    if with_cap:
                        cap_values = bars["marketcap"].to_numpy(np.float64) * 1e6
                        report["cap_rows"] += int(np.isfinite(cap_values).sum())
                        columns["cap"] = pa.array(cap_values, pa.float64())
                    batch.append(pa.table(columns, schema=schema))
                    classification.setdefault(kind, []).append(name)
                    if delisted and kind == "stock":
                        delisted_written.append(name)
                    rows_written += count
                    if len(batch) >= 200:
                        writer.write_table(pa.concat_tables(batch))
                        batch = []
                if batch:
                    writer.write_table(pa.concat_tables(batch))
                _log(progress, f"Built bucket {code + 1}/{BUCKETS}")
    os.replace(temporary, destination)
    digest = sha256(destination)
    manifest = {
        "schema": 1, "builder": "sharadar", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "table": str(destination), "table_sha256": digest, "rows": rows_written, "start": start,
        "sources": {"sharadar": source.identity()},
        "conventions": {"prices": "split-, dividend- and spinoff-adjusted: split-adjusted OHLC x closeadj / close",
                        "volume": "shares scaled so close x volume is traded dollar volume",
                        "close_raw": "traded close (closeunadj); close / close_raw is the adjustment factor",
                        "dividend": "cash per share on the ex-date, traded share basis (actions value x closeunadj / close)",
                        "cap": "market capitalization in dollars (daily marketcap x 1e6)" if with_cap else None,
                        "factor": "not supplied: the snapshot keeps Qlib's identity factor (1)"},
        "input_rows": counts,
        "classification": {kind: {"count": len(names), "examples": sorted(names)[:EXAMPLES]}
                           for kind, names in sorted(classification.items())},
        "delisted_stocks": {"count": len(delisted_written), "examples": sorted(delisted_written)[:EXAMPLES]},
        "duplicate_symbols": {"count": len(source.duplicate_symbols), "examples": source.duplicate_symbols[:EXAMPLES]},
        "not_in_tickers": {"count": len(report["not_in_tickers"]), "examples": report["not_in_tickers"][:EXAMPLES]},
        "listing_filter": {"rule": "bars dated while the security traded off a US exchange (after an exchange move to "
                                   "OTC, or after the day of a 'delisted' action) are dropped",
                           "bars_dropped": report["unlisted_bars"], "examples": report["unlisted_examples"]},
        "duplicate_bars_dropped": report["duplicate_bars"],
        "dividends": {"actions": int(len(paid)), "matched_to_bars": report["dividends_matched"]},
        "cap_rows": report["cap_rows"] if with_cap else None,
        "level_breaks": {"rule": "close x3+ that persists, share volume halved or less, dollar volume within 2x: "
                                 "bars before the jump are set to NaN (series restarts)",
                         "count": len(report["breaks"]), "examples": report["breaks"][-EXAMPLES:]},
        "isolated_spikes": {"rule": "close moves 2x or 1/2 and is back within 25% of the prior close next day; "
                                    "the bar is set to NaN", "rows": report["spike_rows"],
                            "examples": report["spikes"][-EXAMPLES:]},
        "cutoff": None,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    manifest_path = destination.with_name(destination.name + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest
