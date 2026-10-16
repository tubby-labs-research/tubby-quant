"""HistoricalData.net's US daily archive (bought once, delisted stocks included) -> consolidated table.

The archive (historicaldata.net, "Daily archive": one-time purchase, perpetual
licence) holds one CSV per listing lifecycle in ``day_by_symbol/`` from
2003-10-01: ``{SYMBOL}_day.csv`` for listed securities and
``{SYMBOL}_day_delisted_{YYYY-MM-DD}.csv`` for each delisted one, plus
``symbols.csv`` (one row per lifecycle: type, exchange, delisting date, CIK,
FIGI) and a ``manifest.json`` with every file's SHA-256. Each row carries raw
and adjusted OHLCV, VWAP, the cash dividend on its ex-date and the split ratio.

This module writes the long table the other builders write, so
import-stooq-table, the universe, Alpha Lab and the Qlib runner use it
unchanged:

* ``open high low close vwap`` are the archive's ``adj_*`` columns (split- and
  cash-distribution-adjusted), ``volume`` is ``adj_volume`` (adjusted close x
  volume = traded dollars, the same convention as the other tables),
  ``close_raw`` the traded close, ``dividend`` the USD cash per share on its
  ex-date;
* a symbol reused by a later company keeps its plain name for the listed
  lifecycle; earlier lifecycles get the delisting year and month appended
  (``ANET2412``), as Sharadar does with a number;
* securities that paid in a foreign currency have no ``adj_*`` columns before
  that payment (the vendor cannot convert it); their whole series is adjusted
  for splits only, so each security keeps one return definition;
* event-only rows (a dividend dated on a closed day) carry no prices and are
  not bars.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from qlib_yahoo.dataset import DatasetError, sha256

EXAMPLES = 20
FILE_NAME = re.compile(r"(?P<symbol>[^_]+)_day(?:_delisted_(?P<delisted>\d{4}-\d{2}-\d{2}))?\.csv\Z")
COLUMNS = ("date", "open", "high", "low", "close", "volume", "vwap", "transactions", "adj_open", "adj_high", "adj_low",
           "adj_close", "adj_volume", "adj_vwap", "dividend", "dividend_type", "split")
# Security types as the archive writes them (Polygon's reference types). Common equity in any form is a stock.
STOCK_TYPES = frozenset({"CS", "ADRC", "OS", "NYRS", "GDR"})
TYPE_GROUPS = {"ETF": "fund", "ETN": "fund", "ETV": "fund", "ETS": "fund", "FUND": "fund", "PFD": "preferred",
               "ADRP": "preferred", "WARRANT": "warrant", "ADRW": "warrant", "RIGHT": "right", "ADRR": "right",
               "UNIT": "unit", "SP": "other", "BOND": "other", "INDEX": "other", "BASKET": "other", "LT": "other"}
EXCHANGES = {"XNYS": "NYSE", "NYSE": "NYSE", "XNAS": "NASDAQ", "NASDAQ": "NASDAQ", "XASE": "NYSEMKT",
             "NYSEMKT": "NYSEMKT", "NYSE AMERICAN": "NYSEMKT", "AMEX": "NYSEMKT", "ARCX": "NYSEARCA",
             "NYSEARCA": "NYSEARCA", "NYSE ARCA": "NYSEARCA", "BATS": "BATS", "CBOE": "BATS", "IEXG": "IEX"}
OTC = frozenset({"OTC", "OTCM", "OOTC", "PINX", "OTCQX", "OTCQB", "XOTC"})
# Blank-check companies are common stock by type; their names give them away.
SPAC_NAME = re.compile(r"\bacquisition\s+(corp|corporation|company|co|inc|ltd|limited|holdings)\b", re.I)


def _log(progress: Callable[[str], None] | None, message: str) -> None:
    if progress:
        progress(message)


def symbol(text: str) -> str:
    """The archive writes share classes as BRK.B; the tables here use BRK-B."""
    return text.strip().upper().replace(".", "-")


def security_type(kind: str, name: str, exchange: str) -> str:
    kind = (kind or "").strip().upper()
    venue = (exchange or "").strip().upper()
    if venue in OTC:
        return "otc"
    if kind in STOCK_TYPES:
        return "spac" if SPAC_NAME.search(name or "") else "stock"
    return TYPE_GROUPS.get(kind, "other")


class Archive:
    """An unpacked archive folder: ``symbols.csv`` and ``day_by_symbol/`` (manifest optional but checked when present)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.days = self.path / "day_by_symbol"
        if not self.days.is_dir():
            raise DatasetError(f"{self.path} has no day_by_symbol folder (the daily archive)")
        listing = self.path / "symbols.csv"
        if not listing.is_file():
            raise DatasetError(f"{self.path} has no symbols.csv (security types and exchanges)")
        with listing.open(newline="", encoding="utf-8") as handle:
            self.symbols = list(csv.DictReader(handle))
        missing = {"symbol", "type", "exchange", "delisted_at"} - set(self.symbols[0] if self.symbols else {})
        if missing:
            raise DatasetError(f"symbols.csv lacks columns {sorted(missing)}")
        self.listing_sha256 = sha256(listing)
        self.hashes: dict[str, str] = {}
        for candidate in (self.days / "manifest.json", self.path / "manifest.json"):
            if candidate.is_file():
                record = json.loads(candidate.read_text(encoding="utf-8"))
                for item in record.get("files") or []:
                    self.hashes[Path(str(item.get("name", ""))).name] = str(item.get("sha256", ""))
                self.manifest = {"file": str(candidate), "sha256": sha256(candidate),
                                 "product": record.get("product"), "first_day": record.get("first_day"),
                                 "last_day": record.get("last_day")}
                break
        else:
            self.manifest = None

    def identity(self) -> dict[str, Any]:
        return {"path": str(self.path), "provider": "HistoricalData.net US stock daily archive",
                "symbols_csv_sha256": self.listing_sha256, "manifest": self.manifest,
                "licence": "per-user licence; data must not be redistributed (historicaldata.net/terms.html#license)"}


def _split_factor(text: str) -> float:
    """``from:to`` ratios (several joined with +) as the price factor for earlier rows."""
    factor = 1.0
    for part in (text or "").split("+"):
        if ":" in part:
            left, right = part.split(":", 1)
            try:
                a, b = float(left), float(right)
            except ValueError:
                continue
            if a > 0 and b > 0:
                factor *= a / b
    return factor


def read_lifecycle(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """One lifecycle file as columns, with the adjustment rule applied (see the module notes)."""
    import io

    import numpy as np
    import pyarrow as pa
    import pyarrow.csv as pacsv

    data = path.read_bytes()
    if expected_sha256 and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise DatasetError(f"{path.name} differs from the archive manifest")
    numeric = ("open", "high", "low", "close", "volume", "vwap", "adj_open", "adj_high", "adj_low", "adj_close",
               "adj_volume", "adj_vwap", "dividend")
    wanted = ("date", *numeric, "dividend_type", "split")
    try:
        table = pacsv.read_csv(io.BytesIO(data), convert_options=pacsv.ConvertOptions(
            include_columns=list(wanted), strings_can_be_null=True, null_values=[""],
            column_types={**{name: pa.float64() for name in numeric}, "date": pa.string(),
                          "dividend_type": pa.string(), "split": pa.string()}))
    except (pa.ArrowInvalid, KeyError) as error:
        raise DatasetError(f"cannot read {path.name}: {error}") from error
    column = lambda name: table.column(name).to_numpy(zero_copy_only=False).astype(float)  # noqa: E731
    raw = {name: column(name) for name in ("open", "high", "low", "close", "volume", "vwap")}
    adj = {name: column(f"adj_{name}") for name in ("open", "high", "low", "close", "volume", "vwap")}
    dividends = column("dividend")
    kinds = [kind or "" for kind in table.column("dividend_type").to_pylist()]
    splits = [split or "" for split in table.column("split").to_pylist()]
    count = table.num_rows
    priced = np.isfinite(raw["close"])
    split_only = bool((priced & ~np.isfinite(adj["close"])).any())
    if split_only:
        factor = np.ones(count)
        running = 1.0
        for index in range(count - 1, -1, -1):
            factor[index] = running
            running *= _split_factor(splits[index])  # an event on row i adjusts the rows before it
        for name in ("open", "high", "low", "close", "vwap"):
            adj[name] = raw[name] * factor
        with np.errstate(divide="ignore", invalid="ignore"):
            adj["volume"] = raw["volume"] / factor
    foreign = np.array(["CUR:" in kind for kind in kinds], dtype=bool)
    cash = np.where(np.isfinite(dividends) & ~foreign, dividends, 0.0)
    keep = priced & np.isfinite(adj["close"])
    dates = np.array(table.column("date").to_pylist(), dtype="datetime64[D]")
    return {"dates": dates[keep].astype("datetime64[s]"),
            "open": adj["open"][keep], "high": adj["high"][keep], "low": adj["low"][keep], "close": adj["close"][keep],
            "volume": adj["volume"][keep], "vwap": adj["vwap"][keep], "close_raw": raw["close"][keep],
            "dividend": cash[keep], "split_only": split_only, "event_only_rows": int((~priced).sum()),
            "foreign_dividends": int(foreign.sum()), "rows": count}


def lifecycle_names(listing: list[dict[str, str]]) -> dict[tuple[str, str], str]:
    """(symbol, delisted_at) -> the table's ticker: plain when unambiguous, else the delisting month appended."""
    by_symbol: dict[str, list[dict[str, str]]] = {}
    for row in listing:
        if (row.get("status") or "").strip().lower() == "renamed":
            continue  # a previous holder whose rows live in the current holder's file
        by_symbol.setdefault(symbol(row["symbol"]), []).append(row)
    names: dict[tuple[str, str], str] = {}
    for base, rows in by_symbol.items():
        active = [row for row in rows if not (row.get("delisted_at") or "").strip()]
        for row in rows:
            delisted = (row.get("delisted_at") or "").strip()
            key = (row["symbol"].strip(), delisted)
            if not delisted or (len(rows) == 1 and not active):
                names[key] = base
                continue
            name = f"{base}{delisted[2:4]}{delisted[5:7]}"
            if name in names.values():
                name = f"{base}{delisted[2:4]}{delisted[5:7]}{delisted[8:10]}"
            names[key] = name
    return names


def build_table(source: Path, destination: Path, *, start: str | None = None,
                progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Write ``destination`` (Parquet) and ``destination``.manifest.json from an unpacked archive; return the manifest."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from qlib_yahoo.yahoo_table import _level_breaks, _spikes

    archive = Archive(Path(source))
    destination = Path(destination)
    if destination.exists():
        raise DatasetError(f"{destination} exists; choose a new file")
    begin = np.datetime64(start, "s") if start else None
    names = lifecycle_names(archive.symbols)
    info = {(row["symbol"].strip(), (row.get("delisted_at") or "").strip()): row for row in archive.symbols}
    files = sorted(item for item in archive.days.iterdir() if item.is_file() and FILE_NAME.fullmatch(item.name))
    schema = pa.schema([("ticker", pa.string()), ("exchange", pa.string()), ("type", pa.string()), ("date", pa.timestamp("s"))]
                       + [(name, pa.float64()) for name in ("open", "high", "low", "close", "volume", "close_raw", "dividend", "vwap")])
    report: dict[str, Any] = {"classification": {}, "not_in_symbols": [], "split_only": [], "spikes": [], "spike_rows": 0,
                              "breaks": [], "event_only_rows": 0, "foreign_dividends": 0, "exchanges": {}, "delisted_stocks": 0,
                              "suffixed": [], "unverified": 0}
    listed_files = set()
    rows_written = 0
    temporary = destination.with_name(destination.name + ".tmp")
    from concurrent.futures import ThreadPoolExecutor

    def load(path: Path) -> dict[str, Any]:
        expected = archive.hashes.get(path.name)
        return {"series": read_lifecycle(path, expected), "verified": bool(expected)}

    def loaded_in_order():  # a bounded window of files read ahead in parallel
        for first in range(0, len(files), 256):
            chunk = files[first:first + 256]
            yield from zip(chunk, pool.map(load, chunk))

    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer, ThreadPoolExecutor(max_workers=8) as pool:
        batch = []
        for number, (path, loaded) in enumerate(loaded_in_order()):
            match = FILE_NAME.fullmatch(path.name)
            key = (match["symbol"], match["delisted"] or "")
            listed_files.add(key)
            meta = info.get(key)
            if meta is None:
                report["not_in_symbols"].append(path.name)
                kind, exchange = "other", "OTHER"
            else:
                venue = (meta.get("exchange") or "").strip().upper()
                exchange = EXCHANGES.get(venue, "OTC" if venue in OTC else (venue or "OTHER"))
                kind = security_type(meta.get("type", ""), meta.get("name", ""), venue)
                report["exchanges"][exchange] = report["exchanges"].get(exchange, 0) + 1
            ticker = names.get(key) or symbol(match["symbol"])
            if archive.hashes and not loaded["verified"]:
                report["unverified"] += 1
            series = loaded["series"]
            report["event_only_rows"] += series["event_only_rows"]
            report["foreign_dividends"] += series["foreign_dividends"]
            if series["split_only"]:
                report["split_only"].append(ticker)
            if begin is not None:
                keep = series["dates"] >= begin
                for field in ("dates", "open", "high", "low", "close", "volume", "vwap", "close_raw", "dividend"):
                    series[field] = series[field][keep]
            if not len(series["dates"]):
                continue
            flags = _spikes(series["close"])
            if flags.any():
                report["spike_rows"] += int(flags.sum())
                report["spikes"].extend(f"{ticker} {day}" for day in series["dates"][flags].astype("datetime64[D]").astype(str))
                for field in ("open", "high", "low", "close", "volume", "vwap", "close_raw"):
                    series[field] = np.where(flags, np.nan, series[field])
            for index in _level_breaks(series["close"], series["volume"]):
                report["breaks"].append(f"{ticker} {str(series['dates'][index])[:10]} x{series['close'][index] / series['close'][index - 1]:.3g}")
                for field in ("open", "high", "low", "close", "volume", "vwap", "close_raw"):
                    series[field][:index] = np.nan
            count = len(series["dates"])
            batch.append(pa.table({
                "ticker": pa.array([ticker] * count, pa.string()), "exchange": pa.array([exchange] * count, pa.string()),
                "type": pa.array([kind] * count, pa.string()), "date": pa.array(series["dates"], pa.timestamp("s")),
                **{field: pa.array(series[field], pa.float64()) for field in
                   ("open", "high", "low", "close", "volume", "close_raw", "dividend", "vwap")}}, schema=schema))
            report["classification"].setdefault(kind, []).append(ticker)
            if key[1] and kind == "stock":
                report["delisted_stocks"] += 1
            if ticker != symbol(match["symbol"]):
                report["suffixed"].append(f"{match['symbol']} delisted {key[1]} -> {ticker}")
            rows_written += count
            if len(batch) >= 200:
                writer.write_table(pa.concat_tables(batch))
                batch = []
            if progress and (number + 1) % 2000 == 0:
                progress(f"Built {number + 1:,}/{len(files):,} files")
        if batch:
            writer.write_table(pa.concat_tables(batch))
    os.replace(temporary, destination)
    without_file = [f"{s}{' delisted ' + d if d else ''}" for (s, d) in info if (s, d) not in listed_files
                    and (info[(s, d)].get("status") or "").strip().lower() != "renamed"]
    manifest = {
        "schema": 1, "builder": "historicaldata", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "table": str(destination), "table_sha256": sha256(destination), "rows": rows_written, "start": start,
        "sources": {"historicaldata": archive.identity()},
        "conventions": {"prices": "archive adj_* columns: split- and cash-distribution-adjusted (split-only for securities "
                                  "that paid in a foreign currency, listed under split_only_adjusted)",
                        "volume": "adj_volume: adjusted close x volume is traded dollar volume",
                        "close_raw": "traded close", "dividend": "USD cash per share on the ex-date, traded share basis",
                        "vwap": "adj_vwap, the day's volume-weighted price, adjusted like the other prices",
                        "factor": "not supplied: the snapshot keeps Qlib's identity factor (1)"},
        "files": len(files), "manifest_checked": bool(archive.hashes), "files_without_manifest_hash": report["unverified"],
        "classification": {kind: {"count": len(items), "examples": sorted(items)[:EXAMPLES]}
                           for kind, items in sorted(report["classification"].items())},
        "delisted_stocks": {"count": report["delisted_stocks"]},
        "exchanges": dict(sorted(report["exchanges"].items())),
        "reused_symbols": {"rule": "earlier lifecycles of a reused symbol get the delisting year and month appended",
                           "count": len(report["suffixed"]), "examples": report["suffixed"][:EXAMPLES]},
        "split_only_adjusted": {"rule": "no adjusted prices from the archive (a foreign-currency distribution): the whole "
                                        "series is adjusted for splits only and those distributions are left out",
                                "count": len(report["split_only"]), "examples": report["split_only"][:EXAMPLES]},
        "event_only_rows_skipped": report["event_only_rows"], "foreign_currency_dividends_skipped": report["foreign_dividends"],
        "not_in_symbols_csv": {"count": len(report["not_in_symbols"]), "examples": report["not_in_symbols"][:EXAMPLES]},
        "symbols_without_file": {"count": len(without_file), "examples": without_file[:EXAMPLES]},
        "level_breaks": {"rule": "close x3+ that persists, share volume halved or less, dollar volume within 2x: "
                                 "bars before the jump are set to NaN (series restarts)",
                         "count": len(report["breaks"]), "examples": report["breaks"][-EXAMPLES:]},
        "isolated_spikes": {"rule": "close moves 2x or 1/2 and is back within 25% of the prior close next day; "
                                    "the bar is set to NaN", "rows": report["spike_rows"], "examples": report["spikes"][-EXAMPLES:]},
        "cutoff": None,
        "builder_sha256": sha256(Path(__file__)),
    }
    manifest_path = destination.with_name(destination.name + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest
