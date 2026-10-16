"""Yahoo Finance daily chart history with split and dividend events, kept as raw evidence.

The public chart endpoint needs no account. Its ``close`` is adjusted for splits
only and ``adjclose`` also for dividends; split events carry exact ratios. An
archive is fetched once into its own folder under the data root: one gzip file
per ticker holding the untouched response, and a manifest with each response's
SHA-256, so later imports read only verified local bytes. A partly fetched
archive resumes; nothing already fetched is downloaded again.
"""

from __future__ import annotations

import gzip
import hashlib
import http.client
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from qlib_yahoo.dataset import DatasetError

ENDPOINT = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
USER_AGENT = "Mozilla/5.0 (Tubby Labs research; daily corporate actions)"
PROVIDER = "Yahoo Finance chart API v8 (public, no account)"
ARCHIVE_SCHEMA = 1


@dataclass(frozen=True)
class ChartHistory:
    """One ticker's parsed daily history. Dates are exchange-local trading dates."""

    symbol: str
    dates: list[str]
    open: list[float | None]
    high: list[float | None]
    low: list[float | None]
    close: list[float | None]
    adjclose: list[float | None]
    volume: list[float | None]
    splits: list[tuple[str, float, float]] = field(default_factory=list)  # (date, numerator, denominator)
    dividends: list[tuple[str, float]] = field(default_factory=list)  # (ex-date, cash per share, split-adjusted)
    meta: dict[str, Any] = field(default_factory=dict)


def yahoo_symbol(ticker: str) -> str:
    """Stooq and Yahoo both write share classes with a dash (BRK-B); dots become dashes."""
    return ticker.strip().upper().replace(".", "-")


def _local_date(timestamp: int, offset: int) -> str:
    return (datetime.fromtimestamp(int(timestamp), timezone.utc) + timedelta(seconds=offset)).date().isoformat()


def parse_chart(symbol: str, payload: bytes) -> ChartHistory:
    """Parse a chart response; raise DatasetError for an error response or malformed body."""
    try:
        document = json.loads(payload)
    except ValueError as error:
        raise DatasetError(f"{symbol}: chart response is not JSON") from error
    chart = document.get("chart") if isinstance(document, dict) else None
    if not isinstance(chart, dict) or not chart.get("result"):
        detail = (chart or {}).get("error") if isinstance(chart, dict) else None
        raise DatasetError(f"{symbol}: no chart data ({(detail or {}).get('description', 'empty result')})")
    result = chart["result"][0]
    meta = result.get("meta") or {}
    offset = int(meta.get("gmtoffset") or 0)
    stamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    adjusted = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or [None] * len(stamps)
    columns = {name: quote.get(name) or [None] * len(stamps) for name in ("open", "high", "low", "close", "volume")}
    if any(len(values) != len(stamps) for values in (*columns.values(), adjusted)):
        raise DatasetError(f"{symbol}: chart arrays have different lengths")
    events = result.get("events") or {}
    splits = sorted((_local_date(int(v.get("date", k)), offset), float(v["numerator"]), float(v["denominator"]))
                    for k, v in (events.get("splits") or {}).items())
    dividends = sorted((_local_date(int(v.get("date", k)), offset), float(v["amount"]))
                       for k, v in (events.get("dividends") or {}).items())
    for day, numerator, denominator in splits:
        if not (numerator > 0 and denominator > 0):
            raise DatasetError(f"{symbol}: split on {day} has a nonpositive ratio")
    keep = ("symbol", "currency", "exchangeName", "fullExchangeName", "instrumentType", "firstTradeDate",
            "longName", "shortName", "gmtoffset", "exchangeTimezoneName", "dataGranularity")
    return ChartHistory(symbol=symbol, dates=[_local_date(stamp, offset) for stamp in stamps],
                        adjclose=list(adjusted), splits=splits, dividends=dividends,
                        meta={key: meta[key] for key in keep if key in meta}, **columns)


def _request(symbol: str, start: date, end: date, timeout: float) -> tuple[int, bytes]:
    query = urllib.parse.urlencode({
        "period1": int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp()),
        "period2": int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp()),
        "interval": "1d", "events": "div|split", "includeAdjustedClose": "true",
    })
    url = ENDPOINT.format(symbol=urllib.parse.quote(symbol, safe="-")) + "?" + query
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read() or b""


def fetch_chart(symbol: str, start: date, end: date, *, retries: int = 5, timeout: float = 30.0,
                sleep: Callable[[float], None] = time.sleep,
                request: Callable[[str, date, date, float], tuple[int, bytes]] = _request) -> tuple[int, bytes]:
    """Return (HTTP status, body). 404 is a final answer (unknown symbol); 429/5xx and network errors retry."""
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            status, body = request(symbol, start, end, timeout)
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError, OSError) as error:
            status, body = 0, str(error).encode()
        if status == 200 or status == 404 or (400 <= status < 500 and status != 429):
            return status, body
        if attempt < retries:
            sleep(delay + random.uniform(0, delay / 2))
            delay = min(delay * 2, 120.0)
    return status, body


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fetch_archive(tickers: Iterable[str], destination: Path, start: str, end: str, *, workers: int = 6,
                  progress: Callable[[str], None] | None = None,
                  fetch: Callable[..., tuple[int, bytes]] = fetch_chart) -> Path:
    """Fetch every ticker into ``destination`` (created or resumed); return it once complete.

    Each response is stored byte for byte (gzip, no timestamp) and hashed. Tickers
    Yahoo does not know are recorded with their status, not retried forever.
    """
    begin, stop = date.fromisoformat(start), date.fromisoformat(end)
    if begin >= stop:
        raise DatasetError("start must be before end")
    if type(workers) is not int or not 1 <= workers <= 16:
        raise DatasetError("workers must be an integer from 1 to 16")
    wanted = sorted({yahoo_symbol(ticker) for ticker in tickers})
    if not wanted:
        raise DatasetError("no tickers to fetch")
    destination = Path(destination)
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("request", {}).get("start") != start or manifest.get("request", {}).get("end") != end:
            raise DatasetError(f"{destination} was fetched for a different period; use a new folder")
        if manifest.get("status") == "ready" and set(wanted) <= set(manifest["tickers"]):
            return destination
    else:
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "charts").mkdir()
        manifest = {"schema": ARCHIVE_SCHEMA, "provider": PROVIDER, "endpoint": ENDPOINT,
                    "request": {"start": start, "end": end, "interval": "1d", "events": "div|split"},
                    "semantics": {"close": "split-adjusted only", "adjclose": "split and dividend adjusted",
                                  "splits": "numerator new shares per denominator old shares, dated on the ex-date",
                                  "dividends": "cash per share on the ex-date, split-adjusted"},
                    "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "building", "tickers": {}}
        _write_json(manifest_path, manifest)
    done = manifest["tickers"]
    todo = [symbol for symbol in wanted if symbol not in done or done[symbol].get("status") not in (200, 404)]
    manifest["status"] = "building"

    def one(symbol: str) -> tuple[str, dict[str, Any]]:
        try:
            return fetch_one(symbol)
        except Exception as error:  # one broken response must not stop the archive; it is retried on resume
            return symbol, {"status": "error", "error": f"{type(error).__name__}: {error}"[:300],
                            "fetched_at_utc": datetime.now(timezone.utc).isoformat()}

    def fetch_one(symbol: str) -> tuple[str, dict[str, Any]]:
        status, body = fetch(symbol, begin, stop)
        record: dict[str, Any] = {"status": status, "fetched_at_utc": datetime.now(timezone.utc).isoformat()}
        if status == 200:
            try:
                history = parse_chart(symbol, body)
            except DatasetError as error:
                return symbol, {**record, "status": "unparsable", "error": str(error)}
            path = destination / "charts" / f"{symbol}.json.gz"
            path.write_bytes(gzip.compress(body, mtime=0))
            record.update(sha256=hashlib.sha256(body).hexdigest(), bytes=len(body), bars=len(history.dates),
                          first=history.dates[0] if history.dates else None,
                          last=history.dates[-1] if history.dates else None,
                          splits=len(history.splits), dividends=len(history.dividends),
                          instrument_type=history.meta.get("instrumentType"))
        else:
            record["error"] = body[:300].decode("utf-8", "replace")
        return symbol, record

    finished = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for symbol, record in pool.map(one, todo):
            done[symbol] = record
            finished += 1
            if finished % 100 == 0 or finished == len(todo):
                _write_json(manifest_path, manifest)
                if progress:
                    progress(f"Fetched {finished:,}/{len(todo):,} charts")
    failed = sorted(symbol for symbol in wanted if done.get(symbol, {}).get("status") not in (200, 404))
    manifest["status"] = "ready" if not failed else "incomplete"
    manifest["summary"] = {
        "tickers": len(done), "ok": sum(1 for r in done.values() if r.get("status") == 200),
        "unknown_404": sum(1 for r in done.values() if r.get("status") == 404), "failed": failed[:50],
        "failed_count": len(failed),
    }
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    if failed:
        raise DatasetError(f"{len(failed)} tickers failed (e.g. {', '.join(failed[:5])}); run again to resume")
    return destination


class ChartArchive:
    """Read a fetched archive, verifying each response against its recorded SHA-256."""

    def __init__(self, path: Path):
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        if not manifest_path.is_file():
            raise DatasetError(f"not a Yahoo chart archive: {self.path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != ARCHIVE_SCHEMA or self.manifest.get("status") != "ready":
            raise DatasetError(f"Yahoo chart archive is not complete: {self.path}")

    def symbols(self) -> list[str]:
        return sorted(s for s, r in self.manifest["tickers"].items() if r.get("status") == 200)

    def history(self, ticker: str) -> ChartHistory | None:
        symbol = yahoo_symbol(ticker)
        record = self.manifest["tickers"].get(symbol)
        if not record or record.get("status") != 200:
            return None
        body = gzip.decompress((self.path / "charts" / f"{symbol}.json.gz").read_bytes())
        if hashlib.sha256(body).hexdigest() != record["sha256"]:
            raise DatasetError(f"{symbol}: stored chart does not match its recorded SHA-256")
        return parse_chart(symbol, body)

    def identity(self) -> dict[str, Any]:
        manifest_bytes = (self.path / "manifest.json").read_bytes()
        return {"path": str(self.path), "provider": self.manifest["provider"], "request": self.manifest["request"],
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "tickers_ok": self.manifest["summary"]["ok"], "unknown_404": self.manifest["summary"]["unknown_404"]}
