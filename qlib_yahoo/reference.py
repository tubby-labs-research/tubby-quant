"""Security reference data: sector/industry groups for neutralization, kept as raw evidence.

Nasdaq's public stock screener lists every NYSE, Nasdaq and NYSE American stock
with its sector, industry, country and IPO year in one response, without an
account. Classifications are as of the fetch date (not point-in-time); a
company's sector rarely changes, but that limitation is recorded with the data.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from qlib_yahoo.dataset import DatasetError

SCREENER = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true"
BROWSER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
GROUP_FIELDS = ("sector", "industry", "country")


def nasdaq_symbol(symbol: str) -> str:
    """Nasdaq writes share classes as BRK/B; Stooq and Yahoo use BRK-B."""
    return symbol.strip().upper().replace("/", "-").replace("^", "_")


def fetch_screener(destination: Path, *, opener=urllib.request.urlopen) -> Path:
    """Fetch the screener once into ``destination`` (a new folder) with its SHA-256 and time."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    request = urllib.request.Request(SCREENER, headers={"User-Agent": BROWSER_AGENT, "Accept": "application/json"})
    with opener(request, timeout=120) as response:
        body = response.read()
    rows = parse_screener(body)
    (destination / "screener.json.gz").write_bytes(gzip.compress(body, mtime=0))
    manifest = {"schema": 1, "provider": "Nasdaq stock screener (public, no account)", "url": SCREENER,
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(), "sha256": hashlib.sha256(body).hexdigest(),
                "rows": len(rows), "point_in_time": False,
                "note": "sector/industry/country as of the fetch date; used as static groups"}
    temporary = destination / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, destination / "manifest.json")
    return destination


def parse_screener(body: bytes) -> list[dict[str, str]]:
    try:
        rows = json.loads(body)["data"]["rows"]
    except (ValueError, KeyError, TypeError) as error:
        raise DatasetError("Nasdaq screener response has no data rows") from error
    if not isinstance(rows, list) or not rows:
        raise DatasetError("Nasdaq screener response has no data rows")
    return rows


class Screener:
    """A fetched screener, verified against its recorded SHA-256."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        body = gzip.decompress((self.path / "screener.json.gz").read_bytes())
        if hashlib.sha256(body).hexdigest() != self.manifest.get("sha256"):
            raise DatasetError(f"Nasdaq screener file differs from its manifest: {self.path}")
        self.rows = {nasdaq_symbol(row["symbol"]): row for row in parse_screener(body)}

    def identity(self) -> dict[str, Any]:
        return {"path": str(self.path), "provider": self.manifest["provider"],
                "fetched_at_utc": self.manifest["fetched_at_utc"], "sha256": self.manifest["sha256"],
                "point_in_time": False}

    group_limitation = "classification as of the screener fetch, applied to the whole history"

    def groups(self, symbols: Iterable[str]) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """One row per symbol with sector, industry and country ('' when unknown), and coverage counts."""
        table, missing, blank = [], [], 0
        for symbol in sorted(symbols):
            row = self.rows.get(symbol)
            if row is None:
                missing.append(symbol)
            values = {name: ((row or {}).get(name) or "").strip() for name in GROUP_FIELDS}
            blank += not values["sector"]
            table.append({"symbol": symbol, **values})
        coverage = {"symbols": len(table), "not_in_screener": len(missing), "missing_examples": missing[:20],
                    "without_sector": blank,
                    "sectors": len({r["sector"] for r in table if r["sector"]}),
                    "industries": len({r["industry"] for r in table if r["industry"]})}
        return table, coverage


def group_source(path: Path) -> Any:
    """The sector/industry/country source in ``path``: a fetched Nasdaq screener or a Sharadar folder."""
    path = Path(path)
    if (path / "screener.json.gz").is_file():
        return Screener(path)
    from qlib_yahoo.sharadar import SharadarSource

    if SharadarSource.looks_like(path):
        return SharadarSource(path)
    raise DatasetError(f"groups must be a fetch-nasdaq-screener folder or a Sharadar folder: {path}")


def groups_csv(table: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=("symbol", *GROUP_FIELDS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(table)
    return buffer.getvalue().encode("utf-8")


def read_groups(path: Path) -> dict[str, dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {row["symbol"]: {name: row[name] for name in GROUP_FIELDS} for row in csv.DictReader(handle)}


def dataset_groups(dataset: Path, manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    """A snapshot's recorded groups, verified against the manifest hash ({} when it has none)."""
    record = manifest.get("groups") if isinstance(manifest, dict) else None
    if not record:
        return {}
    if record.get("file") != "universe/groups.csv":
        raise DatasetError("unexpected groups file in the dataset manifest")
    path = Path(dataset) / "universe" / "groups.csv"
    if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
        raise DatasetError("dataset groups file is missing or differs from the manifest")
    return read_groups(path)
