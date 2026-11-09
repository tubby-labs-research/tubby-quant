"""Read a verified snapshot into dates x instruments matrices.

Used where a whole cross-section is needed at once (data-quality scans and the
vectorized alpha simulator) instead of Qlib's per-instrument expression engine.
The snapshot is verified first (tree hash, or archive bytes for packed
snapshots), then the Qlib float32 bins are placed by their calendar offsets.
Missing bars stay NaN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from qlib_yahoo.dataset import DatasetError


@dataclass
class Panel:
    dataset: Path
    manifest: dict[str, Any]
    identity: dict[str, Any]
    dates: Any  # numpy datetime64[D], ascending; the snapshot's calendar, or the window load_panel was bounded to
    symbols: list[str]
    fields: dict[str, Any] = field(default_factory=dict)  # name -> float32 (dates, symbols)

    def column(self, symbol: str) -> int:
        try:
            return self.symbols.index(symbol)
        except ValueError:
            raise DatasetError(f"panel has no instrument {symbol}") from None

    def row(self, day: str) -> int:
        import numpy as np

        index = int(np.searchsorted(self.dates, np.datetime64(day, "D")))
        if index >= len(self.dates) or self.dates[index] != np.datetime64(day, "D"):
            raise DatasetError(f"{day} is not a session of this dataset")
        return index


def available_fields(manifest: dict[str, Any]) -> list[str]:
    """Fields every instrument of the snapshot carries."""
    return list(manifest.get("fields") or ("open", "high", "low", "close", "volume", "factor"))


def load_panel(dataset: Path, fields: Sequence[str] = ("open", "high", "low", "close", "volume"),
               symbols: Sequence[str] | None = None, *, start_date: str | None = None,
               end_date: str | None = None) -> Panel:
    """Verify ``dataset`` and return the requested fields for all (or the given) instruments.

    ``start_date`` and ``end_date`` (ISO dates, inclusive) bound the sessions that are read. The
    returned calendar runs from the first session at or after ``start_date`` to the last session at
    or before ``end_date``, and bars outside that window are never deserialized: a plain snapshot
    reads only the bytes of the bounded rows of each instrument's bin file, a packed snapshot copies
    only those rows out of its verified archive. Research code relies on this to keep a reserved
    period out of a process altogether rather than slicing it away after loading. A bound beyond
    the calendar is clipped; a window without a session is an error.
    """
    import numpy as np

    from qlib_yahoo.dataset_integrity import verify_dataset

    dataset = Path(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    known = available_fields(manifest)
    unknown = [name for name in fields if name not in known]
    if unknown:
        raise DatasetError(f"dataset has no field {unknown[0]!r} (available: {', '.join(known)})")
    identity = verify_dataset(dataset, manifest)
    qlib = dataset / "qlib_data"
    calendar = (qlib / "calendars" / "day.txt").read_text(encoding="utf-8").split()
    full = np.asarray(calendar, dtype="datetime64[D]")
    first_row = int(np.searchsorted(full, np.datetime64(start_date, "D"))) if start_date else 0
    last_row = int(np.searchsorted(full, np.datetime64(end_date, "D"), side="right")) if end_date else len(full)
    if last_row <= first_row:
        raise DatasetError(f"no session between {start_date or 'the first session'} and {end_date or 'the last session'}")
    dates = full[first_row:last_row]
    listed = [line.split("\t", 1)[0] for line in (qlib / "instruments" / "all.txt").read_text(encoding="utf-8").splitlines() if line]
    chosen = listed if symbols is None else [s for s in symbols if s in set(listed)]
    if symbols is not None and len(chosen) != len(set(symbols)):
        missing = sorted(set(symbols) - set(listed))
        raise DatasetError(f"dataset has no instrument {missing[0]}")
    entries = None
    if "storage" in manifest:
        from qlib_yahoo.packed import verified_entries

        entries = verified_entries(dataset, manifest["storage"]["archive_sha256"])

    def window(symbol: str, name: str, first: int, count: int) -> tuple[int, int]:
        """Rows [lo, hi) of the calendar where this instrument's bars and the requested window meet."""
        if first < 0 or first + count > len(full):
            raise DatasetError(f"{symbol}/{name} extends beyond the calendar")
        return max(first, first_row), min(first + count, last_row)

    def read(symbol: str, name: str) -> tuple[int, Any] | None:
        """(first calendar row, bars) of the instrument inside the window, or None when it has none there."""
        relative = f"features/{symbol.lower()}/{name}.day.bin"
        if entries is not None:
            data = entries.get(relative)
            if data is None or len(data) < 8:
                return None
            payload = np.frombuffer(data, dtype="<f4")  # a view of the verified bytes, nothing copied yet
            first = int(payload[0])
            lo, hi = window(symbol, name, first, len(payload) - 1)
            return None if hi <= lo else (lo, payload[1 + lo - first:1 + hi - first])
        path = qlib / relative
        if not path.is_file() or path.stat().st_size < 8:
            return None
        first = int(np.fromfile(path, dtype="<f4", count=1)[0])
        lo, hi = window(symbol, name, first, path.stat().st_size // 4 - 1)
        if hi <= lo:
            return None
        return lo, np.fromfile(path, dtype="<f4", count=hi - lo, offset=4 * (1 + lo - first))

    arrays = {name: np.full((len(dates), len(chosen)), np.nan, dtype=np.float32) for name in fields}
    for column, symbol in enumerate(chosen):
        for name in fields:
            bars = read(symbol, name)
            if bars is None:
                continue
            lo, values = bars
            arrays[name][lo - first_row:lo - first_row + len(values), column] = values
    return Panel(dataset=dataset, manifest=manifest, identity=identity, dates=dates, symbols=list(chosen), fields=arrays)


def membership_mask(panel: Panel, universe: str) -> Any:
    """Boolean (dates, symbols) mask of a sealed dated universe of the snapshot."""
    import numpy as np

    from qlib_yahoo.universe import load_dataset_universe

    spans, _record = load_dataset_universe(panel.dataset, panel.manifest, universe)
    mask = np.zeros((len(panel.dates), len(panel.symbols)), dtype=bool)
    position = {symbol: column for column, symbol in enumerate(panel.symbols)}
    for symbol, intervals in spans.items():
        column = position.get(symbol)
        if column is None:
            continue
        for begin, end in intervals:
            first = int(np.searchsorted(panel.dates, np.datetime64(str(begin)[:10], "D")))
            last = int(np.searchsorted(panel.dates, np.datetime64(str(end)[:10], "D"), side="right"))
            mask[first:last, column] = True
    return mask
