"""Artificial demonstration prices for a first session with the console — not market data.

A data folder starts empty, and nothing in the console works without prepared
prices. This module writes a small artificial snapshot in minutes-to-seconds so
a newcomer can simulate an alpha, validate it and run a discovery campaign
before downloading real data. The prices come from the lab's planted-panel
generator: three known signals (short-term reversal, a volume shock and low
volatility) are planted at modest strengths, so the bundled Alpha Lab examples
have something to find and a search that finds nothing on it is visibly broken.

The snapshot uses the layout every console engine reads: Qlib binary features
(open, high, low, close, volume and an identity ``factor``), a calendar,
instrument lists, a dated TOP-N universe and a hash-sealed manifest. The
manifest says plainly that the data is artificial. No network access, and no
writes outside ``directory``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Callable

PROVIDER = "console-demo-planted-v1"
DEFAULT_NAME = "demo-planted-200"
NOTE = ("Artificial demonstration prices for 200 fictional stocks with three planted signals. "
        "Generated locally; no market, economic or investment evidence.")
MIN_HISTORY_SESSIONS = 120


def create_demo_market(directory: str | Path, *, symbols: int = 200, days: int = 1500, seed: int = 11,
                       start: str = "2018-01-02", log: Callable[[str], None] | None = None) -> Path:
    """Write the demonstration snapshot into an empty ``directory`` and verify it; idempotent."""
    try:
        from autoresearch.domains.alpha.planted import ORACLES, generate
    except ImportError as error:  # a framework export without the research domains
        raise ValueError("Demonstration data needs the research-domain package (autoresearch.domains); "
                         "download or import real prices on the Data page instead.") from error
    import numpy as np

    from qlib_yahoo.alphalab.panel import source_identity
    from qlib_yahoo.dataset import tree_sha256

    say = log or (lambda text: None)
    directory = Path(directory).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("provider") != PROVIDER:
            raise ValueError(f"{directory} already holds a dataset that is not the console demonstration; refusing to replace it")
        say("Demonstration data already exists; verifying it.")
        source_identity(directory)
        return directory
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"{directory} is not empty; choose a new folder name for the demonstration data")
    if symbols < 20 or days < 400:
        raise ValueError("the demonstration needs at least 20 symbols and 400 sessions")

    say(f"Generating {symbols} artificial stocks over {days} sessions (seed {seed}) ...")
    data = generate(symbols, days, seed=seed, start=start)
    dates, names = data["dates"], data["symbols"]
    fields = dict(data["fields"])
    fields["factor"] = np.ones_like(fields["close"])  # already adjusted prices: identity factor everywhere

    say("Writing Qlib binary features ...")
    qlib = directory / "qlib_data"
    (qlib / "calendars").mkdir(parents=True)
    (qlib / "instruments").mkdir()
    (qlib / "calendars/day.txt").write_text("\n".join(map(str, dates)) + "\n", encoding="utf-8")
    spans = "".join(f"{s}\t{dates[0]}\t{dates[-1]}\n" for s in names)
    (qlib / "instruments/all.txt").write_text(spans, encoding="utf-8")
    for column, symbol in enumerate(names):
        folder = qlib / "features" / symbol.lower()
        folder.mkdir(parents=True)
        for field, values in fields.items():
            np.concatenate(([0], values[:, column])).astype("<f4").tofile(folder / f"{field}.day.bin")

    say("Building the dated universe ...")
    universe_name = f"top{symbols}"
    first_member = dates[min(MIN_HISTORY_SESSIONS, len(dates) - 1)]
    members = "".join(f"{s}\t{first_member}\t{dates[-1]}\n" for s in names)
    instruments = qlib / "instruments" / f"{universe_name}.txt"
    instruments.write_text(members, encoding="utf-8")
    dollar = (fields["close"] * fields["volume"]).mean(axis=0)
    order = np.argsort(-dollar)
    universe = directory / "universe"
    universe.mkdir()
    membership = universe / "membership.csv.gz"
    rows = "".join(f"{names[i]},{first_member},{dates[-1]},{rank + 1}\n" for rank, i in enumerate(order))
    membership.write_bytes(gzip.compress(("instrument,effective_start,effective_end,rank\n" + rows).encode(), mtime=0))

    say("Sealing the manifest ...")
    end_exclusive = str(dates[-1] + np.timedelta64(1, "D"))
    manifest = {
        "status": "ready", "provider": PROVIDER, "note": NOTE, "artificial": True,
        "planted": {"signals": data["strength"], "seed": seed, "oracles": dict(ORACLES),
                    "timing": "z-score known at close t predicts the close-to-close return of t+2"},
        "symbols": list(names), "fields": ["open", "high", "low", "close", "volume", "factor"],
        "sessions": int(len(dates)), "date_range": {"start_inclusive": str(dates[0]), "end_exclusive": end_exclusive},
        "qlib_binary_tree_sha256": tree_sha256(qlib)[0],
        "universes": {universe_name: {
            "size": int(symbols), "membership_csv": "universe/membership.csv.gz",
            "membership_sha256": hashlib.sha256(membership.read_bytes()).hexdigest(),
            "instruments_file": f"instruments/{universe_name}.txt",
            "instruments_sha256": hashlib.sha256(instruments.read_bytes()).hexdigest(),
            "min_history_sessions": MIN_HISTORY_SESSIONS, "first_effective": str(first_member),
            "full_size_from": str(first_member), "unique_members": int(symbols)}},
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    say("Verifying the snapshot ...")
    source_identity(directory)
    say(f"Demonstration data ready: {directory}")
    return directory
