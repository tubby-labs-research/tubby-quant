"""Prepare artificial Alpha Lab examples for screenshots; never download data.

Requires a new output directory outside the source checkout. Results are ordinary
Alpha Lab library entries, not edited metrics or fabricated chart images.
The 200 symbols are 25 replicas of each of eight artificial paths: the larger
universe only fits the console's TOP200 selector and adds no diversity.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def create_gallery_dataset(output: Path) -> Path:
    """Expand the seed-7403 demo without changing or inventing price paths."""
    from autoresearch.backend import create_demo_dataset
    from qlib_yahoo.alphalab.panel import source_identity
    from qlib_yahoo.dataset import tree_sha256

    template = create_demo_dataset(output / "template")
    dataset = output / "datasets" / "SYNTHETIC-DEMO-not-market-data"
    qlib = dataset / "qlib_data"
    (qlib / "calendars").mkdir(parents=True)
    (qlib / "instruments").mkdir()
    calendar = template / "qlib_data" / "calendars" / "day.txt"
    days = calendar.read_text(encoding="utf-8").split()
    shutil.copyfile(calendar, qlib / "calendars" / "day.txt")
    symbols = [f"SYN{i:03d}" for i in range(200)]
    replicas = {symbol: f"SYN{i % 8:02d}" for i, symbol in enumerate(symbols)}
    for symbol, original in replicas.items():
        source = template / "qlib_data" / "features" / original.lower()
        destination = qlib / "features" / symbol.lower()
        destination.mkdir(parents=True)
        for field in ("open", "high", "low", "close", "volume"):
            shutil.copyfile(source / f"{field}.day.bin", destination / f"{field}.day.bin")
    (qlib / "instruments" / "all.txt").write_text(
        "".join(f"{symbol}\t{days[0]}\t{days[-1]}\n" for symbol in symbols),
        encoding="utf-8")
    membership = dataset / "universe" / "membership.csv.gz"
    membership.parent.mkdir()
    rows = "instrument,effective_start,effective_end,rank\n" + "".join(
        f"{symbol},{days[0]},{days[-1]},{i + 1}\n" for i, symbol in enumerate(symbols))
    membership.write_bytes(gzip.compress(rows.encode("utf-8"), mtime=0))
    manifest = {
        "status": "ready", "provider": "synthetic-readme-replicas-v1",
        "note": "SYNTHETIC ONLY: 200 symbols are 25 replicas of each of eight "
                "seed-7403 artificial paths. No added diversity. Static TOP200 membership "
                "and rank are fixture identifiers, not liquidity rankings. "
                "Artificial data only—not evidence of investment performance.",
        "symbols": symbols, "sessions": len(days),
        "date_range": {"start_inclusive": days[0],
                       "end_exclusive": str(date.fromisoformat(days[-1]) + timedelta(days=1))},
        "fields": ["open", "high", "low", "close", "volume"],
        "replication": {"seed": 7403, "original_paths": 8, "replicas_per_path": 25,
                        "adds_diversity": False, "source_symbols": replicas},
        "qlib_binary_tree_sha256": tree_sha256(qlib)[0],
        "universes": {"top200": {"size": 200, "unique_members": 200,
                                 "first_effective": days[0], "full_size_from": days[0],
                                 "last_effective": days[-1],
                                 "membership_csv": "universe/membership.csv.gz",
                                 "membership_sha256": hashlib.sha256(membership.read_bytes()).hexdigest()}},
    }
    (dataset / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    source_identity(dataset)
    return dataset


def prepare(output: Path) -> dict:
    output = output.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Use a new data directory outside the source checkout.")
    # Do not reuse, mix with, or overwrite an existing user's data root.
    output.mkdir(parents=True, exist_ok=False)

    from qlib_yahoo.alphalab.batch import run_batch
    from qlib_yahoo.alphalab.settings import Settings
    from qlib_yahoo.webapp.service import ConsoleService

    dataset = create_gallery_dataset(output)
    settings = Settings(universe="TOP200", delay=1, decay=4,
                        neutralization="MARKET", truncation=0.01,
                        cost_bps=5.0, warmup=252,
                        start="2024-01-02", end="2024-12-31").validate()
    expressions = ["rank(ts_delta(close, 5))", "-rank(ts_delta(close, 5))"]
    results = run_batch(dataset, expressions, settings, data_root=output,
                        library=output / "alphalab" / "library.sqlite",
                        workers=1, worker_ceiling=1)
    service = ConsoleService(output, settings_path=output / "console-settings.json")
    evidence = []
    for result in results:
        if result.get("error"):
            raise RuntimeError(result["error"])
        pnl = result["summary"]["pnl"]
        label = "SYNTHETIC / " + ("positive result" if pnl > 0 else "negative result")
        service.annotate({"kind": "alphas", "key": result["id"], "label": label})
        evidence.append({"id": result["id"], "expression": result["expression"],
                         "summary": result["summary"]})
    if not (any(row["summary"]["pnl"] > 0 for row in evidence)
            and any(row["summary"]["pnl"] < 0 for row in evidence)):
        raise RuntimeError("Demo no longer produces contrasting outcomes; inspect before capturing.")
    receipt = {"synthetic": True, "warning": "Artificial fixture; no investment evidence.",
               "replication": "200 symbols: 25 replicas of each of eight artificial paths; no added diversity.",
               "dataset": results[0]["dataset"],
               "settings": asdict(settings), "results": evidence}
    service.make_default(str(dataset))
    (output / "screenshot-demo.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
