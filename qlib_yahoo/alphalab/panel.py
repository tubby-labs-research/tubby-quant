"""Content identities for Alpha Lab; verification never relies on timestamps."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qlib_yahoo.dataset import DatasetError, sha256

FIELDS = ("open", "high", "low", "close", "volume", "close_raw", "dividend", "cap", "vwap")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_identity(dataset: Path) -> dict:
    """Verify all consumed sources without allocating the price matrices."""
    from qlib_yahoo.dataset_integrity import verify_dataset
    from qlib_yahoo.panel import available_fields
    from qlib_yahoo.reference import dataset_groups
    from qlib_yahoo.universe import dataset_universes

    dataset = Path(dataset).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    verified = verify_dataset(dataset, manifest)
    dataset_groups(dataset, manifest)
    auxiliary = {}
    for name, record in dataset_universes(manifest).items():
        relative = record.get("membership_csv", "")
        path = dataset / relative
        if (not relative or not path.resolve().is_relative_to(dataset) or path.is_symlink()
                or not path.is_file() or sha256(path) != record.get("membership_sha256")):
            raise DatasetError(f"universe {name} membership file differs from the dataset manifest")
        auxiliary[relative] = sha256(path)
    qlib = dataset / "qlib_data"
    dates = (qlib / "calendars/day.txt").read_text(encoding="utf-8").split()
    symbols = [line.split("\t", 1)[0] for line in
               (qlib / "instruments/all.txt").read_text(encoding="utf-8").splitlines() if line]
    if len(set(symbols)) != len(symbols) or dates != sorted(set(dates)):
        raise DatasetError("Alpha Lab requires unique symbols and ascending unique sessions")
    identity = {"dataset": str(dataset), "verified": verified,
                "qlib_binary_tree_sha256": verified["qlib_binary_tree_sha256"], "manifest_sha256": digest(manifest),
                "auxiliary": auxiliary, "symbols": symbols, "dates": dates,
                "fields": [f for f in FIELDS if f in available_fields(manifest)]}
    identity["source_sha256"] = digest(identity)
    return identity
