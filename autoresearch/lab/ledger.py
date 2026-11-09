"""Append-only, hash-chained trial ledger: every candidate a campaign looks at is counted.

Large automated searches are only honest if the number of things tried is
known. Each line of ``<campaign>.ledger.jsonl`` is one trial event (a screen on
development data, a validation, a review, a holdout access) carrying the SHA-256
of the previous line, so deleting, reordering or editing an earlier entry breaks
the chain. The chain detects accidental corruption and silent edits; it is not
a signature against someone able to rewrite the whole file.

Single writer: the campaign holds the record store's OS lock while appending.
A crash can leave a torn final line without its newline; the next append trims
it and records a ``torn_tail_repaired`` entry, so the loss is itself visible.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import LedgerEntry, LedgerStatus

GENESIS = "0" * 64
STAGES = frozenset({"screen", "validate", "review", "holdout", "submitted", "torn_tail_repaired", "note"})


class LedgerError(ValueError):
    """The ledger chain or an entry is invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def entry_hash(entry: LedgerEntry) -> str:
    body = {key: value for key, value in entry.items() if key != "sha256"}
    return hashlib.sha256(_canonical(body)).hexdigest()


def _split(raw: bytes) -> tuple[list[bytes], bytes]:
    """Complete lines and a torn tail (bytes after the last newline)."""
    if not raw:
        return [], b""
    cut = raw.rfind(b"\n")
    if cut < 0:
        return [], raw
    return raw[:cut].split(b"\n"), raw[cut + 1:]


class TrialLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.is_symlink():
            raise LedgerError("ledger must not be a symlink")
        self._head: tuple[int, str, int] | None = None  # (entries, last sha256, verified byte size)

    def _read(self) -> tuple[list[LedgerEntry], bytes, int]:
        raw = self.path.read_bytes() if self.path.exists() else b""
        lines, torn = _split(raw)
        previous = GENESIS
        out: list[LedgerEntry] = []
        for index, line in enumerate(lines):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise LedgerError(f"ledger line {index + 1} is not JSON") from error
            if not isinstance(entry, dict) or entry.get("seq") != index or entry.get("prev") != previous:
                raise LedgerError(f"ledger chain broken at line {index + 1}")
            if entry.get("sha256") != entry_hash(entry):
                raise LedgerError(f"ledger entry {index + 1} does not match its hash")
            if entry.get("stage") not in STAGES:
                raise LedgerError(f"ledger entry {index + 1} has an unknown stage")
            previous = entry["sha256"]
            out.append(entry)
        return out, torn, len(raw) - len(torn)

    def entries(self) -> list[LedgerEntry]:
        """Every complete entry, verifying sequence numbers and the hash chain."""
        entries, torn, size = self._read()
        if not torn:
            self._head = (len(entries), entries[-1]["sha256"] if entries else GENESIS, size)
        return entries

    def verify(self) -> LedgerStatus:
        try:
            entries, torn, _size = self._read()
        except LedgerError as error:
            return {"ok": False, "problem": str(error)}
        counts: dict[str, int] = {}
        for entry in entries:
            counts[entry["stage"]] = counts.get(entry["stage"], 0) + 1
        return {"ok": True, "entries": len(entries), "head": entries[-1]["sha256"] if entries else GENESIS,
                "stages": counts, "torn_tail_bytes": len(torn)}

    def _write(self, line: bytes) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(descriptor, line)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def append(self, stage: str, **fields: Any) -> LedgerEntry:
        if stage not in STAGES:
            raise LedgerError(f"unknown ledger stage {stage!r}")
        size = self.path.stat().st_size if self.path.exists() else 0
        if self._head is None or self._head[2] != size:
            entries, torn, verified = self._read()
            if torn:
                with open(self.path, "r+b") as handle:
                    handle.truncate(verified)
                    handle.flush()
                    os.fsync(handle.fileno())
            self._head = (len(entries), entries[-1]["sha256"] if entries else GENESIS, verified)
            if torn:
                self.append("torn_tail_repaired", discarded_bytes=len(torn))
        seq, previous, size = self._head
        entry: LedgerEntry = {"seq": seq, "prev": previous, "stage": stage,
                              "created_at": datetime.now(timezone.utc).isoformat(), **fields}
        entry["sha256"] = entry_hash(entry)
        line = _canonical(entry) + b"\n"
        self._write(line)
        self._head = (seq + 1, entry["sha256"], size + len(line))
        return entry
