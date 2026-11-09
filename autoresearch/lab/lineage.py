"""Lineage registry: trial counts and holdout accesses that belong to the data, not to a record store.

A campaign's record store holds its own evidence, but two things belong to the
*data*: the multiple-testing count a review prices selection with, and the one-use
rule of the reserved holdout. Every campaign that validates on the same sealed
periods of the same dataset spends from the same budget, whichever
``research_root`` it writes to. This module keeps that accounting in one canonical
file beside the data::

    <lineage root>/<data identity>.json

keyed inside by the sealed-period lineage identifiers the campaign computes
(``validation_lineage`` and ``holdout_lineage``). The lineage root is
``<data root>/autoresearch/lineage`` for every dataset of this machine unless a
campaign names ``lineage_root`` explicitly (benchmarks and demonstrations do, under
their own root, so they stay isolated by construction); a campaign's report says
which registry it used and whether that is the canonical one.

Writes are atomic and serialized by the same single-writer OS lock pattern the
record store uses (a ``.writer.lock`` beside the registry files), with a bounded
wait because campaigns in different record stores may share a registry. Trials and
scores are keyed by experiment, accesses by candidate key, so a repeated write
after a crash changes nothing. The document carries ``created_at`` and a
``revision`` that every write increments, and campaigns snapshot what they saw into
their trial ledgers, so a replaced, shrunk or deleted registry is detectable
(:func:`verify_registry`).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import JSONObject, LedgerEntry, Lineage

REGISTRY_VERSION = 1


class RegistryMoved(ValueError):
    """The registry was written between a read and the write that depended on it; the caller re-reads and retries."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _holdout_bucket(document: JSONObject, lineage_id: str, lineage: Mapping[str, Any]) -> JSONObject:
    bucket: JSONObject = document["holdout"].setdefault(lineage_id, {"lineage": dict(lineage), "accesses": {},
                                                                     "retired": None, "campaigns": {},
                                                                     "batch": None, "batches": [], "exposed": None})
    bucket.setdefault("accesses", {})
    bucket.setdefault("campaigns", {})
    bucket.setdefault("batches", [])
    bucket.setdefault("exposed", None)
    return bucket


class LineageRegistry:
    """One dataset's lineage ledger: validation trials and holdout accesses across every campaign on it."""

    def __init__(self, root: str | Path, data_identity: str) -> None:
        if not data_identity or not all(ch.isalnum() or ch in "_-" for ch in data_identity):
            raise ValueError("data identity must be a safe identifier")
        self.root = Path(root)
        self.data_identity = data_identity
        self.path = self.root / (data_identity + ".json")
        self._mutex = threading.RLock()
        self._depth = 0

    # ---------------------------------------------------------------- locking
    @contextmanager
    def lock(self, timeout: float = 10.0) -> Iterator[LineageRegistry]:
        """Single-writer OS lock on the registry folder, waiting up to ``timeout`` seconds for another writer."""
        with self._mutex:
            if self._depth:
                self._depth += 1
                try:
                    yield self
                finally:
                    self._depth -= 1
                return
            self.root.mkdir(parents=True, exist_ok=True)
            lock_path = self.root / ".writer.lock"
            if lock_path.is_symlink():
                raise ValueError("lineage registry lock file must not be a symlink")
            handle = open(lock_path, "a+b")
            deadline = time.monotonic() + timeout
            try:
                while True:
                    try:
                        if sys.platform == "win32":
                            import msvcrt
                            handle.seek(0)
                            if not handle.read(1):
                                handle.write(b"0")
                                handle.flush()
                            handle.seek(0)
                            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError as exc:
                        if time.monotonic() >= deadline:
                            raise RuntimeError(f"lineage registry already has a writer: {self.root}") from exc
                        time.sleep(0.05)
            except BaseException:
                handle.close()
                raise
            self._depth = 1
            try:
                yield self
            finally:
                self._depth = 0
                handle.close()

    # ---------------------------------------------------------------- storage
    def read(self) -> JSONObject:
        """The registry document (an empty one, revision 0, when nothing was recorded yet)."""
        if self.path.is_symlink():
            raise ValueError("lineage registry must not be a symlink")
        if not self.path.exists():
            return {"version": REGISTRY_VERSION, "data_identity": self.data_identity, "revision": 0,
                    "validation": {}, "holdout": {}}
        document = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("data_identity") != self.data_identity:
            raise ValueError("lineage registry identity does not match its file")
        document.setdefault("validation", {})
        document.setdefault("holdout", {})
        document.setdefault("revision", 0)
        return document

    def _update(self, change: Callable[[JSONObject], None]) -> JSONObject:
        with self.lock():
            document = self.read()
            change(document)
            document["version"] = REGISTRY_VERSION
            document["revision"] = int(document.get("revision") or 0) + 1
            document["updated_at"] = _now()
            document.setdefault("created_at", document["updated_at"])
            _write_atomic(self.path, document)
            return document

    @staticmethod
    def _campaign(section: JSONObject, lineage_id: str, campaign_id: str, research_root: str,
                  lineage: Mapping[str, Any]) -> JSONObject:
        bucket = section.setdefault(lineage_id, {"lineage": dict(lineage), "campaigns": {}})
        campaigns: dict[str, JSONObject] = bucket.setdefault("campaigns", {})
        entry = campaigns.setdefault(campaign_id, {"research_root": research_root, "trials": {}, "scores": {}})
        entry["research_root"] = research_root
        entry["updated_at"] = _now()
        return entry

    # ------------------------------------------------------------ validation
    def record_trial(self, lineage_id: str, campaign_id: str, experiment_id: str, *, research_root: str,
                     lineage: Mapping[str, Any]) -> None:
        """A validation trial reserved by ``campaign_id`` (counted whether or not it finishes)."""
        def change(document: JSONObject) -> None:
            entry = self._campaign(document["validation"], lineage_id, campaign_id, research_root, lineage)
            entry["trials"].setdefault(experiment_id, _now())

        self._update(change)

    def record_score(self, lineage_id: str, campaign_id: str, experiment_id: str, score: float | None, *,
                     research_root: str, lineage: Mapping[str, Any]) -> None:
        """The published selection score of a trial (also records the trial if its reservation was lost)."""
        def change(document: JSONObject) -> None:
            entry = self._campaign(document["validation"], lineage_id, campaign_id, research_root, lineage)
            entry["trials"].setdefault(experiment_id, _now())
            if score is not None:
                entry["scores"][experiment_id] = score

        self._update(change)

    def validation_view(self, lineage_id: str) -> dict[str, JSONObject]:
        """Per campaign on this lineage: ``trials`` (count), ``scores`` (list) and ``research_root``."""
        campaigns = (self.read()["validation"].get(lineage_id) or {}).get("campaigns") or {}
        view: dict[str, JSONObject] = {}
        for campaign_id, entry in sorted(campaigns.items()):
            scores = entry.get("scores") or {}
            view[campaign_id] = {"trials": len(entry.get("trials") or {}),
                                 "scores": [scores[k] for k in sorted(scores)],
                                 "research_root": entry.get("research_root")}
        return view

    # --------------------------------------------------------------- holdout
    def record_access(self, lineage_id: str, campaign_id: str, key: str, sequence: int, *, research_root: str,
                      lineage: Mapping[str, Any], confirmatory: bool = True) -> None:
        """A holdout read of candidate ``key`` reserved by ``campaign_id`` (its ``sequence``-th across campaigns).

        A read outside a confirmatory batch (``confirmatory`` false: an exploratory read, a per-candidate
        final, a repeated read) **exposes** the lineage: the first such read is kept under ``exposed``,
        and a confirmatory batch declared after it is no longer an untouched test of this holdout.
        """
        def change(document: JSONObject) -> None:
            bucket = _holdout_bucket(document, lineage_id, lineage)
            bucket["campaigns"][campaign_id] = research_root
            current = bucket["accesses"].get(key) or {}
            if int(current.get("sequence", 0)) < sequence:
                bucket["accesses"][key] = {"sequence": sequence, "campaign_id": campaign_id, "at": _now()}
            if not confirmatory and not bucket.get("exposed"):
                bucket["exposed"] = {"at": _now(), "campaign_id": campaign_id, "key": key, "sequence": sequence}

        self._update(change)

    def retire_holdout(self, lineage_id: str, campaign_id: str, *, keys: list[str], level: float, method: str,
                       research_root: str, lineage: Mapping[str, Any], reason: str | None = None) -> None:
        """Close this lineage's holdout for confirmation; later reads by any campaign are exploratory.

        ``method`` says why: ``batch-holm`` after a confirmatory batch decided, ``voided-after-access``
        when a batch was abandoned after one of its keys had already been read (the holdout is then
        no longer an untouched test for the keys that remain).
        """
        def change(document: JSONObject) -> None:
            bucket = _holdout_bucket(document, lineage_id, lineage)
            bucket["campaigns"][campaign_id] = research_root
            if not bucket.get("retired"):
                bucket["retired"] = {"at": _now(), "campaign_id": campaign_id, "keys": list(keys),
                                     "level": level, "method": method, "reason": reason}

        self._update(change)

    def open_batch(self, lineage_id: str, campaign_id: str, batch_id: str, *, keys: list[str], level: float,
                   method: str, research_root: str, lineage: Mapping[str, Any], revision: int | None = None,
                   eligible_not_requested: list[str] | None = None, trials_at_declaration: int | None = None) -> None:
        """Declare a confirmatory holdout batch (its whole family and level) before its first access is made.

        ``revision`` is the registry revision the family's eligibility was computed from: the batch is
        declared only if nothing was written to the registry since, so the trial count a finalist was
        priced at and the batch that reads the holdout are one decision; otherwise :class:`RegistryMoved`
        and the caller recomputes. ``eligible_not_requested`` records the eligible promotions the caller
        left out of the family (``--key``), so a narrowed family is visible in the registry.
        ``trials_at_declaration`` is the lineage-wide validation count the finalists were priced at: a trial
        recorded after the declaration did not take part in selecting this family and does not enter it.
        """
        def change(document: JSONObject) -> None:
            if revision is not None and int(document.get("revision") or 0) != int(revision):
                raise RegistryMoved(f"the lineage registry moved from revision {revision} to {document.get('revision')} "
                                    "while the holdout batch was being declared")
            bucket = _holdout_bucket(document, lineage_id, lineage)
            if bucket.get("batch"):
                raise ValueError("a holdout batch is already open on this lineage: " + str(bucket["batch"].get("batch_id")))
            bucket["campaigns"][campaign_id] = research_root
            bucket["batch"] = {"batch_id": batch_id, "campaign_id": campaign_id, "keys": list(keys), "level": level,
                               "method": method, "opened_at": _now(),
                               "eligible_not_requested": list(eligible_not_requested or []),
                               "trials_at_declaration": trials_at_declaration}

        self._update(change)

    def void_batch(self, lineage_id: str, batch_id: str, campaign_id: str, *, reason: str, retire: bool,
                   keys: list[str], level: float, method: str, research_root: str, lineage: Mapping[str, Any],
                   declared: Mapping[str, Any] | None = None) -> None:
        """Abandon ``campaign_id``'s batch and, when ``retire``, retire the lineage's holdout, in one write.

        Closing and retiring in one registry transaction leaves no revision at which the batch is gone
        but the holdout still open for confirmation, so another record store cannot declare a batch in
        between. The batch may be known to the registry (its current ``batch``) or only to the record
        store that declared it (the registry's record was lost): in the latter case the void is written
        to the history anyway and says so, carrying the declaration the caller still has (``declared``:
        keys, level, opened_at, trials_at_declaration, eligible_not_requested) so the history stays whole.
        """
        def change(document: JSONObject) -> None:
            bucket = _holdout_bucket(document, lineage_id, lineage)
            bucket["campaigns"][campaign_id] = research_root
            batch = bucket.get("batch")
            if batch and batch.get("batch_id") == batch_id:
                if batch.get("campaign_id") != campaign_id:
                    raise ValueError("a holdout batch can only be voided by the campaign that declared it")
                bucket["batch"] = None
                bucket["batches"].append({**batch, "status": "voided", "closed_at": _now(), "reason": reason,
                                          "keys_closed": list(batch.get("keys") or [])})
            elif batch:
                raise ValueError("another holdout batch is open on this lineage: " + str(batch.get("batch_id")))
            else:
                known = dict(declared or {})
                bucket["batches"].append({"batch_id": batch_id, "campaign_id": campaign_id,
                                          "keys": list(known.get("keys") or []), "level": known.get("level"),
                                          "method": known.get("method", method), "opened_at": known.get("opened_at"),
                                          "trials_at_declaration": known.get("trials_at_declaration"),
                                          "eligible_not_requested": list(known.get("eligible_not_requested") or []),
                                          "status": "voided", "closed_at": _now(), "reason": reason, "keys_closed": list(keys),
                                          "note": "voided from the record store's declaration; the registry had lost the batch"})
            if retire and not bucket.get("retired"):
                bucket["retired"] = {"at": _now(), "campaign_id": campaign_id, "keys": list(keys),
                                     "level": level, "method": method, "reason": reason}

        self._update(change)

    def close_batch(self, lineage_id: str, batch_id: str, *, status: str, reason: str | None = None,
                    keys: list[str] | None = None) -> None:
        """Finish (``completed``) or abandon (``voided``) the open batch, keeping it in the lineage's history."""
        def change(document: JSONObject) -> None:
            bucket = _holdout_bucket(document, lineage_id, {})
            batch = bucket.get("batch")
            if batch and batch.get("batch_id") == batch_id:
                bucket["batch"] = None
                bucket["batches"].append({**batch, "status": status, "closed_at": _now(), "reason": reason,
                                          "keys_closed": list(keys if keys is not None else batch["keys"])})

        self._update(change)

    def holdout_view(self, lineage_id: str) -> JSONObject:
        """``accesses`` (key -> sequence, campaign_id, at), ``retired``, ``exposed``, the open ``batch`` and ``campaigns``."""
        bucket = self.read()["holdout"].get(lineage_id) or {}
        return {"accesses": dict(bucket.get("accesses") or {}), "retired": bucket.get("retired"),
                "exposed": bucket.get("exposed"),
                "batch": bucket.get("batch"), "batches": list(bucket.get("batches") or []),
                "campaigns": dict(bucket.get("campaigns") or {})}

    def seeded(self, validation_lineage: str) -> JSONObject | None:
        """How this lineage's entry was first seeded from a record store (campaign, store, counts, time), if it was."""
        bucket = self.read()["validation"].get(validation_lineage) or {}
        value = bucket.get("seeded")
        return dict(value) if isinstance(value, dict) else None

    def knows(self, validation_lineage: str, holdout_lineage: str) -> bool:
        """Whether the registry has any entry for this lineage (trials, accesses, exposure, a batch or a retirement)."""
        if not self.path.exists():
            return False
        document = self.read()
        holdout = document["holdout"].get(holdout_lineage) or {}
        return bool((document["validation"].get(validation_lineage) or {}).get("campaigns")
                    or holdout.get("accesses") or holdout.get("retired") or holdout.get("batch") or holdout.get("exposed"))

    def snapshot(self, validation_lineage: str, holdout_lineage: str) -> JSONObject:
        """What a campaign records in its ledger about this registry: identity, revision and the lineage's counts.

        Deterministic across machines and roots (no paths, no timestamps), so two runs of the same
        campaign record the same snapshot; ``exists`` and ``created_at`` are for the live check only.
        """
        document = self.read()
        campaigns = (document["validation"].get(validation_lineage) or {}).get("campaigns") or {}
        holdout = document["holdout"].get(holdout_lineage) or {}
        retired = holdout.get("retired") or None
        batch = holdout.get("batch") or None
        return {"registry": self.path.name, "revision": int(document.get("revision") or 0),
                "exists": self.path.exists(), "created_at": document.get("created_at"),
                "trials": sum(len(entry.get("trials") or {}) for entry in campaigns.values()),
                "campaigns": len(campaigns),
                "accesses": {key: int(item.get("sequence", 1)) for key, item in sorted((holdout.get("accesses") or {}).items())},
                "retired": None if not retired else {"campaign_id": retired.get("campaign_id"),
                                                     "keys": list(retired.get("keys") or []), "level": retired.get("level"),
                                                     "method": retired.get("method")},
                "batch": None if not batch else batch.get("batch_id"),
                "closed_batches": sorted(str(b.get("batch_id")) for b in holdout.get("batches") or [] if b.get("batch_id")),
                "exposed": bool(holdout.get("exposed")),
                "seeded": bool((document["validation"].get(validation_lineage) or {}).get("seeded"))}

    # ------------------------------------------------------------- migration
    def seed(self, *, validation_lineage: str, holdout_lineage: str, lineage: Lineage, campaign_id: str,
             research_root: str, trials: list[tuple[str, str, float | None]], accesses: list[tuple[str, str, int]],
             retired: JSONObject | None, exposed: JSONObject | None = None) -> None:
        """First contact of a record store with this registry: copy its accounting for a lineage in one write.

        ``trials`` are ``(campaign_id, experiment_id, score)``, ``accesses`` are ``(campaign_id, key,
        sequence)``; both are keyed as the live writers key them, so seeding is idempotent and a
        later live write for the same experiment or key changes nothing. The entry remembers that it
        was seeded (``seeded``: the campaign and record store that did it, the counts, the time), so a
        reader knows its baseline is that store's word rather than live accounting.
        """
        def change(document: JSONObject) -> None:
            marker = {"at": _now(), "campaign_id": campaign_id, "research_root": research_root,
                      "trials": len(trials), "accesses": len(accesses), "retired": bool(retired)}
            for seeded_campaign, experiment_id, score in trials:
                entry = self._campaign(document["validation"], validation_lineage, seeded_campaign, research_root,
                                       lineage["validation"])
                entry["trials"].setdefault(experiment_id, _now())
                if score is not None:
                    entry["scores"].setdefault(experiment_id, score)
            section = document["validation"].setdefault(validation_lineage, {"lineage": dict(lineage["validation"]),
                                                                              "campaigns": {}})
            section["seeded"] = section.get("seeded") or marker
            if accesses or retired or exposed:
                bucket = _holdout_bucket(document, holdout_lineage, lineage["holdout"])
                bucket["seeded"] = bucket.get("seeded") or marker
                if exposed and not bucket.get("exposed"):
                    bucket["exposed"] = dict(exposed)
                for seeded_campaign, key, sequence in accesses:
                    bucket["campaigns"][seeded_campaign] = research_root
                    current = bucket["accesses"].get(key) or {}
                    if int(current.get("sequence", 0)) < sequence:
                        bucket["accesses"][key] = {"sequence": sequence, "campaign_id": seeded_campaign, "at": _now()}
                if retired and not bucket.get("retired"):
                    bucket["retired"] = dict(retired)

        self._update(change)


# ------------------------------------------------------------------- integrity
SNAPSHOT_FIELDS = ("registry", "revision", "trials", "campaigns", "accesses", "retired", "batch", "closed_batches",
                   "exposed", "seeded")


def latest_snapshot(entries: Iterable[LedgerEntry]) -> JSONObject | None:
    """The last ``lineage_snapshot`` note of a trial ledger, or None when the campaign never met a registry."""
    found: JSONObject | None = None
    for entry in entries:
        if entry.get("stage") == "note" and entry.get("decision") == "lineage_snapshot":
            found = {field: entry.get(field) for field in SNAPSHOT_FIELDS}
    return found


def verify_registry(snapshot: JSONObject | None, current: JSONObject) -> JSONObject:
    """Compare a registry's current state with the latest snapshot a campaign's ledger recorded.

    ``registry_reset``: the file is missing or its revision fell below the snapshot's (a replaced
    file). ``registry_shrunk``: validation trials, a recorded holdout access or the retirement went
    away. Growth is normal and never a problem; a campaign that never snapshotted reports
    ``no_snapshot``.
    """
    if snapshot is None:
        return {"status": "no_snapshot", "problems": [], "snapshot": None, "current": current}
    problems: list[str] = []
    reset = shrunk = False
    if not current.get("exists"):
        reset = True
        problems.append("the registry file is missing")
    elif int(current.get("revision") or 0) < int(snapshot.get("revision") or 0):
        reset = True
        problems.append(f"registry revision {current.get('revision')} is below the snapshot's {snapshot.get('revision')}: "
                        "the file was replaced")
    if int(current.get("trials") or 0) < int(snapshot.get("trials") or 0):
        shrunk = True
        problems.append(f"validation trials fell from {snapshot.get('trials')} to {current.get('trials')}")
    accesses = current.get("accesses") or {}
    for key, sequence in sorted((snapshot.get("accesses") or {}).items()):
        if int(accesses.get(key, 0)) < int(sequence):
            shrunk = True
            problems.append(f"holdout access {key} (sequence {sequence}) is missing or lower ({accesses.get(key, 0)})")
    if snapshot.get("retired") and not current.get("retired"):
        shrunk = True
        problems.append("the holdout retirement is gone")
    if snapshot.get("exposed") and not current.get("exposed"):
        shrunk = True
        problems.append("the record of the holdout's exposure is gone")
    opened = snapshot.get("batch")
    if opened and current.get("batch") != opened and opened not in (current.get("closed_batches") or []):
        shrunk = True
        problems.append(f"the open holdout batch {opened} is gone without being completed or voided")
    status = "registry_reset" if reset else "registry_shrunk" if shrunk else "ok"
    return {"status": status, "problems": problems, "snapshot": snapshot, "current": current}
