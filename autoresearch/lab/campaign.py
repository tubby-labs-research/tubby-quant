"""Domain-neutral, checkpointed search campaign with a sealed, budgeted cascade.

One :meth:`LabCampaign.tick` is one round::

    propose (domain generators, agent inbox) ─► dedupe by canonical key
      ─► screen on development data ─► population of the fittest
      ─► shortlist: fitness, domain gates, diversity against accepted/validated
      ─► validate (one counted validation trial each) ─► review ─► promote

Every screen, validation, review and holdout access is appended to the
hash-chained trial ledger, so the number of things tried is never lost;
the domain's review receives that count to price selection (for alphas, the
deflated Sharpe ratio). Validation trials are reserved in the checkpoint before
they run: an interrupted evaluation still counts. Budgets are operational and
may be raised to continue a campaign; policy thresholds are part of the
campaign identity, so changing a gate starts a new campaign instead of
rescuing an old candidate. The reserved holdout is scored only by the explicit
:meth:`finalize`, once per candidate, with reuse refused or labelled.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from autoresearch.records import ResearchStore, stable_id

from .domain import REGISTRY, Domain, load_domain
from .ledger import LedgerError, TrialLedger
from .lineage import SNAPSHOT_FIELDS, LineageRegistry, RegistryMoved, latest_snapshot, verify_registry
from .selection import holm, probabilistic_sharpe
from .selection import pearson as _correlation
from .types import (
    AcceptedEntry,
    CampaignConfig,
    CampaignLimits,
    CampaignRecord,
    CampaignStatus,
    Candidate,
    Checkpoint,
    Decision,
    EvaluationResult,
    ExperimentRecord,
    FinalOutcome,
    FinalRecord,
    HypothesisRecord,
    JSONObject,
    Lineage,
    LineageEvidence,
    Manifest,
    PopulationEntry,
    ProposalContext,
    Review,
    ReviewRecord,
    RoundOutcome,
    RoundSummary,
    ScreenOutcome,
    SelectionSettings,
    Submission,
    TrialContext,
    ValidatedEntry,
    Verdict,
)

LAB_VERSION = 1
DEFAULT_LIMITS: CampaignLimits = {
    "max_rounds": 25,
    "proposals_per_round": 48,
    "max_screen_trials": 2000,
    "validations_per_round": 3,
    "max_validation_trials": 60,
    "max_promotions": 20,
    "population_size": 40,
}
DEFAULT_SELECTION: SelectionSettings = {"max_correlation": None}  # None: the domain's default_max_correlation
#: Policy keys the harness reads itself. They enter the campaign identity like every policy key but are
#: not passed to the domain: how the reserved holdout confirms, and at which familywise level
#: (``None``: the campaign's own level, ``alpha_level`` or ``1 - min_dsr``, else 0.05).
HARNESS_POLICY: JSONObject = {"holdout_confirmation": "batch", "holdout_level": None}
HOLDOUT_CONFIRMATION_MODES = ("batch", "per-candidate")
MAX_INBOX_BYTES = 1_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
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


def append_inbox(path: Path, candidate: Candidate, submitted_by: str, *, key: str | None = None) -> None:
    """Append one submission line; the campaign reads, dedupes and screens it on its next round."""
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be an object")
    line = json.dumps({"candidate": candidate, "submitted_by": str(submitted_by)[:80], "key": key,
                       "submitted_at": utc_now()}, sort_keys=True, allow_nan=False) + "\n"
    if len(line) > 20_000:
        raise ValueError("submitted candidate is too large")
    path = Path(path)
    if path.is_symlink():
        raise ValueError("inbox must not be a symlink")
    if path.exists() and path.stat().st_size + len(line) > MAX_INBOX_BYTES:
        raise ValueError("campaign inbox is full; run the campaign before submitting more")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(descriptor, line.encode())
    finally:
        os.close(descriptor)


def default_lab_root(domain: str) -> Path:
    from qlib_yahoo import default_data_root
    return Path(default_data_root()) / "autoresearch" / "lab" / domain


def validate_lab_config(value: Mapping[str, Any]) -> CampaignConfig:
    if not isinstance(value, dict):
        raise ValueError("campaign configuration must be an object")
    config: dict[str, Any] = copy.deepcopy(value)
    if not isinstance(config.get("name"), str) or not config["name"].strip():
        raise ValueError("campaign name is required")
    if config.get("domain") not in REGISTRY:
        raise ValueError("campaign domain must be one of: " + ", ".join(sorted(REGISTRY)))
    if not isinstance(config.get("settings"), dict):
        raise ValueError("settings must be an object with the domain configuration")
    for section in ("policy", "selection"):
        if not isinstance(config.get(section, {}), dict):
            raise ValueError(f"{section} must be an object")
    seed = config.setdefault("seed", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    limits: dict[str, Any] = {**DEFAULT_LIMITS, **config.get("limits", {})}
    if set(limits) != set(DEFAULT_LIMITS):
        raise ValueError("unknown campaign limit: " + ", ".join(sorted(set(limits) - set(DEFAULT_LIMITS))))
    for key, number in limits.items():
        if type(number) is not int or number < 1:
            raise ValueError(f"{key} must be a positive integer")
    config["limits"] = limits
    selection: dict[str, Any] = {**DEFAULT_SELECTION, **config.get("selection", {})}
    if set(selection) != set(DEFAULT_SELECTION):
        raise ValueError("unknown selection setting")
    threshold = selection["max_correlation"]
    if threshold is not None and not (isinstance(threshold, (int, float)) and 0 < threshold <= 1):
        raise ValueError("selection.max_correlation must be in (0, 1]")
    config["selection"] = selection
    policy = config.get("policy", {})
    if policy.get("holdout_confirmation", HARNESS_POLICY["holdout_confirmation"]) not in HOLDOUT_CONFIRMATION_MODES:
        raise ValueError("policy.holdout_confirmation must be one of: " + ", ".join(HOLDOUT_CONFIRMATION_MODES))
    level = policy.get("holdout_level")
    if level is not None and not (isinstance(level, (int, float)) and not isinstance(level, bool) and 0 < level < 1):
        raise ValueError("policy.holdout_level must be in (0, 1)")
    if "non_confirmatory" in config and not isinstance(config["non_confirmatory"], bool):
        raise ValueError("non_confirmatory must be true or false")
    if "lineage_root" in config and (not isinstance(config["lineage_root"], str) or not config["lineage_root"].strip()):
        raise ValueError("lineage_root must be a path")

    def inspect(obj: object) -> None:
        if isinstance(obj, dict):
            for key, item in obj.items():
                if str(key).lower() in {"api_key", "token", "password", "secret", "authorization"}:
                    raise ValueError("put credentials in environment variables, not campaign files")
                inspect(item)
        elif isinstance(obj, list):
            for item in obj:
                inspect(item)

    inspect(config)
    # Every key the harness reads was checked above; unknown top-level keys are kept
    # on purpose, because they enter the campaign identity.
    return cast(CampaignConfig, config)


def _absolute(value: str, base: Path) -> str:
    candidate = Path(value).expanduser()
    return str(candidate if candidate.is_absolute() else base / candidate)


def load_lab_config(path: Path) -> CampaignConfig:
    path = Path(path).resolve()
    config = validate_lab_config(json.loads(path.read_text(encoding="utf-8")))
    if config.get("research_root"):
        config["research_root"] = _absolute(config["research_root"], path.parent)
    if config.get("runtime_root"):
        config["runtime_root"] = _absolute(config["runtime_root"], path.parent)
    if config.get("lineage_root"):
        config["lineage_root"] = _absolute(config["lineage_root"], path.parent)
    settings = config["settings"]
    for key in ("dataset", "table"):
        if isinstance(settings.get(key), str) and settings[key] and not settings[key].startswith("builtin:"):
            settings[key] = _absolute(settings[key], path.parent)
    return config


def _code_hashes(packages: Iterable[str]) -> dict[str, str]:
    """SHA-256 of the harness and the domain packages, keyed by repository-relative path."""
    import importlib

    base = Path(__file__).resolve().parents[2]
    roots = [Path(__file__).resolve().parent]
    for name in packages:
        source = importlib.import_module(name).__file__
        if source is None:
            raise ValueError(f"cannot hash {name}: the package has no source file")
        roots.append(Path(source).resolve().parent)
    hashes: dict[str, str] = {}
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                label = path.relative_to(base).as_posix()
            except ValueError:
                label = path.name
            hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def _round(value: Any, digits: int = 6) -> Any:
    """Round floats (recursively through dicts and lists) and turn non-finite ones into None."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(f"{value:.{digits}g}")
    if isinstance(value, dict):
        return {key: _round(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item, digits) for item in value]
    return value


def _by_fitness(entry: PopulationEntry) -> tuple[float, str]:
    """Population order: fittest first, unknown fitness last, ties by key."""
    fitness = entry["fitness"]
    return (-(fitness if fitness is not None else -math.inf), entry["key"])


def default_lineage_root() -> Path:
    """The canonical lineage registry folder of this machine: ``<data root>/autoresearch/lineage``.

    Every dataset registers here unless the campaign names ``lineage_root`` explicitly, so a
    dataset copied to another folder (same content, same data identity) still meets the same
    registry. Benchmarks and demonstrations name a registry under their own root explicitly, so
    they stay isolated by construction, and their finals say so (``registry_canonical: false``).
    """
    from qlib_yahoo import default_data_root

    return Path(default_data_root()).expanduser().resolve() / "autoresearch" / "lineage"


def data_identity(domain: str, lineage: Lineage) -> str:
    """The registry file for a dataset: its data identity (content hash) under the domain, independent of periods."""
    validation = lineage["validation"]
    data = validation.get("data") or validation.get("content") or validation
    return stable_id("data", {"domain": domain, "data": data})


def holdout_pvalue(result: EvaluationResult) -> tuple[float | None, str]:
    """One-sided p-value that a final's holdout evidence is null, from what the domain recorded.

    A tabular task reports its paired-bootstrap p-value; a daily signal reports its holdout
    profile (per-day Sharpe, sessions, skewness, kurtosis), priced with the probabilistic
    Sharpe ratio against zero over ``T`` holdout sessions; a bare annualized Sharpe over
    ``days`` sessions falls back to the Gaussian probabilistic Sharpe ratio.
    """
    if result.get("status") != "ok":
        return None, "unavailable"
    metrics = (result.get("metrics") or {}).get("holdout") or {}
    p_value = metrics.get("p_value")
    if isinstance(p_value, (int, float)) and not isinstance(p_value, bool) and math.isfinite(p_value):
        return min(max(float(p_value), 0.0), 1.0), str(metrics.get("p_value_method") or "bootstrap")
    profile = result.get("holdout_profile") or {}
    sharpe, days = profile.get("daily_sharpe"), profile.get("days")
    if isinstance(sharpe, (int, float)) and math.isfinite(sharpe) and isinstance(days, (int, float)) and days >= 2:
        return 1.0 - probabilistic_sharpe(float(sharpe), 0.0, int(days), float(profile.get("skew") or 0.0),
                                          float(profile.get("kurtosis") or 3.0)), "psr"
    annual, days = metrics.get("sharpe"), metrics.get("days")
    if isinstance(annual, (int, float)) and math.isfinite(annual) and isinstance(days, (int, float)) and days >= 2:
        return 1.0 - probabilistic_sharpe(float(annual) / math.sqrt(252.0), 0.0, int(days)), "psr-gaussian"
    return None, "unavailable"


def statistic_note(statistic: str) -> str:
    """How a holdout p-value was obtained and what it assumes about dependence, for the reason text and the console."""
    return {"psr": "PSR; sessions treated as independent",
            "psr-gaussian": "Gaussian PSR; sessions treated as independent",
            "iid-bootstrap": "paired bootstrap; rows resampled independently",
            "moving-block-bootstrap": "moving-block bootstrap on ordered rows",
            "bootstrap": "paired bootstrap",
            "unavailable": "no usable statistic; entered as p = 1"}.get(statistic, statistic)


@dataclass
class LedgerIndex:
    """What the ledger already recorded, for bringing a checkpoint back in line after a crash."""

    screen_keys: set[str] = field(default_factory=set)
    screen: int = 0
    screen_errors: int = 0
    validate: set[str] = field(default_factory=set)
    review: set[str] = field(default_factory=set)


class LabCampaign:
    """A single-writer campaign for one domain; see the module docstring."""

    def __init__(self, config: Mapping[str, Any], *, domain: Domain | None = None) -> None:
        self.config: CampaignConfig = validate_lab_config(config)
        name = self.config["domain"]
        root = default_lab_root(name)
        self.research_root = Path(self.config.get("research_root") or root / "records")
        self.runtime = Path(self.config.get("runtime_root") or root / "runtime")
        self.runtime.mkdir(parents=True, exist_ok=True)
        policy: JSONObject = dict(self.config.get("policy", {}))
        self.harness_policy: JSONObject = {**HARNESS_POLICY, **{k: v for k, v in policy.items() if k in HARNESS_POLICY}}
        self.domain_policy: JSONObject = {k: v for k, v in policy.items() if k not in HARNESS_POLICY}
        if domain is None:
            domain = load_domain(name)(self.config["settings"], self.runtime / "domain", policy=self.domain_policy)
        self.domain: Domain = domain
        public: JSONObject = {key: value for key, value in self.config.items()
                              if key not in {"research_root", "runtime_root", "lineage_root", "settings", "limits"}}
        public["settings"] = self.domain.public_config()
        self.manifest: Manifest = {"lab_version": LAB_VERSION, "domain": name, "config": public,
                                   "identity": self.domain.identity(),
                                   "code": _code_hashes(getattr(self.domain, "code_packages", ()))}
        self.campaign_id = stable_id("campaign", self.manifest)
        lineage = self.domain.lineage()
        self.lineage: Lineage = lineage
        self.validation_lineage = stable_id("lineage", {"domain": name, "validation": lineage["validation"]})
        self.holdout_lineage = stable_id("lineage", {"domain": name, "holdout": lineage["holdout"]})
        self.lineage_root = Path(self.config.get("lineage_root") or default_lineage_root())
        self.registry_canonical = self.lineage_root.resolve() == default_lineage_root()
        self.registry = LineageRegistry(self.lineage_root, data_identity(name, lineage))
        self.store = ResearchStore(self.research_root)
        self.ledger = TrialLedger(self.research_root / "ledgers" / (self.campaign_id + ".jsonl"))
        self.state_path = self.runtime / (self.campaign_id + ".json")
        self.inbox_path = self.runtime / (self.campaign_id + ".inbox.jsonl")
        self.limits: CampaignLimits = self.config["limits"]
        self.state: Checkpoint | None = None

    @property
    def checkpoint(self) -> Checkpoint:
        """The loaded checkpoint; :meth:`tick`, :meth:`finalize` and :meth:`status` load it first."""
        state = self.state
        if state is None:
            raise RuntimeError("the campaign checkpoint is not loaded yet; call status() first")
        return state

    # ------------------------------------------------------------- persistence
    def _fresh(self, created_at: str | None = None) -> Checkpoint:
        return {"campaign_id": self.campaign_id, "domain": self.config["domain"], "round": 0,
                "screen_trials": 0, "screen_errors": 0, "validation_trials": 0, "promotions": 0,
                "seen": [], "population": [], "validated": {}, "accepted": [],
                "validation_scores": [], "pending": None, "round_open": None,
                "holdout_access": {}, "holdout_accessed": False, "generators": {},
                "status": "ready", "created_at": created_at or utc_now()}

    def _load(self) -> None:
        if self.state_path.exists():
            loaded: Checkpoint = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.state = loaded
            if loaded.get("campaign_id") != self.campaign_id:
                raise ValueError("campaign checkpoint identity mismatch")
        else:
            rebuilt = self._rebuild()
            loaded = rebuilt if rebuilt is not None else self._fresh()
            self.state = loaded
            if rebuilt is not None:
                self._save()
                counts = {"round": loaded["round"], "screen_trials": loaded["screen_trials"],
                          "validation_trials": loaded["validation_trials"], "promotions": loaded["promotions"],
                          "holdout_accesses": len(loaded["holdout_access"])}
                self.ledger.append("note", decision="checkpoint_rebuilt", **counts)
                self._event("checkpoint_rebuilt", **counts)
        if self.store.get("campaigns", self.campaign_id) is None:
            record: CampaignRecord = {"id": self.campaign_id, "name": self.config["name"],
                                      **self.manifest, "created_at": loaded["created_at"]}
            self.store.put("campaigns", record)
        self._seed_registry()

    def _seed_registry(self) -> None:
        """First contact of this record store with the lineage registry: carry the store's accounting over.

        Campaigns that ran before the registry existed have their trials, scores, holdout accesses
        and any retirement in the record store only. When the registry has no entry at all for this
        lineage but the store has, those are copied once, in one write, keyed by experiment and
        candidate so the copy is idempotent, and the seeding is recorded in the ledger and the store
        together with a snapshot of the registry. A registry that already has entries for the
        lineage is never changed here: a store that then disagrees with it is reported as such. And
        a registry this campaign has already snapshotted is never seeded again: its disappearance
        is recorded as ``registry_reset`` and reported by :meth:`verify`, not papered over.
        """
        if self.registry.knows(self.validation_lineage, self.holdout_lineage):
            return
        if self.ledger.path.exists():
            status = self.ledger.verify()
            if not status.get("ok"):
                # A store whose own trial ledger does not verify cannot be the registry's baseline: the registry
                # stays empty, the store disagrees with it, and a confirmatory finalize is refused until that is
                # resolved. The refusal is recorded in the store (the ledger cannot be appended to).
                if not any(e.get("kind") == "lineage_seed_refused" and e.get("campaign_id") == self.campaign_id
                           for e in self.store.list("events")):
                    self._event("lineage_seed_refused", reason=str(status.get("problem")))
                return
        entries = self.ledger.entries() if self.ledger.path.exists() else []
        if latest_snapshot(entries) is not None:
            marks = [e for e in entries if e.get("stage") == "note"
                     and e.get("decision") in ("lineage_snapshot", "registry_reset")]
            if marks and marks[-1].get("decision") != "registry_reset":
                current = self.registry.snapshot(self.validation_lineage, self.holdout_lineage)
                self.ledger.append("note", decision="registry_reset", registry=current["registry"],
                                   revision=current["revision"], exists=current["exists"])
                self._event("registry_reset", registry=str(self.registry.path), revision=current["revision"],
                            exists=current["exists"])
            return
        experiments = [e for e in self.store.list("experiments") if e.get("validation_lineage") == self.validation_lineage]
        events = [e for e in self.store.list("events") if e.get("holdout_lineage") == self.holdout_lineage
                  and e.get("kind") in ("holdout_access", "holdout_retired")]
        if not experiments and not events:
            return
        trials: list[tuple[str, str, float | None]] = []
        for experiment in experiments:
            result = experiment.get("result") or {}
            score: float | None = None
            if result.get("status") == "ok":
                try:
                    score = self.domain.selection_score(result)
                except Exception:
                    score = None
            trials.append((str(experiment["campaign_id"]), str(experiment["id"]),
                           _round(score) if score is not None and math.isfinite(score) else None))
        accesses = [(str(e["campaign_id"]), str(e["key"]), int(e.get("access_sequence", 1)))
                    for e in events if e["kind"] == "holdout_access"]
        retired: JSONObject | None = None
        exposed: JSONObject | None = None
        for event in sorted(events, key=lambda e: str(e.get("created_at", ""))):
            if event["kind"] == "holdout_retired" and retired is None:
                retired = {"at": event.get("created_at"), "campaign_id": event["campaign_id"],
                           "keys": list(event.get("keys") or []), "level": event.get("level"),
                           "method": event.get("method") or "batch-holm"}
            elif event["kind"] == "holdout_access" and exposed is None:
                confirmatory = event.get("confirmatory")
                if confirmatory is None:
                    confirmatory = event.get("batch_id") is not None
                if not confirmatory:
                    exposed = {"at": event.get("created_at"), "campaign_id": event["campaign_id"], "key": event["key"],
                               "sequence": int(event.get("access_sequence", 1))}
        self.registry.seed(validation_lineage=self.validation_lineage, holdout_lineage=self.holdout_lineage,
                           lineage=self.lineage, campaign_id=self.campaign_id, research_root=str(self.research_root),
                           trials=trials, accesses=accesses, retired=retired, exposed=exposed)
        counts = {"trials": len(trials), "holdout_accesses": len(accesses), "retired": retired is not None,
                  "campaigns": len({cid for cid, _e, _s in trials} | {cid for cid, _k, _q in accesses})}
        self.ledger.append("note", decision="lineage_seeded", **counts)
        self._event("lineage_seeded", **counts)
        self._snapshot("lineage_seeded")

    def _snapshot(self, reason: str) -> None:
        """Record in the ledger what this campaign saw in the registry (deterministic: no paths, no timestamps)."""
        current = self.registry.snapshot(self.validation_lineage, self.holdout_lineage)
        fields = {key: current[key] for key in SNAPSHOT_FIELDS}
        self.ledger.append("note", decision="lineage_snapshot", reason=reason, **fields)
        self._event("lineage_snapshot", reason=reason, path=str(self.registry.path), **fields)

    def _rebuild(self) -> Checkpoint | None:
        """A checkpoint rebuilt from the trial ledger and the record store when the runtime copy is missing.

        Everything that was counted comes back: screens (and the keys seen), validation trials
        and their scores, review decisions, promotions with their candidates (restored from the
        hypothesis records through :meth:`Domain.restore_candidate`) and holdout accesses. The
        screened population is not recorded anywhere, so the search continues from an empty
        population without re-screening what was seen; promoted candidates lose their diversity
        signature, so later diversity checks cannot compare against them. A trial the lineage
        registry reserved but the ledger never saw still counts.
        """
        cid = self.campaign_id
        entries = self.ledger.entries() if self.ledger.path.exists() else []
        hypotheses = {h["id"]: h for h in self.store.list("hypotheses") if h.get("campaign_id") == cid}
        if not entries and not hypotheses:
            return None
        campaign = self.store.get("campaigns", cid) or {}
        state = self._fresh(campaign.get("created_at"))
        seen: list[str] = []
        generators = state["generators"]
        scores: dict[str, float | None] = {}

        def count(generator: str | None, stage: str | None) -> None:
            bucket = generators.setdefault(generator or "unknown", {})
            bucket[stage or "unknown"] = bucket.get(stage or "unknown", 0) + 1

        for entry in entries:
            stage = entry["stage"]
            if stage == "screen":
                state["screen_trials"] += 1
                if entry.get("status") != "ok":
                    state["screen_errors"] += 1
                if entry.get("key") and entry["key"] not in seen:
                    seen.append(entry["key"])
                state["round"] = max(state["round"], int(entry.get("round") or 0))
                count(entry.get("generator"), "screened")
            elif stage == "validate":
                state["validation_trials"] += 1
                scores[entry.get("experiment_id", "")] = entry.get("score")
                if entry.get("score") is not None:
                    state["validation_scores"].append(entry["score"])
            elif stage == "review":
                key, hid, eid = entry["key"], entry.get("hypothesis_id", ""), entry.get("experiment_id", "")
                hypothesis = hypotheses.get(hid, {})
                generator = str(hypothesis.get("generator") or "unknown")
                decision = str(entry.get("decision") or "unknown")
                count(generator, "validated")
                count(generator, decision)
                state["validated"][key] = {"hypothesis_id": hid, "experiment_id": eid, "decision": decision,
                                           "score": scores.get(eid), "signature": None, "diversity_group": "atomic"}
                if decision == "promote" and key not in {a["key"] for a in state["accepted"]}:
                    review_id = stable_id("review", {"experiment": eid, "policy": self.config.get("policy", {})})
                    review = self.store.get("reviews", review_id) or {}
                    state["accepted"].append({"key": key, "hypothesis_id": hid, "experiment_id": eid,
                                              "review_id": review_id,
                                              "candidate": self.domain.restore_candidate(hypothesis),
                                              "score": scores.get(eid), "signature": None, "generator": generator,
                                              "diversity_group": "atomic", "checks": review.get("checks") or {}})
            elif stage == "holdout":
                key = entry["key"]
                state["holdout_access"][key] = max(state["holdout_access"].get(key, 0), int(entry.get("access_sequence", 1)))
                state["holdout_accessed"] = True
            elif stage == "note" and entry.get("decision") == "redundant" and entry.get("key"):
                redundant: ValidatedEntry = {"decision": "redundant", "score": None}
                if isinstance(entry.get("max_correlation"), (int, float)):
                    redundant["max_correlation"] = float(entry["max_correlation"])
                state["validated"].setdefault(entry["key"], redundant)
        state["seen"] = seen
        state["promotions"] = len(state["accepted"])
        own = self.registry.validation_view(self.validation_lineage).get(cid) or {}
        state["validation_trials"] = max(state["validation_trials"], int(own.get("trials") or 0))
        state["status"] = "running" if entries else "ready"
        return state

    def _save(self) -> None:
        atomic_json(self.state_path, self.state)

    def _event(self, kind: str, **details: object) -> None:
        value: dict[str, object] = {"kind": kind, "campaign_id": self.campaign_id, "domain": self.config["domain"],
                                    "created_at": utc_now(), **details}
        value["id"] = stable_id("event", {**value, "nonce": uuid.uuid4().hex})
        self.store.put("events", _round(value))

    def _count(self, generator: str, stage: str) -> None:
        bucket = self.checkpoint["generators"].setdefault(generator or "unknown", {})
        bucket[stage] = bucket.get(stage, 0) + 1

    # ----------------------------------------------------------------- budgets
    def _budget_reason(self) -> str | None:
        s, limits = self.checkpoint, self.limits
        if s["round"] >= limits["max_rounds"]:
            return "round budget reached"
        if s["validation_trials"] >= limits["max_validation_trials"]:
            return "validation-trial budget reached"
        if s["screen_trials"] >= limits["max_screen_trials"]:
            return "screen-trial budget reached"
        if s["promotions"] >= limits["max_promotions"]:
            return "promotion budget reached"
        return None

    # --------------------------------------------------------------- lineage
    def _family(self) -> LineageEvidence:
        """Evidence shared with other campaigns on the same data: the record store and the lineage registry.

        Validation trials and holdout accesses belong to the data, not to one
        campaign: a campaign started by nudging a gate, editing code or changing
        a seed tests the same validation period and reads the same holdout. They
        are counted across every campaign with the same lineage, in this record
        store and in the registry beside the data, which another ``research_root``
        cannot reset. Where the two disagree the larger count is used and the
        disagreement is reported; a confirmatory finalization refuses it.

        The holdout's retirement and its exposure (a read outside a confirmatory
        batch) are monotone facts, so this record store's own events count as
        evidence of them even when the registry has lost them; an open batch is
        taken from the store only when the registry has no entry for the holdout
        lineage at all. ``revision`` is read before anything else, so a registry
        write that lands while this evidence is assembled is caught when the
        batch is declared against it.
        """
        revision = int(self.registry.read().get("revision") or 0)
        store_campaigns: dict[str, JSONObject] = {}
        for experiment in self.store.list("experiments"):
            if (experiment.get("validation_lineage") == self.validation_lineage
                    and experiment.get("campaign_id") != self.campaign_id):
                bucket = store_campaigns.setdefault(experiment["campaign_id"], {"trials": 0, "scores": []})
                bucket["trials"] += 1
                result = experiment.get("result") or {}
                if result.get("status") == "ok":
                    try:
                        score = self.domain.selection_score(result)
                    except Exception:
                        score = None
                    if score is not None and math.isfinite(score):
                        bucket["scores"].append(_round(score))
        accesses: dict[str, int] = {}
        store_retired: JSONObject | None = None
        store_exposed: JSONObject | None = None
        store_batches: dict[str, JSONObject] = {}
        for event in sorted(self.store.list("events"), key=lambda e: str(e.get("created_at", ""))):
            kind = event.get("kind")
            if kind in ("batch_closed", "batch_voided"):
                store_batches.pop(str(event.get("batch_id")), None)
                continue
            if event.get("holdout_lineage") != self.holdout_lineage:
                continue
            if kind == "holdout_access":
                accesses[event["key"]] = max(accesses.get(event["key"], 0), int(event.get("access_sequence", 1)))
                confirmatory = event.get("confirmatory")
                if confirmatory is None:                     # recorded before the flag existed: a batch read or not
                    confirmatory = event.get("batch_id") is not None
                if not confirmatory and store_exposed is None:
                    store_exposed = {"at": event.get("created_at"), "campaign_id": event.get("campaign_id"),
                                     "key": event["key"], "sequence": int(event.get("access_sequence", 1)),
                                     "source": "store"}
            elif kind == "holdout_retired" and store_retired is None:
                store_retired = {"at": event.get("created_at"), "campaign_id": event.get("campaign_id"),
                                 "keys": list(event.get("keys") or []), "level": event.get("level"),
                                 "method": event.get("method") or "batch-holm", "source": "store"}
            elif kind == "batch_opened":
                store_batches[str(event.get("batch_id"))] = {
                    "batch_id": event.get("batch_id"), "campaign_id": event.get("campaign_id"),
                    "keys": list(event.get("keys") or []), "level": event.get("level"),
                    "opened_at": event.get("created_at"), "source": "store",
                    "trials_at_declaration": event.get("trials_at_declaration"),
                    "eligible_not_requested": list(event.get("eligible_not_requested") or []),
                    "registry_revision": event.get("registry_revision")}
        registry_campaigns = {cid: view for cid, view in self.registry.validation_view(self.validation_lineage).items()
                              if cid != self.campaign_id}
        holdout = self.registry.holdout_view(self.holdout_lineage)
        registry_known = bool(holdout["campaigns"] or holdout["accesses"] or holdout["retired"] or holdout["batch"]
                              or holdout.get("exposed"))
        # A batch the record store declared and never closed is open unless the registry's history says it was
        # completed or voided (a crash between the registry's close and the store's event): a registry that merely
        # lost its open-batch record does not reopen the holdout for a new family.
        closed_in_registry = {str(b.get("batch_id")) for b in holdout.get("batches") or []}
        store_open = [b for bid, b in store_batches.items() if bid not in closed_in_registry]
        store_batch = store_open[0] if store_open else None
        for key, item in holdout["accesses"].items():
            accesses[key] = max(accesses.get(key, 0), int(item.get("sequence", 1)))
        trials = 0
        scores: list[float] = []
        for cid in sorted(set(store_campaigns) | set(registry_campaigns)):
            stored, registered = store_campaigns.get(cid), registry_campaigns.get(cid)
            trials += max(int(stored["trials"]) if stored else 0, int(registered["trials"]) if registered else 0)
            stored_scores = list(stored["scores"]) if stored else []
            registered_scores = [float(s) for s in registered["scores"]] if registered else []
            scores += stored_scores if len(stored_scores) >= len(registered_scores) else registered_scores
        agree = ({cid: int(b["trials"]) for cid, b in store_campaigns.items()}
                 == {cid: int(v["trials"]) for cid, v in registry_campaigns.items()})
        return {"trials": trials, "scores": scores, "accesses": accesses, "registry": str(self.registry.path),
                "agree": agree,
                "store_trials": sum(int(b["trials"]) for b in store_campaigns.values()),
                "registry_trials": sum(int(v["trials"]) for v in registry_campaigns.values()),
                "foreign_campaigns": [{"campaign_id": cid, "research_root": v.get("research_root"), "trials": v["trials"]}
                                      for cid, v in registry_campaigns.items() if cid not in store_campaigns],
                "unregistered_campaigns": sorted(cid for cid in store_campaigns if cid not in registry_campaigns),
                "retired": holdout.get("retired") or store_retired,
                "batch": holdout.get("batch") or store_batch,
                "exposed": holdout.get("exposed") or store_exposed,
                "revision": revision, "registry_known": registry_known}

    def _trial_context(self, family: LineageEvidence) -> TrialContext:
        state = self.checkpoint
        return {"n_validation_trials": family["trials"] + state["validation_trials"],
                "campaign_validation_trials": state["validation_trials"],
                "lineage_validation_trials": family["trials"],
                "validation_scores": family["scores"] + list(state["validation_scores"]),
                "policy": dict(self.domain_policy)}

    def holdout_level(self) -> float:
        """The familywise level of holdout confirmation: ``policy.holdout_level``, else the domain's ``alpha_level``, else 0.05.

        A validation threshold on the deflated Sharpe ratio (``min_dsr``) is a different quantity
        and is not turned into a holdout level.
        """
        configured = self.harness_policy.get("holdout_level")
        if configured is not None:
            return float(configured)
        merged: dict[str, Any] = {**(getattr(self.domain, "gates", None) or {}), **self.domain_policy}
        if "alpha_level" in merged:
            return float(merged["alpha_level"])
        return 0.05

    def _registry_check(self) -> JSONObject:
        """The live registry against this campaign's latest ``lineage_snapshot`` (see :func:`verify_registry`)."""
        current = self.registry.snapshot(self.validation_lineage, self.holdout_lineage)
        try:
            entries = self.ledger.entries() if self.ledger.path.exists() else []
        except LedgerError as error:
            return {"status": "ledger_broken", "problems": [str(error)], "snapshot": None, "current": current}
        return verify_registry(latest_snapshot(entries), current)

    def lineage_summary(self) -> JSONObject:
        """Which lineage registry this campaign uses, whether it is the canonical one, and whether the store agrees."""
        family = self._family()
        current = self.registry.snapshot(self.validation_lineage, self.holdout_lineage)
        batch = family["batch"]
        return {"registry": family["registry"], "canonical": self.registry_canonical,
                "default_registry": str(default_lineage_root()), "revision": current["revision"],
                "data_identity": self.registry.data_identity,
                "validation_lineage": self.validation_lineage, "holdout_lineage": self.holdout_lineage,
                "trials": {"store": family["store_trials"], "registry": family["registry_trials"],
                           "agree": family["agree"]},
                "foreign_campaigns": family["foreign_campaigns"],
                "unregistered_campaigns": family["unregistered_campaigns"],
                "holdout_retired": family["retired"],
                "holdout_exposed": family["exposed"],
                "holdout_accesses": dict(family["accesses"]),
                "seeded": self.registry.seeded(self.validation_lineage),
                "registry_status": self._registry_check()["status"],
                "open_batch": None if not batch else {"batch_id": batch.get("batch_id"), "campaign_id": batch.get("campaign_id"),
                                                      "keys": list(batch.get("keys") or []), "level": batch.get("level"),
                                                      "eligible_not_requested": list(batch.get("eligible_not_requested") or []),
                                                      "trials_at_declaration": batch.get("trials_at_declaration"),
                                                      "source": batch.get("source", "registry")},
                "confirmation": self.harness_policy["holdout_confirmation"], "level": self.holdout_level(),
                "non_confirmatory": bool(self.config.get("non_confirmatory"))}

    # ------------------------------------------------------------------- inbox
    def submit(self, candidate: Candidate, submitted_by: str = "operator") -> dict[str, str]:
        """Queue an externally proposed candidate (agent or person) for the next round.

        Submission grants no evaluation priority and bypasses no gate: the
        candidate is screened, counted and reviewed like any generated one.
        """
        if not isinstance(candidate, dict):
            raise ValueError("candidate must be an object")
        key = self.domain.key({**candidate, "generator": "inbox"})
        append_inbox(self.inbox_path, candidate, submitted_by, key=key)
        return {"status": "queued", "campaign_id": self.campaign_id, "key": key}

    @property
    def _inbox_processing(self) -> Path:
        return self.inbox_path.with_name(self.inbox_path.name + ".processing")

    def _read_inbox(self) -> list[Submission]:
        """Take every queued submission. The inbox is renamed, so new submissions start a new file
        and the size cap applies to what is waiting, not to everything ever submitted. The taken
        file is deleted once the round is saved; after a crash it is read again and the campaign's
        deduplication skips what was already screened."""
        processing = self._inbox_processing
        if not processing.exists() and self.inbox_path.exists():
            os.replace(self.inbox_path, processing)
        if not processing.exists():
            return []
        items: list[Submission] = []
        for line in processing.read_bytes().split(b"\n"):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and isinstance(value.get("candidate"), dict):
                items.append(cast(Submission, value))  # shape checked on the line above
        return items

    def _close_inbox(self) -> None:
        self._inbox_processing.unlink(missing_ok=True)

    # ----------------------------------------------------------------- recover
    def _ledger_index(self) -> LedgerIndex:
        index = LedgerIndex()
        for entry in self.ledger.entries():
            if entry["stage"] == "screen":
                index.screen += 1
                index.screen_keys.add(entry["key"])
                if entry.get("status") != "ok":
                    index.screen_errors += 1
            elif entry["stage"] in ("validate", "review") and entry.get("experiment_id"):
                (index.validate if entry["stage"] == "validate" else index.review).add(entry["experiment_id"])
        return index

    def _recover(self) -> None:
        """Bring the checkpoint in line with the ledger and records after an interrupted round."""
        state = self.checkpoint
        if state.get("round_open"):
            index = self._ledger_index()
            seen = set(state["seen"])
            state["seen"] += sorted(index.screen_keys - seen)
            state["screen_trials"] = index.screen
            state["screen_errors"] = index.screen_errors
            self.ledger.append("note", decision="round_interrupted", round=state["round_open"])
            state["round_open"] = None
            self._save()
        if state.get("pending"):
            self._complete(None, recovering=True)

    # ------------------------------------------------------------------- round
    def tick(self) -> RoundOutcome:
        with self.store.lock():
            self._load()
            self._recover()
            state = self.checkpoint
            reason = self._budget_reason()
            if reason:
                state["status"] = "complete"
                self._save()
                return {"campaign_id": self.campaign_id, "status": "complete", "reason": reason}
            number = state["round"] + 1
            seed = int(hashlib.sha256(f"{self.campaign_id}:{self.config['seed']}:{number}".encode()).hexdigest()[:12], 16)
            submitted = self._read_inbox()
            context: ProposalContext = {"round": number, "rng_seed": seed, "n": self.limits["proposals_per_round"],
                                        "population": copy.deepcopy(state["population"]),
                                        "accepted": copy.deepcopy(state["accepted"]),
                                        "submitted": submitted, "seen": set(state["seen"])}
            try:
                proposals = self.domain.propose(context)
            except Exception as error:
                # A malformed submission must not stall every round: record it and continue without it.
                self.ledger.append("note", decision="propose_failed", round=number,
                                   error=(type(error).__name__ + ": " + str(error))[:300],
                                   dropped_submissions=len(submitted))
                context["submitted"], submitted = [], []
                proposals = self.domain.propose(context)
            # The domain may offer more (first-round templates, seeds); the round budget is the
            # documented proposals_per_round plus whatever agents submitted since the last round.
            proposals = proposals[: self.limits["proposals_per_round"] + len(submitted)]
            fresh: list[Candidate] = []
            keys: list[str] = []
            seen = set(state["seen"])
            remaining = self.limits["max_screen_trials"] - state["screen_trials"]
            for candidate in proposals:
                try:
                    key = self.domain.key(candidate)
                except Exception as error:  # an unparsable proposal is still a counted trial
                    key = stable_id("invalid", {"candidate": candidate})
                    candidate = {**candidate, "_invalid": type(error).__name__ + ": " + str(error)[:300]}
                if key in seen:
                    continue
                if len(fresh) >= remaining:
                    if candidate.get("generator") == "inbox":
                        self.ledger.append("note", key=key, decision="submission_not_screened",
                                           reason="screen-trial budget reached")
                    continue
                seen.add(key)
                fresh.append(candidate)
                keys.append(key)
            state["round_open"] = number
            self._save()
            screened = self.domain.screen([c for c in fresh if "_invalid" not in c]) if fresh else []
            outcomes = iter(screened)
            new_entries: list[PopulationEntry] = []
            for candidate, key in zip(fresh, keys):
                generator = candidate.get("generator", "unknown")
                if "_invalid" in candidate:
                    outcome: ScreenOutcome = {"status": "error", "fitness": None, "metrics": {},
                                              "error": candidate["_invalid"]}
                else:
                    outcome = next(outcomes)
                state["screen_trials"] += 1
                state["seen"].append(key)
                self._count(generator, "screened")
                if outcome.get("status") != "ok":
                    state["screen_errors"] += 1
                self.ledger.append("screen", key=key, round=number, generator=generator,
                                   status=outcome.get("status"), fitness=_round(outcome.get("fitness")),
                                   metrics=_round(outcome.get("metrics") or {}),
                                   error=(outcome.get("error") or "")[:300] or None)
                if outcome.get("status") == "ok" and outcome.get("fitness") is not None:
                    chosen = outcome.get("candidate") or candidate  # the domain may orient the signal
                    new_entries.append({"key": key, "candidate": {k: v for k, v in chosen.items() if k != "_invalid"},
                                        "fitness": _round(outcome["fitness"]), "metrics": _round(outcome.get("metrics") or {}),
                                        "signature": _round(outcome.get("signature") or []), "round": number,
                                        "generator": generator, "validated": False})
            population = state["population"] + new_entries
            population.sort(key=_by_fitness)
            state["population"] = population[: self.limits["population_size"]]
            self._save()
            family = self._family()
            validated: list[Decision] = []
            for entry in self._shortlist():
                if self._budget_reason():
                    break
                validated.append(self._validate_and_review(entry, number, family))
            state["round"] = number
            state["round_open"] = None
            state["status"] = "running"
            self._save()
            self._close_inbox()
            summary: RoundSummary = {"round": number, "proposed": len(proposals), "screened": len(fresh),
                                     "screen_ok": len(new_entries), "submitted": len(submitted),
                                     "validated": len(validated),
                                     "promoted": sum(1 for v in validated if v["decision"] == "promote"),
                                     "best_fitness": state["population"][0]["fitness"] if state["population"] else None}
            self._event("round", **summary)
            return {"campaign_id": self.campaign_id, "status": "round", **summary,
                    "decisions": validated}

    def _shortlist(self) -> list[PopulationEntry]:
        state = self.checkpoint
        threshold = self.config["selection"]["max_correlation"] or self.domain.default_max_correlation
        # Diversity is judged within a group: a blend of atomic signals is compared with
        # other blends, not with its own components (which it is meant to resemble).
        references = [(a.get("diversity_group", "atomic"), a.get("signature")) for a in state["accepted"]]
        references += [(v.get("diversity_group", "atomic"), v.get("signature")) for v in state["validated"].values()]
        chosen: list[PopulationEntry] = []
        for entry in state["population"]:
            if len(chosen) >= self.limits["validations_per_round"]:
                break
            if entry["validated"] or entry["key"] in state["validated"]:
                continue
            if not self.domain.shortlist_ok(entry):
                continue
            group = entry["candidate"].get("diversity_group", "atomic")
            pool = [sig for g, sig in references if g == group]
            pool += [e["signature"] for e in chosen if e["candidate"].get("diversity_group", "atomic") == group]
            correlations = [abs(c) for c in (_correlation(entry["signature"], ref) for ref in pool) if c is not None]
            if correlations and max(correlations) >= threshold:
                entry["validated"] = True  # redundant: never spend a validation trial on it
                state["validated"].setdefault(entry["key"], {"decision": "redundant", "score": None,
                                                             "max_correlation": _round(max(correlations))})
                self.ledger.append("note", key=entry["key"], decision="redundant",
                                   max_correlation=_round(max(correlations)))
                self._count(entry["generator"], "redundant")
                continue
            chosen.append(entry)
        return chosen

    def _validate_and_review(self, entry: PopulationEntry, number: int, family: LineageEvidence) -> Decision:
        state = self.checkpoint
        key, candidate = entry["key"], entry["candidate"]
        name = self.config["domain"]
        hypothesis_id = stable_id("hypothesis", {"campaign": self.campaign_id, "key": key})
        if self.store.get("hypotheses", hypothesis_id) is None:
            # Selected after any holdout of this data was read: never an untouched final test.
            post_holdout = bool(state["holdout_accessed"] or family["accesses"])
            record: HypothesisRecord = {"id": hypothesis_id, "campaign_id": self.campaign_id, "domain": name, "key": key,
                                        "generator": entry["generator"], "parents": candidate.get("parents", []),
                                        "round": number, "screen": entry["metrics"], "screen_fitness": entry["fitness"],
                                        "post_holdout": post_holdout, "validation_lineage": self.validation_lineage,
                                        **self.domain.describe_candidate(candidate), "created_at": utc_now()}
            self.store.put("hypotheses", _round(record))
        state["validation_trials"] += 1
        experiment_id = stable_id("experiment", {"campaign": self.campaign_id, "key": key,
                                                 "attempt": state["validation_trials"]})
        state["pending"] = {"key": key, "hypothesis_id": hypothesis_id, "experiment_id": experiment_id,
                            "started_at": utc_now(), "candidate": candidate, "signature": entry["signature"],
                            "generator": entry["generator"]}
        self._save()  # reserve before running: an interrupted validation still counts
        self.registry.record_trial(self.validation_lineage, self.campaign_id, experiment_id,
                                   research_root=str(self.research_root), lineage=self.lineage["validation"])
        try:
            result = self.domain.validate(candidate)
        except Exception as error:
            result = {"status": "error", "error_kind": "execution", "error": type(error).__name__ + ": " + str(error)[:300]}
        entry["validated"] = True
        return self._complete(result, family=family)

    def _complete(self, result: EvaluationResult | None, *, recovering: bool = False,
                  family: LineageEvidence | None = None) -> Decision:
        """Publish a reserved validation: experiment, ledger, score, review, decision. Idempotent.

        Normal runs call it straight after the domain's validation. After a crash it is called
        with no result and finishes whatever the interrupted run had not yet written, so the
        checkpoint, the record store and the ledger always agree.
        """
        state = self.checkpoint
        pending = state["pending"]
        if pending is None:
            raise RuntimeError("no validation trial is pending")
        key, hypothesis_id, experiment_id = pending["key"], pending["hypothesis_id"], pending["experiment_id"]
        candidate = pending.get("candidate") or {}
        name = self.config["domain"]
        index = self._ledger_index() if recovering else LedgerIndex()
        experiment = self.store.get("experiments", experiment_id)
        if experiment is None:
            if result is None:
                result = {"status": "error", "error_kind": "interrupted",
                          "error": "Worker stopped before the validation result was published; outcome unknown."}
            record: ExperimentRecord = {
                "id": experiment_id, "hypothesis_id": hypothesis_id, "campaign_id": self.campaign_id,
                "experiment_key": key, "domain": name, "validation_lineage": self.validation_lineage,
                "result": result, "created_at": pending["started_at"]}
            if candidate:
                record = {**record, **self.domain.describe_candidate(candidate)}
            experiment = self.store.put("experiments", _round(record))
        outcome: EvaluationResult = experiment["result"]
        score: float | None = None
        if outcome.get("status") == "ok":
            try:
                score = self.domain.selection_score(outcome)
            except Exception:
                score = None
        if experiment_id not in index.validate:
            self.ledger.append("validate", key=key, hypothesis_id=hypothesis_id, experiment_id=experiment_id,
                               status=outcome.get("status"), error_kind=outcome.get("error_kind"), score=_round(score),
                               metrics=_round((outcome.get("metrics") or {}).get("validation", {})))
        self.registry.record_score(self.validation_lineage, self.campaign_id, experiment_id,
                                   _round(score) if score is not None and math.isfinite(score) else None,
                                   research_root=str(self.research_root), lineage=self.lineage["validation"])
        if score is not None and math.isfinite(score):
            state["validation_scores"].append(_round(score))
        generator = pending.get("generator", "unknown")
        self._count(generator, "validated")
        context = self._trial_context(family if family is not None else self._family())
        context["accepted"] = state["accepted"]
        context["signature"] = pending.get("signature")
        review_id = stable_id("review", {"experiment": experiment_id, "policy": self.config.get("policy", {})})
        stored = self.store.get("reviews", review_id)
        if stored is not None:
            review: Review = {"decision": stored["decision"], "reason": stored["reason"],
                              "checks": stored.get("checks", {})}
        else:
            try:
                review = self.domain.review(candidate, outcome, context) if candidate else \
                    {"decision": "repair", "reason": "Interrupted before review; candidate unknown.", "checks": {}}
            except Exception as error:
                review = {"decision": "repair", "reason": "Review failed: " + type(error).__name__, "checks": {}}
            review.setdefault("checks", {})
            review["checks"]["validation_trials"] = {"campaign": context["campaign_validation_trials"],
                                                     "lineage": context["lineage_validation_trials"],
                                                     "total": context["n_validation_trials"]}
            review_record: ReviewRecord = {"id": review_id, "campaign_id": self.campaign_id,
                                           "hypothesis_id": hypothesis_id, "experiment_id": experiment_id,
                                           "decision": review["decision"], "reason": review["reason"],
                                           "checks": review.get("checks", {}), "question": "",
                                           "expected_change": "", "created_at": utc_now()}
            self.store.put("reviews", _round(review_record))
        if experiment_id not in index.review:
            self.ledger.append("review", key=key, hypothesis_id=hypothesis_id, experiment_id=experiment_id,
                               decision=review["decision"], reason=review["reason"][:300])
        group = candidate.get("diversity_group", "atomic")
        state["validated"][key] = {"hypothesis_id": hypothesis_id, "experiment_id": experiment_id,
                                   "decision": review["decision"], "score": _round(score),
                                   "signature": pending.get("signature"), "diversity_group": group}
        self._count(generator, review["decision"])
        if review["decision"] == "promote" and key not in {a["key"] for a in state["accepted"]}:
            state["accepted"].append({"key": key, "hypothesis_id": hypothesis_id, "experiment_id": experiment_id,
                                      "review_id": review_id, "candidate": candidate, "score": _round(score),
                                      "signature": pending.get("signature"), "generator": generator,
                                      "diversity_group": group, "checks": _round(review.get("checks", {}))})
            state["promotions"] += 1
        for entry in state["population"]:
            if entry["key"] == key:
                entry["validated"] = True
        state["pending"] = None
        self._save()
        return {"key": key, "hypothesis_id": hypothesis_id, "decision": review["decision"],
                "reason": review["reason"], "score": _round(score)}

    # ------------------------------------------------------------------- final
    def final_context(self) -> TrialContext:
        return self._trial_context(self._family())

    def _final_eligibility(self, context: TrialContext,
                           keys: list[str] | None) -> tuple[list[AcceptedEntry], list[JSONObject], list[str]]:
        """Promotions the domain still selects at the final trial count, the requested ones it no longer does, and the eligible keys.

        The eligible set is always the domain's ``final_selection`` under the lineage-wide
        trial count at this moment. Explicit ``keys`` select a subset of it: a promoted key
        that no longer passes is refused, returned with the recomputed statistic and never
        scored; a key that was never promoted is an error. The third value is every eligible
        key, requested or not, so a caller can show which eligible promotions a request left out.
        """
        state = self.checkpoint
        known = {a["key"]: a for a in state["accepted"]}
        eligible = {e["key"] for e in self.domain.final_selection(state["accepted"], context)}
        requested = list(known) if keys is None else list(dict.fromkeys(keys))  # a repeated key is one request
        for key in requested:
            if key not in known:
                raise ValueError(f"{key} was not promoted in this campaign")
        selected = [known[key] for key in requested if key in eligible]
        excluded: list[JSONObject] = []
        for key in requested:
            if key in eligible:
                continue
            check = {k: v for k, v in self.domain.final_check(known[key], context).items() if k != "eligible"}
            excluded.append({"key": key, "hypothesis_id": known[key]["hypothesis_id"], "requested": keys is not None,
                             "n_validation_trials": context["n_validation_trials"], **check})
        return selected, excluded, sorted(eligible)

    def final_eligibility(self, keys: list[str] | None = None) -> JSONObject:
        """Read-only: which promotions still pass the domain's final selection at the lineage-wide trial count."""
        with self.store.lock():
            self._load()
            self._recover()
            context = self._trial_context(self._family())
            selected, excluded, eligible = self._final_eligibility(context, keys)
            requested = None if keys is None else set(keys)
            return {"n_validation_trials": context["n_validation_trials"],
                    "eligible": [entry["key"] for entry in selected], "excluded": _round(excluded),
                    "eligible_not_requested": [] if requested is None else [k for k in eligible if k not in requested]}

    def finalize(self, keys: list[str] | None = None, *, allow_reuse: bool = False,
                 non_confirmatory: bool = False, void_batch: str | None = None) -> list[FinalOutcome]:
        """Score selected candidates once on the reserved holdout; record immutable finals.

        The eligible set is recomputed at this moment: accepted candidates that
        still pass the domain's final selection with the final trial count of
        this data's lineage. Explicit ``keys`` select a subset of that set; a
        promoted key that no longer passes is refused and recorded
        (``final_selection_excluded`` in the ledger and the store, with the final
        trial count and the recomputed statistic), never scored.

        Holdout accesses are counted across every campaign on the same holdout
        lineage, in this record store and in the lineage registry beside the
        data, and reserved (ledger, store event, registry, checkpoint) before
        each evaluation. A repeated access is refused unless ``allow_reuse``; a
        reused holdout, or a candidate selected after any holdout of this data
        was read, is never confirmed.

        Confirmation follows ``policy.holdout_confirmation``. ``"batch"`` (the
        default) declares the batch in the registry (``batch_opened``: the whole
        family and its level) before the first access, stores every scored
        holdout result (``holdout_scored`` events), gives each finalist a
        one-sided p-value that its holdout evidence is null (the probabilistic
        Sharpe ratio of its holdout daily PnL over the holdout sessions, or a
        tabular task's bootstrap p-value), applies Holm's step-down across the
        batch at :meth:`holdout_level`, and then retires the lineage's holdout
        for confirmation in the registry: later reads, by any campaign, are
        non-confirmatory. A batch interrupted after some accesses is completed
        by this campaign's next ``finalize`` over the same family, re-reading the
        stored results of the keys already scored, or abandoned explicitly with
        ``void_batch`` (a reason, recorded); another campaign cannot confirm on
        the lineage while the batch is open. ``"per-candidate"`` keeps the
        domain's own verdict per candidate and retires nothing.

        A read is non-confirmatory, and every final ``not_confirmed``, when the
        caller or the campaign file says so, when this record store disagrees
        with the lineage registry, or when the holdout was already retired;
        without the flag the latter two refuse to finalize. Every final says
        whether its registry is the canonical one (``registry_canonical``), and
        every finalize ends with a ``lineage_snapshot`` of the registry in the
        ledger, which :meth:`verify` checks the registry against later.
        """
        with self.store.lock():
            self._load()
            self._recover()
            state = self.checkpoint
            family = self._family()
            context = self._trial_context(family)
            known = {a["key"]: a for a in state["accepted"]}
            for key in dict.fromkeys(keys or []):
                if key not in known:
                    raise ValueError(f"{key} was not promoted in this campaign")
            declared = bool(non_confirmatory or self.config.get("non_confirmatory"))
            exploratory: list[str] = ["declared non-confirmatory"] if declared else []
            if not family["agree"]:
                where = ", ".join(f"{c['campaign_id']} in {c['research_root']}" for c in family["foreign_campaigns"]) \
                    or "campaigns the registry no longer lists: " + ", ".join(family["unregistered_campaigns"])
                message = (f"this record store ({self.research_root}) disagrees with the lineage registry "
                           f"{family['registry']}: trials by other campaigns {family['store_trials']} here versus "
                           f"{family['registry_trials']} registered ({where})")
                if not declared:
                    raise ValueError(message + "; finalize with --non-confirmatory to read the holdout without confirmation")
                exploratory.append(message)
            # The registry against this campaign's last snapshot of it: a replaced, deleted or shrunk registry
            # is not merely reported by verify, it cannot confirm.
            check = self._registry_check()
            registry_status = str(check["status"])
            if registry_status == "ledger_broken":
                raise ValueError("this campaign's trial ledger does not verify (" + "; ".join(str(p) for p in check["problems"])
                                 + "); nothing can be finalized from it")
            if registry_status in ("registry_reset", "registry_shrunk"):
                message = (f"the lineage registry {family['registry']} does not match this campaign's last snapshot of it "
                           f"({registry_status}: " + "; ".join(str(p) for p in check["problems"]) + ")")
                if not declared:
                    raise ValueError(message + "; restore the registry, or finalize with --non-confirmatory to read the "
                                     "holdout without confirmation")
                exploratory.append(message)
            retired = family["retired"]
            if retired:
                how = (("when a batch was voided after reading " + ", ".join(retired.get("keys") or []))
                       if retired.get("method") == "voided-after-access"
                       else f"after a batch of {len(retired.get('keys') or [])}")
                message = (f"the holdout of this lineage was retired for confirmation on {str(retired.get('at', ''))[:19]} by "
                           f"campaign {retired.get('campaign_id')} {how}")
                if not declared:
                    raise ValueError(message + "; it can only be read again as non-confirmatory (--non-confirmatory)")
                exploratory.append(message)
            mode = str(self.harness_policy["holdout_confirmation"])
            level = self.holdout_level()
            open_batch = family["batch"]
            if open_batch and open_batch.get("campaign_id") != self.campaign_id:
                message = (f"campaign {open_batch.get('campaign_id')} has an unfinished holdout batch on this lineage "
                           f"({len(open_batch.get('keys') or [])} keys, opened {str(open_batch.get('opened_at', ''))[:19]}); "
                           "it must be completed or voided from that campaign first")
                if not declared:
                    raise ValueError(message + "; a non-confirmatory read is possible with --non-confirmatory")
                exploratory.append(message)
                open_batch = None
            voided = bool(open_batch and void_batch)
            if open_batch and void_batch:
                batch_ref = str(open_batch["batch_id"])
                # Keys the batch read or reserved: the store's events and this campaign's ledger lines (a crash between
                # the ledger line and the store event leaves the reservation in the ledger alone).
                read = {str(e["key"]) for e in self.store.list("events")
                        if e.get("batch_id") == batch_ref and e.get("kind") in ("holdout_access", "holdout_scored")}
                read |= {str(e["key"]) for e in self.ledger.entries() if e.get("stage") == "holdout" and e.get("batch_id") == batch_ref}
                touched = sorted(read)
                retire_now = bool(touched) and not retired
                # One registry write closes the batch and, when a key of it was read, retires the lineage's holdout: there
                # is no revision at which the batch is gone but the holdout still open for another record store to declare on.
                self.registry.void_batch(self.holdout_lineage, batch_ref, self.campaign_id, reason=void_batch[:300],
                                         retire=retire_now, keys=touched, level=level, method="voided-after-access",
                                         research_root=str(self.research_root), lineage=self.lineage["holdout"],
                                         declared=open_batch)
                self.ledger.append("note", decision="batch_voided", batch_id=batch_ref,
                                   keys=list(open_batch.get("keys") or []), reason=void_batch[:300], read=touched)
                self._event("batch_voided", batch_id=batch_ref, keys=list(open_batch.get("keys") or []),
                            reason=void_batch[:300], read=touched, holdout_lineage=self.holdout_lineage)
                open_batch = None
                if retire_now:
                    # A key of the family was read, so part of the batch's answer is known: the holdout is no longer
                    # an untouched test for the keys that remain.
                    self.ledger.append("note", decision="holdout_retired", keys=touched, level=level, batch=len(touched),
                                       method="voided-after-access")
                    self._event("holdout_retired", holdout_lineage=self.holdout_lineage, keys=touched, level=level,
                                batch=len(touched), method="voided-after-access")
                    self._snapshot("void")
                    message = ("the voided batch had already read " + ", ".join(touched)
                               + ", so the holdout of this lineage is retired for confirmation")
                    if not declared:
                        raise ValueError(message + "; its remaining candidates can only be read as non-confirmatory "
                                         "(--non-confirmatory)")
                    exploratory.append(message)
            if voided:
                # The void (and any retirement) wrote the registry: read the family again so the batch is declared
                # against the revision that includes it.
                family, context = self._family(), self._trial_context(self._family())
            exposed = family["exposed"]
            exposure_informs = bool(exposed)
            if exposed and mode == "batch":
                # A read outside a confirmatory batch (exploratory, per-candidate or repeated) happened on this
                # lineage: a batch declared after it is informed by the holdout. The batch that was already open
                # when the exposure happened may still be completed; its family was declared before.
                completing = bool(open_batch) and str((open_batch or {}).get("opened_at") or "") <= str(exposed.get("at") or "")
                exposure_informs = not completing
                if not completing:
                    message = (f"the holdout of this lineage was read outside a confirmatory batch on "
                               f"{str(exposed.get('at', ''))[:19]} by campaign {exposed.get('campaign_id')} "
                               f"(key {exposed.get('key')}), so a confirmatory batch after it is not an untouched test")
                    if not declared:
                        raise ValueError(message + "; finalize with --non-confirmatory to read the holdout as evidence")
                    exploratory.append(message)
            resumed = bool(open_batch) and mode == "batch" and not exploratory
            if open_batch and not resumed:
                raise ValueError(f"this campaign has an unfinished holdout batch ({len(open_batch.get('keys') or [])} keys); "
                                 "a confirmatory finalize completes it over the same family, or void it with "
                                 "--void-batch REASON")
            scored: dict[str, JSONObject] = {}
            reserved: dict[str, int] = {}     # accesses the interrupted batch reserved but never scored
            already: list[AcceptedEntry] = []
            batch_id: str | None = None
            selected: list[AcceptedEntry] = []
            excluded: list[JSONObject] = []
            eligible: list[str] = []
            accesses: dict[str, int] = {}
            attempt = 0
            while True:
                if resumed and open_batch:
                    batch_id = str(open_batch["batch_id"])
                    level = float(open_batch.get("level") or level)
                    family_keys = [k for k in open_batch.get("keys") or [] if k in known]
                    if keys is not None and set(keys) - set(family_keys):
                        raise ValueError("an unfinished holdout batch is completed over its own family ("
                                         + ", ".join(family_keys) + "); request a subset of it, or void it with --void-batch REASON")
                    for event in self.store.list("events"):
                        if event.get("batch_id") != batch_id:
                            continue
                        if event.get("kind") == "holdout_scored":
                            scored[str(event["key"])] = event
                        elif event.get("kind") == "holdout_access":
                            reserved[str(event["key"])] = int(event.get("access_sequence") or 1)
                    already = [known[k] for k in family_keys if k in scored]
                    pending = [k for k in family_keys if k not in scored]
                    # Keys not yet read are re-checked against today's trial count; the family may only shrink.
                    selected, excluded, eligible = self._final_eligibility(context, pending) if pending else ([], [], [])
                else:
                    selected, excluded, eligible = self._final_eligibility(context, keys)
                accesses = family["accesses"]
                for entry in selected:
                    if entry["key"] in reserved and entry["key"] not in scored:
                        continue        # reserved by this batch before the interruption: the same read, completed now
                    if accesses.get(entry["key"], 0) and not allow_reuse:
                        raise ValueError("holdout already evaluated for " + entry["key"]
                                         + " (in this or another campaign on the same data); repeated access must be"
                                         + " requested explicitly and is labelled reused")
                if resumed or mode != "batch" or not selected or exploratory:
                    break
                # Declare the family against the registry revision its eligibility was computed from: the trial
                # count each finalist was priced at and the batch that reads the holdout are one decision. A
                # registry written meanwhile (another campaign's trials or reads) sends us round once more.
                attempt += 1
                opened = sum(1 for e in self.ledger.entries() if e.get("stage") == "note" and e.get("decision") == "batch_opened")
                batch_keys = [entry["key"] for entry in selected]
                not_requested = [k for k in eligible if k not in set(batch_keys)]
                declared_id = stable_id("batch", {"campaign": self.campaign_id, "keys": batch_keys, "sequence": opened + 1})
                try:
                    self.registry.open_batch(self.holdout_lineage, self.campaign_id, declared_id, keys=batch_keys, level=level,
                                             method="batch-holm", research_root=str(self.research_root),
                                             lineage=self.lineage["holdout"], revision=family["revision"],
                                             eligible_not_requested=not_requested,
                                             trials_at_declaration=int(context["n_validation_trials"]))
                except RegistryMoved as moved:
                    if attempt >= 3:
                        raise ValueError(str(moved) + "; another campaign keeps writing the registry, finalize again") from moved
                    refreshed = self._family()
                    if (refreshed["agree"], refreshed["retired"], refreshed["batch"], refreshed["exposed"]) != \
                            (family["agree"], family["retired"], family["batch"], family["exposed"]):
                        raise ValueError("the lineage registry changed while finalizing (agreement, retirement, an open "
                                         "batch or an exposure moved); finalize again") from moved
                    family, context = refreshed, self._trial_context(refreshed)
                    continue
                batch_id = declared_id
                self.ledger.append("note", decision="batch_opened", batch_id=batch_id, keys=batch_keys, level=level,
                                   eligible_not_requested=not_requested, registry_revision=family["revision"], attempt=attempt,
                                   trials_at_declaration=int(context["n_validation_trials"]))
                self._event("batch_opened", batch_id=batch_id, keys=batch_keys, level=level,
                            holdout_lineage=self.holdout_lineage, eligible_not_requested=not_requested,
                            registry_revision=family["revision"], attempt=attempt,
                            trials_at_declaration=int(context["n_validation_trials"]))
                self._snapshot("batch_opened")      # the declared batch is in the ledger's picture of the registry
                break
            for detail in excluded:
                self.ledger.append("note", decision="final_selection_excluded", **_round(detail))
                self._event("final_selection_excluded", **detail)
            seeded = self.registry.seeded(self.validation_lineage) is not None
            exposed_before = exposure_informs
            # The lineage-wide count the family was priced at: a trial recorded after the declaration did not take part
            # in selecting it and does not enter it (the registry's batch record keeps the count for a resumed batch).
            priced_at = int(((open_batch or {}).get("trials_at_declaration") if resumed else None)
                            or context["n_validation_trials"])
            confirmatory = batch_id is not None and not exploratory     # a batch read; anything else exposes the lineage
            reads: list[tuple[AcceptedEntry, int, bool, EvaluationResult, Verdict]] = []
            for entry in already:   # scored before the interruption: the stored result, not a second read
                event = scored[entry["key"]]
                result: EvaluationResult = dict(event.get("result") or {})
                sequence = int(event.get("access_sequence") or state["holdout_access"].get(entry["key"], 1))
                post_holdout = bool(result.get("post_holdout"))
                verdict: Verdict = self.domain.assess_final(result)
                if verdict["decision"] == "confirmed" and post_holdout:
                    verdict = {"decision": "not_confirmed",
                               "reason": "Selected after a holdout of this data was read; not an untouched final test."}
                reads.append((entry, sequence, post_holdout, result, verdict))
            for entry in selected:
                hypothesis = self.store.get("hypotheses", entry["hypothesis_id"]) or {}
                post_holdout = bool(hypothesis.get("post_holdout"))
                if entry["key"] in reserved:
                    # The interrupted batch reserved this read and never scored it: complete that same access.
                    sequence = reserved[entry["key"]]
                    accesses[entry["key"]] = max(accesses.get(entry["key"], 0), sequence)
                    state["holdout_access"][entry["key"]] = sequence
                    state["holdout_accessed"] = True
                    self.ledger.append("note", decision="holdout_resumed", key=entry["key"], access_sequence=sequence,
                                       batch_id=batch_id)
                else:
                    sequence = accesses.get(entry["key"], 0) + 1
                    accesses[entry["key"]] = sequence
                    state["holdout_access"][entry["key"]] = sequence
                    state["holdout_accessed"] = True
                    self.ledger.append("holdout", key=entry["key"], hypothesis_id=entry["hypothesis_id"],
                                       access_sequence=sequence, non_confirmatory=bool(exploratory), batch_id=batch_id,
                                       confirmatory=confirmatory)
                    self._event("holdout_access", key=entry["key"], holdout_lineage=self.holdout_lineage,
                                access_sequence=sequence, hypothesis_id=entry["hypothesis_id"], batch_id=batch_id,
                                confirmatory=confirmatory)
                    self.registry.record_access(self.holdout_lineage, self.campaign_id, entry["key"], sequence,
                                                research_root=str(self.research_root), lineage=self.lineage["holdout"],
                                                confirmatory=confirmatory)
                self._save()  # reserved in the ledger, the store, the registry and the checkpoint before the evaluation
                try:
                    result = self.domain.evaluate_holdout(entry["candidate"])
                except Exception as error:
                    result = {"status": "error", "error_kind": "execution",
                              "error": type(error).__name__ + ": " + str(error)[:300]}
                result = {**result, "reused": sequence > 1, "access_sequence": sequence, "post_holdout": post_holdout,
                          "non_confirmatory": bool(exploratory)}
                # The scored result is kept before the batch decides, so an interrupted batch can be completed
                # over the same family without reading this holdout a second time.
                self._event("holdout_scored", key=entry["key"], hypothesis_id=entry["hypothesis_id"],
                            access_sequence=sequence, batch_id=batch_id, holdout_lineage=self.holdout_lineage,
                            result=_round(result))
                verdict = self.domain.assess_final(result)
                if verdict["decision"] == "confirmed" and post_holdout:
                    verdict = {"decision": "not_confirmed",
                               "reason": "Selected after a holdout of this data was read; not an untouched final test."}
                reads.append((entry, sequence, post_holdout, result, verdict))
            confirmations = self._confirm_batch(reads, level) if mode == "batch" else \
                {entry["key"]: {"method": "per-candidate"} for entry, *_ in reads}
            outcomes: list[FinalOutcome] = []
            for entry, sequence, post_holdout, result, verdict in reads:
                confirmation = confirmations[entry["key"]]
                decision, reason = verdict["decision"], verdict["reason"]
                if mode == "batch" and decision == "confirmed":
                    if confirmation["holm_rejected"]:
                        reason += (f" Confirmed by Holm across a batch of {confirmation['batch']}: nominal p = "
                                   f"{confirmation['p_value']:.3g} ({confirmation['statistic_note']}) at step-down level "
                                   f"{confirmation['holm_level']:.3g}, familywise level {level:g}.")
                    else:
                        shown = "unavailable" if confirmation["p_value"] is None else f"{confirmation['p_value']:.3g}"
                        decision, reason = "not_confirmed", (
                            f"Holdout nominal p-value {shown} ({confirmation['statistic_note']}) is not below its Holm "
                            f"step-down level {confirmation['holm_level']:.3g} across a batch of {confirmation['batch']} "
                            f"at familywise level {level:g}.")
                if exploratory:
                    note = "Read as non-confirmatory (" + "; ".join(exploratory) + ")."
                    if decision == "confirmed":
                        decision, reason = "not_confirmed", note + " Not an untouched final test."
                    else:
                        reason = reason + " " + note
                record: FinalRecord = {"id": stable_id("final", {"campaign": self.campaign_id, "key": entry["key"],
                                                                 "sequence": sequence}),
                                       "campaign_id": self.campaign_id, "domain": self.config["domain"],
                                       "hypothesis_id": entry["hypothesis_id"], "experiment_id": entry["experiment_id"],
                                       "experiment_key": entry["key"], **self.domain.describe_candidate(entry["candidate"]),
                                       "access_sequence": sequence, "reused": sequence > 1, "post_holdout": post_holdout,
                                       "holdout_lineage": self.holdout_lineage, "result": result,
                                       "decision": decision, "reason": reason, "created_at": utc_now(),
                                       "confirmation": confirmation, "non_confirmatory": bool(exploratory),
                                       "registry_canonical": self.registry_canonical, "batch_id": batch_id,
                                       "lineage_seeded": seeded, "lineage_exposed": exposed_before,
                                       "registry_status": registry_status, "trials_at_declaration": priced_at}
                stored = self.store.put("finals", _round(record))
                outcomes.append({"final_id": stored["id"], "key": entry["key"], "hypothesis_id": entry["hypothesis_id"],
                                 "decision": stored["decision"], "reason": stored["reason"], "reused": sequence > 1,
                                 "post_holdout": post_holdout,
                                 "holdout": (stored["result"].get("metrics") or {}).get("holdout", {}),
                                 "confirmation": stored.get("confirmation") or confirmation,
                                 "non_confirmatory": bool(exploratory),
                                 "registry_canonical": self.registry_canonical, "batch_id": batch_id,
                                 "lineage_seeded": seeded, "lineage_exposed": exposed_before,
                                 "registry_status": registry_status, "trials_at_declaration": priced_at})
            if mode == "batch" and reads and not exploratory:
                batch_keys = [entry["key"] for entry, *_ in reads]
                self.registry.retire_holdout(self.holdout_lineage, self.campaign_id, keys=batch_keys, level=level,
                                             method="batch-holm", research_root=str(self.research_root),
                                             lineage=self.lineage["holdout"])
                self.ledger.append("note", decision="holdout_retired", keys=batch_keys, level=level, batch=len(batch_keys),
                                   method="batch-holm")
                self._event("holdout_retired", holdout_lineage=self.holdout_lineage, keys=batch_keys, level=level,
                            batch=len(batch_keys), method="batch-holm")
                if batch_id is not None:
                    self.registry.close_batch(self.holdout_lineage, batch_id, status="completed", keys=batch_keys)
                    self.ledger.append("note", decision="batch_closed", batch_id=batch_id, keys=batch_keys,
                                       resumed=resumed)
                    self._event("batch_closed", batch_id=batch_id, keys=batch_keys, resumed=resumed,
                                holdout_lineage=self.holdout_lineage)
            self._save()
            self._event("finalized", count=len(outcomes), confirmation=mode, non_confirmatory=bool(exploratory),
                        batch_id=batch_id)
            self._snapshot("finalize")
            return outcomes

    @staticmethod
    def _confirm_batch(reads: list[tuple[AcceptedEntry, int, bool, EvaluationResult, Verdict]],
                       level: float) -> dict[str, JSONObject]:
        """Holm's step-down over one batch of holdout p-values at familywise ``level``.

        Every finalist in the batch is a member of the family, including one the domain already
        declined, so the correction is never weaker than the batch that was actually read; a
        finalist without a usable statistic enters with p = 1.
        """
        pvalues: dict[str, float] = {}
        details: dict[str, tuple[float | None, str]] = {}
        for entry, _sequence, _post_holdout, result, _verdict in reads:
            p_value, statistic = holdout_pvalue(result)
            details[entry["key"]] = (p_value, statistic)
            pvalues[entry["key"]] = 1.0 if p_value is None else p_value
        rejected = holm(pvalues, level) if pvalues else {}
        size = len(pvalues)
        ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
        step_levels = {key: level / (size - rank) for rank, (key, _p) in enumerate(ordered)}
        return {key: {"method": "batch-holm", "level": level, "batch": size, "p_value": details[key][0],
                      "statistic": details[key][1], "statistic_note": statistic_note(details[key][1]),
                      "holm_level": step_levels[key], "holm_rejected": bool(rejected[key])}
                for key in pvalues}

    # ------------------------------------------------------------------ verify
    def verify(self) -> JSONObject:
        """Ledger chain, checkpoint-versus-ledger counts, and the registry against the ledger's latest snapshot."""
        with self.store.lock():
            self._load()
            s = self.checkpoint
            ledger = self.ledger.verify()
            stages = ledger.get("stages") or {}
            consistent = bool(ledger.get("ok")) and stages.get("validate", 0) == s["validation_trials"] \
                and stages.get("screen", 0) == s["screen_trials"]
            entries = self.ledger.entries() if ledger.get("ok") else []
            registry = verify_registry(latest_snapshot(entries),
                                       self.registry.snapshot(self.validation_lineage, self.holdout_lineage))
            registry["path"] = str(self.registry.path)
            registry["canonical"] = self.registry_canonical
            return {"campaign_id": self.campaign_id, "ledger": ledger, "checkpoint_matches_ledger": bool(consistent),
                    "registry": registry, "ok": bool(consistent) and registry["status"] in ("ok", "no_snapshot")}

    # ------------------------------------------------------------------ status
    def status(self) -> CampaignStatus:
        with self.store.lock():
            self._load()
            s = self.checkpoint
            return {"campaign_id": self.campaign_id, "domain": self.config["domain"], "name": self.config["name"],
                    "status": s["status"], "round": s["round"], "screen_trials": s["screen_trials"],
                    "screen_errors": s["screen_errors"], "validation_trials": s["validation_trials"],
                    "promotions": s["promotions"], "population": len(s["population"]),
                    "best_fitness": s["population"][0]["fitness"] if s["population"] else None,
                    "budget": self._budget_reason(), "limits": self.limits,
                    "holdout_accessed": s["holdout_accessed"], "generators": s["generators"]}
