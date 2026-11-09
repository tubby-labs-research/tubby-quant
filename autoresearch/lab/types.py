"""The shapes of what the harness reads and writes: configuration, checkpoint, records, reports.

Every payload here is JSON on disk or on the wire, so these are ``TypedDict``s rather
than classes: a checkpoint loaded with :func:`json.loads` is already the right shape.
``dict[str, Any]`` is kept only where the shape belongs to someone else — a domain's
candidate, metrics and evaluation result — or is open by design (ledger entries and
events carry stage-specific fields).
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

#: A JSON object whose keys the harness does not interpret.
JSONObject = dict[str, Any]
#: A domain's proposal: ``{"expression", ...}`` for alphas, ``{"family", "params", ...}`` for recipes.
Candidate = dict[str, Any]
#: Named numbers from a screen, a validation or a holdout; the domain chooses the names.
Metrics = dict[str, Any]
#: The shared result format of :mod:`autoresearch.backend`: ``status`` plus ``metrics``,
#: ``periods``, ``robustness``, ``model``, ``error`` and ``error_kind`` as the domain fills them.
EvaluationResult = dict[str, Any]
#: One line of the trial ledger: ``seq``, ``prev``, ``stage``, ``created_at``, ``sha256`` and
#: the fields of its stage (``key``, ``decision``, ``experiment_id``, ...).
LedgerEntry = dict[str, Any]
#: A compact development-period signal (bucketed daily PnL or sampled predictions) for diversity.
Signature = list[float]


# ------------------------------------------------------------------- configuration
class CampaignLimits(TypedDict):
    """Operational budgets; raising one continues a campaign without changing its identity."""

    max_rounds: int
    proposals_per_round: int
    max_screen_trials: int
    validations_per_round: int
    max_validation_trials: int
    max_promotions: int
    population_size: int


class SelectionSettings(TypedDict):
    #: ``None`` means the domain's ``default_max_correlation``.
    max_correlation: float | None


class CampaignConfig(TypedDict):
    """A validated campaign file. Unknown top-level keys are kept (they enter the identity)."""

    name: str
    domain: str
    seed: int
    settings: JSONObject
    limits: CampaignLimits
    selection: SelectionSettings
    policy: NotRequired[JSONObject]
    research_root: NotRequired[str]
    runtime_root: NotRequired[str]
    #: Where the lineage registry lives; absent: derived from the data location.
    lineage_root: NotRequired[str]
    #: Declares every holdout read of this campaign exploratory: never confirmed, never retiring the holdout.
    non_confirmatory: NotRequired[bool]


class Manifest(TypedDict):
    """What the campaign identity is hashed from."""

    lab_version: int
    domain: str
    config: JSONObject
    identity: JSONObject
    code: dict[str, str]


class Lineage(TypedDict):
    """What makes two campaigns test the same evidence (see :meth:`Domain.lineage`)."""

    validation: JSONObject
    holdout: JSONObject


# ---------------------------------------------------------------------- domain I/O
class CandidateDescription(TypedDict):
    """Fields a domain stores on hypothesis, experiment and final records."""

    title: str
    family: str
    expression: NotRequired[str]
    generator: NotRequired[str]
    complexity: NotRequired[int | None]
    recipe: NotRequired[JSONObject]


class ScreenOutcome(TypedDict):
    """One development-only score; ``candidate`` is set when the domain re-oriented the signal."""

    status: str
    fitness: float | None
    metrics: Metrics
    signature: NotRequired[Signature]
    error: NotRequired[str]
    candidate: NotRequired[Candidate]


class Review(TypedDict):
    """Decisions: promote, reject, park, repair."""

    decision: str
    reason: str
    checks: NotRequired[JSONObject]


class Verdict(TypedDict):
    """A final assessment: confirmed, not_confirmed or repair."""

    decision: str
    reason: str
    question: NotRequired[str]
    expected_change: NotRequired[str]


class Submission(TypedDict):
    """One line of the campaign inbox."""

    candidate: Candidate
    submitted_by: str
    key: str | None
    submitted_at: str


class DomainInfo(TypedDict):
    name: str
    title: str
    summary: str


# ----------------------------------------------------------------------- checkpoint
class PopulationEntry(TypedDict):
    key: str
    candidate: Candidate
    fitness: float | None
    metrics: Metrics
    signature: Signature
    round: int
    generator: str
    validated: bool


class ValidatedEntry(TypedDict):
    """Why a key will not spend another validation trial: reviewed, or redundant with one that was."""

    decision: str
    score: float | None
    hypothesis_id: NotRequired[str]
    experiment_id: NotRequired[str]
    signature: NotRequired[Signature | None]
    diversity_group: NotRequired[str]
    max_correlation: NotRequired[float]


class AcceptedEntry(TypedDict):
    key: str
    hypothesis_id: str
    experiment_id: str
    review_id: str
    candidate: Candidate
    score: float | None
    signature: Signature | None
    generator: str
    diversity_group: str
    checks: JSONObject


class PendingValidation(TypedDict):
    """A validation trial reserved before it runs, so an interrupted one still counts."""

    key: str
    hypothesis_id: str
    experiment_id: str
    started_at: str
    candidate: Candidate
    signature: Signature | None
    generator: str


class Checkpoint(TypedDict):
    """``<runtime>/<campaign_id>.json``, rewritten atomically after every step."""

    campaign_id: str
    domain: str
    round: int
    screen_trials: int
    screen_errors: int
    validation_trials: int
    promotions: int
    seen: list[str]
    population: list[PopulationEntry]
    validated: dict[str, ValidatedEntry]
    accepted: list[AcceptedEntry]
    validation_scores: list[float]
    pending: PendingValidation | None
    round_open: int | None
    holdout_access: dict[str, int]
    holdout_accessed: bool
    generators: dict[str, dict[str, int]]
    status: str
    created_at: str


class ProposalContext(TypedDict):
    """What a domain sees when proposing a round."""

    round: int
    rng_seed: int
    n: int
    population: list[PopulationEntry]
    accepted: list[AcceptedEntry]
    submitted: list[Submission]
    seen: set[str]


class LineageEvidence(TypedDict):
    """Validation trials, their scores and holdout accesses of other campaigns on the same data.

    Counted from the record store and from the lineage registry beside the data; ``agree``
    says whether the two see the same campaigns with the same trial counts.
    """

    trials: int
    scores: list[float]
    accesses: dict[str, int]
    registry: str
    agree: bool
    store_trials: int
    registry_trials: int
    #: Campaigns the registry knows but this record store does not (another ``research_root``).
    foreign_campaigns: list[JSONObject]
    #: Campaigns this record store knows but the registry does not (a registry that was reset).
    unregistered_campaigns: list[str]
    #: The registry's retirement record once a confirmatory batch read this lineage's holdout.
    retired: JSONObject | None
    #: The holdout batch currently open on this lineage (declared before its first access), if any.
    batch: JSONObject | None
    #: The first read of this lineage's holdout outside a confirmatory batch (``at``, ``campaign_id``, ``key``), if any.
    exposed: JSONObject | None
    #: The registry revision this evidence was read at; a batch is declared against it.
    revision: int
    #: Whether the registry has any entry for the holdout lineage (store-side batch evidence counts only when not).
    registry_known: bool


class TrialContext(TypedDict):
    """Trial counts a review prices selection with; ``accepted`` and ``signature`` are added for reviews."""

    n_validation_trials: int
    campaign_validation_trials: int
    lineage_validation_trials: int
    validation_scores: list[float]
    policy: JSONObject
    accepted: NotRequired[list[AcceptedEntry]]
    signature: NotRequired[Signature | None]


# -------------------------------------------------------------------------- records
class CampaignRecord(TypedDict):
    id: str
    name: str
    lab_version: int
    domain: str
    config: JSONObject
    identity: JSONObject
    code: dict[str, str]
    created_at: str


class HypothesisRecord(TypedDict):
    id: str
    campaign_id: str
    domain: str
    key: str
    generator: str
    parents: list[str]
    round: int
    screen: Metrics
    screen_fitness: float | None
    post_holdout: bool
    validation_lineage: str
    title: str
    family: str
    expression: NotRequired[str]
    complexity: NotRequired[int | None]
    recipe: NotRequired[JSONObject]
    created_at: str


class ExperimentRecord(TypedDict):
    """The description fields are absent when a crashed validation is published without its candidate."""

    id: str
    hypothesis_id: str
    campaign_id: str
    experiment_key: str
    domain: str
    validation_lineage: str
    result: EvaluationResult
    created_at: str
    title: NotRequired[str]
    family: NotRequired[str]
    expression: NotRequired[str]
    generator: NotRequired[str]
    complexity: NotRequired[int | None]
    recipe: NotRequired[JSONObject]


class ReviewRecord(TypedDict):
    id: str
    campaign_id: str
    hypothesis_id: str
    experiment_id: str
    decision: str
    reason: str
    checks: JSONObject
    question: str
    expected_change: str
    created_at: str


class FinalRecord(TypedDict):
    id: str
    campaign_id: str
    domain: str
    hypothesis_id: str
    experiment_id: str
    experiment_key: str
    title: str
    family: str
    expression: NotRequired[str]
    generator: NotRequired[str]
    complexity: NotRequired[int | None]
    recipe: NotRequired[JSONObject]
    access_sequence: int
    reused: bool
    post_holdout: bool
    holdout_lineage: str
    result: EvaluationResult
    decision: str
    reason: str
    created_at: str
    #: How the decision was reached: ``{"method": "batch-holm", "p_value", "holm_level", "holm_rejected", ...}``
    #: or ``{"method": "per-candidate"}``.
    confirmation: NotRequired[JSONObject]
    #: The read was declared non-confirmatory (registry disagreement, retired holdout or an explicit flag).
    non_confirmatory: NotRequired[bool]
    #: Whether the lineage registry used was this machine's canonical one (``<data root>/autoresearch/lineage``).
    registry_canonical: NotRequired[bool]
    #: The confirmatory batch this final belongs to, if any.
    batch_id: NotRequired[str | None]
    #: The registry's entry for this lineage was seeded from a record store, so its baseline is that store's word.
    lineage_seeded: NotRequired[bool]
    #: The lineage's holdout had been read outside a confirmatory batch before this read.
    lineage_exposed: NotRequired[bool]
    #: ``verify_registry`` at finalize: ``ok``, ``no_snapshot``, ``registry_reset`` or ``registry_shrunk``.
    registry_status: NotRequired[str]
    #: The lineage-wide validation count the batch's finalists were priced at when the batch was declared.
    trials_at_declaration: NotRequired[int]


# ------------------------------------------------------------------------- outcomes
class Decision(TypedDict):
    """What one validation trial decided."""

    key: str
    hypothesis_id: str
    decision: str
    reason: str
    score: float | None


class RoundSummary(TypedDict):
    round: int
    proposed: int
    screened: int
    screen_ok: int
    submitted: int
    validated: int
    promoted: int
    best_fitness: float | None


class CampaignComplete(TypedDict):
    """A tick that ran nothing because a budget was reached."""

    campaign_id: str
    status: Literal["complete"]
    reason: str


class RoundResult(TypedDict):
    """A tick that ran one round: the round summary and every validation decision."""

    campaign_id: str
    status: Literal["round"]
    round: int
    proposed: int
    screened: int
    screen_ok: int
    submitted: int
    validated: int
    promoted: int
    best_fitness: float | None
    decisions: list[Decision]


#: What :meth:`LabCampaign.tick` returns; ``status`` tells the two apart.
RoundOutcome = CampaignComplete | RoundResult


class FinalOutcome(TypedDict):
    final_id: str
    key: str
    hypothesis_id: str
    decision: str
    reason: str
    reused: bool
    post_holdout: bool
    holdout: Metrics
    confirmation: NotRequired[JSONObject]
    non_confirmatory: NotRequired[bool]
    registry_canonical: NotRequired[bool]
    batch_id: NotRequired[str | None]
    lineage_seeded: NotRequired[bool]
    lineage_exposed: NotRequired[bool]
    registry_status: NotRequired[str]
    trials_at_declaration: NotRequired[int]


class CampaignStatus(TypedDict):
    campaign_id: str
    domain: str
    name: str
    status: str
    round: int
    screen_trials: int
    screen_errors: int
    validation_trials: int
    promotions: int
    population: int
    best_fitness: float | None
    budget: str | None
    limits: CampaignLimits
    holdout_accessed: bool
    generators: dict[str, dict[str, int]]


# -------------------------------------------------------------------------- reports
class LedgerStatus(TypedDict):
    """:meth:`TrialLedger.verify`: ``ok`` with the chain summary, or ``ok: False`` with the problem."""

    ok: bool
    problem: NotRequired[str]
    entries: NotRequired[int]
    head: NotRequired[str]
    stages: NotRequired[dict[str, int]]
    torn_tail_bytes: NotRequired[int]


class Funnel(TypedDict):
    screened: int
    screen_errors: int
    redundant: int
    validated: int
    promoted: int
    finalized: int
    confirmed: int


class FinalSummary(TypedDict):
    decision: str
    reused: bool
    holdout: dict[str, float]


class LeaderboardRow(TypedDict):
    hypothesis_id: str
    title: str
    family: str
    generator: str
    round: int | None
    screen_fitness: float | None
    decision: str | None
    reason: str
    outcome: str | None
    post_holdout: bool
    #: Named validation numbers (alpha: days, sharpe, ...; ml: score, baseline, ..., metric).
    validation: dict[str, Any]
    final: FinalSummary | None
    deflated_sharpe: NotRequired[float | None]
    expression: NotRequired[str]
    recipe: NotRequired[JSONObject]


class CampaignReport(TypedDict):
    campaign_id: str
    name: str
    #: ``None`` only for a stored campaign record that lacks the field.
    domain: str | None
    status: str
    round: int
    budget: str | None
    #: The configured limits, or ``{}`` when the report is built from records alone.
    limits: CampaignLimits | dict[str, int]
    funnel: Funnel
    generators: dict[str, dict[str, int]]
    leaderboard: list[LeaderboardRow]
    ledger: LedgerStatus
    holdout_accessed: bool
    #: Which lineage registry the campaign used and whether its record store agrees with it.
    lineage: NotRequired[JSONObject]


class CampaignListing(TypedDict):
    campaign_id: str
    name: str
    domain: str | None
    created_at: str | None


class StoreReport(TypedDict):
    campaigns: list[CampaignListing]
    campaign: CampaignReport | None
