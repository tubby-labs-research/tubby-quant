"""The contract every research domain implements, and the registry of domains.

A domain turns the harness's generic loop into research on something concrete.
The harness owns budgets, checkpoints, the trial ledger, immutable records,
diversity screening, holdout accounting and reports; the domain owns what a
candidate is, how it is generated, how it is scored on each sealed period and
what its gates are.

Lifecycle of one candidate::

    propose ─► key (dedupe) ─► screen (development only) ─► population
            ─► shortlist (fitness, gates, diversity) ─► validate (validation period)
            ─► review (gates, robustness, multiple testing) ─► promote
            ─► finalize (reserved holdout, once, explicit operator action)

Sealing is part of the contract. ``screen`` may read development data only,
``validate`` development and validation, and only ``evaluate_holdout`` the
reserved period. Implementations should make that structural (load sliced
data) rather than a convention.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

from .types import (
    AcceptedEntry,
    Candidate,
    CandidateDescription,
    DomainInfo,
    EvaluationResult,
    JSONObject,
    LeaderboardRow,
    Lineage,
    PopulationEntry,
    ProposalContext,
    Review,
    ScreenOutcome,
    TrialContext,
    Verdict,
)

REGISTRY: dict[str, tuple[str, str]] = {
    "alpha": ("autoresearch.domains.alpha.domain", "AlphaDiscoveryDomain"),
    "ml": ("autoresearch.domains.ml.domain", "MLDomain"),
}


class Domain(ABC):
    """Base class; subclasses set ``name``, ``title`` and ``summary``."""

    name = "abstract"
    title = ""
    summary = ""
    #: Python packages whose source is hashed into every campaign identity.
    code_packages: tuple[str, ...] = ()
    #: Signature correlation at or above which a candidate is redundant, unless the campaign sets one.
    default_max_correlation = 0.7

    def __init__(self, config: JSONObject, runtime: Path, policy: JSONObject | None = None) -> None:
        self.config = config
        self.runtime = Path(runtime)
        self.policy_config: JSONObject = dict(policy or {})

    # ---------------------------------------------------------------- identity
    @abstractmethod
    def identity(self) -> JSONObject:
        """Data, code and settings identity. Must not depend on the holdout bounds."""

    def lineage(self) -> Lineage:
        """What makes two campaigns test the same evidence.

        ``validation``: the data and periods every validation trial looks at;
        ``holdout``: the data and reserved period a final reads. Campaigns in
        one record store with the same lineage share their validation-trial
        count and their holdout accesses, whatever their gates, seeds or code.
        The default (the full identity) is safe but narrow; domains should name
        only the data, universe and periods.
        """
        identity = self.identity()
        return {"validation": identity, "holdout": identity}

    def public_config(self) -> JSONObject:
        """The domain configuration as it enters the campaign identity (no local paths)."""
        return dict(self.config)

    # -------------------------------------------------------------- generation
    @abstractmethod
    def propose(self, context: ProposalContext) -> list[Candidate]:
        """New candidates. ``context`` carries round, rng_seed, n, population, accepted and submitted."""

    @abstractmethod
    def key(self, candidate: Candidate) -> str:
        """Canonical identity used to deduplicate candidates across the whole campaign."""

    def describe_candidate(self, candidate: Candidate) -> CandidateDescription:
        """Fields stored on the hypothesis record (title, family, expression or recipe)."""
        return {"title": candidate.get("title", ""), "family": candidate.get("family", "")}

    def restore_candidate(self, description: JSONObject) -> Candidate:
        """A candidate rebuilt from a stored hypothesis record, when the runtime checkpoint is lost.

        The default returns what :meth:`describe_candidate` stored (expression, generator, family,
        title, complexity, parents) and unpacks a recipe into ``family``/``params``, which is what
        the shipped domains need to score a holdout.
        """
        candidate: Candidate = {key: description[key]
                                for key in ("expression", "generator", "family", "title", "complexity", "parents")
                                if key in description}
        recipe = description.get("recipe")
        if isinstance(recipe, dict):
            candidate["family"] = recipe.get("family", candidate.get("family"))
            candidate["params"] = dict(recipe.get("params") or {})
        return candidate

    # ----------------------------------------------------------------- scoring
    @abstractmethod
    def screen(self, candidates: list[Candidate]) -> list[ScreenOutcome]:
        """Development-only scores: ``{"status", "fitness", "metrics", "signature", "error"}`` per candidate."""

    def shortlist_ok(self, screened: PopulationEntry) -> bool:
        """Whether a population entry (key, candidate, fitness, metrics, signature) may spend a validation trial."""
        return screened.get("fitness") is not None

    @abstractmethod
    def validate(self, candidate: Candidate) -> EvaluationResult:
        """Development and validation evaluation in the shared result format."""

    @abstractmethod
    def selection_score(self, result: EvaluationResult) -> float | None:
        """The validation statistic whose cross-trial distribution prices selection."""

    @abstractmethod
    def review(self, candidate: Candidate, result: EvaluationResult, context: TrialContext) -> Review:
        """``{"decision", "reason", "checks"}``; decisions: promote, reject, park, repair."""

    def final_check(self, entry: AcceptedEntry, context: TrialContext) -> JSONObject:
        """Whether one promoted candidate still qualifies for the holdout at the final trial count.

        ``{"eligible": bool, ...}`` with the recomputed selection statistic beside the verdict, so a
        refused finalization is recorded with its numbers. The default keeps every promotion; a
        domain overrides this (per candidate) or ``final_selection`` (set-valued).
        """
        return {"eligible": True}

    def final_selection(self, accepted: list[AcceptedEntry], context: TrialContext) -> list[AcceptedEntry]:
        """Accepted candidates that still qualify for the holdout once the campaign's trial count is final."""
        return [entry for entry in accepted if self.final_check(entry, context).get("eligible", True)]

    # ----------------------------------------------------------------- holdout
    def evaluate_holdout(self, candidate: Candidate) -> EvaluationResult:
        raise ValueError(f"domain {self.name} has no holdout evaluation")

    def assess_final(self, result: EvaluationResult) -> Verdict:
        raise ValueError(f"domain {self.name} has no final assessment")

    # ----------------------------------------------------------------- reports
    def export(self, rows: list[LeaderboardRow], output: Path) -> Path:
        raise ValueError(f"domain {self.name} has no export format")

    # ---------------------------------------------------------------- lifetime
    def release(self) -> None:
        """Free cached data and workspaces at the end of a run; the base domain holds none."""


def load_domain(name: str) -> type[Domain]:
    """The Domain subclass registered under ``name``."""
    if name not in REGISTRY:
        raise ValueError(f"unknown research domain {name!r}; available: {', '.join(sorted(REGISTRY))}")
    module, attribute = REGISTRY[name]
    # The registry names Domain subclasses; importlib cannot tell the checker that.
    return cast(type[Domain], getattr(importlib.import_module(module), attribute))


def catalog() -> list[DomainInfo]:
    out: list[DomainInfo] = []
    for name in sorted(REGISTRY):
        cls = load_domain(name)
        out.append({"name": name, "title": cls.title, "summary": cls.summary})
    return out
