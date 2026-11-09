"""Autonomous alpha discovery: typed FASTEXPR search under a sealed, selection-aware cascade.

Generators (mixed every round):

* ``template`` — mechanism families (reversal, momentum, volatility, volume,
  price-volume, intraday range, liquidity) with sampled windows;
* ``random`` — unit-checked random trees;
* ``mutation`` and ``crossover`` — genetic edits of the fittest screened trees;
* ``seed`` — configured expressions and promoted ideas from Research campaigns;
* ``inbox`` — candidates submitted by an agent (MCP) or a person;
* ``ensemble`` — the equal-weight rank blend of mutually uncorrelated development
  leaders, chosen on development data only, so its validation is a fair test;
* ``combination`` — the equal-weight rank blend of the promoted set. Its parts
  were selected on validation, so only its holdout is independent evidence.

Stages:

1. **Screen** on the development period only, in a workspace whose prices end
   at the development end. Each signal is oriented (an expression and its
   negation are one candidate) and scored by Alpha Lab fitness minus a
   complexity penalty. A compact development PnL signature feeds diversity.
2. **Validate** through the canonical :class:`autoresearch.backend.AlphaLabBackend`
   (development and validation as separate runs, the same records Research
   writes), plus veto-only robustness probes on a workspace that ends at the
   validation end: costs multiplied, the two validation halves, and every
   window scaled down and up.
3. **Review**: the deterministic Research gates (days, Sharpe, returns,
   turnover, drawdown), then the deflated Sharpe ratio over every validation
   trial of the campaign, then the robustness vetoes.
4. **Final selection** recomputes the deflated Sharpe ratio with the
   campaign's final trial count before anything reaches the holdout.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from autoresearch.lab.domain import Domain
from autoresearch.lab.selection import DeflatedSharpe, deflated_sharpe, moments
from autoresearch.lab.types import (
    AcceptedEntry,
    Candidate,
    CandidateDescription,
    EvaluationResult,
    JSONObject,
    Lineage,
    PopulationEntry,
    ProposalContext,
    Review,
    ScreenOutcome,
    TrialContext,
    Verdict,
)
from autoresearch.policy import assess, assess_final

from . import grammar
from .evaluator import SealedEvaluator, bucketed, daily_profile, require_plain_snapshot, sharpe_of

REQUIRED = ("dataset", "universe", "development_start", "development_end", "validation_start",
            "validation_end", "cost_bps")
BACKEND_KEYS = REQUIRED + ("holdout_start", "warmup", "decay", "neutralization", "truncation",
                           "pasteurization", "booksize")
SEARCH_DEFAULTS: dict[str, Any] = {
    "fields": ["open", "high", "low", "close", "vwap", "volume", "returns", "adv20"],
    "windows": [2, 3, 5, 10, 20, 40, 60],
    "max_depth": 4, "max_size": 14,
    "mix": {"template": 0.2, "random": 0.25, "mutation": 0.4, "crossover": 0.15},
    "seeds": [], "seed_research_root": None,
    "complexity_penalty": 0.02, "tournament": 4, "folds": 3,
    "ensemble_size": 5, "ensemble_max_correlation": 0.5,
}
POLICY_DEFAULTS: dict[str, Any] = {
    "min_dev_sharpe": 0.75, "min_validation_days": 60, "min_sharpe": 1.0, "min_returns": 0.0,
    "max_turnover": 0.7, "max_drawdown": 0.3, "min_refinement_improvement": 0.05,
    "min_dsr": 0.95, "dsr_variance": "null", "cost_stress": 2.0, "min_cost_stress_retention": 0.35,
    "min_half_sharpe": 0.0, "min_perturbation_retention": 0.5,
    "min_holdout_days": 20, "min_holdout_sharpe": 0.0,
}
PERTURBATION_FACTORS = (0.75, 1.33)


class AlphaDiscoveryDomain(Domain):
    name = "alpha"
    title = "Autonomous alpha discovery"
    summary = ("Typed FASTEXPR search (templates, random trees, mutation, crossover, agent submissions) "
               "screened on sealed development data, validated by Alpha Lab, priced by the deflated Sharpe "
               "ratio and vetoed by cost, sub-period and parameter robustness.")
    code_packages = ("autoresearch.domains.alpha",)

    def __init__(self, config: JSONObject, runtime: Path, policy: JSONObject | None = None) -> None:
        super().__init__(config, runtime, policy)
        from autoresearch.backend import AlphaLabBackend

        if not isinstance(config, dict):
            raise ValueError("alpha settings must be an object")
        missing = [key for key in REQUIRED if key not in config]
        if missing:
            raise ValueError("alpha settings require " + ", ".join(missing))
        search = {**SEARCH_DEFAULTS, **config.get("search", {})}
        unknown = set(search) - set(SEARCH_DEFAULTS)
        unknown |= set(config) - set(BACKEND_KEYS) - {"search"}
        if unknown:
            raise ValueError("unknown alpha settings: " + ", ".join(sorted(unknown)))
        mix = {**SEARCH_DEFAULTS["mix"], **search["mix"]}
        if set(mix) != set(SEARCH_DEFAULTS["mix"]) or any(not isinstance(v, (int, float)) or v < 0 for v in mix.values()) \
                or sum(mix.values()) <= 0:
            raise ValueError("search.mix must give nonnegative weights for template, random, mutation, crossover")
        search["mix"] = mix
        fields = list(search["fields"])
        if not fields or any(f not in grammar.FIELD_UNITS for f in fields):
            raise ValueError("search.fields must be drawn from " + ", ".join(grammar.FIELD_UNITS))
        windows = [int(w) for w in search["windows"]]
        if not windows or any(w < 1 for w in windows):
            raise ValueError("search.windows must be positive integers")
        self.search = search
        # Before anything verifies the dataset: verifying a packed snapshot caches the whole archive in the process
        # (qlib_yahoo/packed.py), reserved sessions included, so the refusal has to come first.
        require_plain_snapshot(Path(config["dataset"]))
        self.backend = AlphaLabBackend({key: config[key] for key in BACKEND_KEYS if key in config},
                                       self.runtime / "engine")
        self.settings = self.backend.settings
        self.bounds = dict(self.backend.bounds)
        self.space = grammar.Space(fields=tuple(fields), windows=tuple(windows), max_depth=int(search["max_depth"]),
                                   max_size=int(search["max_size"]),
                                   max_lookback=max(2, self.settings.warmup - 2))
        self.evaluator = SealedEvaluator(self.backend.dataset, self.settings, self.bounds)
        self.gates = self.policy(self.policy_config)
        self._identity: JSONObject | None = None

    # ---------------------------------------------------------------- identity
    def identity(self) -> JSONObject:
        if self._identity is None:
            self._identity = self.backend.identity()
        return self._identity

    def lineage(self) -> Lineage:
        data = {"data": self.identity()["data_identity"], "universe": self.settings.universe}
        return {"validation": {**data, **self.bounds},
                "holdout": {**data, "holdout_start": self.config.get("holdout_start")}}

    def public_config(self) -> JSONObject:
        value = {key: item for key, item in self.config.items() if key not in ("dataset", "holdout_start")}
        value["search"] = {**self.search}
        if value["search"].get("seed_research_root"):
            value["search"]["seed_research_root"] = "configured"
        return value

    def policy(self, context_policy: JSONObject | None) -> JSONObject:
        supplied = context_policy or {}
        unknown = set(supplied) - set(POLICY_DEFAULTS)
        if unknown:
            raise ValueError("unknown alpha policy keys: " + ", ".join(sorted(unknown)))
        merged = {**POLICY_DEFAULTS, **supplied}
        if merged["dsr_variance"] not in ("null", "empirical"):
            raise ValueError("dsr_variance must be null or empirical")
        return merged

    # -------------------------------------------------------------- candidates
    def _candidate(self, node: grammar.Node, generator: str, *, title: str = "", parents: Sequence[str] = (),
                   note: str = "") -> Candidate:
        return {"expression": grammar.to_text(node), "generator": generator, "parents": list(parents),
                "family": grammar.family(node), "title": title or grammar.family(node),
                "complexity": grammar.size(node), "note": note}

    def key(self, candidate: Candidate) -> str:
        expression = candidate.get("expression")
        if not isinstance(expression, str) or not expression.strip() or len(expression) > 4096:
            raise ValueError("candidate needs a FASTEXPR expression of at most 4096 characters")
        try:
            node = grammar.from_text(expression)
        except ValueError:
            from autoresearch.records import canonical_expression, stable_id
            return stable_id("alpha", {"opaque": canonical_expression(expression)})
        return grammar.canonical_key(node)

    def describe_candidate(self, candidate: Candidate) -> CandidateDescription:
        return {"title": candidate.get("title", ""), "family": candidate.get("family", ""),
                "expression": candidate.get("expression", ""), "generator": candidate.get("generator", ""),
                "complexity": candidate.get("complexity")}

    def _seeds(self) -> list[tuple[str, str]]:
        seeds = [("configured seed", text) for text in self.search.get("seeds") or []]
        root = self.search.get("seed_research_root")
        if root:
            from autoresearch.handoff import campaign_report
            from autoresearch.records import ResearchStore
            try:
                report = campaign_report(ResearchStore(Path(root)))
                seeds += [("research: " + row.get("title", "")[:60], row["expression"])
                          for row in report["hypotheses"] if row.get("decision") in ("promote", "refine")]
            except (OSError, ValueError):
                pass
        return seeds

    def propose(self, context: ProposalContext) -> list[Candidate]:
        rng = np.random.default_rng(context["rng_seed"])
        n = int(context["n"])
        out: list[Candidate] = []
        for item in context.get("submitted", []):
            # Only the expression and the rationale are taken from a submission: complexity,
            # family and grouping are recomputed, so a submitter cannot buy priority.
            raw = item["candidate"]
            submitted_expression = raw.get("expression")
            expression = submitted_expression if isinstance(submitted_expression, str) else ""
            rationale = str(raw.get("rationale", ""))[:500]
            candidate: Candidate = {"expression": expression, "rationale": rationale, "generator": "inbox",
                                    "submitted_by": str(item.get("submitted_by", ""))[:80],
                                    "title": ("submitted: " + rationale)[:72] if rationale else "submitted"}
            try:
                node = grammar.from_text(expression)
                candidate.update(family=grammar.family(node), complexity=grammar.size(node))
            except ValueError:
                candidate.update(family="submitted", complexity=grammar.text_size(expression))
            out.append(candidate)
        if context["round"] == 1:
            for title, text in self._seeds():
                try:
                    out.append(self._candidate(grammar.from_text(text), "seed", title=title))
                except ValueError:
                    out.append({"expression": text, "generator": "seed", "title": title, "family": "seed",
                                "complexity": grammar.text_size(text)})
            for title, node in grammar.template_trees(every_window=True):
                out.append(self._candidate(node, "template", title=title))
        accepted = context.get("accepted") or []
        atomic = [a for a in accepted if a.get("generator") not in ("ensemble", "combination")]
        if len(atomic) >= 2:
            members = sorted(atomic, key=lambda a: a["key"])[: int(self.search["ensemble_size"])]
            out.append(self._blend(members, "combination", "equal-weight blend of promoted alphas"))
        # The parsed tree of every population entry whose expression parses; genetic edits need them.
        trees: dict[str, grammar.Node] = {}
        for entry in context.get("population", []):
            tree = self._tree(entry)
            if tree is not None:
                trees[entry["key"]] = tree
        population = [e for e in context.get("population", []) if e["key"] in trees]
        ensemble = self._diverse(population)
        if len(ensemble) >= 2:
            out.append(self._blend(ensemble, "ensemble", "equal-weight blend of diverse development leaders"))
        mix = dict(self.search["mix"])
        if len(population) < 2:
            mix["crossover"] = 0.0
        if not population:
            mix["mutation"] = 0.0
        names = list(mix)
        weights = np.array([mix[k] for k in names], dtype=float)
        weights = weights / weights.sum()
        attempts = 0
        while len(out) < n and attempts < n * 6:
            attempts += 1
            kind = names[int(rng.choice(len(names), p=weights))]
            if kind == "template":
                title, node = grammar.template_trees(rng)[int(rng.integers(len(grammar.TEMPLATES)))]
                out.append(self._candidate(node, "template", title=title))
            elif kind == "random":
                tree = grammar.random_tree(rng, self.space)
                if tree is not None:
                    out.append(self._candidate(tree, "random"))
            elif kind == "mutation":
                parent = self._tournament(rng, population)
                edited = grammar.mutate(rng, trees[parent["key"]], self.space)
                if edited is not None:
                    out.append(self._candidate(edited[0], "mutation", parents=[parent["key"]], note=edited[1]))
            else:
                first, second = self._tournament(rng, population), self._tournament(rng, population)
                if first["key"] == second["key"]:
                    continue
                child = grammar.crossover(rng, trees[first["key"]], trees[second["key"]], self.space)
                if child is not None:
                    out.append(self._candidate(child, "crossover", parents=[first["key"], second["key"]]))
        return out

    def _diverse(self, population: list[PopulationEntry]) -> list[PopulationEntry]:
        """Greedy, development-only choice of mutually uncorrelated leaders (no validation information)."""
        from autoresearch.lab.selection import pearson

        chosen: list[PopulationEntry] = []
        for entry in population:  # already sorted by development fitness
            if len(chosen) >= int(self.search["ensemble_size"]):
                break
            if entry["candidate"].get("generator") in ("ensemble", "combination"):
                continue
            if not self.shortlist_ok(entry):
                continue
            if all(abs(pearson(entry["signature"], other["signature"]) or 0.0) < float(self.search["ensemble_max_correlation"])
                   for other in chosen):
                chosen.append(entry)
        return chosen

    def _blend(self, members: Sequence[AcceptedEntry | PopulationEntry], generator: str, title: str) -> Candidate:
        ordered = sorted(members, key=lambda m: m["key"])
        expression = "add(" + ", ".join(_ranked(m["candidate"]["expression"]) for m in ordered) + ")"
        candidate: Candidate = {"expression": expression, "generator": generator, "title": f"{title} ({len(ordered)})",
                                "family": generator, "parents": [m["key"] for m in ordered], "diversity_group": "blend",
                                "complexity": len(ordered)}  # a blend is charged per member, not per node
        if generator == "combination":
            candidate["note"] = "components were selected on validation; only the holdout is independent evidence"
        return candidate

    def _tree(self, entry: PopulationEntry) -> grammar.Node | None:
        try:
            return grammar.from_text(entry["candidate"]["expression"])
        except (KeyError, ValueError):
            return None

    def _tournament(self, rng: np.random.Generator, population: list[PopulationEntry]) -> PopulationEntry:
        size = min(len(population), int(self.search["tournament"]))
        picks = rng.choice(len(population), size=size, replace=False)
        return min((population[int(i)] for i in picks),
                   key=lambda e: (-(e["fitness"] if e["fitness"] is not None else -math.inf), e["key"]))

    # ------------------------------------------------------------------ screen
    def screen(self, candidates: list[Candidate]) -> list[ScreenOutcome]:
        from qlib_yahoo.alphalab.engine import check_expression

        penalty = float(self.search["complexity_penalty"])
        out: list[ScreenOutcome] = []
        for candidate in candidates:
            expression = candidate["expression"]
            problems = check_expression(expression)
            if problems:
                out.append({"status": "error", "fitness": None, "metrics": {}, "error": "; ".join(problems)[:300]})
                continue
            try:
                raw = self.evaluator.simulate(expression, "development")
            except Exception as error:
                out.append({"status": "error", "fitness": None, "metrics": {},
                            "error": (type(error).__name__ + ": " + str(error))[:300]})
                continue
            summary = raw["summary"]
            oriented = candidate
            if summary.get("sharpe") is not None and summary["sharpe"] < 0:
                try:
                    node = grammar.negate(grammar.from_text(expression))
                    flipped_text = grammar.to_text(node)
                except ValueError:
                    flipped_text = f"-({expression})"
                try:
                    flipped = self.evaluator.simulate(flipped_text, "development")
                    if (flipped["summary"].get("sharpe") or -math.inf) > summary["sharpe"]:
                        raw, summary = flipped, flipped["summary"]
                        oriented = {**candidate, "expression": flipped_text, "oriented": "negated"}
                except Exception:
                    pass
            sharpe = summary.get("sharpe")
            pnl = np.asarray(raw["daily"]["pnl"], dtype=float)
            folds = [sharpe_of(chunk) for chunk in np.array_split(pnl, int(self.search["folds"]))]
            finite_folds = [f for f in folds if f is not None and math.isfinite(f)]
            if sharpe is None or len(finite_folds) != len(folds):
                out.append({"status": "error", "fitness": None, "metrics": {}, "error": "undefined development statistics"})
                continue
            complexity = int(candidate.get("complexity") or 1)
            metrics: dict[str, Any] = {key: summary.get(key) for key in ("days", "sharpe", "returns", "turnover", "fitness",
                                                                         "drawdown", "coverage", "margin_bps")}
            metrics["fold_sharpes"] = folds
            # The weakest development fold, not the full-period peak: consistency across
            # sub-periods is what generalizes, and a single lucky stretch cannot carry it.
            out.append({"status": "ok", "fitness": float(min(finite_folds)) - penalty * complexity, "metrics": metrics,
                        "signature": bucketed(pnl), "candidate": oriented})
        return out

    def shortlist_ok(self, screened: PopulationEntry) -> bool:
        metrics = screened.get("metrics") or {}
        sharpe, folds = metrics.get("sharpe"), metrics.get("fold_sharpes") or []
        return (sharpe is not None and sharpe >= self.gates["min_dev_sharpe"]
                and bool(folds) and all(f is not None and f > 0 for f in folds))

    # ---------------------------------------------------------------- validate
    def validate(self, candidate: Candidate) -> EvaluationResult:
        expression = candidate["expression"]
        result = self.backend.evaluate(expression)
        if result.get("status") != "ok":
            return result
        policy = self.gates
        robustness: dict[str, Any] = {}
        base = self.evaluator.simulate(expression, "validation")
        pnl = np.asarray(base["daily"]["pnl"], dtype=float)
        robustness.update(daily_profile(pnl))
        half = len(pnl) // 2
        robustness["half_sharpes"] = [sharpe_of(pnl[:half]), sharpe_of(pnl[half:])]
        stressed = self.evaluator.simulate(expression, "validation",
                                           cost_bps=self.settings.cost_bps * float(policy["cost_stress"]))
        robustness["cost_stress_sharpe"] = stressed["summary"].get("sharpe")
        robustness["cost_stress_multiplier"] = float(policy["cost_stress"])
        # Window perturbation is a probe only when it changes the expression: an unparsable
        # expression or one without a scalable window is recorded as having no probe, so the
        # review cannot read an empty list as a pass.
        perturbed: list[dict[str, Any]] = []
        status = "ok"
        node: grammar.Node | None
        try:
            node = grammar.from_text(expression)
        except ValueError:
            node, status = None, "unparsable"
        if node is not None:
            for factor in PERTURBATION_FACTORS:
                scaled = _scale_windows(node, factor, self.space.max_lookback)
                if scaled != node:
                    perturbed.append({"factor": factor,
                                      "sharpe": self.evaluator.simulate(grammar.to_text(scaled), "validation")["summary"].get("sharpe")})
            if not perturbed:
                status = "no_window"
        robustness["perturbations"] = perturbed
        robustness["perturbation"] = {"kind": "window", "status": status}
        result["robustness"] = robustness
        return result

    def selection_score(self, result: EvaluationResult) -> float | None:
        value = (result.get("robustness") or {}).get("daily_sharpe")
        return float(value) if value is not None and math.isfinite(value) else None

    # ------------------------------------------------------------------ review
    def review(self, candidate: Candidate, result: EvaluationResult, context: TrialContext) -> Review:
        note = None
        if candidate.get("generator") == "combination":
            note = "Its components were selected on validation, so only its holdout is independent evidence."
        return review_signal(result, context, self.policy(context.get("policy")), note=note)

    def final_check(self, entry: AcceptedEntry, context: TrialContext) -> JSONObject:
        policy = self.policy(context.get("policy"))
        return deflated_check(entry, context, policy["min_dsr"], policy["dsr_variance"])

    # ----------------------------------------------------------------- holdout
    def evaluate_holdout(self, candidate: Candidate) -> EvaluationResult:
        """Score the reserved period once, on a panel opened for this call alone.

        The panel comes from :func:`autoresearch.domains.alpha.holdout.open_holdout_panel`, the one
        loader of reserved sessions, and is dropped when this returns; the sealed evaluator's caches
        never see it. The same simulator scores it as the canonical batch runner would, with the
        result recorded exactly as before (``finals`` artifact, holdout settings, engine identity).
        """
        from .holdout import open_holdout_panel

        start, _end = self.backend.holdout_bounds()
        panel = open_holdout_panel(self.backend.dataset, holdout_start=start, warmup=self.settings.warmup)
        result = self.backend.evaluate_holdout(candidate["expression"], panel=panel)
        if result.get("status") == "ok":
            # The holdout's daily PnL lives in the backend's finals artifact; its per-day profile is what
            # a batch confirmation prices with the probabilistic Sharpe ratio over the holdout sessions.
            for relative in result.get("artifacts") or []:
                try:
                    stored = json.loads((self.backend.runtime_root / relative).read_text(encoding="utf-8"))
                    pnl = np.asarray(stored["holdout"]["daily"]["pnl"], dtype=float)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                result["holdout_profile"] = daily_profile(pnl)
                break
        return result

    def assess_final(self, result: EvaluationResult) -> Verdict:
        policy = self.gates
        return assess_final(result, {key: policy[key] for key in ("min_holdout_days", "min_holdout_sharpe")})

    # ---------------------------------------------------------------- lifetime
    def release(self) -> None:
        self.evaluator.release()


def review_signal(result: EvaluationResult, context: TrialContext, policy: JSONObject,
                  note: str | None = None) -> Review:
    """Research gates, then the deflated Sharpe ratio, then veto-only robustness checks.

    Shared by every domain whose candidates are daily long/short signals scored
    by Alpha Lab (FASTEXPR alphas and machine-learning predictions alike).
    """
    gates = {key: policy[key] for key in ("min_validation_days", "min_sharpe", "min_returns", "max_turnover",
                                          "max_drawdown", "min_refinement_improvement")}
    base = assess(result, gates)
    checks: dict[str, Any] = {"gates": base["decision"]}
    if base["decision"] != "promote":
        decision = "reject" if base["decision"] == "refine" else base["decision"]
        return {"decision": decision, "reason": base["reason"], "checks": checks}
    robustness = result.get("robustness") or {}
    dsr = _deflated(robustness, context.get("n_validation_trials", 1), context.get("validation_scores", []),
                    policy.get("dsr_variance", "null"))
    checks["deflated_sharpe"] = dsr
    failures: list[str] = []
    if dsr is None or dsr["dsr"] < policy["min_dsr"]:
        failures.append(f"deflated Sharpe ratio {0 if dsr is None else dsr['dsr']:.3f} below {policy['min_dsr']}"
                        f" after {context.get('n_validation_trials', 1)} validation trials")
    base_sharpe = ((result.get("metrics") or {}).get("validation") or {}).get("sharpe")
    # Stressed costs must leave a stated fraction of the validation Sharpe, not merely a positive one.
    stress = robustness.get("cost_stress_sharpe")
    checks["cost_stress_sharpe"] = stress
    stress_retention: float | None = None if stress is None or not base_sharpe or base_sharpe <= 0 \
        else stress / base_sharpe
    checks["cost_stress_retention"] = stress_retention
    min_stress = policy.get("min_cost_stress_retention", POLICY_DEFAULTS["min_cost_stress_retention"])
    if stress is None or stress <= 0 or stress_retention is None or stress_retention < min_stress:
        failures.append(f"Sharpe at {policy['cost_stress']}x costs is "
                        + ("unavailable" if stress is None else f"{stress:.2f}")
                        + f", below {min_stress:.0%} of the validation Sharpe"
                        + (f" {base_sharpe:.2f}" if base_sharpe else ""))
    halves = robustness.get("half_sharpes") or []
    checks["half_sharpes"] = halves
    if len(halves) != 2 or any(h is None or h <= policy["min_half_sharpe"] for h in halves):
        failures.append("a validation half is not above the configured Sharpe floor")
    # Parameter perturbation (windows for an expression, a recipe's numeric parameters for a
    # model) is a probe only when something was actually perturbed. An empty list is not a pass:
    # the candidate's robustness is unavailable and it is parked, not promoted.
    retention = None
    perturbations = robustness.get("perturbations") or []
    probe: dict[str, Any] = robustness.get("perturbation") or {}
    if perturbations and base_sharpe:
        values = [item.get("sharpe") for item in perturbations]
        retention = None if any(v is None for v in values) else min(values) / base_sharpe
        checks["robustness"] = "probed"
        if retention is None or retention < policy["min_perturbation_retention"]:
            failures.append(f"{probe.get('kind', 'parameter')} perturbations lose too much validation Sharpe")
    else:
        checks["robustness"] = "unavailable"
    checks["perturbation_retention"] = retention
    if failures:
        return {"decision": "reject", "reason": "Robustness or selection checks failed: " + "; ".join(failures),
                "checks": checks}
    if checks["robustness"] == "unavailable":
        why = {"unparsable": "the expression could not be parsed for window scaling",
               "no_window": "the expression has no scalable window",
               "no_parameter": "the recipe has no numeric parameter to perturb"}.get(
            str(probe.get("status") or ""), "no perturbation probe applies to this candidate")
        return {"decision": "park", "checks": checks,
                "reason": "Robustness unavailable: " + why + ". The gates and the deflated Sharpe ratio passed, "
                          "but a candidate cannot be called robust on a vacuous perturbation check; it is parked, "
                          "not promoted."}
    reason = "Validation gates, deflated Sharpe ratio and robustness vetoes passed; eligible for deeper research only."
    if note:
        reason += " " + note
    return {"decision": "promote", "checks": checks, "reason": reason}


def deflated_check(entry: AcceptedEntry, context: TrialContext, min_dsr: float, variance: str = "null") -> JSONObject:
    """One promotion's deflated Sharpe ratio recomputed at the final trial count; eligible when it still clears ``min_dsr``."""
    stored = (entry.get("checks") or {}).get("deflated_sharpe") or {}
    profile = {"daily_sharpe": stored.get("sharpe"), "days": stored.get("observations"),
               "skew": stored.get("skew", 0.0), "kurtosis": stored.get("kurtosis", 3.0)}
    again = _deflated(profile, context["n_validation_trials"], context["validation_scores"], variance)
    dsr = None if again is None else again["dsr"]
    return {"eligible": dsr is not None and dsr >= min_dsr, "statistic": "deflated_sharpe", "dsr": dsr,
            "min_dsr": min_dsr, "benchmark_sharpe": None if again is None else again["benchmark_sharpe"],
            "dsr_at_review": stored.get("dsr"), "trials_at_review": stored.get("n_trials")}


def _ranked(expression: str) -> str:
    """A member's cross-sectional rank for blending; an already ranked member is used as written."""
    try:
        base, _sign = grammar.strip_sign(grammar.from_text(expression))
        if base.op == "rank":
            return expression
    except ValueError:
        pass
    return f"rank({expression})"


def _deflated(profile: dict[str, Any], n_trials: int, scores: Sequence[float | None],
              mode: str = "null") -> DeflatedSharpe | None:
    """Deflated Sharpe ratio of one candidate given every validation trial of the campaign.

    ``mode`` "null" benchmarks against the best of ``n_trials`` draws under the global null
    (Sharpe estimates with variance 1/T): a familywise test comparable to Bonferroni.
    "empirical" uses the observed cross-trial variance as in Bailey and Lopez de Prado, which
    is stricter when the trials include genuinely different signals.
    """
    sharpe, days = profile.get("daily_sharpe"), profile.get("days")
    if sharpe is None or not days or days < 2:
        return None
    finite = [float(s) for s in scores if s is not None and math.isfinite(float(s))]
    variance = None
    if mode == "empirical" and len(finite) >= 2:
        spread = moments(finite)["std"]
        if spread:
            variance = spread ** 2
    skew, kurtosis = float(profile.get("skew") or 0.0), float(profile.get("kurtosis") or 3.0)
    value = deflated_sharpe(float(sharpe), int(days), max(1, int(n_trials)), variance, skew, kurtosis)
    value["skew"] = skew
    value["kurtosis"] = kurtosis
    return value


def _scale_windows(node: grammar.Node, factor: float, ceiling: int) -> grammar.Node:
    args = tuple(_scale_windows(a, factor, ceiling) for a in node.args)
    value = node.value
    if node.op in grammar.TS_SAME + grammar.TS_SCORE + grammar.PAIR_TS:
        value = int(min(ceiling, max(2, round(node.value * factor))))
    return grammar.Node(node.op, args, value)
