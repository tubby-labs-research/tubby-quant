"""Machine-learning research domain: training recipes as hypotheses, bounded training runs as experiments.

The same harness that searches FASTEXPR alphas searches model recipes here.
A candidate is ``{"family", "params"}`` for one of the dependency-free model
families (ridge, logistic, histogram gradient boosting, random forest, MLP).

* **Screen** (cheap): train on a subsample of the development split's first
  75% and score its last 25% (chronological when the task is). Fitness is the
  negated loss for tabular tasks and the mean daily rank information
  coefficient for panel tasks. Predictions on a fixed slice of those rows are
  the diversity signature, so near-identical models do not spend validation
  trials.
* **Validate**: train on the whole development split and score validation.
  Tabular tasks compare against a constant baseline with a paired bootstrap
  p-value (rows resampled independently, or in moving blocks of about T^(1/3)
  rows when the task is chronological; the empirical p-value is floored at
  1/(resamples + 1), never extrapolated); panel predictions become an Alpha Lab
  signal (injected as a data field) with the same Sharpe, cost, sub-period,
  deflated-Sharpe and perturbation review as discovered alphas, the recipe's
  numeric parameters scaled in place of an expression's windows.
* **Review**: tabular promotion requires the improvement's p-value to pass a
  Bonferroni level over every validation trial of the campaign and a minimum
  relative improvement; final selection re-applies it with the final count.
* **Holdout**: the recipe is refit on development plus validation and scored
  once on the reserved rows or sessions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from autoresearch.lab.domain import Domain
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
from autoresearch.policy import assess_final as assess_signal_final

from . import search
from .models import Model, make_model
from .tasks import PanelTask, TabularTask

SEARCH_DEFAULTS: dict[str, Any] = {"families": None, "mix": {"random": 0.45, "mutation": 0.55}, "screen_fraction": 0.5,
                                   "screen_rows": 60_000, "tournament": 3, "signature_rows": 256}
# bootstrap: the smallest attainable p-value is 1 / (bootstrap + 1), so 2000 resamples keep the
# Bonferroni level alpha_level / trials attainable up to 100 lineage-wide validation trials.
TABULAR_POLICY: dict[str, float] = {"alpha_level": 0.05, "min_relative_improvement": 0.02, "bootstrap": 2000,
                                    "min_holdout_improvement": 0.0}
PANEL_POLICY_EXTRA: dict[str, float] = {"min_screen_ic": 0.005}


def _loss(metric: str, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-row losses whose mean (rmse: square root of the mean) is the metric."""
    if metric in ("rmse", "mse"):
        return (y - p) ** 2
    if metric == "mae":
        return np.abs(y - p)
    if metric == "log_loss":
        q = np.clip(p, 1e-12, 1 - 1e-12)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q))
    raise ValueError(f"unsupported metric {metric}")


def _aggregate(metric: str, losses: np.ndarray, axis: int | None = None) -> Any:
    mean = losses.mean(axis=axis)
    return np.sqrt(mean) if metric == "rmse" else mean


def block_length_for(rows: int) -> int:
    """Moving-block length for a series of ``rows`` observations: the ceiling of T^(1/3) (Künsch's rate)."""
    return max(1, int(math.ceil(rows ** (1 / 3)))) if rows > 0 else 1


def paired_bootstrap(metric: str, y: np.ndarray, model: np.ndarray, baseline: np.ndarray, *,
                     resamples: int, seed: int, block_length: int | None = None) -> dict[str, Any]:
    """Improvement of ``model`` over ``baseline`` (baseline loss minus model loss) with a one-sided p-value.

    Rows are resampled independently, or as moving blocks of ``block_length`` consecutive rows
    when the data are a time series (Künsch 1989), so serial dependence in the per-row losses
    widens the interval instead of being ignored. The p-value is the empirical tail
    (1 + #resamples at or below zero) / (resamples + 1): its resolution is one part in
    ``resamples + 1`` and a tail no resample reached is reported at that floor, never
    extrapolated below it.
    """
    lm, lb = _loss(metric, y, model), _loss(metric, y, baseline)
    observed = float(_aggregate(metric, lb) - _aggregate(metric, lm))
    rng = np.random.default_rng(seed)
    n = len(y)
    block = 1 if not block_length else int(min(max(1, block_length), n))
    blocks = -(-n // block)
    diffs = np.empty(resamples)
    for start in range(0, resamples, 200):
        stop = min(resamples, start + 200)
        if block == 1:
            index = rng.integers(0, n, size=(stop - start, n))
        else:
            starts = rng.integers(0, n - block + 1, size=(stop - start, blocks))
            index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(stop - start, blocks * block)[:, :n]
        diffs[start:stop] = _aggregate(metric, lb[index], axis=1) - _aggregate(metric, lm[index], axis=1)
    below = int(np.sum(diffs <= 0))
    p_value = (1 + below) / (resamples + 1)
    return {"improvement": observed, "p_value": float(p_value),
            "p_value_method": "iid-bootstrap" if block == 1 else "moving-block-bootstrap",
            "p_value_floor": 1 / (resamples + 1), "resamples": int(resamples), "block_length": block,
            "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975))}


class MLDomain(Domain):
    name = "ml"
    title = "Machine learning"
    summary = ("Training recipes searched across ridge/logistic, histogram gradient boosting, random forests "
               "and MLPs on tabular tasks or cross-sectional return panels, with a subsample screen, "
               "bootstrap or deflated-Sharpe selection control and a single-use holdout.")
    code_packages = ("autoresearch.domains.ml",)
    # Competent models agree with each other by construction, so only near-identical
    # predictions (a regularization step that changes nothing) count as redundant.
    default_max_correlation = 0.98

    def __init__(self, config: JSONObject, runtime: Path, policy: JSONObject | None = None) -> None:
        super().__init__(config, runtime, policy)
        if not isinstance(config, dict) or config.get("task") not in ("tabular", "panel"):
            raise ValueError("ml settings need task: tabular or panel")
        search_cfg = {**SEARCH_DEFAULTS, **config.get("search", {})}
        unknown = set(search_cfg) - set(SEARCH_DEFAULTS)
        if unknown:
            raise ValueError("unknown ml search settings: " + ", ".join(sorted(unknown)))
        mix = {**SEARCH_DEFAULTS["mix"], **search_cfg["mix"]}
        if set(mix) != set(SEARCH_DEFAULTS["mix"]) or sum(mix.values()) <= 0 or min(mix.values()) < 0:
            raise ValueError("search.mix must give nonnegative weights for random and mutation")
        search_cfg["mix"] = mix
        if not 0 < float(search_cfg["screen_fraction"]) <= 1:
            raise ValueError("search.screen_fraction must be in (0, 1]")
        self.search = search_cfg
        self.seed = int(config.get("seed", 0))
        task_settings = {k: v for k, v in config.items() if k not in ("search", "seed")}
        self.task: TabularTask | PanelTask
        if config["task"] == "tabular":
            self.task = TabularTask(task_settings, seed=self.seed)
        else:
            self.task = PanelTask(task_settings, seed=self.seed, runtime=self.runtime)
        self.problem = self.task.problem
        self.families = search.families_for(self.problem, search_cfg["families"])
        self.gates = self._policy(self.policy_config)
        self._baseline: float | None = None

    # ---------------------------------------------------------------- identity
    def _policy(self, supplied: JSONObject | None) -> JSONObject:
        supplied = dict(supplied or {})
        defaults: dict[str, Any]
        if self.task.kind == "tabular":
            defaults = dict(TABULAR_POLICY)
        else:
            from autoresearch.domains.alpha.domain import POLICY_DEFAULTS
            defaults = {**POLICY_DEFAULTS, **PANEL_POLICY_EXTRA}
            defaults.pop("min_dev_sharpe", None)
        unknown = set(supplied) - set(defaults)
        if unknown:
            raise ValueError("unknown ml policy keys: " + ", ".join(sorted(unknown)))
        return {**defaults, **supplied}

    def identity(self) -> JSONObject:
        return {"task": self.task.identity(), "problem": self.problem}

    def lineage(self) -> Lineage:
        task = self.task.identity()
        if self.task.kind == "panel":
            backend = task["alpha_lab"]
            data = {"data": backend["data_identity"], "universe": self.task.backend.settings.universe}
            return {"validation": {**data, **self.task.backend.bounds, "target": task["target"]},
                    "holdout": {**data, "holdout_start": self.config.get("holdout_start"), "target": task["target"]}}
        rows = {"content": task["content_sha256"], "rows": task["rows"], "split": task["split"],
                "split_mode": task["split_mode"], "split_seed": task["split_seed"], "problem": task["problem"]}
        return {"validation": rows, "holdout": rows}

    def public_config(self) -> JSONObject:
        value = {k: v for k, v in self.config.items() if k not in ("dataset", "holdout_start")}
        if isinstance(value.get("table"), str) and not value["table"].startswith("builtin:"):
            value["table"] = "csv"
        value["search"] = dict(self.search)
        return value

    # -------------------------------------------------------------- candidates
    def key(self, candidate: Candidate) -> str:
        if candidate.get("family") not in self.families:
            raise ValueError(f"family must be one of {', '.join(self.families)}")
        return search.recipe_key(candidate, self.problem, self.task.kind)

    def describe_candidate(self, candidate: Candidate) -> CandidateDescription:
        recipe = search.canonical(candidate, self.problem)
        return {"title": search.title(recipe), "family": recipe["family"], "recipe": recipe,
                "generator": candidate.get("generator", "")}

    def _candidate(self, recipe: dict[str, Any], generator: str, parents: Sequence[str] = ()) -> Candidate:
        return {"family": recipe["family"], "params": recipe["params"], "generator": generator,
                "parents": list(parents)}

    def propose(self, context: ProposalContext) -> list[Candidate]:
        rng = np.random.default_rng(context["rng_seed"])
        n = int(context["n"])
        out: list[Candidate] = []
        for item in context.get("submitted", []):
            candidate = dict(item["candidate"])
            out.append({"family": candidate.get("family"), "params": candidate.get("params") or {},
                        "generator": "inbox", "submitted_by": item.get("submitted_by", "")})
        if context["round"] == 1:
            out += [self._candidate(search.default(f, self.problem), "default") for f in self.families]
        population = context.get("population") or []
        mix = dict(self.search["mix"])
        if not population:
            mix["mutation"] = 0.0
        names = list(mix)
        weights = np.array([mix[k] for k in names], dtype=float)
        weights /= weights.sum()
        guard = 0
        while len(out) < n and guard < n * 5:
            guard += 1
            kind = names[int(rng.choice(len(names), p=weights))]
            if kind == "random":
                family = self.families[int(rng.integers(len(self.families)))]
                out.append(self._candidate(search.sample(rng, family, self.problem), "random"))
            else:
                size = min(len(population), int(self.search["tournament"]))
                picks = rng.choice(len(population), size=size, replace=False)
                parent = min((population[int(i)] for i in picks),
                             key=lambda e: (-(e["fitness"] if e["fitness"] is not None else -math.inf), e["key"]))
                recipe = {"family": parent["candidate"]["family"], "params": parent["candidate"]["params"]}
                try:
                    child = search.mutate(rng, recipe, self.problem, self.families)
                except ValueError:
                    continue
                out.append(self._candidate(child, "mutation", [parent["key"]]))
        return out

    # ------------------------------------------------------------------ helpers
    def _model(self, candidate: Candidate) -> Model:
        recipe = search.canonical(candidate, self.problem)
        return make_model(recipe["family"], candidate.get("params") or {}, task=self.problem, seed=self.seed)

    def _candidate_seed(self, candidate: Candidate) -> int:
        return int(hashlib.sha256(self.key(candidate).encode()).hexdigest()[:8], 16)

    def _block_length(self, rows: np.ndarray) -> int | None:
        """Moving-block length for a chronological tabular task; None resamples rows independently."""
        return block_length_for(len(rows)) if getattr(self.task, "ordered", False) else None

    def _tabular_inner(self, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        task = self.task
        if task.kind != "tabular":
            raise ValueError("the inner development split exists for tabular tasks only")
        X, y = task.development()
        cut = int(len(y) * 0.75)
        train = np.arange(cut)
        if fraction < 1:
            rng = np.random.default_rng(seed)
            train = np.sort(rng.choice(train, size=max(20, int(cut * fraction)), replace=False))
        return X[train], y[train], X[cut:], y[cut:]

    def _baseline_inner(self) -> float:
        if self._baseline is None:
            X, y, Xv, yv = self._tabular_inner(1.0, 0)
            base = make_model("constant", {}, task=self.problem).fit(X, y)
            self._baseline = float(_aggregate(self.task.metric, _loss(self.task.metric, yv, base.predict(Xv))))
        return self._baseline

    # ------------------------------------------------------------------- screen
    def screen(self, candidates: list[Candidate]) -> list[ScreenOutcome]:
        out: list[ScreenOutcome] = []
        for candidate in candidates:
            try:
                out.append(self._screen_one(candidate))
            except Exception as error:
                out.append({"status": "error", "fitness": None, "metrics": {},
                            "error": (type(error).__name__ + ": " + str(error))[:300]})
        return out

    def _screen_one(self, candidate: Candidate) -> ScreenOutcome:
        # Common random numbers: every recipe sees the same screening subsample, so
        # fitness differences come from the recipe and identical models look identical.
        seed = self.seed
        model = self._model(candidate)
        if self.task.kind == "tabular":
            X, y, Xv, yv = self._tabular_inner(float(self.search["screen_fraction"]), seed)
            model.fit(X, y)
            prediction = model.predict(Xv)
            score = float(_aggregate(self.task.metric, _loss(self.task.metric, yv, prediction)))
            baseline = self._baseline_inner()
            signature = prediction[: int(self.search["signature_rows"])].tolist()
            return {"status": "ok", "fitness": -score, "signature": signature,
                    "metrics": {"inner_score": score, "baseline_inner_score": baseline,
                                "relative_improvement": (baseline - score) / baseline if baseline else None,
                                "metric": self.task.metric}}
        rng = np.random.default_rng(seed)
        X, y, _ = self.task.arrays(self.task.rows("development", inner="train"), limit=int(self.search["screen_rows"]), rng=rng)
        model.fit(X, y)
        rows = self.task.rows("development", inner="validate")
        Xv, _yv, flat = self.task.arrays(rows)
        prediction_matrix = np.full(self.task._matrices()["y"].size, np.nan)
        prediction_matrix[flat] = model.predict(Xv)
        prediction_matrix = prediction_matrix.reshape(self.task._matrices()["y"].shape)
        ics = self.task.daily_ic(prediction_matrix, flat)
        if len(ics) < 5:
            return {"status": "error", "fitness": None, "metrics": {}, "error": "too few inner validation days"}
        mean = float(np.mean(ics))
        sample = np.random.default_rng(0).choice(len(flat), size=min(len(flat), int(self.search["signature_rows"])),
                                                 replace=False)
        return {"status": "ok", "fitness": mean, "signature": prediction_matrix.ravel()[flat][np.sort(sample)].tolist(),
                "metrics": {"ic_mean": mean, "ic_ir": mean / float(np.std(ics, ddof=1)) if np.std(ics) > 0 else None,
                            "days": len(ics)}}

    def shortlist_ok(self, screened: PopulationEntry) -> bool:
        metrics = screened.get("metrics") or {}
        if self.task.kind == "tabular":
            improvement = metrics.get("relative_improvement")
            return improvement is not None and improvement > 0
        ic = metrics.get("ic_mean")
        return ic is not None and ic >= self.gates["min_screen_ic"]

    # ----------------------------------------------------------------- validate
    def _save_model(self, candidate: Candidate, model: Model, label: str) -> str:
        directory = self.runtime / "models"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.key(candidate)}-{label}.json"
        path.write_text(json.dumps(model.to_json(), sort_keys=True, allow_nan=False) + "\n")
        return path.relative_to(self.runtime).as_posix()

    def validate(self, candidate: Candidate) -> EvaluationResult:
        model = self._model(candidate)
        if self.task.kind == "tabular":
            X, y = self.task.development()
            Xv, yv = self.task.validation()
            model.fit(X, y)
            prediction = model.predict(Xv)
            baseline = make_model("constant", {}, task=self.problem).fit(X, y).predict(Xv)
            metric = self.task.metric
            stats = paired_bootstrap(metric, yv, prediction, baseline, resamples=int(self.gates["bootstrap"]),
                                     seed=self._candidate_seed(candidate), block_length=self._block_length(yv))
            score = float(_aggregate(metric, _loss(metric, yv, prediction)))
            base = float(_aggregate(metric, _loss(metric, yv, baseline)))
            tabular: dict[str, Any] = {"metric": metric, "score": score, "baseline": base, "rows": int(len(yv)),
                                       "relative_improvement": (base - score) / base if base else None, **stats}
            return {"status": "ok", "metrics": {"validation": tabular}, "periods": {"validation": tabular},
                    "model": model.describe(), "artifact": self._save_model(candidate, model, "validation")}
        rows = self.task.rows("development")
        X, y, _ = self.task.arrays(rows, limit=self.task.max_train_rows, rng=np.random.default_rng(self.seed))
        model.fit(X, y)
        predictions = self.task.predict_matrix(model)
        raw = self.task.simulate(predictions, "validation")
        summary = raw["summary"]
        pnl = np.asarray(raw["daily"]["pnl"], dtype=float)
        from autoresearch.domains.alpha.evaluator import daily_profile, sharpe_of

        robustness = daily_profile(pnl)
        half = len(pnl) // 2
        robustness["half_sharpes"] = [sharpe_of(pnl[:half]), sharpe_of(pnl[half:])]
        stressed = self.task.simulate(predictions, "validation", cost_multiplier=float(self.gates["cost_stress"]))
        robustness["cost_stress_sharpe"] = stressed["summary"].get("sharpe")
        robustness["cost_stress_multiplier"] = float(self.gates["cost_stress"])
        ics = self.task.daily_ic(predictions, self.task.rows("validation"))
        validation: dict[str, Any] = {key: summary.get(key) for key in ("days", "sharpe", "returns", "turnover", "fitness",
                                                                        "drawdown", "margin_bps", "coverage")}
        validation.update(ic_mean=float(np.mean(ics)) if ics else None,
                          ic_ir=float(np.mean(ics) / np.std(ics, ddof=1)) if len(ics) > 2 and np.std(ics) > 0 else None)
        in_sample = self.task.daily_ic(predictions, rows)
        result: EvaluationResult = {
            "status": "ok",
            "metrics": {"development": {"ic_in_sample": float(np.mean(in_sample)) if in_sample else None},
                        "validation": validation},
            "periods": {"validation": {"days": validation["days"]}},
            "settings": {"cost_bps": self.task.backend.settings.cost_bps}, "robustness": robustness,
            "model": model.describe(), "artifact": self._save_model(candidate, model, "validation")}
        # The recipe's numeric parameters scaled down and up stand in for an expression's windows:
        # a signal that survives only one exact hyperparameter setting is not robust. The two refits
        # are the expensive part of a panel validation, so they run gate-first: the review that will
        # judge this result is previewed on the base result (Research gates, the deflated Sharpe ratio
        # at its most lenient count of one trial, cost stress, halves), and only a candidate that would
        # otherwise pass is refit. One the review rejects anyway records the probe as skipped, with the
        # reason; its stored numbers are those of the base result, exactly as before.
        from autoresearch.domains.alpha.domain import PERTURBATION_FACTORS, review_signal

        preview = review_signal(result, {"n_validation_trials": 1, "campaign_validation_trials": 0,
                                         "lineage_validation_trials": 0, "validation_scores": [],
                                         "policy": self.gates}, self.gates)
        perturbed: list[dict[str, Any]] = []
        probe: dict[str, Any] = {"kind": "recipe"}
        if preview["decision"] == "park" and (preview.get("checks") or {}).get("robustness") == "unavailable":
            recipe = search.canonical(candidate, self.problem)
            for factor in PERTURBATION_FACTORS:
                scaled = search.scale_recipe(recipe, factor, self.problem)
                if search.canonical(scaled, self.problem)["params"] == recipe["params"]:
                    continue
                other = make_model(scaled["family"], scaled["params"], task=self.problem, seed=self.seed).fit(X, y)
                perturbed.append({"factor": factor,
                                  "sharpe": self.task.simulate(self.task.predict_matrix(other), "validation")["summary"].get("sharpe")})
            probe["status"] = "ok" if perturbed else "no_parameter"
        else:
            probe.update(status="skipped", reason=preview["reason"][:300])
        robustness["perturbations"] = perturbed
        robustness["perturbation"] = probe
        return result

    def selection_score(self, result: EvaluationResult) -> float | None:
        if self.task.kind == "tabular":
            value = ((result.get("metrics") or {}).get("validation") or {}).get("relative_improvement")
        else:
            value = (result.get("robustness") or {}).get("daily_sharpe")
        return float(value) if value is not None and math.isfinite(value) else None

    # ------------------------------------------------------------------- review
    def review(self, candidate: Candidate, result: EvaluationResult, context: TrialContext) -> Review:
        policy = self._policy(context.get("policy"))
        if result.get("status") != "ok":
            return {"decision": "repair", "reason": "Training or scoring failed; repair the operational error.",
                    "checks": {}}
        if self.task.kind == "panel":
            from autoresearch.domains.alpha.domain import review_signal
            return review_signal(result, context, policy)
        validation = result["metrics"]["validation"]
        trials = max(1, int(context.get("n_validation_trials", 1)))
        level = policy["alpha_level"] / trials
        floor = validation.get("p_value_floor")
        checks: dict[str, Any] = {"p_value": validation["p_value"], "bonferroni_level": level,
                                  "validation_trials": trials, "relative_improvement": validation["relative_improvement"],
                                  "p_value_floor": floor, "p_value_method": validation.get("p_value_method")}
        failures: list[str] = []
        if validation["p_value"] > level:
            message = (f"improvement p-value {validation['p_value']:.4f} above the Bonferroni level "
                       f"{level:.5f} for {trials} validation trials")
            if floor is not None and floor > level:
                message += (f" (the smallest p-value {validation.get('resamples')} resamples can report is "
                            f"{floor:.5f}; raise policy.bootstrap)")
            failures.append(message)
        if (validation["relative_improvement"] or 0) < policy["min_relative_improvement"]:
            failures.append(f"relative improvement below {policy['min_relative_improvement']:.0%}")
        if failures:
            return {"decision": "reject", "reason": "Selection checks failed: " + "; ".join(failures), "checks": checks}
        return {"decision": "promote", "checks": checks,
                "reason": "Beats the constant baseline on validation beyond the multiplicity-adjusted level; "
                          "eligible for deeper research only."}

    def final_check(self, entry: AcceptedEntry, context: TrialContext) -> JSONObject:
        policy = self._policy(context.get("policy"))
        if self.task.kind == "panel":
            from autoresearch.domains.alpha.domain import deflated_check
            return deflated_check(entry, context, policy["min_dsr"], policy["dsr_variance"])
        checks = entry.get("checks") or {}
        trials = max(1, int(context["n_validation_trials"]))
        level = policy["alpha_level"] / trials
        p_value = checks.get("p_value", 1.0)
        return {"eligible": p_value <= level, "statistic": "bonferroni", "p_value": p_value, "bonferroni_level": level,
                "level_at_review": checks.get("bonferroni_level"), "trials_at_review": checks.get("validation_trials")}

    # ------------------------------------------------------------------ holdout
    def evaluate_holdout(self, candidate: Candidate) -> EvaluationResult:
        model = self._model(candidate)
        if self.task.kind == "tabular":
            X, y = self.task.development()
            Xv, yv = self.task.validation()
            Xa, ya = np.concatenate([X, Xv]), np.concatenate([y, yv])
            held = self.task.holdout()
            model.fit(Xa, ya)
            prediction = model.predict(held["X"])
            baseline = make_model("constant", {}, task=self.problem).fit(Xa, ya).predict(held["X"])
            metric = self.task.metric
            stats = paired_bootstrap(metric, held["y"], prediction, baseline, resamples=int(self.gates["bootstrap"]),
                                     seed=self._candidate_seed(candidate) + 1, block_length=self._block_length(held["y"]))
            score = float(_aggregate(metric, _loss(metric, held["y"], prediction)))
            base = float(_aggregate(metric, _loss(metric, held["y"], baseline)))
            tabular: dict[str, Any] = {"metric": metric, "score": score, "baseline": base, "rows": int(len(held["y"])),
                                       "days": int(len(held["y"])),
                                       "relative_improvement": (base - score) / base if base else None, **stats}
            if "probability" in held:
                tabular["bayes_optimal_score"] = float(_aggregate(metric, _loss(metric, held["y"], held["probability"])))
            if self.task.known.get("noise_floor_rmse") is not None and metric == "rmse":
                tabular["noise_floor"] = self.task.known["noise_floor_rmse"]
            return {"status": "ok", "metrics": {"holdout": tabular}, "periods": {"holdout": tabular},
                    "model": model.describe(), "artifact": self._save_model(candidate, model, "final")}
        # The refit reads the sealed matrices only (development and validation rows, targets realized by
        # the validation end); the reserved sessions enter through the task's holdout path alone.
        rows = self.task.rows("train_final")
        X, y, _ = self.task.arrays(rows, limit=self.task.max_train_rows, rng=np.random.default_rng(self.seed))
        model.fit(X, y)
        raw = self.task.evaluate_holdout(model)
        summary = raw["summary"]
        holdout = {key: summary.get(key) for key in ("days", "sharpe", "returns", "turnover", "fitness", "drawdown")}
        from autoresearch.domains.alpha.evaluator import daily_profile

        # The per-day profile of the holdout PnL is what a batch confirmation prices with the
        # probabilistic Sharpe ratio over the holdout sessions.
        profile = daily_profile(np.asarray(raw["daily"]["pnl"], dtype=float))
        return {"status": "ok", "metrics": {"holdout": holdout}, "periods": {"holdout": holdout},
                "settings": {"cost_bps": self.task.backend.settings.cost_bps, "holdout_evaluated": True},
                "holdout_profile": profile,
                "model": model.describe(), "artifact": self._save_model(candidate, model, "final")}

    def assess_final(self, result: EvaluationResult) -> Verdict:
        if self.task.kind == "panel":
            return assess_signal_final(result, {k: self.gates[k] for k in ("min_holdout_days", "min_holdout_sharpe")})
        if result.get("status") != "ok":
            return {"decision": "repair", "reason": "Holdout training or scoring failed."}
        if result.get("reused"):
            return {"decision": "not_confirmed", "reason": "A reused holdout is further research, not an untouched final test."}
        holdout = result["metrics"]["holdout"]
        if holdout["improvement"] <= self.gates["min_holdout_improvement"]:
            return {"decision": "not_confirmed", "reason": "No improvement over the constant baseline on the holdout."}
        if holdout["p_value"] > self.gates["alpha_level"]:
            return {"decision": "not_confirmed", "reason": "Holdout improvement is not significant at the configured level."}
        return {"decision": "confirmed",
                "reason": "Reserved holdout agreed with validation; evidence for deeper research only."}

    def release(self) -> None:
        if hasattr(self.task, "release"):
            self.task.release()
