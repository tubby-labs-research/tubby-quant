"""A single-writer, checkpointed source -> hypothesis -> experiment controller."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import default_research_root, default_runtime_root, validate_config
from .discovery import discover
from .models import make_model
from .policy import assess, assess_final
from .records import ResearchStore, canonical_expression, stable_id
from qlib_yahoo.alphalab.fastexpr import ExpressionError


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _public_config(config):
    value = copy.deepcopy(config)
    for key in ("research_root", "runtime_root"):
        value.pop(key, None)
    value["backend"].pop("dataset", None)
    for section in ("model", "discovery"):
        value[section].pop("fixture_path", None)
    value["model"].pop("api_key_env", None)
    if "endpoint" in value["model"]:
        value["model"]["endpoint_sha256"] = hashlib.sha256(value["model"].pop("endpoint").encode()).hexdigest()
    # Source bodies and fixture model outputs remain local inputs. Their hashes
    # bind the configuration, while selected evidence travels with hypotheses.
    for section, keys in (("discovery", ("fixture_sources",)),
                          ("model", ("hypotheses", "refinements"))):
        for key in keys:
            if key in value[section]:
                value[section][key + "_sha256"] = stable_id("input", value[section].pop(key))
    return value


class Controller:
    """Each tick performs one durable transition; completed records survive reset."""

    def __init__(self, config, *, backend=None, model=None, discoverer=None):
        self.config = validate_config(config)
        self.store = ResearchStore(Path(config.get("research_root", default_research_root())))
        self.runtime = Path(config.get("runtime_root", default_runtime_root()))
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.model = model or make_model(self.config["model"])
        if backend is None:
            from .backend import AlphaLabBackend
            backend = AlphaLabBackend(self.config["backend"], self.runtime / "engine")
        self.backend = backend
        self.discoverer = discoverer or discover
        self.backend_identity = backend.identity()
        model_identity = self.model.identity() if hasattr(self.model, "identity") else {"provider": "injected"}
        self.manifest = {"config": _public_config(self.config), "backend": self.backend_identity,
                         "model": model_identity, "controller_version": 1}
        self.manifest["code"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path(__file__).parent.glob("*.py"))}
        self.manifest["fixture_inputs"] = {
            section: hashlib.sha256(Path(self.config[section]["fixture_path"]).read_bytes()).hexdigest()
            for section in ("model", "discovery") if self.config[section].get("fixture_path")}
        self.campaign_id = stable_id("campaign", self.manifest)
        self.state_path = self.runtime / (self.campaign_id + ".json")
        self.sources_path = self.runtime / (self.campaign_id + "-sources.json")
        self.limits = self.config["limits"]
        self.state = None

    def _load(self):
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("campaign_id") != self.campaign_id:
                raise ValueError("campaign checkpoint identity mismatch")
        else:
            self.state = {"campaign_id": self.campaign_id, "queue": [], "hypotheses": [],
                          "families": {}, "evaluations": 0, "model_calls": 0,
                          "daily_calls": {}, "last_discovery": 0, "discovery_errors": 0,
                          "completed": 0, "status": "ready", "created_at": utc_now()}
        if self.store.get("campaigns", self.campaign_id) is None:
            self.store.put("campaigns", {"id": self.campaign_id, **self.manifest,
                                         "created_at": self.state["created_at"]})

    def _save(self):
        atomic_json(self.state_path, self.state)

    def _event(self, kind, **details):
        value = {"kind": kind, "campaign_id": self.campaign_id,
                 "created_at": utc_now(), **details}
        value["id"] = stable_id("event", {**value, "nonce": uuid.uuid4().hex})
        self.store.put("events", value)

    def _key(self, expression):
        return stable_id("evaluation", {"expression": canonical_expression(expression),
                                        "backend": self.backend_identity})

    def _model_call(self, method, *args):
        day = datetime.now(timezone.utc).date().isoformat()
        if self.state["model_calls"] >= self.limits["max_model_calls"]:
            raise BudgetReached("campaign model-call budget reached")
        if self.state["daily_calls"].get(day, 0) >= self.limits["daily_model_calls"]:
            raise BudgetReached("daily model-call budget reached")
        # Reserve before request: interrupted or failed calls still consume budget.
        self.state["model_calls"] += 1
        self.state["daily_calls"][day] = self.state["daily_calls"].get(day, 0) + 1
        self._save()
        return getattr(self.model, method)(*args)

    def _sources(self):
        if self.sources_path.exists():
            return json.loads(self.sources_path.read_text(encoding="utf-8"))
        return []

    def _bind_sources(self):
        if hasattr(self.model, "bind_sources"):
            self.model.bind_sources(self._sources())

    def _enqueue(self, proposal, parent=None, depth=0):
        content = copy.deepcopy(proposal)
        content["depth"] = depth
        if parent:
            content["parent_experiment_id"] = parent["id"]
        if self.state.get("holdout_accessed"):
            # Selection after a holdout access is no longer untouched by it.
            content["post_holdout"] = True
        expression = content.get("expression", "")
        try:
            key = self._key(expression)
        except (ValueError, SyntaxError, ExpressionError):
            # Malformed candidates still receive durable evidence and repair.
            key = stable_id("invalid", {"expression": expression, "backend": self.backend_identity})
        previous = self.store.find_experiment(key)
        family = " ".join(str(content.get("family", "unclassified")).lower().split())
        if (len(self.state["hypotheses"]) >= self.limits["max_hypotheses"]
                or self.state["families"].get(family, 0) >= self.limits["max_per_family"]):
            self._event("hypothesis_budget", family=family)
            return False
        if any(job["experiment_key"] == key for job in self.state["queue"]):
            return False
        identity = {k: v for k, v in content.items() if k not in {"id", "created_at"}}
        hid = stable_id("hypothesis", identity)
        if hid in self.state["hypotheses"]:
            return False
        content["id"] = hid
        if self.store.get("hypotheses", hid) is None:
            self.store.put("hypotheses", {**content, "created_at": utc_now()})
        self.state["hypotheses"].append(hid)
        self.state["families"][family] = self.state["families"].get(family, 0) + 1
        self.state["queue"].append({"hypothesis_id": hid, "experiment_key": key,
                                    "phase": "evaluate", "errors": 0, "depth": depth})
        if previous:
            self._event("duplicate_experiment", experiment_key=key, hypothesis_id=hid,
                        reused_experiment_id=previous["id"])
        return True

    def _discover(self):
        if time.time() - self.state["last_discovery"] < self.config["discovery_interval_seconds"]:
            return {"status": "idle", "reason": "waiting for the next discovery interval"}
        self.state["last_discovery"] = time.time()
        self._save()
        try:
            sources = self.discoverer(self.config["discovery"])
            if not sources:
                self._event("no_sources")
                return {"status": "idle", "reason": "no retrievable sources"}
            previous = {source["id"]: source for source in self._sources()}
            previous.update({source["id"]: source for source in sources})
            for source in sources:
                meta = {k: v for k, v in source.items() if k != "text"}
                meta["text_sha256"] = hashlib.sha256(source["text"].encode()).hexdigest()
                meta["text_characters"] = len(source["text"])
                old = self.store.get("sources", source["id"])
                if old is None:
                    self.store.put("sources", meta)
                elif any(old.get(k) != v for k, v in meta.items() if k != "retrieved_at"):
                    raise ValueError("source ID conflicts with preserved evidence")
            atomic_json(self.sources_path, list(previous.values()))
            history = self.store.related_hypotheses(" ".join(s["title"] for s in sources), limit=8)
            proposals = self._model_call("propose", sources, history)
            count = sum(self._enqueue(p) for p in proposals)
            self.state["discovery_errors"] = 0
            self._event("discovered", source_ids=[s["id"] for s in sources], hypotheses_queued=count)
            return {"status": "discovered", "sources": len(sources), "queued": count}
        except BudgetReached as error:
            return {"status": "budget", "reason": str(error)}
        except Exception as error:
            self.state["discovery_errors"] += 1
            self._event("discovery_error", error_type=type(error).__name__)
            return {"status": "error", "stage": "discovery", "error_type": type(error).__name__}

    def _evaluate(self, job, hypothesis):
        if self.backend.identity() != self.backend_identity:
            raise RuntimeError("dataset or engine changed; restart to create a newly identified campaign")
        experiment = self.store.get("experiments", job.get("experiment_id", "")) if job.get("experiment_id") else None
        if experiment is None and job.get("experiment_id"):
            # A prior worker reserved this attempt but did not publish a result.
            # Preserve the unknown outcome; any retry reserves a new budget slot.
            experiment = {"id": job["experiment_id"], "hypothesis_id": hypothesis["id"],
                          "experiment_key": job["experiment_key"], "expression": hypothesis["expression"],
                          "parent_experiment_id": hypothesis.get("parent_experiment_id"),
                          "campaign_id": self.campaign_id, "backend_identity": self.backend_identity,
                          "result": {"status": "error", "error_kind": "interrupted",
                                     "error": "Worker stopped before result publication; outcome unknown."},
                          "created_at": job["started_at"]}
            self.store.put("experiments", experiment)
        if experiment is None:
            experiment = self.store.find_experiment(job["experiment_key"])
        if experiment is None:
            if "experiment_id" not in job:
                if self.state["evaluations"] >= self.limits["max_experiments"]:
                    return {"status": "budget", "reason": "experiment budget reached"}
                self.state["evaluations"] += 1
                job["experiment_id"] = stable_id("experiment", {
                    "campaign": self.campaign_id, "key": job["experiment_key"],
                    "attempt": self.state["evaluations"]})
                job["started_at"] = utc_now()
                self._save()
            try:
                problems = self.backend.validate(hypothesis["expression"])
                result = ({"status": "error", "error_kind": "expression", "errors": problems}
                          if problems else self.backend.evaluate(hypothesis["expression"]))
            except Exception as error:
                result = {"status": "error", "error_kind": "execution",
                          "error": type(error).__name__}
            experiment = {"id": job["experiment_id"], "hypothesis_id": hypothesis["id"],
                          "experiment_key": job["experiment_key"], "expression": hypothesis["expression"],
                          "parent_experiment_id": hypothesis.get("parent_experiment_id"),
                          "campaign_id": self.campaign_id, "backend_identity": self.backend_identity,
                          "result": result, "created_at": job["started_at"]}
            self.store.put("experiments", experiment)
        job["experiment_id"] = experiment["id"]
        job["phase"] = "review"
        self._save()
        return {"status": "evaluated", "experiment_id": experiment["id"],
                "outcome": experiment["result"]["status"]}

    def _review(self, job, hypothesis):
        experiment = self.store.get("experiments", job["experiment_id"])
        parent = self.store.get("experiments", hypothesis["parent_experiment_id"]) if hypothesis.get("parent_experiment_id") else None
        review = assess(experiment["result"], self.config.get("policy", {}),
                        parent["result"] if parent else None)
        rid = stable_id("review", {"experiment": experiment["id"], "hypothesis": hypothesis["id"],
                                    "policy": self.config.get("policy", {})})
        if self.store.get("reviews", rid) is None:
            self.store.put("reviews", {"id": rid, "hypothesis_id": hypothesis["id"],
                                       "experiment_id": experiment["id"], **review, "created_at": utc_now()})
        job["review_id"] = rid
        decision = review["decision"]
        if decision == "repair" and experiment["result"].get("error_kind") != "expression":
            job["errors"] += 1
            if job["errors"] < self.limits["max_job_errors"]:
                job.pop("experiment_id", None)
                job["phase"] = "evaluate"
            else:
                self._event("parked_operational_failure", hypothesis_id=hypothesis["id"])
                self._finish(job)
        elif decision in {"refine", "repair"} and job["depth"] < self.limits["max_refinement_depth"]:
            job["phase"] = "refine"
        else:
            self._finish(job)
        return {"status": "reviewed", "decision": decision, "reason": review["reason"]}

    def _finish(self, job):
        self.state["queue"].remove(job)
        self.state["completed"] += 1

    def _refine(self, job, hypothesis):
        experiment = self.store.get("experiments", job["experiment_id"])
        experiment = {**experiment, "review": self.store.get("reviews", job["review_id"])}
        self._bind_sources()
        history = self.store.related_hypotheses(hypothesis["mechanism"], limit=8)
        try:
            children = self._model_call("refine", hypothesis, experiment, history)
        except BudgetReached as error:
            return {"status": "budget", "reason": str(error)}
        except Exception as error:
            job["errors"] += 1
            self._event("refinement_error", hypothesis_id=hypothesis["id"], error_type=type(error).__name__)
            if job["errors"] >= self.limits["max_job_errors"]:
                self._finish(job)
            return {"status": "error", "stage": "refinement", "error_type": type(error).__name__}
        count = sum(self._enqueue(child, experiment, job["depth"] + 1) for child in children)
        self._finish(job)
        self._event("refined", parent_experiment_id=experiment["id"], queued=count)
        return {"status": "refined", "queued": count}

    def tick(self):
        with self.store.lock():
            self._load()
            if self.state["queue"]:
                job = self.state["queue"][0]
                hypothesis = self.store.get("hypotheses", job["hypothesis_id"])
                result = getattr(self, "_" + job["phase"])(job, hypothesis)
            elif self.state["evaluations"] >= self.limits["max_experiments"]:
                result = {"status": "complete", "reason": "experiment budget reached"}
            elif len(self.state["hypotheses"]) >= self.limits["max_hypotheses"]:
                result = {"status": "complete", "reason": "hypothesis budget reached"}
            elif self.state["model_calls"] >= self.limits["max_model_calls"]:
                result = {"status": "complete", "reason": "model-call budget reached"}
            else:
                result = self._discover()
            self.state["status"] = result["status"]
            self._save()
            return {"campaign_id": self.campaign_id, **result}

    def status(self):
        with self.store.lock():
            self._load()
            return {**self.state, "queued": len(self.state["queue"])}

    # ------------------------------------------------------------- final evaluation
    def _campaign_experiment(self, experiment_id):
        experiment = self.store.get("experiments", experiment_id)
        if experiment is None or experiment.get("campaign_id") != self.campaign_id:
            return None
        return experiment

    def _promoted(self):
        """Latest promoted experiment per hypothesis, in review order, for this campaign."""
        reviews = sorted(self.store.list("reviews"), key=lambda r: (str(r.get("created_at", "")), r["id"]))
        chosen = {}
        for review in reviews:
            if review.get("decision") != "promote":
                continue
            experiment = self._campaign_experiment(review.get("experiment_id", ""))
            if experiment is not None and experiment["result"].get("status") == "ok":
                chosen[review["hypothesis_id"]] = experiment
        return [(hypothesis_id, experiment, "promoted") for hypothesis_id, experiment in chosen.items()]

    def _latest_successful(self, hypothesis_id):
        experiments = [e for e in self.store.list("experiments")
                       if e.get("hypothesis_id") == hypothesis_id and e.get("campaign_id") == self.campaign_id
                       and isinstance(e.get("result"), dict) and e["result"].get("status") == "ok"]
        if not experiments:
            raise ValueError(f"hypothesis {hypothesis_id} has no successful evaluation in this campaign")
        return max(experiments, key=lambda e: (str(e.get("created_at", "")), e["id"]))

    def finalize(self, hypothesis_ids=None, *, allow_reuse=False):
        """Score selected hypotheses once on the reserved holdout and record immutable finals.

        Default selection: hypotheses whose latest review promoted them. Each
        expression's holdout access is reserved in the checkpoint before the
        evaluation runs, so an interrupted run still counts. A second access of
        the same expression is refused unless ``allow_reuse`` is set, in which
        case it is recorded with its access sequence and labelled ``reused``;
        the policy never confirms a reused holdout. Afterwards every new
        hypothesis in this campaign is labelled ``post_holdout``.
        """
        if not hasattr(self.backend, "evaluate_holdout"):
            raise ValueError("the configured backend has no holdout evaluation")
        with self.store.lock():
            self._load()
            if self.backend.identity() != self.backend_identity:
                raise RuntimeError("dataset or engine changed; restart to create a newly identified campaign")
            if hypothesis_ids is None:
                selected = self._promoted()
            else:
                selected = []
                for hypothesis_id in hypothesis_ids:
                    if self.store.get("hypotheses", hypothesis_id) is None:
                        raise ValueError(f"unknown hypothesis {hypothesis_id}")
                    selected.append((hypothesis_id, self._latest_successful(hypothesis_id), "explicit"))
            access = self.state.setdefault("holdout_access", {})
            for _hypothesis_id, experiment, _selection in selected:
                if access.get(experiment["experiment_key"], 0) and not allow_reuse:
                    raise ValueError("holdout already evaluated for " + _hypothesis_id
                                     + "; repeated access must be requested explicitly and is labelled reused")
            outcomes = []
            for hypothesis_id, experiment, selection in selected:
                hypothesis = self.store.get("hypotheses", hypothesis_id)
                key = experiment["experiment_key"]
                sequence = access.get(key, 0) + 1
                access[key] = sequence
                self.state["holdout_accessed"] = True
                self._event("holdout_access", hypothesis_id=hypothesis_id, experiment_id=experiment["id"],
                            access_sequence=sequence)
                self._save()  # reserve before the evaluation: an interruption still consumed the access
                try:
                    result = self.backend.evaluate_holdout(hypothesis["expression"])
                except Exception as error:
                    result = {"status": "error", "error_kind": "execution", "error": type(error).__name__}
                result = {**result, "reused": sequence > 1, "access_sequence": sequence}
                verdict = assess_final(result, self.config.get("policy", {}))
                record = {"id": stable_id("final", {"campaign": self.campaign_id, "key": key, "sequence": sequence}),
                          "campaign_id": self.campaign_id, "hypothesis_id": hypothesis_id,
                          "experiment_id": experiment["id"], "experiment_key": key,
                          "expression": hypothesis["expression"], "selection": selection,
                          "access_sequence": sequence, "reused": sequence > 1,
                          "backend_identity": self.backend_identity, "result": result,
                          "decision": verdict["decision"], "reason": verdict["reason"], "created_at": utc_now()}
                stored = self.store.put("finals", record)
                self._event("finalized", hypothesis_id=hypothesis_id, final_id=stored["id"],
                            decision=stored["decision"], reused=stored["reused"])
                outcomes.append({"final_id": stored["id"], "hypothesis_id": hypothesis_id,
                                 "experiment_id": experiment["id"], "selection": selection,
                                 "decision": stored["decision"], "reason": stored["reason"],
                                 "reused": stored["reused"], "access_sequence": sequence,
                                 "holdout": (result.get("metrics") or {}).get("holdout", {})})
            self._save()
            return outcomes


class BudgetReached(RuntimeError):
    pass
