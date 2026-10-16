"""Bounded local research control; reads and configuration never invoke a model."""
from __future__ import annotations

import ast
import copy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid
import time
from datetime import datetime, timezone

from autoresearch.config import validate_config
from autoresearch.controller import atomic_json
from autoresearch.backend import _date
from autoresearch.discovery import discover
from autoresearch.models import AnthropicModel, validate_candidates
from autoresearch.models import APIAgentModel, ModelError
from autoresearch.agents import load_agents, credential_ready
from autoresearch.policy import DEFAULTS
from autoresearch.records import stable_id
from qlib_yahoo.alphalab.settings import Settings
from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.alphalab.fastexpr import parse
from qlib_yahoo.webapp.research_series import period_series
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_COLLECTIONS = ("campaigns", "hypotheses", "experiments", "reviews", "sources", "events")
_HIDDEN = {"code", "code_sha256", "backend_identity", "fixture_inputs", "fixture_path", "fixture_sources", "hypotheses_sha256", "refinements_sha256", "api_key_env", "endpoint", "research_root", "runtime_root", "artifacts", "api_key", "password", "secret", "token", "authorization"}


class ResearchService:
    def __init__(self, data_root, source_root=ROOT):
        self.data_root = Path(data_root).resolve()
        self.source_root = Path(source_root).resolve()
        self.root = self.data_root / "autoresearch"
        self.config_root = self.root / "configs"
        self.records_root = self.root / "records"
        self.runtime_root = self.root / "runtime"
        self.demo_root = self.root / "demo"
        self._lock = threading.RLock()
        self._process = None
        self._closed = False
        self._stop_requested = False
        self._stopping = False
        self._worker = self._idle()
        self._connections = {}

    @staticmethod
    def _idle():
        return dict(running=False, status="idle", campaign_id=None, config_id=None, demo=False,
                    continuous=False, reviewed=False, phase=None, evaluations=0, queued=0, last_error=None, log=[])

    @staticmethod
    def _id(value):
        if not isinstance(value, str) or not _SAFE.fullmatch(value):
            raise ValueError("Invalid research id")
        return value

    def _inside(self, root, path):
        root = Path(root).resolve()
        path = Path(path)
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("Path must stay inside its managed research folder")
        return path

    def _config_path(self, value):
        self._inside(self.data_root, self.config_root)
        return self._inside(self.config_root, self.config_root / (self._id(value) + ".json"))

    def _clean(self, value, depth=0, *, keep_inputs=False, _redactions=None):
        if _redactions is None:
            names = {"ANTHROPIC_API_KEY"}
            try:
                names.update(row["config"].get("api_key_env") for row in load_agents(self.data_root).values())
            except ModelError:
                pass
            names.update(k for k in os.environ if re.search(r"KEY|TOKEN|SECRET|PASSWORD", k, re.I))
            _redactions = {os.environ.get(name, "") for name in names if name and os.environ.get(name, "")}
        if depth > 12:
            return None
        if isinstance(value, dict):
            return {k: self._clean(v, depth + 1, keep_inputs=keep_inputs, _redactions=_redactions) for k, v in list(value.items())[:100]
                    if (k.lower() not in _HIDDEN or (keep_inputs and k == "fixture_sources")) and not k.lower().endswith("_path")}
        if isinstance(value, list):
            return [self._clean(v, depth + 1, keep_inputs=keep_inputs, _redactions=_redactions) for v in value[:300]]
        if isinstance(value, str):
            # Cover custom agent credential names as well as the legacy SDK key.
            for secret in _redactions:
                value = value.replace(secret, "[redacted]")
            value = re.sub(r"\bsk-[\w-]{8,}", "[redacted]", value)
            for root in (self.data_root, self.source_root):
                value = value.replace(str(root), "[local path]")
            return re.sub(r"(?:/Users/|/home/|/private/|/tmp/|/var/folders/)[^\s'\"<>]*", "[local path]", value)[:4000]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def _read(self, path):
        if path.stat().st_size > 4_000_000:
            raise ValueError("Research record is too large")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Research record must be an object")
        return value

    def _validate(self, data):
        if not isinstance(data, dict):
            raise ValueError("Research configuration must be an object")
        allowed = {"id", "name", "backend", "model", "discovery", "policy", "limits"}
        if data.keys() - allowed:
            raise ValueError("Unknown research configuration fields")
        if not isinstance(data.get("limits", {}), dict):
            raise ValueError("limits must be an object")
        cfg = validate_config({k: v for k, v in data.items() if k != "id"})
        if len(cfg["name"]) > 200:
            raise ValueError("Campaign name exceeds 200 characters")
        backend = cfg["backend"]
        required = {"dataset", "universe", "development_start", "development_end", "validation_start", "validation_end", "cost_bps"}
        if required - backend.keys() or backend.keys() - required - {"holdout_start", "warmup"}:
            raise ValueError("Backend requires dataset, universe, chronological dates and cost_bps; only warmup and holdout_start are optional")
        dataset = backend["dataset"]
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("Select a local dataset")
        folder = self._inside(self.data_root, self.data_root / "datasets")
        path = self._inside(folder, Path(dataset) if Path(dataset).is_absolute() else folder / dataset)
        backend["dataset"] = path.relative_to(folder).as_posix()
        if not backend["dataset"] or backend["dataset"] == ".":
            raise ValueError("Select a dataset folder")
        if not isinstance(backend["universe"], str) or not backend["universe"].strip():
            raise ValueError("Select an explicit universe")
        ds, de, vs, ve = [_date(backend[k], k) for k in ("development_start", "development_end", "validation_start", "validation_end")]
        if not ds <= de < vs <= ve:
            raise ValueError("Development and validation must be chronological and nonoverlapping")
        if backend.get("holdout_start"):
            if _date(backend["holdout_start"], "holdout_start") <= ve:
                raise ValueError("Holdout must start after validation")
        elif "holdout_start" in backend:
            backend.pop("holdout_start")
        if type(backend["cost_bps"]) not in (int, float) or not math.isfinite(backend["cost_bps"]):
            raise ValueError("cost_bps must be finite")
        try:
            settings = Settings(**{k: backend[k] for k in ("universe", "cost_bps", "warmup") if k in backend}).validate()
        except DatasetError as error:
            raise ValueError(str(error)) from error
        if settings.warmup < 2:
            raise ValueError("warmup must be at least 2")
        model = cfg["model"]
        reviewed = model.get("provider") == "reviewed"
        model_keys = {"provider", "model", "research_focus"} | ({"hypotheses", "refinements"} if reviewed else {"agent_id"} if model.get("provider") == "agent" else set())
        if model.keys() - model_keys or model.get("provider") not in {"anthropic", "reviewed", "agent"}:
            raise ValueError("Select a configured agent, Anthropic or reviewed ideas; endpoints and credentials are server-side")
        if model.get("provider") == "agent":
            self._id(model.get("agent_id"))
        focus = model.get("research_focus", "")
        if not isinstance(focus, str) or len(focus) > 2000:
            raise ValueError("research_focus must be text of at most 2000 characters")
        if not isinstance(model.get("model"), str) or not model["model"].strip() or len(model["model"]) > 200:
            raise ValueError("Provide an explicit model id of at most 200 characters")
        discovery = cfg["discovery"]
        if reviewed:
            self._validate_reviewed(model, discovery)
        else:
            if model["provider"] == "anthropic":
                AnthropicModel(model)  # validation only, no client or network
            if discovery.keys() - {"arxiv_queries", "max_results"}:
                raise ValueError("Unknown discovery settings")
            queries = discovery.get("arxiv_queries", [])
            if not isinstance(queries, list) or not 1 <= len(queries) <= 10 or any(not isinstance(q, str) or not q.strip() or len(q) > 2000 for q in queries):
                raise ValueError("Provide one to ten nonempty arXiv queries")
            discover({**discovery, "arxiv_queries": [], "network_enabled": False})
        policy = cfg.get("policy", {})
        if policy.keys() - DEFAULTS.keys():
            raise ValueError("Unknown research policy threshold")
        for key, value in {**DEFAULTS, **policy}.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Policy thresholds must be finite numbers")
            if (key == "min_validation_days" and (type(value) is not int or value < 2)) or (key in {"max_turnover", "max_drawdown", "min_refinement_improvement"} and value < 0):
                raise ValueError("Invalid research policy threshold: " + key)
        cfg.pop("discovery_interval_seconds")
        cfg.pop("idle_seconds")
        return cfg

    def _validate_reviewed(self, model, discovery):
        if discovery.keys() - {"fixture_sources", "arxiv_queries", "network_enabled", "max_results"}:
            raise ValueError("Unknown reviewed-source settings")
        if discovery.get("arxiv_queries") or discovery.get("network_enabled", False) is not False:
            raise ValueError("Reviewed ideas use saved source snapshots with networking disabled")
        if type(discovery.get("max_results", 10)) is not int or not 1 <= discovery.get("max_results", 10) <= 10:
            raise ValueError("Reviewed max_results must be between 1 and 10")
        sources = discovery.get("fixture_sources")
        if not isinstance(sources, list) or not 1 <= len(sources) <= 10:
            raise ValueError("Provide one to ten reviewed source snapshots")
        seen = set()
        for row in sources:
            if not isinstance(row, dict) or set(row) != {"id", "title", "url", "text"}:
                raise ValueError("Each source requires exactly id, title, url and text")
            self._id(row["id"])
            if row["id"] in seen:
                raise ValueError("Reviewed source ids must be unique")
            seen.add(row["id"])
            for key in ("title", "url", "text"):
                if not isinstance(row[key], str) or not row[key].strip() or len(row[key]) > 2000:
                    raise ValueError("Source fields must be nonempty text of at most 2000 characters")
            url = urlparse(row["url"])
            if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password:
                raise ValueError("Source URLs must be public HTTP or HTTPS references")
        normalized = discover({"fixture_sources": sources, "network_enabled": False, "max_results": 10})
        base_keys = {"title", "mechanism", "family", "expression", "source_ids", "evidence", "expected_horizon", "required_fields", "rationale"}
        # Extract the engine's operator names without importing numpy or the evaluator.
        operators = {"add", "subtract", "multiply", "divide", "and", "or", "not"}
        for filename, names in (("engine.py", {"TS", "REGISTRY", "ALIASES"}), ("ops.py", {"ELEMENTWISE"})):
            tree = ast.parse((ROOT / "qlib_yahoo/alphalab" / filename).read_text())
            for node in tree.body:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
                if any(isinstance(t, ast.Name) and t.id in names for t in targets):
                    value = node.value
                    if isinstance(value, ast.Dict):
                        operators.update(k.value for k in value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str))
                    elif isinstance(value, ast.DictComp):
                        for gen in value.generators:
                            if isinstance(gen.iter, (ast.Tuple, ast.List)):
                                operators.update(k.value for k in gen.iter.elts if isinstance(k, ast.Constant) and isinstance(k.value, str))
        for section in ("hypotheses", "refinements"):
            values = model.get(section, [])
            if not isinstance(values, list) or len(values) > 20 or (section == "hypotheses" and not values):
                raise ValueError("Provide one to twenty hypotheses and at most twenty refinements")
            for candidate in values:
                allowed = base_keys | ({"parent_experiment_id", "question", "expected_change"} if section == "refinements" else set())
                if not isinstance(candidate, dict) or set(candidate) != allowed:
                    raise ValueError("Reviewed candidates must match the strict hypothesis schema")
                if any(isinstance(v, str) and len(v) > (4096 if k == "expression" else 2000) for k, v in candidate.items()):
                    raise ValueError("Reviewed candidate fields exceed their text limit")
                if not isinstance(candidate.get("evidence"), list) or any(not isinstance(e, dict) or set(e) != {"source_id", "quote"} for e in candidate["evidence"]):
                    raise ValueError("Evidence requires exactly source_id and quote")
            validate_candidates(values, normalized, max_candidates=20,
                                parent_experiment_id="$parent" if section == "refinements" else None)
            for candidate in values:
                try:
                    parse(candidate["expression"])
                except DatasetError as error:
                    raise ValueError(str(error)) from error
                calls = re.findall(r"\b([A-Za-z_]\w*)\s*\(", candidate["expression"])
                if any(name not in operators for name in calls):
                    raise ValueError("Reviewed expression contains an unknown operator")
        discovery["arxiv_queries"] = []
        discovery["network_enabled"] = False
        discovery["max_results"] = 10

    def config(self, config_id):
        with self._lock:
            path = self._config_path(config_id)
            if not path.exists():
                raise ValueError("Research configuration not found")
            return self._clean({"id": config_id, **self._validate(self._read(path))}, keep_inputs=True)

    def save(self, data):
        with self._lock:
            cfg = self._validate(data)
            config_id = self._id(data.get("id") or "config_" + uuid.uuid4().hex[:16])
            if cfg["model"]["provider"] == "reviewed" and data.get("id"):
                if self.busy() and self._worker["config_id"] == config_id:
                    raise ValueError("Stop the worker before revising its reviewed ideas")
                config_id = "config_" + uuid.uuid4().hex[:16]
            if self.busy() and self._worker["config_id"] == config_id:
                raise ValueError("Stop the worker before editing its active configuration")
            path = self._config_path(config_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(path, cfg)
            return self._clean({"id": config_id, **cfg}, keep_inputs=True)

    def busy(self):
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def _resolved_model(self, model):
        if model["provider"] != "agent":
            return copy.deepcopy(model)
        row = load_agents(self.data_root).get(model["agent_id"])
        if row is None:
            raise ModelError("Selected research agent is no longer configured on this server")
        if model["model"] != row["config"]["model"]:
            raise ModelError("Agent model changed; select the agent again and save a new setup")
        return {**row["config"], "research_focus": model.get("research_focus", "")}

    def agent_catalog(self):
        """Configuration readiness is offline; reachability requires explicit Check."""
        rows = [{"id": "reviewed", "name": "Reviewed source-backed ideas", "provider": "reviewed",
                 "provider_name": "Reviewed plan", "model": "session-reviewed", "ready": True,
                 "state": "offline", "message": "Uses your cited plan; no model connection needed."},
                {"id": "anthropic", "name": "Anthropic · direct model", "provider": "anthropic",
                 "provider_name": "Anthropic", "model": "", "ready": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
                 "state": "configured" if os.environ.get("ANTHROPIC_API_KEY", "").strip() else "missing_credential",
                 "message": "Credential available; connection checked when research starts." if os.environ.get("ANTHROPIC_API_KEY", "").strip() else "Ask the server administrator to provide the Anthropic credential."}]
        issues = []
        try:
            for agent_id, row in load_agents(self.data_root).items():
                cfg = row["config"]
                model = APIAgentModel(cfg)
                ready = credential_ready(cfg)
                cached = self._connections.get(agent_id, {})
                valid = cached.get("fingerprint") == model.fingerprint() and time.monotonic() - cached.get("time", 0) < 60
                connection = cached["result"] if valid and ready else {}
                rows.append({"id": "agent:" + agent_id, "agent_id": agent_id, "name": row["name"],
                             "provider": "agent", "provider_name": cfg["provider_name"], "model": cfg["model"],
                             "ready": ready, "state": connection.get("state", "configured" if ready else "missing_credential"),
                             "checked_at": connection.get("checked_at"),
                             "message": connection.get("message", "Configured; connection has not been checked." if ready else "Server credential is missing for this agent.")})
        except ModelError as error:
            issues.append(str(error))
        return {"agents": rows, "issues": issues}

    def check_agent(self, data):
        if not isinstance(data, dict) or set(data) != {"agent_id"}:
            raise ValueError("Connection check requires a configured agent id")
        agent_id = self._id(data["agent_id"])
        row = load_agents(self.data_root).get(agent_id)
        if row is None:
            raise ValueError("Research agent is not configured")
        model = APIAgentModel(row["config"])
        result = {"agent_id": agent_id, "checked_at": datetime.now(timezone.utc).isoformat()}
        if not credential_ready(row["config"]):
            result.update(state="missing_credential", message="Server credential is missing for this agent.")
        else:
            try:
                model.check_connection()  # GET health only; never a generation request
                result.update(state="connected", message="Agent answered the health check. No model call was made.")
            except ModelError as error:
                result.update(state="unavailable", message=str(error))
        with self._lock:
            self._connections[agent_id] = {"fingerprint": model.fingerprint(), "time": time.monotonic(), "result": result}
        return self._clean(result)

    def _issues(self, cfg):
        issues = []
        folder = self.data_root / "datasets" / cfg["backend"]["dataset"]
        if not folder.is_dir() or not (folder / "manifest.json").is_file():
            issues.append("Dataset is unavailable or has no manifest")
        if cfg["model"]["provider"] == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            issues.append("ANTHROPIC_API_KEY is not available to this app")
        if cfg["model"]["provider"] == "agent":
            try:
                model = self._resolved_model(cfg["model"])
                if not credential_ready(model):
                    issues.append("Server credential is missing for the selected agent")
            except ModelError as error:
                issues.append(str(error))
        return issues

    def start(self, data):
        with self._lock:
            if not isinstance(data, dict) or data.keys() - {"id", "continuous", "allow_model"}:
                raise ValueError("Start requires a managed configuration id")
            if type(data.get("continuous", False)) is not bool:
                raise ValueError("continuous must be true or false")
            requested_id = self._id(data.get("id"))
            cfg = {"id": requested_id, **self._validate(self._read(self._config_path(requested_id)))}
            reviewed = cfg["model"]["provider"] == "reviewed"
            if not reviewed and data.get("allow_model") is not True:
                raise ValueError("Explicit allow_model=true is required for live model calls")
            issues = self._issues(cfg)
            if issues:
                raise ValueError("; ".join(issues))
            config_id = cfg.pop("id")
            records = self._inside(self.data_root, self.records_root)
            runtime = self._inside(self.data_root, self.runtime_root / config_id)
            runtime.mkdir(parents=True, exist_ok=True)
            cfg["backend"]["dataset"] = str((self.data_root / "datasets" / cfg["backend"]["dataset"]).resolve())
            cfg["discovery"]["network_enabled"] = not reviewed
            if reviewed:
                cfg["discovery"]["snapshot_evidence_level"] = "excerpt"
                cfg["model"]["provider"] = "fixture"
                cfg["model"]["reviewed"] = True
                cfg["model"]["max_candidates"] = max(len(cfg["model"].get("hypotheses", [])), len(cfg["model"].get("refinements", [])), 1)
            else:
                cfg["model"] = self._resolved_model(cfg["model"])
            cfg.update(research_root=str(records), runtime_root=str(runtime))
            path = self._inside(runtime, runtime / "campaign.json")
            if self.busy() or self._stopping or self._closed:
                raise ValueError("A research worker is already running or the app is closing")
            atomic_json(path, cfg)
            return self._spawn(path, config_id, False, data.get("continuous", False), reviewed=reviewed)

    def demo(self):
        with self._lock:
            if self.busy() or self._stopping or self._closed:
                raise ValueError("A research worker is already running or the app is closing")
            self._inside(self.data_root, self.demo_root)
            from autoresearch.demo import create_demo
            path = self._inside(self.demo_root, create_demo(self.demo_root))
            cfg = self._read(path)
            # Only internally created synthetic inputs are admitted to this path.
            for key in ("research_root", "runtime_root"):
                self._inside(self.demo_root, cfg[key])
            self._inside(self.demo_root, cfg["backend"]["dataset"])
            if (cfg["model"].get("provider") != "fixture"
                    or any("fixture_path" in cfg[section] for section in ("model", "discovery"))
                    or cfg["discovery"].get("network_enabled") is not False
                    or cfg["discovery"].get("arxiv_queries")):
                raise ValueError("Demo configuration must use offline fixtures")
            return self._spawn(path, "demo", True, False)

    def _spawn(self, path, config_id, demo, continuous, *, reviewed=False):
        if self.busy() or self._stopping or self._closed:
            raise ValueError("A research worker is already running or the app is closing")
        args = [sys.executable, "-m", "autoresearch", "run", str(path), "--continuous" if continuous else "--once"]
        if not demo and not reviewed:
            args.append("--allow-model")
        process = subprocess.Popen(args, cwd=self.source_root, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._stop_requested = False
        self._process = process
        self._worker = {**self._idle(), "running": True, "status": "starting", "config_id": config_id,
                        "demo": demo, "continuous": continuous, "reviewed": reviewed}
        threading.Thread(target=self._drain, args=(process,), daemon=True).start()
        return copy.deepcopy(self._worker)

    def _drain(self, process):
        try:
            for line in process.stdout:
                line = line.strip()[:8000]
                with self._lock:
                    if self._process is not process:
                        return
                    public = self._clean(line)
                    self._worker["log"] = (self._worker["log"] + [public])[-100:]
                    try:
                        update = json.loads(line)
                    except ValueError:
                        self._worker["last_error"] = public[:1000]
                        continue
                    if isinstance(update, dict):
                        for key in ("status", "campaign_id", "evaluations", "queued"):
                            if key in update:
                                self._worker[key] = self._clean(update[key])
                        self._worker["phase"] = self._clean(update.get("phase", update.get("stage", update.get("status"))))
            code = process.wait()
            with self._lock:
                if self._process is process:
                    self._worker["running"] = False
                    self._worker["status"] = "error" if code and not self._stop_requested else "stopped"
                    if code and not self._stop_requested and not self._worker["last_error"]:
                        self._worker["last_error"] = "Research worker exited with code " + str(code)
        finally:
            if process.stdout:
                process.stdout.close()

    def stop(self, *, _grace=65):
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return copy.deepcopy(self._worker)
            self._stop_requested = True
            self._stopping = True
            self._worker["status"] = "stopping"
            process.terminate()
        try:
            process.wait(timeout=_grace)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        with self._lock:
            self._stopping = False
            self._worker.update(running=False, status="stopped")
            return copy.deepcopy(self._worker)

    def close(self):
        with self._lock:
            self._closed = True
        self.stop(_grace=5)

    def _records(self, root, collection):
        if not root.exists():
            return []
        self._inside(self.data_root if root.is_relative_to(self.data_root) else self.source_root, root)
        folder = self._inside(root, root / collection)
        if not folder.exists():
            return []
        rows = []
        for path in sorted(folder.glob("*.json"))[-1000:]:
            try:
                self._inside(root, path)
                row = self._read(path)
                if row.get("id") == path.stem:
                    rows.append(row)
            except (ValueError, OSError):
                continue
        return sorted(rows, key=lambda r: (str(r.get("created_at", "")), r["id"]))

    def _roots(self):
        # Prefer managed records on ID collisions; never merge campaign evidence
        # across stores or open a writable ResearchStore for legacy discovery.
        return [(self.records_root, False), (self.source_root / "research" / "alphas", False),
                (self.demo_root / "research", True)]

    def _checkpoint(self, campaign_id, demo):
        folders = [self.demo_root / "runtime"] if demo else list(self.runtime_root.glob("*")) if self.runtime_root.exists() else []
        for folder in folders:
            try:
                self._inside(self.data_root, folder)
                path = self._inside(folder, folder / (campaign_id + ".json"))
                if path.exists():
                    state = self._read(path)
                    if state.get("campaign_id") == campaign_id:
                        state["queued"] = len(state.get("queue", []))
                        return state, "demo" if demo else folder.name
            except (ValueError, OSError):
                continue
        return {}, None

    def _summary(self, row, root, demo, records=None):
        records = records or {c: self._records(root, c) for c in _COLLECTIONS}
        state, config_id = self._checkpoint(row["id"], demo)
        ids = set(state.get("hypotheses", []))
        reused = {r.get("reused_experiment_id") for r in records["events"] if r.get("campaign_id") == row["id"]}
        experiments = [r for r in records["experiments"] if r.get("campaign_id") == row["id"] or r["id"] in reused]
        ids.update(r.get("hypothesis_id") for r in experiments)
        exp_ids = {r["id"] for r in experiments}
        hypotheses = [r for r in records["hypotheses"] if r["id"] in ids or r.get("campaign_id") == row["id"]]
        source_ids = {s for r in hypotheses for s in r.get("source_ids", [])}
        cfg = row.get("config", {})
        current_review_ids = set()
        for hypothesis in hypotheses:
            reused_ids = {r.get("reused_experiment_id") for r in records["events"]
                          if r.get("campaign_id") == row["id"] and r.get("hypothesis_id") == hypothesis["id"]}
            for experiment in experiments:
                if experiment.get("hypothesis_id") == hypothesis["id"] or experiment["id"] in reused_ids:
                    current_review_ids.add(stable_id("review", {"experiment": experiment["id"],
                                           "hypothesis": hypothesis["id"], "policy": cfg.get("policy", {})}))
        counts = {"hypotheses": len(hypotheses), "experiments": len(experiments),
                  "reviews": sum(r["id"] in current_review_ids for r in records["reviews"]),
                  "sources": sum(r["id"] in source_ids for r in records["sources"])}
        status = state.get("status", "recorded")
        owns_worker = self._worker.get("campaign_id") == row["id"]
        if owns_worker and self._worker.get("status") == "error":
            status = "error"
        elif state.get("queued", 0) and not (owns_worker and self.busy()):
            status = "interrupted"
        return {"id": row["id"], "name": cfg.get("name", row["id"]), "demo": demo,
                "reviewed": bool(cfg.get("model", {}).get("reviewed")),
                "provider": "reviewed" if cfg.get("model", {}).get("reviewed") else cfg.get("model", {}).get("provider"),
                "model_id": cfg.get("model", {}).get("model"),
                "agent_id": cfg.get("model", {}).get("agent_id"),
                "provider_name": cfg.get("model", {}).get("provider_name"),
                "created_at": row.get("created_at"), "counts": counts, "status": status, "queued": state.get("queued", 0),
                "limits": cfg.get("limits", {}), "config_id": config_id}

    def overview(self):
        with self._lock:
            configs = []
            if self.config_root.exists():
                self._inside(self.data_root, self.config_root)
                for path in sorted(self.config_root.glob("*.json"))[:100]:
                    try:
                        cfg = self.config(path.stem)
                        issues = self._issues(cfg)
                        configs.append({**cfg, "provider": cfg["model"]["provider"], "model": cfg["model"]["model"], "ready": not issues, "issues": issues})
                    except (ValueError, OSError):
                        configs.append({"id": path.stem, "name": path.stem, "ready": False, "issues": ["Invalid managed configuration"]})
            campaigns = []
            seen = set()
            for root, demo in self._roots():
                records = {c: self._records(root, c) for c in _COLLECTIONS}
                for row in records["campaigns"]:
                    if row["id"] not in seen:
                        campaigns.append(self._summary(row, root, demo, records))
                        seen.add(row["id"])
            worker = copy.deepcopy(self._worker)
            worker["running"] = self.busy()
            if worker.get("campaign_id"):
                state, _ = self._checkpoint(worker["campaign_id"], worker["demo"])
                worker.update({k: state[k] for k in ("evaluations", "queued") if k in state})
            return self._clean({"configs": configs, "campaigns": campaigns, "worker": worker,
                                "connections": self.agent_catalog(),
                                "credential_ready": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())})

    def detail(self, campaign_id):
        self._id(campaign_id)
        for root, demo in self._roots():
            records = {c: self._records(root, c) for c in _COLLECTIONS}
            manifest = next((r for r in records["campaigns"] if r["id"] == campaign_id), None)
            if manifest is None:
                continue
            state, _ = self._checkpoint(campaign_id, demo)
            ids = set(state.get("hypotheses", []))
            events = [r for r in records["events"] if r.get("campaign_id") == campaign_id]
            reused = {r.get("reused_experiment_id") for r in events}
            experiments = [r for r in records["experiments"] if r.get("campaign_id") == campaign_id or r["id"] in reused]
            ids.update(r.get("hypothesis_id") for r in experiments)
            hypotheses = [r for r in records["hypotheses"] if r["id"] in ids or r.get("campaign_id") == campaign_id]
            exp_ids = {r["id"] for r in experiments}
            reviews = [r for r in records["reviews"] if r.get("experiment_id") in exp_ids or r.get("campaign_id") == campaign_id]
            source_ids = {s for r in hypotheses for s in r.get("source_ids", [])}
            sources = [r for r in records["sources"] if r["id"] in source_ids]
            all_exp = {r["id"]: r for r in records["experiments"]}
            enriched = []
            current_review_ids = set()
            policy = manifest.get("config", {}).get("policy", {})
            review_lookup = {r["id"]: r for r in reviews}
            for row in hypotheses:
                reused_ids = {r.get("reused_experiment_id") for r in events if r.get("hypothesis_id") == row["id"]}
                linked = [r for r in experiments if r.get("hypothesis_id") == row["id"] or r["id"] in reused_ids]
                review_ids = [stable_id("review", {"experiment": e["id"], "hypothesis": row["id"], "policy": policy}) for e in linked]
                current_review_ids.update(review_ids)
                latest_review = review_lookup.get(review_ids[-1]) if review_ids else None
                lineage, seen, parent = [], set(), row.get("parent_experiment_id")
                while parent and parent not in seen and len(lineage) < 20:
                    seen.add(parent)
                    ancestor = all_exp.get(parent)
                    if ancestor is None:
                        break
                    lineage.append({k: ancestor.get(k) for k in ("id", "hypothesis_id", "expression", "parent_experiment_id")})
                    parent = ancestor.get("parent_experiment_id")
                enriched.append({**row, "experiments": linked, "result": linked[-1].get("result") if linked else None,
                                 "latest_review": latest_review,
                                 "sources": [r for r in sources if r["id"] in row.get("source_ids", [])], "lineage": lineage})
            campaign = {**manifest, **self._summary(manifest, root, demo, records)}
            return self._clean({"campaign": campaign, "hypotheses": enriched, "experiments": experiments,
                                "reviews": [r for r in reviews if r["id"] in current_review_ids], "sources": sources,
                                "events": events[-300:],
                                "checkpoint": state})
        raise ValueError("Research campaign not found")

    def series(self, campaign_id, experiment_id):
        """Authorize through the selected campaign, then resolve only its saved artifact."""
        self._id(campaign_id)
        self._id(experiment_id)
        for root, demo in self._roots():
            campaigns = self._records(root, "campaigns")
            if not any(row["id"] == campaign_id for row in campaigns):
                continue
            reused = {row.get("reused_experiment_id") for row in self._records(root, "events")
                      if row.get("campaign_id") == campaign_id}
            experiment = next((row for row in self._records(root, "experiments") if row["id"] == experiment_id
                               and (row.get("campaign_id") == campaign_id or row["id"] in reused)), None)
            if experiment is None:
                raise ValueError("Test does not belong to the selected campaign")
            public = {"campaign_id": campaign_id, "experiment_id": experiment_id,
                      "reused": experiment.get("campaign_id") != campaign_id, "periods": {}}
            result = experiment.get("result", {})
            if result.get("status") != "ok":
                return {**public, "status": "failed", "message": "This test failed; no completed daily PnL series is available."}
            origin = experiment.get("campaign_id")
            if not isinstance(origin, str) or not any(row["id"] == origin for row in campaigns):
                return {**public, "status": "missing", "message": "The original test's runtime is unavailable on this machine."}
            _, config_id = self._checkpoint(origin, demo)
            if not config_id:
                return {**public, "status": "missing", "message": "Daily series are not retained on this machine. Summary evidence is still available."}
            paths = result.get("artifacts", [])
            try:
                if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str):
                    raise ValueError("Invalid artifact reference")
                path = Path(paths[0])
                if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
                    raise ValueError("Invalid artifact path")
                engine = self.demo_root / "runtime/engine" if demo else self.runtime_root / self._id(config_id) / "engine"
                self._inside(self.data_root, engine)
                artifact_path = self._inside(engine, engine / path)
                if not artifact_path.is_file():
                    return {**public, "status": "missing", "message": "The saved daily series file is unavailable. Summary evidence is still available."}
                artifact = self._read(artifact_path)
                periods = {period: period_series(artifact, result, period) for period in ("development", "validation")}
                if periods["development"]["dates"][-1] >= periods["validation"]["dates"][0]:
                    raise ValueError("Periods overlap")
                booksize = result["settings"]["booksize"]
                if type(booksize) not in (int, float) or not math.isfinite(booksize) or booksize <= 0:
                    raise ValueError("Invalid book size")
                # Do not use _clean's generic 300-item display truncation on daily series.
                return {**public, "status": "ready", "periods": periods, "booksize": booksize,
                        "return_denominator": booksize / 2, "semantics": "Sum of daily net PnL after costs; fixed notional, not compounded. Each period starts at zero."}
            except (ValueError, OSError, KeyError, TypeError, OverflowError):
                return {**public, "status": "invalid", "message": "The retained series failed integrity or date checks; it cannot be charted."}
        raise ValueError("Research campaign not found")
