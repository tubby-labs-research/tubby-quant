"""Explicit campaign configuration; credentials never belong in JSON files."""

from __future__ import annotations

import copy
import json
from pathlib import Path


DEFAULT_LIMITS = {
    "max_experiments": 30,
    "max_hypotheses": 30,
    "max_per_family": 6,
    "max_refinement_depth": 2,
    "max_model_calls": 30,
    "daily_model_calls": 10,
    "max_job_errors": 2,
}


def default_research_root() -> Path:
    from qlib_yahoo import default_data_root
    return Path(default_data_root()) / "autoresearch" / "records"


def default_runtime_root() -> Path:
    from qlib_yahoo import default_data_root
    return Path(default_data_root()) / "autoresearch" / "runtime"


def validate_config(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("campaign configuration must be an object")
    cfg = copy.deepcopy(value)
    if not isinstance(cfg.get("name"), str) or not cfg["name"].strip():
        raise ValueError("campaign name is required")
    for name in ("backend", "model", "discovery"):
        if not isinstance(cfg.get(name), dict):
            raise ValueError(f"{name} configuration is required")
    if not isinstance(cfg.get("policy", {}), dict):
        raise ValueError("policy must be an object")
    limits = {**DEFAULT_LIMITS, **cfg.get("limits", {})}
    if set(limits) != set(DEFAULT_LIMITS):
        raise ValueError("unknown campaign limit")
    for key, number in limits.items():
        if type(number) is not int or number < (0 if key == "max_refinement_depth" else 1):
            raise ValueError(f"{key} must be a positive integer (depth may be zero)")
    cfg["limits"] = limits
    for key, default in (("discovery_interval_seconds", 3600), ("idle_seconds", 30)):
        number = cfg.setdefault(key, default)
        if type(number) is not int or number < 1:
            raise ValueError(f"{key} must be a positive integer")
    # Key names are allowed; actual credentials are environment-only.
    def inspect(obj):
        if isinstance(obj, dict):
            for key, item in obj.items():
                if key.lower() in {"api_key", "token", "password", "secret", "authorization"}:
                    raise ValueError("put credentials in environment variables, not campaign files")
                inspect(item)
        elif isinstance(obj, list):
            for item in obj:
                inspect(item)
    inspect(cfg)
    return cfg


def load_config(path: Path) -> dict:
    path = Path(path).resolve()
    cfg = validate_config(json.loads(path.read_text(encoding="utf-8")))
    for section, key in (("backend", "dataset"), ("model", "fixture_path"),
                         ("discovery", "fixture_path")):
        if cfg[section].get(key):
            candidate = Path(cfg[section][key]).expanduser()
            cfg[section][key] = str(candidate if candidate.is_absolute() else path.parent / candidate)
    for key in ("research_root", "runtime_root"):
        if cfg.get(key):
            candidate = Path(cfg[key]).expanduser()
            cfg[key] = str(candidate if candidate.is_absolute() else path.parent / candidate)
    return cfg
