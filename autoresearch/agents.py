"""Local administrator-owned connection catalog. Browser requests only select IDs."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .models import APIAgentModel, ModelError


def load_agents(data_root):
    path = Path(os.environ.get("QUANT_RESEARCH_AGENTS_FILE") or Path(data_root) / "autoresearch" / "agents.local.json")
    if not path.exists():
        return {}
    try:
        if path.stat().st_size > 100_000:
            raise ValueError("catalog too large")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {"agents"} or not isinstance(value["agents"], list) or len(value["agents"]) > 20:
            raise ValueError("invalid catalog")
        result = {}
        required = {"id", "name", "provider_name", "model", "endpoint"}
        optional = {"api_key_env", "max_candidates", "max_tokens", "timeout_seconds", "revision"}
        for row in value["agents"]:
            if not isinstance(row, dict) or required - row.keys() or row.keys() - required - optional:
                raise ValueError("invalid connection fields")
            if (not isinstance(row["id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", row["id"])
                    or row["id"] in result or not isinstance(row["name"], str) or not row["name"].strip() or len(row["name"]) > 200):
                raise ValueError("invalid connection identity")
            if "revision" in row and (not isinstance(row["revision"], str) or len(row["revision"]) > 200):
                raise ValueError("invalid revision")
            config = {"provider": "agent", "agent_id": row["id"], **{k: v for k, v in row.items() if k not in {"id", "name"}}}
            APIAgentModel(config)  # offline validation only
            result[row["id"]] = {"name": row["name"], "config": config}
        return result
    except (OSError, ValueError, TypeError) as exc:
        raise ModelError("Research agent catalog is invalid; ask the server administrator to check it") from exc


def credential_ready(config):
    key_env = config.get("api_key_env")
    return not key_env or bool(os.environ.get(key_env, "").strip())
