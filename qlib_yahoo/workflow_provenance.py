"""Small source/request identities for submitted work, never editor polling.

This is a wrapper identity, not a claim that a Git commit describes a dirty
checkout. The selected numerical dependencies are named rather than discovered
by recursively scanning the repository.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import math
from pathlib import Path
import platform
import sys

from .workflow_records import canonical_json_bytes

LAB_SOURCES = (
    "alphalab/batch.py", "alphalab/engine.py", "alphalab/fastexpr.py",
    "alphalab/ops.py", "alphalab/panel.py", "alphalab/settings.py",
    "alphalab/simulate.py", "classification.py", "dataset.py",
    "dataset_integrity.py", "panel.py", "packed.py", "reference.py", "universe.py",
    "sec_features.py", "sec_catalog.py", "sec_resolution.py", "sec_registry.py",
    "sec_registry_definitions.json", "sec_names.py", "sec_concepts.py",
    "sec_context_names.py", "sec_context_lookup.py", "sec_execution.py",
    "workflow_provenance.py",
)


def lab_implementation():
    """Hash bounded local implementation plus interpreter/numeric versions."""
    import numpy as np
    import pandas as pd

    root = Path(__file__).resolve().parent
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             for name in LAB_SOURCES}
    value = {
        "schema_version": 1, "policy": "lab-source-v1", "files": files,
        "runtime": {"python": platform.python_version(),
                    "implementation": sys.implementation.name,
                    "numpy": np.__version__, "pandas": pd.__version__},
    }
    value["sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return value


def request_value(value):
    """Record even invalid requests without pretending NaN is a valid setting."""
    if is_dataclass(value) and not isinstance(value, type):
        return request_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            return {"invalid_request_type": "mapping with non-string keys"}
        return {key: request_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [request_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_number": str(value)}
    if value is None or type(value) in (str, int, float, bool):
        return value
    return {"invalid_request_type": type(value).__name__}


def frozen_lab_result(result):
    """Separate reproducible evaluation from a changing comparison cohort."""
    from .alphalab.simulate import json_safe

    frozen = {key: value for key, value in result.items()
              if key not in {"cached", "attempt_id", "result_digest", "batch_attempt_id",
                             "self_correlation", "most_correlated_id"}}
    if "checks" in frozen:
        frozen["checks"] = [check for check in frozen["checks"]
                            if check.get("name") != "SELF_CORRELATION"]
    return json_safe(frozen)
