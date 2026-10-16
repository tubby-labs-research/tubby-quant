"""Artificial, offline demonstration using the real local Alpha Lab engine."""

from pathlib import Path
from .backend import create_demo_dataset
from .controller import atomic_json


def create_demo(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "campaign.json"
    if config_path.exists():
        return config_path
    dataset = create_demo_dataset(root / "synthetic-data")
    source = {"id": "source_demo", "title": "Artificial pipeline fixture",
              "url": "https://example.invalid/autoresearch-fixture",
              "text": "This artificial fixture tests whether past five-day price changes predict later returns. "
                      "A ten-day comparison tests sensitivity to the measurement horizon."}
    base = {"title": "Five-day continuation fixture", "family": "synthetic-price-change",
            "mechanism": "Artificial diagnostic: compare continuation and reversal of recent price changes.",
            "expression": "rank(ts_delta(close, 5))", "source_ids": ["source_demo"],
            "evidence": [{"source_id": "source_demo", "quote": "past five-day price changes predict later returns"}],
            "expected_horizon": "one trading session", "required_fields": ["close"],
            "rationale": "Tests plumbing using synthetic prices; no economic claim."}
    child = {**base, "title": "Ten-day continuation fixture", "expression": "rank(ts_delta(close, 10))",
             "parent_experiment_id": "$parent", "question": "Does a longer measurement horizon reduce noise?",
             "expected_change": "A more stable validation result if the five-day measurement is noisy."}
    cfg = {"name": "offline-artificial-demo", "research_root": str(root / "research"),
           "runtime_root": str(root / "runtime"),
           "backend": {"dataset": str(dataset), "universe": "TOP8", "cost_bps": 5,
                       "development_start": "2023-06-01", "development_end": "2023-12-29",
                       "validation_start": "2024-01-01", "validation_end": "2024-09-30",
                       "holdout_start": "2024-10-01", "warmup": 100},
           "model": {"provider": "fixture", "max_candidates": 3,
                     "hypotheses": [base, {**base, "title": "Five-day reversal fixture",
                                          "expression": "-rank(ts_delta(close, 5))"}],
                     "refinements": [child]},
           "discovery": {"network_enabled": False, "fixture_sources": [source]},
           # Artificially high gate exercises the refinement branch in the demo.
           "policy": {"min_validation_days": 60, "min_sharpe": 100.0},
           "limits": {"max_experiments": 6, "max_hypotheses": 6, "max_per_family": 6,
                      "max_refinement_depth": 1, "max_model_calls": 6, "daily_model_calls": 6,
                      "max_job_errors": 2}}
    atomic_json(config_path, cfg)
    return config_path
