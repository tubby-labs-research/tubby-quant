"""Selected immutable Lab evidence -> actual signed Qlib validation.

This boundary does not map FASTEXPR to Qlib expressions. The candidate adapter
reproduces the selected Lab target weights; the existing signed engine consumes
those targets at the following open with its ordinary accounting.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from .dataset import DatasetError
from .workflow_provenance import request_value
from .workflow_records import EvidenceStore
from .workflow_submission import parent_attempts, submission_context


def execution_options(params):
    """Explicit versioned validation policy; Lab screening defaults stay put."""
    allowed = {"attempt_id", "initial_cash", "cost_per_side", "annual_borrow_rate"}
    if not isinstance(params, dict) or set(params) - allowed:
        raise DatasetError("validation accepts an attempt_id, initial_cash, cost_per_side and annual_borrow_rate only")
    if not isinstance(params.get("attempt_id"), str):
        raise DatasetError("choose a recorded Lab attempt")
    values = {"initial_cash": params.get("initial_cash", 100_000.0),
              "cost_per_side": params.get("cost_per_side", .0005),
              "annual_borrow_rate": params.get("annual_borrow_rate", .004)}
    for key, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise DatasetError(f"{key} must be a finite nonnegative number")
    if values["initial_cash"] <= 0:
        raise DatasetError("initial_cash must be positive")
    return values


def preview_validation(data_root, params):
    from .candidate_workflow import preview_candidate

    options = execution_options(params)
    preview = preview_candidate(data_root, params["attempt_id"])
    return {
        **preview, "execution": {
            "policy": "candidate-next-open-v1", **options,
            "annual_cash_rate": 0.0, "cash_borrowing": False,
            "strategy_mode": "signed", "signal_transform": "raw",
            "gross_exposure": 1.0,
            "gross_exposure_meaning": "input ceiling only; supplied smaller targets are not scaled up",
            "report_level": "summary", "holdings": "never",
            "omissions": ["real borrow availability and recalls", "broker margin/settlement",
                          "market impact", "taxes", "unmodeled exchange and regulatory charges"],
        },
        "claims": "Reproduced targets and simulated execution are not investment readiness.",
    }


def validate_candidate(data_root, params, *, cancel_file=None):
    """Run only in the isolated job worker; retain every failure and parent."""
    from .candidate_workflow import prepare_candidate_targets
    from .backtest import run_backtest
    from .analysis import validate_bundle

    root = Path(data_root).resolve()
    store = EvidenceStore(root)
    parents = list(parent_attempts() or [])
    source = params.get("attempt_id") if isinstance(params, dict) else None
    # An invalid identity can still be retained as a rejected submission, without
    # inventing a valid parent reference.
    parent_error = None
    if source:
        try:
            store.get(source)
            parents.append(source)
        except (ValueError, FileNotFoundError) as error:
            parent_error = error
    attempt = store.begin("candidate-validation", {"parameters": request_value(params)},
                          parents=list(dict.fromkeys(parents)))
    phase = "preview"
    try:
        if parent_error:
            raise DatasetError(f"Cannot load the selected candidate attempt: {parent_error}")
        preview = preview_validation(root, params)
        phase = "reproduction"
        from .target_bundle import _path
        target_parent = _path(root / "workflow" / "targets")
        target_parent.mkdir(parents=True, exist_ok=True)
        prepared = prepare_candidate_targets(root, source, root / "workflow" / "targets" / attempt["id"])
        if prepared["preview"]["result_digest"] != preview["result_digest"]:
            raise DatasetError("candidate evidence changed after preview")
        execution = preview["execution"]
        phase = "execution"
        with submission_context(store, attempt):
            run = run_backtest(
                Path(preview["dataset"]), "<supplied Lab targets>",
                start=preview["period"]["start"], end=preview["period"]["end"],
                initial_cash=execution["initial_cash"], cost_per_side=execution["cost_per_side"],
                annual_borrow_rate=execution["annual_borrow_rate"], annual_cash_rate=0.0,
                strategy_mode="signed", signal_transform="raw", gross_exposure=1.0,
                universe=preview["execution_universe"],
                warmup_sessions=preview["settings"]["warmup"],
                report_level="summary", holdings="never", runtime_profile="auto",
                target_bundle=prepared["bundle"], _cancel_file=cancel_file,
            )
        phase = "reconciliation"
        checked = validate_bundle(run)
        checked.assert_unchanged()
        result = {
            "run_path": str(run), "candidate": preview, "target_bundle": prepared["bundle"],
            "qualification": prepared["qualification"], "reproduction": prepared["reproduction"],
            "reconciliation": {"status": "passed", "checker": "analysis.validate_bundle",
                               "scope": "sealed saved signed-account ledger"},
            "manifest_sha256": hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest(),
            "summary": checked.summary,
        }
        store.finish(attempt["id"], "completed", result=result, artifacts={"run_path": str(run)})
        return Path(run)
    except BaseException as error:
        status = ("cancelled" if isinstance(error, KeyboardInterrupt) else
                  "rejected" if phase == "preview" else "failed")
        store.finish(attempt["id"], status,
                     error={"phase": phase, "type": type(error).__name__, "message": str(error)})
        raise
