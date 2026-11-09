"""One fresh USD simulation account for one/two previously validated signals.

Use immutable target artifacts from linked candidate validations, not a sum of
their return series. This is a historical simulation, not a persistent paper
account. Numerical neutralization remains part of each supplied candidate.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from .dataset import DatasetError
from .portfolio import RiskLimits
from .workflow_records import EvidenceStore, canonical_json_bytes
from .workflow_provenance import request_value
from .workflow_submission import parent_attempts, submission_context


LIMIT_NAMES = ("gross", "net", "name", "group_gross", "group_net", "turnover", "liquidity")
_FIELDS = {"validation_ids", "coefficients", "account_id", "initial_cash",
           "cost_per_side", "annual_borrow_rate", "limits", "group_overlay", "group_level"}


def _allocation_sources():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("portfolio.py", "portfolio_workflow.py", "target_bundle.py")}


def portfolio_options(params):
    """Validate small plain inputs only, without datasets, panels or Qlib."""
    if not isinstance(params, dict) or set(params) - _FIELDS:
        raise DatasetError("unrecognized portfolio options")
    ids, coefficients = params.get("validation_ids"), params.get("coefficients")
    if (not isinstance(ids, list) or len(ids) not in (1, 2)
            or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids)):
        raise DatasetError("select one or two distinct completed candidate-validation attempts")
    if (not isinstance(coefficients, list) or len(coefficients) != len(ids)
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
                   for x in coefficients) or math.fsum(coefficients) > 1):
        raise DatasetError("provide one capital fraction per signal; fractions must be nonnegative and total at most one")
    account = params.get("account_id")
    if (not isinstance(account, str) or not account.strip() or account != account.strip()
            or len(account) > 120 or any(ord(c) < 32 for c in account)):
        raise DatasetError("give this historical simulation a short account name")
    from .validation_workflow import execution_options
    execution = execution_options({"attempt_id": ids[0], **{
        k: params[k] for k in ("initial_cash", "cost_per_side", "annual_borrow_rate") if k in params}})
    limits = params.get("limits")
    if not isinstance(limits, dict) or set(limits) != set(LIMIT_NAMES):
        raise DatasetError("declare all seven risk limits explicitly; null disables an individual limit")
    try:
        validated_limits = asdict(RiskLimits(**limits))
    except (TypeError, ValueError) as error:
        raise DatasetError(str(error)) from error
    overlay, level = params.get("group_overlay"), params.get("group_level")
    if overlay is not None and (not isinstance(overlay, str) or not overlay.strip()):
        raise DatasetError("group_overlay must be an explicit path or null")
    if level not in (None, "sector", "industry", "country"):
        raise DatasetError("group_level must be sector, industry, country or null")
    needs_groups = limits["group_gross"] is not None or limits["group_net"] is not None
    if needs_groups and (not overlay or not level):
        raise DatasetError("enabled group limits require an explicit dated group overlay and level")
    if (overlay is None) != (level is None):
        raise DatasetError("provide both a group overlay and its level, or neither")
    return {
        "validation_ids": list(ids), "coefficients": [float(c) for c in coefficients],
        "account_id": account, **execution, "limits": validated_limits,
        "group_overlay": None if overlay is None else str(Path(overlay).expanduser().resolve()),
        "group_level": level,
    }


def _validation(store, identity):
    """Verify compact validation receipts; defer large artifact reads to Run."""
    record = store.get(identity)
    if (record["kind"] != "candidate-validation" or record["status"] != "completed"
            or record["error"] is not None or not record["result_digest"]):
        raise DatasetError("portfolio inputs must be completed candidate-validation attempts")
    result = store.get_result(record["result_digest"])
    if (not isinstance(result, dict) or result.get("reproduction", {}).get("status") != "matched"
            or result.get("reconciliation", {}).get("status") != "passed"):
        raise DatasetError("selected validation lacks reproduced targets and checked signed-ledger evidence")
    from .candidate_workflow import preview_candidate
    saved = result.get("candidate") or {}
    source_id = saved.get("source_attempt_id")
    preview = preview_candidate(store.root.parent, source_id)
    if (preview["result_digest"] != saved.get("result_digest")
            or source_id not in record["parents"]
            or preview["source_bindings"] != saved.get("source_bindings")):
        raise DatasetError("validation no longer binds the exact selected candidate evidence")
    run = Path(result["run_path"])
    # Guard local request paths before even the cheap manifest read.
    from .target_bundle import _path
    run = _path(run)
    if not run.is_relative_to(store.root.parent / "runs"):
        raise DatasetError("selected validation run is outside this data root")
    raw = (run / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != result.get("manifest_sha256"):
        raise DatasetError("selected validation manifest changed")
    manifest = json.loads(raw)
    target = (manifest.get("target_bundle") or {}).get("identity") or {}
    envelope = result.get("target_bundle") or {}
    if (manifest.get("status") != "completed" or manifest.get("input_mode") != "target_weights"
            or not envelope.get("path") or not envelope.get("sha256")
            or target.get("path") != envelope["path"]
            or target.get("sha256") != envelope["sha256"]):
        raise DatasetError("selected validation manifest target binding differs")
    return record, result, preview


def preview_portfolio(data_root, params):
    """Metadata-only comparison; validation artifacts are checked again on Run."""
    options = portfolio_options(params)
    root = Path(data_root).resolve()
    if (options["group_overlay"] is not None
            and not Path(options["group_overlay"]).is_relative_to(root / "classifications")):
        raise DatasetError("portfolio group overlay must belong to this data root's classifications")
    store = EvidenceStore(root)
    bound = [_validation(store, item) for item in options["validation_ids"]]
    first = bound[0][2]
    for _, _, preview in bound:
        if (preview["dataset"], preview["period"], preview["execution_universe"],
                preview["source_bindings"]["price_identity_sha256"]) != (
                first["dataset"], first["period"], first["execution_universe"],
                first["source_bindings"]["price_identity_sha256"]):
            raise DatasetError("portfolio legs require identical price source, period and prepared universe")
        if not Path(preview["dataset"]).is_relative_to(root / "datasets"):
            raise DatasetError("portfolio dataset must belong to this data root")
    sources = [result["candidate"]["source_attempt_id"] for _, result, _ in bound]
    if len(set(sources)) != len(sources):
        raise DatasetError("two validations of the same candidate attempt are not two signals")
    return {
        "schema_version": 1, "request": options,
        "legs": [{"validation_id": record["id"], "validation_result_digest": record["result_digest"],
                  "candidate_attempt_id": preview["source_attempt_id"],
                  "result_digest": preview["result_digest"], "expression": preview["expression"],
                  "coefficient": coefficient, "qualification": result["qualification"]}
                 for (record, result, preview), coefficient in zip(bound, options["coefficients"])],
        "period": first["period"], "dataset": first["dataset"],
        "execution_universe": first["execution_universe"],
        "warmup_sessions": max(item[2]["settings"]["warmup"] for item in bound),
        "policy": {"version": "portfolio_fixed_v1", "account_id": options["account_id"],
                   "initial_cash": options["initial_cash"], "coefficients": options["coefficients"],
                   "limits": options["limits"], "account_lifetime": "one fresh historical simulation",
                   "neutralization": "already part of each candidate; allocation applies none",
                   "unused_fraction": 1 - math.fsum(options["coefficients"])},
        "execution": {
            **{key: options[key] for key in ("initial_cash", "cost_per_side", "annual_borrow_rate")},
            "annual_cash_rate": 0.0, "cash_borrowing": False, "restricted_short_proceeds": True,
            "timing": "saved close decisions, next-session-open whole-share execution",
            "report_level": "summary", "holdings": "never",
        },
        "risk_contract": {
            "limits": options["limits"],
            "disabled_limits": [key for key, value in options["limits"].items() if value is None],
            "turnover": "full absolute traded reference notional divided by equity",
            "liquidity": "absolute proposed/executed shares divided by prior-session shares volume",
            "missing_inputs": "block rebalance; do not invent prices, volume or groups",
            "violation_policy": "block rebalance, retain actual positions and report residual exposures",
            "group_overlay": options["group_overlay"], "group_level": options["group_level"],
            "omissions": ["real borrow availability/recalls", "broker margin/settlement",
                          "market impact", "taxes", "live trading"],
        },
        "qualification": {
            "status": "exploratory_simulation",
            "source_validation_receipts": "verified",
            "large_source_and_target_artifacts": "pending explicit Run verification",
            "limitations": ["shared allocation does not improve the underlying data qualification",
                           "fixed weights are capital fractions, not a portfolio optimizer",
                           "risk limits do not certify investment readiness"],
        },
    }


def run_portfolio(data_root, params, *, cancel_file=None):
    """Actual shared-capital signed execution. Called only in an isolated worker."""
    from .analysis import validate_bundle
    from .backtest import run_backtest
    from .portfolio import NormalizedTargets, allocate_fixed, make_fixed_policy
    from .preparation import check_cancel
    from .target_bundle import _path, load_bundle, write_bundle

    root, phase = Path(data_root).resolve(), "preview"
    store = EvidenceStore(root)
    parents = list(parent_attempts() or [])
    requested_ids = params.get("validation_ids") if isinstance(params, dict) else None
    for identity in requested_ids if isinstance(requested_ids, list) else []:
        try:
            store.get(identity)
            parents.append(identity)
        except (ValueError, FileNotFoundError):
            pass  # Invalid submitted identities still get a rejected record.
    attempt = store.begin("portfolio-validation", {"parameters": request_value(params)},
                          parents=list(dict.fromkeys(parents)))
    try:
        allocation_sources = _allocation_sources()
        preview = preview_portfolio(root, params)
        options = preview["request"]
        phase = "verified_target_reuse"
        inputs, verified = [], []
        for identity, leg in zip(options["validation_ids"], preview["legs"]):
            check_cancel(cancel_file)
            record, saved, candidate = _validation(store, identity)
            if record["result_digest"] != leg["validation_result_digest"]:
                raise DatasetError("validation evidence changed after preview")
            checked = validate_bundle(Path(saved["run_path"]))
            checked.assert_unchanged()
            target = load_bundle(saved["target_bundle"])
            metadata = target["metadata"]
            if (metadata["attempt_ids"] != [candidate["source_attempt_id"]]
                    or metadata["candidate_result_digests"] != [candidate["result_digest"]]
                    or metadata["weight_policy"] != "lab_close_targets_v1"):
                raise DatasetError("validation targets no longer bind the selected Lab evidence")
            inputs.append(NormalizedTargets(
                dates=tuple(target["dates"]), symbols=tuple(target["symbols"]),
                weights=target["weights"], candidate_result_digest=candidate["result_digest"],
                attempt_id=candidate["source_attempt_id"],
                dataset_manifest_sha256=metadata["dataset_manifest_sha256"],
                price_identity=canonical_json_bytes(metadata["price_identity"]).decode("utf-8")))
            verified.append(checked)
        phase = "allocation"
        policy = make_fixed_policy(inputs, options["coefficients"],
                                   account_id=options["account_id"], capital=options["initial_cash"],
                                   policy_id="portfolio-" + attempt["id"])
        allocated = allocate_fixed(policy, inputs)
        parent = _path(root / "workflow" / "targets")
        parent.mkdir(parents=True, exist_ok=True)
        metadata = {**allocated.to_metadata(),
                    # Preserve the native target contract's structured price identity.
                    "price_identity": json.loads(inputs[0].price_identity),
                    "validation_ids": options["validation_ids"],
                    "validation_result_digests": [leg["validation_result_digest"] for leg in preview["legs"]],
                    "portfolio_attempt_id": attempt["id"], "risk_limits": options["limits"],
                    "allocation_sources_sha256": allocation_sources}
        if _allocation_sources() != allocation_sources:
            raise DatasetError("allocation implementation changed during preparation")
        bundle = write_bundle(parent / attempt["id"], dates=allocated.dates,
                              symbols=allocated.symbols, weights=allocated.weights, metadata=metadata)
        del allocated, inputs
        check_cancel(cancel_file)
        phase = "execution"
        with submission_context(store, attempt):
            run = run_backtest(
                Path(preview["dataset"]), "<fixed allocation of validated Lab targets>",
                start=preview["period"]["start"], end=preview["period"]["end"],
                initial_cash=options["initial_cash"], cost_per_side=options["cost_per_side"],
                annual_borrow_rate=options["annual_borrow_rate"], annual_cash_rate=0.0,
                strategy_mode="signed", signal_transform="raw", gross_exposure=1.0,
                universe=preview["execution_universe"], warmup_sessions=preview["warmup_sessions"],
                report_level="summary", holdings="never", runtime_profile="auto",
                target_bundle=bundle, _cancel_file=cancel_file,
                portfolio_risk={"schema_version": 1, "limits": options["limits"],
                                "group_overlay": options["group_overlay"],
                                "group_level": options["group_level"],
                                "provenance": {
                                    "portfolio_attempt_id": attempt["id"],
                                    "policy_id": metadata["allocation_policy"]["policy_id"],
                                    "validation_ids": options["validation_ids"],
                                    "allocation_sources_sha256": allocation_sources}})
        phase = "reconciliation"
        checked = validate_bundle(run)
        checked.assert_unchanged()
        for source in verified:
            source.assert_unchanged()
        if _allocation_sources() != allocation_sources:
            raise DatasetError("allocation implementation changed during execution")
        counts = checked.summary["portfolio_risk"]["counts"]
        exceptions = (counts["blocked_rebalances"]
                      + counts["post_fill"]["blocked"] + counts["post_fill"]["missing_inputs"])
        qualification = {
            **preview["qualification"],
            "large_source_and_target_artifacts": "verified",
            "source_drift": "no_change_observed_during_request",
            "shared_ledger_reconciliation": "passed",
            "risk_outcome": "exceptions_recorded" if exceptions else "checks_passed",
            "risk_counts": counts,
        }
        result = {
            "run_path": str(run), "portfolio": preview, "allocation": metadata,
            "target_bundle": bundle, "qualification": qualification,
            "manifest_sha256": hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest(),
            "reconciliation": {"status": "passed", "checker": "analysis.validate_bundle",
                               "scope": "one shared signed cash/share account"},
            "summary": checked.summary,
        }
        store.finish(attempt["id"], "completed", result=result, artifacts={"run_path": str(run)})
        return Path(run)
    except BaseException as error:
        store.finish(attempt["id"],
                     "cancelled" if isinstance(error, KeyboardInterrupt) else
                     "rejected" if phase == "preview" else "failed",
                     error={"phase": phase, "type": type(error).__name__, "message": str(error)})
        raise
