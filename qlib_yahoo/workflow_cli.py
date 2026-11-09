"""Structured human/agent entrypoints for immutable workflow evidence."""
import json
import sys
import time
from pathlib import Path


def add_parser(commands):
    parser = commands.add_parser("workflow", help="Inspect recorded attempts and validate a saved Lab candidate.")
    sub = parser.add_subparsers(dest="workflow_command", required=True)
    listing = sub.add_parser("attempts", help="List recent submitted attempts, including failures.")
    listing.add_argument("--limit", type=int, default=100)
    detail = sub.add_parser("show", help="Read a sealed attempt and its compact result.")
    detail.add_argument("attempt_id")
    sub.add_parser("recover", help="Verify owners of unfinished attempts; mark only confirmed dead owners interrupted/unknown.")
    for name in ("preview", "declare", "trial", "freeze", "final", "view"):
        action = sub.add_parser("chronology-" + name, help="Chronological protocol/selection/evaluation: " + name)
        action.add_argument("--request", type=Path, required=True,
                            help="Plain JSON parameters matching POST workflow-chronology-" + name)
    for name in ("portfolio-preview", "portfolio-run"):
        action = sub.add_parser(name, help="Preview or execute one shared-capital portfolio of validated candidates.")
        action.add_argument("--request", type=Path, required=True,
                            help="JSON request containing validation_ids, coefficients, account_id and explicit risk limits.")
    for name in ("preview", "validate"):
        action = sub.add_parser(name, help=("Preview the candidate execution contract." if name == "preview"
                                            else "Validate a saved candidate in an isolated signed worker."))
        action.add_argument("attempt_id")
        action.add_argument("--initial-cash", type=float, default=100_000.0)
        action.add_argument("--cost-per-side", type=float, default=.0005)
        action.add_argument("--annual-borrow-rate", type=float, default=.004)


def _service_job_result(service, start_method, params, *, chronology=False):
    """Start through the HTTP service boundary and wait for sealed evidence.

    A stop can arrive before the service registers its worker's cancel hook (or
    before that worker can accept it). Keep retrying at the polling interval
    until accepted or terminal; a terminal state still needs its finished seal.
    """
    from .jobs import JobCancelled
    from .dataset import DatasetError
    started = getattr(service, start_method)(params)
    since = 0
    pending_cancel = False
    while True:
        try:
            job = service.job(started["id"], since)
            for line in job["log"]:
                print(line, file=sys.stderr, flush=True)
            since = job["log_index"]
            # _start seals console evidence before setting finished. Do not
            # terminate the CLI while its daemon job thread is still sealing.
            if job["finished"] is not None:
                break
            if pending_cancel and job["state"] == "running":
                pending_cancel = not service.cancel()
        except KeyboardInterrupt:
            pending_cancel = True
            try:
                pending_cancel = not service.cancel()
            except KeyboardInterrupt:
                pass
        # Even rejected startup cancellation must yield to the worker. Keep
        # repeated Ctrl+C during the pause pending without leaving the waiter.
        try:
            time.sleep(.05)
        except KeyboardInterrupt:
            pending_cancel = True
    label = "Chronology" if chronology else "Workflow"
    if job["state"] == "cancelled":
        raise JobCancelled(job["error"] or label + " job cancelled")
    if job["state"] != "completed":
        raise DatasetError(job["error"] or label + " job failed")
    if not chronology:
        return {"run_path": job["result"]}
    return {"schema_version": 1, "result_kind": "chronology", "job_id": job["id"],
            "attempt_id": job["attempt_id"], "run_path": job["result"], "receipt": job["receipt"]}


def _chronology_result(service, action, params):
    """Compatibility entrypoint using the same decorated boundary as HTTP."""
    if action not in ("trial", "final"):
        return getattr(service, "workflow_chronology_" + action)(params)
    return _service_job_result(service, "start_chronology_" + action, params, chronology=True)


def main(args):
    from .webapp.service import ConsoleService

    service = ConsoleService(args.data_root)
    if args.workflow_command.startswith("chronology-"):
        from .target_bundle import _json as strict_json
        params = strict_json(args.request.read_bytes())
        result = _chronology_result(service, args.workflow_command[len("chronology-"):], params)
    elif args.workflow_command == "attempts":
        result = service.workflow_attempts(args.limit)
    elif args.workflow_command == "show":
        result = service.workflow_detail(args.attempt_id)
    elif args.workflow_command == "recover":
        result = service.workflow_recover()
    elif args.workflow_command in ("portfolio-preview", "portfolio-run"):
        from .target_bundle import _json as strict_json
        params = strict_json(args.request.read_bytes())
        if args.workflow_command == "portfolio-preview":
            result = service.workflow_portfolio_preview(params)
        else:
            result = _service_job_result(service, "start_portfolio_validation", params)
    else:
        params = {key: getattr(args, key) for key in
                  ("attempt_id", "initial_cash", "cost_per_side", "annual_borrow_rate")}
        if args.workflow_command == "preview":
            result = service.workflow_preview(params)
        else:
            result = _service_job_result(service, "start_candidate_validation", params)
    print(json.dumps(result, indent=2, allow_nan=False))
