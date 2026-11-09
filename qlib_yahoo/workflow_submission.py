"""Shared submitted-invocation context for console, workers and Python callers.

Metadata reads and editor checks deliberately do not enter this context.
Thread workers copy only explicit attempt IDs; a process-local context is never
used as proof of durable completion.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import inspect
from pathlib import Path

from .dataset import DatasetError
from .workflow_provenance import request_value
from .workflow_records import EvidenceStore

_CURRENT = ContextVar("quant_submitted_attempt", default=None)


def current_submission():
    return _CURRENT.get()


def parent_attempts():
    current = current_submission()
    return [current["id"]] if current else None


@contextmanager
def submission_context(store, attempt):
    token = _CURRENT.set({"store": store, "id": attempt["id"]})
    try:
        yield
    finally:
        _CURRENT.reset(token)


def record_console_submission(kind):
    """Persist a button/API submission before its ordinary cheap validation."""
    def decorate(method):
        @wraps(method)
        def call(self, params):
            store = EvidenceStore(self.data_root)
            attempt = store.begin(kind, {"parameters": request_value(params)})
            try:
                with submission_context(store, attempt):
                    response = method(self, params)
                # Normal asynchronous starts are completed by the job thread.
                if not isinstance(response, dict) or response.get("attempt_id") != attempt["id"]:
                    store.finish(attempt["id"], "completed", result=request_value(response))
                return response
            except BaseException as error:
                status = ("cancelled" if isinstance(error, KeyboardInterrupt) else
                          "rejected" if isinstance(error, (DatasetError, ValueError, TypeError)) else "failed")
                if store.get(attempt["id"])["status"] == "pending":
                    store.finish(attempt["id"], status,
                                 error={"type": type(error).__name__, "message": str(error)})
                raise
        return call
    return decorate


def run_worker_submission(request, dispatch):
    """Use the same parent contract for the structured worker and console."""
    request = dict(request)
    envelope = request.pop("_workflow", None)
    if envelope is not None and (not isinstance(envelope, dict)
                                 or set(envelope) != {"data_root", "parents"}):
        raise ValueError("invalid workflow submission envelope")
    data_root = envelope["data_root"] if envelope else request.get("data_root")
    if data_root is None and request.get("kind") == "backtest" and request.get("dataset"):
        dataset = Path(request["dataset"]).resolve()
        if dataset.parent.name == "datasets" and dataset.is_dir():
            data_root = dataset.parent.parent
    if data_root is None:
        return dispatch(request)
    store = EvidenceStore(data_root)
    attempt = store.begin("worker-" + str(request.get("kind")), request_value(request),
                          parents=envelope["parents"] if envelope else None)
    try:
        with submission_context(store, attempt):
            result = dispatch(request)
        store.finish(attempt["id"], "completed", result={"result_path": str(result)})
        return result
    except BaseException as error:
        status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        store.finish(attempt["id"], status, error={"type": type(error).__name__, "message": str(error)})
        raise


def record_backtest_submission(method):
    """Retain direct CLI/Python invocations as well as worker child attempts.

Legacy nonstandard dataset layouts keep working. Without an inherited storage
context, only the established <data_root>/datasets/<snapshot> layout supplies a
safe implicit destination for evidence; invalid arbitrary paths are not a reason
to create workflow directories in an unrelated ancestor.
    """
    signature = inspect.signature(method)

    @wraps(method)
    def call(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        current = current_submission()
        if current:
            store = current["store"]
        else:
            dataset = Path(bound.arguments["dataset"]).resolve()
            if dataset.parent.name != "datasets" or not dataset.is_dir():
                return method(*args, **kwargs)
            store = EvidenceStore(dataset.parent.parent)
        source = Path(method.__code__.co_filename)
        attempt = store.begin("qlib-validation", request_value(dict(bound.arguments)),
                              parents=parent_attempts(),
                              source={"wrapper_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        try:
            with submission_context(store, attempt):
                run = method(*args, **kwargs)
            manifest = Path(run) / "manifest.json"
            result = {"run_path": str(run),
                      "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
            store.finish(attempt["id"], "completed", result=result, artifacts={"run_path": str(run)})
            return run
        except BaseException as error:
            store.finish(attempt["id"], "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                         error={"type": type(error).__name__, "message": str(error)})
            raise
    return call
