"""Read process ownership before classifying an unfinished invocation.

No process is signaled or terminated. A missing PID or changed creation time is
evidence of death; age, a lock file, permission denial or an unavailable process
inspector is not. Remote-host records are deliberately left unknown.
"""
import socket

from .workflow_records import EvidenceStore


def owner_alive(owner):
    if owner.get("host") != socket.gethostname():
        return None
    try:
        import psutil
    except ImportError:
        return None
    try:
        process = psutil.Process(owner["pid"])
        born = process.create_time()
        if owner.get("create_time") is None:
            return None
        if abs(born - owner["create_time"]) > .01:
            return False  # The PID was reused by a different process.
        return process.is_running()
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, OSError, ValueError, KeyError):
        return None


def recover(data_root):
    store = EvidenceStore(data_root)
    recovered = store.recover_pending(owner_alive)
    remaining = [row for row in store.list_attempts(verify_results=False, limit=2**63 - 1)
                 if row["status"] == "pending"]
    return {
        "recovered": [row["id"] for row in recovered], "remaining_pending": len(remaining),
        "meaning": "Interrupted/unknown is not a completed simulation. Live or unverifiable owners remain pending.",
    }
