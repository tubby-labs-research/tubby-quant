"""Thread hygiene for lab processes: one BLAS thread each unless told otherwise.

The lab's numerics are many small matrix products (a network layer on a few
hundred rows, a ridge solve on a dozen columns, a rank correlation per
session). OpenBLAS starts one thread per core and keeps those threads spinning
between calls, which gains nothing on products this small and collapses as soon
as two lab processes share a machine: two eight-second demonstrations run side
by side took more than ten minutes each (2026-11-08, twelve cores), and a
benchmark that overlapped other work ran eight times slower than alone. So
every lab entry point caps the BLAS pool of its own process before any
numerical work, and every worker it starts does the same.
``TUBBY_LAB_BLAS_THREADS`` or ``tubby-lab --blas-threads N`` raise the cap on a
machine the lab has to itself. The cap changes wall time only, never a number.
"""

from __future__ import annotations

import os
from typing import TypedDict

#: The lab's own variable: the cap this process applied and its children inherit.
ENV_VAR = "TUBBY_LAB_BLAS_THREADS"
DEFAULT_THREADS = 1
#: Read by OpenBLAS, OpenMP runtimes, MKL, Apple's Accelerate and numexpr when they load.
LIBRARY_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                     "NUMEXPR_NUM_THREADS")


class ThreadPool(TypedDict):
    library: str
    threads_before: int


class ThreadLimit(TypedDict):
    threads: int
    #: ``threadpoolctl`` when the pools already loaded were capped too; ``environment`` when only
    #: libraries loaded from now on (and child processes) are.
    method: str
    pools: list[ThreadPool]


def requested_threads(threads: int | None = None) -> int:
    """The cap to apply: the explicit value, else ``TUBBY_LAB_BLAS_THREADS``, else one."""
    if threads is None:
        raw = os.environ.get(ENV_VAR, "").strip()
        if not raw:
            return DEFAULT_THREADS
        if not raw.isdigit() or int(raw) < 1:
            raise ValueError(f"{ENV_VAR} must be a positive integer, not {raw!r}")
        return int(raw)
    if type(threads) is not int or threads < 1:
        raise ValueError(f"--blas-threads must be a positive integer, not {threads!r}")
    return threads


def limit_blas_threads(threads: int | None = None) -> ThreadLimit:
    """Cap this process's BLAS and OpenMP pools to ``threads`` for the rest of its life.

    The environment is set first, so libraries that load later, and child processes, which
    inherit it, start capped; ``threadpoolctl`` then caps the pools already loaded. Without
    ``threadpoolctl`` the environment is all that is applied, which suffices whenever the cap
    runs before numpy is imported, as it does in ``tubby-lab``.
    """
    count = requested_threads(threads)
    os.environ[ENV_VAR] = str(count)
    for name in LIBRARY_VARIABLES:
        os.environ[name] = str(count)
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError:
        return {"threads": count, "method": "environment", "pools": []}
    pools: list[ThreadPool] = [{"library": str(pool.get("internal_api") or pool.get("user_api") or "?"),
                                "threads_before": int(pool.get("num_threads") or 0)}
                               for pool in threadpool_info()]
    threadpool_limits(limits=count)
    return {"threads": count, "method": "threadpoolctl", "pools": pools}
