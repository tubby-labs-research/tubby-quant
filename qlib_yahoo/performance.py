"""Small, process-local timing and peak RSS telemetry for worker stages."""

from __future__ import annotations

import os
import platform
import threading
import time
from contextlib import contextmanager
from typing import Iterator

try:
    import psutil
except ImportError:  # Bare hosts can still time stages.
    psutil = None


def _worker_rss() -> int | None:
    if psutil is None:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss
    except (OSError, psutil.Error):
        return None


class RunPerformance:
    """Measure sequential worker stages; repeated names accumulate time and peak RSS.

    ``with RunPerformance() as perf:`` and ``with perf.stage("load")`` are the
    public interface. ``snapshot()`` returns ``pid``, ``platform``, and a
    ``stages`` mapping to ``elapsed_seconds`` and ``peak_rss_bytes`` (or None).
    RSS is sampled every 0.25 seconds while a stage is active, so short spikes
    between samples may be missed. Only the current process is measured.
    """

    def __init__(self) -> None:
        self._stages: dict[str, dict[str, float | int | None]] = {}
        self._active: tuple[str, float] | None = None
        self._peak: int | None = None
        self._lock = threading.Lock()

    def __enter__(self) -> RunPerformance:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        return None

    def _sample(self) -> None:
        rss = _worker_rss()
        if rss is not None:
            with self._lock:
                self._peak = max(self._peak or 0, rss)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time one stage, including failed stages; stages cannot overlap."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("stage name must be a nonempty string")
        with self._lock:
            if self._active is not None:
                raise RuntimeError("performance stages cannot overlap")
            self._active = (name, time.perf_counter())
            self._peak = None

        stop = threading.Event()

        def sample_until_stopped() -> None:
            while not stop.wait(0.25):
                self._sample()

        sampler = threading.Thread(target=sample_until_stopped, daemon=True)
        completed = False
        try:
            self._sample()
            sampler.start()
            print(f"Started {name}", flush=True)
            yield
            completed = True
        finally:
            stop.set()
            if sampler.ident is not None:
                sampler.join()
            self._sample()
            with self._lock:
                elapsed = max(0.0, time.perf_counter() - self._active[1])
                previous = self._stages.get(name, {"elapsed_seconds": 0.0, "peak_rss_bytes": None})
                old_peak = previous["peak_rss_bytes"]
                peak = max(value for value in (old_peak, self._peak) if value is not None) if (
                    old_peak is not None or self._peak is not None
                ) else None
                self._stages[name] = {
                    "elapsed_seconds": previous["elapsed_seconds"] + elapsed,
                    "peak_rss_bytes": peak,
                }
                self._active = None
                self._peak = None
            memory = f", peak RSS {peak / 1024**2:.1f} MiB" if peak is not None else ""
            print(f"{'Completed' if completed else 'Failed'} {name} in {elapsed:.2f}s{memory}", flush=True)

    measure = stage

    def call(self, name, function, *args, **kwargs):
        """Measure a single function call without changing its return value."""
        with self.stage(name):
            return function(*args, **kwargs)

    def snapshot(self) -> dict[str, object]:
        """Return completed stages plus the current stage's elapsed time and peak."""
        with self._lock:
            stages = {name: values.copy() for name, values in self._stages.items()}
            if self._active is not None:
                name, started = self._active
                previous = stages.get(name, {"elapsed_seconds": 0.0, "peak_rss_bytes": None})
                old_peak = previous["peak_rss_bytes"]
                stages[name] = {
                    "elapsed_seconds": previous["elapsed_seconds"] + max(0.0, time.perf_counter() - started),
                    "peak_rss_bytes": max(value for value in (old_peak, self._peak) if value is not None) if (
                        old_peak is not None or self._peak is not None
                    ) else None,
                }
        return {"pid": os.getpid(), "platform": platform.system(), "stages": stages}
