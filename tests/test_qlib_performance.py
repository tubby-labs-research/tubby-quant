"""Deterministic checks for worker timing, RSS, and sampler cleanup."""

import io
import os
import threading
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from qlib_yahoo import performance


class RunPerformanceTests(unittest.TestCase):
    def test_snapshot_tracks_active_and_repeated_stages(self):
        clock = iter((10.0, 11.5, 12.0, 20.0, 21.0))
        memory = iter((100, 200, 150, 300))
        with patch.object(performance.time, "perf_counter", side_effect=lambda: next(clock)), \
             patch.object(performance, "_worker_rss", side_effect=lambda: next(memory)), \
             redirect_stdout(io.StringIO()) as output:
            with performance.RunPerformance() as perf:
                with perf.stage("load"):
                    active = perf.snapshot()
                with perf.stage("load"):
                    pass
                result = perf.snapshot()

        self.assertEqual(active["pid"], os.getpid())
        self.assertEqual(active["stages"]["load"], {"elapsed_seconds": 1.5, "peak_rss_bytes": 100})
        self.assertEqual(result["stages"]["load"], {"elapsed_seconds": 3.0, "peak_rss_bytes": 300})
        self.assertIsInstance(result["platform"], str)
        self.assertIn("Started load", output.getvalue())
        self.assertIn("Completed load", output.getvalue())

    def test_exception_stops_sampler_and_records_stage(self):
        threads = []
        real_thread = threading.Thread

        def make_thread(*args, **kwargs):
            thread = real_thread(*args, **kwargs)
            threads.append(thread)
            return thread

        with patch.object(performance.threading, "Thread", side_effect=make_thread), \
             patch.object(performance, "_worker_rss", return_value=123), \
             redirect_stdout(io.StringIO()):
            perf = performance.RunPerformance()
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with perf.stage("score"):
                    raise RuntimeError("failed")
        self.assertEqual(len(threads), 1)
        self.assertTrue(threads[0].daemon)
        self.assertFalse(threads[0].is_alive())
        self.assertGreaterEqual(perf.snapshot()["stages"]["score"]["elapsed_seconds"], 0)
        self.assertEqual(perf.snapshot()["stages"]["score"]["peak_rss_bytes"], 123)

    def test_missing_psutil_keeps_timing(self):
        clock = iter((1.0, 2.0))
        with patch.object(performance, "psutil", None), \
             patch.object(performance.time, "perf_counter", side_effect=lambda: next(clock)), \
             redirect_stdout(io.StringIO()):
            with performance.RunPerformance() as perf:
                with perf.stage("plain"):
                    pass
            self.assertEqual(perf.snapshot()["stages"]["plain"], {
                "elapsed_seconds": 1.0, "peak_rss_bytes": None,
            })

    def test_rejects_overlapping_stages(self):
        with patch.object(performance, "_worker_rss", return_value=None), redirect_stdout(io.StringIO()):
            with performance.RunPerformance() as perf:
                with perf.stage("outer"):
                    with self.assertRaisesRegex(RuntimeError, "overlap"):
                        with perf.stage("inner"):
                            pass
        self.assertEqual(list(perf.snapshot()["stages"]), ["outer"])

    def test_measure_alias_prints_flushed_progress(self):
        with patch.object(performance, "_worker_rss", return_value=None), \
             patch("builtins.print") as progress:
            with performance.RunPerformance() as perf:
                with perf.measure("simulate"):
                    pass
        self.assertEqual(progress.call_count, 2)
        self.assertEqual(progress.call_args_list[0].kwargs, {"flush": True})
        self.assertEqual(progress.call_args_list[1].kwargs, {"flush": True})


if __name__ == "__main__":
    unittest.main()
