"""Thread hygiene: a lab process caps its own BLAS pool and every worker it starts does the same.

The cap changes wall time only. These tests check the precedence of the setting, that the
cap reaches both the environment (libraries loaded later, child processes) and the pools
already loaded, and that ``tubby-lab`` applies it before any command runs and refuses
nonsense. The process's pools are restored afterwards.
"""

from __future__ import annotations

import contextlib
import io
import os
import unittest
from unittest import mock

from autoresearch.lab import cli, runtime

try:  # the restoration needs the same optional library the cap uses
    import numpy  # noqa: F401
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:  # pragma: no cover - dependency-light environments
    threadpool_info = threadpool_limits = None  # type: ignore[assignment]

_ORIGINAL = {pool["user_api"]: int(pool["num_threads"]) for pool in threadpool_info()} if threadpool_info else {}


def tearDownModule() -> None:
    if threadpool_limits is not None and _ORIGINAL:
        threadpool_limits(limits=_ORIGINAL)


class RequestedThreadsTests(unittest.TestCase):
    def test_default_is_one_and_the_environment_or_an_explicit_value_overrides_it(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(runtime.ENV_VAR, None)
            self.assertEqual(runtime.requested_threads(), 1)
            os.environ[runtime.ENV_VAR] = " 4 "
            self.assertEqual(runtime.requested_threads(), 4)
            self.assertEqual(runtime.requested_threads(2), 2, "an explicit value wins over the environment")
            for bad in ("0", "-3", "two", "1.5"):
                os.environ[runtime.ENV_VAR] = bad
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, runtime.ENV_VAR):
                    runtime.requested_threads()
            for bad_value in (0, -1, True, 2.0):
                with self.subTest(bad=bad_value), self.assertRaisesRegex(ValueError, "positive integer"):
                    runtime.requested_threads(bad_value)  # type: ignore[arg-type]


class LimitTests(unittest.TestCase):
    def test_the_cap_reaches_the_environment_and_every_loaded_pool(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            applied = runtime.limit_blas_threads(1)
            self.assertEqual(applied["threads"], 1)
            for name in runtime.LIBRARY_VARIABLES + (runtime.ENV_VAR,):
                self.assertEqual(os.environ[name], "1", name)
            if threadpool_info is None:
                self.assertEqual(applied["method"], "environment")
                self.assertEqual(applied["pools"], [])
                return
            self.assertEqual(applied["method"], "threadpoolctl")
            blas = [pool for pool in threadpool_info() if pool["user_api"] == "blas"]
            self.assertTrue(blas, "numpy is installed but exposes no BLAS pool")
            self.assertEqual({pool["num_threads"] for pool in blas}, {1}, blas)
            self.assertEqual({pool["library"] for pool in applied["pools"]} >= {pool["internal_api"] for pool in blas}, True)

    def test_a_larger_cap_is_applied_as_given(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            applied = runtime.limit_blas_threads(2)
            self.assertEqual(applied["threads"], 2)
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "2")
            if threadpool_info is not None:
                for pool in threadpool_info():
                    if pool["user_api"] == "blas":
                        self.assertLessEqual(pool["num_threads"], 2, pool)


class CommandLineTests(unittest.TestCase):
    def test_the_flag_caps_the_process_before_the_command_runs(self):
        with mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch.object(cli, "limit_blas_threads", wraps=cli.limit_blas_threads) as cap, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(["--blas-threads", "2", "domains"]), 0)
            cap.assert_called_once_with(2)
            self.assertEqual(os.environ[runtime.ENV_VAR], "2")
            self.assertIn("alpha", out.getvalue())
        with mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch.object(cli, "limit_blas_threads", wraps=cli.limit_blas_threads) as cap, \
                contextlib.redirect_stdout(io.StringIO()):
            os.environ.pop(runtime.ENV_VAR, None)
            self.assertEqual(cli.main(["domains"]), 0)
            cap.assert_called_once_with(None)
            self.assertEqual(os.environ[runtime.ENV_VAR], "1", "the default cap is one thread")

    def test_nonsense_is_refused_before_anything_runs(self):
        with mock.patch.dict(os.environ, {}, clear=False), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(cli.main(["--blas-threads", "0", "domains"]), 2)
            self.assertIn("positive integer", err.getvalue())
        with mock.patch.dict(os.environ, {runtime.ENV_VAR: "three"}), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(cli.main(["domains"]), 2)
            self.assertIn(runtime.ENV_VAR, err.getvalue())


if __name__ == "__main__":
    unittest.main()
