"""Unwired binary64-load replay candidate, enabled only by a verified build.

No build, download or installation is triggered. Use QLIB_SESSION_LOOPS_MANIFEST
with an isolated signed child. Optional QLIB_CYTHON_REPLAY_RECEIPT measures the
ENTIRE try_execute_orders boundary (eligibility, rows, packing, kernel and replay,
including candidate guards and owned snapshot), never a replay-only speed claim.
Synthetic timing is diagnostic; it does not establish worker acceptance.
"""
import ast
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CHILD = "QLIB_CYTHON_REPLAY_CHILD"
MANIFEST = "QLIB_SESSION_LOOPS_MANIFEST"


def boundary_parts(bridge):
    """Extract the actual current Python prefix/suffix, without a copied oracle.

    The prefix executes EVERY existing guard/pack/kernel operation. The suffix
    is compiled from the unchanged original AST, using the original globals.
    Fail if its known boundary changed rather than silently selecting a subset.
    """
    tree = ast.parse(inspect.getsource(bridge.try_execute_orders))
    original = tree.body[0]
    matches = [i for i, statement in enumerate(original.body)
               if isinstance(statement, ast.Assign) and len(statement.targets) == 1
               and isinstance(statement.targets[0], ast.Name)
               and statement.targets[0].id == "execute_result"]
    if len(matches) != 1:
        raise RuntimeError("Python replay boundary changed; review required")
    split = matches[0]
    suffix = ast.FunctionDef(
        name="reference_replay", args=ast.arguments(
            posonlyargs=[], args=[ast.arg(arg=name) for name in
                                ("position", "accum", "orders", "rows", "result")],
            kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=original.body[split:], decorator_list=[])
    original.name = "prepare_boundary"
    original.body = original.body[:split] + [ast.Return(value=ast.Tuple(
        elts=[ast.Name(id=name, ctx=ast.Load()) for name in
              ("position", "accum", "orders", "rows", "result")], ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[original, suffix], type_ignores=[]))
    namespace = dict(bridge.__dict__)
    exec(compile(module, inspect.getsourcefile(bridge.try_execute_orders), "exec"), namespace)
    return namespace["prepare_boundary"], namespace["reference_replay"]


class LauncherTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get(MANIFEST), "explicit reviewed extension receipt required")
    @unittest.skipIf(os.environ.get(CHILD) == "1", "isolated child")
    def test_reviewed_signed_child(self):
        environment = os.environ.copy()
        environment[CHILD] = "1"
        program = (
            "from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); "
            "import unittest; suite=unittest.defaultTestLoader.loadTestsFromName("
            "'tests.test_qlib_cython_replay.ReplayChild'); "
            "r=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not r.wasSuccessful())")
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=240)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(os.environ.get(CHILD) == "1", "requires verified extension child")
class ReplayChild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_qlib_ordered_signed import OrderedChild
        from tests.test_qlib_session_loops import load_reviewed_extension
        OrderedChild.setUpClass()
        cls.fixture = OrderedChild()
        cls.extension, cls.load_seconds, cls.manifest = load_reviewed_extension()
        cls.prepare, cls.reference = map(staticmethod, boundary_parts(cls.fixture.bridge))

    def complete(self, exchange, account, orders, *, candidate, compiled=False):
        prepared = self.prepare(exchange, account, orders, compiled=compiled)
        if prepared is NotImplemented or prepared == []:
            return prepared
        position, accum, original_orders, rows, result = prepared
        if candidate:
            replay = self.extension.try_replay_records(position, accum, original_orders, rows, result.records)
            if replay is not NotImplemented:
                return replay
        return self.reference(*prepared)

    def pair(self, specifications, *, holdings=None, cash=1000., frame=None, compiled=False):
        f = self.fixture
        outcomes = []
        for candidate in (False, True):
            account, orders = f.account(cash, holdings), f.orders(specifications)
            exchange = f.exchange(True, frame)
            prepared = self.prepare(exchange, account, orders, compiled=compiled)
            self.assertIsNot(prepared, NotImplemented)
            if prepared == []:
                result = []
            elif candidate:
                result = self.extension.try_replay_records(*prepared[:4], prepared[4].records)
                self.assertIsNot(result, NotImplemented, "Supported native fixture must use candidate")
            else:
                result = self.reference(*prepared)
            for i, row in enumerate(result):
                self.assertIs(row[0], orders[i])
            outcomes.append((f.state(account, orders), [(o.stock_id, f.freeze(values))
                            for o, *values in result],
                            None if prepared == [] else prepared[4].records.tobytes()))
        self.assertEqual(outcomes[1], outcomes[0])
        # A second oracle calls the actual native per-order engine.
        account, orders = f.account(cash, holdings), f.orders(specifications)
        native, error = f.native(f.exchange(False, frame), account, orders)
        self.assertIsNone(error)
        self.assertEqual((f.state(account, orders), native), outcomes[0][:2])

    def test_reversals_duplicate_symbols_close_reopen_dictionary_order_and_bits(self):
        f = self.fixture
        buy, sell = f.Order.BUY, f.Order.SELL
        specs = [("AAA", 7, sell), ("BBB", 2, buy), ("AAA", 9, buy),
                 ("BBB", 4, sell), ("AAA", 6, sell), ("AAA", 2, buy), ("CCC", 1, sell)]
        for price in (.1, 3., f.np.nextafter(10., 0.), f.np.nextafter(10., f.np.inf)):
            with self.subTest(price=price):
                self.pair(specs, holdings={"CCC": 3, "AAA": 4, "BBB": -2},
                          frame=f.frame({"AAA": price, "BBB": 20., "CCC": 30.}))

    def test_partial_zero_cash_skips_preserve_prior_partial_and_factor(self):
        f = self.fixture
        self.pair([("AAA", 100, f.Order.BUY), ("BBB", 0, f.Order.SELL)], cash=25.)
        self.pair([("AAA", 1, f.Order.BUY), ("BBB", 1, f.Order.SELL)], cash=0.)
        for price in (float("nan"), 0., -1.):
            self.pair([("AAA", 1, f.Order.BUY), ("BBB", 1, f.Order.SELL)],
                      frame=f.frame({"AAA": price, "BBB": 20., "CCC": 30.}))
        limited = f.frame()
        limited.loc[("AAA", f.dates[0]), "limit_buy"] = True
        self.pair([("AAA", 2, f.Order.BUY), ("AAA", 2, f.Order.SELL)], frame=limited)

    def test_empty_replay_keeps_accumulator_types_and_bits(self):
        f = self.fixture
        account = f.account()
        before = f.state(account, [])
        self.assertEqual(self.extension.try_replay_records(account.current_position, account.accum_info,
                         [], [], f.np.empty((0, 14), dtype=f.np.float64)), [])
        self.assertEqual(f.state(account, []), before)
        self.pair([])

    def prepared(self):
        f = self.fixture
        account, orders = f.account(1000., {"AAA": 3}), f.orders([
            ("AAA", 2, f.Order.SELL), ("BBB", 1, f.Order.BUY)])
        prepared = self.prepare(f.exchange(True), account, orders)
        self.assertIsNot(prepared, NotImplemented)
        return account, orders, prepared

    def test_records_shape_dtype_layout_and_status_decline_before_mutation(self):
        f = self.fixture
        for kind in ("shape", "dtype", "layout", "subclass", "status", "factor_set", "nan_cash"):
            account, orders, prepared = self.prepared()
            records = prepared[4].records.copy()
            if kind == "shape": records = records[:, :-1]
            elif kind == "dtype": records = records.astype(f.np.float32)
            elif kind == "layout": records = f.np.asfortranarray(records)
            elif kind == "subclass": records = records.view(type("CustomArray", (f.np.ndarray,), {}))
            elif kind == "status": records[1, 12] = .5
            elif kind == "factor_set": records[1, 13] = 2.
            elif kind == "nan_cash": records[1, 4] = float("nan")
            before = f.state(account, orders)
            with self.subTest(kind=kind):
                self.assertIs(self.extension.try_replay_records(*prepared[:4], records), NotImplemented)
                self.assertEqual(f.state(account, orders), before)

    def test_custom_dict_metadata_methods_orders_and_duplicate_objects_decline(self):
        f = self.fixture
        class Callback:
            def __init__(self, trace): self.trace = trace
            def __del__(self): self.trace.append("destructor")

        for kind in ("dict", "metadata", "entry_metadata", "method", "order_subclass", "duplicate", "alias_entry", "descriptor"):
            with self.subTest(kind=kind):
                account, orders, prepared = self.prepared()
                position, accum, _, rows, result = prepared
                calls = []
                held = Callback(calls)  # Each destructor owns its original trace.
                if kind == "dict": position.position = type("CustomDict", (dict,), {})(position.position)
                elif kind == "metadata": orders[1].signed_partial_reason = held
                elif kind == "entry_metadata": position.position["AAA"]["research_metadata"] = held
                elif kind == "method": position._init_stock = lambda *args, trace=calls: trace.append("method")
                elif kind == "order_subclass": orders[1].__class__ = type("CustomOrder", (f.Order,), {})
                elif kind == "duplicate": orders[1] = orders[0]
                elif kind == "alias_entry": position.position["BBB"] = position.position["AAA"]
                before = f.state(account, orders)
                guard = patch.object(f.Order, "stock_id", property(lambda self, trace=calls: trace.append("property")), create=True)
                with (guard if kind == "descriptor" else patch.dict({}, {})):
                    self.assertIs(self.extension.try_replay_records(position, accum, orders, rows, result.records), NotImplemented)
                self.assertEqual(f.state(account, orders), before)
                self.assertEqual(calls, [])

    def test_unrelated_retained_metadata_is_not_a_replay_callback(self):
        f = self.fixture
        outcomes = []
        held = object()
        for candidate in (False, True):
            account, orders = f.account(1000., {"AAA": 3}), f.orders([
                ("AAA", 2, f.Order.SELL), ("BBB", 1, f.Order.BUY)])
            account.current_position._init_stock("UNTOUCHED", 1., 2.)
            account.current_position.position["UNTOUCHED"]["metadata"] = held
            orders[0].research_metadata = held
            prepared = self.prepare(f.exchange(True), account, orders)
            self.assertIsNot(prepared, NotImplemented)
            if candidate:
                result = self.extension.try_replay_records(*prepared[:4], prepared[4].records)
                self.assertIsNot(result, NotImplemented)
            else:
                result = self.reference(*prepared)
            outcomes.append((f.state(account, orders), [(o.stock_id, f.freeze(values)) for o, *values in result]))
        self.assertEqual(outcomes[1], outcomes[0])

    def test_late_factor_error_fallback_has_actual_native_error_and_state_prefix(self):
        f = self.fixture
        for symbol, frame in (("BBB", f.frame(factors={"BBB": 2})), ("MISSING", f.frame())):
            specs = [("AAA", 2, f.Order.BUY), (symbol, 1, f.Order.SELL)]
            outcomes = []
            for candidate in (False, True):
                account, orders = f.account(), f.orders(specs)
                exchange = f.exchange(True, frame)
                before = f.state(account, orders)
                self.assertIs(self.complete(exchange, account, orders, candidate=candidate), NotImplemented)
                self.assertEqual(f.state(account, orders), before)
                native, error = f.native(exchange, account, orders)
                outcomes.append((native, error, f.state(account, orders)))
            account, orders = f.account(), f.orders(specs)
            native, error = f.native(f.exchange(False, frame), account, orders)
            self.assertEqual(outcomes[0], (native, error, f.state(account, orders)))
            self.assertEqual(outcomes[1], outcomes[0])
            if symbol == "BBB": self.assertEqual(error[0], "NotImplementedError")

    def test_optional_complete_boundary_probe(self):
        from datetime import datetime, timezone
        import hashlib

        receipt = os.environ.get("QLIB_CYTHON_REPLAY_RECEIPT")
        if not receipt:
            self.skipTest("complete-boundary timing not explicitly requested")
        destination = Path(receipt).absolute()
        if destination.is_symlink() or destination.exists() or not destination.parent.is_dir():
            raise ValueError("Use a new receipt in an existing directory")
        compiled = os.environ.get("QLIB_CYTHON_REPLAY_COMPILED") == "1"
        payload = {"schema_version": 1, "kind": "synthetic_complete_ordered_boundary",
                   "goal_acceptance": False, "status": "running", "stage": "identity",
                   "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "stocks": 3000, "requested_pairs": 6, "completed_pairs": 0,
                   "compiled_kernel_requested": compiled, "compiled_kernel_activated": False,
                   "extension_load_seconds": self.load_seconds,
                   "activation_seconds": None, "first_pair": None, "pairs": [],
                   "fallback_counts": {"python_prefix": 0, "candidate_prefix": 0,
                                       "candidate_replay_to_python": 0},
                   "includes": ["original eligibility", "quote rows", "packing", "kernel",
                                "replay guards", "immutable snapshot", "ordered replay"],
                   "limits": "Synthetic complete boundary. Fixtures excluded equally; extension/compiler activation separate. No worker/full-report acceptance."}
        # Exclusive reservation precedes identities, activation and every probe
        # measurement. Keep progress on the SAME owned handle, including errors.
        with destination.open("x", encoding="utf-8") as output:
            def persist():
                encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
                output.seek(0)
                output.write(encoded)
                output.truncate()
                output.flush()
                os.fsync(output.fileno())

            persist()
            try:
                f = self.fixture
                def digest(path):
                    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
                payload["identity"] = {
                    "python": sys.version, "executable": sys.executable,
                    "numpy": f.np.__version__,
                    "manifest_path": str(Path(os.environ[MANIFEST]).absolute()),
                    "manifest_sha256": digest(os.environ[MANIFEST]),
                    "reviewed_build": self.manifest,
                    "extension_loaded_path": str(Path(self.extension.__file__).absolute()),
                    "extension_actual_sha256": digest(self.extension.__file__),
                    "actual_source_sha256": {name: digest(ROOT / name) for name in (
                        "qlib_yahoo/session_loops.pyx", "qlib_yahoo/ordered_signed.py",
                        "qlib_yahoo/ordered_signed_kernel.py", "tests/test_qlib_cython_replay.py")}}
                import qlib
                native_root = Path(qlib.__file__).parent
                payload["identity"]["native_runtime"] = {
                    "version": qlib.__version__, "path": str(native_root),
                    "build_sha256": digest(native_root / "_tubby_build.json")}
                payload["stage"] = "compiler_activation"
                core = f.bridge.kernel
                existing = getattr(core._compiled, "signatures", ())
                payload["compiler_signatures_before_probe"] = [str(item) for item in existing]
                persist()
                if compiled:
                    compiler_path = os.environ.get("QLIB_CYTHON_REPLAY_COMPILER_PATH")
                    if not compiler_path:
                        raise ValueError("Compiled probe requires explicit isolated QLIB_CYTHON_REPLAY_COMPILER_PATH")
                    started = time.perf_counter()
                    try:
                        # Actual reviewed path/version checks; imports only.
                        # First JIT remains inside the first session measurement.
                        payload["compiler_identity"] = f.bridge.activate_compiler(compiler_path)
                        import numba
                        import llvmlite
                        payload["compiler_identity"]["module_sha256"] = {
                            str(Path(module.__file__).absolute()): digest(module.__file__)
                            for module in (numba, llvmlite)}
                        payload["compiled_kernel_activated"] = True
                    finally:
                        payload["activation_seconds"] = time.perf_counter() - started
                        persist()
                else:
                    payload["activation_seconds"] = 0.

                def measured_boundary(exchange, account, orders, candidate):
                    # Mirror complete() but count each actual fallback explicitly;
                    # these branches and counters are timed for both variants.
                    prepared = self.prepare(exchange, account, orders, compiled=compiled)
                    if prepared is NotImplemented:
                        label = "candidate_prefix" if candidate else "python_prefix"
                        payload["fallback_counts"][label] += 1
                        return NotImplemented
                    if prepared == []:
                        return []
                    if candidate:
                        result = self.extension.try_replay_records(*prepared[:4], prepared[4].records)
                        if result is not NotImplemented:
                            return result
                        payload["fallback_counts"]["candidate_replay_to_python"] += 1
                    return self.reference(*prepared)

                payload["stage"] = "fixture_preparation"
                persist()
                names = [f"R{i:04d}" for i in range(3000)]
                frame = f.frame({name: 10. for name in names})
                specifications = [(name, 3, f.Order.SELL if i % 2 else f.Order.BUY)
                                  for i, name in enumerate(names)]
                for pair in range(6):
                    trial = {"index": pair, "status": "running", "samples": [],
                             "first_use_compilation_expected": bool(compiled and pair == 0 and not existing)}
                    payload["pairs"].append(trial)
                    if pair == 0: payload["first_pair"] = trial
                    outcomes = []
                    for candidate in ((False, True) if pair % 2 == 0 else (True, False)):
                        variant = "candidate" if candidate else "python"
                        payload["stage"] = f"pair_{pair}_{variant}_fixture"
                        persist()
                        account, orders = f.account(100000.), f.orders(specifications)
                        exchange = f.exchange(True, frame)
                        payload["stage"] = f"pair_{pair}_{variant}_boundary"
                        persist()
                        sample = {"variant": variant, "status": "running"}
                        trial["samples"].append(sample)
                        started = time.perf_counter()
                        try:
                            result = measured_boundary(exchange, account, orders, candidate)
                        finally:
                            sample["seconds"] = time.perf_counter() - started
                            sample["fallback_counts_after"] = dict(payload["fallback_counts"])
                            sample["compiler_signatures_after"] = [str(item) for item in getattr(core._compiled, "signatures", ())]
                            persist()
                        self.assertIsNot(result, NotImplemented)
                        if compiled:
                            self.assertTrue(sample["compiler_signatures_after"], "Compiled kernel did not execute")
                        outcomes.append((f.state(account, orders), [(o.stock_id, f.freeze(values))
                                         for o, *values in result]))
                        sample["status"] = "completed"
                        persist()
                    payload["stage"] = f"pair_{pair}_parity"
                    self.assertEqual(outcomes[0], outcomes[1])
                    trial.update(status="completed", exact_parity=True)
                    payload["completed_pairs"] += 1
                    persist()
                warm = {variant: [sample["seconds"] for trial in payload["pairs"][1:]
                                  for sample in trial["samples"] if sample["variant"] == variant]
                        for variant in ("python", "candidate")}
                payload["warm_samples_seconds"] = warm
                payload["warm_medians_seconds"] = {key: statistics.median(value) for key, value in warm.items()}
                payload["all_medians_seconds"] = {variant: statistics.median(
                    sample["seconds"] for trial in payload["pairs"] for sample in trial["samples"]
                    if sample["variant"] == variant) for variant in warm}
                payload["candidate_exercised_without_fallback"] = not any(payload["fallback_counts"].values())
                payload.update(status="completed", stage="finished",
                               finished_at_utc=datetime.now(timezone.utc).isoformat())
                persist()
            except BaseException as exc:
                payload.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                               finished_at_utc=datetime.now(timezone.utc).isoformat())
                persist()
                raise


if __name__ == "__main__":
    unittest.main()
