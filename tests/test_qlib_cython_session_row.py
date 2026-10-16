"""Unwired native-row/date-suffix differentials; never builds or installs.

Supply a NEW reviewed V5 QLIB_SESSION_LOOPS_MANIFEST, then run this module's
LauncherTests. Optional QLIB_CYTHON_ROW_RECEIPT writes complete-row paired
diagnostics (native prefix + helper guards/views + native price/cache suffix),
including cached calls. Never run alongside a full-worker timing measurement.
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
MANIFEST = "QLIB_SESSION_LOOPS_MANIFEST"
CHILD = "QLIB_CYTHON_ROW_CHILD"


def row_with_suffix(native, helper, counts=None):
    """Graft ONLY the current comparison expression, retaining native callbacks.

    This is a test/probe caller, not an installed exchange binding. Unsupported
    helper input evaluates the original expression in-place; no prefix restart.
    Source structure changes fail closed and require review.
    """
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(native)))
    method = tree.body[0]
    matches = [node for node in ast.walk(method)
               if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp)
               and ast.unparse(node.test).startswith("row == len(index)")]
    if len(matches) != 1:
        raise RuntimeError("Native comparison boundary changed; review required")
    condition = matches[0]
    arguments = ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in
                              ("index", "row", "start_ns", "end_ns")],
                              kwonlyargs=[], kw_defaults=[], defaults=[])
    original = ast.FunctionDef(name="native_range_rejected", args=arguments,
                              body=[ast.Return(value=condition.test)], decorator_list=[])
    # Original expression uses self only for the two immutable scalar bounds.
    class Bounds(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                if node.attr in ("_session_start_ns", "_session_end_ns"):
                    return ast.Name(id="start_ns" if node.attr == "_session_start_ns"
                                    else "end_ns", ctx=ast.Load())
            return self.generic_visit(node)
    original = Bounds().visit(original)
    namespace = dict(native.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[])),
                 inspect.getsourcefile(native), "exec"), namespace)
    reference = namespace["native_range_rejected"]

    def rejected(index, row, start_ns, end_ns):
        result = helper(index, row, start_ns, end_ns)
        if counts is not None:
            key = "declined" if result is NotImplemented else "used"
            counts[key] = counts.get(key, 0) + 1
        return reference(index, row, start_ns, end_ns) if result is NotImplemented else not result

    namespace["candidate_range_rejected"] = rejected
    condition.test = ast.parse(
        "candidate_range_rejected(index, row, self._session_start_ns, self._session_end_ns)",
        mode="eval").body
    method.name = "candidate_row"
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(native), "exec"), namespace)
    return namespace["candidate_row"]


class LauncherTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get(MANIFEST), "new reviewed V5 extension receipt required")
    @unittest.skipIf(os.environ.get(CHILD) == "1", "isolated child")
    def test_verified_native_child(self):
        environment = os.environ.copy()
        environment[CHILD] = "1"
        program = (
            "from qlib_yahoo.runtime import ensure_signed_runtime; ensure_signed_runtime(); "
            "import unittest; suite=unittest.defaultTestLoader.loadTestsFromName("
            "'tests.test_qlib_cython_session_row.RowChild'); "
            "r=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not r.wasSuccessful())")
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(os.environ.get(CHILD) == "1", "requires verified native extension child")
class RowChild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_qlib_fast_signed import SignedChild
        from tests.test_qlib_session_loops import load_reviewed_extension
        SignedChild.setUpClass()
        cls.fixture = SignedChild()
        cls.extension, cls.load_seconds, cls.manifest = load_reviewed_extension()
        if not hasattr(cls.extension, "try_valid_range"):
            raise RuntimeError("V5 date suffix required; preserved V4 is insufficient")
        cls.native = staticmethod(cls.fixture.FastExchange._session_row)
        cls.candidate = staticmethod(row_with_suffix(cls.native, cls.extension.try_valid_range))

    def exact(self, actual, expected):
        from tests.test_qlib_session_loops import NativeChild
        comparison = NativeChild()
        comparison.fixture = self.fixture
        comparison.exact(actual, expected)

    def reference_range(self, index, row, start, end):
        return not (row == len(index) or index[row] > end
                    or (row > 0 and index[row - 1] >= start)
                    or (row + 1 < len(index) and index[row + 1] <= end))

    def test_exact_range_duplicates_nat_endpoints_and_error_policies(self):
        np = self.fixture.np
        values = [-9223372036854775808, -9223372036854775807, -1, 0, 1,
                  9223372036854775807]
        callbacks = []
        class Log:
            def write(self, message): callbacks.append(message)
        for policy in ("ignore", "warn", "raise", "call", "log"):
            with np.errstate(all=policy, call=Log() if policy == "log" else
                             lambda *args: callbacks.append(args)):
                for a in values:
                    for b in values:
                        bounds = np.array([a, b], dtype="int64").view("datetime64[ns]")
                        for ticks in ([a], [a, a], [a, 0, b], [b, a, -1]):
                            index = np.array(ticks, dtype="int64").view("datetime64[ns]")
                            for row in range(len(index) + 1):
                                self.assertIs(self.extension.try_valid_range(index, row, *bounds),
                                              self.reference_range(index, row, *bounds))
                empty = np.array([], dtype="datetime64[ns]")
                self.assertIs(self.extension.try_valid_range(empty, 0, *bounds), False)
        self.assertEqual(callbacks, [])

    def test_unsupported_inputs_decline_without_conversions_or_attributes(self):
        np = self.fixture.np
        index = np.array([0, 1, 2], dtype="int64").view("datetime64[ns]")
        start, end = index[0], index[1]
        class Hostile:
            def __getattribute__(self, name): raise AssertionError("attribute callback")
            def __index__(self): raise AssertionError("index callback")
        class Array(np.ndarray):
            def __getattribute__(self, name): raise AssertionError("array callback")
        unaligned = np.ndarray(shape=(3,), dtype="datetime64[ns]", buffer=bytearray(25), offset=1)
        for arguments in ((Hostile(), 0, start, end), (index, Hostile(), start, end),
                          (index, 0, Hostile(), end), (index, 0, start, Hostile()),
                          (index.view(Array), 0, start, end), (index, True, start, end),
                          (index, np.int64(0), start, end), (index, -1, start, end),
                          (index, 4, start, end), (index, 10 ** 100, start, end),
                          (index[::2], 0, start, end), (index[::-1], 0, start, end),
                          (unaligned, 0, start, end), (index.reshape(1, 3), 0, start, end),
                          (index.astype("datetime64[ms]"), 0, start, end),
                          (index.astype(">M8[ns]"), 0, start, end),
                          (index, 0, start.astype("datetime64[ms]"), end)):
            self.assertIs(self.extension.try_valid_range(*arguments), NotImplemented)
        with patch.object(np, "dtype", side_effect=AssertionError("dtype callback")):
            # Captured immutable native dtype; this suffix never calls np.dtype.
            self.assertIs(self.extension.try_valid_range(index, 0, start, end), False)

    def pair_rows(self, *, configure=None, weekend=False):
        f = self.fixture
        outputs = []
        for candidate in (False, True):
            exchange = f.exchange(True)
            if candidate:
                exchange._range_validator = self.extension.try_valid_range
                exchange.reset_session_cache()
            start = f.dates[0]
            end = start + f.pd.Timedelta(days=2 if weekend else 0, hours=23)
            self.assertTrue(exchange.begin_session(start, end))
            trace = []
            if configure: configure(exchange, trace)
            results = []
            method = self.native  # Actual caller, including its live binding guards/counts.
            try:
                for symbol in ("BBB", "AAA", "MISSING", "CCC", "AAA"):
                    result = method(exchange, symbol, start, end)
                    results.append(result)
                    if result is not None:
                        self.assertIs(result, exchange._session_rows[symbol])
                error = None
            except Exception as exc:
                error = (type(exc), str(exc))
            outputs.append((results, list(exchange._session_rows.items()), trace, error))
        self.exact(outputs[1], outputs[0])
        return outputs[0]

    def test_native_rows_lazy_order_scalar_types_readonly_and_mutable_views(self):
        for weekend in (False, True):
            result = self.pair_rows(weekend=weekend)
            self.assertEqual([key for key, _ in result[1]], ["BBB", "AAA", "CCC"])
            def configure(exchange, trace):
                for panel in exchange.quote.data.values():
                    panel.indices[0].idx_list.flags.writeable = False
            self.pair_rows(weekend=weekend, configure=configure)
        np = self.fixture.np
        index = np.array([0, 1, 2], dtype="int64").view("datetime64[ns]")
        self.assertIs(self.extension.try_valid_range(index, 1, index[1], index[1]), True)
        index[0] = index[1]
        self.assertIs(self.extension.try_valid_range(index, 1, index[1], index[1]), False)

    def test_prefix_callbacks_once_mutations_and_late_exception_cache_prefix(self):
        for fail in (False, True):
            def configure(exchange, trace):
                class Mutable(dict):
                    def get(self, key, default=None):
                        trace.append(("quote", key))
                        if key == "BBB": self.pop("CCC", None)
                        if key == "AAA" and fail: raise LookupError("native prefix error")
                        return super().get(key, default)
                exchange.quote.data = Mutable(exchange.quote.data)
                panel = exchange.quote.data["BBB"]
                class Dates(dict):
                    def get(self, key, default=None):
                        trace.append("date")
                        # Change a captured neighboring date before comparisons.
                        panel.indices[0].idx_list[1] = key
                        return super().get(key, default)
                panel.indices[0].index_map = Dates(panel.indices[0].index_map)
            result = self.pair_rows(configure=configure)
            self.assertEqual(result[2].count("date"), 1)
            self.assertEqual(result[2].count(("quote", "BBB")), 1)

    def test_decline_resumes_only_comparison_preserving_native_index_error(self):
        for row in (-1, 99, self.fixture.np.int64(0)):
            def configure(exchange, trace):
                panel = exchange.quote.data["BBB"]
                class Dates(dict):
                    def get(self, key, default=None):
                        trace.append("date")
                        return row
                panel.indices[0].index_map = Dates(panel.indices[0].index_map)
            result = self.pair_rows(configure=configure)
            self.assertEqual(result[2].count("date"), 1)

    def test_native_layout_fallback_and_price_field_suffix_are_unchanged(self):
        f, np = self.fixture, self.fixture.np
        for weekend in (False, True):
            def noncontiguous(exchange, trace):
                for panel in exchange.quote.data.values():
                    labels = panel.indices[0].idx_list
                    panel.indices[0].idx_list = np.repeat(labels, 2)[::2]
            self.pair_rows(configure=noncontiguous, weekend=weekend)
            for value in (np.nan, np.inf, -np.inf, -0., np.nextafter(10., 0.)):
                def price(exchange, trace):
                    panel = exchange.quote.data["BBB"]
                    panel.data[:, panel.indices[1].index_map["$open"]] = value
                self.pair_rows(configure=price, weekend=weekend)
            for field in ("$open", "$factor", "limit_sell"):
                def missing(exchange, trace):
                    exchange.quote.data["BBB"].indices[1].index_map.pop(field)
                self.pair_rows(configure=missing, weekend=weekend)

    def test_bound_descriptors_decline_before_reordering_native_reads(self):
        f = self.fixture
        results = []
        for selected in (False, True):
            exchange = f.exchange(True)
            if selected:
                exchange._range_validator = self.extension.try_valid_range
                exchange.reset_session_cache()
            start, end = f.dates[0], f.dates[0] + f.pd.Timedelta(hours=23)
            exchange.begin_session(start, end)
            bounds = {name: exchange.__dict__[name] for name in ("_session_start_ns", "_session_end_ns")}
            trace = []
            def getter(name):
                def read(instance):
                    trace.append(name)
                    return bounds[name]
                return property(read)
            with patch.object(f.FastExchange, "_session_start_ns", getter("_session_start_ns"), create=True), \
                    patch.object(f.FastExchange, "_session_end_ns", getter("_session_end_ns"), create=True):
                row = self.native(exchange, "AAA", start, end)
            results.append((row, trace))
            if selected:
                self.assertEqual(exchange.range_counts, {"compiled_checks": 0, "native_checks": 1})
        self.exact(results[1], results[0])
        # The lazy date-map prefix reads start once. Row zero's comparison
        # then reads end twice; the optional decline must retain that order.
        self.assertEqual(results[0][1], ["_session_start_ns", "_session_end_ns", "_session_end_ns"])

    @unittest.skipUnless(os.environ.get("QLIB_CYTHON_ROW_RECEIPT"), "explicit complete-row probe not requested")
    def test_complete_row_probe(self):
        path = Path(os.environ["QLIB_CYTHON_ROW_RECEIPT"]).absolute()
        # Exclusive reservation prevents overwriting successful/failed evidence.
        with path.open("x", encoding="utf-8") as handle:
            handle.write('{"status":"reserved","goal_acceptance":false}\n')
            handle.flush(); os.fsync(handle.fileno())
        receipt = {"status": "running", "goal_acceptance": False,
                   "scope": "synthetic complete row with native prefix/suffix and all helper costs",
                   "timed_instrumentation": "candidate activation/decline count dictionary updates included",
                   "excludes": "child startup, fixture preparation and extension loading (load reported separately)",
                   "extension_load_seconds": self.load_seconds, "manifest": self.manifest,
                   "pairs": [], "symbol_count": 3000, "index_days": 1255}
        def persist():
            temporary = path.with_name(path.name + ".progress.tmp")
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(receipt, handle, indent=2); handle.write("\n")
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, path)
        persist()
        try:
            f, np, pd = self.fixture, self.fixture.np, self.fixture.pd
            symbols = [f"S{i:04d}" for i in range(3000)]
            # Real native panels, representative 1255-day date-map scale.
            preparation = time.perf_counter()
            dates = pd.bdate_range("2021-01-01", periods=1255)
            index = pd.MultiIndex.from_product([symbols, dates], names=["instrument", "datetime"])
            frame = pd.DataFrame({"$open": 10., "$close": 11., "$volume": 123.,
                                  "$factor": 1., "limit_buy": False, "limit_sell": False}, index=index)
            exchange = f.exchange(True, frame)
            receipt["fixture_preparation_seconds"] = time.perf_counter() - preparation
            start, end = dates[600], dates[600] + pd.Timedelta(hours=23)
            for number in range(6):
                pair = {"number": number, "seconds": {}, "counts": {}}
                reference = None
                for name in (("python", "candidate") if number % 2 == 0 else ("candidate", "python")):
                    exchange._range_validator = self.extension.try_valid_range if name == "candidate" else None
                    exchange.reset_session_cache(); exchange.begin_session(start, end)
                    method = self.native
                    started = time.perf_counter()
                    rows = [method(exchange, symbol, start, end) for symbol in symbols]
                    # Cache hits must preserve identity and skip the helper.
                    hits = [method(exchange, symbol, start, end) for symbol in symbols]
                    pair["seconds"][name] = time.perf_counter() - started
                    self.assertTrue(all(a is b for a, b in zip(rows, hits)))
                    self.assertEqual(list(exchange._session_rows), symbols)
                    if reference is not None: self.exact(rows, reference)
                    reference = rows
                    if name == "candidate":
                        self.assertEqual(exchange.range_counts, {"compiled_checks": 3000, "native_checks": 0})
                        pair["counts"][name] = exchange.range_counts.copy()
                receipt["pairs"].append(pair); persist()
            receipt["median_seconds"] = {name: statistics.median(pair["seconds"][name]
                for pair in receipt["pairs"]) for name in ("python", "candidate")}
            receipt["status"] = "completed"; persist()
        except (Exception, KeyboardInterrupt) as exc:
            receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            persist(); raise


if __name__ == "__main__":
    unittest.main()
