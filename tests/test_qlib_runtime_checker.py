"""Standalone bounded checker coverage; never import the legacy Brain tools."""

import importlib.util
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("runtime_checker_tests", ROOT / "tools/qlib_runtime_check.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ReportPayloadCheckerTests(unittest.TestCase):
    @staticmethod
    def report(path, *, uuid="first", value=1, names=None, renderer=None):
        path.mkdir()
        names = ["account.html", "signal_ic.html"] if names is None else names
        (path / "report_manifest.json").write_text(json.dumps({
            "status": "completed", "charts": names, "renderer_source_sha256": "source",
            "plotly_version": "test", "level": "detailed", "holdings_rendered": False,
            **(renderer or {}),
        }), encoding="utf-8")
        (path / "report_index.html").write_text("dashboard", encoding="utf-8")
        (path / "plotly.min.js").write_text("offline-library", encoding="utf-8")
        for name in names:
            if Path(name).name != name:
                continue
            arguments = [uuid, [{"y": [value, None], "name": "signal"}], {"title": "net"}, {"responsive": True}]
            (path / name).write_text("Plotly.newPlot( " + ", ".join(map(json.dumps, arguments)) + " );", encoding="utf-8")

    def test_fresh_dom_ids_are_ignored_but_every_chart_value_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary) / "left", Path(temporary) / "right"
            self.report(left)
            self.report(right, uuid="another")
            self.assertEqual(checker.compare_reports(left, right)["chart_count"], 2)
            path = right / "signal_ic.html"
            path.write_text(path.read_text(encoding="utf-8").replace('[1, null]', '[2, null]'), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "signal_ic"):
                checker.compare_reports(left, right)

    def test_offline_plotting_library_cannot_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary) / "left", Path(temporary) / "right"
            self.report(left)
            self.report(right)
            (right / "plotly.min.js").write_text("altered-library", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "offline Plotly bytes"):
                checker.compare_reports(left, right)

    def test_equal_python_numbers_do_not_relax_json_type_or_float_bit_parity(self):
        for first, second in ((-0.0, 0.0), (1, 1.0), (True, 1)):
            self.assertEqual(first, second)
            self.assertFalse(checker.json_values_exact({"nested": [first]}, {"nested": [second]}))
            with tempfile.TemporaryDirectory() as temporary:
                left, right = Path(temporary) / "left", Path(temporary) / "right"
                self.report(left, value=first)
                self.report(right, value=second)
                with self.assertRaisesRegex(AssertionError, "account.html"):
                    checker.compare_reports(left, right)
        self.assertTrue(checker.json_values_exact({"nested": [-0.0, 1.0, None]}, {"nested": [-0.0, 1.0, None]}))
        self.assertFalse(checker.json_values_exact({"a": 1, "b": 2}, {"b": 2, "a": 1}))
        self.assertFalse(checker.json_values_exact([], [None]))
        self.assertFalse(checker.json_values_exact((1,), (1,)))

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional native table dependencies unavailable")
    def test_table_float_bits_include_signed_zero_and_null_mask(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary) / "left.parquet", Path(temporary) / "right.parquet"
            for dtype in (pa.float32(), pa.float64()):
                pq.write_table(pa.table({"value": pa.array([-0.0, None, 1.0], type=dtype)}), left)
                pq.write_table(pa.table({"value": pa.array([-0.0, None, 1.0], type=dtype)}), right)
                self.assertEqual(checker.compare_tables(left, right), 3)
                pq.write_table(pa.table({"value": pa.array([0.0, None, 1.0], type=dtype)}), right)
                with self.assertRaisesRegex(AssertionError, "floating bits"):
                    checker.compare_tables(left, right)

    def test_missing_inventory_and_redirected_chart_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right, invalid = [root / name for name in ("left", "right", "invalid")]
            self.report(left)
            self.report(right, names=["account.html"])
            with self.assertRaisesRegex(AssertionError, "chart inventory"):
                checker.compare_reports(left, right)
            self.report(invalid, names=["../outside.html"])
            with self.assertRaises(AssertionError):
                checker.compare_reports(invalid, invalid)

    def test_chart_parser_does_not_accept_extra_or_incomplete_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "chart.html"
            for content in ('Plotly.newPlot("id", [], {}, {}) Plotly.newPlot("other", [], {}, {})',
                            'Plotly.newPlot("id", [], {}, {} extra)'):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(AssertionError):
                    checker.chart_payload(path)

    def test_renderer_change_requires_exact_reviewed_pair_and_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary) / "left", Path(temporary) / "right"
            self.report(left, renderer=checker.REVIEWED_RENDERERS[0])
            self.report(right, renderer=checker.REVIEWED_RENDERERS[1])
            with self.assertRaisesRegex(AssertionError, "renderer_source"):
                checker.compare_reports(left, right)
            proof = checker.compare_reports(left, right, allow_reviewed_renderer_change=True)
            self.assertEqual(proof["reviewed_renderer_transition"]["after"], checker.REVIEWED_RENDERERS[1])
            path = right / "report_manifest.json"
            status = json.loads(path.read_text())
            status["renderer_dependencies_sha256"]["holding_series.py"] = "unknown"
            path.write_text(json.dumps(status))
            with self.assertRaisesRegex(AssertionError, "unreviewed renderer"):
                checker.compare_reports(left, right, allow_reviewed_renderer_change=True)

    def test_reviewed_renderer_change_never_relaxes_chart_or_request_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary) / "left", Path(temporary) / "right"
            self.report(left, renderer=checker.REVIEWED_RENDERERS[0])
            self.report(right, renderer=checker.REVIEWED_RENDERERS[1])
            path = right / "report_manifest.json"
            status = json.loads(path.read_text())
            status["holdings"] = "never"
            path.write_text(json.dumps(status))
            with self.assertRaisesRegex(AssertionError, "holdings"):
                checker.compare_reports(left, right, allow_reviewed_renderer_change=True)
            status.pop("holdings")
            path.write_text(json.dumps(status))
            path = right / "signal_ic.html"
            path.write_text(path.read_text().replace('[1, null]', '[2, null]'))
            with self.assertRaisesRegex(AssertionError, "signal_ic"):
                checker.compare_reports(left, right, allow_reviewed_renderer_change=True)


class FinancialSummaryCheckerTests(unittest.TestCase):
    @staticmethod
    def bundle(path, summary):
        path.mkdir()
        (path / "daily_report.csv").write_bytes(b"account\n100000\n")
        (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        sealed = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in path.iterdir()}
        (path / "manifest.json").write_text(json.dumps({
            "status": "completed", "backtest": {}, "artifacts_sha256": sealed,
        }), encoding="utf-8")

    def test_metrics_and_funding_require_scalar_types_and_float_bits(self):
        for key in ("metrics", "funding_state"):
            for first, second in ((-0.0, 0.0), (1, 1.0), (True, 1)):
                with self.subTest(key=key, first=repr(first), second=repr(second)), \
                        tempfile.TemporaryDirectory() as temporary:
                    left, right = Path(temporary) / "left", Path(temporary) / "right"
                    self.bundle(left, {key: {"value": first}})
                    self.bundle(right, {key: {"value": second}})
                    with self.assertRaisesRegex(AssertionError, key):
                        checker.compare_runs(left, right)

    def test_financial_summary_nested_order_is_exact(self):
        for key in ("metrics", "metrics_status", "funding_state"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                left, right = Path(temporary) / "left", Path(temporary) / "right"
                self.bundle(left, {key: {"a": 1, "b": 2}})
                self.bundle(right, {key: {"b": 2, "a": 1}})
                with self.assertRaisesRegex(AssertionError, key):
                    checker.compare_runs(left, right)

    def test_exact_financial_summary_roundtrip_still_passes(self):
        summary = {"metrics": {"sharpe": -0.0, "observations": 1255},
                   "metrics_status": {"sharpe": "defined"},
                   "funding_state": {"cash": 1.0, "restricted": 0.0}}
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary) / "left", Path(temporary) / "right"
            self.bundle(left, summary)
            self.bundle(right, summary)
            self.assertEqual(checker.compare_runs(left, right)["status"], "exact parity")


class ReviewedEngineTransitionTests(unittest.TestCase):
    def identities(self):
        base = {"repository": "https://github.com/tubby-labs-research/qlib",
                "upstream_commit": "da920b7f954f48ab1bb64117c976710de198373e",
                "installed_package_verified": True, "build_dirty": False,
                "platform": "win32-amd64", "platform_tag": "cp312-cp312-win_amd64",
                "base_wheel_sha256": "official-base", "build_method": "verified-wheel"}
        return (
            {**base, "commit": "da54853c7598599f74ff7776dc979b74106eb1d0", "version": "0.9.7+tubby.2",
             "wheel_sha256": "d7474a7bf3e5aa4dc68461d0531339d92692cc827fa02080ece89da99321a243"},
            {**base, "commit": "d10a5a8ca397ea088da8f5c996e8bd042eea27e0", "version": "0.9.7+tubby.3",
             "wheel_sha256": "2838655eda89662e0e13e42c4066501217f91ff8b09b4363b6b62fb8197b06dd"},
        )

    def test_known_clean_build_transition_is_explicit_and_recorded(self):
        first, second = self.identities()
        result = checker.reviewed_engine_transition(first, second)
        self.assertEqual(result["before"], first)
        self.assertEqual(result["after"], second)

    def test_ordered_hook_candidate_requires_the_exact_reviewed_windows_pair(self):
        old, native = self.identities()
        candidate = {**native, "commit": "8ab6249aa6143cb0675259d5ff986d294e918ef4",
                     "version": "0.9.7+tubby.4",
                     "wheel_sha256": "9157e9554a46b30e7ec4e83f7d9f03022c6b8781816da5170587d3fe6ec4e1af"}
        for first in (old, native):
            self.assertIn("ordered-session", checker.reviewed_engine_transition(first, candidate)["scope"])
            for key, value in (("wheel_sha256", "other-build"), ("build_dirty", True),
                               ("installed_package_verified", False), ("version", native["version"]),
                               ("commit", "unknown"), ("platform", "darwin-universal2")):
                with self.subTest(before=first["version"], key=key), self.assertRaises(AssertionError):
                    checker.reviewed_engine_transition(first, {**candidate, key: value})
        with self.assertRaisesRegex(AssertionError, "unreviewed engine transition"):
            checker.reviewed_engine_transition(candidate, native)

    def test_candidate_mac_build_is_not_implied_by_the_windows_build(self):
        _old, native = self.identities()
        native.update(platform="darwin-universal2", platform_tag="cp312-cp312-macosx_10_13_universal2",
                      wheel_sha256="a8387ccca0d9908d3f3df77f7247efc4d12c209cd2952bd104b6335f14d7fbd1")
        candidate = {**native, "commit": "8ab6249aa6143cb0675259d5ff986d294e918ef4", "version": "0.9.7+tubby.4",
                     "wheel_sha256": "9157e9554a46b30e7ec4e83f7d9f03022c6b8781816da5170587d3fe6ec4e1af"}
        with self.assertRaisesRegex(AssertionError, "reviewed engine platform differs"):
            checker.reviewed_engine_transition(native, candidate)

    def test_unknown_tampered_unverified_and_changed_builds_are_rejected(self):
        first, original = self.identities()
        for key, value in (("commit", "unknown"), ("wheel_sha256", "tampered"),
                           ("installed_package_verified", False), ("build_dirty", True),
                           ("platform", "darwin-universal2"), ("platform_tag", "changed"),
                           ("base_wheel_sha256", "changed"), ("version", "unknown"),
                           ("repository", "other"), ("upstream_commit", "other")):
            with self.subTest(key=key):
                second = deepcopy(original)
                second[key] = value
                with self.assertRaises(AssertionError):
                    checker.reviewed_engine_transition(first, second)


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional PyArrow unavailable")
class BoundedTableCheckerTests(unittest.TestCase):
    @staticmethod
    def bundle(path, *, stream, order_values=None, daily=False, decisions=False):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path.mkdir()
        (path / "daily_report.csv").write_text("account\n100000\n", encoding="utf-8")
        (path / "summary.json").write_text(json.dumps({"metrics": {}, "metrics_status": "defined"}), encoding="utf-8")
        if order_values is not None:
            pq.write_table(pa.table({"deal_amount": order_values}), path / "qlib_order_indicators.parquet")
        if daily:
            (path / "qlib_trade_indicators.csv").write_text("ffr\n1\n", encoding="utf-8")
        if decisions:
            with gzip.open(path / "signed_decisions.json.gz", "wt", encoding="utf-8") as source:
                json.dump([{"orders": []}], source)
        sealed = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in path.iterdir()}
        (path / "manifest.json").write_text(json.dumps({
            "status": "completed", "backtest": {}, "artifacts_sha256": sealed,
            "history_storage": {"mode": "stream" if stream else "memory"},
        }), encoding="utf-8")

    def test_reviewed_engine_change_requires_opt_in_without_relaxing_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary) / "left", Path(temporary) / "right"
            self.bundle(left, stream=True, daily=True, decisions=True, order_values=[1.0])
            self.bundle(right, stream=True, daily=True, decisions=True, order_values=[1.0])
            identities = ReviewedEngineTransitionTests().identities()
            for path, identity in zip((left, right), identities):
                manifest = json.loads((path / "manifest.json").read_text())
                manifest["signed_engine_identity"] = identity
                (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(AssertionError, "signed_engine_identity"):
                checker.compare_runs(left, right)
            result = checker.compare_runs(left, right, allow_reviewed_engine_change=True)
            self.assertEqual(result["decision_sessions"], 1)
            self.assertIsNotNone(result["reviewed_engine_transition"])
            manifest["backtest"]["cash"] = 999
            (right / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(AssertionError, "cash"):
                checker.compare_runs(left, right, allow_reviewed_engine_change=True)

    def test_streamed_order_indicator_presence_and_values_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right, missing, memory = [root / name for name in ("left", "right", "missing", "memory")]
            self.bundle(left, stream=True, order_values=[1.0, -2.0])
            self.bundle(right, stream=True, order_values=[1.0, -3.0])
            self.bundle(missing, stream=True)
            self.bundle(memory, stream=False)
            with self.assertRaises(AssertionError):
                checker.compare_runs(left, right)
            with self.assertRaisesRegex(AssertionError, "indicator presence differs"):
                checker.compare_runs(left, missing)
            # An older memory run never exported the extra order table.
            self.assertEqual(checker.compare_runs(memory, left)["status"], "exact parity")

    def test_order_indicators_exact_across_row_group_boundaries(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary) / "left.parquet", Path(temporary) / "right.parquet"
            table = pa.table({"instrument": ["A", "B", "C"] * 10_001,
                              "deal_amount": [0.0, -1.0, float("nan")] * 10_001})
            pq.write_table(table, left, row_group_size=10_000)
            pq.write_table(table, right, row_group_size=3_007)
            self.assertEqual(checker.compare_tables(left, right), 30_003)
            changed = table.set_column(1, "deal_amount", pa.array([0.0, -2.0, float("nan")] * 10_001))
            pq.write_table(changed, right, row_group_size=3_007)
            with self.assertRaises(AssertionError):
                checker.compare_tables(left, right)

    def test_streamed_daily_indicators_and_decisions_cannot_disappear(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, daily_missing, decisions_missing = [root / name for name in ("left", "daily", "decisions")]
            self.bundle(left, stream=True, daily=True, decisions=True)
            self.bundle(daily_missing, stream=True, decisions=True)
            self.bundle(decisions_missing, stream=True, daily=True)
            with self.assertRaisesRegex(AssertionError, "daily indicator presence differs"):
                checker.compare_runs(left, daily_missing)
            with self.assertRaisesRegex(AssertionError, "decision history presence differs"):
                checker.compare_runs(left, decisions_missing)
