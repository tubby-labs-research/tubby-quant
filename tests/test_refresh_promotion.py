"""Offline refresh promotion ordering; all acquisition and scanning are mocked."""

import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from qlib_yahoo.dataset import DatasetError
from qlib_yahoo.refresh import refresh
from qlib_yahoo.workspace import set_default_dataset


def clean_report():
    return {
        "totals": dict.fromkeys((
            "possible_unadjusted_splits", "possible_unadjusted_reverse_splits",
            "possible_unadjusted_splits_in_universe", "extreme_moves_in_universe",
            "stale_member_days", "inconsistent_ohlc_bars", "members_that_ended"), 0),
        "by_year": [{"year": 2025, "member_days": 20, "member_days_without_close": 0}],
    }


class RefreshPromotionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.pointer = self.root / "latest.json"
        self.previous = b'{"dataset": "previous-snapshot", "preserve": true}\n'
        self.pointer.write_bytes(self.previous)
        self.snapshot = self.root / "datasets" / "candidate"
        self.events = []
        self.report = clean_report()

        def mocked(name, **kwargs):
            return self.stack.enter_context(patch(name, **kwargs))

        mocked("qlib_yahoo.reference.fetch_screener", return_value=self.root / "screener")
        mocked("qlib_yahoo.refresh.ticker_list", return_value=["AAA"])
        mocked("qlib_yahoo.yahoo_chart.fetch_archive")
        mocked("qlib_yahoo.yahoo_table.build_table", return_value={"cutoff": {"dropped_from": "2026-01-01"}})

        def prepare(*args, **kwargs):
            self.assertFalse(kwargs["make_latest"])
            self.assert_previous()
            self.events.append("prepare")
            return self.snapshot

        self.prepare = mocked("qlib_yahoo.stooq_table.prepare_stooq_table", side_effect=prepare)

        def scan(snapshot):
            self.assertEqual(snapshot, self.snapshot)
            self.assert_previous()
            self.events.append("scan")
            return self.report

        self.scan = mocked("qlib_yahoo.data_quality.scan", side_effect=scan)
        self.summary = mocked("qlib_yahoo.data_quality.summary_lines", return_value=[])
        # Exercise the real atomic default setter without preparing a real dataset.
        self.dataset_summary = mocked("qlib_yahoo.workspace.dataset_summary",
                                      return_value={"status": "ready", "price_eligible": True})

        def promote(root, snapshot):
            self.assertEqual(self.events, ["prepare", "scan"])
            self.assert_previous()
            self.assertTrue(self.receipt()["qualified"])
            self.assertFalse(self.receipt()["promoted"])
            self.events.append("promote")
            set_default_dataset(root, snapshot)

        self.promote = mocked("qlib_yahoo.workspace.set_default_dataset", side_effect=promote)

    def assert_previous(self):
        self.assertEqual(self.pointer.read_bytes(), self.previous)

    def receipt(self):
        paths = list((self.root / "sources").glob("yahoo-table-*/refresh.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text(encoding="utf-8"))

    def test_requested_promotion_follows_scan_and_persisted_qualification(self):
        record = refresh(self.root, make_latest=True)
        self.assertEqual(self.events, ["prepare", "scan", "promote"])
        self.assertEqual(json.loads(self.pointer.read_text())["dataset"], str(self.snapshot))
        self.assertEqual(record, self.receipt())
        self.assertTrue(record["qualified"])
        self.assertTrue(record["promoted"])
        self.assertEqual(record["quality"], self.report["totals"])
        self.assertEqual(record["qualification_scope"], "basic_price_scan_only_not_pit_readiness")

    def test_default_argument_never_promotes_clean_candidate(self):
        record = refresh(self.root)
        self.assert_previous()
        self.promote.assert_not_called()
        self.assertTrue(record["qualified"])
        self.assertFalse(record["promotion_requested"])
        self.assertFalse(record["promoted"])
        self.assertEqual(record["status"], "qualified")

    def test_each_anomaly_blocks_requested_promotion(self):
        for key in clean_report()["totals"]:
            if key == "members_that_ended":
                continue
            with self.subTest(key=key):
                # Each invocation needs its own dated-output root.
                root = self.root / key
                root.mkdir()
                pointer = root / "latest.json"
                pointer.write_bytes(self.previous)
                self.report = clean_report()
                self.report["totals"][key] = 1
                with self.assertRaisesRegex(DatasetError, key):
                    refresh(root, make_latest=True)
                self.assertEqual(pointer.read_bytes(), self.previous)
                receipt_path = next((root / "sources").glob("yahoo-table-*/refresh.json"))
                record = json.loads(receipt_path.read_text())
                self.assertEqual(record["status"], "rejected")
                self.assertFalse(record["qualified"])
                self.assertFalse(record["promoted"])
        self.promote.assert_not_called()

    def test_missing_close_blocks_even_without_promotion_request(self):
        self.report["by_year"][0]["member_days_without_close"] = 1
        with self.assertRaisesRegex(DatasetError, "member_days_without_close"):
            refresh(self.root)
        self.assert_previous()
        self.assertEqual(self.receipt()["status"], "rejected")
        self.promote.assert_not_called()

    def test_ended_listings_and_low_prices_are_informational(self):
        self.report["totals"]["members_that_ended"] = 3
        self.report["by_year"][0].update(member_days_below_1=5, member_days_below_5=10)
        record = refresh(self.root, make_latest=True)
        self.assertTrue(record["promoted"])

    def test_scan_exception_preserves_default_and_original_error(self):
        error = OSError("scan unavailable")
        self.scan.side_effect = error
        with self.assertRaises(OSError) as raised:
            refresh(self.root, make_latest=True)
        self.assertIs(raised.exception, error)
        self.assert_previous()
        self.promote.assert_not_called()
        self.assertEqual(self.receipt()["status"], "scan_failed")
        self.assertFalse(self.receipt()["qualified"])
        self.assertIn("scan unavailable", self.receipt()["error"])

    def test_malformed_or_empty_scan_fails_closed(self):
        reports = [None, {}, {"totals": {}}, {**clean_report(), "by_year": []}]
        for invalid in (None, -1, True, 0.0, "0"):
            report = clean_report()
            report["totals"]["stale_member_days"] = invalid
            reports.append(report)
        for invalid in (0, -1, True, "20"):
            report = clean_report()
            report["by_year"][0]["member_days"] = invalid
            reports.append(report)
        for index, report in enumerate(reports):
            with self.subTest(report=report):
                root = self.root / str(index)
                root.mkdir()
                self.report = copy.deepcopy(report)
                with self.assertRaises((TypeError, KeyError, ValueError)):
                    refresh(root, make_latest=True)
                self.assertFalse((root / "latest.json").exists())
                self.assert_previous()
        self.promote.assert_not_called()

    def test_summary_failure_blocks_promotion(self):
        self.summary.side_effect = ValueError("unreadable report")
        with self.assertRaisesRegex(ValueError, "unreadable report"):
            refresh(self.root, make_latest=True)
        self.assert_previous()
        self.promote.assert_not_called()
        self.assertEqual(self.receipt()["status"], "scan_failed")

    def test_default_setter_failure_is_recorded(self):
        self.dataset_summary.return_value = {"status": "incomplete", "price_eligible": True}
        with self.assertRaisesRegex(ValueError, "Only a ready dataset"):
            refresh(self.root, make_latest=True)
        self.assert_previous()
        self.assertTrue(self.receipt()["qualified"])
        self.assertFalse(self.receipt()["promoted"])
        self.assertEqual(self.receipt()["status"], "promotion_failed")

    def test_receipt_failure_prevents_promotion(self):
        original_write = Path.write_text

        def write(path, *args, **kwargs):
            if path.name == "refresh.json":
                raise OSError("receipt unavailable")
            return original_write(path, *args, **kwargs)

        with patch.object(Path, "write_text", write):
            with self.assertRaisesRegex(OSError, "receipt unavailable"):
                refresh(self.root, make_latest=True)
        self.assert_previous()
        self.promote.assert_not_called()


if __name__ == "__main__":
    unittest.main()
