"""Dated liquidity universes: classification, causal ranking, spans and score masking."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from qlib_yahoo.dataset import DatasetError, sha256
from qlib_yahoo.universe import UniverseRule, exclusion_reason


class ClassificationTests(unittest.TestCase):
    def test_common_stock_rule(self):
        cases = {
            ("AAPL", "stock"): None, ("BRK-B", "stock"): None, ("GOOGL", "stock"): None,
            ("SPY", "etf"): "not_stock_type", ("PMT_A", "stock"): "preferred_or_series",
            ("OXY-WS", "stock"): "non_common_symbol", ("BEBE-U", "stock"): "warrant_unit_right_or_when_issued",
            ("JACS-R", "stock"): "warrant_unit_right_or_when_issued", ("MKC-V", "stock"): "warrant_unit_right_or_when_issued",
            ("ADIG-W", "stock"): "warrant_unit_right_or_when_issued", ("CON", "stock"): "windows_reserved_name",
            ("1ABC", "stock"): "non_common_symbol", ("A.B", "stock"): "non_common_symbol",
            ("ZVZZT", "stock"): "exchange_test_symbol", ("ZXZZT", "stock"): "exchange_test_symbol",
            ("ZIEXT", "stock"): "exchange_test_symbol", ("MTEST-A", "stock"): "exchange_test_symbol",
            ("ZBZX", "stock"): "exchange_test_symbol", ("ZVV", "stock"): "exchange_test_symbol",
            ("ZION", "stock"): None, ("ZVRA", "stock"): None, ("IBO", "stock"): None, ("TEST", "stock"): None,
        }
        for (ticker, kind), expected in cases.items():
            with self.subTest(ticker=ticker):
                self.assertEqual(exclusion_reason(ticker, kind), expected)

    def test_rule_validation(self):
        UniverseRule().validate()
        for rule in (UniverseRule(name="all"), UniverseRule(name="Top3000"), UniverseRule(size=0),
                     UniverseRule(adv_sessions=0), UniverseRule(min_history_sessions=True)):
            with self.subTest(rule=rule), self.assertRaises(DatasetError):
                rule.validate()


@unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is optional outside the Qlib environment")
class RankingTests(unittest.TestCase):
    def panel(self, sessions=90, tickers=("AAA", "BBB", "CCC", "DDD", "EEE")):
        import numpy as np
        import pandas as pd

        dates = pd.bdate_range("2024-01-02", periods=sessions)
        rng = np.random.default_rng(7)
        dollars = pd.DataFrame(rng.uniform(1e6, 5e6, (sessions, len(tickers))), index=dates, columns=list(tickers))
        valid = pd.DataFrame(True, index=dates, columns=list(tickers))
        return dollars, valid

    def test_rank_sessions_are_month_ends_except_the_open_month(self):
        import pandas as pd
        from qlib_yahoo.universe import rank_sessions

        sessions = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01", "2024-03-04"])
        self.assertEqual([f"{d:%Y-%m-%d}" for d in rank_sessions(sessions)], ["2024-01-31", "2024-02-29"])

    def test_membership_is_causal_sized_and_effective_next_session(self):
        from qlib_yahoo.universe import liquidity_membership, rank_sessions

        dollars, valid = self.panel()
        rule = UniverseRule(name="top2", size=2, adv_sessions=5, min_history_sessions=10)
        membership, rankings = liquidity_membership(dollars, valid, rule)
        ranks = rank_sessions(dollars.index)
        self.assertEqual(list(rankings["ranked_on"]), list(ranks))
        self.assertTrue((membership.groupby("ranked_on").size() == 2).all())
        sessions = list(dollars.index)
        for row in rankings.itertuples():
            self.assertEqual(sessions.index(row.effective_start), sessions.index(row.ranked_on) + 1)
        # Changing data after a ranking session must not change that ranking.
        first = ranks[0]
        changed = dollars.copy()
        changed.loc[changed.index > first, "EEE"] = 1e12
        changed_membership, _ = liquidity_membership(changed, valid, rule)
        before = membership[membership["ranked_on"] == first][["instrument", "rank"]].reset_index(drop=True)
        after = changed_membership[changed_membership["ranked_on"] == first][["instrument", "rank"]].reset_index(drop=True)
        self.assertTrue(before.equals(after))
        later = changed_membership[changed_membership["ranked_on"] == ranks[-1]]
        self.assertIn("EEE", set(later["instrument"]))

    def test_adv_is_the_trailing_window_mean(self):
        from qlib_yahoo.universe import liquidity_membership

        dollars, valid = self.panel(sessions=45)
        rule = UniverseRule(name="top5", size=5, adv_sessions=5, min_history_sessions=1)
        membership, _ = liquidity_membership(dollars, valid, rule)
        row = membership.iloc[0]
        position = list(dollars.index).index(row["ranked_on"])
        expected = dollars[row["instrument"]].iloc[position - 4: position + 1].mean()
        self.assertAlmostEqual(row["adv"], expected, delta=expected * 1e-9)

    def test_history_invalid_bars_and_ties(self):
        from qlib_yahoo.universe import liquidity_membership

        dollars, valid = self.panel(sessions=45)
        dollars.loc[:, :] = 1e6  # all tied: alphabetical order decides
        young = dollars.copy()
        valid.loc[valid.index[:25], "BBB"] = False  # too little history at the first ranking
        young.loc[valid.index[:25], "BBB"] = 0.0
        rule = UniverseRule(name="top2", size=2, adv_sessions=5, min_history_sessions=15)
        membership, rankings = liquidity_membership(young, valid, rule)
        first = membership[membership["ranked_on"] == rankings["ranked_on"].iloc[0]]
        self.assertEqual(list(first["instrument"]), ["AAA", "CCC"])
        # A session without a valid bar makes the ticker ineligible that ranking.
        valid.loc[rankings["ranked_on"].iloc[-1], "AAA"] = False
        later, _ = liquidity_membership(young, valid, rule)
        self.assertNotIn("AAA", set(later[later["ranked_on"] == rankings["ranked_on"].iloc[-1]]["instrument"]))

    def test_shortfall_is_recorded_not_filled(self):
        from qlib_yahoo.universe import liquidity_membership

        dollars, valid = self.panel(sessions=45)
        membership, rankings = liquidity_membership(dollars, valid, UniverseRule(name="top9", size=9, adv_sessions=5, min_history_sessions=1))
        self.assertTrue((rankings["members"] == 5).all())
        self.assertTrue((rankings["eligible"] == 5).all())

    def test_spans_merge_and_file_roundtrip(self):
        import pandas as pd
        from qlib_yahoo.universe import (liquidity_membership, membership_spans, members_between, read_instrument_file,
                                         write_instrument_file)

        dollars, valid = self.panel(sessions=130)
        dollars["AAA"] = 9e9
        dollars.loc[dollars.index[70:95], "BBB"] = 1.0  # drops out for one period, then returns
        membership, _ = liquidity_membership(dollars, valid, UniverseRule(name="top2", size=2, adv_sessions=5, min_history_sessions=1))
        spans = membership_spans(membership)
        self.assertEqual(len(spans["AAA"]), 1)
        self.assertEqual(spans["AAA"][0][1], dollars.index[-1])
        self.assertGreaterEqual(len(spans["BBB"]), 2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "top2.txt"
            write_instrument_file(path, spans)
            self.assertEqual(read_instrument_file(path), spans)
            path.write_text("AAA\t2024-01-02\t2024-03-01\nAAA\t2024-02-01\t2024-04-01\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "overlap"):
                read_instrument_file(path)
        self.assertIn("AAA", members_between(spans, pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-05")))

    def test_a_ranking_without_members_splits_spans(self):
        import pandas as pd
        from qlib_yahoo.universe import membership_spans

        day = pd.Timestamp
        rankings = pd.DataFrame({"effective_start": [day("2024-02-01"), day("2024-03-01"), day("2024-04-01")],
                                 "effective_end": [day("2024-02-29"), day("2024-03-28"), day("2024-04-30")]})
        membership = rankings.drop(index=1).assign(instrument="AAA")  # nobody qualified in March
        self.assertEqual(membership_spans(membership, rankings)["AAA"],
                         [(day("2024-02-01"), day("2024-02-29")), (day("2024-04-01"), day("2024-04-30"))])
        self.assertEqual(len(membership_spans(membership)["AAA"]), 1)  # without rankings the gap is invisible

    def test_mask_keeps_only_member_observations(self):
        import pandas as pd
        from qlib_yahoo.universe import mask_to_members

        index = pd.MultiIndex.from_product([pd.bdate_range("2024-01-01", periods=6), ["AAA", "BBB"]],
                                           names=["datetime", "instrument"])
        scores = pd.Series(range(12), index=index, dtype=float)
        spans = {"AAA": [(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")), (pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08"))]}
        kept = mask_to_members(scores, spans)
        self.assertEqual(set(kept.index.get_level_values("instrument")), {"AAA"})
        self.assertEqual([f"{d:%m-%d}" for d in kept.index.get_level_values("datetime")], ["01-02", "01-03", "01-05", "01-08"])

    def test_dataset_universe_checks(self):
        import pandas as pd
        from qlib_yahoo.universe import load_dataset_universe, write_instrument_file

        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            (dataset / "qlib_data" / "instruments").mkdir(parents=True)
            path = dataset / "qlib_data" / "instruments" / "top2.txt"
            write_instrument_file(path, {"AAA": [(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-01"))]})
            manifest = {"symbols": ["AAA"], "universes": {"top2": {"instruments_file": "instruments/top2.txt",
                                                                    "instruments_sha256": sha256(path)}}}
            spans, record = load_dataset_universe(dataset, manifest, "top2")
            self.assertEqual(set(spans), {"AAA"})
            with self.assertRaisesRegex(DatasetError, "available: top2"):
                load_dataset_universe(dataset, manifest, "top9")
            with self.assertRaisesRegex(DatasetError, "outside the dataset"):
                load_dataset_universe(dataset, {**manifest, "symbols": ["BBB"]}, "top2")
            path.write_text(path.read_text() + "BBB\t2024-01-02\t2024-01-05\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "differs from the dataset manifest"):
                load_dataset_universe(dataset, json.loads(json.dumps(manifest)), "top2")


if __name__ == "__main__":
    unittest.main()
