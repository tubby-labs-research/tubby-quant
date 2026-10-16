"""Independent portfolio-neutrality, classification alignment and causality fixtures."""

from __future__ import annotations

import importlib.util
import time
import unittest
from unittest.mock import patch

HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
if HAVE_NUMPY:
    import numpy as np

    from qlib_yahoo.alphalab.simulate import ENGINE_VERSION, Settings, Workspace, _truncate, simulate, weights
    from qlib_yahoo.dataset import DatasetError


def reference(alpha, codes, cap):
    """Small independent oracle: per-group mean, then bisection for each capped side."""
    raw = np.zeros_like(alpha, dtype=float)
    known = np.isfinite(codes) & (codes >= 0) & np.isfinite(alpha)
    for code in np.unique(codes[known]):
        selected = known & (codes == code)
        raw[selected] = alpha[selected] - alpha[selected].mean()
    gross = np.abs(raw).sum()
    if gross == 0:
        return raw
    raw /= gross
    out = np.zeros_like(raw)
    for code in np.unique(codes[known]):
        group = known & (codes == code)
        positive, negative = group & (raw > 0), group & (raw < 0)
        target = min(raw[positive].sum(), -raw[negative].sum())
        if cap:
            target = min(target, positive.sum() * cap, negative.sum() * cap)
        for selected, sign in ((positive, 1), (negative, -1)):
            values = np.abs(raw[selected])
            if not target or not len(values):
                continue
            if not cap:
                out[selected] = sign * values * target / values.sum()
                continue
            low, high = 0.0, cap / values.min()
            for _ in range(90):
                mid = (low + high) / 2
                if np.minimum(values * mid, cap).sum() < target:
                    low = mid
                else:
                    high = mid
            out[selected] = sign * np.minimum(values * ((low + high) / 2), cap)
    return out


@unittest.skipUnless(HAVE_NUMPY, "NumPy is optional outside the Qlib environment")
class NeutralizationTests(unittest.TestCase):
    def book(self, alpha, codes=None, cap=.25, mode="MARKET", **kwargs):
        alpha = np.asarray(alpha, dtype=float)
        if alpha.ndim == 1:
            alpha = alpha[None, :]
        mask = np.ones_like(alpha, dtype=bool)
        return weights(alpha, mask, mask, Settings(delay=0, neutralization=mode, truncation=cap),
                       {} if codes is None else {mode.lower(): np.asarray(codes)}, **kwargs)

    def test_confirmed_outlier_cap_has_equal_sides(self):
        w = self.book([20, 1, 0, 0, 0])[0]
        np.testing.assert_allclose(w, [.25, -.25 * 3.2 / 15.8, *([-.25 * 4.2 / 15.8] * 3)], atol=1e-15)
        self.assertAlmostEqual(w.sum(), 0, places=14)
        self.assertAlmostEqual(np.abs(w).sum(), .5)

    def test_unequal_groups_do_not_share_constrained_budget(self):
        alpha = np.array([-1, 1, 20, 1, 0, 0, 0.])
        codes = np.array([0, 0, 1, 1, 1, 1, 1])
        w = self.book(alpha, codes, mode="INDUSTRY")[0]
        np.testing.assert_allclose(w[:2], [-1 / 33.6, 1 / 33.6], atol=1e-15)
        self.assertAlmostEqual(np.abs(w[2:]).sum(), .5)
        np.testing.assert_allclose(w, reference(alpha, codes, .25), atol=1e-14)

    def test_infeasible_cap_accepts_smaller_neutral_book(self):
        w = self.book([10, 1, 0, 0, 0], cap=.01)[0]
        self.assertAlmostEqual(w[w > 0].sum(), .01)
        self.assertAlmostEqual(-w[w < 0].sum(), .01)
        self.assertLessEqual(np.abs(w).max(), .01 + 1e-15)

    def test_water_filling_redistributes_only_on_same_side(self):
        alpha = np.array([10, 2, 1, -7, -6.])  # mean zero; first long saturates
        w = self.book(alpha, cap=.3)[0]
        np.testing.assert_allclose(w, [.3, 2 / 15, 1 / 15, -7 / 26, -6 / 26], atol=1e-14)

    def test_signed_groups_singletons_constant_and_unknown(self):
        alpha = [-4, 4, -10, 10, 900, 1000, -1000, 7, 7]
        codes = [0, 0, 1, 1, 2, np.nan, -1, 3, 3]
        for mode in ("INDUSTRY", "SECTOR", "COUNTRY"):
            w = self.book(alpha, codes, mode=mode)[0]
            np.testing.assert_allclose(w, reference(np.array(alpha), np.array(codes), .25), atol=1e-14)
            np.testing.assert_array_equal(w[4:], 0)

    def test_market_needs_no_classification(self):
        np.testing.assert_allclose(self.book([1, 2], cap=0), [[-.5, .5]])

    def test_precise_float64_demeaning_avoids_operator_float32(self):
        alpha = np.array([1e12 + 1, 1e12 + 2, 1e12 + 3, 1e12 + 8])
        with patch("qlib_yahoo.alphalab.ops.group_neutralize", side_effect=AssertionError("float32 path")):
            w = self.book(alpha, [0] * 4, cap=0, mode="SECTOR")[0]
        self.assertLessEqual(abs(w.sum()), 1e-10)
        self.assertAlmostEqual(np.abs(w).sum(), 1)
        np.testing.assert_allclose(w, [-2.5 / 9, -1.5 / 9, -.5 / 9, .5], atol=1e-14)
        # Near the float64 range limit, intermediate means must not overflow.
        w = self.book([1e308, -1e308, 1e307, -1e307], cap=0)[0]
        self.assertTrue(np.isfinite(w).all())
        self.assertLessEqual(abs(w.sum()), 1e-10)

    def test_masks_nonfinite_and_empty_days(self):
        alpha = np.array([[1, 3, np.inf, 100], [np.nan] * 4, [2] * 4])
        members = np.array([[True, True, True, False]] * 3)
        diagnostics = {}
        w = weights(alpha, members, np.ones_like(members), Settings(delay=0, truncation=0), {},
                    diagnostics=diagnostics)
        np.testing.assert_array_equal(w, [[-.5, .5, 0, 0], [0] * 4, [0] * 4])
        self.assertEqual(diagnostics["inactive_neutralized_days"], 2)
        self.assertEqual(diagnostics["classification_coverage"], 1)

    def test_sparse_codes_and_huge_singleton_do_not_erase_small_group(self):
        w = self.book([-1e-100, 1e-100, 1e308], [1e12, 1e12, 2e12], cap=0, mode="SECTOR")
        np.testing.assert_allclose(w, [[-.5, .5, 0]], atol=1e-14)

    def test_classification_only_outside_eligible_members_is_unavailable(self):
        alpha = np.array([[1, 2, 3, 4.]])
        members = np.array([[True, True, False, True]])
        tradable = np.array([[True, True, True, False]])
        with self.assertRaisesRegex(DatasetError, "finite eligible names"):
            weights(alpha, members, tradable, Settings(delay=0, neutralization="INDUSTRY"),
                    {"industry": np.array([np.nan, -1, 0, 0])})

    def test_pasteurization_off_allows_classified_nonmember_positions(self):
        alpha = np.array([[1, 2, -4, 4.]])
        members = np.array([[True, True, False, False]])
        tradable = np.ones_like(members)
        groups = {"industry": np.array([np.nan, -1, 0, 0])}
        with self.assertRaisesRegex(DatasetError, "finite eligible names"):
            weights(alpha, members, tradable, Settings(delay=0, neutralization="INDUSTRY"), groups)
        diagnostics = {}
        w = weights(alpha, members, tradable,
                    Settings(delay=0, neutralization="INDUSTRY", truncation=.25, pasteurization=False),
                    groups, diagnostics=diagnostics)
        np.testing.assert_allclose(w, [[0, 0, -.25, .25]], atol=1e-14)
        self.assertEqual(diagnostics["classification_coverage"], .5)
        self.assertEqual(diagnostics["inactive_neutralized_days"], 0)

    def test_pasteurization_coverage_and_position_set_on_off(self):
        alpha = np.array([[-2, 2, -8, 8.]])
        members = np.array([[True, True, False, False]])
        tradable = np.ones_like(members)
        groups = {"industry": np.array([0, 0, 1, np.nan])}
        for enabled, expected_coverage in ((True, 1), (False, .75)):
            diagnostics = {}
            w = weights(alpha, members, tradable,
                        Settings(delay=0, neutralization="INDUSTRY", truncation=0, pasteurization=enabled),
                        groups, diagnostics=diagnostics)
            np.testing.assert_allclose(w, [[-.5, .5, 0, 0]], atol=1e-14)
            self.assertEqual(diagnostics["classification_coverage"], expected_coverage)

    def test_none_is_historical_truncation_even_for_neutral_expression(self):
        alpha = np.array([[15.8, -3.2, -4.2, -4.2, -4.2]])
        expected = _truncate(alpha / np.abs(alpha).sum(axis=1, keepdims=True), .25)
        np.testing.assert_array_equal(self.book(alpha, mode="NONE"), expected)
        self.assertGreater(abs(expected.sum()), .1)

    def test_missing_and_wholly_unknown_classification_raise(self):
        for codes in (None, [np.nan, -1], [np.inf, np.nan]):
            with self.assertRaisesRegex(DatasetError, "classification"):
                self.book([1, 2], codes, mode="COUNTRY")

    def test_dated_codes_follow_current_day_without_future_leak(self):
        alpha = np.tile([-3, -1, 1, 3.], (12, 1))
        codes = np.tile([0, 0, 1, 1], (12, 1)).astype(float)
        codes[6:] = [0, 1, 0, 1]
        w = self.book(alpha, codes, cap=0, mode="SECTOR")
        for row in range(12):
            np.testing.assert_allclose(w[row], reference(alpha[row], codes[row], 0), atol=1e-14)
        changed = codes.copy()
        changed[8:] = np.nan
        np.testing.assert_array_equal(self.book(alpha, changed, cap=0, mode="SECTOR")[:8], w[:8])
        np.testing.assert_array_equal(self.book(alpha[:8], codes[:8], cap=0, mode="SECTOR"), w[:8])

    def test_delay_decay_are_causal_with_dated_groups(self):
        rng = np.random.default_rng(12)
        alpha = rng.normal(size=(30, 8))
        codes = np.tile(np.arange(8) // 2, (30, 1))
        mask = np.ones_like(alpha, bool)
        settings = Settings(delay=1, decay=4, neutralization="INDUSTRY", truncation=.1)
        w = weights(alpha, mask, mask, settings, {"industry": codes})
        altered = alpha.copy()
        altered[15:] *= -1000
        np.testing.assert_array_equal(weights(altered, mask, mask, settings, {"industry": codes})[:15], w[:15])

    def test_random_numeric_invariants_and_independent_oracle(self):
        rng = np.random.default_rng(840)
        for cap in (0, .001, .03, .2, 1):
            alpha = rng.normal(size=(25, 37)) * rng.lognormal(0, 3, size=(25, 37))
            codes = rng.integers(-1, 7, size=alpha.shape).astype(float)
            alpha[rng.random(alpha.shape) < .1] = np.nan
            codes[rng.random(alpha.shape) < .1] = np.nan
            w = self.book(alpha, codes, cap=cap, mode="INDUSTRY")
            self.assertTrue(np.isfinite(w).all())
            self.assertTrue((np.abs(w).sum(axis=1) <= 1 + 1e-10).all())
            if cap:
                self.assertLessEqual(np.abs(w).max(), cap + 1e-10)
            for row in range(len(w)):
                np.testing.assert_allclose(w[row], reference(alpha[row], codes[row], cap), atol=1e-10, rtol=1e-10)
                for code in np.unique(codes[row][np.isfinite(codes[row]) & (codes[row] >= 0)]):
                    self.assertLessEqual(abs(w[row, codes[row] == code].sum()), 1e-10)
            market = self.book(alpha, cap=cap)
            for row in range(len(market)):
                np.testing.assert_allclose(market[row], reference(alpha[row], np.zeros(alpha.shape[1]), cap),
                                           atol=1e-10, rtol=1e-10)

    def test_bounded_performance_fixture(self):
        rng = np.random.default_rng(1)
        alpha = rng.normal(size=(252, 3000))
        codes = np.tile(np.arange(3000) % 40, (252, 1))
        codes[126:] = (codes[126:] + np.arange(3000) % 3) % 40
        started = time.perf_counter()
        diagnostics = {}
        w = self.book(alpha, codes, cap=.001, mode="INDUSTRY", diagnostics=diagnostics)
        elapsed = time.perf_counter() - started
        print(f"\nneutralization fixture 252 x 3000, 40 dated groups: {elapsed:.3f}s")
        self.assertLess(elapsed, 15, "bounded local fixture exceeded a generous regression limit")
        self.assertLessEqual(diagnostics["max_abs_group_net_exposure"], 1e-10)
        self.assertLessEqual(np.abs(w).max(), .001 + 1e-10)

    def test_chunk_boundaries_do_not_change_weights_or_diagnostics(self):
        from qlib_yahoo.alphalab import simulate as simulator

        rng = np.random.default_rng(81)
        alpha = rng.standard_normal((145, 91), dtype=np.float32)
        original = alpha.copy()
        codes = rng.integers(-1, 9, size=alpha.shape)
        mask = np.ones_like(alpha, bool)
        settings = Settings(delay=0, neutralization="INDUSTRY", truncation=.02)
        diagnostic_sets, books = [], []
        for chunk_rows in (1, 17, 64):
            diagnostics = {}
            with patch.object(simulator, "NEUTRALIZATION_CHUNK_ROWS", chunk_rows), \
                    patch.object(simulator, "_neutral_chunk", wraps=simulator._neutral_chunk) as run:
                books.append(weights(alpha, mask, mask, settings, {"industry": codes},
                                     diagnostics=diagnostics, test_start=70))
                self.assertTrue(all(call.args[0].shape[0] <= chunk_rows for call in run.call_args_list))
            diagnostic_sets.append(diagnostics)
        np.testing.assert_array_equal(alpha, original)
        for book, diagnostics in zip(books, diagnostic_sets):
            np.testing.assert_allclose(book, books[0], atol=1e-14, rtol=1e-14)
            self.assertEqual(diagnostics, diagnostic_sets[0])
        for row in range(len(alpha)):
            np.testing.assert_allclose(books[0][row], reference(alpha[row].astype(float), codes[row], .02), atol=1e-12)

    def test_market_fast_path_never_sorts_codes(self):
        alpha = np.tile([20, 1, 0, 0, 0.], (145, 1))
        with patch("qlib_yahoo.alphalab.simulate.np.unique", side_effect=AssertionError("market sort")):
            diagnostics = {}
            w = self.book(alpha, diagnostics=diagnostics)
        np.testing.assert_allclose(w.sum(axis=1), 0, atol=1e-14)
        self.assertLessEqual(diagnostics["max_abs_group_net_exposure"], 1e-10)

    def test_classified_warmup_in_other_chunk_cannot_validate_empty_test(self):
        alpha = np.tile([1, 2, 3, 4.], (145, 1))
        codes = np.full(alpha.shape, -1)
        codes[:64] = [0, 0, 1, 1]
        with self.assertRaisesRegex(DatasetError, "selected test period"):
            self.book(alpha, codes, mode="INDUSTRY", test_start=70)


@unittest.skipUnless(HAVE_NUMPY, "NumPy is optional outside the Qlib environment")
class WorkspaceNeutralizationTests(unittest.TestCase):
    def workspace(self, codes):
        ws = Workspace.__new__(Workspace)
        ws.dates = np.arange("2024-01-01", "2024-02-20", dtype="datetime64[D]")
        ws.symbols = ["A", "B", "C", "D"]
        ws.fields = {"close": np.tile([10, 20, 30, 40.], (len(ws.dates), 1))}
        ws.identity = {"fixture": "dated-neutralization"}
        ws.groups = {"industry": np.asarray(codes, dtype=float)}
        ws._universes = {"TOP4": np.ones((len(ws.dates), 4), bool)}
        return ws

    def settings(self):
        return Settings(universe="TOP4", start="2024-01-11", end="2024-02-10", warmup=3,
                        delay=0, neutralization="INDUSTRY", truncation=.25)

    def test_context_slices_dated_codes_and_preserves_static_codes(self):
        codes = np.tile([0, 0, 1, 1], (50, 1))
        codes[10:] = [0, 1, 0, 1]
        ws = self.workspace(codes)
        ctx = ws.context(7, 40, ws.universe("TOP4"))
        np.testing.assert_array_equal(ctx.groups["industry"], codes[7:41])
        self.assertEqual(ctx.groups["industry"].shape, ctx.members.shape)
        ws.groups["industry"] = np.array([0, 0, 1, 1])
        self.assertIs(ws.context(7, 40, ws.universe("TOP4")).groups["industry"], ws.groups["industry"])

    def test_invalid_shapes_raise_in_workspace_and_weights(self):
        for shape in ((4, 1), (49, 4), (50, 3), (1, 4), (50, 4, 1), ()):
            ws = self.workspace(np.zeros(shape))
            with self.assertRaisesRegex(DatasetError, "shape"):
                ws.context(7, 40, ws.universe("TOP4"))
            with self.assertRaisesRegex(DatasetError, "shape"):
                weights(ws.fields["close"], ws.universe("TOP4"), ws.universe("TOP4"), self.settings(), ws.groups)

    def test_simulation_warmup_alignment_summary_and_partial_coverage(self):
        codes = np.tile([0, 0, 1, 1.], (50, 1))
        codes[10:20] = [0, 1, 0, 1]
        codes[20] = np.nan  # one empty test day
        codes[21:, 3] = np.nan
        ws = self.workspace(codes)
        result = simulate("close", self.settings(), ws, keep_weights=True)
        expected = np.vstack([reference(ws.fields["close"][r], codes[r], .25) for r in range(10, 41)])
        np.testing.assert_allclose(result["weights"], expected, atol=1e-14)
        summary = result["summary"]
        self.assertEqual(result["engine"], "alphalab-3")
        self.assertLessEqual(summary["max_abs_group_net_exposure"], 1e-10)
        self.assertEqual(summary["inactive_neutralized_days"], 1)
        self.assertAlmostEqual(summary["classification_coverage"], (124 - 4 - 20) / 124)
        self.assertTrue(all(np.isscalar(summary[k]) for k in
                            ("max_abs_group_net_exposure", "classification_coverage", "inactive_neutralized_days")))

    def test_warmup_and_future_classification_cannot_validate_empty_test(self):
        codes = np.full((50, 4), np.nan)
        codes[:10] = [0, 0, 1, 1]
        codes[41:] = [0, 0, 1, 1]
        with self.assertRaisesRegex(DatasetError, "selected test period"):
            simulate("close", self.settings(), self.workspace(codes))

    def test_coverage_denominator_is_finite_eligible_test_members(self):
        ws = self.workspace(np.tile([0, 0, np.nan, np.nan], (50, 1)))
        ws.fields["close"][10:41, 3] = np.nan
        ws._universes["TOP4"][10:41, 2] = False
        summary = simulate("close", self.settings(), ws)["summary"]
        self.assertEqual(summary["classification_coverage"], 1)

    def test_singleton_classifications_are_available_but_inactive(self):
        ws = self.workspace([0, 1, 2, 3])
        summary = simulate("close", self.settings(), ws)["summary"]
        self.assertEqual(summary["classification_coverage"], 1)
        self.assertEqual(summary["inactive_neutralized_days"], 31)
        self.assertEqual(summary["gross_exposure"], 0)

    def test_historical_result_identity_is_separate(self):
        from qlib_yahoo.alphalab import batch

        with patch.object(batch, "ENGINE_VERSION", "alphalab-2"):
            old = batch.alpha_id("close", self.settings(), "fixture")
        self.assertEqual(ENGINE_VERSION, "alphalab-3")
        self.assertNotEqual(old, batch.alpha_id("close", self.settings(), "fixture"))


if __name__ == "__main__":
    unittest.main()
