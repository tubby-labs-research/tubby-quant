"""Alpha Lab operators against naive references, including ties, empty sides, overflow and option errors.

Complements tests/test_alphalab.py: every operator here is checked against a loop
or closed form on data with ties and missing values. Semantics follow the
documented conventions in qlib_yahoo/alphalab/ops.py; where BRAIN's own choice
is not public (tie direction of ts_arg_*, dense decay), the test pins the
project's documented behaviour rather than asserting equivalence with BRAIN.
"""

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

HAVE_PANDAS = bool(importlib.util.find_spec("pandas") and importlib.util.find_spec("numpy"))

if HAVE_PANDAS:
    import numpy as np


def naive_ts(fn, x, d):
    out = np.full(x.shape, np.nan)
    for t in range(d - 1, len(x)):
        for j in range(x.shape[1]):
            window = x[t - d + 1:t + 1, j].astype(np.float64)
            if not np.isnan(window).any():
                out[t, j] = fn(window)
    return out


def average_rank(values):
    """1-based average ranks of a 1-D array without NaN, ties averaged."""
    ranks = np.empty(len(values))
    for i, v in enumerate(values):
        ranks[i] = np.sum(values < v) + (np.sum(values == v) + 1) / 2
    return ranks


@unittest.skipUnless(HAVE_PANDAS, "numpy and pandas are optional outside the Qlib environment")
class TimeSeriesReferenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        # Rounded values create ties; a few holes exercise the full-window rule.
        self.x = np.round(rng.normal(size=(70, 6)), 1).astype(np.float32)
        self.y = np.round(rng.normal(size=(70, 6)), 1).astype(np.float32)
        self.x[7, 1] = np.nan
        self.x[20:23, 3] = np.nan
        self.y[9, 0] = np.nan

    def close(self, got, want, **kwargs):
        np.testing.assert_allclose(np.asarray(got, dtype=np.float64), want, rtol=2e-5, atol=2e-6, equal_nan=True, **kwargs)

    def test_window_statistics(self):
        from qlib_yahoo.alphalab import ops

        d = 6
        self.close(ops.ts_min(self.x, d), naive_ts(np.min, self.x, d))
        self.close(ops.ts_zscore(self.x, d), naive_ts(lambda w: (w[-1] - w.mean()) / w.std(ddof=1) if w.std(ddof=1) > 0 else np.nan, self.x, d))
        self.close(ops.ts_ir(self.x, d), naive_ts(lambda w: w.mean() / w.std(ddof=1) if w.std(ddof=1) > 0 else np.nan, self.x, d))
        self.close(ops.ts_product(self.x, 4), naive_ts(np.prod, self.x, 4))
        self.close(ops.ts_count_nans(self.x, d), naive_ts(lambda w: 0.0, np.nan_to_num(self.x), d)
                   + np.vstack([np.full((d - 1, 6), np.nan),
                                np.array([[np.isnan(self.x[t - d + 1:t + 1, j]).sum() for j in range(6)]
                                          for t in range(d - 1, len(self.x))], dtype=float)]))
        want = np.full(self.x.shape, np.nan)
        for t in range(d - 1, len(self.x)):
            for j in range(6):
                a, b = self.x[t - d + 1:t + 1, j].astype(float), self.y[t - d + 1:t + 1, j].astype(float)
                if not (np.isnan(a).any() or np.isnan(b).any()):
                    want[t, j] = np.cov(a, b)[0, 1]
        self.close(ops.ts_covariance(self.x, self.y, d), want)

    def test_argmax_argmin_ties_prefer_the_most_recent_day_and_chunking_is_invisible(self):
        from qlib_yahoo.alphalab import ops

        d = 5
        recent_max = naive_ts(lambda w: int(np.argmax(w[::-1])), self.x, d)
        recent_min = naive_ts(lambda w: int(np.argmin(w[::-1])), self.x, d)
        self.close(ops.ts_arg_max(self.x, d), recent_max)
        self.close(ops.ts_arg_min(self.x, d), recent_min)
        with patch.object(ops, "_ARG_CHUNK_CELLS", 7):  # one or two rows per block
            self.close(ops.ts_arg_max(self.x, d), recent_max)
            self.close(ops.ts_arg_min(self.x, d), recent_min)
        self.assertTrue(np.isnan(ops.ts_arg_max(self.x, 80)).all())  # window longer than the history

    def test_backfill_and_dense_decay(self):
        from qlib_yahoo.alphalab import ops

        d = 3
        want = self.x.astype(np.float64).copy()
        for j in range(6):
            last = None
            for t in range(len(self.x)):
                if not np.isnan(self.x[t, j]):
                    last = t
                elif last is not None and t - last <= d:
                    want[t, j] = self.x[last, j]
        self.close(ops.ts_backfill(self.x, d), want)
        weights = np.arange(d, 0, -1, dtype=float)  # today heaviest
        dense = np.full(self.x.shape, np.nan)
        for t in range(d - 1, len(self.x)):
            for j in range(6):
                window = self.x[t - d + 1:t + 1, j][::-1].astype(float)
                ok = ~np.isnan(window)
                if ok.any():
                    dense[t, j] = np.dot(window[ok], weights[ok]) / weights[ok].sum()
        self.close(ops.ts_decay_linear(self.x, d, dense=True), dense)
        sparse = ops.ts_decay_linear(self.x, d)
        self.assertTrue(np.isnan(sparse[20:25, 3]).all())  # a hole in the window gives NaN by default

    def test_regression_return_types(self):
        from qlib_yahoo.alphalab import ops

        d = 8
        rows = range(d - 1, len(self.x))
        fits = {}
        for t in rows:
            for j in range(6):
                a, b = self.y[t - d + 1:t + 1, j].astype(float), self.x[t - d + 1:t + 1, j].astype(float)
                if not (np.isnan(a).any() or np.isnan(b).any()) and np.ptp(b) > 0:
                    slope, intercept = np.polyfit(b, a, 1)
                    fits[(t, j)] = (a[-1] - (intercept + slope * b[-1]), intercept, intercept + slope * b[-1],
                                    np.corrcoef(a, b)[0, 1] ** 2)
        for rettype, position in ((0, 0), (1, 1), (3, 2), (6, 3)):
            got = ops.ts_regression(self.y, self.x, d, rettype=rettype)
            for (t, j), values in fits.items():
                self.assertAlmostEqual(float(got[t, j]), values[position], places=3, msg=f"rettype {rettype} at {t},{j}")


@unittest.skipUnless(HAVE_PANDAS, "numpy and pandas are optional outside the Qlib environment")
class CrossSectionReferenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.x = np.round(rng.normal(size=(12, 9)), 1).astype(np.float32)
        self.x[2, 4] = np.nan
        self.members = np.ones_like(self.x, dtype=bool)
        self.members[:, 8] = False
        self.groups = np.array([0, 0, 1, 1, 1, 2, 2, np.nan, 0])

    def close(self, got, want):
        np.testing.assert_allclose(np.asarray(got, dtype=np.float64), want, rtol=2e-5, atol=2e-6, equal_nan=True)

    def test_rank_averages_ties(self):
        from qlib_yahoo.alphalab import ops

        got = ops.rank(self.x, self.members)
        for t in range(len(self.x)):
            row = self.x[t, :8].astype(float)
            ok = ~np.isnan(row)
            want = np.full(9, np.nan)
            want[np.flatnonzero(ok)] = (average_rank(row[ok]) - 1) / (ok.sum() - 1)
            self.close(got[t], want)

    def test_normalize_winsorize_and_scale_sides(self):
        from qlib_yahoo.alphalab import ops

        row = self.x[0, :8].astype(float)
        self.close(ops.normalize(self.x, self.members)[0, :8], row - row.mean())
        self.close(ops.normalize(self.x, self.members, useStd=True, limit=0.5)[0, :8],
                   np.clip((row - row.mean()) / row.std(), -0.5, 0.5))
        self.close(ops.winsorize(self.x, self.members, std=1.0)[0, :8],
                   np.clip(row, row.mean() - row.std(), row.mean() + row.std()))
        scaled = ops.scale(self.x, self.members, scale=3.0)
        self.close(np.nansum(np.abs(scaled), axis=1), np.full(len(self.x), 3.0))
        sides = ops.scale(self.x, self.members, longscale=2.0, shortscale=0.5)
        has_long = np.nansum(np.where(self.x[:, :8] > 0, 1.0, 0.0), axis=1) > 0
        has_short = np.nansum(np.where(self.x[:, :8] < 0, 1.0, 0.0), axis=1) > 0
        self.close(np.nansum(np.where(sides > 0, sides, 0.0), axis=1), np.where(has_long, 2.0, 0.0))
        self.close(-np.nansum(np.where(sides < 0, sides, 0.0), axis=1), np.where(has_short, 0.5, 0.0))
        self.assertTrue(np.isnan(sides[:, 8]).all())
        self.assertTrue(np.isnan(sides[2, 4]))
        all_positive = np.abs(self.x) + np.float32(0.1)
        one_sided = ops.scale(all_positive, self.members, longscale=1.0, shortscale=1.0)
        self.close(np.nansum(one_sided, axis=1), np.ones(len(self.x)))  # the empty short side is 0, not 0/0
        self.assertEqual(int(np.isnan(one_sided[:, :8]).sum()), 1)  # only the missing input stays missing

    def test_vector_neut_is_orthogonal_to_the_demeaned_regressor(self):
        from qlib_yahoo.alphalab import ops

        y = np.round(np.random.default_rng(9).normal(size=self.x.shape), 1).astype(np.float32)
        got = ops.vector_neut(self.x, y, self.members)
        for t in range(len(self.x)):
            a, b = self.x[t, :8].astype(float), y[t, :8].astype(float)
            ok = ~np.isnan(a) & ~np.isnan(b)
            a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
            residual = got[t, :8][ok].astype(float)
            self.assertAlmostEqual(float(np.dot(residual, b)), 0.0, places=4)
            self.close(residual, a - np.dot(a, b) / np.dot(b, b) * b)

    def test_group_statistics_against_per_group_loops(self):
        from qlib_yahoo.alphalab import ops

        mean = ops.group_mean(self.x, self.groups, self.members)
        zscore = ops.group_zscore(self.x, self.groups, self.members)
        ranked = ops.group_rank(self.x, self.groups, self.members)
        for t in range(len(self.x)):
            for code in (0, 1, 2):
                columns = [j for j in range(8) if self.groups[j] == code and not np.isnan(self.x[t, j])]
                values = self.x[t, columns].astype(float)
                self.close(mean[t, columns], np.full(len(columns), values.mean()))
                std = values.std()
                self.close(zscore[t, columns], (values - values.mean()) / std if std > 0 else np.full(len(columns), np.nan))
                want = (average_rank(values) - 1) / (len(values) - 1) if len(values) > 1 else np.array([0.5])
                self.close(ranked[t, columns], want)
        for out in (mean, zscore, ranked):
            self.assertTrue(np.isnan(out[:, 7]).all())  # unknown group
            self.assertTrue(np.isnan(out[:, 8]).all())  # outside the universe
            self.assertTrue(np.isnan(out[2, 4]))

    def test_group_zscore_is_stable_for_offset_series(self):
        from qlib_yahoo.alphalab import ops

        base = self.x.astype(np.float64)
        offset = base + 1e7  # E[x^2] - E[x]^2 loses ~2 decimal digits here; the two-pass form does not
        self.close(ops.group_zscore(offset, self.groups, self.members)[:, :7],
                   ops.group_zscore(base, self.groups, self.members)[:, :7])

    def test_large_group_labels_do_not_allocate_by_label_value(self):
        from qlib_yahoo.alphalab import ops

        huge = np.array([1e9, 1e9, 2e9, 2e9, 2e9, 3e9, 3e9, np.nan, 1e9])
        codes, groups = ops._group_codes(huge, self.x.shape)
        self.assertEqual(groups, 3)
        self.assertEqual(sorted(set(codes[0].tolist())), [-1, 0, 1, 2])
        self.close(ops.group_neutralize(self.x, huge, self.members), ops.group_neutralize(self.x, self.groups, self.members))
        self.close(ops.group_rank(self.x, huge, self.members), ops.group_rank(self.x, self.groups, self.members))

    def test_bucket_edges(self):
        from qlib_yahoo.alphalab import ops

        values = np.array([[0.0, 0.05, 0.1, 0.55, 0.95, 1.0, np.nan, 0.3, 0.75]], dtype=np.float32)
        members = np.ones_like(values, dtype=bool)
        got = ops.bucket(values, members, range="0,1,0.1")
        self.close(got[0], [0, 0, 1, 5, 9, 9, np.nan, 3, 7])
        custom = ops.bucket(values, members, buckets="0,0.5,1")
        self.close(custom[0], [0, 0, 0, 1, 1, 1, np.nan, 0, 1])


@unittest.skipUnless(HAVE_PANDAS, "numpy and pandas are optional outside the Qlib environment")
class ElementwiseAndErrorTests(unittest.TestCase):
    def context(self):
        from qlib_yahoo.alphalab.engine import Context

        rng = np.random.default_rng(2)
        shape = (30, 5)
        close = (np.exp(np.cumsum(rng.normal(0, 0.02, shape), axis=0)) * 50).astype(np.float32)
        fields = {"close": close, "open": close * np.float32(0.99), "high": close * np.float32(1.01),
                  "low": close * np.float32(0.98), "volume": np.full(shape, 1e6, dtype=np.float32)}
        return Context(dates=np.arange(shape[0]), symbols=[f"S{i}" for i in range(shape[1])], fields=fields,
                       members=np.ones(shape, dtype=bool), groups={"sector": np.array([0, 0, 1, 1, 1], dtype=float)})

    def test_overflow_becomes_missing_not_infinite(self):
        from qlib_yahoo.alphalab import ops
        from qlib_yahoo.alphalab.engine import evaluate

        big = np.array([[100.0, 1.0]], dtype=np.float32)
        exp = ops.ELEMENTWISE["exp"](big)
        self.assertTrue(np.isnan(exp[0, 0]) and np.isfinite(exp[0, 1]))
        self.assertFalse(np.isinf(ops.power(np.array([[1e6]], dtype=np.float32), 7)).any())
        value = evaluate("volume ^ 7", self.context())
        self.assertTrue(np.isnan(value).all())
        value = evaluate("scale(exp(close / 100))", self.context())  # no inf poisoning the row
        self.assertTrue(np.isfinite(value).all())

    def test_elementwise_references(self):
        from qlib_yahoo.alphalab import ops

        a = np.array([[-2.0, 0.0, 3.0, np.nan]], dtype=np.float32)
        b = np.array([[1.0, 2.0, np.nan, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(ops.signed_power(a, 2)[0], [-4.0, 0.0, 9.0, np.nan], equal_nan=True)
        np.testing.assert_allclose(ops.elementwise_min(a, b)[0], [-2.0, 0.0, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(ops.elementwise_max(a, b)[0], [1.0, 2.0, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(ops.ELEMENTWISE["log"](a)[0], [np.nan, np.nan, np.log(3.0), np.nan], equal_nan=True)
        np.testing.assert_allclose(ops.ELEMENTWISE["sqrt"](a)[0], [np.nan, 0.0, np.sqrt(3.0), np.nan], equal_nan=True)
        np.testing.assert_allclose(ops._binary("&&", a, b)[0], [0.0, 0.0, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(ops._binary("||", a, b)[0], [1.0, 1.0, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(ops.if_else(a, b, -b)[0], [-1.0, -2.0, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(ops.clamp(a, -1.0, 1.0)[0], [-1.0, 0.0, 1.0, np.nan], equal_nan=True)

    def test_option_type_errors_are_expression_errors(self):
        from qlib_yahoo.alphalab.fastexpr import ExpressionError
        from qlib_yahoo.alphalab.engine import evaluate

        context = self.context()
        for expression in ('ts_rank(close, 5, constant="a")', 'winsorize(close, std="a")', 'hump(close, hump="a")',
                           'clamp(close, lower="a")', 'normalize(close, limit="a")', 'scale(close, scale="a")',
                           'ts_regression(close, open, 5, rettype="a")', 'bucket(rank(close), buckets="a,b")',
                           "clamp(close, 1, -1)"):
            with self.subTest(expression=expression), self.assertRaises(ExpressionError) as caught:
                evaluate(expression, context)
            self.assertIn(expression.split("(")[0], str(caught.exception))

    def test_weights_apply_cost_decay_and_truncation(self):
        from qlib_yahoo.alphalab import ops
        from qlib_yahoo.alphalab.settings import Settings
        from qlib_yahoo.alphalab.simulate import _truncate, weights

        rng = np.random.default_rng(4)
        alpha = rng.normal(size=(40, 6)).astype(np.float32)
        members = np.ones_like(alpha, dtype=bool)
        tradable = np.ones_like(alpha, dtype=bool)
        settings = Settings(universe="TOP6", delay=1, decay=4, neutralization="NONE", truncation=0.0).validate()
        got = weights(alpha, members, tradable, settings, {})
        shifted = np.full_like(alpha, np.nan, dtype=np.float64)
        shifted[1:] = alpha[:-1]
        decayed = ops.ts_decay_linear(shifted, 4, dense=True).astype(np.float64)
        decayed = np.where(np.isfinite(decayed), decayed, 0.0)
        want = decayed / np.maximum(np.abs(decayed).sum(axis=1, keepdims=True), 1e-12)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
        capped = _truncate(want, 0.2)
        self.assertLessEqual(float(np.abs(capped).max()), 0.2 + 1e-6)
        invested = np.abs(want).sum(axis=1) > 0
        self.assertTrue(invested[4:].all())
        np.testing.assert_allclose(np.abs(capped[invested]).sum(axis=1), np.ones(invested.sum()), rtol=1e-4)
        self.assertEqual(float(np.abs(capped[~invested]).sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
