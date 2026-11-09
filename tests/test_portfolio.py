"""Hand-known S3 allocation/risk cases; no Qlib, market data or PnL ledger."""
from dataclasses import asdict, replace
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import unittest
from unittest import mock

if importlib.util.find_spec("numpy") is not None:
    import numpy as np

import qlib_yahoo.portfolio as portfolio

from qlib_yahoo.portfolio import (
    AllocationLeg, FixedAllocationPolicy, NormalizedTargets, RiskLimits,
    allocate_fixed, evaluate_risk, make_fixed_policy,
)


def targets(rows=((.5, -.25), (0., 0.)), **changes):
    options = dict(dates=("2024-01-05", "2024-01-08"), symbols=("B", "A"),
                   weights=rows, candidate_result_digest="a" * 64,
                   attempt_id="attempt-1", dataset_manifest_sha256="b" * 64,
                   price_identity="synthetic-price-v1")
    options.update(changes)
    return NormalizedTargets(**options)


def policy(inputs, coefficients):
    return make_fixed_policy(inputs, coefficients, account_id="one-usd-account",
                             capital=1000, policy_id="fixed-budget-1")


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required for target matrices")
class AllocationTests(unittest.TestCase):
    def test_one_signal_subunit_gross_one_capital(self):
        source = targets()
        result = allocate_fixed(policy([source], [.5]), [source])
        np.testing.assert_array_equal(result.weights, ((.25, -.125), (0., 0.)))
        np.testing.assert_array_equal(result.target_notionals, ((250., -125.), (0., 0.)))
        self.assertEqual(result.dates, source.dates)
        self.assertEqual(result.symbols, ("B", "A"))

    def test_two_signals_share_budget_without_renormalization(self):
        first = targets()
        second = targets(((-.25, .5), (.125, -.125)),
                         candidate_result_digest="c" * 64, attempt_id="attempt-2")
        result = allocate_fixed(policy([first, second], [.5, .25]), [first, second])
        np.testing.assert_array_equal(result.weights, ((.1875, 0.), (.03125, -.03125)))
        np.testing.assert_array_equal(result.target_notionals[0], (187.5, 0.))
        metadata = result.to_metadata()
        self.assertEqual(metadata["attempt_ids"], ["attempt-1", "attempt-2"])
        self.assertEqual(metadata["input_target_digests"], [first.digest, second.digest])
        self.assertEqual([x["coefficient"] for x in metadata["allocation_policy"]["legs"]], [.5, .25])
        self.assertEqual(metadata["weight_policy"], "portfolio_fixed_v1")

    def test_zero_signal_and_zero_coefficient(self):
        for source, coefficient in ((targets(((0., 0.), (0., 0.))), 1), (targets(), 0)):
            np.testing.assert_array_equal(allocate_fixed(policy([source], [coefficient]), [source]).weights,
                                          ((0., 0.), (0., 0.)))

    def test_short_direction_and_reversal_preserved(self):
        source = targets(((-.75, .25), (.75, -.25)))
        np.testing.assert_array_equal(allocate_fixed(policy([source], [1]), [source]).weights, source.weights)

    def test_budget_and_capital_rejected(self):
        first, second = targets(), targets(attempt_id="attempt-2")
        for coefficients in ((.6, .5), (-.1, .5), (math.nan, 0), (True, 0)):
            with self.subTest(coefficients=coefficients), self.assertRaises(ValueError):
                policy([first, second], coefficients)
        for capital in (0, -1, math.inf, True):
            with self.subTest(capital=capital), self.assertRaises(ValueError):
                make_fixed_policy([first], [1], account_id="account", capital=capital, policy_id="p")

    def test_count_duplicate_and_policy_version_rejected(self):
        source = targets()
        for inputs, coefficients in (([], []), ([source], []), ([source] * 3, [0] * 3),
                                     ([source, source], [.5, .5])):
            with self.subTest(inputs=len(inputs)), self.assertRaises(ValueError):
                policy(inputs, coefficients)
        with self.assertRaises(ValueError):
            replace(policy([source], [1]), version="future")
        with self.assertRaises(ValueError):
            AllocationLeg("bad", "a" * 64, "a", .5)
        with self.assertRaises(ValueError):
            FixedAllocationPolicy("p", "a", 1, ())

    def test_missing_nonfinite_and_unnormalized_input_rejected(self):
        cases = [dict(weights=((None, 0), (0, 0))), dict(weights=((math.inf, 0), (0, 0))),
                 dict(weights=((math.nan, 0), (0, 0))), dict(weights=((True, 0), (0, 0))),
                 dict(weights=((.75, -.5), (0, 0))), dict(weights=((.1,), (0,))),
                 dict(weights=()), dict(candidate_result_digest=""), dict(attempt_id=""),
                 dict(dataset_manifest_sha256="bad"), dict(price_identity=""),
                 dict(dates=("2024-01-08", "2024-01-05")), dict(dates=("bad", "2024-01-08")),
                 dict(symbols=("A", "A"))]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                targets(**changes)

    def test_exact_axis_and_dataset_alignment_required_even_zero_coefficient(self):
        source = targets()
        for changes in (dict(symbols=("A", "B")), dict(dates=("2024-01-04", "2024-01-08")),
                        dict(dataset_manifest_sha256="d" * 64), dict(price_identity="different")):
            other = targets(attempt_id="attempt-2", **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                allocate_fixed(policy([source, other], [1, 0]), [source, other])

    def test_pinned_matrix_and_lineage_cannot_be_substituted(self):
        source = targets()
        pinned = policy([source], [1])
        for other in (targets(attempt_id="other"), targets(candidate_result_digest="c" * 64),
                      targets(((.5, -.125), (0, 0)))):
            with self.assertRaises(ValueError):
                allocate_fixed(pinned, [other])

    def test_caller_mutation_detached_and_metadata_detached(self):
        rows = [[.5, -.25], [0, 0]]
        source = targets(rows)
        pinned = policy([source], [1])
        identity = source.digest
        rows[0][0] = 99
        self.assertEqual(source.digest, identity)
        result = allocate_fixed(pinned, [source])
        metadata = result.to_metadata()
        metadata["allocation_policy"]["capital"] = 999
        self.assertEqual(result.policy.capital, 1000)

    def test_owned_contiguous_readonly_storage_and_dtype(self):
        for dtype in (np.float32, np.float64):
            original = np.array([[.5, -.25], [0., 0.]], dtype=dtype, order="F")
            source = targets(original)
            self.assertEqual(source.weights.dtype, dtype)
            self.assertTrue(source.weights.flags.c_contiguous)
            self.assertTrue(source.weights.flags.owndata)
            self.assertFalse(source.weights.flags.writeable)
            self.assertFalse(np.shares_memory(original, source.weights))
            original[0, 0] = 99
            self.assertEqual(source.weights[0, 0], .5)
            with self.assertRaises(ValueError):
                source.weights[0, 0] = 1
            result = allocate_fixed(policy([source], [.5]), [source])
            self.assertEqual(result.weights.dtype, np.float64)
            self.assertFalse(result.weights.flags.writeable)
            self.assertFalse(np.shares_memory(result.weights, source.weights))
            notionals = result.target_notionals
            self.assertIsInstance(notionals, np.ndarray)
            self.assertFalse(notionals.flags.writeable)
            self.assertFalse(np.shares_memory(notionals, result.weights))
            self.assertIsNot(notionals, result.target_notionals)

    def test_mutating_exposed_owner_storage_is_detected_even_zero_coefficient(self):
        source = targets(np.array([[.5, -.25], [0., 0.]], dtype=np.float32))
        pinned = policy([source], [0])
        previous = source.digest
        source.weights.setflags(write=True)  # frozen dataclass is not array immutability
        source.weights[0, 0] = .25
        source.weights.setflags(write=False)
        self.assertEqual(source.digest, previous)  # cached identity is the original seal
        with self.assertRaisesRegex(ValueError, "differs from construction seal"):
            allocate_fixed(pinned, [source])
        with self.assertRaisesRegex(ValueError, "differs from construction seal"):
            policy([source], [1])

    def test_cached_identity_does_not_rescan_storage(self):
        source = targets()
        with mock.patch.object(portfolio, "_matrix_digest", side_effect=AssertionError("rehashed")):
            self.assertEqual(source.digest, source.digest)

    def test_storage_shape_change_cannot_reuse_policy(self):
        source = targets()
        pinned = policy([source], [1])
        source.weights.shape = (1, 4)
        with self.assertRaisesRegex(ValueError, "construction seal"):
            allocate_fixed(pinned, [source])

    def test_binary_digest_dtype_axes_and_canonical_byte_order(self):
        first = targets(np.array([[.5, -.25], [0., 0.]], dtype="<f4"))
        second = targets(first.weights.astype("<f8"))
        big_endian = targets(first.weights.astype(">f4"))
        self.assertNotEqual(first.digest, second.digest)
        self.assertEqual(first.digest, big_endian.digest)
        self.assertEqual(big_endian.weights.dtype.str, "<f4")
        self.assertNotEqual(first.digest, replace(first, symbols=("A", "B")).digest)
        # Independently construct the specified length-prefixed header + row bytes.
        header = dict(dates=first.dates, symbols=first.symbols,
                      candidate_result_digest=first.candidate_result_digest,
                      attempt_id=first.attempt_id, dataset_manifest_sha256=first.dataset_manifest_sha256,
                      price_identity=first.price_identity, digest_format="portfolio_targets_binary_v1",
                      dtype="<f4", shape=(2, 2), order="C")
        encoded = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        expected = hashlib.sha256(len(encoded).to_bytes(8, "big") + encoded)
        expected.update(memoryview(first.weights).cast("B"))
        self.assertEqual(first.digest, expected.hexdigest())

    def test_dtype_rounding_tolerance_never_renormalizes(self):
        for dtype in (np.float32, np.float64):
            source = targets(np.array([[.25, -.125], [0., 0.]], dtype=dtype))
            result = allocate_fixed(policy([source], [1]), [source])
            np.testing.assert_array_equal(result.weights, source.weights)
            self.assertEqual(np.abs(result.weights[0]).sum(), .375)
        # float32 epsilon is NOT an economic gross allowance. Accumulate in f64.
        with self.assertRaisesRegex(ValueError, "gross exceeds"):
            targets(np.array([[.4, .6], [0., 0.]], dtype=np.float32))
        tiny = targets(np.array([[.5, .5 + 5e-13], [0., 0.]], dtype=np.float64))
        np.testing.assert_array_equal(allocate_fixed(policy([tiny], [1]), [tiny]).weights, tiny.weights)
        with self.assertRaisesRegex(ValueError, "gross exceeds"):
            targets(np.array([[.5, .5 + 2e-12], [0., 0.]], dtype=np.float64))

    def test_array_dtype_shape_and_nonfinite_rejections(self):
        for values in (np.ones((2, 2), dtype=int), np.zeros((2, 2), dtype=bool),
                       np.zeros((2, 2), dtype=np.float16), np.zeros((2, 2), dtype=object),
                       np.zeros((2, 2, 1)), np.full((2, 2), np.nan, dtype=np.float32)):
            with self.subTest(dtype=values.dtype, shape=values.shape), self.assertRaises(ValueError):
                targets(values)

    def test_moderate_matrix_single_output_bounded_scratch_and_one_hash_per_leg(self):
        # Structural resource check, deliberately no timing/RSS threshold or benchmark.
        shape = (257, 513)  # crosses four complete scratch chunks and a final row
        dates = tuple(np.datetime_as_string(x, unit="D") for x in
                      np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + shape[0]))
        symbols = tuple(f"S{i}" for i in range(shape[1]))
        left = targets(np.full(shape, 1 / 2048, dtype=np.float32), dates=dates, symbols=symbols)
        right = targets(np.full(shape, -1 / 4096, dtype=np.float64), dates=dates, symbols=symbols,
                        attempt_id="second")
        pinned = policy([left, right], [.5, .25])
        with mock.patch.object(portfolio, "_matrix_digest", wraps=portfolio._matrix_digest) as hashes, \
                mock.patch.object(np, "empty", wraps=np.empty) as allocations, \
                mock.patch.object(portfolio, "_owned_weights", side_effect=AssertionError("full recopy")):
            result = allocate_fixed(pinned, [left, right])
            metadata = result.to_metadata()
            self.assertEqual(hashes.call_count, 2)
            self.assertEqual([call.args[0] for call in allocations.call_args_list],
                             [shape, (64, shape[1])])
        self.assertEqual(result.weights.nbytes, shape[0] * shape[1] * 8)
        self.assertTrue(result.weights.flags.owndata)
        self.assertIs(result.inputs[0], left)
        self.assertIs(result.inputs[1], right)
        self.assertTrue(np.all(result.weights == 3 / 16384))
        self.assertLess(len(json.dumps(metadata)), 5000)

    def test_no_nested_matrix_serialization_in_implementation(self):
        tree = ast.parse(inspect.getsource(portfolio))
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                    node.func.id if isinstance(node.func, ast.Name) else "")
                if name in {"asdict", "tolist"}:
                    forbidden.append((name, node.lineno))
        self.assertEqual(forbidden, [])


class RiskTests(unittest.TestCase):
    def evaluate(self, **changes):
        options = dict(phase="pre_order", holdings={"A": 2}, trades={"A": 3},
                       target_weights={"A": .5}, equity=100, reference_prices={"A": 10},
                       limits=RiskLimits(gross=1, net=1, name=1, turnover=1),
                       free_cash=80, projected_free_cash=50)
        options.update(changes)
        return evaluate_risk(**options)

    def test_preorder_hand_known_full_turnover_and_no_input_mutation(self):
        held, orders = {"A": 2}, {"A": 3}
        result = self.evaluate(holdings=held, trades=orders)
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.assessed_holdings, {"A": 5})
        self.assertEqual(result.metrics["assessed"]["gross"], .5)
        self.assertEqual(result.metrics["turnover"], .3)
        self.assertEqual(result.metrics["holdings_notionals_usd"], {"A": 20})
        self.assertEqual(result.version, "portfolio_risk_v1")
        self.assertEqual(result.limits.turnover, 1)
        self.assertEqual(result.residual_weights, {"A": 0})
        self.assertEqual((held, orders), ({"A": 2}, {"A": 3}))

    def test_shorts_and_reversal_use_absolute_traded_notional(self):
        result = self.evaluate(holdings={"A": -2}, trades={"A": 5}, target_weights={"A": .3})
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.assessed_holdings, {"A": 3})
        self.assertEqual(result.metrics["turnover"], .5)
        short = self.evaluate(holdings={"A": -5}, trades={}, target_weights={"A": -.5})
        self.assertEqual(short.metrics["assessed"]["net"], -.5)
        self.assertEqual(short.metrics["assessed"]["gross"], .5)

    def test_target_and_executable_concentration_both_checked(self):
        result = self.evaluate(limits=RiskLimits(name=.4))
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.violations, ("assessed:name", "target:name"))

    def test_joint_gross_net_name_group_turnover_liquidity_and_cash(self):
        result = self.evaluate(limits=RiskLimits(gross=.4, net=.4, name=.4,
                                               group_gross=.4, group_net=.4,
                                               turnover=.2, liquidity=.1),
                               groups={"A": "sector"}, liquidity={"A": 10}, projected_free_cash=-1)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(len(result.violations), 13)
        self.assertIn("trades:turnover", result.violations)
        self.assertIn("trades:liquidity:A", result.violations)
        self.assertIn("funding:projected_cash_loan", result.violations)
        self.assertEqual(result.assessed_holdings, {"A": 5})

    def test_cash_and_turnover_infeasibility_never_rescaled(self):
        result = self.evaluate(projected_free_cash=-5, limits=RiskLimits(turnover=.2))
        self.assertEqual(result.violations, ("funding:projected_cash_loan", "trades:turnover"))
        self.assertEqual(result.metrics["turnover"], .3)
        self.assertEqual(result.residual_weights["A"], 0)

    def test_unknown_group_and_missing_liquidity_fail_closed_jointly(self):
        result = self.evaluate(limits=RiskLimits(name=.4, group_gross=.7, liquidity=.1),
                               groups={"A": None}, liquidity={})
        self.assertEqual(result.status, "missing_inputs")
        self.assertEqual(result.missing_inputs, ("groups:A", "liquidity:A"))
        self.assertIn("assessed:name", result.violations)
        self.assertEqual(result, self.evaluate(limits=RiskLimits(name=.4, group_gross=.7, liquidity=.1),
                                               groups={"A": None}, liquidity={}))

    def test_group_gross_counts_both_sides_net_cancels(self):
        result = self.evaluate(holdings={"A": 5, "B": -5}, trades={},
                               target_weights={"A": .5, "B": -.5},
                               reference_prices={"A": 10, "B": 10}, groups={"A": "G", "B": "G"},
                               limits=RiskLimits(group_gross=.75, group_net=0))
        self.assertEqual(result.violations, ("assessed:group_gross:G", "target:group_gross:G"))
        self.assertEqual(result.metrics["assessed"]["groups"]["G"], {"gross": 1, "net": 0})

    def test_unknown_group_sentinel_never_becomes_a_bucket(self):
        for group in ("UNKNOWN", "__unknown__", "", None, -1):
            with self.subTest(group=group):
                result = self.evaluate(groups={"A": group}, limits=RiskLimits(group_gross=1))
                self.assertEqual(result.missing_inputs, ("groups:A",))
                self.assertEqual(result.status, "missing_inputs")

    def test_missing_groups_for_target_only_and_residual_only(self):
        for held, target in (({}, {"A": .5}), ({"A": 5}, {})):
            result = self.evaluate(holdings=held, trades={}, target_weights=target,
                                   limits=RiskLimits(group_net=0))
            self.assertEqual(result.status, "missing_inputs")
            self.assertIn("groups:A", result.missing_inputs)

    def test_zero_liquidity_is_violation_nonfinite_is_missing(self):
        result = self.evaluate(liquidity={"A": 0}, limits=RiskLimits(liquidity=.1))
        self.assertEqual(result.violations, ("trades:liquidity:A",))
        for volume in (None, math.nan, math.inf, -1):
            with self.subTest(volume=volume):
                result = self.evaluate(liquidity={"A": volume}, limits=RiskLimits(liquidity=.1))
                self.assertEqual(result.status, "missing_inputs")

    def test_disabled_inputs_not_required_zero_limits_enabled(self):
        self.assertEqual(self.evaluate().status, "pass")
        self.assertEqual(self.evaluate(limits=RiskLimits(turnover=0)).status, "blocked")
        result = self.evaluate(holdings={}, trades={}, target_weights={}, reference_prices={},
                               limits=RiskLimits(group_gross=0, liquidity=0), free_cash=100,
                               projected_free_cash=None)
        self.assertEqual(result.status, "pass")

    def test_postfill_does_not_add_fills_again(self):
        result = self.evaluate(phase="post_fill", holdings={"A": 5}, trades={"A": 3}, free_cash=50)
        self.assertEqual(result.assessed_holdings, {"A": 5})
        self.assertEqual(result.metrics["turnover"], .3)
        self.assertEqual(result.status, "pass")

    def test_blocked_fill_zero_target_preserves_actual_held_exposure(self):
        result = self.evaluate(phase="post_fill", holdings={"A": -5}, trades={},
                               target_weights={"A": 0}, limits=RiskLimits(gross=.4))
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.assessed_holdings, {"A": -5})
        self.assertEqual(result.metrics["assessed"]["gross"], .5)
        self.assertEqual(result.metrics["target"]["gross"], 0)
        self.assertEqual(result.residual_weights, {"A": .5})

    def test_nonpositive_equity_and_passive_unfunded_state_retained(self):
        for equity in (0, -100):
            result = self.evaluate(phase="post_fill", equity=equity, holdings={"A": -5},
                                   trades={}, free_cash=-2)
            self.assertEqual(result.status, "blocked")
            self.assertEqual(result.violations, ("equity:nonpositive", "funding:negative_free_cash"))
            self.assertEqual(result.assessed_holdings, {"A": -5})
            self.assertEqual(result.metrics["unfunded_cash_shortfall_usd"], 2)
            self.assertNotIn("assessed", result.metrics)

    def test_insolvent_state_does_not_waive_missing_enabled_inputs(self):
        result = self.evaluate(equity=-1, limits=RiskLimits(group_gross=1, liquidity=.1))
        self.assertEqual(result.status, "missing_inputs")
        self.assertEqual(result.missing_inputs, ("groups:A", "liquidity:A"))
        self.assertIn("equity:nonpositive", result.violations)

    def test_missing_cash_projection_does_not_assume_short_proceeds_spendable(self):
        result = self.evaluate(holdings={}, trades={"A": -3}, target_weights={"A": -.3},
                               projected_free_cash=None)
        self.assertEqual(result.status, "missing_inputs")
        self.assertEqual(result.missing_inputs, ("projected_free_cash",))
        self.assertEqual(self.evaluate(free_cash=None).status, "missing_inputs")

    def test_missing_nonfinite_prices_holdings_targets_equity(self):
        for changes in (dict(reference_prices={}), dict(reference_prices={"A": math.nan}),
                        dict(reference_prices={"A": 0}), dict(reference_prices={"A": -10}),
                        dict(holdings={"A": math.inf}), dict(trades={"A": math.nan}),
                        dict(target_weights={"A": None}), dict(equity=math.inf), dict(holdings=None)):
            with self.subTest(changes=changes):
                self.assertEqual(self.evaluate(**changes).status, "missing_inputs")

    def test_invalid_limits_and_phase_rejected(self):
        for value in (-1, math.inf, math.nan, True):
            with self.assertRaises(ValueError):
                RiskLimits(gross=value)
        with self.assertRaises(ValueError):
            self.evaluate(phase="close")

    def test_overflow_fails_closed_instead_of_passing_or_crashing(self):
        for changes in (
                dict(holdings={"A": 1e308}),
                dict(holdings={"A": 1e307, "B": 1e307}, trades={},
                     reference_prices={"A": 10, "B": 10}, equity=1),
                dict(equity=5e-324),
                dict(equity=10 ** 1000)):
            with self.subTest(changes=list(changes)):
                result = self.evaluate(**changes)
                self.assertEqual(result.status, "missing_inputs")
                json.dumps(asdict(result), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
