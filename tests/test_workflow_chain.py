"""Actual saved-Lab -> signed validation -> one-account S3 portfolio chain."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_qlib_target_handoff import installed


def make_snapshot(root):
    import pandas as pd
    from test_qlib_yahoo_execution import _snapshot
    from qlib_yahoo.dataset import sha256, tree_sha256

    source = _snapshot(root, sessions_count=65, include_benchmark=False)
    dataset = root / "datasets" / "synthetic"
    dataset.parent.mkdir()
    source.rename(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text())
    days = pd.bdate_range("2024-01-02", periods=65)
    membership = dataset / "membership.csv"
    membership.write_text("instrument,rank,effective_start,effective_end\n" + "".join(
        f"{symbol},{rank},{days[0]:%Y-%m-%d},{days[-1]:%Y-%m-%d}\n"
        for rank, symbol in enumerate(manifest["symbols"], 1)))
    native_membership = dataset / "qlib_data" / "instruments" / "top6.txt"
    native_membership.write_text("".join(
        f"{symbol}\t{days[0]:%Y-%m-%d}\t{days[-1]:%Y-%m-%d}\n" for symbol in manifest["symbols"]))
    tree, count = tree_sha256(dataset / "qlib_data")
    manifest.update(status="ready", provider="synthetic hand-known workflow fixture",
                    benchmark=None, fields=["open", "high", "low", "close", "volume", "factor"],
                    qlib_binary_tree_sha256=tree, qlib_binary_file_count=count,
                    universes={"top6": {"size": 6, "min_history_sessions": 1, "membership_csv": "membership.csv",
                                       "instruments_file": "instruments/top6.txt",
                                       "instruments_sha256": sha256(native_membership),
                                       "membership_sha256": sha256(membership)}})
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    return dataset, days


def run_chain():
    import numpy as np
    import pandas as pd
    from qlib_yahoo.analysis import read_instrument_table, table_artifact, validate_bundle
    from qlib_yahoo.dataset import sha256
    from qlib_yahoo.alphalab.batch import run_batch
    from qlib_yahoo.alphalab.settings import Settings
    from qlib_yahoo.validation_workflow import validate_candidate
    from qlib_yahoo.workflow_records import EvidenceStore
    from qlib_yahoo.target_bundle import load_bundle
    from qlib_yahoo.portfolio_workflow import preview_portfolio, run_portfolio
    from qlib_yahoo.portfolio import RiskLimits
    from dataclasses import asdict

    # Workspace-local temporary files also work in Windows isolated subprocesses
    # whose inherited system-temp ACL belongs to a different sandbox identity.
    with tempfile.TemporaryDirectory(prefix="workflow-chain-test-", dir=ROOT) as folder:
        root = Path(folder).resolve()
        dataset, days = make_snapshot(root)
        settings = Settings(universe="TOP6", start=f"{days[10]:%Y-%m-%d}",
                            end=f"{days[45]:%Y-%m-%d}", warmup=8, truncation=.2, cost_bps=5)
        results = run_batch(dataset, ["rank(close)", "-rank(ts_delta(close, 2))"],
                            settings, data_root=root, workers=1)
        assert all("error" not in item for item in results), results
        store = EvidenceStore(root)
        validations = []
        input_targets = []
        for result in results:
            run = validate_candidate(root, {"attempt_id": result["attempt_id"]})
            manifest = json.loads((run / "manifest.json").read_text())
            assert manifest["input_mode"] == "target_weights"
            target = manifest["target_bundle"]["metadata"]
            assert target["attempt_ids"] == [result["attempt_id"]]
            assert target["candidate_result_digests"] == [result["result_digest"]]
            direct = store.get(manifest["workflow"]["attempt_id"])
            assert direct["status"] == "completed", direct
            parent = store.get(direct["parents"][0])
            assert parent["kind"] == "candidate-validation" and parent["status"] == "completed", parent
            validations.append(parent["id"])
            checked = store.get_result(parent["result_digest"])
            assert checked["reconciliation"]["status"] == "passed"
            assert checked["reproduction"]["status"] == "matched"
            targets = load_bundle(checked["target_bundle"])
            assert targets["dates"][0] == f"{days[9]:%Y-%m-%d}"
            assert targets["dates"][-1] == f"{days[44]:%Y-%m-%d}"
            input_targets.append(targets)
        parameters = {
            "validation_ids": validations, "coefficients": [.4, .6], "account_id": "connected-fixture",
            "limits": asdict(RiskLimits(gross=1.0, net=.2, name=.25, turnover=2, liquidity=.1)),
            "initial_cash": 100_000, "cost_per_side": .0005, "annual_borrow_rate": .004,
        }
        preview = preview_portfolio(root, parameters)
        assert preview["policy"]["initial_cash"] == 100_000
        assert preview["policy"]["coefficients"] == [.4, .6]
        assert [leg["candidate_attempt_id"] for leg in preview["legs"]] == [r["attempt_id"] for r in results]

        # Independent arithmetic on the saved per-leg matrices: do not use the
        # allocation implementation as its own expected-value oracle.
        left, right = input_targets
        assert left["dates"] == right["dates"] and left["symbols"] == right["symbols"]
        expected64 = .4 * left["weights"].astype(np.float64) + .6 * right["weights"].astype(np.float64)
        expected = expected64.astype(np.float32)
        assert (np.abs(expected64).sum(axis=1) < 1).all(), "fixture needs no gross-cap rounding"

        def check_portfolio(run, requested, *, allow_blocks=False):
            checked = validate_bundle(run)
            checked.assert_unchanged()
            manifest, summary = checked.manifest, checked.summary
            assert manifest["status"] == "completed" and manifest["stage"] == "sealed"
            assert manifest["backtest"]["cash"] == 100_000  # never two independent full-capital accounts
            assert summary["backtest"]["cash"] == 100_000
            contract = manifest["backtest"]["execution_contract"]
            assert contract["cash_borrowing"] is False and contract["restricted_short_proceeds"] is True
            assert contract["annual_borrow_rate"] == .004
            assert contract["open_cost"] == contract["close_cost"] == .0005

            direct = store.get(manifest["workflow"]["attempt_id"])
            assert direct["status"] == "completed", direct
            parents = [store.get(identity) for identity in direct["parents"]]
            portfolio_attempts = [item for item in parents if item["kind"] == "portfolio-validation"]
            assert len(portfolio_attempts) == 1, parents
            attempt = portfolio_attempts[0]
            assert attempt["status"] == "completed" and set(validations).issubset(attempt["parents"])
            receipt = store.get_result(attempt["result_digest"])
            assert receipt["run_path"] == str(run)
            assert receipt["manifest_sha256"] == sha256(run / "manifest.json")
            assert receipt["reconciliation"] == {"status": "passed", "checker": "analysis.validate_bundle",
                                                 "scope": "one shared signed cash/share account"}
            assert receipt["portfolio"]["request"] == requested
            allocation = receipt["allocation"]
            assert allocation["portfolio_attempt_id"] == attempt["id"]
            assert allocation["validation_ids"] == validations
            assert allocation["validation_result_digests"] == [store.get(x)["result_digest"] for x in validations]
            assert allocation["attempt_ids"] == [item["attempt_id"] for item in results]
            assert allocation["candidate_result_digests"] == [item["result_digest"] for item in results]
            policy = allocation["allocation_policy"]
            assert policy["account_id"] == requested["account_id"] and policy["capital"] == 100_000
            assert policy["version"] == "portfolio_fixed_v1"
            assert policy["policy_id"] == "portfolio-" + attempt["id"]
            assert [leg["coefficient"] for leg in policy["legs"]] == [.4, .6]
            assert [leg["target_digest"] for leg in policy["legs"]] == allocation["input_target_digests"]
            combined = load_bundle(receipt["target_bundle"])
            assert combined["dates"] == left["dates"] and combined["symbols"] == left["symbols"]
            assert combined["metadata"] == allocation
            np.testing.assert_array_equal(combined["weights"], expected)
            assert combined["identity"]["precision"]["gross_cap_ulp_reductions"] == 0
            assert manifest["target_bundle"]["identity"]["sha256"] == receipt["target_bundle"]["sha256"]
            assert manifest["target_bundle"]["metadata"] == allocation

            target_table = read_instrument_table(table_artifact(run, "signed_weights"))
            index = pd.MultiIndex.from_product([pd.to_datetime(combined["dates"]), combined["symbols"]],
                                               names=["datetime", "instrument"])
            saved_weights = target_table.set_index(["datetime", "instrument"])["weight"].reindex(index)
            assert saved_weights.notna().all()
            np.testing.assert_array_equal(saved_weights.to_numpy().reshape(expected.shape), expected)
            decisions = json.loads(gzip.decompress((run / "signed_decisions.json.gz").read_bytes()))
            assert len(decisions) == len(expected)
            for row, decision in zip(expected, decisions):
                np.testing.assert_array_equal([decision["target_weights"][s] for s in combined["symbols"]], row)

            risk = manifest["portfolio_risk"]
            assert risk == summary["portfolio_risk"] == receipt["summary"]["portfolio_risk"]
            qualification = receipt["qualification"]
            assert qualification["large_source_and_target_artifacts"] == "verified"
            assert qualification["shared_ledger_reconciliation"] == "passed"
            assert qualification["source_drift"] == "no_change_observed_during_request"
            assert qualification["risk_outcome"] == ("exceptions_recorded" if allow_blocks else "checks_passed")
            assert qualification["risk_counts"] == summary["portfolio_risk"]["counts"]
            assert receipt["portfolio"]["qualification"]["large_source_and_target_artifacts"] == "pending explicit Run verification"
            assert risk["policy"]["limits"] == requested["limits"]
            assert risk["policy"]["schema_version"] == 1
            assert risk["policy"]["provenance"] == {
                "portfolio_attempt_id": attempt["id"], "policy_id": policy["policy_id"],
                "validation_ids": validations,
                "allocation_sources_sha256": allocation["allocation_sources_sha256"]}
            for name, digest in allocation["allocation_sources_sha256"].items():
                assert sha256(ROOT / "qlib_yahoo" / name) == digest
            risk_path = run / risk["artifact"]
            assert manifest["artifacts_sha256"][risk_path.name] == sha256(risk_path)
            events = [json.loads(line) for line in gzip.decompress(risk_path.read_bytes()).decode().splitlines()]
            pre = [item for item in events if item["phase"] == "pre_order"]
            post = [item for item in events if item["phase"] == "post_fill"]
            assert len(events) == 2 * len(expected) and len(pre) == len(post) == len(expected)
            assert [item["session"] for item in pre] == [str(day.date()) for day in days[10:46]]
            assert [item["session"] for item in post] == [item["session"] for item in pre]
            assert [item["decision_session"] for item in pre] == combined["dates"]
            assert [item["decision_session"] for item in post] == combined["dates"]
            assert risk["counts"]["sessions"] == len(expected)
            for phase, rows in (("pre_order", pre), ("post_fill", post)):
                assert risk["counts"][phase] == {
                    status: sum(item["status"] == status for item in rows)
                    for status in ("pass", "blocked", "missing_inputs")}
                assert all(not item["missing_inputs"] for item in rows), rows
                if not allow_blocks:
                    assert all(item["status"] == "pass" for item in rows), rows
            assert risk["counts"]["blocked_rebalances"] == sum(item["status"] != "pass" for item in pre)

            daily = checked.frame
            positions = read_instrument_table(table_artifact(run, "positions"))
            for number, event in enumerate(post):
                held = positions.loc[positions.datetime == pd.Timestamp(event["session"])]
                amounts = {row.instrument: float(row.amount) for row in held.itertuples() if row.instrument}
                assert amounts == event["actual_holdings"]
                assert all(quantity == int(quantity) for quantity in amounts.values())
                np.testing.assert_allclose(event["metrics"]["free_cash_usd"], daily.free_cash.iloc[number], atol=1e-9)
            assert (daily.free_cash >= 0).all() and (daily.restricted_cash >= 0).all()
            assert daily.restricted_cash.max() > 0
            np.testing.assert_allclose(daily.account, daily.cash_total + daily.long_value - daily.short_liability)
            np.testing.assert_allclose(daily.cash_total, daily.free_cash + daily.restricted_cash)
            np.testing.assert_allclose(daily.cost_dollars,
                                       daily.transaction_cost_dollars + daily.borrow_cost_dollars)
            assert daily.transaction_cost_dollars.sum() > 0 and daily.borrow_cost_dollars.sum() > 0
            # ACT/365 includes calendar gaps; use actual prior CLOSE liability.
            gaps = daily.index.to_series().diff().dt.days.fillna(0).to_numpy()
            expected_borrow = daily.short_liability.shift(1).fillna(0).to_numpy() * .004 * gaps / 365
            np.testing.assert_allclose(daily.borrow_cost_dollars, expected_borrow, rtol=1e-12, atol=1e-9)
            from qlib.data import D
            opens = D.features(combined["symbols"], ["$open"], start_time=daily.index[0],
                               end_time=daily.index[-1], freq="day", disk_cache=0)["$open"]
            expected_cost = [sum(quantity * float(opens.loc[(symbol, pd.Timestamp(event["session"]))])
                                 for symbol, quantity in event["executed_gross_shares"].items()) * .0005
                             for event in post]
            np.testing.assert_allclose(daily.transaction_cost_dollars, expected_cost, rtol=1e-12, atol=1e-9)
            np.testing.assert_allclose(post[0]["metrics"]["equity_usd"],
                                       100_000 - daily.transaction_cost_dollars.iloc[0], atol=1e-9)
            checked.assert_unchanged()
            return pre, post

        run = run_portfolio(root, parameters)
        pre, post = check_portfolio(run, preview["request"])
        assert any(any(q < 0 for q in item["actual_holdings"].values()) for item in post)
        assert any(any(q > 0 for q in item["actual_holdings"].values()) for item in post)

        # One additional actual request: initially separate groups later merge.
        # An explicit dated concentration violation must retain already-held
        # shares, rather than merely demonstrating that an empty account stays empty.
        from qlib_yahoo.classification import create_classification_overlay
        name_peak = float(np.max(np.abs(expected)))
        gross = np.abs(expected.astype(np.float64)).sum(axis=1)
        eligible_changes = np.flatnonzero((np.arange(len(gross)) >= 2) & (gross > name_peak * 1.25))
        assert len(eligible_changes), "fixture needs a genuinely diversified target for the group check"
        change = int(eligible_changes[len(eligible_changes) // 2])
        group_cap = (name_peak + float(gross[change])) / 2
        classifications = []
        for symbol in left["symbols"]:
            classifications.extend([
                {"instrument": symbol, "available_at": "2024-01-01T00:00:00Z",
                 "sector": symbol, "industry": None, "country": "US"},
                {"instrument": symbol, "available_at": left["dates"][change] + "T00:00:00Z",
                 "sector": "concentrated", "industry": None, "country": "US"},
            ])
        overlay = create_classification_overlay(dataset, root / "classifications" / "concentration", classifications)
        blocked_parameters = {**parameters, "account_id": "connected-concentration-block",
                              "limits": {**parameters["limits"], "group_gross": group_cap},
                              "group_overlay": str(overlay), "group_level": "sector"}
        blocked_preview = preview_portfolio(root, blocked_parameters)
        blocked_run = run_portfolio(root, blocked_parameters)
        blocked_pre, blocked_post = check_portfolio(blocked_run, blocked_preview["request"], allow_blocks=True)
        assert blocked_pre[change]["status"] == "blocked", blocked_pre[change]
        assert any("group_gross:concentrated" in reason for reason in blocked_pre[change]["violations"])
        assert blocked_post[change - 1]["actual_holdings"]
        assert blocked_post[change]["actual_holdings"] == blocked_post[change - 1]["actual_holdings"]
        assert blocked_post[change]["executed_trades"] == {}
        assert blocked_post[change]["residual_weights"]
        assert all(item["status"] == "pass" for item in blocked_pre[:change])
        print("WORKFLOW_CHAIN_OK: two saved candidates -> actual validations -> 0.4/0.6 shared account; "
              "sealed pre/post risk, exact targets, reconciled costs/borrow, held-position concentration block")


class WorkflowChainTests(unittest.TestCase):
    @unittest.skipUnless(installed(), "installed signed derivative/dependencies unavailable")
    def test_two_saved_candidates_reach_actual_shared_portfolio_and_risk(self):
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--chain-child"],
            cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=240,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-20000:])
        self.assertIn("WORKFLOW_CHAIN_OK", result.stdout)


if __name__ == "__main__":
    if "--chain-child" in sys.argv:
        run_chain()
    else:
        unittest.main()
