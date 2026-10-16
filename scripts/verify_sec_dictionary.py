"""Opt-in real-catalog acceptance for formula-first SEC names.

No downloads or source writes. --run adds two public-example Lab results and
one short native Qlib run under the supplied data root. Other outputs go into
the explicitly new receipt directory; never run this in ordinary CI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--start", default="2025-02-03")
    parser.add_argument("--end", default="2025-03-31")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from qlib_yahoo.sec_registry import dictionary_status, search_dictionary
    from qlib_yahoo.sec_resolution import resolve_config
    from qlib_yahoo.sec_features import materialize, load_arrays, close_arrays
    from qlib_yahoo.webapp.service import ConsoleService, ServiceError
    from verify_sec_runs import wait_job
    import numpy as np

    receipt = {"status": "failed", "run_requested": args.run, "catalog": str(args.catalog),
               "dataset": str(args.dataset), "start": args.start, "end": args.end}
    tick = time.monotonic()
    try:
        config = {"catalog": str(args.catalog), "mode": "dictionary"}
        service = ConsoleService(args.data_root, settings_path=args.output / "settings.json")
        query_start = time.monotonic()
        page = search_dictionary(args.catalog, query="inventory", limit=10)
        receipt["search_seconds"] = time.monotonic() - query_start
        assert page["items"][0]["name"] == "sec_inventory", "Common inventory definition not ranked first"
        receipt["dictionary"] = dictionary_status(args.catalog)
        expressions = "rank(sec_inventory / sec_assets)\nrank(sec_liabilities / sec_assets)"
        query_start = time.monotonic()
        checked = service.alpha_check(expressions, str(args.dataset), sec=config)
        receipt["check_seconds"] = time.monotonic() - query_start
        assert checked["ok"], checked
        assert {row["name"] for row in checked["sec_inputs"]} == {"sec_assets", "sec_inventory", "sec_liabilities"}
        assert service.check_alpha("$sec_inventory / $sec_assets", sec=config)["ok"]
        for expression in ("rank(sec_inventroy / sec_assets)", "nonsense(sec_inventory)", "sec_assets +",
                           "ts_mean(sec_assets)", "rank(sec_assets, bogus=2)"):
            with patch.object(service, "_start", side_effect=AssertionError("Invalid formula launched a worker")):
                try:
                    service.start_alpha({"expressions": expression, "dataset": str(args.dataset), "sec": config})
                except ServiceError:
                    pass
                else:
                    raise AssertionError("Invalid formula accepted")
        receipt["invalid_formulas_rejected_before_worker"] = True

        resolved = resolve_config(["sec_inventory / sec_assets"], config, verify=True)
        dates = np.array(["2025-01-30", "2025-01-31", "2025-02-03"], dtype="datetime64[D]")
        first = materialize(args.catalog, resolved["bindings"], dates, ["AAPL.US", "MSFT.US"], args.output / "panels")
        arrays = load_arrays(first)
        try:
            assert arrays["sec_inventory"][2, 0] == np.float32(6911000000), "Known AAPL inventory differs"
            expected = arrays["sec_assets"].copy()
        finally:
            close_arrays(arrays)
        # Different alias and field combination, identical underlying recipe.
        with patch("qlib_yahoo.sec_features._build_field", side_effect=AssertionError("Repeated field construction")):
            second = materialize(args.catalog, {"sec_saved_assets": resolved["bindings"]["sec_assets"]},
                                 dates, ["AAPL.US", "MSFT.US"], args.output / "panels")
        arrays = load_arrays(second)
        try:
            np.testing.assert_array_equal(arrays["sec_saved_assets"], expected)
        finally:
            close_arrays(arrays)
        receipt["known_value_and_per_field_reuse"] = True

        if args.run:
            params = {"dataset": str(args.dataset), "sec": config, "expressions": expressions,
                      "start": args.start, "end": args.end, "universe": "TOP3000",
                      "cost_bps": 5, "neutralization": "MARKET", "workers": 2}
            cold = wait_job(service, service.start_alpha(params))
            batch = json.loads(Path(cold["result"]).read_text(encoding="utf-8"))
            assert len(batch["ids"]) == 2 and not batch["errors"], batch
            from qlib_yahoo.alphalab.batch import Library
            library = Library(args.data_root / "alphalab" / "library.sqlite")
            results = [library.get(identifier) for identifier in batch["ids"]]
            assert all(r["sec"]["registry"] == resolved["registry"] for r in results)
            warm = wait_job(service, service.start_alpha(params))
            native = wait_job(service, service.start_backtest({
                "dataset": str(args.dataset), "sec": config, "alpha": "$sec_inventory / $sec_assets",
                "start": args.start, "end": args.end, "universe": "top3000", "strategy_mode": "signed",
                "signal_transform": "zscore", "borrow_rate": 0.004, "cost_bps": 5}))
            manifest = json.loads((Path(native["result"]) / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["status"] == "completed"
            assert manifest["sec"]["registry"] == resolved["registry"]
            receipt.update(lab=cold, lab_cached=warm, native=native, lab_ids=batch["ids"])
        receipt["status"] = "passed"
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - tick
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "receipt": str(args.output / "receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
