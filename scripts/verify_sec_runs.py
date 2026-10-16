"""Exercise SEC inputs through the actual console service and child workers.

Explicit local acceptance only: adds two public-example Alpha Lab results and
one signed Qlib run under the given data root. Never edits preferences or sources.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def wait_job(service, job):
    cursor = 0
    while True:
        state = service.job(job["id"], cursor)
        for line in state["log"]:
            print(line, flush=True)
        cursor = state["log_index"]
        if state["state"] != "running":
            if state["state"] != "completed":
                raise RuntimeError(state["error"] or state["state"])
            return state
        time.sleep(0.25)


def main():
    from qlib_yahoo.webapp.service import ConsoleService

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new receipt path")
    service = ConsoleService(args.data_root)
    bindings = {alias: {"tag": tag, "taxonomy": "us-gaap/", "unit": "USD", "qtrs": qtrs,
                        "segments": "", "coreg": ""}
                for alias, tag, qtrs in (("sec_income_q", "NetIncomeLoss", 1),
                                         ("sec_equity", "StockholdersEquity", 0),
                                         ("sec_inventory", "InventoryNet", 0),
                                         ("sec_assets", "Assets", 0))}
    sec = {"catalog": str(args.catalog), "bindings": bindings}
    common = {"dataset": str(args.dataset), "start": args.start, "end": args.end,
              "cost_bps": 5, "sec": sec}
    lab_params = {**common, "expressions": "rank(sec_income_q / sec_equity)\nrank(sec_inventory / sec_assets)",
                  "universe": "TOP3000", "delay": 1, "neutralization": "MARKET", "workers": 2}
    cold = wait_job(service, service.start_alpha(lab_params))
    batch = json.loads(Path(cold["result"]).read_text(encoding="utf-8"))
    assert len(batch["ids"]) == 2 and not batch["errors"], "Lab alpha failed"
    from qlib_yahoo.alphalab.batch import Library
    library = Library(args.data_root / "alphalab" / "library.sqlite")
    results = [library.get(identifier) for identifier in batch["ids"]]
    assert all(r and r.get("sec") for r in results), "Missing SEC provenance"
    warm = wait_job(service, service.start_alpha(lab_params))
    native = wait_job(service, service.start_backtest({**common, "alpha": "$sec_income_q / $sec_equity",
                                                      "universe": "top3000", "strategy_mode": "signed",
                                                      "signal_transform": "zscore", "borrow_rate": 0.004}))
    run = Path(native["result"])
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("sec"), "Native run lacks SEC identity"
    assert manifest.get("status") == "completed", "Native run incomplete"
    result = {"status": "passed", "catalog": str(args.catalog), "dataset": str(args.dataset),
              "start": args.start, "end": args.end,
              "lab_cold": cold, "lab_cached": warm, "native": native,
              "lab_ids": [r["id"] for r in results]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({"status": "passed", "receipt": str(args.output),
                      "seconds": {"lab_cold": cold["elapsed"], "lab_cached": warm["elapsed"],
                                  "native": native["elapsed"]}}, indent=2))


if __name__ == "__main__":
    main()
