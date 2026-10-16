"""Verify a built SEC catalog against independent source counts and filing values.

Writes a local verification receipt. Does not alter raw data, catalogs, prices,
preferences or run trading simulations. See docs/SEC_FIELD_ACCESS.md for the
independently read filing values and the availability convention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def verify(catalog: Path, audit: Path, cache: Path) -> dict:
    import numpy as np
    from qlib_yahoo.sec_catalog import read_catalog, search_tags, query_rows
    from qlib_yahoo.sec_features import materialize, load_arrays

    started = time.monotonic()
    manifest = read_catalog(catalog, verify=True)
    independent = json.loads(audit.read_text(encoding="utf-8"))
    if "csv.reader logical" not in independent["method"]:
        raise ValueError("Require an independent logical-record audit, not preliminary line counts")
    expected = {item["name"]: item for item in independent["archives"]}
    observed = {Path(item["path"]).name: item for item in manifest["sources"]}
    assert set(expected) == set(observed), "Source archive coverage differs"
    for name, source in expected.items():
        assert source["sha256"] == observed[name]["sha256"], name
        for table, record in source["tables"].items():
            assert observed[name]["table_counts"][table] == record["rows"], (name, table)
            assert set(record["columns"]) <= set(manifest["tables"][table]["columns"]), (name, table)
    counts = {name: record["rows"] for name, record in manifest["tables"].items()}
    assert counts == independent["rows"], "Aggregate source counts differ"
    print("Every source archive/hash/table/count matched independent logical-record audit", flush=True)

    bindings = {alias: {"tag": tag, "taxonomy": "us-gaap/", "unit": "USD", "qtrs": duration,
                         "segments": "", "coreg": ""}
                for alias, tag, duration in (("sec_income_q", "NetIncomeLoss", 1),
                                             ("sec_income_ytd", "NetIncomeLoss", 2),
                                             ("sec_inventory", "InventoryNet", 0))}
    timings = {}
    for query in ("NetIncomeLoss", "InventoryNet", "operating cash"):
        tick = time.monotonic()
        found = search_tags(catalog, query, limit=10)
        timings[query] = {"seconds": time.monotonic() - tick, "matches": found["total"]}
        if query != "operating cash":
            assert found["items"], query
    rows = query_rows(catalog, "num", {"adsh": "0000320193-25-000008", "tag": "NetIncomeLoss",
                                       "ddate": "20241231", "qtrs": "1", "uom": "USD",
                                       "segments": "", "coreg": ""})
    assert {row["value"] for row in rows} == {"36330000000.0000"}, "Raw decimal/provenance mismatch"
    dates = np.array(["2025-01-30", "2025-01-31", "2025-02-03"], dtype="datetime64[D]")
    tick = time.monotonic()
    handle = materialize(catalog, bindings, dates, ["AAPL.US", "MSFT.US"], cache, progress=print)
    arrays = load_arrays(handle)
    try:
        assert arrays["sec_income_q"][1, 1] == np.float32(24108000000)
        assert arrays["sec_income_q"][2, 0] == np.float32(36330000000)
        assert arrays["sec_income_ytd"][1, 1] == np.float32(48775000000)
        assert arrays["sec_inventory"][2, 0] == np.float32(6911000000)
        assert arrays["sec_income_q"][0, 1] != np.float32(24108000000)
        assert arrays["sec_income_q"][1, 0] != np.float32(36330000000)
        panel = {key: [[float(v) if np.isfinite(v) else None for v in row] for row in value]
                 for key, value in arrays.items()}
    finally:
        for value in arrays.values():
            value._mmap.close()
    return {"status": "passed", "catalog_sha256": manifest["catalog_sha256"],
            "archive_count": len(expected), "row_counts": counts, "search": timings,
            "known_answer_dates": list(map(str, dates)), "known_answer_symbols": ["AAPL.US", "MSFT.US"],
            "known_answer_panel": panel, "panel_seconds": time.monotonic() - tick,
            "diagnostics": handle["diagnostics"], "elapsed_seconds": time.monotonic() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new receipt path")
    result = verify(args.catalog, args.audit, args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
