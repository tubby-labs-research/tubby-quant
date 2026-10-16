"""Bounded exact parity checks for saved Qlib runtime optimization trials."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct


def json_values_exact(left, right):
    """Compare decoded JSON types/order and binary64 bits, including signed zero.

    Python equality alone accepts 1 == 1.0 and -0.0 == 0.0. Container/compression
    bytes and decimal spelling may differ; observable decoded values may not.
    """
    pending = [(left, right)]
    while pending:
        first, second = pending.pop()
        if type(first) is not type(second):
            return False
        if type(first) is dict:
            if list(first) != list(second):
                return False
            pending.extend((value, second[key]) for key, value in first.items())
        elif type(first) is list:
            if len(first) != len(second):
                return False
            pending.extend(zip(first, second))
        elif type(first) is float:
            if struct.pack("!d", first) != struct.pack("!d", second):
                return False
        elif type(first) in (str, int, bool, type(None)):
            if first != second:
                return False
        else:
            return False  # Not a decoded JSON value.
    return True


def decision_records(path):
    """Decode one session at a time rather than loading a multi-year JSON array."""
    decoder = json.JSONDecoder()
    with gzip.open(path, "rt", encoding="utf-8") as source:
        buffer, ended = "", False
        while True:
            buffer = buffer.lstrip()
            if buffer[:1] in ("[", ","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                if buffer[1:].strip() or any(chunk.strip() for chunk in iter(lambda: source.read(64 * 1024), "")):
                    raise ValueError("trailing decision data")
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if ended:
                    raise ValueError("incomplete decision history")
                more = source.read(64 * 1024)
                ended = not more
                buffer += more
                if len(buffer) > 64 * 1024**2:
                    raise ValueError("decision record exceeds bounded check limit")
                continue
            if not isinstance(value, dict):
                raise ValueError("invalid decision record")
            yield value
            buffer = buffer[end:]


def compare_tables(left, right):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    first, second = pq.ParquetFile(left), pq.ParquetFile(right)
    assert first.schema_arrow.remove_metadata() == second.schema_arrow.remove_metadata(), (left.name, "schema")
    assert first.metadata.num_rows == second.metadata.num_rows, (left.name, "rows")
    a, b = iter(first.iter_batches(batch_size=16_384)), iter(second.iter_batches(batch_size=16_384))
    x, y = next(a, None), next(b, None)
    while x is not None and y is not None:
        size = min(x.num_rows, y.num_rows)
        # Pandas compares NaNs as missing rather than IEEE NaN != NaN.
        import pandas as pd
        first_slice, second_slice = x.slice(0, size), y.slice(0, size)
        pd.testing.assert_frame_equal(first_slice.to_pandas(), second_slice.to_pandas(), check_exact=True)
        for index, field in enumerate(first.schema_arrow):
            if pa.types.is_floating(field.type):
                a_col, b_col = first_slice.column(index), second_slice.column(index)
                valid = a_col.is_valid().to_numpy(zero_copy_only=False)
                np.testing.assert_array_equal(valid, b_col.is_valid().to_numpy(zero_copy_only=False),
                                              err_msg=f"{left.name}: {field.name} null mask")
                a_values, b_values = (column.to_numpy(zero_copy_only=False) for column in (a_col, b_col))
                unsigned = np.dtype(f"u{a_values.dtype.itemsize}")
                # Arrow null payload bytes are not observable data. All valid
                # floating values, including signed zeros/NaN payloads, are exact.
                np.testing.assert_array_equal(a_values.view(unsigned)[valid], b_values.view(unsigned)[valid],
                                              err_msg=f"{left.name}: {field.name} floating bits")
        x = x.slice(size) if size < x.num_rows else next(a, None)
        y = y.slice(size) if size < y.num_rows else next(b, None)
    assert x is None and y is None
    return first.metadata.num_rows


def chart_payload(path: Path):
    """Read Plotly's JSON arguments without evaluating JavaScript or its DOM UUID."""
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    marker = "Plotly.newPlot("
    assert source.count(marker) == 1, (path.name, "one chart call required")
    source = source.split(marker, 1)[1]
    decoder, values = json.JSONDecoder(), []
    for index in range(4):
        value, end = decoder.raw_decode(source.lstrip())
        values.append(value)
        source = source.lstrip()[end:].lstrip()
        if index < 3:
            assert source.startswith(","), (path.name, "chart arguments")
            source = source[1:]
    assert source.startswith(")"), (path.name, "chart closing delimiter")
    return values[1:]  # data, layout and display config; each render has a fresh DOM UUID


REVIEWED_RENDERERS = (
    {"renderer_source_sha256": "3a7de598450e0ed96c34b2c9bec547309988adfefc133f557ae435a5d6c29f1f",
     "renderer_dependencies_sha256": {}},
    {"renderer_source_sha256": "848473fdd9e5fcf26641e3338cbe934827e9cff0a30f5ccdaf9beaced17b4511",
     "renderer_dependencies_sha256": {
         "holding_series.py": "eb15b14a34674c7196d643d6b6d40107d963f1b235ce1dafbf7dd12707089d4e"}},
)


def renderer_identity(report):
    return {"renderer_source_sha256": report.get("renderer_source_sha256"),
            "renderer_dependencies_sha256": report.get("renderer_dependencies_sha256", {})}


def compare_reports(left: Path, right: Path, *, allow_reviewed_renderer_change=False):
    """Require every declared chart and exact plotted data/layout/config parity."""
    left, right = Path(left), Path(right)
    first, second = [json.loads((path / "report_manifest.json").read_text(encoding="utf-8"))
                     for path in (left, right)]
    assert first["status"] == second["status"] == "completed"
    assert first["charts"] == second["charts"] and first["charts"], "chart inventory"
    assert len(set(first["charts"])) == len(first["charts"]), "duplicate charts"
    for name in first["charts"]:
        assert isinstance(name, str) and Path(name).name == name and Path(name).suffix == ".html", name
        a, b = left / name, right / name
        assert not a.is_symlink() and not b.is_symlink(), name
        assert json_values_exact(chart_payload(a), chart_payload(b)), (name, "data/layout/config")
    before, after = renderer_identity(first), renderer_identity(second)
    renderer_transition = None
    if before != after:
        assert allow_reviewed_renderer_change, "renderer_source_sha256/dependencies"
        assert (before, after) == REVIEWED_RENDERERS, "unreviewed renderer transition"
        renderer_transition = {"before": before, "after": after,
                               "scope": "RT-4 guarded sparse signed holding-series assembly; all charts still compared"}
    for key in ("plotly_version", "level", "holdings_rendered", "holdings", "threshold", "min_observations"):
        assert first.get(key) == second.get(key), key
    for path in (left, right):
        assert (path / "report_index.html").is_file(), "report dashboard"
        assert (path / "plotly.min.js").is_file(), "offline Plotly library"
    digests = []
    for path in (left / "plotly.min.js", right / "plotly.min.js"):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digests.append(digest.hexdigest())
    assert digests[0] == digests[1], "offline Plotly bytes"
    return {"left": str(left), "right": str(right), "status": "exact chart data/layout/config parity",
            "charts": first["charts"], "chart_count": len(first["charts"]),
            "reviewed_renderer_transition": renderer_transition}


def reviewed_engine_transition(first, second):
    """Allow only explicitly reviewed source/build pairs, never arbitrary pins.

    Both builds must be source-verified and clean. This acknowledges different
    code identities; it does not relax any execution setting or saved-output
    comparison. Unknown pins/builds continue to fail closed.
    """
    assert isinstance(first, dict) and isinstance(second, dict), "reviewed engine identities required"
    identities = {
        "da54853c7598599f74ff7776dc979b74106eb1d0": {
            "version": "0.9.7+tubby.2", "builds": {
                "win32-amd64": "d7474a7bf3e5aa4dc68461d0531339d92692cc827fa02080ece89da99321a243",
                "darwin-universal2": "7e02a5cc190fa886f723b84eea99ddb0e9a23f05a43fe20675735830b4617563"}},
        "d10a5a8ca397ea088da8f5c996e8bd042eea27e0": {
            "version": "0.9.7+tubby.3", "builds": {
                "win32-amd64": "2838655eda89662e0e13e42c4066501217f91ff8b09b4363b6b62fb8197b06dd",
                "darwin-universal2": "a8387ccca0d9908d3f3df77f7247efc4d12c209cd2952bd104b6335f14d7fbd1"}},
        "8ab6249aa6143cb0675259d5ff986d294e918ef4": {
            "version": "0.9.7+tubby.4", "builds": {
                "win32-amd64": "9157e9554a46b30e7ec4e83f7d9f03022c6b8781816da5170587d3fe6ec4e1af"}},
    }
    transitions = {
        ("da54853c7598599f74ff7776dc979b74106eb1d0", "d10a5a8ca397ea088da8f5c996e8bd042eea27e0"):
            "RT-1 immutable native executor date cache; all saved outputs still compared",
        ("d10a5a8ca397ea088da8f5c996e8bd042eea27e0", "8ab6249aa6143cb0675259d5ff986d294e918ef4"):
            "RT-3 opt-in generic ordered-session executor hook; all saved outputs still compared",
        ("da54853c7598599f74ff7776dc979b74106eb1d0", "8ab6249aa6143cb0675259d5ff986d294e918ef4"):
            "RT-1 date cache plus RT-3 opt-in generic ordered-session hook; all saved outputs still compared",
    }
    commits = first.get("commit"), second.get("commit")
    assert commits in transitions, "unreviewed engine transition"
    before, after = (identities[commit] for commit in commits)
    platform = first.get("platform")
    assert (platform == second.get("platform") and platform in before["builds"]
            and platform in after["builds"]), "reviewed engine platform differs"
    for identity, approved in zip((first, second), (before, after)):
        assert identity.get("wheel_sha256") == approved["builds"][platform], "unreviewed engine build"
        assert identity.get("version") == approved["version"], "reviewed engine version differs"
        assert identity.get("installed_package_verified") is True and identity.get("build_dirty") is False, "engine not verified clean"
        assert identity.get("repository") == "https://github.com/tubby-labs-research/qlib", "engine repository differs"
        assert identity.get("upstream_commit") == "da920b7f954f48ab1bb64117c976710de198373e", "engine upstream differs"
    for key in ("base_wheel_sha256", "build_method", "platform_tag"):
        assert first.get(key) and first.get(key) == second.get(key), f"engine {key} differs"
    return {"before": first, "after": second, "scope": transitions[commits]}


def compare_runs(left: Path, right: Path, *, allow_reviewed_engine_change=False):
    from itertools import zip_longest

    left, right = Path(left), Path(right)
    manifests = [json.loads((run / "manifest.json").read_text(encoding="utf-8")) for run in (left, right)]
    assert all(item["status"] == "completed" for item in manifests)
    both_streamed = all(
        isinstance(item.get("history_storage"), dict) and item["history_storage"].get("mode") == "stream"
        for item in manifests
    )
    for run, manifest in zip((left, right), manifests):
        for name, expected in manifest["artifacts_sha256"].items():
            assert isinstance(name, str) and Path(name).name == name and name not in (".", "..")
            path = run / name
            assert not path.is_symlink(), name
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            assert digest.hexdigest() == expected, (name, "sealed artifact hash")
    for key in ("alpha", "qlib_binary_tree_sha256", "dataset_manifest_sha256"):
        assert manifests[0].get(key) == manifests[1].get(key), key
    engine_transition = None
    first, second = (item.get("signed_engine_identity") for item in manifests)
    if first != second:
        assert allow_reviewed_engine_change, "signed_engine_identity"
        engine_transition = reviewed_engine_transition(first, second)
    for name in ("start", "end", "cash", "warmup_sessions", "universe", "execution_contract", "trade_symbols"):
        assert manifests[0]["backtest"].get(name) == manifests[1]["backtest"].get(name), name
    assert (left / "daily_report.csv").read_bytes() == (right / "daily_report.csv").read_bytes(), "daily ledger bytes"
    # Older runs did not save this native output. When both variants provide
    # it, also prove that discarding retained per-order objects changes no
    # native daily indicator (the new streamed per-order table is additional).
    indicator = "qlib_trade_indicators.csv"
    if (left / indicator).is_file() and (right / indicator).is_file():
        assert (left / indicator).read_bytes() == (right / indicator).read_bytes(), "native daily indicator bytes"
    elif (left / indicator).is_file() != (right / indicator).is_file() and both_streamed:
        raise AssertionError("streamed native daily indicator presence differs")
    summaries = [json.loads((run / "summary.json").read_text(encoding="utf-8")) for run in (left, right)]
    for key in ("metrics", "metrics_status", "funding_state"):
        assert json_values_exact(summaries[0].get(key), summaries[1].get(key)), key
    tables = {}
    for name in ("alpha_scores", "signed_weights", "signal_forward_labels", "positions"):
        a, b = left / (name + ".parquet"), right / (name + ".parquet")
        assert a.exists() == b.exists(), name
        if a.exists():
            tables[name] = compare_tables(a, b)
    # Stage 5+ streamed histories also save native per-order indicators. Compare
    # their entire logical table, not only the daily aggregated indicator CSV.
    # Earlier memory-history runs did not export this additional artifact.
    name = "qlib_order_indicators"
    a, b = left / (name + ".parquet"), right / (name + ".parquet")
    if a.exists() and b.exists():
        tables[name] = compare_tables(a, b)
    elif a.exists() != b.exists() and both_streamed:
        raise AssertionError("streamed native order indicator presence differs")
    count = 0
    decisions = "signed_decisions.json.gz"
    assert (left / decisions).exists() == (right / decisions).exists(), "decision history presence differs"
    if (left / decisions).exists():
        missing = object()
        for first, second in zip_longest(decision_records(left / "signed_decisions.json.gz"),
                                         decision_records(right / "signed_decisions.json.gz"), fillvalue=missing):
            assert json_values_exact(first, second), f"decision session {count}"
            count += 1
    return {"left": str(left), "right": str(right), "status": "exact parity", "table_rows": tables,
            "decision_sessions": count, "daily_csv_bytes_equal": True,
            "reviewed_engine_transition": engine_transition}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", action="store_true", help="Compare saved report directories instead of run cores.")
    parser.add_argument("--reviewed-engine-change", action="store_true", help="Allow only recorded clean tubby.2/3/4 source/build pairs; compare every output exactly.")
    parser.add_argument("--reviewed-renderer-change", action="store_true", help="Allow only the recorded RT-4 renderer/helper transition; compare every chart exactly.")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    result = (compare_reports(args.left, args.right, allow_reviewed_renderer_change=args.reviewed_renderer_change) if args.reports else
              compare_runs(args.left, args.right, allow_reviewed_engine_change=args.reviewed_engine_change))
    print(json.dumps(result, indent=2))
