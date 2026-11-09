"""Tiny real SEC/group sources for selected-workflow acceptance.

No API stubs or Qlib initialization. All sources, catalogs, dictionaries, panels
and results belong beneath the caller's disposable root. ``build_sec_group_fixture``
is suitable for Main's installed-runtime chain extension as well as unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class SECGroupFixture:
    root: Path
    dataset: Path
    days: object
    symbols: list[str]
    catalog: Path
    overlay: Path
    settings: object
    recipes: dict
    names: dict
    initial_values: list[float]

    @property
    def sec(self):
        return {"catalog": str(self.catalog), "mode": "dictionary"}

    @property
    def expression(self):
        return f"{self.names['base']} / close"

    def record(self, expression=None, settings=None, sec=None):
        from qlib_yahoo.alphalab.batch import run_batch
        return run_batch(self.dataset, [expression or self.expression], settings or self.settings,
                         data_root=self.root, sec=self.sec if sec is None else sec,
                         group_overlay=self.overlay, workers=1)[0]

    def prepare(self, result, name="targets"):
        from qlib_yahoo.candidate_workflow import prepare_candidate_targets
        return prepare_candidate_targets(self.root, result["attempt_id"], self.root / name)

    def selected_panel(self, name="base"):
        """Copy actual selected materialization; close its maps before returning."""
        from qlib_yahoo.sec_features import materialize, load_arrays, close_arrays
        from qlib_yahoo.sec_resolution import resolve_config
        public_name = self.names[name]
        config = resolve_config([public_name], self.sec, dialect="lab", verify=True)
        handle = materialize(self.catalog, config["bindings"], self.days.strftime("%Y-%m-%d").tolist(),
                             self.symbols, self.root / "cache/sec")
        arrays = load_arrays(handle)
        try:
            return arrays[public_name].copy(), handle
        finally:
            close_arrays(arrays)

    def groups(self):
        from qlib_yahoo.classification import classification_identity, load_classification_groups
        from qlib_yahoo.alphalab.batch import _aligned_groups
        from types import SimpleNamespace
        overlay = load_classification_groups(self.dataset, self.overlay,
                                             cache_root=self.root / "cache/classifications",
                                             expected_identity=classification_identity(self.dataset, self.overlay))
        workspace = SimpleNamespace(symbols=self.symbols, dates=self.days.to_numpy().astype("datetime64[D]"))
        return _aligned_groups(overlay, workspace), overlay


def build_sec_group_fixture(root, *, future_filing=700., future_revision=450.,
                            future_industry="B", unknown_symbols=(), all_unknown=False,
                            late_initial=False, include_base=True):
    """Build six issuers × 65 business sessions through the production importers.

    Initial facts are available before the price calendar unless ``late_initial``
    is true. A newer filing at session 30 and a revision at 34 become eligible
    after the SEC two-calendar-day bound. A classification event exactly at
    session 38's New York open becomes usable at session 39. Alternative recipes
    differ ONLY in currency, quarter duration, segment or co-registrant context.
    """
    from qlib_yahoo.alphalab.settings import Settings
    from qlib_yahoo.classification import create_classification_overlay
    from qlib_yahoo.sec_catalog import build_catalog
    from qlib_yahoo.sec_registry import COMMON_FIELDS, build_dictionary, name_for_spec
    from qlib_yahoo.sec_execution import build_execution_index
    try:
        from .test_workflow_chain import make_snapshot
        from .test_sec_catalog import tsv
    except ImportError:
        from test_workflow_chain import make_snapshot
        from test_sec_catalog import tsv

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset, days = make_snapshot(root)
    symbols = json.loads((dataset / "manifest.json").read_text())["symbols"]
    source = root / "sec-source"
    source.mkdir()
    recipes = {"base": dict(COMMON_FIELDS["sec_assets"])}
    recipes.update({name: {**recipes["base"], **changes} for name, changes in (
        ("currency", {"unit": "EUR"}), ("quarter", {"qtrs": 1}),
        ("segment", {"segments": "Axis=One"}), ("coreg", {"coreg": "Subsidiary"}))})
    initial_values = [100., 140., 210., 310., 440., 590.]
    submissions, facts = [], []

    def filing(cik, suffix, accepted, period, value, *, alternatives=False):
        accession = f"{cik:010d}-24-{suffix:06d}"
        submissions.append([accession, str(cik), accepted, accepted[:10].replace("-", ""), "10-Q"])
        if include_base:
            facts.append([accession, "Assets", "us-gaap/2023", period, 0, "USD", "", "", value])
        if alternatives:
            for offset, name in enumerate(("currency", "quarter", "segment", "coreg"), 1):
                spec = recipes[name]
                facts.append([accession, spec["tag"], "us-gaap/2023", period, spec["qtrs"],
                              spec["unit"], spec["segments"], spec["coreg"], 1000 * offset + value])

    initial_day = days[20].strftime("%Y-%m-%d") if late_initial else "2023-12-28"
    for cik, value in enumerate(initial_values, 1):
        filing(cik, 1, initial_day + " 16:30:00", "20230930", value, alternatives=True)
    filing(1, 2, days[30].strftime("%Y-%m-%d") + " 16:30:00", "20231231", future_filing)
    filing(2, 2, days[34].strftime("%Y-%m-%d") + " 16:30:00", "20230930", future_revision)
    tables = {
        "sub": tsv(["adsh", "cik", "accepted", "filed", "form"], submissions),
        "tag": tsv(["tag", "version", "custom", "tlabel", "doc", "datatype", "iord"],
                   [["Assets", "us-gaap/2023", "0", "Assets", "Synthetic total assets", "monetary", "I"]]),
        "num": tsv(["adsh", "tag", "version", "ddate", "qtrs", "uom", "segments", "coreg", "value"], facts),
        "pre": tsv(["adsh", "tag", "version", "stmt"],
                   [[row[0], "Assets", "us-gaap/2023", "BS"] for row in submissions]),
    }
    for name, content in tables.items():
        (source / f"{name}.txt").write_text(content, encoding="utf-8")
    ticker = root / "tickers.json"
    ticker.write_text(json.dumps({"fields": ["cik", "ticker"],
                                 "data": [[cik, symbol] for cik, symbol in enumerate(symbols, 1)]}))
    catalog = root / "sec/catalogs/synthetic"
    build_catalog([source], catalog, ticker_map=ticker)
    build_dictionary(catalog)
    build_execution_index(catalog)
    names = {name: name_for_spec(catalog, spec)["name"] for name, spec in recipes.items()
             if include_base or name != "base"}
    events = []
    if not all_unknown:
        events = [{"instrument": symbol, "available_at": "2023-12-28T00:00:00Z",
                   "industry": "A" if index < 3 else "B", "sector": "Synthetic division", "country": "US"}
                  for index, symbol in enumerate(symbols) if symbol not in unknown_symbols]
        if symbols[2] not in unknown_symbols:
            events.append({"instrument": symbols[2], "available_at": days[38].strftime("%Y-%m-%d") + "T14:30:00Z",
                           "industry": future_industry, "sector": "Synthetic division", "country": "US"})
    overlay = create_classification_overlay(
        dataset, root / "classifications/synthetic", events,
        source_metadata={"availability_policy": "synthetic dates; filing availability is inferred",
                         "historical_identity_limit": "synthetic current ticker-to-CIK proxy"},
        taxonomy_metadata={"industry": "synthetic SIC-like groups A/B", "sector": "synthetic division",
                           "country": "synthetic business-address country"})
    settings = Settings(universe="TOP6", start=days[10].strftime("%Y-%m-%d"),
                        end=days[45].strftime("%Y-%m-%d"), warmup=8, delay=1,
                        neutralization="INDUSTRY", truncation=.4, cost_bps=5)
    return SECGroupFixture(root, dataset, days, symbols, catalog, overlay, settings,
                           recipes, names, initial_values)
