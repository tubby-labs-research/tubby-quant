"""Reconstruct one immutable Lab attempt; never initialize Qlib or read Library.

Preview hashes only the bounded implementation files and reads recorded JSON.
Source byte verification and selected panel preparation belong to the explicit
materialization call. Targets are already transformed close decisions, not scores.
"""
from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import math
from pathlib import Path

from .dataset import DatasetError
from .workflow_records import EvidenceStore, canonical_json_bytes


ASSUMPTIONS = {
    "lab": "fixed-book adjusted close-to-close PnL; configured turnover costs; no stock borrow",
    "validation": "supplied close-decision weights executed at following session open by signed Qlib",
    "transforms": "Lab delay, decay, pasteurization, neutralization and cap already applied; no second transform",
    "weight_scale": "raw weights, gross at most one; subunit gross stays subunit",
    "comparison": "different fill timing, cash/share funding, costs and borrow; equal returns are not promised",
}


def _same(left, right, label):
    if left != right:
        raise DatasetError(f"candidate {label} differs from recorded evidence")


def _adapter_sources():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("candidate_workflow.py", "target_bundle.py")}


def _settings(value, *, canonical=True):
    from .alphalab.settings import Settings

    if not isinstance(value, dict) or set(value) != {field.name for field in fields(Settings)}:
        raise DatasetError("candidate has missing or unrecognized settings; no defaults supplied")
    for name in ("delay", "decay", "warmup"):
        if type(value[name]) is not int:
            raise DatasetError(f"candidate setting {name} must be an integer")
    if type(value["pasteurization"]) is not bool:
        raise DatasetError("candidate pasteurization must be boolean")
    for name in ("truncation", "booksize", "cost_bps"):
        if type(value[name]) not in (int, float) or not math.isfinite(value[name]):
            raise DatasetError(f"candidate setting {name} must be finite")
    for name in ("universe", "neutralization"):
        if not isinstance(value[name], str) or not value[name]:
            raise DatasetError(f"candidate setting {name} must be text")
    from datetime import date
    for name in ("start", "end", "oos_start"):
        if value[name] is not None:
            if not isinstance(value[name], str) or date.fromisoformat(value[name]).isoformat() != value[name]:
                raise DatasetError(f"candidate setting {name} must be an ISO date")
    settings = Settings(**value).validate()
    if canonical:
        _same(asdict(settings), value, "canonical settings")
    return settings


def _price_binding(identity):
    from .alphalab.batch import digest

    base = {key: value for key, value in identity.items()
            if key not in ("sec", "classification", "source_sha256")}
    needed = {"dataset", "verified", "manifest_sha256", "qlib_binary_tree_sha256",
              "auxiliary", "symbols", "dates", "fields"}
    if not needed.issubset(base) or not base["dates"] or not base["symbols"]:
        raise DatasetError("candidate is legacy or missing recorded price source bindings")
    base["source_sha256"] = digest(base)
    return base


def _combined(price, classification, sec):
    from .alphalab.batch import digest

    value = dict(price)
    # Match batch's order exactly: price, optional classification, selected SEC.
    for key, source in (("classification", classification), ("sec", sec)):
        if source is not None:
            value = {**value, key: source}
            value["source_sha256"] = digest(value)
    return value


def _native_universe(price, settings):
    """Resolve the same named membership, never silently widen a Lab subuniverse."""
    name = settings.universe.lower()
    records = price.get("universes") or {}
    record = records.get(name)
    if (not isinstance(record, dict) or record.get("instruments_file") != f"instruments/{name}.txt"
            or not record.get("instruments_sha256") or type(record.get("min_history_sessions")) is not int
            or record["min_history_sessions"] < 1):
        raise DatasetError("selected Lab universe has no recorded native Qlib membership file; "
                           "this handoff supports explicitly prepared matching universes, not implicit TOP-N subsets")
    return name


def read_candidate(data_root, attempt_id, *, _allow_observation_filter=False):
    """Read and validate a completed evaluation by ATTEMPT ID, never cache ID.

    No source-data scan, SEC lookup/materialization, panel load or Qlib init.
    Returns detached plain JSON ``{attempt, result}``; raises DatasetError.
    """
    try:
        from .alphalab import batch
        from .alphalab.simulate import ENGINE_VERSION
        from .sec_resolution import field_dependencies
        from .workflow_provenance import lab_implementation

        store = EvidenceStore(data_root)
        attempt = store.get(attempt_id)
        if (attempt["kind"] != "lab-evaluation" or attempt["status"] not in ("completed", "reused")
                or attempt["error"] is not None or not attempt["result_digest"]):
            raise DatasetError("select a successful or reused Lab evaluation with immutable evidence; failed/legacy attempts are unsupported")
        result = store.get_result(attempt["result_digest"])
        if (not isinstance(result, dict) or "error" in result or result.get("cached")
                or result.get("engine") != ENGINE_VERSION):
            raise DatasetError("candidate error or legacy/missing engine")
        implementation = result.get("implementation")
        if not implementation or (attempt.get("source") or {}).get("implementation") != implementation:
            raise DatasetError("candidate missing recorded implementation source binding")
        _same(lab_implementation(), implementation, "implementation/code or runtime")
        settings = _settings(result.get("settings"))
        filtered = settings.evaluation_sessions is not None
        if filtered and not _allow_observation_filter:
            raise DatasetError("observation-filtered chronology evidence is not a continuous trading candidate; "
                               "use the chronology selection workflow, not Qlib target validation")
        if not settings.pasteurization:
            raise DatasetError("candidate pasteurization=False is unsupported: handoff requires recorded eligible-universe targets; settings were not changed")
        if settings.warmup < 1:
            raise DatasetError("candidate warmup<1 cannot supply the preceding close decision or native runner warmup")
        request = attempt["request"]
        _same(_settings(request.get("settings"), canonical=False), settings, "submitted settings")
        expression = result.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise DatasetError("candidate missing expression")
        _same(" ".join(request["expression"].split()), expression, "submitted expression")
        if not result.get("candidate_id") or result["candidate_id"] != attempt["candidate_id"]:
            raise DatasetError("candidate semantic identity missing or mismatched")
        identity = result.get("dataset")
        if not isinstance(identity, dict):
            raise DatasetError("candidate missing recorded source bindings")
        price = _price_binding(identity)
        _native_universe(price, settings)
        _same(str(Path(request["dataset"]).resolve()), price["dataset"], "submitted dataset")
        group = identity.get("classification")
        consumed = batch._uses_classifications(expression, settings)
        if consumed:
            if (not isinstance(group, dict) or not group.get("path") or not request.get("group_overlay")
                    or not group.get("events_sha256") or not group.get("manifest_sha256")
                    or (result.get("classification") or {}).get("source_identity") != group):
                raise DatasetError("group-aware handoff requires recorded dated classification bindings; no static fallback")
            _same(str(Path(request["group_overlay"]).absolute()), group["path"], "classification path")
        elif group is not None:
            raise DatasetError("candidate contains unconsumed classification binding")
        sec = identity.get("sec")
        dependencies = {name for name in field_dependencies([expression], dialect="lab") if name.startswith("sec_")}
        if dependencies:
            saved = result.get("sec") or {}
            if (not isinstance(sec, dict) or not sec.get("catalog") or not sec.get("source_identity")
                    or set(sec.get("bindings", {})) != dependencies
                    or any(saved.get(key) != value for key, value in sec.items())):
                raise DatasetError("candidate missing recorded selected SEC source/recipe bindings")
            submitted_sec = request.get("sec")
            if not isinstance(submitted_sec, dict) or not submitted_sec.get("catalog"):
                raise DatasetError("candidate missing submitted SEC source binding")
            _same(str(Path(submitted_sec["catalog"]).expanduser().resolve()), sec["catalog"],
                  "submitted SEC catalog")
            if saved.get("mode") == "dictionary" and not sec.get("registry"):
                raise DatasetError("candidate missing recorded SEC dictionary registry")
        elif sec is not None:
            raise DatasetError("candidate contains unconsumed SEC binding")
        _same(_combined(price, group, sec), identity, "source identity composition")
        _same(batch.alpha_id(expression, settings, identity["source_sha256"], implementation),
              result.get("id"), "Lab cache identity")
        daily = result.get("daily") or {}
        if (not daily.get("dates") or not result.get("summary")
                or len(daily.get("pnl", [])) != len(daily["dates"])
                or len(daily.get("turnover", [])) != len(daily["dates"])):
            raise DatasetError("candidate missing original daily/summary numerical evidence")
        canonical_json_bytes(result)  # Never turn nonfinite evidence into null.
        _same(result["summary"].get("start"), daily["dates"][0], "summary start")
        _same(result["summary"].get("end"), daily["dates"][-1], "summary end")
        calendar = price["dates"]
        if (calendar != sorted(set(calendar)) or daily["dates"][0] not in calendar
                or daily["dates"][-1] not in calendar):
            raise DatasetError("candidate missing ordered recorded calendar for validation period")
        start = calendar.index(settings.start if filtered else daily["dates"][0])
        end = calendar.index(settings.end if filtered else daily["dates"][-1])
        if start < settings.warmup:
            raise DatasetError("candidate recorded warmup requires more preceding sessions; no clipping permitted")
        if end >= len(calendar) - 1:
            raise DatasetError("candidate validation end requires a later boundary session for the native runner")
        if filtered:
            expected = settings.evaluation_sessions
            _same(expected, daily['dates'], 'scored daily calendar')
            if any(day not in calendar[start:end + 1] for day in expected):
                raise DatasetError('scored session outside observed calendar')
            evaluation = result.get('evaluation') or {}
            for key, value in {'policy': 'observation-filter-v1', 'observed_start': calendar[start],
                               'observed_end': calendar[end], 'observed_count': end - start + 1,
                               'selected_sessions': expected, 'selected_count': len(expected)}.items():
                _same(evaluation.get(key), value, 'observation filter ' + key)
        else:
            _same(calendar[start:end + 1], daily["dates"], "daily calendar")
        return {"attempt": attempt, "result": result}
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        raise DatasetError(f"cannot reconstruct Lab candidate: {error}") from error


def _preview(candidate):
    result, attempt = candidate["result"], candidate["attempt"]
    identity = result["dataset"]
    price = _price_binding(identity)
    return {
        "schema_version": 1, "attempt_id": attempt["id"], "source_attempt_id": attempt["id"],
        "candidate_id": attempt["candidate_id"],
        "result_digest": attempt["result_digest"], "expression": result["expression"],
        "source_status": attempt["status"],
        "settings": result["settings"], "dataset": price["dataset"],
        "execution_universe": _native_universe(price, _settings(result["settings"])),
        "validation_window": {"start": result["daily"]["dates"][0], "end": result["daily"]["dates"][-1]},
        "period": {"start": result["summary"]["start"], "end": result["summary"]["end"]},
        "differences": list(ASSUMPTIONS.values()),
        "source_bindings": {"price_identity_sha256": price["source_sha256"],
                            "classification": identity.get("classification"), "sec": identity.get("sec"),
                            "implementation_sha256": result["implementation"]["sha256"]},
        "assumptions": dict(ASSUMPTIONS), "weight_policy": "lab_close_targets_v1",
        "qualification": {"status": "not_verified", "use": "exploratory_signed_validation",
                          "checks": {"attempt_and_result_seals": "verified", "implementation": "matched",
                                     "source_bytes": "pending", "numerical_reproduction": "pending"},
                          "limitations": ["source integrity and economic point-in-time validity are separate",
                                          "price, membership and share-adjustment limits require selected-input verification"],
                          "note": "preview verifies recorded evidence and code only; source bytes and reproduction are pending"},
    }


def preview_candidate(data_root, attempt_id):
    """Cheap explicit preview; source bytes/catalogs and dense panels are untouched."""
    return _preview(read_candidate(data_root, attempt_id))


def _verify_sources(result):
    from .alphalab import batch

    recorded = result["dataset"]
    price = batch.source_identity(Path(recorded["dataset"]))
    _same(price, _price_binding(recorded), "price source")
    group = None
    if "classification" in recorded:
        from .classification import classification_identity
        group = classification_identity(price["dataset"], recorded["classification"]["path"])
        _same(group, recorded["classification"], "dated classification source")
    sec = None
    if "sec" in recorded:
        from .sec_resolution import resolve_config
        from .sec_features import _verify_catalog, feature_identity
        saved = recorded["sec"]
        config = {"catalog": saved["catalog"], "bindings": saved["bindings"],
                  "mode": "dictionary" if "registry" in saved else "legacy"}
        if "registry" in saved:
            config["registry"] = saved["registry"]
        resolved = resolve_config([result["expression"]], config, dialect="lab", verify=True)
        _same(resolved["bindings"], saved["bindings"], "canonical SEC recipes")
        _same(resolved.get("registry"), saved.get("registry"), "SEC registry")
        catalog = Path(resolved["catalog"])
        _verify_catalog(catalog, force=True)
        sec = {"catalog": str(catalog), "bindings": resolved["bindings"],
               "source_identity": feature_identity(catalog, resolved["bindings"]),
               **({"registry": resolved["registry"]} if "registry" in resolved else {})}
    _same(_combined(price, group, sec), recorded, "selected source identities")
    return price, group, sec


def _compare(actual, saved, path="result"):
    """Compare every original numerical output, including nulls and structure."""
    if isinstance(saved, dict):
        if not isinstance(actual, dict) or actual.keys() != saved.keys():
            raise DatasetError(f"candidate reproduction structure differs: {path}")
        for key in saved:
            _compare(actual[key], saved[key], path + "." + key)
    elif isinstance(saved, list):
        if not isinstance(actual, list) or len(actual) != len(saved):
            raise DatasetError(f"candidate reproduction length differs: {path}")
        for index, (a, b) in enumerate(zip(actual, saved)):
            _compare(a, b, f"{path}[{index}]")
    elif type(saved) in (int, float) and type(actual) in (int, float):
        if not (math.isfinite(actual) and math.isfinite(saved)
                and math.isclose(actual, saved, rel_tol=1e-10, abs_tol=1e-9)):
            raise DatasetError(f"candidate numerical reproduction differs: {path}")
    elif actual != saved:
        raise DatasetError(f"candidate reproduction differs: {path}")


def prepare_candidate_targets(data_root, attempt_id, destination):
    """Verify inputs, reproduce original numbers, then publish a NEW target bundle.

    Source integrity is distinct from economic point-in-time qualification. The
    current supported use is explicitly exploratory signed validation. No Qlib
    execution, downloads, catalog build, parameter substitution or ledger writes.
    """
    import numpy as np
    from .alphalab import batch
    from .alphalab.simulate import simulate
    from .panel import available_fields, load_panel
    from .target_bundle import write_bundle
    from .workflow_provenance import lab_implementation

    adapter_sources = _adapter_sources()
    candidate = read_candidate(data_root, attempt_id)
    result = candidate["result"]
    preview = _preview(candidate)
    arrays = {}
    try:
        price, group, sec = _verify_sources(result)
        dataset = Path(price["dataset"])
        manifest_bytes = (dataset / "manifest.json").read_bytes()
        import json
        manifest = json.loads(manifest_bytes)
        _same(batch.digest(manifest), price["manifest_sha256"], "price manifest")
        required = {"open", "high", "low", "close", "volume", "factor"}
        if not required.issubset(available_fields(manifest)):
            raise DatasetError("candidate signed handoff requires OHLCV and factor fields")
        workspace = batch.Workspace(dataset)
        if (workspace.symbols != price["symbols"] or sorted(workspace.fields) != sorted(price["fields"])
                or [str(day) for day in workspace.dates] != price["dates"]):
            raise DatasetError("candidate loaded price calendar/symbols/fields differ")
        _same(workspace.panel.manifest, manifest, "loaded manifest")
        workspace.identity = result["dataset"]
        if group:
            from .classification import load_classification_groups
            overlay = load_classification_groups(dataset, group["path"],
                cache_root=Path(data_root) / "cache" / "classifications", expected_identity=group)
            workspace.groups = batch._aligned_groups(overlay, workspace)
        else:
            workspace.groups = {}  # Static labels cannot enter a handoff.
        if sec:
            from .sec_features import materialize, load_arrays
            handle = materialize(Path(sec["catalog"]), sec["bindings"], workspace.dates, workspace.symbols,
                                 Path(data_root) / "cache" / "sec", force_verify=False)
            arrays = load_arrays(handle)
            if set(arrays) != set(sec["bindings"]) or any(
                    value.shape != (len(workspace.dates), len(workspace.symbols))
                    or value.dtype != np.dtype("float32") for value in arrays.values()):
                raise DatasetError("candidate selected SEC panel fields/shapes/dtypes differ")
            workspace.fields.update(arrays)
        settings = _settings(result["settings"])
        first, start, end = workspace.window(settings)
        if start - first < 1:
            raise DatasetError("candidate missing warmup for preceding close decision")
        reproduced = simulate(result["expression"], settings, workspace, keep_decision_targets=True)
        for key in ("daily", "summary", "yearly", "periods", "checks"):
            _compare(reproduced[key], result.get(key), key)
        targets = reproduced["decision_targets"]
        matrix = targets["weights"]
        if not np.isfinite(matrix).all():
            raise DatasetError("candidate reconstructed targets contain nonfinite weights")
        members = workspace.universe(settings.universe)[start - 1:end]
        if np.any((matrix != 0) & ~members):
            raise DatasetError("candidate nonzero target outside dated membership; no remasking permitted")
        from .universe import load_dataset_universe
        native_spans, _ = load_dataset_universe(dataset, manifest, preview["execution_universe"])
        decision_dates = np.asarray(targets["dates"], dtype="datetime64[D]")
        for column, symbol in enumerate(workspace.symbols):
            allowed = np.zeros(len(decision_dates), dtype=bool)
            for begin, finish in native_spans.get(symbol, ()):
                allowed |= ((decision_dates >= np.datetime64(begin.date()))
                            & (decision_dates <= np.datetime64(finish.date())))
            if np.any((matrix[:, column] != 0) & ~allowed):
                raise DatasetError("Lab targets disagree with native dated membership for " + symbol)
        factor_panel = load_panel(dataset, ["factor"])
        if (factor_panel.symbols != workspace.symbols
                or not np.array_equal(factor_panel.dates, workspace.dates)):
            raise DatasetError("candidate factor calendar/symbols differ")
        factors = factor_panel.fields["factor"][start - 1:end + 1]
        close = workspace.fields["close"][start - 1:end + 1]
        eligible = workspace.universe(settings.universe)[start - 1:end + 1]
        if np.any(eligible & np.isfinite(close) & (~np.isfinite(factors) | (factors != 1))):
            raise DatasetError("candidate signed validation requires finite factor=1 on eligible price bars")
        # Native signed validation also checks present factors outside each
        # member's span, for symbols eligible anywhere in the decision window.
        selected_factors = factors[:, members.any(axis=0)]
        present = selected_factors[~np.isnan(selected_factors)]
        if not np.isfinite(present).all() or np.any(present != 1):
            raise DatasetError("candidate signed validation requires factor=1 wherever an eligible symbol has a bar")
        _verify_sources(result)  # No load-time source changes may publish targets.
        _same((dataset / "manifest.json").read_bytes(), manifest_bytes, "raw manifest bytes")
        _same(lab_implementation(), result["implementation"], "implementation during reconstruction")
        qualification = {
            "schema_version": 1, "use": "exploratory_signed_validation", "status": "exploratory_only",
            "checks": {"attempt_and_result_seals": "verified", "implementation": "matched",
                       "source_bytes": "verified", "numerical_reproduction": "matched"},
            "limitations": ["byte seals do not certify economic point-in-time validity",
                            "historical identity, delistings, membership and share-adjustment limitations remain",
                            "Lab fixed-book returns differ from next-open cash/share execution"],
            "integrity": {"price": price["verified"], "membership": "recorded checksums and loaded mask verified",
                          "sec": "selected source and recipes verified" if sec else "not_consumed",
                          "groups": "dated source and alignment verified" if group else "not_consumed",
                          "factor": "identity factor verified; no corporate-action/share ledger"},
            "price": {"provider": manifest.get("provider"), "normalization": manifest.get("normalization"),
                      "limitations": manifest.get("limitations", []),
                      "economic_pit": "not established by byte seals; historical identity/delistings and share-adjustment limitations remain"},
            "membership": {"universe": settings.universe,
                           "limitations": "dated membership is not proof of unbiased historical security coverage"},
            "sec": {"source": sec, "limitations": "inferred filing availability and current ticker mapping; missing coverage stays missing; no economic PIT certification"} if sec else None,
            "groups": {"source": group, "coverage": dict(overlay.coverage),
                       "limitations": "dated taxonomy/availability and security mapping are source-specific; coverage is not completeness or economic PIT"} if group else None,
            "source_links": {"dataset": str(dataset / "manifest.json"),
                             "attempt_id": attempt_id, "result_digest": preview["result_digest"]},
        }
        preview["qualification"] = qualification
        reproduction = {"status": "matched", "result_digest": preview["result_digest"],
                        "compared": ["daily", "summary", "yearly", "periods", "checks"],
                        "relative_tolerance": 1e-10, "absolute_tolerance": 1e-9,
                        "decision_start": targets["dates"][0], "decision_end": targets["dates"][-1],
                        "sessions": len(targets["dates"])}
        metadata = {"dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "price_identity": price, "candidate_result_digests": [preview["result_digest"]],
                    "attempt_ids": [attempt_id], "weight_policy": "lab_close_targets_v1",
                    "assumptions": dict(ASSUMPTIONS), "qualification": qualification,
                    "reproduction": reproduction, "candidate_id": preview["candidate_id"],
                    "source_links": qualification["source_links"],
                    "adapter_sources_sha256": adapter_sources}
        _same(_adapter_sources(), adapter_sources, "adapter implementation during reconstruction")
        bundle = write_bundle(destination, dates=targets["dates"], symbols=targets["symbols"],
                              weights=matrix, metadata=metadata)
        _same(_adapter_sources(), adapter_sources, "adapter implementation during publication")
        return {"bundle": bundle, "preview": preview, "qualification": qualification,
                "reproduction": reproduction}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as error:
        raise DatasetError(f"cannot prepare candidate targets: {error}") from error
    finally:
        batch._close_sec_arrays(arrays)


__all__ = ["read_candidate", "preview_candidate", "prepare_candidate_targets"]
