"""Selected SEC facts on a shared, immutable daily research calendar.

Catalog imports stay sparse. This module materializes only explicit bindings;
its arrays are consumed identically by Alpha Lab and the native Qlib overlay.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import keyword
import os
from pathlib import Path
import re
import shutil
import tempfile
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation


POLICY = {
    "version": "sec-selected-1",
    "availability": "max(accepted_date,filed_date)+2_calendar_days;session_ge",
    "selection": "latest_period_then_latest_acceptance;conflict_blocks",
    "expiry_months": 15,
    "mapping": "current_SEC_tickers;US_suffix_and_dot_dash_normalized",
    "dtype": "float32;overflow_missing",
}
_ALIAS = re.compile(r"sec_[A-Za-z0-9_]+\Z")
_SPEC_KEYS = {"tag", "taxonomy", "unit", "qtrs", "segments", "coreg"}
_VERIFIED_CATALOGS = {}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_bindings(config: dict | None) -> dict:
    """Validate direct alias/spec mappings, or a config containing ``bindings``."""
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("SEC bindings must be a dictionary")
    source = config.get("bindings", config)
    if not isinstance(source, dict):
        raise ValueError("SEC bindings must be a dictionary")
    result = {}
    for alias, spec in sorted(source.items()):
        if not isinstance(alias, str) or not _ALIAS.fullmatch(alias) or keyword.iskeyword(alias):
            raise ValueError(f"Unsafe SEC alias: {alias!r}")
        if (not isinstance(spec, dict) or set(spec) - _SPEC_KEYS
                or not {"tag", "taxonomy", "unit", "qtrs"} <= set(spec)):
            raise ValueError(f"{alias}: specify tag, taxonomy, unit, qtrs, segments and coreg")
        copied = dict(spec)
        copied.setdefault("segments", "")
        copied.setdefault("coreg", "")
        for key in _SPEC_KEYS - {"qtrs"}:
            if not isinstance(copied[key], str):
                raise ValueError(f"{alias}: {key} must be a string")
        if any(not copied[key] or copied[key] != copied[key].strip()
               for key in ("tag", "taxonomy", "unit")):
            raise ValueError(f"{alias}: tag, taxonomy and unit must be explicit")
        tax = copied["taxonomy"]
        if tax.endswith("/") and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*/", tax):
            raise ValueError(f"{alias}: custom taxonomy selectors must be exact versions")
        if isinstance(copied["qtrs"], bool) or not isinstance(copied["qtrs"], int) or copied["qtrs"] < 0:
            raise ValueError(f"{alias}: qtrs must be a nonnegative integer")
        result[alias] = copied
    return result


def selected_aliases(expressions: list[str], bindings: dict) -> dict:
    """Select external SEC fields, not function names or sequential local variables."""
    from qlib_yahoo.sec_resolution import field_dependencies

    bindings = load_bindings(bindings)
    used = {name for name in field_dependencies(expressions) if name.startswith("sec_")}
    for name in sorted(used):
        if name not in bindings:
            raise ValueError(f"Unknown SEC field: {name}")
    return {name: bindings[name] for name in sorted(used)}


def feature_identity(catalog: Path, bindings: dict) -> dict:
    """Read sealed metadata identities only; never scan numeric tables or tickers."""
    from qlib_yahoo.sec_catalog import read_catalog

    catalog = Path(catalog).resolve()
    manifest = read_catalog(catalog, verify=False)
    catalog_sha = manifest.get("catalog_sha256")
    if not isinstance(catalog_sha, str) or not re.fullmatch("[0-9a-f]{64}", catalog_sha):
        raise ValueError("SEC catalog requires a canonical catalog_sha256")
    ticker_sha = manifest.get("artifacts_sha256", {}).get("tickers.json")
    if not isinstance(ticker_sha, str) or not re.fullmatch("[0-9a-f]{64}", ticker_sha):
        raise ValueError("SEC catalog requires sealed tickers.json")
    return {"catalog_sha256": catalog_sha, "mapping_sha256": ticker_sha,
            "bindings": load_bindings(bindings), "policy": dict(POLICY)}


def _catalog_guard(catalog):
    actual = sorted(p for p in catalog.rglob("*") if p.is_file())
    return tuple((p.relative_to(catalog).as_posix(), p.stat().st_size,
                  p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in actual)


def _verify_catalog(catalog, *, force=False):
    """Force hashes at execution boundaries; stat guards suffice within that run."""
    from qlib_yahoo.sec_catalog import read_catalog

    catalog = Path(catalog).resolve()
    manifest = read_catalog(catalog, verify=False)
    if not manifest.get("numeric_materialization_supported"):
        raise ValueError("SEC product needs an explicit numeric dimension adapter")
    before = _catalog_guard(catalog)
    key = (str(catalog), manifest["catalog_sha256"])
    if force or _VERIFIED_CATALOGS.get(key) != before:
        # A failed forced check must invalidate the earlier successful receipt.
        _VERIFIED_CATALOGS.pop(key, None)
        read_catalog(catalog, verify=True)
        if _catalog_guard(catalog) != before:
            raise ValueError("SEC catalog changed during verification")
        _VERIFIED_CATALOGS[key] = before
    return before


def _symbol(value):
    value = str(value).strip().upper()
    if value.endswith(".US"):
        value = value[:-3]
    return value.replace(".", "-")


def _cik(value):
    try:
        text = str(value).strip()
        return str(int(text)) if text.isdigit() and int(text) > 0 else None
    except (ValueError, TypeError):
        return None


def _mapping(path, symbols, diagnostics):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and "fields" in data and "data" in data:
        rows = [dict(zip(data["fields"], row)) for row in data["data"]]
    elif isinstance(data, dict):
        rows = list(data.values())
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError("Unsupported SEC ticker mapping schema")
    candidates = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid SEC ticker mapping row")
        ticker = row.get("ticker")
        cik = _cik(row.get("cik", row.get("cik_str")))
        if ticker and cik:
            candidates.setdefault(_symbol(ticker), set()).add(cik)
    output = {}
    for column, symbol in enumerate(symbols):
        choices = candidates.get(_symbol(symbol), set())
        if len(choices) == 1:
            output.setdefault(next(iter(choices)), []).append(column)
        elif choices:
            diagnostics["ambiguous_symbols"].append(str(symbol))
        else:
            diagnostics["unmapped_symbols"].append(str(symbol))
    return output


def _day(value):
    text = str(value or "").strip()
    if re.fullmatch(r"\d{8}", text):
        return datetime.strptime(text, "%Y%m%d").date()
    return date.fromisoformat(text)


def _accepted(value):
    text = str(value or "").strip()
    if re.fullmatch(r"\d{14}", text):
        stamp = datetime.strptime(text, "%Y%m%d%H%M%S")
    else:
        # A date alone is not an acceptance clock.
        if "T" not in text and " " not in text:
            raise ValueError("Missing acceptance clock")
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    # Retain the source's date; normalize offset-aware clocks for ordering.
    from datetime import timezone
    ordered = stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)
    return stamp.date(), ordered


def _expires(period):
    months = period.year * 12 + period.month - 1 + POLICY["expiry_months"]
    year, month0 = divmod(months, 12)
    month = month0 + 1
    return date(year, month, min(period.day, calendar.monthrange(year, month)[1]))


def _matches(row, spec):
    version = str(row.get("version", ""))
    tax = spec["taxonomy"]
    return (str(row.get("tag", "")) == spec["tag"]
            and (version.startswith(tax) if tax.endswith("/") else version == tax)
            and str(row.get("uom", row.get("unit", ""))) == spec["unit"]
            and str(row.get("qtrs", "")) == str(spec["qtrs"])
            and str(row.get("segments") or "") == spec["segments"]
            and str(row.get("coreg") or "") == spec["coreg"])


def _numeric(values, diagnostics):
    import numpy as np

    if len(values) != 1 or None in values:
        diagnostics["blocked_winners"] += 1
        return np.float32(np.nan)
    value = next(iter(values))
    if value.copy_abs() > Decimal(str(np.finfo(np.float32).max)):
        diagnostics["overflow_winners"] += 1
        return np.float32(np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.float32(str(value))
    if not np.isfinite(converted):
        diagnostics["overflow_winners"] += 1
        return np.float32(np.nan)
    return converted


def _build_field(catalog, spec, dates, mapping, diagnostics):
    import numpy as np
    from qlib_yahoo.sec_catalog import iter_facts

    output = np.full((len(dates), diagnostics["symbol_count"]), np.nan, dtype=np.float32)
    events = {}
    for row in iter_facts(catalog, spec):
        if not _matches(row, spec):
            continue
        cik = _cik(row.get("cik"))
        if cik not in mapping:
            continue
        if (row.get("ambiguous_sub") or row.get("sub_ambiguous")
                or row.get("filing_conflict") or row.get("filing_missing")):
            diagnostics["ineligible_rows"] += 1
            continue
        try:
            period = _day(row.get("ddate"))
            accepted_day, accepted = _accepted(row.get("accepted"))
            bound = max(accepted_day, _day(row.get("filed"))) + timedelta(days=2)
            if period > max(accepted_day, _day(row.get("filed"))):
                raise ValueError("Future reported period")
        except (ValueError, TypeError, OverflowError):
            diagnostics["ineligible_rows"] += 1
            continue
        try:
            value = Decimal(str(row.get("value", "")).strip())
            if not value.is_finite():
                value = None
        except InvalidOperation:
            value = None
        events.setdefault(cik, []).append((bound, period, accepted, value))
    for cik, records in events.items():
        records.sort(key=lambda event: (event[0], event[1], event[2]))
        periods = {}
        offset = 0
        winning_period = None
        while offset < len(records):
            bound = records[offset][0]
            while offset < len(records) and records[offset][0] == bound:
                _, period, accepted, value = records[offset]
                prior = periods.get(period)
                if prior is None or accepted > prior[0]:
                    periods[period] = (accepted, {value})
                elif accepted == prior[0]:
                    prior[1].add(value)
                if winning_period is None or period > winning_period:
                    winning_period = period
                offset += 1
            start = int(np.searchsorted(dates, np.datetime64(bound), side="left"))
            next_bound = records[offset][0] if offset < len(records) else date.max
            end = min(int(np.searchsorted(dates, np.datetime64(next_bound), side="left")),
                      int(np.searchsorted(dates, np.datetime64(_expires(winning_period)), side="right")))
            if start < end:
                output[start:end, mapping[cik]] = _numeric(periods[winning_period][1], diagnostics)
    return output


def _field_identity(identity, spec):
    """Numeric definition identity, deliberately independent of binding names."""
    return {key: identity[key] for key in
            ("catalog_sha256", "mapping_sha256", "policy", "dates", "symbols")} | {"spec": spec}


def _cache_path(path):
    """Reject redirected cache objects, including Windows junctions."""
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("SEC cache path is redirected")
    return path


def _checked_array(path, sha, identity, alias):
    import numpy as np

    path = _cache_path(path)
    before = path.stat() if path.is_file() else None
    if before is None or _hash(path) != sha:
        raise ValueError(f"SEC cached field hash mismatch: {alias}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if values.shape != (len(identity["dates"]), len(identity["symbols"])) or values.dtype != np.dtype("float32"):
            raise ValueError(f"SEC cached field shape/dtype mismatch: {alias}")
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f"SEC cached field changed during loading: {alias}")
        return values
    except BaseException:
        close_arrays({alias: values})
        raise


def _read_field(root, identity, *, validate_array=True):
    key = _digest(identity)
    target = _cache_path(root / "fields-v1" / key)
    receipt = json.loads(_cache_path(target / "field.json").read_text(encoding="utf-8"))
    if receipt.get("identity") != identity or receipt.get("identity_sha256") != key:
        raise ValueError("SEC cached field identity mismatch")
    if validate_array:
        close_arrays({key: _checked_array(target / "values.npy", receipt.get("sha256"), identity, key)})
    counters = receipt.get("diagnostics")
    if not isinstance(counters, dict) or set(counters) != {
            "ineligible_rows", "blocked_winners", "overflow_winners"} or any(
                type(value) is not int or value < 0 for value in counters.values()):
        raise ValueError("SEC cached field diagnostics mismatch")
    return receipt


def _publish_directory(staging, target):
    """Publish complete objects, tolerating short Windows no-delete-share locks.

    The caller validates any competing winner. Never replace it or retry its
    validation failures. Persistent access denial retains the original error.
    """
    import time

    for attempt in range(5):
        if _cache_path(target).exists():
            return
        try:
            os.rename(staging, target)
            return
        except OSError as exc:
            if _cache_path(target).exists():
                return
            # Directory renames can report ACCESS_DENIED (5), as well as
            # SHARING_VIOLATION (32), for a temporarily open child file. Do not
            # generalize to errno EACCES or other platforms/errors. An actual
            # permissions failure still raises after five attempts / 0.75s.
            if os.name != "nt" or getattr(exc, "winerror", None) not in (5, 32) or attempt == 4:
                raise
            time.sleep(0.05 * 2 ** attempt)


def materialize(catalog: Path, bindings: dict, dates, symbols, cache_root: Path, progress=None,
                *, force_verify=True) -> dict:
    """Reuse definition-keyed panels in alias-sensitive handles.

    Standalone calls force source verification before any cache hit.
    A runner that called ``_verify_catalog(catalog, force=True)`` at its current
    invocation boundary may pass ``force_verify=False`` to avoid repeated hashes.
    Calendar indexes are absolute and full length.
    """
    import numpy as np

    catalog = Path(catalog).resolve()
    bindings = load_bindings(bindings)
    calendar_values = np.asarray(dates, dtype="datetime64[D]")
    if calendar_values.ndim != 1 or np.isnat(calendar_values).any():
        raise ValueError("SEC calendar must contain finite ordered dates")
    if len(calendar_values) > 1 and not (calendar_values[1:] > calendar_values[:-1]).all():
        raise ValueError("SEC calendar must be unique and increasing")
    symbols = list(map(str, symbols))
    if len(symbols) != len(set(symbols)):
        raise ValueError("SEC symbols must be unique")
    source_guard = _verify_catalog(catalog, force=force_verify)
    metadata = feature_identity(catalog, bindings)
    identity = dict(metadata, dates=list(map(str, calendar_values)), symbols=symbols)
    identity_sha = _digest(identity)
    root = _cache_path(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = _cache_path(root / identity_sha)
    if target.exists():
        handle = json.loads(_cache_path(target / "handle.json").read_text(encoding="utf-8"))
        handle["path"] = str(target)
        if handle.get("identity") != identity or handle.get("identity_sha256") != identity_sha:
            raise ValueError("SEC cached identity mismatch")
        close_arrays(load_arrays(handle))
        if _verify_catalog(catalog) != source_guard or feature_identity(catalog, bindings) != metadata:
            raise ValueError("SEC source files changed during materialization")
        return handle
    diagnostics = {"ambiguous_symbols": [], "unmapped_symbols": [], "symbol_count": len(symbols),
                   "ineligible_rows": 0, "blocked_winners": 0, "overflow_winners": 0,
                   "winner_counter_unit": "selected_field_issuer_intervals",
                   "mapping_is_current_proxy": True}
    mapping = _mapping(catalog / "tickers.json", symbols, diagnostics)
    staging = Path(tempfile.mkdtemp(prefix=".sec-selected-", dir=root))
    try:
        hashes = {}
        references = {}
        object_root = _cache_path(root / "fields-v1")
        object_root.mkdir(exist_ok=True)
        for index, (alias, spec) in enumerate(bindings.items()):
            if progress:
                progress(f"Materializing SEC field {alias} ({index + 1}/{len(bindings)})")
            field_identity = _field_identity(identity, spec)
            key = _digest(field_identity)
            field_target = _cache_path(object_root / key)
            if not field_target.exists():
                field_staging = Path(tempfile.mkdtemp(prefix=".f-", dir=object_root))
                try:
                    counters = {name: 0 for name in
                                ("ineligible_rows", "blocked_winners", "overflow_winners")}
                    field_diagnostics = dict(diagnostics, **counters)
                    values = _build_field(catalog, spec, calendar_values, mapping, field_diagnostics)
                    np.save(field_staging / "values.npy", values, allow_pickle=False)
                    sha = _hash(field_staging / "values.npy")
                    close_arrays({alias: _checked_array(field_staging / "values.npy", sha, identity, alias)})
                    if feature_identity(catalog, bindings) != metadata or _verify_catalog(catalog) != source_guard:
                        raise ValueError("SEC source files changed during materialization")
                    receipt = {"identity": field_identity, "identity_sha256": key, "sha256": sha,
                               "diagnostics": {name: field_diagnostics[name] for name in counters}}
                    (field_staging / "field.json").write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
                    _publish_directory(field_staging, field_target)
                finally:
                    if field_staging.exists():
                        shutil.rmtree(field_staging)
            receipt = _read_field(root, field_identity)
            hashes[alias] = receipt["sha256"]
            references[alias] = key
            for name, count in receipt["diagnostics"].items():
                diagnostics[name] += count
        # Detect source/mapping mutation during materialization before committing.
        if feature_identity(catalog, bindings) != metadata:
            raise ValueError("SEC catalog identity changed during materialization")
        if _verify_catalog(catalog) != source_guard:
            raise ValueError("SEC source files changed during materialization")
        handle = {"path": str(target), "identity": identity, "identity_sha256": identity_sha,
                  "fields": list(bindings), "filehashes": hashes, "diagnostics": diagnostics,
                  "cache_schema": "sec-field-refs-1", "field_objects": references}
        (staging / "handle.json").write_text(json.dumps(handle, sort_keys=True, indent=2), encoding="utf-8")
        _publish_directory(staging, target)
        winner = json.loads(_cache_path(target / "handle.json").read_text(encoding="utf-8"))
        winner["path"] = str(target)
        if winner.get("identity") != identity or winner.get("identity_sha256") != identity_sha:
            raise ValueError("SEC competing cache identity mismatch")
        close_arrays(load_arrays(winner))
        return winner
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_arrays(handle: dict) -> dict:
    """Verify immutable files before opening read-only memory maps."""
    identity = handle["identity"]
    if _digest(identity) != handle.get("identity_sha256"):
        raise ValueError("SEC handle identity mismatch")
    fields = handle.get("fields")
    if fields != sorted(identity["bindings"]) or set(handle.get("filehashes", {})) != set(fields):
        raise ValueError("SEC handle field inventory mismatch")
    arrays = {}
    try:
        schema = handle.get("cache_schema")
        if schema not in (None, "sec-field-refs-1"):
            raise ValueError("SEC unknown cache schema")
        references = handle.get("field_objects", {})
        base = _cache_path(handle["path"])
        if schema and (base.name != handle["identity_sha256"] or set(references) != set(fields)):
            raise ValueError("SEC handle field object inventory mismatch")
        for alias in fields:
            if not _ALIAS.fullmatch(alias):
                raise ValueError("Unsafe SEC cached field")
            if schema:
                field_identity = _field_identity(identity, identity["bindings"][alias])
                key = _digest(field_identity)
                if references[alias] != key:
                    raise ValueError("SEC handle field object identity mismatch")
                receipt = _read_field(base.parent, field_identity, validate_array=False)
                if receipt["sha256"] != handle["filehashes"][alias]:
                    raise ValueError(f"SEC cached field hash mismatch: {alias}")
                path = base.parent / "fields-v1" / key / "values.npy"
            else:
                path = base / f"{alias}.npy"
            arrays[alias] = _checked_array(path, handle["filehashes"][alias], identity, alias)
    except BaseException:
        close_arrays(arrays)
        raise
    return arrays


def close_arrays(arrays: dict) -> None:
    """Release owned maps; callers must stop using the arrays and their views."""
    seen = set()
    for values in arrays.values():
        mmap = getattr(values, "_mmap", None)
        if mmap is not None and id(mmap) not in seen:
            mmap.close()
            seen.add(id(mmap))
