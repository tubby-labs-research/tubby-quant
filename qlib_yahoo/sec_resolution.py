"""Resolve expression dependencies to shared SEC recipes without reading facts.

Old explicit bindings remain a separate, reproducible legacy mode. Dictionary
mode never interprets an arbitrary nickname as a new economic definition.
"""
from __future__ import annotations

import ast
import difflib
import io
from pathlib import Path
import tokenize

from qlib_yahoo.dataset import DatasetError


def _native_syntax(expression):
    # Dollar signs inside a FASTEXPR string are not native feature syntax.
    try:
        return any(token.string == "$" and token.type not in (tokenize.STRING, tokenize.COMMENT)
                   for token in tokenize.generate_tokens(io.StringIO(expression).readline))
    except (tokenize.TokenError, IndentationError):
        return "$" in expression


def field_dependencies(expressions, *, dialect="auto") -> set[str]:
    """External names only: calls, keyword labels, strings and locals are not fields.

    Assignment scope is sequential. A name used before its assignment is still
    an external dependency; defining a local later cannot hide an unknown input.
    Native Qlib syntax is also validated without importing the Qlib runtime.
    """
    if dialect not in {"auto", "lab", "qlib"}:
        raise ValueError("Choose lab or qlib expression syntax")
    found = set()
    for expression in expressions:
        if not isinstance(expression, str):
            raise ValueError("Expressions must be strings")
        if not expression.strip() or expression.lstrip().startswith("#"):
            continue
        native = dialect == "qlib" or dialect == "auto" and _native_syntax(expression)
        if native:
            from qlib_yahoo.expressions import _FEATURE, _parse_alpha

            candidates = {name for name in _FEATURE.findall(expression) if name.startswith("sec_")}
            tree = _parse_alpha(expression, extra_features=candidates)
            found.update(node.id[9:] for node in ast.walk(tree)
                         if isinstance(node, ast.Name) and node.id.startswith("_feature_"))
        else:
            from qlib_yahoo.alphalab.fastexpr import names, parse

            program = parse(expression)
            assigned = {"true", "false"}
            for binding, value in program.assignments:
                found.update(names(value) - assigned)
                assigned.add(binding)
            found.update(names(program.result) - assigned)
    return found


def config_mode(sec: dict) -> str:
    """A supplied old bindings object is not consent to reinterpret its names."""
    mode = sec.get("mode", "legacy" if "bindings" in sec else "dictionary")
    if mode not in {"legacy", "dictionary"}:
        raise ValueError("SEC mode must be dictionary or legacy")
    return mode


def _unknown(name, catalog, common):
    # Error reporting must not run a global full-text query. Suggestions from
    # the small checked-in common-name map are advisory, never substitutions.
    nearby = difflib.get_close_matches(name, sorted(common), n=1, cutoff=0.65)
    suffix = f" Did you mean {nearby[0]!r}?" if nearby else " Search the SEC dictionary for an official name."
    return ValueError(f"Unknown SEC field: {name}.{suffix}")


def resolve_config(expressions, sec, *, dialect="auto", verify=False):
    """Resolve used inputs; workers verify only their sealed execution sections.

    Returns None for an unconfigured price-only expression. A selected catalog
    with no SEC dependencies is retained without opening its dictionary or facts.
    Every consumer must still apply its normal operator and dataset validation.
    """
    # Keep unrelated price-only calls on their existing syntax/validation path.
    potential = any(isinstance(text, str) and "sec_" in text.lower() for text in expressions)
    used = {name for name in field_dependencies(expressions, dialect=dialect)
            if name.startswith("sec_")} if potential else set()
    if sec is None:
        if used:
            raise DatasetError(f"SEC alias {sorted(used)[0]!r} requires a SEC catalog; choose one in SEC fields")
        return None
    if not isinstance(sec, dict) or not isinstance(sec.get("catalog"), str) or not sec["catalog"].strip():
        raise ValueError("SEC configuration requires a catalog path")
    if set(sec) - {"catalog", "bindings", "mode", "registry"}:
        raise ValueError("Unsupported SEC configuration keys")
    mode = config_mode(sec)
    catalog = Path(sec["catalog"]).expanduser().resolve()
    from qlib_yahoo.sec_features import load_bindings, selected_aliases

    supplied = load_bindings({"bindings": sec.get("bindings", {})})
    if mode == "legacy":
        return {"catalog": str(catalog), "mode": mode,
                "bindings": selected_aliases(expressions, supplied) if used else {}}
    result = {"catalog": str(catalog), "mode": mode, "bindings": {}}
    if not used:
        return result
    from qlib_yahoo.sec_registry import COMMON_FIELDS, dictionary_status, lookup_fields
    from qlib_yahoo.sec_execution import execution_status, verify_execution_fields
    from qlib_yahoo.sec_names import curated_defs

    status = dictionary_status(catalog)
    if status["status"] != "ready":
        if status["status"] == "unsupported":
            raise ValueError("This SEC source supports raw inspection only; numeric inputs need a schema adapter")
        raise ValueError("SEC dictionary is not prepared for this catalog. Open SEC fields and Prepare dictionary once.")
    keys = ("version", "catalog_sha256", "dictionary_sha256")
    provenance = {key: status[key] for key in keys}
    expected = sec.get("registry")
    if expected is not None and (
        not isinstance(expected, dict) or set(expected) - {*keys, "execution"}
        or any(expected.get(key) != provenance[key] for key in keys)
    ):
        raise ValueError("Saved SEC dictionary identity differs; restore the original dictionary or explicitly select a new source")
    entries = lookup_fields(catalog, sorted(used))
    reviewed_names = {**curated_defs(), **COMMON_FIELDS}
    for name in sorted(used):
        if name not in entries:
            if name in reviewed_names:
                raise ValueError(f"SEC field {name!r} is defined but unavailable in this catalog; "
                                 "its required unit, period or context was not observed")
            raise _unknown(name, catalog, reviewed_names)
        spec = load_bindings({name: entries[name]["spec"]})[name]
        if name in supplied and supplied[name] != spec:
            raise ValueError(f"Saved SEC field {name!r} has a different definition. "
                             "Keep its legacy bindings or explicitly migrate the formula; it was not replaced.")
        result["bindings"][name] = spec
    execution = execution_status(catalog)
    execution_identity = execution.get("identity") if execution["status"] == "ready" else None
    if expected is not None and "execution" in expected and expected["execution"] != execution_identity:
        raise ValueError("Saved SEC execution index identity differs; restore the original index or explicitly select a new source")
    if verify:
        if execution_identity is None:
            raise ValueError("SEC execution index is not ready. Open SEC fields and Prepare run index once. "
                             + execution.get("message", ""))
        # Readable public names are an overlay, not new execution artifacts.
        # Verify each original name/spec pair in its sealed shard, then compare
        # the exact selector under every requested spelling. Aliases may share
        # one recipe, so verify its storage name only once.
        storage_names = {name: entries[name].get('storage_name', name) for name in used}
        verified = verify_execution_fields(catalog, sorted(set(storage_names.values())),
                                           expected_identity=execution_identity)
        verified_bindings = {name: verified['fields'].get(storage) for name, storage in storage_names.items()}
        if (verified["identity"] != execution_identity
                or set(verified['fields']) != set(storage_names.values())
                or verified_bindings != result["bindings"]):
            raise ValueError("SEC selected definitions changed during verification")
    if execution_identity is not None:
        provenance["execution"] = execution_identity
    # Metadata remains stable while resolving; no disk-size scan at this point.
    after = dictionary_status(catalog)
    if any(after.get(key) != provenance[key] for key in keys):
        raise ValueError("SEC dictionary changed while resolving inputs; retry")
    result["registry"] = provenance
    return result
