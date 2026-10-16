"""A deliberately small, auditable subset of Qlib daily expressions."""

from __future__ import annotations

import ast
import math
import re

from qlib_yahoo.dataset import DatasetError


FEATURES = frozenset({"open", "high", "low", "close", "volume", "factor"})
OPERATORS = frozenset({
    "Abs", "Sign", "Log", "Power", "Greater", "Less", "If", "Ref",
    "Mean", "Sum", "Std", "Max", "Min", "Rank", "Delta", "Slope",
    "WMA", "EMA", "Corr", "Cov",
})
_FEATURE = re.compile(r"\$([A-Za-z][A-Za-z0-9_]*)")
_BARE_ALIAS = re.compile(r"\b_feature_[A-Za-z0-9_]*\b")
_ARITIES = {
    **{name: 1 for name in ("Abs", "Sign", "Log")},
    **{name: 2 for name in ("Power", "Greater", "Less", "Ref", "Mean", "Sum", "Std",
                             "Max", "Min", "Rank", "Delta", "Slope", "WMA", "EMA")},
    **{name: 3 for name in ("If", "Corr", "Cov")},
}
_WINDOW_INDEX = {name: (2 if name in ("Corr", "Cov") else 1)
                 for name in ("Ref", "Mean", "Sum", "Std", "Max", "Min", "Rank",
                              "Delta", "Slope", "WMA", "EMA", "Corr", "Cov")}
MAX_ALPHA_CHARS = 4096
MAX_ALPHA_NODES = 256
MAX_LITERAL = 1_000_000_000_000
MAX_WINDOW = 4096


def _literal_window(node: ast.AST) -> int:
    if isinstance(node, ast.Constant):
        value = node.value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        if type(node.operand.value) is not int:
            raise DatasetError("rolling window must be a literal finite integer")
        value = -node.operand.value
    else:
        raise DatasetError("rolling window must be a literal finite integer")
    if type(value) is not int or abs(value) > MAX_WINDOW:
        raise DatasetError("rolling window must be a bounded literal integer")
    return value


def _check(node: ast.AST, features=FEATURES) -> bool:
    """Validate an AST and report whether it contains a market feature."""
    if isinstance(node, ast.Expression):
        return _check(node.body, features)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in OPERATORS or node.keywords:
            raise DatasetError("alpha uses an unsupported operator or call")
        name = node.func.id
        if len(node.args) != _ARITIES[name]:
            raise DatasetError(f"{name} requires {_ARITIES[name]} arguments")
        if name in _WINDOW_INDEX:
            window = _literal_window(node.args[_WINDOW_INDEX[name]])
            if window == 0 or (name != "Ref" and window < 1) or (name == "Ref" and window < 0):
                raise DatasetError("future, expanding, zero or invalid rolling window is unsupported")
        dependencies = [_check(arg, features) for index, arg in enumerate(node.args)
                        if name not in _WINDOW_INDEX or index != _WINDOW_INDEX[name]]
        if not any(dependencies):
            raise DatasetError(f"{name} must depend on an allowed market feature")
        return True
    elif isinstance(node, ast.Name):
        if not node.id.startswith("_feature_"):
            raise DatasetError(f"alpha uses unsupported name: {node.id}")
        if node.id[9:] not in features:
            raise DatasetError(f"alpha uses unsupported feature: ${node.id[9:]}")
        return True
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        left = _check(node.left, features)
        right = _check(node.right, features)
        if isinstance(node.op, ast.Pow) and not (left or right):
            raise DatasetError("constant-only power is unsupported")
        return left or right
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        if _check(node.operand, features):
            raise DatasetError("Qlib expressions do not support a sign before a feature term; write -1 * (...) instead")
        return False
    elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if abs(node.value) > MAX_LITERAL or not math.isfinite(node.value):
            raise DatasetError("alpha numeric literal is nonfinite or too large")
        return False
    else:
        raise DatasetError("alpha syntax is outside the vetted numeric Qlib subset")


def _parse_alpha(alpha: str, *, extra_features=()) -> ast.Expression:
    if not isinstance(alpha, str) or not alpha.strip() or "$$" in alpha:
        raise DatasetError("alpha must be a nonempty daily feature expression")
    if len(alpha) > MAX_ALPHA_CHARS:
        raise DatasetError(f"alpha exceeds {MAX_ALPHA_CHARS} characters")
    if _BARE_ALIAS.search(alpha):
        raise DatasetError("alpha features must use $field syntax")
    transformed = _FEATURE.sub(lambda match: "_feature_" + match.group(1), alpha.strip())
    try:
        tree = ast.parse(transformed, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as error:
        raise DatasetError(f"invalid alpha syntax: {error}") from error
    if sum(1 for _ in ast.walk(tree)) > MAX_ALPHA_NODES:
        raise DatasetError(f"alpha exceeds {MAX_ALPHA_NODES} syntax nodes")
    if not _check(tree, FEATURES | frozenset(extra_features)):
        raise DatasetError("alpha must depend on an allowed market feature")
    return tree


def check_alpha_syntax(alpha: str, *, extra_features=()) -> None:
    """Fast, stdlib-only structural check; Qlib still verifies dependency windows."""
    _parse_alpha(alpha, extra_features=extra_features)


def validate_alpha(alpha: str, *, extra_features=()) -> int:
    """Return finite required history; reject future or uncertain history.

    Qlib's own expression object supplies the authoritative extended window.
    Zero-length windows are rejected because Qlib treats them as expanding or
    first-observation references, which cannot be certified by a finite warmup.
    """
    extra_features = frozenset(extra_features)
    tree = _parse_alpha(alpha, extra_features=extra_features)
    from qlib.data.base import Expression, Feature
    from qlib.data import ops

    try:
        namespace = {name: getattr(ops, name) for name in OPERATORS}
        namespace.update({"_feature_" + name: Feature(name) for name in FEATURES | extra_features})
        expression = eval(compile(tree, "<validated-alpha>", "eval"), {"__builtins__": {}}, namespace)
        if not isinstance(expression, Expression):
            raise DatasetError("alpha must depend on an allowed market feature")
        left, right = expression.get_extended_window_size()
        if not all(math.isfinite(v) and v >= 0 for v in (left, right)) or right > 0:
            raise DatasetError("alpha requires future data or unbounded history")
        return int(math.ceil(left))
    except DatasetError:
        raise
    except (SyntaxError, TypeError, ValueError, AttributeError, NameError, OverflowError) as error:
        raise DatasetError(f"unsupported alpha expression: {error}") from error


def sec_selection(expressions: list[str], sec: dict | None):
    """Resolve shared dictionary or preserved legacy fields, without arrays/Qlib."""
    from pathlib import Path
    from qlib_yahoo.sec_resolution import resolve_config

    try:
        resolved = resolve_config(expressions, sec)
        return (Path(resolved["catalog"]), resolved["bindings"]) if resolved else (None, {})
    except (ValueError, TypeError, KeyError) as error:
        raise DatasetError(str(error)) from error
