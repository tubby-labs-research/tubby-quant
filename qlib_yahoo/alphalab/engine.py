"""Evaluate a parsed FASTEXPR program on a panel of prices.

Data fields (dates x instruments, split- and dividend-adjusted where prices):

    open high low close volume      as stored in the snapshot
    returns                         close / previous close - 1
    vwap                            the stored daily VWAP when the snapshot has one, else (high + low + close) / 3
    dollar_volume                   close x volume (traded dollars)
    adv<d>                          mean dollar volume over d days, e.g. adv20
    close_raw, dividend             traded close and cash dividend, when the snapshot has them
    cap                             market capitalisation, when the snapshot has it

Groups for group_* operators and neutralization: market, sector, industry,
country (the last three when the snapshot records groups).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from qlib_yahoo.alphalab import ops
from qlib_yahoo.alphalab.fastexpr import (Binary, Call, ExpressionError, Name, Num, Program, Str, Ternary, Unary,
                                          parse)

BASE_FIELDS = ("open", "high", "low", "close", "volume")
OPTIONAL_FIELDS = ("close_raw", "dividend", "cap", "vwap")
GROUPS = ("market", "sector", "industry", "country")
ADV = re.compile(r"adv(\d{1,3})\Z")


@dataclass
class Context:
    """What an expression can read: stored fields, the member mask and group codes."""

    dates: Any
    symbols: list[str]
    fields: dict[str, np.ndarray]
    members: np.ndarray
    groups: dict[str, np.ndarray] = field(default_factory=dict)  # name -> float codes per symbol (NaN unknown)
    cache: dict[Any, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.members.shape

    def field(self, name: str) -> np.ndarray:
        if name in self.cache:
            return self.cache[name]
        value = self._field(name)
        self.cache[name] = value
        return value

    def _field(self, name: str) -> np.ndarray:
        if name in self.fields:
            return self.fields[name]
        if name == "returns":
            close = self.field("close").astype(np.float64)
            out = np.full(close.shape, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[1:] = close[1:] / close[:-1] - 1
            return ops._clean(out)
        if name == "vwap":
            return ops._clean((self.field("high").astype(np.float64) + self.field("low") + self.field("close")) / 3)
        if name == "dollar_volume":
            return ops._clean(self.field("close").astype(np.float64) * self.field("volume"))
        match = ADV.fullmatch(name)
        if match:
            return ops.ts_mean(self.field("dollar_volume"), int(match.group(1)))
        available = sorted({*self.fields, "returns", "vwap", "dollar_volume", "adv20"})
        raise ExpressionError(f"unknown data field {name!r}; available: {', '.join(available)} (adv<d> for any d)")

    def group(self, name: str) -> np.ndarray:
        if name == "market":
            return np.zeros(len(self.symbols))
        if name in self.groups:
            return self.groups[name]
        if name in ("subindustry", "sector", "industry", "country"):
            raise ExpressionError(f"group {name!r} is not available for this dataset "
                                  f"(available: {', '.join(['market', *sorted(self.groups)])})")
        raise ExpressionError(f"unknown group {name!r}")


def _scalar(value: Any, what: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ExpressionError(f"{what} must be a number, not a data series")


# name -> (function, positional parameter kinds, keyword defaults). Kinds: "x" a series (numbers broadcast),
# "n" a number, "s" a string, "g" a group, "*" any number of series.
TS = {name: (getattr(ops, name), ("x", "n"), {}) for name in (
    "ts_delay", "ts_delta", "ts_mean", "ts_sum", "ts_std_dev", "ts_max", "ts_min", "ts_skewness", "ts_kurtosis",
    "ts_zscore", "ts_ir", "ts_product", "ts_arg_max", "ts_arg_min", "ts_backfill", "ts_count_nans")}
REGISTRY: dict[str, tuple[Callable[..., Any], tuple[str, ...], dict[str, Any]]] = {
    **TS,
    "ts_rank": (ops.ts_rank, ("x", "n"), {"constant": 0}),
    "ts_decay_linear": (ops.ts_decay_linear, ("x", "n"), {"dense": False}),
    "ts_corr": (ops.ts_corr, ("x", "x", "n"), {}),
    "ts_covariance": (ops.ts_covariance, ("x", "x", "n"), {}),
    "ts_regression": (ops.ts_regression, ("x", "x", "n"), {"lag": 0, "rettype": 0}),
    "hump": (ops.hump, ("x",), {"hump": 0.01}),
    "trade_when": (ops.trade_when, ("x", "x", "x"), {}),
    "days_from_last_change": (ops.days_from_last_change, ("x",), {}),
    "rank": (ops.rank, ("x",), {"rate": 2}),
    "zscore": (ops.zscore, ("x",), {}),
    "normalize": (ops.normalize, ("x",), {"useStd": False, "limit": 0.0}),
    "scale": (ops.scale, ("x",), {"scale": 1.0, "longscale": None, "shortscale": None}),
    "winsorize": (ops.winsorize, ("x",), {"std": 4.0}),
    "vector_neut": (ops.vector_neut, ("x", "x"), {}),
    "group_neutralize": (ops.group_neutralize, ("x", "g"), {}),
    "group_mean": (ops.group_mean, ("x", "g"), {}),
    "group_zscore": (ops.group_zscore, ("x", "g"), {}),
    "group_rank": (ops.group_rank, ("x", "g"), {}),
    "bucket": (ops.bucket, ("x",), {"range": "0,1,0.1", "buckets": None}),
    "if_else": (ops.if_else, ("x", "x", "x"), {}),
    "power": (ops.power, ("x", "x"), {}),
    "signed_power": (ops.signed_power, ("x", "x"), {}),
    "clamp": (ops.clamp, ("x",), {"lower": -np.inf, "upper": np.inf}),
    "min": (ops.elementwise_min, ("*",), {}),
    "max": (ops.elementwise_max, ("*",), {}),
    **{name: (fn, ("x",), {}) for name, fn in ops.ELEMENTWISE.items()},
}
ALIASES = {
    "delay": "ts_delay", "delta": "ts_delta", "ts_std": "ts_std_dev", "stddev": "ts_std_dev", "std": "ts_std_dev",
    "correlation": "ts_corr", "covariance": "ts_covariance", "decay_linear": "ts_decay_linear", "sum": "ts_sum",
    "product": "ts_product", "clip": "clamp", "demean": "normalize", "ts_skew": "ts_skewness", "neg": "reverse",
}
# Operators whose cross-section is restricted to the simulation universe.
MEMBERS = {"rank", "zscore", "normalize", "scale", "winsorize", "vector_neut", "group_neutralize", "group_mean",
           "group_zscore", "group_rank", "bucket"}
KEYWORD_RENAMES = {"std": "std", "usestd": "useStd", "range": "range", "buckets": "buckets"}


def operator_names() -> list[str]:
    return sorted({*REGISTRY, *ALIASES, "add", "subtract", "multiply", "divide", "and", "or", "not"})


class Evaluator:
    def __init__(self, context: Context):
        self.context = context
        self.variables: dict[str, Any] = {}
        self.memo: dict[Any, Any] = {}

    def run(self, program: Program) -> np.ndarray:
        for name, value in program.assignments:
            if name in REGISTRY or name in ALIASES:
                raise ExpressionError(f"{name!r} is an operator name and cannot be assigned")
            self.variables[name] = self.value(value)
            # The RHS reads the old bindings; cached AST values expire once a binding changes.
            self.memo.clear()
        result = self.value(program.result)
        if not isinstance(result, np.ndarray) or result.shape != self.context.shape:
            raise ExpressionError("the alpha must be a data series (it evaluates to a constant)")
        return result

    def value(self, node: Any) -> Any:
        key = node
        if key in self.memo:
            return self.memo[key]
        out = self._value(node)
        self.memo[key] = out
        return out

    def series(self, node: Any, where: str) -> Any:
        value = self.value(node)
        if isinstance(value, str):
            raise ExpressionError(f"{where}: expected a series or number, got the text {value!r}")
        return value

    def _value(self, node: Any) -> Any:
        if isinstance(node, Num):
            return node.value
        if isinstance(node, Str):
            return node.value
        if isinstance(node, Name):
            if node.name in self.variables:
                return self.variables[node.name]
            if node.name in GROUPS:
                raise ExpressionError(f"{node.name!r} is a group; use it inside a group_* operator")
            return self.context.field(node.name)
        if isinstance(node, Unary):
            operand = self.series(node.operand, f"'{node.op}'")
            if node.op == "-":
                return ops._binary("-", 0.0, operand) if isinstance(operand, np.ndarray) else -operand
            if node.op == "+":
                return operand
            return ops._binary("==", operand, 0.0) if isinstance(operand, np.ndarray) else float(operand == 0)
        if isinstance(node, Binary):
            left, right = self.series(node.left, node.op), self.series(node.right, node.op)
            if not isinstance(left, np.ndarray) and not isinstance(right, np.ndarray):
                return float(ops._binary(node.op, np.asarray([left]), np.asarray([right]))[0])
            return ops._binary(node.op, left, right)
        if isinstance(node, Ternary):
            return ops.if_else(self.series(node.condition, "?:"), self.series(node.then, "?:"),
                               self.series(node.otherwise, "?:"))
        if isinstance(node, Call):
            return self.call(node)
        raise ExpressionError(f"cannot evaluate {node!r}")

    def call(self, node: Call) -> Any:
        name = ALIASES.get(node.name, node.name)
        if name in ("add", "subtract", "multiply", "divide", "and", "or"):
            op = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "and": "&&", "or": "||"}[name]
            if len(node.args) < 2 or node.kwargs:
                raise ExpressionError(f"{node.name}() takes two or more series")
            result = self.series(node.args[0], node.name)
            for arg in node.args[1:]:
                result = ops._binary(op, result, self.series(arg, node.name))
            return result
        if name == "not":
            if len(node.args) != 1:
                raise ExpressionError("not() takes one series")
            return ops._binary("==", self.series(node.args[0], "not"), 0.0)
        if name not in REGISTRY:
            raise ExpressionError(f"unknown operator {node.name}(); see the operator list")
        function, kinds, defaults = REGISTRY[name]
        args: list[Any] = []
        if kinds == ("*",):
            if len(node.args) < 2:
                raise ExpressionError(f"{node.name}() takes two or more series")
            values = [self.series(a, node.name) for a in node.args]
            shape = next((v.shape for v in values if isinstance(v, np.ndarray)), None)
            if shape is None:
                raise ExpressionError(f"{node.name}() needs at least one data series")
            return function(*[v if isinstance(v, np.ndarray) else np.full(shape, v, dtype=np.float32) for v in values])
        # positional arguments may also be given by keyword for the numeric ones (e.g. d=20)
        given = list(node.args)
        extra = dict(node.kwargs)
        if len(given) > len(kinds) + len(defaults):
            raise ExpressionError(f"{node.name}() takes at most {len(kinds) + len(defaults)} arguments")
        keywords = {}
        for position, kind in enumerate(kinds):
            if position < len(given):
                arg = given[position]
            else:
                key = {"n": "d"}.get(kind)
                if key and key in extra:
                    arg = extra.pop(key)
                else:
                    raise ExpressionError(f"{node.name}() is missing argument {position + 1}")
            if kind == "g":
                if not isinstance(arg, Name):
                    group = self.series(arg, node.name)
                    if not isinstance(group, np.ndarray):
                        raise ExpressionError(f"{node.name}(): the group must be a group name or bucket(...)")
                    args.append(group)
                elif arg.name in self.variables:
                    args.append(self.variables[arg.name])
                else:
                    args.append(self.context.group(arg.name))
            elif kind == "n":
                args.append(_scalar(self.series(arg, node.name), f"{node.name}() argument {position + 1}"))
            else:
                value = self.series(arg, node.name)
                args.append(value if isinstance(value, np.ndarray)
                            else np.full(self.context.shape, value, dtype=np.float32))
        # remaining positional arguments map onto the keyword parameters in order
        for position, (key, _default) in enumerate(defaults.items()):
            index = len(kinds) + position
            if index < len(given):
                keywords[key] = self._option(given[index], node.name)
        lowered = {key.lower(): key for key in defaults}
        for key, arg in extra.items():
            canonical = lowered.get(key.lower())
            if canonical is None:
                raise ExpressionError(f"{node.name}() has no parameter {key!r}")
            keywords[canonical] = self._option(arg, node.name)
        if name in MEMBERS:
            keywords["members"] = self.context.members
        return function(*args, **keywords)

    def _option(self, node: Any, where: str) -> Any:
        value = self.value(node)
        if isinstance(value, np.ndarray):
            raise ExpressionError(f"{where}(): options must be numbers, text or true/false")
        return value


def check_expression(text: str, fields: set[str] | None = None) -> list[str]:
    """Check syntax, call signatures and names without evaluating data."""
    from qlib_yahoo.alphalab.fastexpr import names

    try:
        program = parse(text)
    except ExpressionError as error:
        return [str(error)]
    known_fields = set(fields or ()) | {*BASE_FIELDS, *OPTIONAL_FIELDS, "returns", "vwap", "dollar_volume"}
    assigned = {name for name, _ in program.assignments}
    problems = []

    def walk(node: Any) -> None:
        if isinstance(node, Call):
            name = ALIASES.get(node.name, node.name)
            if name not in REGISTRY and name not in ("add", "subtract", "multiply", "divide", "and", "or", "not"):
                problems.append(f"unknown operator {node.name}()")
            else:
                problem = _call_signature_problem(node, name)
                if problem:
                    problems.append(problem)
            for arg in node.args:
                walk(arg)
            for _, arg in node.kwargs:
                walk(arg)
        elif isinstance(node, (Unary,)):
            walk(node.operand)
        elif isinstance(node, Binary):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, Ternary):
            walk(node.condition)
            walk(node.then)
            walk(node.otherwise)

    for _, value in program.assignments:
        walk(value)
    walk(program.result)
    for name in sorted(names(program) - assigned - known_fields - set(GROUPS) - {"true", "false"}):
        if not ADV.fullmatch(name):
            problems.append(f"unknown field {name!r}")
    return problems


def _call_signature_problem(node: Call, name: str) -> str | None:
    """Use the evaluator's public argument metadata, not the array function signature.

    Cross-sectional functions receive an implicit members mask. Required series
    and groups are positional; required numbers also accept ``d=``. Extra
    positional values map to defaults in insertion order, and option keywords
    are case-insensitive. Positional options may be overridden by keywords, as
    in Evaluator.call. No arrays, values or operator functions are evaluated.
    """
    if name == "not":
        if len(node.args) != 1 or node.kwargs:
            return f"{node.name}() takes one series and no keywords"
        return None
    if name in ("add", "subtract", "multiply", "divide", "and", "or") or REGISTRY[name][1] == ("*",):
        if len(node.args) < 2 or node.kwargs:
            return f"{node.name}() takes two or more series and no keywords"
        return None
    _, kinds, defaults = REGISTRY[name]
    if len(node.args) > len(kinds) + len(defaults):
        return f"{node.name}() takes at most {len(kinds) + len(defaults)} arguments"
    extra = dict(node.kwargs)
    for position, kind in enumerate(kinds):
        if position < len(node.args):
            continue
        if kind == "n" and "d" in extra:
            extra.pop("d")
        else:
            return f"{node.name}() is missing argument {position + 1}"
    allowed = {key.lower() for key in defaults}
    for key in extra:
        if key.lower() not in allowed:
            return f"{node.name}() has no parameter {key!r}"
    return None


def evaluate(expression: str, context: Context) -> np.ndarray:
    """Parse and evaluate ``expression``; the result has the context's (dates, instruments) shape."""
    program = parse(expression)
    evaluator = Evaluator(context)
    # true/false are accepted as option values
    evaluator.variables.update({"true": 1.0, "false": 0.0})
    return evaluator.run(program)
