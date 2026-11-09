"""Typed FASTEXPR expression trees for search: generation, mutation, crossover, rendering.

Search operates on a small, unit-aware subset of FASTEXPR so that generated
candidates are meaningful more often than random strings would be. Every node
has a unit:

    P price level   V share volume   D dollar amount   L log level
    R ratio / return (dimensionless) S cross-sectional or time-series score

Rules reject combinations that are almost never sensible (adding a price to a
volume, dividing by a return that can be zero) and the root must be
dimensionless (R or S). Rendering produces ordinary FASTEXPR text that the
Alpha Lab parser reads back; :func:`from_text` converts supported FASTEXPR into
trees so seeds, templates and agent submissions can be mutated too.

The canonical key is sign-insensitive: an expression and its negation are the
same candidate, because the screen orients each signal on development data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import numpy as np

FIELD_UNITS = {"open": "P", "high": "P", "low": "P", "close": "P", "vwap": "P",
               "volume": "V", "dollar_volume": "D", "adv20": "D", "returns": "R"}
DIMENSIONLESS = frozenset({"R", "S"})
LEVELS = frozenset({"P", "V", "D"})

TS_SAME = ("ts_mean", "ts_delay", "ts_delta", "ts_max", "ts_min", "ts_std_dev", "ts_decay_linear", "ts_sum")
TS_SCORE = ("ts_rank", "ts_zscore", "ts_arg_max", "ts_arg_min")
CS_SCORE = ("rank", "zscore")
UNARY_SAME = ("abs",)
BINARY = ("add", "sub", "mul", "div")
INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
PAIR_TS = ("ts_corr",)
GROUP_OPS = ("group_neutralize", "group_rank")
ALIASES = {"delay": "ts_delay", "delta": "ts_delta", "ts_std": "ts_std_dev", "stddev": "ts_std_dev",
           "std": "ts_std_dev", "correlation": "ts_corr", "decay_linear": "ts_decay_linear", "sum": "ts_sum",
           "reverse": "neg", "neg": "neg"}
DEFAULT_WINDOWS = (2, 3, 5, 10, 20, 40, 60)


@dataclass(frozen=True)
class Node:
    op: str                      # "field", "const", "neg", "log", "signed_power" or an operator name
    args: tuple = ()
    value: Any = None            # field name, constant, window, exponent or group name


# --------------------------------------------------------------------- units
def unit(node: Node) -> str | None:
    """The node's unit, or None when the combination is invalid."""
    op = node.op
    if op == "field":
        return FIELD_UNITS.get(node.value)
    if op == "const":
        return "R"
    child = [unit(a) for a in node.args]
    if any(u is None for u in child):
        return None
    if op == "neg" or op in UNARY_SAME:
        return child[0]
    if op in CS_SCORE or op in TS_SCORE or op in GROUP_OPS and op == "group_rank":
        return "S"
    if op == "group_neutralize":
        return child[0]
    if op == "log":
        return "L" if child[0] in LEVELS else None
    if op == "signed_power":
        return child[0] if child[0] in DIMENSIONLESS else None
    if op in TS_SAME:
        if op in ("ts_delta", "ts_std_dev") and child[0] == "L":
            return "R"
        return child[0]
    if op in PAIR_TS:
        return None if node.args[0] == node.args[1] else "S"
    a, b = child
    if op in ("add", "sub"):
        if node.args[0] == node.args[1]:
            return None  # x - x and x + x carry no signal beyond x
        if a == b:
            return "R" if a == "L" else a
        if a in DIMENSIONLESS and b in DIMENSIONLESS:
            return "R"
        return None
    if op == "mul":
        if a in DIMENSIONLESS and b in DIMENSIONLESS:
            return "S" if a == b == "S" else "R"
        if a in DIMENSIONLESS:
            return b
        if b in DIMENSIONLESS:
            return a
        return None
    if op == "div":
        if node.args[0] == node.args[1] or a != b:
            return None
        positive = a in LEVELS or node.args[1].op == "ts_std_dev"
        return "R" if positive else None
    return None


def size(node: Node) -> int:
    return 1 + sum(size(a) for a in node.args)


def depth(node: Node) -> int:
    return 1 + max((depth(a) for a in node.args), default=0)


def windows_of(node: Node) -> list[int]:
    out = [node.value] if node.op in TS_SAME + TS_SCORE + PAIR_TS else []
    for arg in node.args:
        out += windows_of(arg)
    return out


def max_window(node: Node) -> int:
    """Longest lookback in sessions (nested windows add up)."""
    own = node.value if node.op in TS_SAME + TS_SCORE + PAIR_TS else 0
    if node.op == "field" and node.value == "adv20":
        own = 20
    return own + max((max_window(a) for a in node.args), default=0)


# ----------------------------------------------------------------- rendering
def _number(value: float) -> str:
    """Exact round-trip text: a rendered constant parses back to the same float."""
    value = float(value)
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def to_text(node: Node) -> str:
    op = node.op
    if op == "field":
        return node.value
    if op == "const":
        return _number(node.value)
    if op == "neg":
        inner = to_text(node.args[0])
        return "-" + inner  # binary operands are already parenthesised by to_text
    if op in BINARY:
        return f"({to_text(node.args[0])} {INFIX[op]} {to_text(node.args[1])})"
    if op in TS_SAME + TS_SCORE:
        return f"{op}({to_text(node.args[0])}, {int(node.value)})"
    if op in PAIR_TS:
        return f"{op}({to_text(node.args[0])}, {to_text(node.args[1])}, {int(node.value)})"
    if op == "signed_power":
        return f"signed_power({to_text(node.args[0])}, {_number(node.value)})"
    if op in GROUP_OPS:
        return f"{op}({to_text(node.args[0])}, {node.value})"
    return f"{op}({to_text(node.args[0])})"


def strip_sign(node: Node) -> tuple[Node, int]:
    sign = 1
    while node.op == "neg":
        node, sign = node.args[0], -sign
    return node, sign


def negate(node: Node) -> Node:
    base, sign = strip_sign(node)
    return base if sign < 0 else Node("neg", (base,))


def canonical_key(node: Node) -> str:
    """Sign-insensitive identity from the parsed Alpha Lab AST of the rendered text."""
    from qlib_yahoo.alphalab.fastexpr import parse

    base, _ = strip_sign(node)
    program = parse(to_text(base))

    def walk(value):
        if is_dataclass(value):
            return {"type": type(value).__name__, **{f.name: walk(getattr(value, f.name)) for f in fields(value)}}
        if isinstance(value, tuple):
            return [walk(item) for item in value]
        return value

    body = json.dumps(walk(program), sort_keys=True, separators=(",", ":"))
    return "alpha_" + hashlib.sha256(body.encode()).hexdigest()[:40]


# ------------------------------------------------------------------- parsing
SUPPORTED_CALLS = set(TS_SAME + TS_SCORE + CS_SCORE + UNARY_SAME + PAIR_TS + GROUP_OPS + ("log", "signed_power", "neg"))


def from_text(text: str) -> Node:
    """Convert supported FASTEXPR into a tree; raise ValueError for anything outside the search subset."""
    from qlib_yahoo.alphalab.fastexpr import Binary, Call, ExpressionError, Name, Num, Unary, parse

    try:
        program = parse(text)
    except ExpressionError as error:  # a RuntimeError subclass; callers expect ValueError
        raise ValueError(str(error)) from None
    if program.assignments:
        raise ValueError("assignments are outside the search subset")

    def number(value) -> float:
        if isinstance(value, Num):
            return float(value.value)
        if isinstance(value, Unary) and value.op == "-" and isinstance(value.operand, Num):
            return -float(value.operand.value)
        raise ValueError("expected a number")

    def window(value) -> int:
        result = number(value)
        if not result.is_integer() or result < 1:
            raise ValueError("windows must be positive integers")
        return int(result)

    def convert(value) -> Node:
        if isinstance(value, Num):
            return Node("const", (), float(value.value))
        if isinstance(value, Name):
            if value.name not in FIELD_UNITS:
                raise ValueError(f"field {value.name!r} is outside the search subset")
            return Node("field", (), value.name)
        if isinstance(value, Unary):
            if value.op == "-":
                return negate(convert(value.operand))
            if value.op == "+":
                return convert(value.operand)
            raise ValueError("logical operators are outside the search subset")
        if isinstance(value, Binary):
            ops = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
            if value.op not in ops:
                raise ValueError(f"operator {value.op!r} is outside the search subset")
            return Node(ops[value.op], (convert(value.left), convert(value.right)))
        if isinstance(value, Call):
            name = ALIASES.get(value.name, value.name)
            if value.kwargs:
                raise ValueError("keyword options are outside the search subset")
            args = value.args
            if name == "neg" and len(args) == 1:
                return negate(convert(args[0]))
            if name in ("add", "multiply") and len(args) >= 2:
                # n-ary sums and products (blends) fold into nested binary nodes
                op = "add" if name == "add" else "mul"
                node = convert(args[0])
                for arg in args[1:]:
                    node = Node(op, (node, convert(arg)))
                return node
            if name in ("subtract", "divide") and len(args) == 2:
                op = {"subtract": "sub", "divide": "div"}[name]
                return Node(op, (convert(args[0]), convert(args[1])))
            if name not in SUPPORTED_CALLS:
                raise ValueError(f"{value.name}() is outside the search subset")
            if name in TS_SAME + TS_SCORE and len(args) == 2:
                return Node(name, (convert(args[0]),), window(args[1]))
            if name in PAIR_TS and len(args) == 3:
                return Node(name, (convert(args[0]), convert(args[1])), window(args[2]))
            if name == "signed_power" and len(args) == 2:
                return Node(name, (convert(args[0]),), number(args[1]))
            if name in GROUP_OPS and len(args) == 2 and isinstance(args[1], Name):
                return Node(name, (convert(args[0]),), args[1].name)
            if name in CS_SCORE + UNARY_SAME + ("log",) and len(args) == 1:
                return Node(name, (convert(args[0]),))
            raise ValueError(f"{value.name}() has an unsupported argument shape")
        raise ValueError("unsupported expression element")

    return convert(program.result)


def text_size(text: str) -> int:
    """Node count of any parsable FASTEXPR (for complexity outside the search subset); 1 if unparsable."""
    from dataclasses import is_dataclass as _is_dc
    from qlib_yahoo.alphalab.fastexpr import ExpressionError, parse

    try:
        program = parse(text)
    except ExpressionError:
        return 1

    def count(value) -> int:
        if _is_dc(value):
            return 1 + sum(count(getattr(value, f.name)) for f in fields(value))
        if isinstance(value, tuple):
            return sum(count(item) for item in value)
        return 0

    return max(1, count(program.result) + sum(count(v) for _, v in program.assignments))


# ---------------------------------------------------------------- generation
@dataclass(frozen=True)
class Space:
    fields: tuple = ("open", "high", "low", "close", "vwap", "volume", "returns", "adv20")
    windows: tuple = DEFAULT_WINDOWS
    max_depth: int = 4
    max_size: int = 14
    max_lookback: int = 120
    groups: tuple = ()

    def admits(self, node: Node) -> bool:
        u = unit(node)
        return (u is not None and u in DIMENSIONLESS and depth(node) <= self.max_depth + 1
                and size(node) <= self.max_size and max_window(node) <= self.max_lookback)


def _terminal(rng, space: Space) -> Node:
    return Node("field", (), space.fields[int(rng.integers(len(space.fields)))])


def _window(rng, space: Space) -> int:
    return int(space.windows[int(rng.integers(len(space.windows)))])


def grow(rng, space: Space, budget: int) -> Node:
    """A random subtree of at most ``budget`` levels; units are checked by the caller."""
    if budget <= 1 or rng.random() < 0.18:
        return _terminal(rng, space)
    roll = rng.random()
    if roll < 0.30:
        return Node(TS_SAME[int(rng.integers(len(TS_SAME)))], (grow(rng, space, budget - 1),), _window(rng, space))
    if roll < 0.42:
        return Node(TS_SCORE[int(rng.integers(len(TS_SCORE)))], (grow(rng, space, budget - 1),), _window(rng, space))
    if roll < 0.55:
        return Node(CS_SCORE[int(rng.integers(len(CS_SCORE)))], (grow(rng, space, budget - 1),))
    if roll < 0.80:
        op = BINARY[int(rng.integers(len(BINARY)))]
        return Node(op, (grow(rng, space, budget - 1), grow(rng, space, budget - 1)))
    if roll < 0.88:
        return Node("ts_corr", (grow(rng, space, budget - 1), grow(rng, space, budget - 1)),
                    int(max(5, _window(rng, space))))
    if roll < 0.93:
        return Node("log", (grow(rng, space, budget - 1),))
    if roll < 0.96 and space.groups:
        return Node(GROUP_OPS[int(rng.integers(len(GROUP_OPS)))], (grow(rng, space, budget - 1),),
                    space.groups[int(rng.integers(len(space.groups)))])
    if roll < 0.98:
        return Node("signed_power", (grow(rng, space, budget - 1),), float(rng.choice([0.5, 2.0])))
    return Node("neg", (grow(rng, space, budget - 1),))


def orient_root(node: Node) -> Node:
    """Make the root dimensionless by ranking a level-valued tree."""
    u = unit(node)
    if u in DIMENSIONLESS:
        return node
    if u is not None:
        return Node("rank", (node,))
    return node


def random_tree(rng, space: Space, attempts: int = 60) -> Node | None:
    for _ in range(attempts):
        node = orient_root(grow(rng, space, int(rng.integers(2, space.max_depth + 1))))
        if space.admits(node) and node.op != "field":
            return node
    return None


def paths(node: Node, prefix: tuple = ()) -> list[tuple]:
    out = [prefix]
    for index, arg in enumerate(node.args):
        out += paths(arg, prefix + (index,))
    return out


def get(node: Node, path: tuple) -> Node:
    for index in path:
        node = node.args[index]
    return node


def replace(node: Node, path: tuple, new: Node) -> Node:
    if not path:
        return new
    args = list(node.args)
    args[path[0]] = replace(args[path[0]], path[1:], new)
    return Node(node.op, tuple(args), node.value)


def mutate(rng, node: Node, space: Space, attempts: int = 30) -> tuple[Node, str] | None:
    """One random structural edit that keeps the tree admissible, with the edit's name."""
    for _ in range(attempts):
        kind = str(rng.choice(["window", "point", "subtree", "wrap", "hoist", "shrink"],
                              p=[0.28, 0.22, 0.2, 0.15, 0.08, 0.07]))
        all_paths = paths(node)
        path = all_paths[int(rng.integers(len(all_paths)))]
        target = get(node, path)
        candidate = None
        if kind == "window":
            spots = [p for p in all_paths if get(node, p).op in TS_SAME + TS_SCORE + PAIR_TS]
            if spots:
                spot = spots[int(rng.integers(len(spots)))]
                old = get(node, spot)
                choices = [w for w in space.windows if w != old.value]
                if choices:
                    candidate = replace(node, spot, Node(old.op, old.args, int(rng.choice(choices))))
        elif kind == "point":
            for family in (TS_SAME, TS_SCORE, CS_SCORE, BINARY, tuple(space.fields)):
                if (target.op in family) or (target.op == "field" and target.value in family):
                    options = [x for x in family if x not in (target.op, target.value)]
                    if not options:
                        break
                    choice = options[int(rng.integers(len(options)))]
                    new = Node("field", (), choice) if target.op == "field" else Node(choice, target.args, target.value)
                    candidate = replace(node, path, new)
                    break
        elif kind == "subtree":
            candidate = replace(node, path, grow(rng, space, int(rng.integers(1, 3))))
        elif kind == "wrap":
            wrapper = str(rng.choice(["rank", "ts_decay_linear", "ts_mean", "ts_rank", "neg", "ts_zscore"]))
            if wrapper in ("rank", "neg"):
                new = Node(wrapper, (target,))
            else:
                new = Node(wrapper, (target,), _window(rng, space))
            candidate = replace(node, path, new)
        elif kind == "hoist" and path:
            candidate = target
        elif kind == "shrink" and target.args:
            candidate = replace(node, path, target.args[int(rng.integers(len(target.args)))])
        if candidate is not None:
            candidate = orient_root(candidate)
            if candidate != node and space.admits(candidate) and candidate.op != "field":
                return candidate, kind
    return None


def crossover(rng, first: Node, second: Node, space: Space, attempts: int = 30) -> Node | None:
    """Replace a random subtree of ``first`` with a unit-compatible subtree of ``second``."""
    donors = paths(second)
    receivers = paths(first)
    for _ in range(attempts):
        at = receivers[int(rng.integers(len(receivers)))]
        donor = get(second, donors[int(rng.integers(len(donors)))])
        if unit(donor) != unit(get(first, at)):
            continue
        child = orient_root(replace(first, at, donor))
        if child not in (first, second) and space.admits(child) and child.op != "field":
            return child
    return None


# ------------------------------------------------------------------- labels
def family(node: Node) -> str:
    """A coarse mechanism label from fields and operators, for reports and budgets."""
    tags: set[str] = set()

    def walk(n: Node) -> None:
        if n.op == "field":
            if n.value in ("volume", "adv20", "dollar_volume"):
                tags.add("volume")
            elif n.value in ("open", "high", "low"):
                tags.add("intraday")
            elif n.value == "vwap":
                tags.add("vwap")
            else:
                tags.add("price")
        elif n.op in ("ts_delta", "ts_delay") or (n.op == "field" and n.value == "returns"):
            tags.add("change")
        elif n.op == "ts_std_dev":
            tags.add("volatility")
        elif n.op == "ts_corr":
            tags.add("correlation")
        elif n.op in ("ts_rank", "ts_zscore", "ts_arg_max", "ts_arg_min", "ts_max", "ts_min"):
            tags.add("extremes")
        for arg in n.args:
            walk(arg)

    walk(node)
    return "+".join(sorted(tags)) or "other"


TEMPLATES = (
    ("short-term reversal", "-rank(ts_delta(close, {n}))", (3, 5, 10)),
    ("medium-term momentum", "rank(ts_delta(close, {n}))", (20, 40, 60)),
    ("risk-adjusted momentum", "rank(ts_mean(returns, {n}) / ts_std_dev(returns, {n}))", (20, 40, 60)),
    ("low volatility", "-rank(ts_std_dev(returns, {n}))", (10, 20, 60)),
    ("abnormal volume", "rank(volume / ts_mean(volume, {n}))", (10, 20, 40)),
    ("price-volume divergence", "-ts_corr(rank(close), rank(volume), {n})", (5, 10, 20)),
    ("close location in range", "rank((close - low) / (high - low))", (1,)),
    ("vwap deviation", "-rank((close - vwap) / vwap)", (1,)),
    ("liquidity", "-rank(log(adv20))", (1,)),
    ("decayed reversal", "-rank(ts_decay_linear(returns, {n}))", (5, 10, 20)),
    ("range volatility", "-rank(ts_mean((high - low) / close, {n}))", (5, 10, 20)),
    ("lottery demand", "-rank(ts_max(returns, {n}))", (10, 20)),
    ("volume-confirmed move", "rank(ts_sum(returns * (volume / ts_mean(volume, 20)), {n}))", (3, 5, 10)),
    ("time-series rank of price", "-ts_rank(close, {n})", (10, 20, 40)),
)


def template_trees(rng=None, every_window: bool = False) -> list[tuple[str, Node]]:
    out = []
    for title, pattern, choices in TEMPLATES:
        options = choices if every_window or rng is None else (int(rng.choice(choices)),)
        for window in options:
            out.append((title, from_text(pattern.format(n=window))))
    return out


def finite_or_none(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def rng_for(seed: int):
    return np.random.default_rng(seed)
