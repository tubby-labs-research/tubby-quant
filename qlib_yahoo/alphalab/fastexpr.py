"""Parser for WorldQuant-style FASTEXPR alpha expressions.

Grammar (lowest to highest precedence)::

    program    := statement (';' statement)* [';']
    statement  := NAME '=' expr | expr          # the last statement is the alpha
    expr       := or ['?' expr ':' expr]
    or         := and ('||' and)*
    and        := compare ('&&' compare)*
    compare    := sum (('=='|'!='|'<'|'<='|'>'|'>=') sum)*
    sum        := product (('+'|'-') product)*
    product    := power (('*'|'/') power)*
    power      := unary ['^' power]
    unary      := ('-'|'+'|'!') unary | atom
    atom       := NUMBER | STRING | NAME | NAME '(' args ')' | '(' expr ')'
    args       := [arg (',' arg)*],  arg := [NAME '='] expr

Nothing is evaluated here and no Python code is ever executed: the result is a
small tree that the evaluator interprets with whitelisted operators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from qlib_yahoo.dataset import DatasetError


class ExpressionError(DatasetError):
    """A user-facing problem with an alpha expression, with its position."""


@dataclass(frozen=True)
class Num:
    value: float


@dataclass(frozen=True)
class Str:
    value: str


@dataclass(frozen=True)
class Name:
    name: str


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...] = field(default=())


@dataclass(frozen=True)
class Unary:
    op: str
    operand: Any


@dataclass(frozen=True)
class Binary:
    op: str
    left: Any
    right: Any


@dataclass(frozen=True)
class Ternary:
    condition: Any
    then: Any
    otherwise: Any


@dataclass(frozen=True)
class Program:
    assignments: tuple[tuple[str, Any], ...]
    result: Any


TOKEN = re.compile(r"""
    (?P<space>\s+)
  | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)
  | (?P<name>[A-Za-z_][A-Za-z0-9_.]*)
  | (?P<string>"[^"]*"|'[^']*')
  | (?P<op>\|\||&&|==|!=|<=|>=|[-+*/^<>!?:(),;=])
""", re.VERBOSE)
MAX_LENGTH = 10_000
MAX_DEPTH = 60


def tokenize(text: str) -> list[tuple[str, str, int]]:
    if len(text) > MAX_LENGTH:
        raise ExpressionError(f"expression is longer than {MAX_LENGTH} characters")
    tokens, position = [], 0
    while position < len(text):
        match = TOKEN.match(text, position)
        if not match:
            raise ExpressionError(f"unexpected character {text[position]!r} at position {position + 1}")
        kind = match.lastgroup
        if kind != "space":
            tokens.append((kind, match.group(), position))
        position = match.end()
    tokens.append(("end", "", len(text)))
    return tokens


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = tokenize(text)
        self.index = 0
        self.depth = 0

    def peek(self, offset: int = 0) -> tuple[str, str, int]:
        return self.tokens[min(self.index + offset, len(self.tokens) - 1)]

    def take(self) -> tuple[str, str, int]:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def expect(self, value: str) -> None:
        kind, text, position = self.take()
        if text != value or kind not in ("op",):
            found = "the end" if kind == "end" else repr(text)
            raise ExpressionError(f"expected {value!r} at position {position + 1}, found {found}")

    def at(self, *values: str) -> bool:
        kind, text, _ = self.peek()
        return kind == "op" and text in values

    def program(self) -> Program:
        assignments: list[tuple[str, Any]] = []
        result = None
        while True:
            if self.peek()[0] == "end":
                break
            if self.peek()[0] == "name" and self.peek(1)[1] == "=" and self.peek(1)[0] == "op":
                name = self.take()[1]
                self.take()
                assignments.append((name, self.expr()))
                result = None
            else:
                result = self.expr()
            if self.at(";"):
                self.take()
                continue
            if self.peek()[0] != "end":
                kind, text, position = self.peek()
                raise ExpressionError(f"unexpected {text!r} at position {position + 1}")
        if result is None:
            if not assignments:
                raise ExpressionError("the expression is empty")
            name, value = assignments.pop()  # "a = ...;" as the last statement: the alpha is a
            result = value
        return Program(tuple(assignments), result)

    def expr(self) -> Any:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ExpressionError("expression is nested too deeply")
        try:
            condition = self.binary(0)
            if self.at("?"):
                self.take()
                then = self.expr()
                self.expect(":")
                return Ternary(condition, then, self.expr())
            return condition
        finally:
            self.depth -= 1

    LEVELS = (("||",), ("&&",), ("==", "!=", "<", "<=", ">", ">="), ("+", "-"), ("*", "/"))

    def binary(self, level: int) -> Any:
        if level == len(self.LEVELS):
            return self.power()
        left = self.binary(level + 1)
        while self.at(*self.LEVELS[level]):
            op = self.take()[1]
            left = Binary(op, left, self.binary(level + 1))
        return left

    def power(self) -> Any:
        base = self.unary()
        if self.at("^"):
            self.take()
            return Binary("^", base, self.power())
        return base

    def unary(self) -> Any:
        if self.at("-", "+", "!"):
            op = self.take()[1]
            return Unary(op, self.unary())
        return self.atom()

    def atom(self) -> Any:
        kind, text, position = self.take()
        if kind == "number":
            return Num(float(text))
        if kind == "string":
            return Str(text[1:-1])
        if kind == "name":
            if self.at("("):
                self.take()
                args, kwargs = [], []
                if not self.at(")"):
                    while True:
                        if self.peek()[0] == "name" and self.peek(1)[1] == "=" and self.peek(1)[0] == "op":
                            key = self.take()[1]
                            self.take()
                            kwargs.append((key, self.expr()))
                        else:
                            if kwargs:
                                raise ExpressionError(f"positional argument after keyword in {text}()")
                            args.append(self.expr())
                        if self.at(","):
                            self.take()
                            continue
                        break
                self.expect(")")
                return Call(text.lower(), tuple(args), tuple(kwargs))
            return Name(text.lower())
        if kind == "op" and text == "(":
            inner = self.expr()
            self.expect(")")
            return inner
        found = "the end of the expression" if kind == "end" else repr(text)
        raise ExpressionError(f"unexpected {found} at position {position + 1}")


def parse(text: str) -> Program:
    """Parse FASTEXPR ``text`` into a Program; raise ExpressionError with a position on mistakes."""
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("the expression is empty")
    try:
        return _Parser(text).program()
    except RecursionError:
        raise ExpressionError("expression is nested too deeply") from None


def names(node: Any) -> set[str]:
    """Every bare name the tree refers to (fields, groups or assigned variables)."""
    if isinstance(node, Program):
        found = set()
        for _, value in node.assignments:
            found |= names(value)
        return found | names(node.result)
    if isinstance(node, Name):
        return {node.name}
    if isinstance(node, Call):
        found = set()
        for arg in node.args:
            found |= names(arg)
        for _, value in node.kwargs:
            found |= names(value)
        return found
    if isinstance(node, Unary):
        return names(node.operand)
    if isinstance(node, Binary):
        return names(node.left) | names(node.right)
    if isinstance(node, Ternary):
        return names(node.condition) | names(node.then) | names(node.otherwise)
    return set()
