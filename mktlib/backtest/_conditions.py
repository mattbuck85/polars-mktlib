from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from mktlib.backtest._types import TradeSide


def _ref(b: str | float | PriceExpr) -> pl.Expr:
    """Resolve a column name, literal, or PriceExpr to a Polars expression."""
    if isinstance(b, PriceExpr):
        return b.resolve()
    return pl.col(b) if isinstance(b, str) else pl.lit(b)


# ---------------------------------------------------------------------------
# Composable price expressions
# ---------------------------------------------------------------------------


class PriceExpr:
    """Base for composable price expressions that resolve to ``pl.Expr``."""

    __slots__ = ()

    def resolve(self) -> pl.Expr:
        raise NotImplementedError

    # --- arithmetic operators ---

    def __add__(self, other: PriceExpr | float | int) -> _BinOp:
        return _BinOp(self, _coerce(other), "+")

    def __radd__(self, other: float | int) -> _BinOp:
        return _BinOp(_coerce(other), self, "+")

    def __sub__(self, other: PriceExpr | float | int) -> _BinOp:
        return _BinOp(self, _coerce(other), "-")

    def __rsub__(self, other: float | int) -> _BinOp:
        return _BinOp(_coerce(other), self, "-")

    def __mul__(self, other: PriceExpr | float | int) -> _BinOp:
        return _BinOp(self, _coerce(other), "*")

    def __rmul__(self, other: float | int) -> _BinOp:
        return _BinOp(_coerce(other), self, "*")

    def __truediv__(self, other: PriceExpr | float | int) -> _BinOp:
        return _BinOp(self, _coerce(other), "/")

    def __rtruediv__(self, other: float | int) -> _BinOp:
        return _BinOp(_coerce(other), self, "/")

    def __mod__(self, other: PriceExpr | float | int) -> _BinOp:
        return _BinOp(self, _coerce(other), "%")

    def __rmod__(self, other: float | int) -> _BinOp:
        return _BinOp(_coerce(other), self, "%")

    def __neg__(self) -> _BinOp:
        return _BinOp(Lit(0.0), self, "-")


@dataclass(frozen=True, slots=True)
class Col(PriceExpr):
    """Column reference — resolves to ``pl.col(name)``."""

    name: str

    def resolve(self) -> pl.Expr:
        return pl.col(self.name)


@dataclass(frozen=True, slots=True)
class Lit(PriceExpr):
    """Literal constant — resolves to ``pl.lit(value)``."""

    value: float

    def resolve(self) -> pl.Expr:
        return pl.lit(self.value)


@dataclass(frozen=True, slots=True)
class _BinOp(PriceExpr):
    """Binary arithmetic node (internal)."""

    left: PriceExpr
    right: PriceExpr
    op: str

    def resolve(self) -> pl.Expr:
        l = self.left.resolve()
        r = self.right.resolve()
        match self.op:
            case "+":
                return l + r
            case "-":
                return l - r
            case "*":
                return l * r
            case "/":
                return l / r
            case "%":
                return l % r
            case _:  # pragma: no cover
                msg = f"Unknown op: {self.op}"
                raise ValueError(msg)


def _coerce(v: PriceExpr | float | int) -> PriceExpr:
    """Wrap a raw numeric into ``Lit`` if needed."""
    if isinstance(v, PriceExpr):
        return v
    return Lit(float(v))


def _coerce_base(v: PriceExpr | str | float) -> PriceExpr:
    """Coerce a base price to PriceExpr: str -> Col, float -> Lit."""
    if isinstance(v, PriceExpr):
        return v
    if isinstance(v, str):
        return Col(v)
    return Lit(float(v))


@dataclass(frozen=True, slots=True)
class Pct(PriceExpr):
    """Price offset by ``pct``% from ``base``.

    Positive ``pct`` -> above, negative -> below.

    ``Pct("close", 1.0)``  -> ``close * 1.01``  (1% above)
    ``Pct("close", -0.5)`` -> ``close * 0.995`` (0.5% below)
    """

    base: PriceExpr | str | float
    pct: float

    def resolve(self) -> pl.Expr:
        return _coerce_base(self.base).resolve() * (1.0 + self.pct / 100.0)


@dataclass(frozen=True, slots=True)
class EntryRef(PriceExpr):
    """Column value snapshotted at the entry signal bar, forward-filled.

    The engine creates ``_entry_{col}`` columns automatically when it
    detects ``EntryRef`` nodes in the exit condition tree.

    ``EntryRef("close")`` resolves to ``pl.col("_entry_close")``.
    """

    col: str

    def resolve(self) -> pl.Expr:
        return pl.col(f"_entry_{self.col}")


class Condition:
    """Base class for signal conditions that resolve to boolean ``pl.Expr``."""

    trade_side: TradeSide | None = None

    def resolve(self) -> pl.Expr:
        raise NotImplementedError

    def __and__(self, other: Condition) -> All:
        return All(self, other)

    def __or__(self, other: Condition) -> Any_:
        return Any_(self, other)

    def __invert__(self) -> Not:
        return Not(self)


@dataclass(frozen=True, slots=True)
class Crossover(Condition):
    """``a`` crosses above ``b`` (column name or constant)."""

    a: str
    b: str | float
    trade_side: TradeSide | None = None

    def resolve(self) -> pl.Expr:
        ref = _ref(self.b)
        prev_ref = ref.shift(1) if isinstance(self.b, str) else ref
        return (pl.col(self.a) > ref) & (pl.col(self.a).shift(1) <= prev_ref)


@dataclass(frozen=True, slots=True)
class Crossunder(Condition):
    """``a`` crosses below ``b`` (column name or constant)."""

    a: str
    b: str | float
    trade_side: TradeSide | None = None

    def resolve(self) -> pl.Expr:
        ref = _ref(self.b)
        prev_ref = ref.shift(1) if isinstance(self.b, str) else ref
        return (pl.col(self.a) < ref) & (pl.col(self.a).shift(1) >= prev_ref)


@dataclass(frozen=True, slots=True)
class PriceIsAbove(Condition):
    """``a > b`` (column name, constant, or PriceExpr)."""

    a: str | PriceExpr
    b: str | float | PriceExpr
    trade_side: TradeSide | None = None

    def resolve(self) -> pl.Expr:
        return _ref(self.a) > _ref(self.b)


@dataclass(frozen=True, slots=True)
class PriceIsBelow(Condition):
    """``a < b`` (column name, constant, or PriceExpr)."""

    a: str | PriceExpr
    b: str | float | PriceExpr
    trade_side: TradeSide | None = None

    def resolve(self) -> pl.Expr:
        return _ref(self.a) < _ref(self.b)


@dataclass(frozen=True, slots=True)
class IsRising(Condition):
    """Column value is greater than its value ``period`` bars ago."""

    col: str
    period: int = 1
    trade_side: TradeSide | None = None

    def resolve(self) -> pl.Expr:
        return pl.col(self.col) > pl.col(self.col).shift(self.period)


@dataclass(frozen=True, slots=True)
class IsFalling(Condition):
    """Column value is less than its value ``period`` bars ago."""

    col: str
    period: int = 1
    trade_side: TradeSide | None = None

    def resolve(self) -> pl.Expr:
        return pl.col(self.col) < pl.col(self.col).shift(self.period)


@dataclass(frozen=True, slots=True)
class Custom(Condition):
    """User-supplied polars expression — must evaluate to a boolean column."""

    expr: pl.Expr
    trade_side: TradeSide | None = None

    def resolve(self) -> pl.Expr:
        return self.expr


# --- Combinators ---


@dataclass(frozen=True, slots=True)
class All(Condition):
    """Both conditions must be true (``a & b``)."""

    left: Condition
    right: Condition
    trade_side: TradeSide | None = None

    def resolve(self) -> pl.Expr:
        return self.left.resolve() & self.right.resolve()


@dataclass(frozen=True, slots=True)
class Any_(Condition):
    """Either condition is true (``a | b``)."""

    left: Condition
    right: Condition
    trade_side: TradeSide | None = None

    def resolve(self) -> pl.Expr:
        return self.left.resolve() | self.right.resolve()


@dataclass(frozen=True, slots=True)
class Not(Condition):
    """Invert a condition (``~a``)."""

    inner: Condition
    trade_side: TradeSide | None = None

    def resolve(self) -> pl.Expr:
        return ~self.inner.resolve()
