from __future__ import annotations

from dataclasses import dataclass

import polars as pl


def _ref(b: str | float) -> pl.Expr:
    """Resolve a column name or literal to a Polars expression."""
    return pl.col(b) if isinstance(b, str) else pl.lit(b)


class Condition:
    """Base class for signal conditions that resolve to boolean ``pl.Expr``."""

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

    def resolve(self) -> pl.Expr:
        ref = _ref(self.b)
        prev_ref = ref.shift(1) if isinstance(self.b, str) else ref
        return (pl.col(self.a) > ref) & (pl.col(self.a).shift(1) <= prev_ref)


@dataclass(frozen=True, slots=True)
class Crossunder(Condition):
    """``a`` crosses below ``b`` (column name or constant)."""

    a: str
    b: str | float

    def resolve(self) -> pl.Expr:
        ref = _ref(self.b)
        prev_ref = ref.shift(1) if isinstance(self.b, str) else ref
        return (pl.col(self.a) < ref) & (pl.col(self.a).shift(1) >= prev_ref)


@dataclass(frozen=True, slots=True)
class PriceIsAbove(Condition):
    """``a > b`` (column name or constant)."""

    a: str
    b: str | float

    def resolve(self) -> pl.Expr:
        return pl.col(self.a) > _ref(self.b)


@dataclass(frozen=True, slots=True)
class PriceIsBelow(Condition):
    """``a < b`` (column name or constant)."""

    a: str
    b: str | float

    def resolve(self) -> pl.Expr:
        return pl.col(self.a) < _ref(self.b)


@dataclass(frozen=True, slots=True)
class IsRising(Condition):
    """Column value is greater than its value ``period`` bars ago."""

    col: str
    period: int = 1

    def resolve(self) -> pl.Expr:
        return pl.col(self.col) > pl.col(self.col).shift(self.period)


@dataclass(frozen=True, slots=True)
class IsFalling(Condition):
    """Column value is less than its value ``period`` bars ago."""

    col: str
    period: int = 1

    def resolve(self) -> pl.Expr:
        return pl.col(self.col) < pl.col(self.col).shift(self.period)


# --- Combinators ---


@dataclass(frozen=True, slots=True)
class All(Condition):
    """Both conditions must be true (``a & b``)."""

    left: Condition
    right: Condition

    def resolve(self) -> pl.Expr:
        return self.left.resolve() & self.right.resolve()


@dataclass(frozen=True, slots=True)
class Any_(Condition):
    """Either condition is true (``a | b``)."""

    left: Condition
    right: Condition

    def resolve(self) -> pl.Expr:
        return self.left.resolve() | self.right.resolve()


@dataclass(frozen=True, slots=True)
class Not(Condition):
    """Invert a condition (``~a``)."""

    inner: Condition

    def resolve(self) -> pl.Expr:
        return ~self.inner.resolve()
