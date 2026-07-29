from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from mktlib.backtest._types import TradeSide


def _ref(b: str | float | ColExpr) -> pl.Expr:
    """Resolve a column name, literal, or ColExpr to a Polars expression."""
    if isinstance(b, ColExpr):
        return b.resolve()
    return pl.col(b) if isinstance(b, str) else pl.lit(b)


# ---------------------------------------------------------------------------
# Composable column expressions
# ---------------------------------------------------------------------------


def _coerce_cmp(v: ColExpr | str | float | int) -> ColExpr:
    """Coerce a comparison operand to ColExpr."""
    if isinstance(v, ColExpr):
        return v
    if isinstance(v, str):
        return Col(v)
    return Lit(float(v))


class ColExpr:
    """Base for composable column expressions that resolve to ``pl.Expr``."""

    __slots__ = ()

    def resolve(self) -> pl.Expr:
        raise NotImplementedError

    # --- arithmetic operators ---

    def __add__(self, other: ColExpr | float | int) -> _BinOp:
        return _BinOp(self, _coerce(other), "+")

    def __radd__(self, other: float | int) -> _BinOp:
        return _BinOp(_coerce(other), self, "+")

    def __sub__(self, other: ColExpr | float | int) -> _BinOp:
        return _BinOp(self, _coerce(other), "-")

    def __rsub__(self, other: float | int) -> _BinOp:
        return _BinOp(_coerce(other), self, "-")

    def __mul__(self, other: ColExpr | float | int) -> _BinOp:
        return _BinOp(self, _coerce(other), "*")

    def __rmul__(self, other: float | int) -> _BinOp:
        return _BinOp(_coerce(other), self, "*")

    def __truediv__(self, other: ColExpr | float | int) -> _BinOp:
        return _BinOp(self, _coerce(other), "/")

    def __rtruediv__(self, other: float | int) -> _BinOp:
        return _BinOp(_coerce(other), self, "/")

    def __mod__(self, other: ColExpr | float | int) -> _BinOp:
        return _BinOp(self, _coerce(other), "%")

    def __rmod__(self, other: float | int) -> _BinOp:
        return _BinOp(_coerce(other), self, "%")

    def __neg__(self) -> _BinOp:
        return _BinOp(Lit(0.0), self, "-")

    # --- comparison operators (return Conditions) ---

    def __gt__(self, other: ColExpr | str | float | int) -> ValueGT:
        return ValueGT(self, _coerce_cmp(other))

    def __ge__(self, other: ColExpr | str | float | int) -> ValueGTE:
        return ValueGTE(self, _coerce_cmp(other))

    def __lt__(self, other: ColExpr | str | float | int) -> ValueLT:
        return ValueLT(self, _coerce_cmp(other))

    def __le__(self, other: ColExpr | str | float | int) -> ValueLTE:
        return ValueLTE(self, _coerce_cmp(other))


# Backward-compat alias
PriceExpr = ColExpr


@dataclass(frozen=True, slots=True)
class Col(ColExpr):
    """Column reference — resolves to ``pl.col(name)``."""

    name: str

    def resolve(self) -> pl.Expr:
        return pl.col(self.name)


@dataclass(frozen=True, slots=True)
class Lit(ColExpr):
    """Literal constant — resolves to ``pl.lit(value)``."""

    value: float

    def resolve(self) -> pl.Expr:
        return pl.lit(self.value)


@dataclass(frozen=True, slots=True)
class _BinOp(ColExpr):
    """Binary arithmetic node (internal)."""

    left: ColExpr
    right: ColExpr
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


def _coerce(v: ColExpr | float | int) -> ColExpr:
    """Wrap a raw numeric into ``Lit`` if needed."""
    if isinstance(v, ColExpr):
        return v
    return Lit(float(v))


def _coerce_base(v: ColExpr | str | float) -> ColExpr:
    """Coerce a base price to ColExpr: str -> Col, float -> Lit."""
    if isinstance(v, ColExpr):
        return v
    if isinstance(v, str):
        return Col(v)
    return Lit(float(v))


@dataclass(frozen=True, slots=True)
class Pct(ColExpr):
    """Price offset by ``pct``% from ``base``.

    Positive ``pct`` -> above, negative -> below.

    ``Pct("close", 1.0)``  -> ``close * 1.01``  (1% above)
    ``Pct("close", -0.5)`` -> ``close * 0.995`` (0.5% below)
    """

    base: ColExpr | str | float
    pct: float

    def resolve(self) -> pl.Expr:
        return _coerce_base(self.base).resolve() * (1.0 + self.pct / 100.0)


@dataclass(frozen=True, slots=True)
class EntryRef(ColExpr):
    """Column value snapshotted at the entry signal bar, forward-filled.

    The engine creates ``_entry_{col}`` columns automatically when it
    detects ``EntryRef`` nodes in the exit condition tree.

    ``EntryRef("close")`` resolves to ``pl.col("_entry_close")``.

    .. warning::

        The snapshot is taken at **every** raw entry signal, including one that
        fires while a position is already open. Position tracking suppresses
        such a signal, but the snapshot does not — so the reference moves
        mid-trade and the exit is measured against a bar the trade did not
        begin on. Exact for an edge-triggered entry paired with its own
        complement; not otherwise. :class:`~mktlib.backtest.Bracket` with
        ``rearm=True`` shares this latching rule and raises when it detects the
        case; ``EntryRef`` predates that check and stays silent.
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
class ValueGT(Condition):
    """``a > b`` (column name, constant, or ColExpr)."""

    a: str | ColExpr
    b: str | float | ColExpr
    trade_side: TradeSide | None = None

    def resolve(self) -> pl.Expr:
        return _ref(self.a) > _ref(self.b)


@dataclass(frozen=True, slots=True)
class ValueGTE(Condition):
    """``a >= b`` (column name, constant, or ColExpr)."""

    a: str | ColExpr
    b: str | float | ColExpr
    trade_side: TradeSide | None = None

    def resolve(self) -> pl.Expr:
        return _ref(self.a) >= _ref(self.b)


@dataclass(frozen=True, slots=True)
class ValueLT(Condition):
    """``a < b`` (column name, constant, or ColExpr)."""

    a: str | ColExpr
    b: str | float | ColExpr
    trade_side: TradeSide | None = None

    def resolve(self) -> pl.Expr:
        return _ref(self.a) < _ref(self.b)


@dataclass(frozen=True, slots=True)
class ValueLTE(Condition):
    """``a <= b`` (column name, constant, or ColExpr)."""

    a: str | ColExpr
    b: str | float | ColExpr
    trade_side: TradeSide | None = None

    def resolve(self) -> pl.Expr:
        return _ref(self.a) <= _ref(self.b)


# Backward-compat aliases
PriceIsAbove = ValueGT
PriceIsBelow = ValueLT


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


@dataclass(frozen=True, slots=True)
class Limit(Condition):
    """Exit condition with same-bar fill at a limit price.

    Wraps an exit condition (typically ``ValueGT/GTE/LT/LTE``) so that
    when the inner condition fires on bar ``t``, the position exits on
    that same bar at *price* — not at the next bar's open (mktlib's
    default fill-at-next-open semantics). Intended for take-profit /
    stop-loss strategies where the fill price is known in advance.

    When *price* is omitted, the fill price is auto-extracted from the
    right-hand side of the wrapped comparison — the typical TP/SL idiom
    ``high >= TP`` → fill at ``TP``. Pass an explicit *price* expression
    for trailing stops or decoupled trigger/fill (e.g. ``price=Col(
    "trailing_stop")`` with the comparison also against that column).

    v1 scope: only honored at the top level of ``exit_cond``. Nested
    use inside ``All`` / ``Any_`` / ``Not`` is treated as a plain
    boolean condition with no same-bar semantics. Bracket patterns
    (``Any_(TP, SL)``) are planned for a later release.
    """

    inner: Condition
    price: ColExpr | str | float | int | None = None
    trade_side: TradeSide | None = None

    def resolve(self) -> pl.Expr:
        return self.inner.resolve()

    def resolve_price(self) -> pl.Expr:
        if self.price is not None:
            return _ref(self.price)
        if isinstance(self.inner, (ValueGT, ValueGTE, ValueLT, ValueLTE)):
            return _ref(self.inner.b)
        msg = (
            f"Limit auto-extract requires inner to be "
            f"ValueGT/GTE/LT/LTE; got {type(self.inner).__name__}. "
            "Pass price= explicitly."
        )
        raise ValueError(msg)
