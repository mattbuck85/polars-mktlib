"""Protective take-profit / stop-loss exits for the backtest engine.

A :class:`Bracket` attaches two resting exit orders to every position the
strategy opens: a **take-profit limit** and a **stop-loss stop-market**.
They are checked against each bar's OHLC from the entry-fill bar onward,
and the first one to trigger closes the position *on that bar*, at the
bracket's own fill price — ahead of whatever the strategy's ``exit()``
condition would have done at the next bar's open.

Like :class:`~mktlib.backtest.Cost`, the model is **primitives only** —
floats, column names and a policy string, never a callable.  A callable
would be invisible to a consumer's cache key, and two runs whose keys
collide but whose brackets differ would silently serve each other's
results.

Fill semantics
--------------
These mirror a conventional event-driven OHLC broker, where a
long position's bracket is a SELL limit plus a SELL stop (and a short
position's is a BUY limit plus a BUY stop):

=====  ===  =======================  ====================
side   leg  trigger                  fill price
=====  ===  =======================  ====================
long   TP   ``high >= tp``           ``max(open, tp)``
long   SL   ``low <= sl``            ``min(open, sl)``
short  TP   ``low <= tp``            ``min(open, tp)``
short  SL   ``high >= sl``           ``max(open, sl)``
=====  ===  =======================  ====================

The ``max``/``min`` against the bar's open is what makes a gap honest: a
long stop at 95 on a bar that opens at 90 fills at 90, not at 95.

Level specification
-------------------
Both legs accept either a ``float`` or a ``str``:

``float``
    A **fraction of the entry fill price**, latched when the position
    fills.  For a long, ``take_profit=0.02`` is ``entry * 1.02`` and
    ``stop_loss=0.01`` is ``entry * 0.99``; for a short the multipliers
    mirror.  This matches the usual live convention of computing bracket
    levels from the position's average fill price.

``str``
    A column of **absolute price levels**, latched at the entry *signal*
    bar — the same bar :class:`~mktlib.backtest.EntryRef` snapshots, one
    bar before the fill.  Use this for levels the strategy derived from
    data it had at signal time, e.g. ``close + atr * mult``.

Examples
--------
A 2% target with a 1% stop::

    from mktlib.backtest import Bracket, run

    result = run(df, strategy, bracket=Bracket(take_profit=0.02, stop_loss=0.01))

An ATR stop the strategy computed in ``init()``, with no target::

    result = run(df, strategy, bracket=Bracket(stop_loss="atr_stop"))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import polars as pl

#: Discriminator values written to :data:`BRACKET_KIND_COLUMN`.
TAKE_PROFIT = "take_profit"
STOP_LOSS = "stop_loss"

#: Same-bar resolution policies for :attr:`Bracket.both_touch`.
BothTouch = Literal["stop_first", "take_profit_first"]
_BOTH_TOUCH_POLICIES: frozenset[str] = frozenset({"stop_first", "take_profit_first"})

# Internal columns materialized by the engine and dropped before the
# BacktestResult is handed back.
BRACKET_LEVEL_COLUMN = "_bracket_level"
BRACKET_KIND_COLUMN = "_bracket_kind"
BRACKET_EXIT_COLUMN = "_bracket_exit_bar"
BRACKET_POST_COLUMN = "_bracket_post"
BRACKET_SEEN_COLUMN = "_bracket_seen"
BRACKET_CANDIDATE_COLUMN = "_bracket_candidate"
BLOCK_COLUMN = "_bracket_block"
ENTRY_FILL_COLUMN = "_bracket_entry_fill"
TP_LEVEL_COLUMN = "_bracket_tp"
SL_LEVEL_COLUMN = "_bracket_sl"

BRACKET_COLUMNS: tuple[str, ...] = (
    BRACKET_LEVEL_COLUMN,
    BRACKET_KIND_COLUMN,
    BRACKET_EXIT_COLUMN,
    BRACKET_POST_COLUMN,
    BRACKET_SEEN_COLUMN,
    BRACKET_CANDIDATE_COLUMN,
    BLOCK_COLUMN,
    ENTRY_FILL_COLUMN,
    TP_LEVEL_COLUMN,
    SL_LEVEL_COLUMN,
)


@dataclass(frozen=True, slots=True)
class Bracket:
    """Protective take-profit / stop-loss levels attached to every position.

    At least one of *take_profit* / *stop_loss* must be given. The bracket
    is armed on the **entry fill bar** and is checked on every bar the
    position is live, including that first one — a gap straight through
    the stop on the entry bar fills at that bar's open.

    Parameters
    ----------
    take_profit
        Profit target. A ``float`` is a fraction of the entry fill price
        (``0.02`` = 2% in the trade's favour); a ``str`` names a column of
        absolute price levels, read on the entry *signal* bar. Must be
        finite, and strictly positive when numeric.
    stop_loss
        Protective stop, same two forms — a ``float`` is a fraction of the
        entry fill price *against* the trade.
    both_touch
        How to resolve a bar whose high **and** low tag both legs.
        ``"stop_first"`` (default) books the loss; ``"take_profit_first"``
        books the gain.
    rearm
        When ``True``, a bracket exit releases the position so a later entry
        signal can open a new trade, instead of closing the block for good.
        **Requires ``str`` level specs**; a ``float`` spec is a fraction of the
        entry *fill* price, which is not known until the position is known,
        which under re-arm depends on the levels — see Notes. Off by default,
        because turning it on changes results.

    Notes
    -----
    **Within-bar ordering is unknowable from OHLC.** A bar that reaches
    both levels records no information about which came first — the same
    open/high/low/close is produced by a path that stopped out and by one
    that took profit. *both_touch* is therefore a stated assumption, not a
    measurement, and no OHLC backtest can validate it. Re-run with both
    policies to bound the true result; a strategy whose edge lives inside
    that band has no edge that this data can demonstrate.

    **The default deliberately diverges from submission-order OCO.** A live
    bracket is commonly an OCO pair whose take-profit leg is submitted
    first, filled in submission order — so on a both-touch bar the realized
    policy is ``"take_profit_first"``. mktlib defaults to ``"stop_first"``
    because a backtest should not book the favourable resolution of an
    ambiguity it cannot observe. Pass ``both_touch="take_profit_first"`` to
    reproduce submission-order behaviour.

    **By default a bracket exit does not re-arm the entry signal.** The
    position is closed for the remainder of the block the entry opened; the
    next trade needs the strategy's ``exit()`` condition to fire and a fresh
    entry signal after it. A strategy whose *only* exit is the bracket
    therefore trades exactly **once** and then sits flat — measured at 1 trade
    against 10,023 over the same 500k bars. Pass ``rearm=True`` for the
    live-like behaviour.

    **Why ``rearm`` needs ``str`` levels.** Re-arm works by feeding the touch
    into the position state, so the position depends on the levels. A ``float``
    spec scales the entry *fill* price, which is only known once the position
    is known — circular. A ``str`` spec names a column read at the entry
    *signal* bar, which is knowable beforehand, so the recurrence closes.

    **``rearm`` re-anchors on every raw entry signal**, matching what
    :class:`~mktlib.backtest.EntryRef` already does. That is exact when an entry
    signal cannot fire while a position is open, and wrong when it can — a
    condition that is true on half the bars would re-anchor the level on almost
    every held bar, quietly turning a fixed bracket into a trailing one. The
    engine detects this and raises rather than returning a number; it is not a
    warning, because the failure inflates results.

    Requires ``high`` and ``low`` columns in the input DataFrame.
    """

    take_profit: float | str | None = None
    stop_loss: float | str | None = None
    both_touch: BothTouch = "stop_first"
    rearm: bool = False

    def __post_init__(self) -> None:
        for name in ("take_profit", "stop_loss"):
            value: object = getattr(self, name)
            if value is None or isinstance(value, str):
                if isinstance(value, str) and not value:
                    msg = f"Bracket.{name} must be a non-empty column name"
                    raise ValueError(msg)
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                msg = (
                    f"Bracket.{name} must be a fraction, a column name or None, "
                    f"got {value!r}"
                )
                raise TypeError(msg)
            if not math.isfinite(value):
                msg = f"Bracket.{name} must be finite, got {value!r}"
                raise ValueError(msg)
            if value <= 0:
                msg = f"Bracket.{name} must be strictly positive, got {value!r}"
                raise ValueError(msg)
        if self.take_profit is None and self.stop_loss is None:
            msg = (
                "Bracket requires at least one of take_profit / stop_loss; "
                "an empty bracket would never close a position."
            )
            raise ValueError(msg)
        if self.both_touch not in _BOTH_TOUCH_POLICIES:
            msg = (
                f"Bracket.both_touch must be one of {sorted(_BOTH_TOUCH_POLICIES)}, "
                f"got {self.both_touch!r}"
            )
            raise ValueError(msg)
        if not isinstance(self.rearm, bool):
            msg = f"Bracket.rearm must be a bool, got {self.rearm!r}"
            raise TypeError(msg)
        if self.rearm:
            numeric = [
                name
                for name in ("take_profit", "stop_loss")
                if isinstance(getattr(self, name), (int, float))
                and not isinstance(getattr(self, name), bool)
            ]
            if numeric:
                msg = (
                    f"Bracket(rearm=True) requires column-name levels, but "
                    f"{', '.join(numeric)} is a fraction. A fraction scales the "
                    "entry fill price, which is not known until the position is "
                    "known — and under re-arm the position depends on the levels. "
                    "Pass a column of absolute price levels instead, computed "
                    "from whatever the entry signal bar makes available."
                )
                raise NotImplementedError(msg)

    @property
    def level_columns(self) -> tuple[str, ...]:
        """Names of any input columns the two legs read absolute levels from."""
        return tuple(
            spec for spec in (self.take_profit, self.stop_loss) if isinstance(spec, str)
        )


def level_expr(
    spec: float | str,
    *,
    leg: str,
    is_long: bool,
    entry_clean_col: str,
    entry_fill_col: str,
) -> pl.Expr:
    """Per-bar bracket level for one leg, held for the life of the position.

    A ``str`` *spec* snapshots that column on the entry signal bar; a
    ``float`` scales the latched entry fill price by the side-appropriate
    multiplier.
    """
    if isinstance(spec, str):
        return (
            pl.when(pl.col(entry_clean_col))
            .then(pl.col(spec))
            .otherwise(None)
            .forward_fill()
            .cast(pl.Float64)
        )
    # A take-profit sits in the trade's favour, a stop against it.
    favourable = leg == TAKE_PROFIT
    up = favourable == is_long
    multiplier = 1.0 + spec if up else 1.0 - spec
    return pl.col(entry_fill_col) * pl.lit(multiplier, dtype=pl.Float64)


def held_expr(
    *,
    entry_col: str,
    exit_col: str,
    touch: pl.Expr,
) -> pl.Expr:
    """Position state with a bracket touch treated as a close — the re-arm path.

    This is the ordinary ``_position`` recurrence with *touch* OR-ed into its
    zero branch. Entry keeps priority over both, matching the non-bracket
    engine: a bracket closing inside bar *t* and an entry signal firing at
    bar *t*'s close are not in conflict — the first ends a trade intrabar, the
    second opens one that fills at *t+1*.

    Because the forward-fill already latches the first zero, "first trigger in
    the block wins" falls out for free, and the position re-arms on the next
    entry. That is the whole mechanism: no block ids, no window.

    *touch* is deliberately **ungated** — it is not masked by "were we actually
    holding?". It does not need to be: forcing ``0`` on a bar where the position
    is already flat is a no-op, so the ungated recurrence and the gated one
    agree everywhere. Gating would reintroduce the circularity the re-arm path
    exists to avoid, since the gate is the very state being computed.
    ``tests/backtest/test_bracket_rearm_recurrence.py`` pins that equivalence
    against a sequential reference implementation.
    """
    return (
        pl.when(pl.col(entry_col))
        .then(pl.lit(1))
        .when(pl.col(exit_col) | touch)
        .then(pl.lit(0))
        .otherwise(pl.lit(None))
        .forward_fill()
        .fill_null(0)
    )


def trigger_expr(*, leg: str, is_long: bool, level_col: str) -> pl.Expr:
    """Boolean: does this bar's range reach *leg*'s level?

    Emitted per compile-time-known side — never derived by multiplying a
    comparison through a signed constant.
    """
    if leg == TAKE_PROFIT:
        if is_long:
            return pl.col("high") >= pl.col(level_col)
        return pl.col("low") <= pl.col(level_col)
    if is_long:
        return pl.col("low") <= pl.col(level_col)
    return pl.col("high") >= pl.col(level_col)


def fill_expr(*, leg: str, is_long: bool, level_col: str) -> pl.Expr:
    """Realized fill price for *leg*, honouring an adverse or favourable gap."""
    if leg == TAKE_PROFIT:
        if is_long:
            return pl.max_horizontal(pl.col("open"), pl.col(level_col))
        return pl.min_horizontal(pl.col("open"), pl.col(level_col))
    if is_long:
        return pl.min_horizontal(pl.col("open"), pl.col(level_col))
    return pl.max_horizontal(pl.col("open"), pl.col(level_col))
