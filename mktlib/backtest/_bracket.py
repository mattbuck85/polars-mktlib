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
    A column of **absolute price levels**, latched on the bar the position
    *opens* — the same bar :class:`~mktlib.backtest.EntryRef` snapshots,
    one bar before the fill.  Use this for levels the strategy derived from
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

#: Level-anchoring policies for :attr:`Bracket.anchor`.
Anchor = Literal["position", "signal"]
_ANCHOR_POLICIES: frozenset[str] = frozenset({"position", "signal"})

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
# Materialized only under ``anchor="signal"``; absent otherwise.
ANCHOR_FILL_COLUMN = "_bracket_anchor_fill"
RESIGNAL_COLUMN = "_bracket_resignal"

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
    ANCHOR_FILL_COLUMN,
    RESIGNAL_COLUMN,
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
    anchor
        Which entry signal the levels are measured from. ``"position"``
        (default) latches both legs once, on the entry that opened the
        position, and holds them for the life of the trade. ``"signal"``
        re-latches them on every later entry signal that fires while the
        position is still open.

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

    **A bracket exit does not re-arm the entry signal.** The position is
    closed for the remainder of the block the entry opened; the next trade
    needs the strategy's ``exit()`` condition to fire and a fresh entry
    signal after it. Live, a still-true entry condition would re-enter on
    the very next bar; that difference is intentional, since re-entering a
    position that just stopped out is rarely what a bracket user wants.

    **A re-anchor is armed on the following bar.** Under
    ``anchor="signal"`` an entry signal on bar ``k`` that fires while the
    position is held is observed at that bar's close, so the level in force
    *during* bar ``k`` is still the previous one and the new level applies
    from ``k + 1``. A ``float`` leg re-anchors to ``open[k + 1]`` — the
    price a fresh entry on that signal would have filled at — and a ``str``
    leg re-reads its column on bar ``k``. A leg tagged on bar ``k``
    therefore closes the position before the re-anchor takes effect.

    Re-anchoring moves the protective levels only. The position opened
    once, trade P&L still measures from the original entry fill, and a
    re-signal never re-opens or extends a block: block boundaries are fixed
    before the bracket is applied.

    Under ``flatten_eod`` a re-signal on a session-last bar does not
    re-anchor, since the position is flattened at that bar's open. With
    ``flatten_eod=False`` a re-signal on a session's last bar anchors a
    ``float`` leg to the next session's opening price, so an overnight gap
    carries the levels with it.

    An entry condition that is **level**-triggered rather than
    edge-triggered stays true on consecutive bars, so under
    ``anchor="signal"`` it re-latches on every one of them and the bracket
    becomes a trailing one.

    *anchor* governs bracket levels only.
    :class:`~mktlib.backtest.EntryRef` snapshots are unaffected and
    continue to latch at the entry that opened the position.

    Requires ``high`` and ``low`` columns in the input DataFrame.
    """

    take_profit: float | str | None = None
    stop_loss: float | str | None = None
    both_touch: BothTouch = "stop_first"
    anchor: Anchor = "position"

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
        if self.anchor not in _ANCHOR_POLICIES:
            msg = (
                f"Bracket.anchor must be one of {sorted(_ANCHOR_POLICIES)}, "
                f"got {self.anchor!r}"
            )
            raise ValueError(msg)

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
    resignal_col: str | None = None,
) -> pl.Expr:
    """Per-bar bracket level for one leg, held for the life of the position.

    A ``str`` *spec* snapshots that column on the bar the position
    **opens** — the realized entry transition *entry_clean_col*, not every
    bar the entry condition fires on. The two coincide for the signal that
    opened the position and nowhere else: a signal that fires while the
    position is already held is suppressed by the position machinery and
    latches nothing. A ``float`` scales the latched entry fill price by the
    side-appropriate multiplier.

    Under ``anchor="signal"`` the level also moves on entry signals that
    fire while the position is held. The two spec kinds take that through
    different doors:

    *float*
        entirely in the **caller**, which passes an *entry_fill_col* that
        re-latches on those bars. This function's float body is the same
        multiplication either way.
    *str*
        here, via *resignal_col* — a boolean column marking those bars.
        When it is ``None`` the expression is exactly the position-anchored
        one.

    The re-latch is ``shift(1)``-ed because the signal on bar ``k`` is
    observed at that bar's close: the position is live throughout ``k`` on
    the level it already had, and the new level is in force from ``k + 1``.
    An unshifted re-latch would move the level *during* the bar the signal
    fired on.

    ``coalesce`` takes the opening latch first. The two collide on one bar
    only under ``flatten_eod``, where a session-last signal is both a
    re-signal against the outgoing position and — deferred to the next
    session's first bar — the signal that opens the next one. That bar
    belongs to the new position, so its own latch wins. The caller
    independently keeps session-last bars out of *resignal_col*, which
    empties the relatch on that bar; measured, either defence alone
    resolves the collision correctly and this ordering is the second one.
    """
    if isinstance(spec, str):
        initial = pl.when(pl.col(entry_clean_col)).then(pl.col(spec)).otherwise(None)
        if resignal_col is None:
            return initial.forward_fill().cast(pl.Float64)
        relatch = (
            pl.when(pl.col(resignal_col)).then(pl.col(spec)).otherwise(None).shift(1)
        )
        return pl.coalesce(initial, relatch).forward_fill().cast(pl.Float64)
    # A take-profit sits in the trade's favour, a stop against it.
    favourable = leg == TAKE_PROFIT
    up = favourable == is_long
    multiplier = 1.0 + spec if up else 1.0 - spec
    return pl.col(entry_fill_col) * pl.lit(multiplier, dtype=pl.Float64)


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
