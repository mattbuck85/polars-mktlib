"""Per-fill transaction costs for the backtest engine.

The engine is **share-count free** — it only ever composes price relatives,
never a quantity or a notional.  That rules out per-share or per-order fee
schedules (an IB-style ``$0.005/share, $1.00 minimum`` cannot be expressed
without knowing the size of the order), so the cost model here is stated in
**basis points of notional, per side**.

The model is deliberately restricted to *primitives*: floats and a column
name, never a callable.  A callable cost model would be invisible to a
consumer's cache key (a closure has no stable identity), and two runs whose
keys collide but whose cost transforms differ would silently serve each
other's results.

Costs are charged **at the fill**, on the entry bar and on the exit bar, in
both the returns series and ``trades.pnl``.  They are never applied as a
post-hoc transform on the returns series: such a transform cannot see trade
boundaries and would charge holding bars too.

Examples
--------
Flat 1 bp commission plus 0.5 bp of assumed slippage on each side::

    from mktlib.backtest import Cost, run

    result = run(df, strategy, cost=Cost(commission_bps=1.0, slippage_bps=0.5))

Per-bar slippage driven by a column the strategy computed (e.g. half the
quoted spread in bps), stacked on top of a flat commission::

    result = run(df, strategy, cost=Cost(commission_bps=1.0, slippage_col="half_spread_bps"))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

#: Internal per-bar cost column materialized by the engine and dropped
#: before :class:`~mktlib.backtest.BacktestResult` is returned.
COST_COLUMN = "_cost_bps"


@dataclass(frozen=True, slots=True)
class Cost:
    """Per-side transaction cost in basis points of notional.

    All fields default to zero, so ``Cost()`` is an exact no-op: a run with
    ``cost=Cost()`` produces results identical to a run without a cost model.

    Parameters
    ----------
    commission_bps
        Flat commission charged on **each** side of a round trip, in basis
        points (``1.0`` = 0.01%). Must be finite and non-negative.
    slippage_bps
        Flat slippage charged on each side, in basis points. Kept separate
        from *commission_bps* purely for bookkeeping — the engine sums them.
        Must be finite and non-negative.
    slippage_col
        Optional column name holding **additional** per-bar slippage in
        basis points, read on the fill bar. Stacks on top of the two flat
        terms. The column must exist in the DataFrame the engine sees (i.e.
        after the strategy's ``init()`` hook has run) and must be numeric,
        finite, non-null and non-negative **on every fill bar**. All four
        conditions are enforced by :func:`validate_slippage_col`; values on
        non-fill bars are never read and are unconstrained, so a column
        that is null during an indicator warm-up is fine as long as it is
        populated wherever a fill actually lands.

    Notes
    -----
    Basis points, not dollars-per-share, because the engine never knows the
    share count. Convert a per-share schedule yourself at the call site:
    ``bps = 1e4 * fee_per_share / expected_fill_price``.

    Costs are subtracted from the bar return, never multiplied by the trade
    side — a cost reduces the realized return of a short exactly as it does
    a long.
    """

    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    slippage_col: str | None = None

    def __post_init__(self) -> None:
        for name in ("commission_bps", "slippage_bps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                msg = f"Cost.{name} must be a real number, got {value!r}"
                raise TypeError(msg)
            if not math.isfinite(value):
                msg = f"Cost.{name} must be finite, got {value!r}"
                raise ValueError(msg)
            if value < 0:
                msg = f"Cost.{name} must be non-negative, got {value!r}"
                raise ValueError(msg)
        if self.slippage_col is not None:
            if not isinstance(self.slippage_col, str):
                msg = f"Cost.slippage_col must be a column name or None, got {self.slippage_col!r}"
                raise TypeError(msg)
            if not self.slippage_col:
                msg = "Cost.slippage_col must be a non-empty column name"
                raise ValueError(msg)

    @property
    def flat_bps(self) -> float:
        """Sum of the two flat per-side terms, in basis points."""
        return self.commission_bps + self.slippage_bps


def cost_bps_expr(cost: Cost) -> pl.Expr:
    """Per-bar cost in basis points, to be read **on the fill bar**."""
    expr = pl.lit(cost.flat_bps, dtype=pl.Float64)
    if cost.slippage_col is not None:
        expr = expr + pl.col(cost.slippage_col)
    return expr


def validate_slippage_col(
    signals: pl.DataFrame,
    cost: Cost,
    *,
    fill_bar: pl.Expr,
) -> None:
    """Reject a ``slippage_col`` that would silently produce a wrong number.

    Checked on the *fill bars only* — the bars whose cost the engine
    actually reads (``fill_bar`` is that mask). Four failure modes, each
    of which is silent rather than loud if left unguarded:

    missing column
        Raises rather than letting Polars fail deep inside the return
        expression with an unrecognizable message.
    non-numeric dtype
        Same reasoning.
    null on a fill bar
        ``flat + null`` is null, the null propagates through the fill-bar
        return expression, and the engine's terminal ``fill_null(0.0)``
        then replaces that bar's **real return with zero** while the
        corresponding ``trades.pnl`` goes null. A missing quote on one
        fill bar would silently rewrite the P&L of the whole run.
    negative or non-finite on a fill bar
        A negative slippage is a *rebate* — it would pay the strategy for
        trading and inflate every metric downstream. The scalar fields are
        already guarded in :meth:`Cost.__post_init__`; the column has to
        be guarded here because its values only exist at run time.

    Raises
    ------
    ValueError
        If the column is missing, or violates any of the fill-bar
        conditions.
    TypeError
        If the column exists but is not of a numeric dtype.
    """
    col = cost.slippage_col
    if col is None:
        return

    if col not in signals.columns:
        msg = (
            f"Cost.slippage_col={col!r} not found in DataFrame columns: "
            f"{sorted(signals.columns)}"
        )
        raise ValueError(msg)

    if not signals.schema[col].is_numeric():
        msg = (
            f"Cost.slippage_col={col!r} must be a numeric column of basis "
            f"points, got dtype {signals.schema[col]}"
        )
        raise TypeError(msg)

    on_fill = signals.filter(fill_bar).get_column(col)
    if on_fill.is_empty():
        return

    n_null = int(on_fill.is_null().sum())
    if n_null:
        msg = (
            f"Cost.slippage_col={col!r} is null on {n_null} fill bar(s). The "
            "cost is read unshifted on the bar a fill lands, so a null there "
            "would propagate into that bar's return and be silently zeroed — "
            "replacing a real return with 0.0 and nulling the trade's pnl. "
            "Fill or forward-fill the column before passing it to run()."
        )
        raise ValueError(msg)

    values = on_fill.cast(pl.Float64)
    if not bool(values.is_finite().all()):
        msg = (
            f"Cost.slippage_col={col!r} has non-finite value(s) on at least "
            "one fill bar; slippage in basis points must be finite."
        )
        raise ValueError(msg)

    n_negative = int((values < 0.0).sum())
    if n_negative:
        msg = (
            f"Cost.slippage_col={col!r} is negative on {n_negative} fill "
            f"bar(s) (minimum {values.min()}). A negative slippage is a "
            "rebate that would pay the strategy for trading; Cost models a "
            "cost, so per-bar slippage must be non-negative."
        )
        raise ValueError(msg)
