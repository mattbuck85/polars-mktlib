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
        after the strategy's ``init()`` hook has run) and must be numeric
        and non-null on fill bars.

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
