"""pandera schemas for mktlib.backtest output DataFrames."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class ReturnsSchema(pa.DataFrameModel):
    """BacktestResult.returns schema: (date, return)."""

    date: pl.Date
    return_: float = pa.Field(alias="return")


class TradesSchema(pa.DataFrameModel):
    """BacktestResult.trades schema: (entry_date, exit_date, side, pnl, bars_held)."""

    entry_date: pl.Date
    exit_date: pl.Date
    side: pl.Int8 = pa.Field(isin=[-1, 1])
    pnl: float
    bars_held: int = pa.Field(ge=0)


class IntradayTradesSchema(pa.DataFrameModel):
    """``BacktestResult.trades`` for an intraday backtest.

    Identical contract to :class:`TradesSchema`, but with ``Datetime`` dates.
    ``run()`` passes the input frame's ``date`` dtype straight through, so an
    intraday input emits ``Datetime`` here — which means ``TradesSchema``, which
    pins ``pl.Date``, has only ever covered daily input.
    """

    entry_date: pl.Datetime
    exit_date: pl.Datetime
    side: pl.Int8 = pa.Field(isin=[-1, 1])
    pnl: float
    bars_held: int = pa.Field(ge=0)


class SignalsSchemaBase(pa.DataFrameModel):
    """BacktestResult.signals schema (base columns only, strict=False)."""

    entry: bool = pa.Field(alias="_entry", nullable=True)
    exit: bool = pa.Field(alias="_exit", nullable=True)  # noqa: A003
    position: pl.Int32 = pa.Field(alias="_position")
    side: pl.Int8 = pa.Field(alias="_side")

    class Config:
        strict = False


class PortfolioWeightsSchema(pa.DataFrameModel):
    """Canonical ``to_portfolio_weights_df()`` output: ``(instrument, weight)``.

    ``instrument`` is ``Utf8`` (ticker-like strings); ``weight`` is a
    non-negative ``Float64``. The full validator in
    ``mktlib.backtest._weights`` additionally enforces non-empty,
    no-nulls, no-NaN, unique instruments, and ``sum(weight) > 0``.
    """

    instrument: pl.Utf8
    weight: float = pa.Field(ge=0)


class WeightedReturnsSchema(pa.DataFrameModel):
    """``MultiBacktestResult.returns`` when instrument_weights is set.

    Portfolio-aggregated time series: one row per date, same shape as
    the single-instrument ``BacktestResult.returns``.
    """

    date: pl.Date
    return_: float = pa.Field(alias="return")
