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


class SignalsSchemaBase(pa.DataFrameModel):
    """BacktestResult.signals schema (base columns only, strict=False)."""

    entry: bool = pa.Field(alias="_entry", nullable=True)
    exit: bool = pa.Field(alias="_exit", nullable=True)  # noqa: A003
    position: pl.Int32 = pa.Field(alias="_position")
    side: pl.Int8 = pa.Field(alias="_side")

    class Config:
        strict = False
