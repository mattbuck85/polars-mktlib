"""pandera schemas for mktlib.reports output DataFrames."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class CoercedReturnsSchema(pa.DataFrameModel):
    """coerce_returns() output: (date, return)."""

    date: pl.Date
    return_: float = pa.Field(alias="return")


class MonthlyReturnsSchema(pa.DataFrameModel):
    """monthly_returns() output: (year, month, monthly_return)."""

    year: pl.Int32
    month: pl.Int8 = pa.Field(ge=1, le=12)
    monthly_return: float


class YearlyReturnsSchema(pa.DataFrameModel):
    """yearly_returns() output: (year, yearly_return)."""

    year: pl.Int32
    yearly_return: float
