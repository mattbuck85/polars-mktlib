"""Schema validation tests for mktlib.reports output DataFrames."""

from __future__ import annotations

import polars as pl

from mktlib.reports._compat import coerce_returns
from mktlib.reports._stats import monthly_returns, yearly_returns

from tests.schemas.reports import (
    CoercedReturnsSchema,
    MonthlyReturnsSchema,
    YearlyReturnsSchema,
)


def _returns_fixture() -> pl.DataFrame:
    """Small returns DataFrame spanning multiple months and years."""
    dates = pl.date_range(
        pl.date(2023, 11, 1), pl.date(2024, 2, 28), eager=True
    )
    n = len(dates)
    return pl.DataFrame(
        {
            "date": dates,
            "return": [0.001 * ((-1) ** i) for i in range(n)],
        }
    )


class TestCoercedReturnsSchema:
    def test_from_dataframe(self):
        df = _returns_fixture()
        result = coerce_returns(df)
        CoercedReturnsSchema.validate(result)

    def test_from_series(self):
        series = pl.Series("return", [0.01, -0.005, 0.002, 0.003])
        result = coerce_returns(series)
        CoercedReturnsSchema.validate(result)


class TestMonthlyReturnsSchema:
    def test_basic(self):
        df = _returns_fixture()
        result = monthly_returns(df)
        MonthlyReturnsSchema.validate(result)
        # Should span Nov 2023 through Feb 2024
        assert result.height >= 4

    def test_compounded_false(self):
        df = _returns_fixture()
        result = monthly_returns(df, compounded=False)
        MonthlyReturnsSchema.validate(result)


class TestYearlyReturnsSchema:
    def test_basic(self):
        df = _returns_fixture()
        result = yearly_returns(df)
        YearlyReturnsSchema.validate(result)
        # Should have 2023 and 2024
        assert result.height == 2

    def test_compounded_false(self):
        df = _returns_fixture()
        result = yearly_returns(df, compounded=False)
        YearlyReturnsSchema.validate(result)
