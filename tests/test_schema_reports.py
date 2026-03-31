"""Schema validation tests for mktlib.reports output DataFrames."""

from __future__ import annotations

import polars as pl
import pytest

from mktlib.reports._compat import coerce_returns
from mktlib.reports._stats import monthly_returns, yearly_returns

from tests.schemas.reports import (
    CoercedReturnsSchema,
    MonthlyReturnsSchema,
    TradesInputSchema,
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


def _trades_fixture() -> pl.DataFrame:
    """Small trades DataFrame for schema validation."""
    from datetime import date
    return pl.DataFrame(
        {
            "entry_date": [date(2024, 1, 2), date(2024, 1, 10), date(2024, 1, 20)],
            "exit_date": [date(2024, 1, 5), date(2024, 1, 15), date(2024, 1, 25)],
            "side": pl.Series([1, -1, 1], dtype=pl.Int8),
            "pnl": [0.02, -0.01, 0.015],
            "bars_held": [3, 5, 5],
        }
    )


class TestTradesInputSchema:
    def test_valid_trades(self):
        df = _trades_fixture()
        TradesInputSchema.validate(df)

    def test_exit_date_after_entry_date(self):
        df = _trades_fixture()
        assert (df["exit_date"] >= df["entry_date"]).all()

    def test_bars_held_non_negative(self):
        df = _trades_fixture()
        assert (df["bars_held"] >= 0).all()

    def test_invalid_negative_bars_held_fails(self):
        import pandera.errors as pa_errors
        bad = _trades_fixture().with_columns(pl.Series("bars_held", [-1, 5, 5]))
        with pytest.raises(pa_errors.SchemaError):
            TradesInputSchema.validate(bad)
