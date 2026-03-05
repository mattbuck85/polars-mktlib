from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from mktlib.reports._compat import coerce_benchmark, coerce_returns


class TestCoerceReturns:
    def test_polars_dataframe_with_named_columns(self):
        df = pl.DataFrame(
            {
                "date": [
                    dt.date(2024, 1, 2),
                    dt.date(2024, 1, 3),
                    dt.date(2024, 1, 4),
                ],
                "return": [0.01, -0.02, 0.015],
            }
        )
        result = coerce_returns(df)
        assert result.columns == ["date", "return"]
        assert result["date"].dtype == pl.Date
        assert len(result) == 3

    def test_polars_dataframe_infers_columns(self):
        df = pl.DataFrame(
            {
                "timestamp": [dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
                "pct_change": [0.01, -0.02],
            }
        )
        result = coerce_returns(df)
        assert result.columns == ["date", "return"]
        assert result["return"].to_list() == [0.01, -0.02]

    def test_polars_dataframe_datetime_cast_to_date(self):
        df = pl.DataFrame(
            {
                "date": [
                    dt.datetime(2024, 1, 2, 10, 30),
                    dt.datetime(2024, 1, 3, 10, 30),
                ],
                "return": [0.01, -0.02],
            }
        )
        result = coerce_returns(df)
        assert result["date"].dtype == pl.Date

    def test_polars_series(self):
        s = pl.Series("returns", [0.01, -0.02, 0.015])
        result = coerce_returns(s)
        assert result.columns == ["date", "return"]
        assert len(result) == 3
        # Synthetic business days start from 2000-01-03 (Monday)
        assert result["date"][0] == dt.date(2000, 1, 3)

    def test_polars_series_skips_weekends(self):
        s = pl.Series("r", [0.01] * 6)
        result = coerce_returns(s)
        dates = result["date"].to_list()
        # 5 weekdays + skip weekend + 1 Monday
        assert dates[4] == dt.date(2000, 1, 7)  # Friday
        assert dates[5] == dt.date(2000, 1, 10)  # Monday (skipped Sat/Sun)

    def test_pandas_series(self):
        pd = pytest.importorskip("pandas")
        idx = pd.DatetimeIndex(
            [
                dt.datetime(2024, 1, 2),
                dt.datetime(2024, 1, 3),
                dt.datetime(2024, 1, 4),
            ]
        )
        s = pd.Series([0.01, -0.02, 0.015], index=idx, name="returns")
        result = coerce_returns(s)
        assert result.columns == ["date", "return"]
        assert result["date"].dtype == pl.Date
        assert len(result) == 3

    def test_pandas_series_with_timezone(self):
        pd = pytest.importorskip("pandas")
        idx = pd.DatetimeIndex(
            [dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 3)],
            tz="US/Eastern",
        )
        s = pd.Series([0.01, -0.02], index=idx, name="r")
        result = coerce_returns(s)
        assert result["date"].dtype == pl.Date

    def test_invalid_dataframe_raises(self):
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="Cannot infer"):
            coerce_returns(df)


class TestCoerceBenchmark:
    def test_none_returns_none(self):
        assert coerce_benchmark(None) is None

    def test_delegates_to_coerce_returns(self):
        s = pl.Series("bench", [0.005, -0.01])
        result = coerce_benchmark(s)
        assert result is not None
        assert result.columns == ["date", "return"]
