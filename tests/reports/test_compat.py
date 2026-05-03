from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from mktlib.reports._compat import (
    _ensure_daily,
    coerce_benchmark,
    coerce_returns,
)


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
        assert len(result) == 3

    def test_polars_dataframe_infers_columns(self):
        df = pl.DataFrame(
            {
                "timestamp": [dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
                "pct_change": [0.01, -0.02],
            }
        )
        result = coerce_returns(df)
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

    def test_polars_series(self):
        s = pl.Series("returns", [0.01, -0.02, 0.015])
        result = coerce_returns(s)
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
        assert len(result) == 3

    def test_pandas_series_with_timezone(self):
        pd = pytest.importorskip("pandas")
        idx = pd.DatetimeIndex(
            [dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 3)],
            tz="US/Eastern",
        )
        s = pd.Series([0.01, -0.02], index=idx, name="r")
        result = coerce_returns(s)

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


class TestEnsureDaily:
    """Sub-daily aggregation: collapses multi-row-per-date to compounded daily."""

    def test_idempotent_byte_for_byte_on_unique_dates(self):
        """Already-daily input must pass through unchanged — no FP drift."""
        df = pl.DataFrame({
            "date": [
                dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4),
            ],
            "return": [0.01, -0.02, 0.03],
        })
        out = _ensure_daily(df)
        assert out.equals(df)

    def test_collapses_intraday_to_compounded_daily(self):
        df = pl.DataFrame({
            "date": [
                dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2024, 1, 3),
            ],
            "return": [0.01, 0.02, -0.01],
        })
        out = _ensure_daily(df)
        assert out.height == 2
        # 2024-01-02 → (1.01)(1.02) - 1 = 0.0302
        row1 = out.filter(pl.col("date") == dt.date(2024, 1, 2)).row(0, named=True)
        assert row1["return"] == pytest.approx(0.0302, abs=1e-12)
        row2 = out.filter(pl.col("date") == dt.date(2024, 1, 3)).row(0, named=True)
        assert row2["return"] == pytest.approx(-0.01, abs=1e-12)

    def test_idempotent_under_double_call(self):
        df = pl.DataFrame({
            "date": [
                dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2024, 1, 3),
            ],
            "return": [0.01, 0.02, -0.01],
        })
        once = _ensure_daily(df)
        twice = _ensure_daily(once)
        assert twice.equals(once)

    def test_sorts_unsorted_input(self):
        df = pl.DataFrame({
            "date": [dt.date(2024, 1, 5), dt.date(2024, 1, 3), dt.date(2024, 1, 4)],
            "return": [0.03, -0.01, 0.02],
        })
        out = _ensure_daily(df)
        assert out["date"].to_list() == [
            dt.date(2024, 1, 3), dt.date(2024, 1, 4), dt.date(2024, 1, 5),
        ]

    def test_coerce_returns_collapses_intraday(self):
        """`coerce_returns` invokes `_ensure_daily` as the final step."""
        df = pl.DataFrame({
            "date": [
                dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2024, 1, 3),
            ],
            "return": [0.01, 0.02, -0.01],
        })
        out = coerce_returns(df)
        assert out["date"].n_unique() == out.height

    def test_rejects_null_returns(self):
        """Null returns surface as ValueError rather than silently dropping
        through the (1+r).product() aggregate (polars skips nulls by default,
        which would produce a misleading daily figure)."""
        df = pl.DataFrame({
            "date": [
                dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2024, 1, 3),
            ],
            "return": [0.01, None, -0.01],
        })
        with pytest.raises(ValueError, match="null"):
            _ensure_daily(df)
