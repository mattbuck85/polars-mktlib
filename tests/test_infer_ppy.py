"""Tests for _infer_ppy helper in mktlib.reports."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import polars as pl
import pytest

from mktlib.reports import _infer_ppy


class TestInferPPY:
    def test_daily_data_returns_252(self):
        """1 bar per day → 252."""
        dates = [datetime(2024, 1, i + 1, tzinfo=timezone.utc) for i in range(20)]
        df = pl.DataFrame({"date": dates, "return": [0.01] * 20})
        assert _infer_ppy(df) == 252

    def test_minute_data_returns_approx_98280(self):
        """390 bars per day → 252 * 390 = 98280."""
        base = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
        dates = [base + timedelta(minutes=i) for i in range(390)]
        df = pl.DataFrame({"date": dates, "return": [0.001] * 390})
        assert _infer_ppy(df) == 252 * 390

    def test_multi_day_minute_data(self):
        """Two days of 390 bars each → 252 * 390."""
        day1 = [datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(390)]
        day2 = [datetime(2024, 1, 3, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(390)]
        dates = day1 + day2
        df = pl.DataFrame({"date": dates, "return": [0.001] * 780})
        assert _infer_ppy(df) == 252 * 390

    def test_five_minute_data(self):
        """78 bars per day (5-min) → 252 * 78 = 19656."""
        base = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
        dates = [base + timedelta(minutes=i * 5) for i in range(78)]
        df = pl.DataFrame({"date": dates, "return": [0.001] * 78})
        assert _infer_ppy(df) == 252 * 78

    def test_empty_dataframe_returns_252(self):
        """Empty df → fallback 252."""
        df = pl.DataFrame({"date": [], "return": []}).cast({"date": pl.Datetime("us", "UTC"), "return": pl.Float64})
        assert _infer_ppy(df) == 252

    def test_missing_date_column_returns_252(self):
        """No 'date' column → fallback 252."""
        df = pl.DataFrame({"return": [0.01, 0.02]})
        assert _infer_ppy(df) == 252

    def test_uneven_bars_uses_median(self):
        """Days with different bar counts → uses median."""
        # Day 1: 390 bars, Day 2: 380 bars → median = 385
        day1 = [datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(390)]
        day2 = [datetime(2024, 1, 3, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(380)]
        dates = day1 + day2
        df = pl.DataFrame({"date": dates, "return": [0.001] * 770})
        ppy = _infer_ppy(df)
        assert ppy == 252 * 385

    def test_single_bar_returns_252(self):
        """Single bar → 1 bar/day → 252."""
        df = pl.DataFrame({
            "date": [datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)],
            "return": [0.01],
        })
        assert _infer_ppy(df) == 252
