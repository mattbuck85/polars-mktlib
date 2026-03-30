"""Tests for _infer_ppy helper in mktlib.reports."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import polars as pl

from mktlib.reports import _infer_ppy


class TestInferPPY:
    def test_daily_data_returns_252(self):
        """~100 rows → daily bucket (252)."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        dates = [base + timedelta(days=i) for i in range(100)]
        df = pl.DataFrame({"date": dates, "return": [0.01] * 100})
        assert _infer_ppy(df) == 252

    def test_minute_data_returns_98280(self):
        """780 rows (> 400) → minutely bucket (98280)."""
        day1 = [datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(390)]
        day2 = [datetime(2024, 1, 3, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(390)]
        dates = day1 + day2
        df = pl.DataFrame({"date": dates, "return": [0.001] * 780})
        assert _infer_ppy(df) == 252 * 390

    def test_weekly_data(self):
        """20 rows (≤ 52) → weekly bucket (52)."""
        dates = [datetime(2024, 1, i + 1, tzinfo=timezone.utc) for i in range(20)]
        df = pl.DataFrame({"date": dates, "return": [0.01] * 20})
        assert _infer_ppy(df) == 52

    def test_boundary_52_is_weekly(self):
        """Exactly 52 rows → weekly bucket."""
        dates = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(weeks=i) for i in range(52)]
        df = pl.DataFrame({"date": dates, "return": [0.01] * 52})
        assert _infer_ppy(df) == 52

    def test_boundary_53_is_daily(self):
        """53 rows → daily bucket."""
        dates = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(53)]
        df = pl.DataFrame({"date": dates, "return": [0.01] * 53})
        assert _infer_ppy(df) == 252

    def test_boundary_400_is_daily(self):
        """400 rows → still daily bucket."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        dates = [base + timedelta(days=i) for i in range(400)]
        df = pl.DataFrame({"date": dates, "return": [0.01] * 400})
        assert _infer_ppy(df) == 252

    def test_boundary_401_is_minutely(self):
        """401 rows → minutely bucket."""
        base = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
        dates = [base + timedelta(minutes=i) for i in range(401)]
        df = pl.DataFrame({"date": dates, "return": [0.001] * 401})
        assert _infer_ppy(df) == 252 * 390

    def test_empty_dataframe_returns_252(self):
        """Empty df → fallback 252."""
        df = pl.DataFrame({"date": [], "return": []}).cast({"date": pl.Datetime("us", "UTC"), "return": pl.Float64})
        assert _infer_ppy(df) == 252

    def test_single_bar(self):
        """1 row (≤ 52) → weekly bucket."""
        df = pl.DataFrame({
            "date": [datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)],
            "return": [0.01],
        })
        assert _infer_ppy(df) == 52
