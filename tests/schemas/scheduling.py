"""Validation helpers for mktlib.scheduling output DataFrames.

Timezone parameterization makes a static DataFrameModel impractical,
so this module provides assertion-based helpers instead.
"""

from __future__ import annotations

import polars as pl


def validate_schedule(
    df: pl.DataFrame,
    tz: str,
    has_breaks: bool = False,
) -> None:
    """Assert structural invariants of an exchange schedule DataFrame."""
    assert df.columns[:3] == ["date", "market_open", "market_close"]
    assert df["date"].dtype == pl.Date
    assert df["market_open"].dtype == pl.Datetime("us", tz)
    assert df["market_close"].dtype == pl.Datetime("us", tz)
    assert (df["market_close"] > df["market_open"]).all()
    if has_breaks:
        assert "break_start" in df.columns
        assert "break_end" in df.columns
        assert df["break_start"].dtype == pl.Datetime("us", tz)
        assert df["break_end"].dtype == pl.Datetime("us", tz)


def validate_valid_days(days: pl.Series) -> None:
    """Assert structural invariants of valid_days() output."""
    assert days.dtype == pl.Date
    assert len(days) > 0
    # Days should be sorted ascending
    assert days.is_sorted()
