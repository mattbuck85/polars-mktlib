from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from mktlib.scheduling._mixins import _align_tz, _tz
from mktlib.scheduling import get_calendar
from mktlib.scheduling.calendar import ExchangeCalendar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _series(dts: list[datetime], tz: str | None = None) -> pl.Series:
    """Build a Datetime series with optional timezone."""
    s = pl.Series("ts", dts)
    if tz is not None:
        s = s.dt.replace_time_zone(tz)
    return s


# ---------------------------------------------------------------------------
# Unit tests for _align_tz
# ---------------------------------------------------------------------------

class TestAlignTz:
    """Unit tests for the _align_tz helper."""

    def test_both_none_returns_unchanged(self):
        """Naive target + naive reference -> return target unchanged."""
        target = _series([datetime(2024, 1, 2, 10, 0)])
        reference = _series([datetime(2024, 1, 2, 9, 30)])
        result = _align_tz(target, reference)
        assert _tz(result) is None
        assert result.to_list() == target.to_list()

    def test_strip_tz_when_ref_naive(self):
        """Tz-aware target + naive reference -> strip timezone."""
        target = _series(
            [datetime(2024, 1, 2, 15, 0)], tz="America/New_York"
        )
        reference = _series([datetime(2024, 1, 2, 10, 0)])
        result = _align_tz(target, reference)
        assert _tz(result) is None

    def test_add_tz_when_tgt_naive(self):
        """Naive target + tz-aware reference -> add reference's timezone."""
        target = _series([datetime(2024, 1, 2, 10, 0)])
        reference = _series(
            [datetime(2024, 1, 2, 14, 30)], tz="America/New_York"
        )
        result = _align_tz(target, reference)
        assert _tz(result) == "America/New_York"

    def test_same_tz_returns_unchanged(self):
        """Both UTC -> return target unchanged."""
        target = _series([datetime(2024, 1, 2, 14, 30)], tz="UTC")
        reference = _series([datetime(2024, 1, 2, 14, 30)], tz="UTC")
        result = _align_tz(target, reference)
        assert _tz(result) == "UTC"
        assert result.to_list() == target.to_list()

    def test_different_tz_converts(self):
        """target=America/New_York, ref=UTC -> convert target to UTC.

        This is the bug: the current code falls through and returns the
        target unchanged, causing Polars to raise SchemaError on
        cross-timezone comparison.
        """
        target = _series(
            [datetime(2024, 1, 2, 9, 30)], tz="America/New_York"
        )
        reference = _series([datetime(2024, 1, 2, 14, 30)], tz="UTC")
        result = _align_tz(target, reference)
        assert _tz(result) == "UTC"

    def test_convert_preserves_moment(self):
        """After conversion the underlying instant is the same.

        09:30 America/New_York on 2024-01-02 == 14:30 UTC.
        """
        target = _series(
            [datetime(2024, 1, 2, 9, 30)], tz="America/New_York"
        )
        reference = _series([datetime(2024, 1, 2, 14, 30)], tz="UTC")
        result = _align_tz(target, reference)
        expected_utc = datetime(2024, 1, 2, 14, 30, tzinfo=ZoneInfo("UTC"))
        actual = result.to_list()[0]
        assert actual == expected_utc


# ---------------------------------------------------------------------------
# Integration test: filter_market_hours with UTC bars
# ---------------------------------------------------------------------------

class TestFilterMarketHoursUtcBars:
    """filter_market_hours must work when bars are in UTC and the
    calendar is in America/New_York (NYSE)."""

    @pytest.fixture
    def nyse(self):
        return get_calendar("XNYS")

    def test_filter_market_hours_utc_bars(self, nyse: ExchangeCalendar):
        """UTC bars covering a full day should keep only those falling
        inside NYSE market hours (09:30-16:00 ET = 14:30-21:00 UTC).

        On 2024-01-02 (first trading day of 2024):
        - 14:00 UTC = 09:00 ET  -> pre-market, excluded
        - 14:30 UTC = 09:30 ET  -> market open, included
        - 18:00 UTC = 13:00 ET  -> mid-day, included
        - 20:59 UTC = 15:59 ET  -> last tradeable minute, included
        - 21:00 UTC = 16:00 ET  -> at close, excluded (open-frame)
        - 22:00 UTC = 17:00 ET  -> post-market, excluded
        """
        bars = [
            datetime(2024, 1, 2, 14, 0, tzinfo=ZoneInfo("UTC")),   # pre
            datetime(2024, 1, 2, 14, 30, tzinfo=ZoneInfo("UTC")),  # open
            datetime(2024, 1, 2, 18, 0, tzinfo=ZoneInfo("UTC")),   # mid
            datetime(2024, 1, 2, 20, 59, tzinfo=ZoneInfo("UTC")),  # last min
            datetime(2024, 1, 2, 21, 0, tzinfo=ZoneInfo("UTC")),   # close
            datetime(2024, 1, 2, 22, 0, tzinfo=ZoneInfo("UTC")),   # post
        ]
        df = pl.DataFrame({"date": bars})
        result = nyse.filter_market_hours(df)

        expected_times_utc = [
            datetime(2024, 1, 2, 14, 30, tzinfo=ZoneInfo("UTC")),
            datetime(2024, 1, 2, 18, 0, tzinfo=ZoneInfo("UTC")),
            datetime(2024, 1, 2, 20, 59, tzinfo=ZoneInfo("UTC")),
        ]
        assert result.height == 3
        assert result["date"].to_list() == expected_times_utc
