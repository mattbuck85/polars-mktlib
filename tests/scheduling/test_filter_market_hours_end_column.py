"""Tests for the optional ``end_column`` on ``Calendar.filter_market_hours``.

The default predicate keeps rows whose label lies in
``[market_open, market_close - 1min]``, which encodes a one-minute,
left-labelled bar.  Frames whose bars have variable duration -- volume bars,
dollar bars, tick bars -- lose the last rows of every session, because a bar
that *starts* inside the final minute but *ends* at or before the close is
labelled after ``market_close - 1min``.  ``end_column`` filters by the bar's
end instead, so those rows survive.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from mktlib.scheduling import get_calendar
from mktlib.scheduling.calendar import ExchangeCalendar, ExchangeCalendarWithBreaks

NY = ZoneInfo("America/New_York")
TOKYO = ZoneInfo("Asia/Tokyo")


@pytest.fixture
def nyse():
    return get_calendar("XNYS")


@pytest.fixture
def jpx():
    return get_calendar("XTKS")


def _bars(spans: list[tuple[datetime, datetime]]) -> pl.DataFrame:
    """A variable-duration bar frame: one ``date`` label plus its ``end``."""
    return pl.DataFrame(
        {
            "date": [start for start, _ in spans],
            "end": [end for _, end in spans],
        }
    )


class TestEndColumn:
    def test_variable_width_bars_keep_the_closing_rows_when_an_end_column_is_given(
        self, nyse: ExchangeCalendar
    ):
        """Three volume bars open inside the final minute and close by 16:00.

        NYSE 2024-01-02 closes at 16:00, so the final minute is
        15:59:00-16:00:00.  All three bars start strictly inside it, so the
        default label predicate (``date <= 15:59:00``) drops all three even
        though every one of them ends at or before the close.
        """
        spans = [
            # ordinary in-session bars, kept either way
            (datetime(2024, 1, 2, 9, 30, 0, tzinfo=NY), datetime(2024, 1, 2, 9, 31, 0, tzinfo=NY)),
            (datetime(2024, 1, 2, 12, 0, 0, tzinfo=NY), datetime(2024, 1, 2, 12, 0, 30, tzinfo=NY)),
            # three bars opening inside the final minute
            (datetime(2024, 1, 2, 15, 59, 10, tzinfo=NY), datetime(2024, 1, 2, 15, 59, 30, tzinfo=NY)),
            (datetime(2024, 1, 2, 15, 59, 30, tzinfo=NY), datetime(2024, 1, 2, 15, 59, 50, tzinfo=NY)),
            # ends exactly at the close: in-session, end is inclusive
            (datetime(2024, 1, 2, 15, 59, 50, tzinfo=NY), datetime(2024, 1, 2, 16, 0, 0, tzinfo=NY)),
        ]
        df = _bars(spans)

        without = nyse.filter_market_hours(df)
        assert without.height == 2, "today's behaviour: the three closing bars are dropped"

        with_end = nyse.filter_market_hours(df, end_column="end")
        assert with_end.height == 5
        assert with_end["date"].to_list() == [start for start, _ in spans]

    def test_a_bar_ending_after_the_close_is_dropped_even_if_it_starts_before_it(
        self, nyse: ExchangeCalendar
    ):
        """The end check must discriminate in the other direction too.

        Without this, ``end_column`` would degenerate into "keep everything
        that starts in session", which would admit a bar spilling past the
        close -- exactly the lookahead the close bound exists to prevent.
        """
        spans = [
            (datetime(2024, 1, 2, 15, 59, 0, tzinfo=NY), datetime(2024, 1, 2, 16, 0, 0, tzinfo=NY)),
            # starts in session, ends 30s after the close
            (datetime(2024, 1, 2, 15, 59, 50, tzinfo=NY), datetime(2024, 1, 2, 16, 0, 30, tzinfo=NY)),
        ]
        df = _bars(spans)

        result = nyse.filter_market_hours(df, end_column="end")
        assert result.height == 1
        assert result["date"].to_list() == [spans[0][0]]

    def test_minute_bars_are_byte_identical_with_and_without_the_end_column(
        self, nyse: ExchangeCalendar
    ):
        """Regular 1-minute bars: the two predicates agree exactly.

        A minute bar labelled 15:59 ends at 16:00 and is in-session under
        both rules; one labelled 16:00 ends at 16:01 and is out under both.
        """
        start = datetime(2024, 1, 2, 8, 0, tzinfo=NY)
        labels = [start + timedelta(minutes=i) for i in range(9 * 60)]  # 08:00-17:00
        df = pl.DataFrame(
            {"date": labels, "end": [d + timedelta(minutes=1) for d in labels]}
        )

        without = nyse.filter_market_hours(df)
        with_end = nyse.filter_market_hours(df, end_column="end")

        assert without.height == with_end.height
        assert without.equals(with_end)

    def test_the_default_path_is_unchanged(self, nyse: ExchangeCalendar):
        """An ``end`` column present but not passed must not change anything.

        Guards against the fix reading the column by name rather than only
        when the caller opts in.
        """
        spans = [
            (datetime(2024, 1, 2, 10, 0, 0, tzinfo=NY), datetime(2024, 1, 2, 10, 0, 30, tzinfo=NY)),
            (datetime(2024, 1, 2, 15, 59, 0, tzinfo=NY), datetime(2024, 1, 2, 15, 59, 30, tzinfo=NY)),
            (datetime(2024, 1, 2, 15, 59, 40, tzinfo=NY), datetime(2024, 1, 2, 16, 0, 0, tzinfo=NY)),
            (datetime(2024, 1, 2, 16, 0, 0, tzinfo=NY), datetime(2024, 1, 2, 16, 0, 30, tzinfo=NY)),
        ]
        df = _bars(spans)

        result = nyse.filter_market_hours(df)
        assert result.height == 2
        assert result["date"].to_list() == [spans[0][0], spans[1][0]]
        assert result.columns == ["date", "end"]

    def test_break_calendar_still_excludes_the_lunch_break_with_an_end_column(
        self, jpx: ExchangeCalendarWithBreaks
    ):
        """JPX 2024-01-04: 09:00-11:30, break, 12:30-15:00.

        ``end_column`` must widen only the session close, not re-admit bars
        sitting inside the lunch break.
        """
        spans = [
            (datetime(2024, 1, 4, 9, 30, 0, tzinfo=TOKYO), datetime(2024, 1, 4, 9, 30, 20, tzinfo=TOKYO)),
            # squarely inside the lunch break -- must stay excluded
            (datetime(2024, 1, 4, 11, 45, 0, tzinfo=TOKYO), datetime(2024, 1, 4, 11, 50, 0, tzinfo=TOKYO)),
            (datetime(2024, 1, 4, 12, 0, 0, tzinfo=TOKYO), datetime(2024, 1, 4, 12, 10, 0, tzinfo=TOKYO)),
            (datetime(2024, 1, 4, 12, 30, 0, tzinfo=TOKYO), datetime(2024, 1, 4, 12, 30, 30, tzinfo=TOKYO)),
            # opens inside the final minute, closes exactly at 15:00
            (datetime(2024, 1, 4, 14, 59, 40, tzinfo=TOKYO), datetime(2024, 1, 4, 15, 0, 0, tzinfo=TOKYO)),
        ]
        df = _bars(spans)

        result = jpx.filter_market_hours(df, end_column="end")

        kept = result["date"].to_list()
        assert spans[1][0] not in kept, "lunch-break bar must remain excluded"
        assert spans[2][0] not in kept, "lunch-break bar must remain excluded"
        assert spans[0][0] in kept
        assert spans[3][0] in kept
        assert spans[4][0] in kept, "closing bar must be retained via end_column"
        assert result.height == 3


class TestCallerColumnCollision:
    """A caller column must never collide with the filter's internals.

    ``filter_market_hours`` adds a join key to the caller's frame and joins
    schedule columns onto it. If the caller already carries a column of the
    same name, Polars suffixes the *incoming* one, so the mask reads the
    caller's column instead of the schedule's and the predicate silently
    compares against unrelated data. The internals are therefore prefixed
    ``__fmh_``, which is why these names are safe to hold.
    """

    @pytest.mark.parametrize(
        "colliding", ["_upper", "_mkt_open", "_bar_date", "_last_min", "_upper_bound"]
    )
    def test_a_caller_column_named_like_an_internal_is_ignored(
        self, nyse: ExchangeCalendar, colliding: str
    ):
        """The default path must return the same rows whatever the extra column."""
        bars = [
            datetime(2024, 1, 2, 8, 0, tzinfo=NY),  # pre-market
            datetime(2024, 1, 2, 10, 0, tzinfo=NY),  # in session
            datetime(2024, 1, 2, 17, 0, tzinfo=NY),  # post-market
        ]
        clean = pl.DataFrame({"date": bars})
        collided = clean.with_columns(pl.lit("x").alias(colliding))

        expected = nyse.filter_market_hours(clean)
        result = nyse.filter_market_hours(collided)

        assert result.height == expected.height == 1
        assert result["date"].to_list() == expected["date"].to_list()
        assert colliding in result.columns

    @pytest.mark.parametrize("colliding", ["_upper", "_mkt_open", "_bar_date"])
    def test_a_caller_column_named_like_an_internal_is_ignored_with_end_column(
        self, nyse: ExchangeCalendar, colliding: str
    ):
        """Same guarantee on the ``end_column`` path."""
        spans = [
            (datetime(2024, 1, 2, 10, 0, 0, tzinfo=NY), datetime(2024, 1, 2, 10, 0, 30, tzinfo=NY)),
            (datetime(2024, 1, 2, 15, 59, 50, tzinfo=NY), datetime(2024, 1, 2, 16, 0, 0, tzinfo=NY)),
            (datetime(2024, 1, 2, 17, 0, 0, tzinfo=NY), datetime(2024, 1, 2, 17, 0, 30, tzinfo=NY)),
        ]
        clean = _bars(spans)
        collided = clean.with_columns(pl.lit("x").alias(colliding))

        expected = clean.pipe(lambda d: nyse.filter_market_hours(d, end_column="end"))
        result = nyse.filter_market_hours(collided, end_column="end")

        assert result.height == expected.height == 2
        assert result["date"].to_list() == expected["date"].to_list()


class TestStraddlingAndNullEnds:
    def test_a_bar_straddling_the_lunch_break_is_kept(
        self, jpx: ExchangeCalendarWithBreaks
    ):
        """A bar that opens before the break and closes after it is KEPT.

        This is the documented contract, not an accident, and it is reachable
        only with variable-duration bars: a fixed-width minute bar cannot
        span JPX's 60-minute break, so the label-based break test never had
        to decide the case. Such a bar carries break-period activity into the
        session. Both the default and the ``end_column`` path keep it, since
        the break exclusion tests the *label*, which sits in the morning.
        """
        spans = [
            # opens 11:00 (morning), closes 13:00 (afternoon) -- straddles
            (datetime(2024, 1, 4, 11, 0, 0, tzinfo=TOKYO), datetime(2024, 1, 4, 13, 0, 0, tzinfo=TOKYO)),
            # squarely inside the break, for contrast: dropped
            (datetime(2024, 1, 4, 11, 45, 0, tzinfo=TOKYO), datetime(2024, 1, 4, 11, 50, 0, tzinfo=TOKYO)),
        ]
        df = _bars(spans)

        with_end = jpx.filter_market_hours(df, end_column="end")
        assert with_end["date"].to_list() == [spans[0][0]]

        default = jpx.filter_market_hours(df)
        assert default["date"].to_list() == [spans[0][0]]

    def test_a_row_whose_end_is_null_is_dropped(self, nyse: ExchangeCalendar):
        """A null in *end_column* drops the row rather than keeping it.

        ``null <= close`` is null, so the mask is null and the row is
        filtered out. That is the safe direction -- an unbounded bar is not
        asserted to be in-session -- but it is silent, so it is pinned here.
        """
        df = pl.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 2, 10, 0, tzinfo=NY),
                    datetime(2024, 1, 2, 11, 0, tzinfo=NY),
                ],
                "end": [
                    datetime(2024, 1, 2, 10, 0, 30, tzinfo=NY),
                    None,
                ],
            },
            schema={
                "date": pl.Datetime("us", "America/New_York"),
                "end": pl.Datetime("us", "America/New_York"),
            },
        )

        result = nyse.filter_market_hours(df, end_column="end")
        assert result.height == 1
        assert result["date"].to_list() == [datetime(2024, 1, 2, 10, 0, tzinfo=NY)]
