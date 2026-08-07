"""The two `_flatten.py` defects found in the 0.16.0 review: #82 and #84.

Both are about *duplication of state that should have had one owner*.

#82 — the schedule cache keyed itself on ``id(calendar)``, which forced a
parallel list of pinned calendars to exist so the ids could not be recycled.
``get_calendar`` returns a fresh object per call, so the key never hit and both
structures grew one entry per run.

#84 — ``_flatten._align_tz`` was a verbatim copy of
``mktlib.scheduling._mixins._align_tz`` taken *before* the latter grew its
``convert_time_zone`` branch (``CHANGELOG.md`` records that fix). The copy
therefore re-introduced a ``SchemaError`` that had already been fixed once.
The test that matters most here is not either behaviour test but
``test_flatten_reuses_the_scheduling_helper`` — the drift is the defect.
"""

from __future__ import annotations

import datetime
import gc
import weakref

import polars as pl

from mktlib.backtest._engine import run
from mktlib.backtest import _flatten
from mktlib.backtest._conditions import Crossover, Crossunder
from mktlib.backtest._flatten import FlattenSchedule, build_flatten_masks
from mktlib.scheduling import get_calendar
from mktlib.scheduling.calendar import ExchangeCalendar
from mktlib.scheduling.rules import HolidayRule

_TWO_SESSIONS = (datetime.date(2024, 1, 2), datetime.date(2024, 1, 3))

#: Tue…Fri of the same ISO week — used where the ``days="weekly"`` path needs
#: the week's closing session to actually be in the data.
_WEEK_TO_FRIDAY = (
    datetime.date(2024, 1, 2),
    datetime.date(2024, 1, 3),
    datetime.date(2024, 1, 4),
    datetime.date(2024, 1, 5),
)


class _Cross:
    """Enter on the crossover, never exit — the flatten owns the exit."""

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Crossunder:
        return Crossunder("never_a", "never_b")


def _naive_bars(
    days: tuple[datetime.date, ...] = _TWO_SESSIONS,
) -> list[datetime.datetime]:
    """15-minute bars, 09:30 … 15:45 New-York-local, tz-naive."""
    out: list[datetime.datetime] = []
    for day in days:
        ts = datetime.datetime(day.year, day.month, day.day, 9, 30)
        end = datetime.datetime(day.year, day.month, day.day, 15, 45)
        while ts <= end:
            out.append(ts)
            ts += datetime.timedelta(minutes=15)
    return out


def _frame(dates: pl.Series | list[datetime.datetime]) -> pl.DataFrame:
    dates = pl.Series("date", dates) if isinstance(dates, list) else dates
    n = dates.len()
    fast = [1.0] * n
    for i in range(2, n):
        fast[i] = 3.0
    return pl.DataFrame({
        "date": dates,
        "open": [100.0 + i * 0.1 for i in range(n)],
        "high": [100.6 + i * 0.1 for i in range(n)],
        "low": [99.5 + i * 0.1 for i in range(n)],
        "close": [100.05 + i * 0.1 for i in range(n)],
        "fast": fast,
        "slow": [2.0] * n,
        "never_a": [0.0] * n,
        "never_b": [5.0] * n,
    })


def _utc_frame() -> pl.DataFrame:
    """The same bars, stamped in UTC rather than left naive.

    XNYS is an ``America/New_York`` calendar, so this is the tz-aware ×
    tz-aware, *different zone* case — the branch ``_flatten._align_tz`` was
    missing.
    """
    naive = pl.Series("date", _naive_bars())
    utc = naive.dt.replace_time_zone("America/New_York").dt.convert_time_zone("UTC")
    return _frame(utc)


def _adhoc_calendar(
    *,
    close_time: datetime.time = datetime.time(16, 0),
    holidays: list[HolidayRule] | None = None,
) -> ExchangeCalendar:
    """Two of these share a *name* and differ in a field the schedule reads."""
    return ExchangeCalendar(
        "MYX",
        timezone="America/New_York",
        open_time=datetime.time(9, 30),
        close_time=close_time,
        holidays=holidays or [],
    )


# ---------------------------------------------------------------------------
# #84 — tz-aware bars against a tz-aware calendar
# ---------------------------------------------------------------------------


class TestAlignTz:
    def test_flatten_reuses_the_scheduling_helper(self) -> None:
        """One implementation of "align these bars to this calendar's zone".

        This is the fix for #84, stated directly. The behaviour test below
        pins the symptom; this pins the cause. A second copy of the helper is
        how the ``convert_time_zone`` branch came to be fixed in one place and
        missing in the other, and re-copying it would re-arm that trap without
        failing any behavioural test the day it happens.
        """
        from mktlib.scheduling import _mixins

        assert _flatten._align_tz is _mixins._align_tz

    def test_utc_bars_against_a_new_york_calendar_run(self) -> None:
        """``run(flatten="eod")`` on UTC bars raises ``SchemaError`` today."""
        result = run(
            _utc_frame(),
            _Cross(),
            calendar=get_calendar("XNYS"),
            flatten="eod",
        )
        assert result.returns.height > 0
        assert result.trades.height > 0

    def test_utc_bars_flatten_at_the_new_york_close(self) -> None:
        """The flatten bar is the session's last bar, in NY terms, not UTC.

        A ``replace_time_zone`` that reinterpreted 14:30 UTC as 14:30 New York
        would still produce *a* flatten bar; it would be the wrong one. The
        assertion is therefore on the timestamps, not on the count.
        """
        df = _utc_frame()
        mask = build_flatten_masks(
            df["date"], get_calendar("XNYS"), FlattenSchedule()
        )[0]
        flattened = (
            df.filter(mask)["date"]
            .dt.convert_time_zone("America/New_York")
            .dt.time()
            .to_list()
        )
        # XNYS closes at 16:00; the last 15-minute bar starts at 15:45.
        assert flattened == [datetime.time(15, 45)] * len(_TWO_SESSIONS)

    def test_naive_bars_still_flatten_at_the_close(self) -> None:
        """The accept-twin: the pre-existing naive path is untouched."""
        df = _frame(_naive_bars())
        mask = build_flatten_masks(
            df["date"], get_calendar("XNYS"), FlattenSchedule()
        )[0]
        flattened = df.filter(mask)["date"].dt.time().to_list()
        assert flattened == [datetime.time(15, 45)] * len(_TWO_SESSIONS)


# ---------------------------------------------------------------------------
# #82 — the schedule cache
# ---------------------------------------------------------------------------


class TestScheduleCache:
    def test_fresh_calendar_objects_share_one_cache_entry(self) -> None:
        """``get_calendar`` returns a new object per call; the cache must hit.

        Under the ``id(calendar)`` key this grows by one entry per iteration —
        200 schedules plus 200 pinned calendars — which is the leak #82
        reports. Under a value key every iteration hits the same entry.
        """
        df = _frame(_naive_bars())
        before = len(_flatten._schedule_cache)
        for _ in range(200):
            run(
                df,
                _Cross(),
                calendar=get_calendar("XNYS"),
                flatten="eod",
            )
        grown = len(_flatten._schedule_cache) - before
        assert grown <= 1, f"cache grew by {grown} entries over 200 runs"

    def test_fresh_calendar_objects_share_one_week_cache_entry(self) -> None:
        """The same, for the ``days="weekly"`` cache — it has the same key."""
        # Tue…Fri, so the week's closing session is present and the run does
        # not emit a "selects no session" warning 200 times.
        df = _frame(_naive_bars(_WEEK_TO_FRIDAY))
        before = len(_flatten._week_last_cache)
        for _ in range(200):
            run(
                df,
                _Cross(),
                calendar=get_calendar("XNYS"),
                flatten="eow",
            )
        grown = len(_flatten._week_last_cache) - before
        assert grown <= 1, f"week cache grew by {grown} entries over 200 runs"

    def test_the_cache_holds_no_strong_reference_to_a_calendar(self) -> None:
        """A calendar the caller has dropped must become collectable.

        ``_schedule_cache_pins`` held every calendar that ever produced an
        entry, forever, so this is the retention half of the leak: even a
        caller that reuses one calendar object could not get it back. A value
        key needs no pin, so the pin list should not exist at all.
        """
        assert not hasattr(_flatten, "_schedule_cache_pins")

        df = _frame(_naive_bars())
        cal = get_calendar("XNYS")
        run(df, _Cross(), calendar=cal, flatten="eod")
        ref = weakref.ref(cal)
        del cal
        gc.collect()
        assert ref() is None

    def test_calendars_differing_only_in_close_time_do_not_collide(self) -> None:
        """The accept-twin, and the hazard the ``id()`` key existed to avoid.

        Two ad-hoc calendars share the name ``MYX`` and differ only in their
        close. A key of ``(name, start, end)`` would hand the second one the
        first one's schedule and silently move its flatten bar. The value key
        has to be wide enough to tell them apart.
        """
        df = _frame(_naive_bars())
        late = build_flatten_masks(
            df["date"], _adhoc_calendar(close_time=datetime.time(16, 0)),
            FlattenSchedule(),
        )[0]
        early = build_flatten_masks(
            df["date"], _adhoc_calendar(close_time=datetime.time(12, 0)),
            FlattenSchedule(),
        )[0]
        assert df.filter(late)["date"].dt.time().to_list() == [
            datetime.time(15, 45)
        ] * len(_TWO_SESSIONS)
        assert df.filter(early)["date"].dt.time().to_list() == [
            datetime.time(11, 45)
        ] * len(_TWO_SESSIONS)

    def test_calendars_differing_only_in_holidays_do_not_collide(self) -> None:
        """The same completeness question, one field further out.

        ``schedule()`` reads holidays as well as hours, so a key built from
        hours alone is incomplete: two ``MYX`` calendars with the same session
        times and different holiday rules produce different sessions, and the
        second must not be served the first one's answer. This is the repo's
        recurring cache-key-completeness failure, asked of the new key before
        it ships rather than after.
        """
        df = _frame(_naive_bars())
        # 2024-01-03 is a Wednesday; closing the exchange that day removes its
        # session entirely, so only 01-02 can carry a flatten bar.
        jan3 = HolidayRule(name="ad-hoc", month=1, day=3)
        open_all = build_flatten_masks(
            df["date"], _adhoc_calendar(), FlattenSchedule()
        )[0]
        closed_jan3 = build_flatten_masks(
            df["date"], _adhoc_calendar(holidays=[jan3]), FlattenSchedule()
        )[0]
        assert df.filter(open_all)["date"].dt.date().to_list() == list(_TWO_SESSIONS)
        assert df.filter(closed_jan3)["date"].dt.date().to_list() == [
            _TWO_SESSIONS[0]
        ]
