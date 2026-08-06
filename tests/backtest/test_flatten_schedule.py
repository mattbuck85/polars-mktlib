"""Flatten scheduling: the mask, the schedule types, and the engine wiring.

The invariant tests here are load-bearing for code that was *deleted* rather
than kept. See ``test_position_is_always_zero_on_the_flatten_bar``.
"""

from __future__ import annotations

import datetime
import logging

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mktlib.backtest._conditions import Crossover, Crossunder
from mktlib.backtest._engine import run
from mktlib.backtest._flatten import (
    FlattenSchedule,
    Weekday,
    build_flatten_masks,
    resolve_flatten,
)
from mktlib.scheduling import get_calendar

# 15-minute bars, 09:30 … 15:45 inclusive. XNYS closes at 16:00, so the last
# bar of each session starts 15 minutes before the close.
_BARS_PER_SESSION = 26

_TWO_SESSIONS = (datetime.date(2024, 1, 2), datetime.date(2024, 1, 3))


class _Cross:
    """Enter on the crossover, never exit on its own — the flatten owns the exit."""

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Crossunder:
        return Crossunder("never_a", "never_b")


def _sessions(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    """Real trading days from the calendar — never a hand-written weekday list."""
    return get_calendar("XNYS").schedule(start, end)["date"].to_list()


def _dates(days: tuple[datetime.date, ...] | list[datetime.date]) -> list[datetime.datetime]:
    out: list[datetime.datetime] = []
    for day in days:
        ts = datetime.datetime(day.year, day.month, day.day, 9, 30)
        end = datetime.datetime(day.year, day.month, day.day, 15, 45)
        while ts <= end:
            out.append(ts)
            ts += datetime.timedelta(minutes=15)
    return out


def _frame(
    days: tuple[datetime.date, ...] | list[datetime.date],
    entry_signal_bar: int = 2,
) -> pl.DataFrame:
    dates = _dates(days)
    n = len(dates)
    fast = [1.0] * n
    for i in range(entry_signal_bar, n):
        fast[i] = 3.0
    return pl.DataFrame(
        {
            "date": dates,
            "open": [100.0 + i * 0.1 for i in range(n)],
            "high": [100.6 + i * 0.1 for i in range(n)],
            "low": [99.5 + i * 0.1 for i in range(n)],
            "close": [100.05 + i * 0.1 for i in range(n)],
            "fast": fast,
            "slow": [2.0] * n,
            "never_a": [0.0] * n,
            "never_b": [5.0] * n,
        }
    )


def _flatten_dates(df: pl.DataFrame, schedule: FlattenSchedule) -> list[datetime.datetime]:
    """The bar timestamps this schedule force-closes on."""
    mask = build_flatten_masks(df["date"], get_calendar("XNYS"), schedule)[0].to_list()
    return [d for d, m in zip(df["date"].to_list(), mask, strict=True) if m]


# ---------------------------------------------------------------------------
# Invariants the deleted `& ~_is_entry_bar` guard rested on
# ---------------------------------------------------------------------------


class TestFlattenBarInvariant:
    def test_position_is_always_zero_on_the_flatten_bar(self) -> None:
        """``_position`` is structurally 0 on every flatten bar.

        This is why the post-flatten return zeroing needs no ``_is_entry_bar``
        guard: at flatten+1 ``_pos_d1`` is 0, so ``_is_entry_bar`` — which
        requires ``_pos_d1 == 1`` — cannot be true there.

        Revert-check (performed 2026-08-06): changing the ``_position``
        expression's exit branch from ``_exit | flatten_bar`` to ``_exit``
        turns this red with "_position must be 0 on flatten bar 25
        (2024-01-02 15:45:00), got 1". The assertion discriminates.
        """
        cal = get_calendar("XNYS")
        df = _frame(_TWO_SESSIONS, entry_signal_bar=2)
        result = run(df, _Cross(), calendar=cal, flatten_eod=True)

        mask = build_flatten_masks(df["date"], cal, FlattenSchedule())[0].to_list()
        positions = result.signals["_position"].to_list()

        flatten_idx = [i for i, m in enumerate(mask) if m]
        assert flatten_idx == [_BARS_PER_SESSION - 1, 2 * _BARS_PER_SESSION - 1], (
            "fixture assumption: one flatten bar at the end of each session"
        )
        for i in flatten_idx:
            assert positions[i] == 0, (
                f"_position must be 0 on flatten bar {i} "
                f"({df['date'][i]}), got {positions[i]}"
            )

    def test_a_position_is_actually_held_before_the_flatten_bar(self) -> None:
        """Guards the test above against passing because nothing ever opens."""
        cal = get_calendar("XNYS")
        df = _frame(_TWO_SESSIONS, entry_signal_bar=2)
        result = run(df, _Cross(), calendar=cal, flatten_eod=True)
        assert sum(result.signals["_position"].to_list()) > 0, (
            "fixture opened no position; the invariant is vacuous"
        )


# ---------------------------------------------------------------------------
# Schedule type
# ---------------------------------------------------------------------------


class TestFlattenScheduleType:
    def test_default_is_daily(self) -> None:
        assert FlattenSchedule().days == "daily"

    def test_weekday_set_normalizes_to_frozenset_and_is_hashable(self) -> None:
        a = FlattenSchedule(days={5})
        b = FlattenSchedule(days={Weekday.FRI})
        c = FlattenSchedule(days=frozenset({Weekday.FRI}))
        assert a == b == c
        assert hash(a) == hash(b) == hash(c)
        assert {a: 1}[c] == 1, "must be usable as a cache key"
        assert isinstance(a.days, frozenset)

    def test_empty_weekday_set_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="selects no session"):
            FlattenSchedule(days=frozenset())

    def test_unknown_day_string_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not valid"):
            FlattenSchedule(days="monthly")  # type: ignore[arg-type]

    def test_out_of_range_weekday_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not valid"):
            FlattenSchedule(days={9})


class TestResolveFlatten:
    @pytest.mark.parametrize(
        ("flatten", "flatten_eod", "expected"),
        [
            (None, False, None),
            (None, True, FlattenSchedule(days="daily")),
            (False, False, None),
            (True, False, FlattenSchedule(days="daily")),
            ("eod", False, FlattenSchedule(days="daily")),
            ("eow", False, FlattenSchedule(days="weekly")),
        ],
    )
    def test_resolution_table(
        self, flatten: object, flatten_eod: bool, expected: FlattenSchedule | None
    ) -> None:
        assert resolve_flatten(flatten, flatten_eod=flatten_eod) == expected  # type: ignore[arg-type]

    def test_both_spellings_together_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot both be set"):
            resolve_flatten("eod", flatten_eod=True)

    def test_unknown_alias_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a known alias"):
            resolve_flatten("eom", flatten_eod=False)  # type: ignore[arg-type]

    def test_wrong_type_is_refused(self) -> None:
        with pytest.raises(TypeError, match="must be a bool"):
            resolve_flatten(42, flatten_eod=False)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Day selection
# ---------------------------------------------------------------------------


class TestDaySelection:
    def test_daily_marks_the_last_bar_of_every_session(self) -> None:
        df = _frame(_TWO_SESSIONS)
        marks = _flatten_dates(df, FlattenSchedule(days="daily"))
        assert marks == [
            datetime.datetime(2024, 1, 2, 15, 45),
            datetime.datetime(2024, 1, 3, 15, 45),
        ]

    def test_weekly_marks_only_the_last_session_of_each_week(self) -> None:
        # Tue 2024-01-02 … Fri 2024-01-12: two full weeks (Jan 1 is a holiday).
        days = _sessions(datetime.date(2024, 1, 2), datetime.date(2024, 1, 12))
        df = _frame(days)
        marks = _flatten_dates(df, FlattenSchedule(days="weekly"))
        assert marks == [
            datetime.datetime(2024, 1, 5, 15, 45),   # Friday, week 1
            datetime.datetime(2024, 1, 12, 15, 45),  # Friday, week 2
        ]

    def test_weekly_falls_back_to_thursday_when_friday_is_a_holiday(self) -> None:
        """Good Friday 2024-03-29. ``weekly`` is holiday-aware for free."""
        days = _sessions(datetime.date(2024, 3, 25), datetime.date(2024, 3, 29))
        assert datetime.date(2024, 3, 29) not in days, "fixture: Good Friday is closed"
        df = _frame(days)
        marks = _flatten_dates(df, FlattenSchedule(days="weekly"))
        assert marks == [datetime.datetime(2024, 3, 28, 15, 45)]

    def test_explicit_friday_set_does_not_flatten_a_friday_holiday_week(self) -> None:
        """The documented literal-vs-aware split, pinned side by side with the above."""
        days = _sessions(datetime.date(2024, 3, 25), datetime.date(2024, 3, 29))
        df = _frame(days)
        assert _flatten_dates(df, FlattenSchedule(days={Weekday.FRI})) == []

    def test_explicit_weekday_set_selects_that_weekday(self) -> None:
        days = _sessions(datetime.date(2024, 1, 2), datetime.date(2024, 1, 12))
        df = _frame(days)
        marks = _flatten_dates(df, FlattenSchedule(days={Weekday.WED}))
        assert marks == [
            datetime.datetime(2024, 1, 3, 15, 45),
            datetime.datetime(2024, 1, 10, 15, 45),
        ]

    def test_iso_week_rollover_does_not_merge_december_into_january(self) -> None:
        """The classic ``year * 100 + week`` bug: Dec 29 - Jan 3 spans a year."""
        days = _sessions(datetime.date(2026, 12, 28), datetime.date(2027, 1, 6))
        df = _frame(days)
        marks = _flatten_dates(df, FlattenSchedule(days="weekly"))
        # Week of Mon 2026-12-28 ends on its last *trading* day; the following
        # ISO week is a separate group with its own flatten.
        assert len(marks) == 2, marks
        assert marks[0].date() >= datetime.date(2026, 12, 30)
        assert marks[0].date() <= datetime.date(2027, 1, 1)
        assert marks[1].date() >= datetime.date(2027, 1, 4)

    def test_weekly_flattens_on_a_final_partial_week(self) -> None:
        # Mon 2024-01-08 … Wed 2024-01-10 — the week never reaches Friday.
        days = _sessions(datetime.date(2024, 1, 8), datetime.date(2024, 1, 10))
        df = _frame(days)
        marks = _flatten_dates(df, FlattenSchedule(days="weekly"))
        assert marks == [datetime.datetime(2024, 1, 10, 15, 45)]

    def test_schedule_selecting_no_session_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = _frame(_TWO_SESSIONS)
        with caplog.at_level(logging.WARNING, logger="mktlib.backtest._flatten"):
            marks = _flatten_dates(df, FlattenSchedule(days={Weekday.SUN}))
        assert marks == []
        assert "selects no session" in caplog.text


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------


class TestEngineWiring:
    def test_eod_alias_is_byte_identical_to_flatten_eod(self) -> None:
        cal = get_calendar("XNYS")
        df = _frame(_TWO_SESSIONS)
        legacy = run(df, _Cross(), calendar=cal, flatten_eod=True)
        modern = run(df, _Cross(), calendar=cal, flatten="eod")
        assert_frame_equal(legacy.returns, modern.returns)
        assert_frame_equal(legacy.trades, modern.trades)
        assert_frame_equal(legacy.signals, modern.signals)

    def test_flatten_false_is_byte_identical_to_no_flatten(self) -> None:
        cal = get_calendar("XNYS")
        df = _frame(_TWO_SESSIONS)
        a = run(df, _Cross(), calendar=cal)
        b = run(df, _Cross(), calendar=cal, flatten=False)
        assert_frame_equal(a.returns, b.returns)
        assert_frame_equal(a.trades, b.trades)

    def test_weekly_holds_a_position_across_a_session_boundary(self) -> None:
        """The whole point of ``eow``: overnight exposure inside the week."""
        cal = get_calendar("XNYS")
        days = _sessions(datetime.date(2024, 1, 2), datetime.date(2024, 1, 5))
        df = _frame(days, entry_signal_bar=2)
        eod = run(df, _Cross(), calendar=cal, flatten="eod")
        eow = run(df, _Cross(), calendar=cal, flatten="eow")
        assert eow.trades.height == 1
        assert eow.trades["exit_date"][0] == datetime.datetime(2024, 1, 5, 15, 45)
        assert eow.trades["bars_held"][0] > eod.trades["bars_held"][0]

    def test_flatten_and_flatten_eod_together_is_refused(self) -> None:
        cal = get_calendar("XNYS")
        df = _frame(_TWO_SESSIONS)
        with pytest.raises(ValueError, match="cannot both be set"):
            run(df, _Cross(), calendar=cal, flatten="eod", flatten_eod=True)

    def test_flatten_schedule_without_calendar_is_refused(self) -> None:
        df = _frame(_TWO_SESSIONS)
        with pytest.raises(ValueError, match=r"flatten=\.\.\..*requires a calendar"):
            run(df, _Cross(), flatten="eow")

    def test_legacy_flatten_eod_error_still_names_flatten_eod(self) -> None:
        """A caller who wrote flatten_eod must not be told about `flatten`."""
        df = _frame(_TWO_SESSIONS)
        with pytest.raises(ValueError, match="flatten_eod=True requires a calendar"):
            run(df, _Cross(), flatten_eod=True)


class TestNoInternalColumnLeak:
    @pytest.mark.parametrize("flatten", [None, "eod", "eow"])
    def test_flatten_bar_column_never_reaches_the_caller(
        self, flatten: str | None
    ) -> None:
        cal = get_calendar("XNYS")
        df = _frame(_TWO_SESSIONS)
        result = run(df, _Cross(), calendar=cal, flatten=flatten)  # type: ignore[arg-type]
        assert "_flatten_bar" not in result.signals.columns
