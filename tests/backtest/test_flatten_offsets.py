"""``minutes_before_close`` and ``block_entry_minutes_before_close``.

The two offsets are independent. Together they answer "close the position
early" and "stop opening new ones" separately, which the single session-last
bar could not express.
"""

from __future__ import annotations

import datetime
import logging

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mktlib.backtest._conditions import Crossover, Crossunder, PriceIsAbove
from mktlib.backtest._cost import Cost
from mktlib.backtest._engine import run
from mktlib.backtest._flatten import FlattenSchedule, build_flatten_masks
from mktlib.scheduling import ExchangeCalendar, get_calendar


class _Cross:
    """Enter on the crossover, never exit on its own — the flatten owns the exit."""

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Crossunder:
        return Crossunder("never_a", "never_b")


class _Level:
    """Level entry: re-arms on **every** bar, so re-entry is always attempted.

    An edge entry (``Crossover``) fires once and never again, which makes it
    useless for testing that something *prevents* a re-entry — nothing was
    going to re-enter anyway. Any assertion about the block window stopping a
    re-open has to be made against an entry that would otherwise re-open.
    """

    def entry(self) -> PriceIsAbove:
        return PriceIsAbove("fast", "slow")

    def exit(self) -> Crossunder:
        return Crossunder("never_a", "never_b")


def _sessions(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    return get_calendar("XNYS").schedule(start, end)["date"].to_list()


def _frame(
    days: list[datetime.date],
    times: tuple[tuple[int, int], ...],
    signal_at: datetime.datetime | None = None,
) -> pl.DataFrame:
    """Frame on an arbitrary intraday grid; crossover fires once at *signal_at*."""
    dates = [
        datetime.datetime(d.year, d.month, d.day, hh, mm)
        for d in days
        for hh, mm in times
    ]
    n = len(dates)
    opens = [100.0 + i * 0.5 for i in range(n)]
    return pl.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": [o + 1.0 for o in opens],
            "low": [o - 1.0 for o in opens],
            "close": [o + 0.25 for o in opens],
            "fast": [3.0 if (signal_at is not None and d >= signal_at) else 1.0
                     for d in dates],
            "slow": [2.0] * n,
            "never_a": [0.0] * n,
            "never_b": [5.0] * n,
        }
    )


def _flatten_dates(
    df: pl.DataFrame, schedule: FlattenSchedule
) -> list[datetime.datetime]:
    mask = build_flatten_masks(df["date"], get_calendar("XNYS"), schedule)[0]
    return [d for d, m in zip(df["date"].to_list(), mask.to_list(), strict=True) if m]


def _blocked_dates(
    df: pl.DataFrame, schedule: FlattenSchedule
) -> list[datetime.datetime]:
    mask = build_flatten_masks(df["date"], get_calendar("XNYS"), schedule)[1]
    return [d for d, m in zip(df["date"].to_list(), mask.to_list(), strict=True) if m]


# A morning pair plus a dense tail, so there are real bars on either side of a
# cutoff. The 09:30 bar exists so a 10:00 signal is a genuine *crossover*: with
# the signal on the very first bar, `fast` starts above `slow` and never
# crosses, which makes every downstream assertion pass vacuously.
_TIMES = ((9, 30), (10, 0), (15, 0), (15, 15), (15, 30), (15, 45))
_TWO = (datetime.date(2024, 1, 2), datetime.date(2024, 1, 3))

#: A signal bar that is early enough to open a position and hold it into the
#: close, and late enough to be a real crossover.
_MORNING = datetime.datetime(2024, 1, 2, 10, 0)

#: A true 15-minute grid, 09:30 .. 15:45. Its last bar starts exactly at
#: ``close - 15``, which is what makes that bar both the flatten bar and the
#: first blocked bar under ``minutes_before_close=15``.
_T15 = tuple(
    ((datetime.datetime(2024, 1, 2, 9, 30) + datetime.timedelta(minutes=15 * i)).hour,
     (datetime.datetime(2024, 1, 2, 9, 30) + datetime.timedelta(minutes=15 * i)).minute)
    for i in range(26)
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestOffsetValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"minutes_before_close": True},
            {"block_entry_minutes_before_close": True},
            {"minutes_before_close": 1.5},
        ],
    )
    def test_non_int_minutes_are_refused(self, kwargs: dict[str, object]) -> None:
        """bool must be refused explicitly: ``isinstance(True, int)`` is True,
        so an unguarded check reads ``minutes_before_close=True`` as 1 minute."""
        with pytest.raises(TypeError, match="non-negative int"):
            FlattenSchedule(**kwargs)  # type: ignore[arg-type]

    def test_negative_minutes_are_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            FlattenSchedule(minutes_before_close=-1)

    def test_block_defaults_to_minutes_before_close(self) -> None:
        s = FlattenSchedule(minutes_before_close=60)
        assert s.block_entry_minutes_before_close == 60

    def test_explicit_zero_block_is_kept(self) -> None:
        s = FlattenSchedule(minutes_before_close=60, block_entry_minutes_before_close=0)
        assert s.block_entry_minutes_before_close == 0

    def test_default_and_explicit_restatement_are_one_schedule(self) -> None:
        """Canonicalization — the consumer's cache key is built from these fields."""
        a = FlattenSchedule(minutes_before_close=30)
        b = FlattenSchedule(
            minutes_before_close=30, block_entry_minutes_before_close=30
        )
        assert a == b
        assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# minutes_before_close
# ---------------------------------------------------------------------------


class TestMinutesBeforeClose:
    def test_moves_the_flatten_bar_off_the_session_last_bar(self) -> None:
        df = _frame(list(_TWO), _TIMES)
        # 16:00 - 45m = 15:15 -> last bar at or before 15:15 is 15:15
        assert _flatten_dates(df, FlattenSchedule(minutes_before_close=45)) == [
            datetime.datetime(2024, 1, 2, 15, 15),
            datetime.datetime(2024, 1, 3, 15, 15),
        ]

    def test_offset_below_the_bar_grid_is_a_noop(self) -> None:
        """The offset is wall-clock, not a bar count."""
        df = _frame(list(_TWO), _TIMES)
        assert _flatten_dates(df, FlattenSchedule(minutes_before_close=10)) == (
            _flatten_dates(df, FlattenSchedule(minutes_before_close=0))
        )

    def test_respects_an_early_close(self) -> None:
        """2024-07-03 is a 13:00 half-day: 30 minutes before is 12:30."""
        times = ((9, 30), (12, 0), (12, 30), (12, 45), (12, 55))
        df = _frame([datetime.date(2024, 7, 3)], times)
        assert _flatten_dates(df, FlattenSchedule(minutes_before_close=30)) == [
            datetime.datetime(2024, 7, 3, 12, 30)
        ]

    def test_offset_longer_than_the_session_warns_and_flattens_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = _frame(list(_TWO), _TIMES)
        with caplog.at_level(logging.WARNING, logger="mktlib.backtest._flatten"):
            marks = _flatten_dates(df, FlattenSchedule(minutes_before_close=600))
        assert marks == []
        assert "longer than every selected session" in caplog.text

    def test_forced_exit_lands_on_the_offset_bar_and_fills_at_its_open(self) -> None:
        df = _frame(list(_TWO), _TIMES, signal_at=_MORNING)
        res = run(
            df, _Cross(), calendar=get_calendar("XNYS"),
            flatten=FlattenSchedule(minutes_before_close=45),
        )
        assert res.trades.height >= 1, "fixture opened nothing; assertion is vacuous"
        assert res.trades["exit_date"][0] == datetime.datetime(2024, 1, 2, 15, 15)

        # A session-forced exit fills at the flatten bar's OWN open, not the
        # next bar's. Assert that through pnl, which is the only place the
        # realized fill price is observable from the public result.
        def _open_at(hh: int, mm: int) -> float:
            return df.filter(
                pl.col("date") == datetime.datetime(2024, 1, 2, hh, mm)
            )["open"][0]

        entry_fill = _open_at(15, 0)   # next open after the 10:00 signal
        exit_fill = _open_at(15, 15)   # the flatten bar's own open
        assert res.trades["pnl"][0] == pytest.approx(
            (exit_fill - entry_fill) / entry_fill
        )
        assert res.trades["bars_held"][0] == 1

    def test_returns_are_zero_on_bars_after_the_flatten_bar(self) -> None:
        df = _frame(list(_TWO), _TIMES, signal_at=_MORNING)
        res = run(
            df, _Cross(), calendar=get_calendar("XNYS"),
            flatten=FlattenSchedule(minutes_before_close=45),
        )
        assert res.trades.height >= 1, "fixture opened nothing; assertion is vacuous"
        held = res.returns.filter(
            pl.col("date") == datetime.datetime(2024, 1, 2, 15, 0)
        )["return"][0]
        assert held != 0.0, "fixture never earned a return; assertion is vacuous"
        after = res.returns.filter(
            pl.col("date").is_in([
                datetime.datetime(2024, 1, 2, 15, 30),
                datetime.datetime(2024, 1, 2, 15, 45),
            ])
        )
        assert after["return"].to_list() == [0.0, 0.0]

    def test_flatten_bar_pays_exactly_one_exit_fill_under_cost(self) -> None:
        """The price / index / cost chains in _extract_trades must agree."""
        df = _frame(list(_TWO), _TIMES, signal_at=_MORNING)
        schedule = FlattenSchedule(minutes_before_close=45)
        cal = get_calendar("XNYS")
        gross = run(df, _Cross(), calendar=cal, flatten=schedule)
        net = run(df, _Cross(), calendar=cal, flatten=schedule, cost=Cost(15.0))

        flat_bar = datetime.datetime(2024, 1, 2, 15, 15)
        g = gross.returns.filter(pl.col("date") == flat_bar)["return"][0]
        n = net.returns.filter(pl.col("date") == flat_bar)["return"][0]
        assert n - g == pytest.approx(-15e-4, rel=1e-9), "one exit side, 15 bps"

        # Bars after the flatten pay nothing at all.
        for ts in (datetime.datetime(2024, 1, 2, 15, 30),
                   datetime.datetime(2024, 1, 2, 15, 45)):
            assert net.returns.filter(pl.col("date") == ts)["return"][0] == 0.0


# ---------------------------------------------------------------------------
# block_entry_minutes_before_close
# ---------------------------------------------------------------------------


class TestBlockEntry:
    def test_block_window_marks_the_expected_bars(self) -> None:
        df = _frame(list(_TWO), _TIMES)
        blocked = _blocked_dates(
            df, FlattenSchedule(block_entry_minutes_before_close=45)
        )
        # 16:00 - 45m = 15:15 -> bars at or after 15:15
        assert blocked == [
            datetime.datetime(2024, 1, 2, 15, 15),
            datetime.datetime(2024, 1, 2, 15, 30),
            datetime.datetime(2024, 1, 2, 15, 45),
            datetime.datetime(2024, 1, 3, 15, 15),
            datetime.datetime(2024, 1, 3, 15, 30),
            datetime.datetime(2024, 1, 3, 15, 45),
        ]

    def test_block_zero_marks_nothing(self) -> None:
        df = _frame(list(_TWO), _TIMES)
        assert _blocked_dates(df, FlattenSchedule()) == []

    def test_block_zero_is_byte_identical_to_plain_eod(self) -> None:
        cal = get_calendar("XNYS")
        df = _frame(list(_TWO), _TIMES, signal_at=_MORNING)
        a = run(df, _Cross(), calendar=cal, flatten="eod")
        b = run(
            df, _Cross(), calendar=cal,
            flatten=FlattenSchedule(block_entry_minutes_before_close=0),
        )
        assert_frame_equal(a.returns, b.returns)
        assert_frame_equal(a.trades, b.trades)
        assert_frame_equal(a.signals, b.signals)

    def test_blocked_entry_is_dropped_not_deferred(self) -> None:
        """A signal inside the window opens nothing — not now, not next session."""
        df = _frame(list(_TWO), _TIMES, signal_at=datetime.datetime(2024, 1, 2, 15, 30))
        res = run(
            df, _Cross(), calendar=get_calendar("XNYS"),
            flatten=FlattenSchedule(block_entry_minutes_before_close=45),
        )
        assert res.trades.height == 0, res.trades

    def test_the_same_signal_opens_a_trade_without_the_block(self) -> None:
        """Guards the test above against passing for an unrelated reason."""
        df = _frame(list(_TWO), _TIMES, signal_at=datetime.datetime(2024, 1, 2, 15, 30))
        res = run(df, _Cross(), calendar=get_calendar("XNYS"), flatten="eod")
        assert res.trades.height == 1

    def test_default_block_closes_the_overnight_hole(self) -> None:
        """An early flatten with the default block leaves no overnight exposure.

        This is the whole reason ``block_entry_minutes_before_close`` defaults
        to ``minutes_before_close``: with the flatten bar off the session's
        last bar, in-session bars remain afterwards, and without a block window
        a fresh entry re-opens on one of them and carries overnight.

        Uses :class:`_Level`, whose entry re-arms every bar. With an edge entry
        this assertion holds trivially — nothing would have re-entered — and
        the block window is never exercised at all.
        """
        days = _sessions(datetime.date(2024, 1, 2), datetime.date(2024, 1, 4))
        df = _frame(days, _TIMES, signal_at=_MORNING)
        cal = get_calendar("XNYS")
        # cutoff 15:20 -> flatten bar 15:15; bars 15:30 and 15:45 remain, and a
        # level entry WILL re-open on them unless the window stops it.
        blocked = run(
            df, _Level(), calendar=cal,
            flatten=FlattenSchedule(minutes_before_close=40),
        )
        unblocked = run(
            df, _Level(), calendar=cal,
            flatten=FlattenSchedule(
                minutes_before_close=40, block_entry_minutes_before_close=0
            ),
        )

        def _day1(res: object) -> list[int]:
            return (
                res.signals.filter(  # type: ignore[attr-defined]
                    pl.col("date").dt.date() == datetime.date(2024, 1, 2)
                )["_position"].to_list()
            )

        held_blocked, held_unblocked = _day1(blocked), _day1(unblocked)
        assert 1 in held_blocked, "fixture never held a position on day 1"
        # The control proves the block is load-bearing: without it the position
        # re-opens after the flatten and is still on at the session boundary.
        assert held_unblocked[-1] == 1, (
            "control failed: nothing re-entered even with the window open, so "
            "the assertion below would hold vacuously"
        )
        assert held_blocked[-1] == 0, (
            "position survived to the session boundary despite an early flatten"
        )

    def test_block_applies_only_on_sessions_the_schedule_flattens(self) -> None:
        """Under `weekly`, a Wednesday 15:30 signal is not in any block window."""
        days = _sessions(datetime.date(2024, 1, 2), datetime.date(2024, 1, 5))
        df = _frame(days, _TIMES, signal_at=datetime.datetime(2024, 1, 3, 15, 30))
        res = run(
            df, _Cross(), calendar=get_calendar("XNYS"),
            flatten=FlattenSchedule(
                days="weekly", block_entry_minutes_before_close=45
            ),
        )
        assert res.trades.height == 1
        assert res.trades["exit_date"][0].date() == datetime.date(2024, 1, 5)


# ---------------------------------------------------------------------------
# Ordering: defer, THEN block
# ---------------------------------------------------------------------------


class TestDeferralThenBlockOrdering:
    """The flatten bar is not always inside the block window.

    On a grid that does not divide the two cutoffs evenly, the flatten bar can
    start strictly before ``close - block_minutes`` while the next bar starts
    after it. Blocking before deferring would let the deferral write an entry
    into the window it had just cleared.
    """

    _GRID = ((9, 30), (15, 40), (15, 45), (15, 50), (15, 55))
    # N = M = 7 against a 16:00 close -> cutoff 15:53.
    #   flatten bar  = last bar <= 15:53 = 15:50   (NOT itself blocked)
    #   block window = bars >= 15:53     = {15:55}
    _SCHEDULE = FlattenSchedule(
        minutes_before_close=7, block_entry_minutes_before_close=7
    )

    def test_fixture_puts_the_flatten_bar_outside_the_block_window(self) -> None:
        df = _frame(list(_TWO), self._GRID)
        assert _flatten_dates(df, self._SCHEDULE) == [
            datetime.datetime(2024, 1, 2, 15, 50),
            datetime.datetime(2024, 1, 3, 15, 50),
        ]
        assert _blocked_dates(df, self._SCHEDULE) == [
            datetime.datetime(2024, 1, 2, 15, 55),
            datetime.datetime(2024, 1, 3, 15, 55),
        ]

    def test_deferral_cannot_smuggle_an_entry_into_the_block_window(self) -> None:
        df = _frame(
            list(_TWO), self._GRID, signal_at=datetime.datetime(2024, 1, 2, 15, 50)
        )
        res = run(df, _Cross(), calendar=get_calendar("XNYS"), flatten=self._SCHEDULE)
        sig = res.signals

        at_1555 = sig.filter(pl.col("date") == datetime.datetime(2024, 1, 2, 15, 55))
        assert at_1555["_entry"][0] is False, (
            "the deferral wrote an entry into the block window"
        )
        assert at_1555["_position"][0] == 0
        next_open = sig.filter(pl.col("date") == datetime.datetime(2024, 1, 3, 9, 30))
        assert next_open["_position"][0] == 0, "overnight exposure leaked"

    def test_a_blocked_flatten_bar_drops_rather_than_defers(self) -> None:
        """The other half of the ordering: the flatten bar IS in the window.

        On a 15-minute grid the last bar starts exactly at ``close - 15``, so
        ``minutes_before_close=15`` makes that bar both the flatten bar and the
        first blocked bar. Deferring from it would carry the signal to the next
        *session's* open — which no window covers — and fill hours late. The
        documented behaviour is to drop.
        """
        schedule = FlattenSchedule(minutes_before_close=15)
        df = _frame(list(_TWO), _T15, signal_at=datetime.datetime(2024, 1, 2, 15, 45))
        assert datetime.datetime(2024, 1, 2, 15, 45) in _flatten_dates(df, schedule)
        assert datetime.datetime(2024, 1, 2, 15, 45) in _blocked_dates(df, schedule)

        res = run(df, _Cross(), calendar=get_calendar("XNYS"), flatten=schedule)
        assert res.trades.height == 0, (
            f"signal was deferred across the session boundary: "
            f"{res.trades.select('entry_date').to_dicts()}"
        )

    def test_the_same_signal_opens_a_trade_without_the_block(self) -> None:
        """Control for the test above — the signal itself is real."""
        df = _frame(list(_TWO), _T15, signal_at=datetime.datetime(2024, 1, 2, 15, 45))
        res = run(
            df, _Cross(), calendar=get_calendar("XNYS"),
            flatten=FlattenSchedule(
                minutes_before_close=15, block_entry_minutes_before_close=0
            ),
        )
        assert res.trades.height == 1
        assert res.trades["entry_date"][0] == datetime.datetime(2024, 1, 3, 9, 30)


class TestWholeSessionBlockDiagnostic:
    """A window covering an entire session must not fail silently."""

    @pytest.mark.parametrize(
        ("schedule", "why"),
        [
            (FlattenSchedule(block_entry_minutes_before_close=390),
             "a whole NYSE session"),
            (FlattenSchedule(minutes_before_close=1800),
             "seconds passed where minutes were meant"),
        ],
    )
    def test_a_window_covering_every_bar_warns(
        self, schedule: FlattenSchedule, why: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = _frame(list(_TWO), _TIMES, signal_at=_MORNING)
        with caplog.at_level(logging.WARNING, logger="mktlib.backtest._flatten"):
            res = run(df, _Cross(), calendar=get_calendar("XNYS"), flatten=schedule)
        assert res.trades.height == 0, "fixture assumption: nothing can open"
        assert "covers every bar" in caplog.text, (
            f"a window that blocks {why} produced an empty backtest silently"
        )
        assert "units" in caplog.text, "the warning should name the likely cause"

    def test_a_normal_window_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Guards the diagnostic against firing on every ordinary run."""
        df = _frame(list(_TWO), _TIMES, signal_at=_MORNING)
        with caplog.at_level(logging.WARNING, logger="mktlib.backtest._flatten"):
            run(df, _Cross(), calendar=get_calendar("XNYS"),
                flatten=FlattenSchedule(block_entry_minutes_before_close=45))
        assert "covers every bar" not in caplog.text


class TestScheduleCacheIdentity:
    """Two calendars sharing a name must not share a cached schedule.

    The cache stores ``market_close``, and ``market_close`` now sets the
    ``minutes_before_close`` cutoff and the entry-block window as well as the
    session-last bar — so a collision moves all three.
    """

    @staticmethod
    def _adhoc(close_time: datetime.time) -> ExchangeCalendar:
        return ExchangeCalendar(
            "ADHOC-SAME-NAME",
            timezone="America/New_York",
            open_time=datetime.time(9, 30),
            close_time=close_time,
            holidays=[],
        )

    def test_same_name_different_close_do_not_collide(self) -> None:
        late = self._adhoc(datetime.time(16, 0))
        early = self._adhoc(datetime.time(13, 30))
        assert late.name == early.name, "fixture: the names must collide"

        df = _frame([datetime.date(2024, 1, 2)], _TIMES)
        dates = df["date"]
        sched_late = build_flatten_masks(dates, late, FlattenSchedule())
        sched_early = build_flatten_masks(dates, early, FlattenSchedule())

        late_bar = [d for d, m in zip(dates.to_list(), sched_late[0].to_list(),
                                      strict=True) if m]
        early_bar = [d for d, m in zip(dates.to_list(), sched_early[0].to_list(),
                                       strict=True) if m]
        # 16:00 close -> the grid's last bar, 15:45.
        assert late_bar == [datetime.datetime(2024, 1, 2, 15, 45)]
        # 13:30 close -> every 15:xx bar is outside the session, so the last
        # bar of the session is 10:00. Sharing the cache entry would have
        # produced 15:45 here.
        assert early_bar == [datetime.datetime(2024, 1, 2, 10, 0)], (
            "the 13:30-closing calendar reused the 16:00 calendar's schedule"
        )


class TestSessionsWithNoBarBeforeCutoff:
    """The third warning path — reachable, and previously untested."""

    def test_a_session_with_no_bar_before_the_cutoff_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Two sessions; the second carries only a late bar, after the cutoff.
        early = [(9, 30), (10, 0), (15, 0)]
        d1 = _frame([datetime.date(2024, 1, 2)], tuple(early))
        d2 = _frame([datetime.date(2024, 1, 3)], ((15, 45),))
        df = pl.concat([d1, d2]).sort("date")
        with caplog.at_level(logging.WARNING, logger="mktlib.backtest._flatten"):
            marks = _flatten_dates(df, FlattenSchedule(minutes_before_close=45))
        # 16:00 - 45m = 15:15; only day 1 has a bar at or before it.
        assert marks == [datetime.datetime(2024, 1, 2, 15, 0)]
        assert "no bar at or before the cutoff" in caplog.text
