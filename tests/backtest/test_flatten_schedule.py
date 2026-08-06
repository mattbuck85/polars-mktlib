"""Flatten scheduling: the mask, the schedule types, and the engine wiring.

The invariant tests here are load-bearing for code that was *deleted* rather
than kept. See ``test_position_is_always_zero_on_the_flatten_bar``.
"""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from mktlib.backtest._conditions import Crossover, Crossunder
from mktlib.backtest._engine import run
from mktlib.backtest._flatten import build_flatten_mask
from mktlib.scheduling import get_calendar

# 15-minute bars, 09:30 … 15:45 inclusive. XNYS closes at 16:00, so the last
# bar of each session starts 15 minutes before the close.
_BARS_PER_SESSION = 26


class _Cross:
    """Enter on the crossover, never exit on its own — the flatten owns the exit."""

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Crossunder:
        return Crossunder("never_a", "never_b")


def _dates(days: tuple[datetime.date, ...]) -> list[datetime.datetime]:
    out: list[datetime.datetime] = []
    for day in days:
        ts = datetime.datetime(day.year, day.month, day.day, 9, 30)
        end = datetime.datetime(day.year, day.month, day.day, 15, 45)
        while ts <= end:
            out.append(ts)
            ts += datetime.timedelta(minutes=15)
    return out


def _frame(
    days: tuple[datetime.date, ...],
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


_TWO_SESSIONS = (datetime.date(2024, 1, 2), datetime.date(2024, 1, 3))


class TestFlattenBarInvariant:
    """What the deleted ``& ~_is_entry_bar`` guard rested on."""

    def test_position_is_always_zero_on_the_flatten_bar(self) -> None:
        """``_position`` is structurally 0 on every flatten bar.

        This is why the post-flatten return zeroing needs no ``_is_entry_bar``
        guard: at flatten+1 ``_pos_d1`` is 0, so ``_is_entry_bar`` — which
        requires ``_pos_d1 == 1`` — cannot be true there.

        Revert-check (performed manually, 2026-08-06): changing the
        ``_position`` expression's ``& ~pl.col(FLATTEN_BAR_COLUMN)`` to
        ``& pl.lit(True)`` turns this test red at the first flatten bar. The
        assertion discriminates.
        """
        cal = get_calendar("XNYS")
        df = _frame(_TWO_SESSIONS, entry_signal_bar=2)
        result = run(df, _Cross(), calendar=cal, flatten_eod=True)

        mask = build_flatten_mask(df["date"], cal).to_list()
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
        held = result.signals["_position"].to_list()
        assert sum(held) > 0, "fixture opened no position; the invariant is vacuous"


class TestFlattenMask:
    def test_mask_marks_the_last_bar_of_each_session(self) -> None:
        cal = get_calendar("XNYS")
        df = _frame(_TWO_SESSIONS)
        mask = build_flatten_mask(df["date"], cal).to_list()
        assert sum(mask) == 2
        assert mask[_BARS_PER_SESSION - 1] is True
        assert mask[2 * _BARS_PER_SESSION - 1] is True

    def test_mask_is_a_boolean_series(self) -> None:
        cal = get_calendar("XNYS")
        df = _frame(_TWO_SESSIONS)
        assert build_flatten_mask(df["date"], cal).dtype == pl.Boolean


class TestNoInternalColumnLeak:
    @pytest.mark.parametrize("flatten_eod", [True, False])
    def test_flatten_bar_column_never_reaches_the_caller(
        self, *, flatten_eod: bool
    ) -> None:
        cal = get_calendar("XNYS")
        df = _frame(_TWO_SESSIONS)
        result = run(df, _Cross(), calendar=cal, flatten_eod=flatten_eod)
        assert "_flatten_bar" not in result.signals.columns
