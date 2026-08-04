"""``EntryRef`` under ``flatten_eod``: the anchor must follow the deferred entry.

``flatten_eod`` defers an entry signal that lands on a session-last bar to the
first bar of the next session — the position opens there, not where the signal
fired. The ``EntryRef`` snapshot used to be built *before* that rewrite, so it
latched on the bar the deferral had moved away from, and the exit was measured
against a bar in the previous session.

No golden scenario combines ``flatten_eod`` with ``EntryRef``, so nothing caught
it. This is that missing scenario.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import polars as pl
import pytest

from mktlib.backtest import Condition, Crossover, EntryRef, Pct, ValueGT, run
from mktlib.scheduling import get_calendar

_SESSIONS = (
    datetime.date(2024, 1, 2),
    datetime.date(2024, 1, 3),
    datetime.date(2024, 1, 4),
)


@dataclass(frozen=True, slots=True)
class _CrossEntryFarTarget:
    """Crossover in, and a target so distant the position never closes.

    Keeping the position open for the rest of the frame means the anchor stays
    observable, which is the point of the fixture.
    """

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Condition:
        return ValueGT("close", Pct(EntryRef("close"), 500.0))


@pytest.fixture
def deferred_entry_frame() -> pl.DataFrame:
    """90-minute bars whose crossover lands exactly on a session-last bar.

    ``close`` rises by 1.0 per bar so the anchor identifies its own bar
    unambiguously — 104.0 is the session-last bar, 105.0 the bar the deferred
    entry actually opens on.
    """
    stamps: list[datetime.datetime] = []
    for day in _SESSIONS:
        ts = datetime.datetime(day.year, day.month, day.day, 9, 30)
        while ts.hour < 16:
            stamps.append(ts)
            ts += datetime.timedelta(minutes=90)

    n = len(stamps)
    close = [100.0 + i for i in range(n)]
    fast = [1.0] * n
    slow = [2.0] * n
    session_last = max(
        i for i, t in enumerate(stamps) if t.date() == _SESSIONS[0]
    )
    fast[session_last:] = [3.0] * (n - session_last)

    return pl.DataFrame(
        {
            "date": stamps,
            "open": close,
            "high": [c * 1.001 for c in close],
            "low": [c * 0.999 for c in close],
            "close": close,
            "fast": fast,
            "slow": slow,
        }
    )


def test_anchor_follows_the_deferred_entry(
    deferred_entry_frame: pl.DataFrame,
) -> None:
    result = run(
        deferred_entry_frame,
        _CrossEntryFarTarget(),
        calendar=get_calendar("XNYS"),
        flatten_eod=True,
    )
    signals = result.signals.with_row_index()

    opening = signals.filter(
        (pl.col("_position") == 1) & (pl.col("_position").shift(1) == 0)
    )
    assert opening.height == 1, "fixture must open exactly one position"
    row = opening.row(0, named=True)

    # The deferral moved the entry into the second session.
    assert row["date"].date() == _SESSIONS[1]
    # ...and the anchor must be that bar's close, not the session-last bar's.
    assert row["_entry_close"] == pytest.approx(row["close"])


def test_anchor_is_not_the_session_last_bar(
    deferred_entry_frame: pl.DataFrame,
) -> None:
    """State the regression directly, in the values it actually took.

    Before the fix the anchor read 104.0 — the session-last bar the deferral
    moved away from — while the position opened on the 105.0 bar.
    """
    result = run(
        deferred_entry_frame,
        _CrossEntryFarTarget(),
        calendar=get_calendar("XNYS"),
        flatten_eod=True,
    )
    held = result.signals.filter(pl.col("_position") == 1)
    assert held["_entry_close"].n_unique() == 1
    assert held["_entry_close"][0] == pytest.approx(105.0)
