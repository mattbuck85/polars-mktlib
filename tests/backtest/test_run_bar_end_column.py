"""``run(..., bar_end_column=...)`` reaches the calendar's end-column filter.

``run()`` applies ``filter_market_hours`` itself whenever a calendar is bound,
on the single-symbol, dual-strategy and multi-instrument paths alike. Without a
keyword of its own, a ``run()`` caller with variable-duration bars has no way
to opt in to the end-column bound: the closing bars of every session are
dropped before the engine ever sees them, and the calendar-level parameter is
unreachable from this entry point.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from mktlib.backtest import run
from mktlib.scheduling import get_calendar

NY = ZoneInfo("America/New_York")

# NYSE 2024-01-02 closes at 16:00. The last three bars all open strictly
# inside the final minute (15:59:00-16:00:00) and all close at or before the
# close, so the default label bound drops them and the end bound keeps them.
_SPANS = [
    (datetime(2024, 1, 2, 9, 30, 0, tzinfo=NY), datetime(2024, 1, 2, 9, 31, 0, tzinfo=NY)),
    (datetime(2024, 1, 2, 12, 0, 0, tzinfo=NY), datetime(2024, 1, 2, 12, 0, 30, tzinfo=NY)),
    (datetime(2024, 1, 2, 15, 59, 10, tzinfo=NY), datetime(2024, 1, 2, 15, 59, 30, tzinfo=NY)),
    (datetime(2024, 1, 2, 15, 59, 30, tzinfo=NY), datetime(2024, 1, 2, 15, 59, 50, tzinfo=NY)),
    (datetime(2024, 1, 2, 15, 59, 50, tzinfo=NY), datetime(2024, 1, 2, 16, 0, 0, tzinfo=NY)),
]


class _AlwaysLong:
    def entry(self) -> pl.Expr:
        return pl.col("sig") > 0.5

    def exit(self) -> pl.Expr:
        return pl.col("sig") < 0.0


class _NeverShort:
    def entry(self) -> pl.Expr:
        return pl.col("sig") < 0.0

    def exit(self) -> pl.Expr:
        return pl.col("sig") > 0.5


def _frame(instrument: str | None = None) -> pl.DataFrame:
    close = [100.0 + i for i in range(len(_SPANS))]
    data: dict[str, list[object]] = {
        "date": [start for start, _ in _SPANS],
        "end": [end for _, end in _SPANS],
        "open": list(close),
        "high": [c * 1.001 for c in close],
        "low": [c * 0.999 for c in close],
        "close": list(close),
        "sig": [1.0] * len(_SPANS),
    }
    if instrument is not None:
        data["symbol"] = [instrument] * len(_SPANS)
    return pl.DataFrame(data)


@pytest.fixture
def nyse():
    return get_calendar("XNYS")


def test_run_drops_variable_width_closing_bars_without_the_keyword(nyse) -> None:
    """The defect, at the entry point: 5 bars in, 2 bars reach the engine."""
    result = run(_frame(), _AlwaysLong(), calendar=nyse)
    assert result.signals.height == 2


def test_run_keeps_variable_width_closing_bars_with_bar_end_column(nyse) -> None:
    """All five bars reach the engine when the end column is threaded through."""
    result = run(_frame(), _AlwaysLong(), calendar=nyse, bar_end_column="end")
    assert result.signals.height == 5
    assert result.signals["date"].to_list() == [start for start, _ in _SPANS]


def test_run_dual_strategy_path_honours_bar_end_column(nyse) -> None:
    """The dual-strategy path filters separately and must thread it too."""
    result = run(
        _frame(),
        _AlwaysLong(),
        short_strategy=_NeverShort(),
        calendar=nyse,
        bar_end_column="end",
    )
    assert result.signals.height == 5


def test_run_multi_instrument_path_honours_bar_end_column(nyse) -> None:
    """The multi-instrument path filters once on the full frame."""
    df = pl.concat([_frame("AAA"), _frame("BBB")])
    result = run(
        df,
        _AlwaysLong(),
        calendar=nyse,
        instrument_col="symbol",
        bar_end_column="end",
    )
    for instrument in ("AAA", "BBB"):
        assert result[instrument].signals.height == 5, instrument


def test_run_default_is_unchanged_when_bar_end_column_is_omitted(nyse) -> None:
    """An ``end`` column present but not named must change nothing.

    This is the accepts-what-it-should companion: the new keyword must not
    start filtering on a column merely because the frame happens to carry it.
    """
    without = run(_frame(), _AlwaysLong(), calendar=nyse)
    explicit_none = run(
        _frame(), _AlwaysLong(), calendar=nyse, bar_end_column=None
    )
    assert without.signals.height == 2
    assert without.signals.equals(explicit_none.signals)
