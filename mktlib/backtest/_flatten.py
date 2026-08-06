"""Everything in :mod:`mktlib.backtest` that knows what a *day* is.

The engine resolves entries, exits, fills and returns without any notion of a
trading session. The single exception is the forced session close, and this
module is where that exception is confined: it turns a calendar plus a schedule
into a boolean mask, and the engine consumes nothing but the mask.

That split is what keeps the compiled resolver out of scope. ``mktlib-scan``
takes ``session_last: Option<&[bool]>`` — opaque booleans with no concept of a
day — so changing *when* a position is force-closed never touches the
bit-identical Python/native contract.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from mktlib.scheduling import ExchangeCalendar

#: Boolean column marking every bar on which an open position is force-closed.
#: Internal — materialized by :func:`build_flatten_mask` and dropped before the
#: engine returns, so it never appears in a caller-visible frame.
FLATTEN_BAR_COLUMN = "_flatten_bar"


def _to_date(dt: datetime.datetime | datetime.date) -> datetime.date:
    """Convert datetime to date if needed, for calendar API compatibility."""
    if isinstance(dt, datetime.datetime):
        return dt.date()
    return dt


def _tz(series: pl.Series) -> str | None:
    """Extract timezone from a Datetime series, or None."""
    return series.dtype.time_zone  # type: ignore[union-attr]


def _align_tz(target: pl.Series, reference: pl.Series) -> pl.Series:
    """Align *target* timezone to match *reference*."""
    ref_tz = _tz(reference)
    tgt_tz = _tz(target)
    if ref_tz is None and tgt_tz is not None:
        return target.dt.replace_time_zone(None)
    if ref_tz is not None and tgt_tz is None:
        return target.dt.replace_time_zone(ref_tz)
    return target


# ---------------------------------------------------------------------------
# Schedule cache — avoids recomputing calendar.schedule() across mask calls
# ---------------------------------------------------------------------------

_schedule_cache: dict[tuple[str, datetime.date, datetime.date], pl.DataFrame] = {}


def _get_schedule(calendar: ExchangeCalendar, dates: pl.Series) -> pl.DataFrame:
    """Return cached schedule DataFrame for the calendar covering *dates*."""
    start = _to_date(dates.min())  # type: ignore[arg-type]
    end = _to_date(dates.max())  # type: ignore[arg-type]
    key = (calendar.name, start, end)
    if key not in _schedule_cache:
        _schedule_cache[key] = calendar.schedule(start, end)
    return _schedule_cache[key]


def build_flatten_mask(
    dates: pl.Series,
    calendar: ExchangeCalendar,
) -> pl.Series:
    """Boolean series: True on the last bar of each trading session.

    Finds the actual last bar per session from the data rather than assuming
    a fixed offset, so this works for any candle size (1min, 5min, 15min, …).
    """
    sched = _get_schedule(calendar, dates)
    open_times = _align_tz(sched["market_open"], dates)
    close_times = _align_tz(sched["market_close"], dates)

    sessions = pl.DataFrame({
        "market_open": open_times,
        "market_close": close_times,
    })

    bar_df = dates.to_frame("date").with_row_index("_idx")
    joined = bar_df.join_asof(sessions, left_on="date", right_on="market_open")

    joined = joined.filter(pl.col("date") < pl.col("market_close"))
    last_per_session = joined.group_by("market_open").agg(
        pl.col("date").max().alias("last_bar"),
    )

    last_bars = last_per_session["last_bar"].to_list()
    return dates.is_in(last_bars)
