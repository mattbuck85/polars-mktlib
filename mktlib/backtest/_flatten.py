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
import enum
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from mktlib.scheduling import ExchangeCalendar

logger = logging.getLogger(__name__)

#: Boolean column marking every bar on which an open position is force-closed.
#: Internal — materialized by :func:`build_flatten_mask` and dropped before the
#: engine returns, so it never appears in a caller-visible frame.
FLATTEN_BAR_COLUMN = "_flatten_bar"


class Weekday(enum.IntEnum):
    """ISO weekday: MON=1 … SUN=7.

    Matches :meth:`polars.Expr.dt.weekday` and :meth:`datetime.date.isoweekday`
    — **not** :meth:`datetime.date.weekday`, which is 0-based. Getting that
    wrong shifts every selection by one day, silently.
    """

    MON = 1
    TUE = 2
    WED = 3
    THU = 4
    FRI = 5
    SAT = 6
    SUN = 7


#: Which sessions get a forced close.
#:
#: ``"daily"``
#:     Every session.
#: ``"weekly"``
#:     The last session of each ISO week *that has bars*. Holiday-aware for
#:     free: when Friday is a holiday, Thursday is that week's last session.
#: An explicit set of :class:`Weekday`
#:     Deliberately literal. ``{Weekday.FRI}`` means Friday and nothing else,
#:     so a week whose Friday is a holiday does not flatten at all. That is the
#:     whole reason both forms exist — pick ``"weekly"`` when you mean "end of
#:     week", the explicit set when you mean a particular weekday.
FlattenDays = Literal["daily", "weekly"] | frozenset[Weekday]

#: What the constructor accepts. Any iterable of weekday-ish values is taken and
#: normalized to a ``frozenset[Weekday]``, so ``{Weekday.FRI}``, ``{5}`` and
#: ``[Weekday.FRI]`` all produce the same — hashable — schedule.
FlattenDaysArg = Literal["daily", "weekly"] | Iterable[Weekday | int]

_VALID_DAY_STRINGS = ("daily", "weekly")


@dataclass(frozen=True, slots=True)
class FlattenSchedule:
    """When open positions are force-closed.

    ``FlattenSchedule()`` is end-of-day flattening — identical to the legacy
    ``flatten_eod=True``.

    After construction ``days`` is always ``"daily"``, ``"weekly"``, or a
    ``frozenset[Weekday]`` — see :data:`FlattenDays`.
    """

    days: FlattenDaysArg = "daily"

    def __post_init__(self) -> None:
        days = self.days
        if isinstance(days, str):
            if days not in _VALID_DAY_STRINGS:
                msg = (
                    f"days={days!r} is not valid; expected one of "
                    f"{_VALID_DAY_STRINGS} or a set of Weekday"
                )
                raise ValueError(msg)
            return

        # Normalize any iterable of weekday-ish values to a frozenset of
        # Weekday. A bare `set` would make this frozen dataclass unhashable,
        # which silently breaks every cache key built from it downstream.
        try:
            normalized = frozenset(Weekday(int(d)) for d in days)
        except (TypeError, ValueError) as exc:
            msg = (
                f"days={days!r} is not valid; expected one of "
                f"{_VALID_DAY_STRINGS} or an iterable of Weekday (ISO 1-7)"
            )
            raise ValueError(msg) from exc
        if not normalized:
            msg = (
                "days=frozenset() selects no session, so nothing would ever be "
                "flattened. Pass flatten=False if that is what you meant."
            )
            raise ValueError(msg)
        object.__setattr__(self, "days", normalized)


#: What ``run(flatten=...)`` accepts.
Flatten = bool | Literal["eod", "eow"] | FlattenSchedule | None

_FLATTEN_ALIASES: dict[str, FlattenSchedule] = {
    "eod": FlattenSchedule(days="daily"),
    "eow": FlattenSchedule(days="weekly"),
}

_BOTH_SET_MSG = (
    "flatten=... and flatten_eod=True cannot both be set. flatten_eod is the "
    "legacy spelling of flatten='eod'; pass only one."
)


def resolve_flatten(
    flatten: Flatten,
    *,
    flatten_eod: bool,
) -> FlattenSchedule | None:
    """Collapse the two spellings into one schedule, or ``None`` for no flatten."""
    if flatten is not None and flatten_eod:
        raise ValueError(_BOTH_SET_MSG)
    if flatten is None:
        return FlattenSchedule() if flatten_eod else None
    if flatten is True:
        return FlattenSchedule()
    if flatten is False:
        return None
    if isinstance(flatten, FlattenSchedule):
        return flatten
    if isinstance(flatten, str):
        try:
            return _FLATTEN_ALIASES[flatten]
        except KeyError:
            msg = (
                f"flatten={flatten!r} is not a known alias; expected one of "
                f"{sorted(_FLATTEN_ALIASES)}, a bool, or a FlattenSchedule"
            )
            raise ValueError(msg) from None
    msg = (
        f"flatten must be a bool, {sorted(_FLATTEN_ALIASES)}, or a "
        f"FlattenSchedule; got {type(flatten).__name__}"
    )
    raise TypeError(msg)


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


#: Monday of the ISO week containing ``_session_date``, as an epoch-day integer.
#:
#: A Date column in polars is Int32 days-since-epoch, so subtracting
#: ``weekday - 1`` lands exactly on that week's Monday with plain integer
#: arithmetic. Deliberately not ``year * 100 + week`` (wrong every year at the
#: Dec/Jan rollover) and not ``dt.truncate("1w")`` (epoch-anchored on a
#: Thursday, so its "weeks" run Thu-Wed).
_WEEK_KEY = pl.col("_session_date").cast(pl.Int32) - (
    pl.col("_session_date").dt.weekday() - 1
)


def _flatten_sessions(joined: pl.DataFrame, schedule: FlattenSchedule) -> pl.DataFrame:
    """Restrict *joined* to the bars of sessions this schedule flattens.

    Selection runs over sessions that actually carry bars, not over the raw
    calendar. That is what makes ``"weekly"`` holiday-aware without a holiday
    table of its own: if Friday has no bars, Thursday is the week's last
    session and takes the flatten.
    """
    days = schedule.days
    if days == "daily":
        return joined
    if days == "weekly":
        last_per_week = (
            joined.group_by(_WEEK_KEY.alias("_wk"))
            .agg(pl.col("_session_date").max().alias("_session_date"))
            .get_column("_session_date")
            .to_list()
        )
        return joined.filter(pl.col("_session_date").is_in(last_per_week))
    return joined.filter(
        pl.col("_session_date").dt.weekday().is_in([int(d) for d in days])
    )


def build_flatten_mask(
    dates: pl.Series,
    calendar: ExchangeCalendar,
    schedule: FlattenSchedule,
) -> pl.Series:
    """Boolean series: True on every bar that force-closes an open position.

    Finds the actual last bar per session from the data rather than assuming
    a fixed offset, so this works for any candle size (1min, 5min, 15min, …).
    """
    sched = _get_schedule(calendar, dates)
    open_times = _align_tz(sched["market_open"], dates)
    close_times = _align_tz(sched["market_close"], dates)

    sessions = pl.DataFrame({
        # The calendar's own session label — the *trading day*. Not derived
        # from ``market_open``, which lands on the previous calendar date on
        # any exchange with a negative open offset (CME Globex, FX).
        "_session_date": sched["date"],
        "market_open": open_times,
        "market_close": close_times,
    })

    bar_df = dates.to_frame("date").with_row_index("_idx")
    joined = bar_df.join_asof(sessions, left_on="date", right_on="market_open")

    joined = joined.filter(pl.col("date") < pl.col("market_close"))
    joined = _flatten_sessions(joined, schedule)

    if joined.height == 0:
        logger.warning(
            "flatten schedule %r selects no session in the data range "
            "%s … %s — nothing will be force-closed",
            schedule,
            dates.min(),
            dates.max(),
        )
        return pl.Series([False] * dates.len(), dtype=pl.Boolean)

    last_per_session = joined.group_by("market_open").agg(
        pl.col("date").max().alias("last_bar"),
    )

    last_bars = last_per_session["last_bar"].to_list()
    return dates.is_in(last_bars)
