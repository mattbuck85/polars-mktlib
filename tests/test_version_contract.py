"""Results that must not move with the installed polars version.

mktlib supports a range of polars versions (see ``pyproject.toml``).  Each
test here pins an output whose value is produced by a polars primitive
that is free to change between releases, so the suite goes red on the
version that changes it rather than on a downstream consumer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from mktlib.backtest import strategy_artifact
from mktlib.data import Innovations, Process, monte_carlo
from mktlib.scheduling import get_calendar

# ---------------------------------------------------------------------------
# trading_index: a session that holds no timestamp contributes no row
# ---------------------------------------------------------------------------
#
# ``trading_index`` builds one list of timestamps per session and explodes
# it.  Exploding an empty list yields one null row on polars 1.x and no row
# on polars 2.x, so a session with no timestamp at the requested period
# must be dropped explicitly for the two to agree.


@pytest.mark.parametrize(
    ("exchange", "period", "closed"),
    [
        # Regular calendar: one 6.5h session, the open interval holds no
        # multiple of the period.
        ("XNYS", "1d", "none"),
        ("XNYS", "1d", "right"),
        ("XNYS", "7h", "none"),
        # Break calendar: both the morning and the afternoon range are
        # empty, so their concatenation is an empty list.
        ("XTKS", "1d", "none"),
        ("XTKS", "1d", "right"),
        ("XTKS", "3h", "none"),
        ("XHKG", "3h", "none"),
    ],
)
def test_trading_index_empty_sessions_yield_no_rows(
    exchange: str, period: str, closed: Literal["none", "right"]
) -> None:
    cal = get_calendar(exchange)
    out = cal.trading_index(
        "2024-03-04", "2024-03-06", period=period, closed=closed
    )
    assert out.null_count() == 0
    assert out.len() == 0
    assert out.name == "datetime"
    assert out.dtype == pl.Datetime("us", cal.timezone)


def test_trading_index_drops_only_the_empty_session() -> None:
    """A short session between two full ones leaves no null behind.

    2024-07-03 is an NYSE early close (09:30-13:00).  With a 4h period and
    an open interval, the full sessions hold 13:30 and the short one holds
    nothing.
    """
    cal = get_calendar("XNYS")
    out = cal.trading_index(
        "2024-07-02", "2024-07-05", period="4h", closed="none"
    )
    tz = ZoneInfo("America/New_York")
    assert out.to_list() == [
        datetime(2024, 7, 2, 13, 30, tzinfo=tz),
        datetime(2024, 7, 5, 13, 30, tzinfo=tz),
    ]


def test_trading_index_break_calendar_drops_only_the_empty_session() -> None:
    """A half-day session between two full ones leaves no null behind.

    Hong Kong: 09:30-12:00 and 13:00-16:00.  At 2h30m, open interval, the
    morning of a full session holds nothing and its afternoon holds 15:30,
    so a full session contributes one timestamp from a list whose first
    half is empty.  2024-02-09 closes at 12:00: both halves are empty and
    the session contributes nothing.
    """
    cal = get_calendar("XHKG")
    sessions = cal.schedule("2024-02-08", "2024-02-14")["market_close"]
    assert [(t.day, t.hour) for t in sessions.to_list()] == [
        (8, 16), (9, 12), (14, 16),
    ]
    out = cal.trading_index(
        "2024-02-08", "2024-02-14", period="2h30m", closed="none"
    )
    tz = ZoneInfo("Asia/Hong_Kong")
    assert out.to_list() == [
        datetime(2024, 2, d, 15, 30, tzinfo=tz) for d in (8, 14)
    ]


# ---------------------------------------------------------------------------
# Tripwires: values derived from polars internals
# ---------------------------------------------------------------------------
#
# Measured identical on polars 1.44.2 and 2.0.0.  Neither value is
# guaranteed by polars; a failure here means a polars release changed the
# underlying primitive, and that stored digests or seeded results change
# with it.


class _CustomCompare:
    def entry(self) -> pl.Expr:
        return pl.col("a") > 0

    def exit(self) -> pl.Expr:
        return pl.col("a") < 0


class _CustomRolling:
    def entry(self) -> pl.Expr:
        return pl.col("close").rolling_mean(3) > pl.col("close").shift(1)

    def exit(self) -> pl.Expr:
        return pl.col("close").is_null()


class _CustomLiteral:
    def entry(self) -> pl.Expr:
        return pl.lit(True)

    def exit(self) -> pl.Expr:
        return pl.lit(False)


@pytest.mark.parametrize(
    ("strategy", "digest"),
    [
        (_CustomCompare(), "e9872528c5495aee"),
        (_CustomRolling(), "23ebab088ad1f10e"),
        (_CustomLiteral(), "d4d843858d17e575"),
    ],
    ids=["compare", "rolling", "literal"],
)
def test_custom_expression_digest_is_stable(
    strategy: Any, digest: str
) -> None:
    """``strategy_artifact`` hashes ``Expr.meta.serialize()`` for a raw
    expression, so the digest follows polars' expression serialization."""
    first = strategy_artifact(strategy)
    assert first == strategy_artifact(strategy)
    assert first == digest


def _bootstrap_signs(seed: int) -> str:
    """Signs of the per-step log increments of a two-valued bootstrap.

    With residuals ``[-1, 1]`` every increment is ``-0.00125 +/- 0.05``, so
    its sign is exactly the residual that ``Series.sample`` drew.
    """
    out = monte_carlo(
        Process.GBM,
        n_simulations=4,
        n=17,
        seed=seed,
        volatility=0.05,
        drift=0.0,
        dt=1.0,
        innovations=Innovations.BOOTSTRAP,
        residuals=pl.Series("r", [-1.0, 1.0]),
    )
    inc = out.with_columns(
        pl.col("price").log().diff().over("simulation").alias("inc")
    )["inc"].drop_nulls()
    return "".join("+" if v > 0 else "-" for v in inc.to_list())


def test_bootstrap_stream_is_stable() -> None:
    """The bootstrap innovations come from ``Series.sample(seed=)``, so the
    seeded stream follows polars' sampling implementation."""
    first = _bootstrap_signs(42)
    assert first == _bootstrap_signs(42)
    assert first != _bootstrap_signs(43)
    assert first == (
        "+---+-+++-+-+---"
        "----+-+------+-+"
        "-+-+-+---+-+----"
        "++-+++---+++++--"
    )
