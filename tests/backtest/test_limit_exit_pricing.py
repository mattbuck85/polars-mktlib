"""Limit-exit fill pricing: the two places ``trades.pnl`` and ``returns`` disagree.

Both defects were found by code review of the flatten-schedule work and both
predate it. They are here rather than in ``test_limit_exit.py`` because they are
about the *price a fill is booked at*, not about whether the limit fires.

The shared shape: ``_limit_price`` is materialized on **every** bar, not only on
bars where the limit actually fills. Anything that reads it without first
checking that the limit fired reads a level the market never traded.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import polars as pl
import pytest

from mktlib.backtest import Col, Condition, Crossover, Limit, ValueGTE, run
from mktlib.scheduling import get_calendar


@dataclass(frozen=True, slots=True)
class _CrossEntryTakeProfit:
    """Enter on a crossover; exit only when ``high`` reaches the ``tp`` column."""

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Condition:
        return Limit(ValueGTE(Col("high"), Col("tp")), price="tp")


# ---------------------------------------------------------------------------
# 1. A flatten-forced exit must not be priced at the untouched limit level
# ---------------------------------------------------------------------------


def _session_frame() -> pl.DataFrame:
    """One XNYS session, 15-minute bars, with a take-profit far out of reach.

    The crossover opens a position early; ``tp`` sits 2% above every high, so
    the limit never fires and the session close is what actually flattens the
    position.
    """
    ts = [
        datetime.datetime(2024, 1, 2, 9, 30) + datetime.timedelta(minutes=15 * i)
        for i in range(26)
    ]
    n = len(ts)
    closes = [100.0 - i * 0.05 for i in range(n)]   # gently declining tape
    opens = [c + 0.01 for c in closes]
    return pl.DataFrame({
        "date": ts,
        "open": opens,
        "high": [c + 0.05 for c in closes],
        "low": [c - 0.05 for c in closes],
        "close": closes,
        "fast": [1.0, 1.0] + [3.0] * (n - 2),   # crossover at bar 2
        "slow": [2.0] * n,
        "tp": [c * 1.02 for c in closes],       # never reached
    })


#: ``fast`` crosses ``slow`` at bar 2, and a signal fills at the NEXT bar's
#: open, so the position is opened at bar 3's open.
_ENTRY_FILL_BAR = 3
#: The session's last bar, where the flatten fills at that bar's own open.
_FLATTEN_FILL_BAR = 25


def test_flatten_forced_exit_is_not_priced_at_the_untouched_limit() -> None:
    """The limit never fires; the flatten fills at the flatten bar's own open.

    ``_limit_price`` carries the take-profit level on every bar, so a trade
    extractor that branches on ``_limit_price.is_not_null()`` books the exit at
    a price the tape never printed.
    """
    df = _session_frame()
    res = run(df, _CrossEntryTakeProfit(), calendar=get_calendar("XNYS"),
              flatten_eod=True)

    assert res.trades.height == 1, "fixture must produce exactly one trade"
    assert (df["high"] < df["tp"]).all(), "fixture: the limit must never fire"

    entry_fill = df["open"][_ENTRY_FILL_BAR]
    exit_fill = df["open"][_FLATTEN_FILL_BAR]
    expected = (exit_fill - entry_fill) / entry_fill

    assert res.trades["pnl"][0] == pytest.approx(expected, rel=1e-12), (
        "flatten-forced exit was booked at the take-profit level rather than "
        "at the flatten bar's open"
    )
    # The tape declines, so this trade loses. A positive pnl means the exit was
    # priced at the +2% take-profit that never traded.
    assert res.trades["pnl"][0] < 0.0


def test_trades_pnl_agrees_with_returns_when_the_limit_never_fires() -> None:
    """The two views of the same run must not disagree.

    ``mktlib.reports`` reads ``trades["pnl"]``; the equity curve reads
    ``returns``. When they disagree a losing strategy can report a 100% win
    rate.
    """
    df = _session_frame()
    res = run(df, _CrossEntryTakeProfit(), calendar=get_calendar("XNYS"),
              flatten_eod=True)

    equity_from_returns = float((1.0 + res.returns["return"]).product())
    equity_from_trades = float((1.0 + res.trades["pnl"]).product())
    assert equity_from_trades == pytest.approx(equity_from_returns, rel=1e-9), (
        f"trades.pnl implies {equity_from_trades:.6f} but returns implies "
        f"{equity_from_returns:.6f}"
    )


# ---------------------------------------------------------------------------
# 2. A same-bar limit fill must be measured from the entry fill, not prev_close
# ---------------------------------------------------------------------------


def test_same_bar_limit_entry_is_measured_from_the_entry_fill() -> None:
    """Limit fires on the very bar the entry filled.

    The position is opened at that bar's ``open`` and closed at the limit
    inside the same bar, so the return is ``(limit - open) / open``. Measuring
    from the previous close credits an overnight gap that was never held.
    """
    df = pl.DataFrame({
        "date": [datetime.date(2024, 1, d) for d in (1, 2, 3, 4)],
        # bar 2 gaps up from a 100.0 close to a 110.0 open and reaches 120.0,
        # so the 115.0 take-profit fills inside the bar the entry opened in.
        "open":  [100.0, 100.0, 110.0, 116.0],
        "high":  [100.0, 100.0, 120.0, 116.0],
        "low":   [100.0, 100.0, 109.0, 116.0],
        "close": [100.0, 100.0, 116.0, 116.0],
        "fast":  [1.0, 3.0, 3.0, 3.0],   # crossover at bar 1
        "slow":  [2.0] * 4,
        "tp":    [115.0] * 4,
    })
    res = run(df, _CrossEntryTakeProfit())

    entry_fill = 110.0   # bar 2's open
    limit_fill = 115.0
    expected = (limit_fill - entry_fill) / entry_fill

    assert res.returns["return"][2] == pytest.approx(expected, rel=1e-12), (
        "same-bar limit fill was measured from the previous close, crediting "
        "a gap the position never held"
    )
    assert res.trades["pnl"][0] == pytest.approx(expected, rel=1e-12)
