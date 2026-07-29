"""Schema validation tests for mktlib.backtest engine output."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest

from mktlib.backtest import Bracket, Cost, Crossover, Crossunder, run

from tests.schemas.backtest import (
    IntradayTradesSchema,
    ReturnsSchema,
    SignalsSchemaBase,
    TradesSchema,
)


@dataclass(frozen=True, slots=True)
class SimpleCrossStrategy:
    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Crossunder:
        return Crossunder("fast", "slow")


@pytest.fixture()
def ohlcv() -> pl.DataFrame:
    """Synthetic data where fast crosses above slow at bar 2, below at bar 5."""
    return pl.DataFrame(
        {
            "date": pl.date_range(
                pl.date(2024, 1, 1), pl.date(2024, 1, 8), eager=True
            ),
            "open": [100.0, 100.5, 101.5, 103.5, 105.5, 103.5, 99.5, 97.5],
            "high": [101.0, 101.5, 103.0, 105.0, 106.0, 104.0, 100.0, 98.0],
            "low": [99.0, 100.0, 101.0, 103.0, 104.0, 99.0, 98.5, 97.0],
            "close": [100.5, 101.5, 102.5, 104.5, 103.0, 99.0, 98.0, 97.5],
            "fast": [100.0, 100.5, 101.5, 103.0, 104.0, 102.0, 99.0, 97.5],
            "slow": [100.0, 100.0, 100.5, 101.0, 102.0, 103.0, 102.0, 100.0],
        }
    )


@pytest.fixture()
def intraday_ohlcv() -> pl.DataFrame:
    """The same crossover pattern on contiguous 1-minute bars, one session.

    Every bar is one minute after the last and they all fall on the same
    calendar day, so "bars between the fills" and "calendar days between the
    dates" cannot be confused for one another here.
    """
    return pl.DataFrame(
        {
            "date": pl.datetime_range(
                pl.datetime(2024, 1, 1, 9, 30),
                pl.datetime(2024, 1, 1, 9, 37),
                interval="1m",
                eager=True,
            ),
            "open": [100.0, 100.5, 101.5, 103.5, 105.5, 103.5, 99.5, 97.5],
            "high": [101.0, 101.5, 103.0, 105.0, 106.0, 104.0, 100.0, 98.0],
            "low": [99.0, 100.0, 101.0, 103.0, 104.0, 99.0, 98.5, 97.0],
            "close": [100.5, 101.5, 102.5, 104.5, 103.0, 99.0, 98.0, 97.5],
            "fast": [100.0, 100.5, 101.5, 103.0, 104.0, 102.0, 99.0, 97.5],
            "slow": [100.0, 100.0, 100.5, 101.0, 102.0, 103.0, 102.0, 100.0],
        }
    )


@pytest.fixture()
def flat_ohlcv() -> pl.DataFrame:
    """No crossover: fast always below slow → no trades."""
    return pl.DataFrame(
        {
            "date": pl.date_range(
                pl.date(2024, 1, 1), pl.date(2024, 1, 5), eager=True
            ),
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "fast": [90.0] * 5,
            "slow": [100.0] * 5,
        }
    )


class TestBacktestSchemas:
    def test_returns(self, ohlcv: pl.DataFrame):
        result = run(ohlcv, SimpleCrossStrategy())
        ReturnsSchema.validate(result.returns)

    def test_trades(self, ohlcv: pl.DataFrame):
        result = run(ohlcv, SimpleCrossStrategy())
        TradesSchema.validate(result.trades)
        # Cross-column invariant
        if result.trades.height > 0:
            assert (
                result.trades["exit_date"] >= result.trades["entry_date"]
            ).all()

    def test_signals(self, ohlcv: pl.DataFrame):
        result = run(ohlcv, SimpleCrossStrategy())
        SignalsSchemaBase.validate(result.signals)


class TestBarsHeldCountsBars:
    """``bars_held`` counts BARS, as the documented schema says it does.

    Stated on contiguous 1-minute bars inside a single session, where the two
    readings diverge maximally: every trade spans zero calendar days, so a
    day-based computation collapses the whole column to 0.

    The definition is ``exit_fill_bar - entry_fill_bar``. Entries always fill at
    the next bar's open, so the entry fill bar is ``entry_signal_row + 1``; the
    exit fill bar follows the same priority order ``exit_price`` uses — a
    bracket or limit fills inside its own bar, everything else at the next
    open. That makes the two cases below differ by exactly one bar.
    """

    def test_next_open_exit_counts_bar_intervals(
        self, intraday_ohlcv: pl.DataFrame
    ) -> None:
        result = run(intraday_ohlcv, SimpleCrossStrategy())
        IntradayTradesSchema.validate(result.trades)
        assert result.trades.height > 0, "fixture must produce a trade"

        # Entry signal 09:31 → fills bar 09:32. Exit signal 09:35 → fills bar
        # 09:36. Four bars held.
        trade = result.trades.row(0, named=True)
        assert trade["bars_held"] == 4

        # Both legs fill at the next open, so the fill-to-fill distance is the
        # signal-to-signal distance — which on 1-minute bars is the minute gap.
        expected = (
            result.trades["exit_date"] - result.trades["entry_date"]
        ).dt.total_minutes()
        assert result.trades["bars_held"].equals(expected, check_dtypes=False)

    def test_same_bar_exit_is_one_bar_shorter(
        self, intraday_ohlcv: pl.DataFrame
    ) -> None:
        """A bracket fills inside the bar that tagged it, not at the next open,
        so it is held exactly one bar less than the signal dates suggest.

        The take-profit is deliberately wide enough that the level is not tagged
        on the entry fill bar itself. A bracket that closes on its own entry bar
        is genuinely ``bars_held == 0``, which is also what the calendar-day
        computation returns — such a case cannot tell the two apart.
        """
        result = run(
            intraday_ohlcv,
            SimpleCrossStrategy(),
            bracket=Bracket(take_profit=0.04),
        )
        IntradayTradesSchema.validate(result.trades)
        assert result.trades.height > 0, "fixture must trigger the bracket"

        # Entry fills bar 09:32 at 101.5; the level 105.56 is first tagged by
        # the 09:34 bar's high of 106.0, and fills inside that bar. Two bars.
        trade = result.trades.row(0, named=True)
        assert trade["bars_held"] == 2

        minutes = (
            result.trades["exit_date"] - result.trades["entry_date"]
        ).dt.total_minutes()
        assert result.trades["bars_held"].equals(minutes - 1, check_dtypes=False)


class TestCostLeavesSchemasUnchanged:
    """A ``cost=`` run must not add, drop or retype a single column."""

    def test_returns(self, ohlcv: pl.DataFrame):
        result = run(ohlcv, SimpleCrossStrategy(), cost=Cost(commission_bps=5.0))
        ReturnsSchema.validate(result.returns)

    def test_trades(self, ohlcv: pl.DataFrame):
        result = run(ohlcv, SimpleCrossStrategy(), cost=Cost(commission_bps=5.0))
        TradesSchema.validate(result.trades)

    def test_signals(self, ohlcv: pl.DataFrame):
        base = run(ohlcv, SimpleCrossStrategy())
        result = run(ohlcv, SimpleCrossStrategy(), cost=Cost(commission_bps=5.0))
        SignalsSchemaBase.validate(result.signals)
        assert result.signals.schema == base.signals.schema


class TestBracketLeavesSchemasUnchanged:
    """A ``bracket=`` run must not add, drop or retype a single column.

    The bracket working columns (``_bracket_level``, ``_bracket_kind``, the
    per-leg levels, the block id) are internal and dropped before the
    result is handed back — this is what pins that.
    """

    BRACKET = Bracket(take_profit=0.01, stop_loss=0.01)

    def test_returns(self, ohlcv: pl.DataFrame):
        result = run(ohlcv, SimpleCrossStrategy(), bracket=self.BRACKET)
        ReturnsSchema.validate(result.returns)

    def test_trades(self, ohlcv: pl.DataFrame):
        result = run(ohlcv, SimpleCrossStrategy(), bracket=self.BRACKET)
        TradesSchema.validate(result.trades)
        assert result.trades.height > 0, "fixture must actually trigger the bracket"

    def test_signals(self, ohlcv: pl.DataFrame):
        base = run(ohlcv, SimpleCrossStrategy())
        result = run(ohlcv, SimpleCrossStrategy(), bracket=self.BRACKET)
        SignalsSchemaBase.validate(result.signals)
        assert result.signals.schema == base.signals.schema

    def test_signals_with_cost(self, ohlcv: pl.DataFrame):
        base = run(ohlcv, SimpleCrossStrategy())
        result = run(
            ohlcv,
            SimpleCrossStrategy(),
            bracket=self.BRACKET,
            cost=Cost(commission_bps=5.0),
        )
        SignalsSchemaBase.validate(result.signals)
        assert result.signals.schema == base.signals.schema


class TestEmptyTrades:
    def test_no_signal(self, flat_ohlcv: pl.DataFrame):
        result = run(flat_ohlcv, SimpleCrossStrategy())
        ReturnsSchema.validate(result.returns)
        TradesSchema.validate(result.trades)
        assert result.trades.height == 0
        SignalsSchemaBase.validate(result.signals)
