"""Schema validation tests for mktlib.backtest engine output."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest

from mktlib.backtest import Bracket, Cost, Crossover, Crossunder, run

from tests.schemas.backtest import ReturnsSchema, SignalsSchemaBase, TradesSchema


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
