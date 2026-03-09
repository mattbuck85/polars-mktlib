from __future__ import annotations

from mktlib.backtest._conditions import Crossover, Crossunder
from mktlib.backtest._types import TradeSide


class MacdCrossover:
    """MACD crossover strategy.

    Expects the DataFrame to have columns for MACD line and signal line
    (e.g. added via ``polars-talib``).
    """

    def entry(self) -> Crossover:
        return Crossover("macd", "macd_signal")

    def exit(self) -> Crossunder:
        return Crossunder("macd", "macd_signal")


class MacdCrossoverShort:
    """MACD crossover strategy for short side.

    Entry when MACD crosses *below* signal (bearish), exit when it crosses
    *above* (bullish).  The entry condition carries ``TradeSide.SHORT`` so the
    engine automatically applies short-side return logic.
    """

    def entry(self) -> Crossunder:
        return Crossunder("macd", "macd_signal", trade_side=TradeSide.SHORT)

    def exit(self) -> Crossover:
        return Crossover("macd", "macd_signal")
