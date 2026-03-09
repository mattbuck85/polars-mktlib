from __future__ import annotations

from dataclasses import dataclass

from mktlib.backtest._conditions import Crossover, Crossunder
from mktlib.backtest._types import TradeSide


@dataclass(frozen=True, slots=True)
class MacdCrossover:
    """MACD crossover strategy.

    Expects the DataFrame to have columns for MACD line and signal line
    (e.g. added via ``polars-talib``).
    """

    macd_col: str = "macd"
    signal_col: str = "macd_signal"

    def entry(self) -> Crossover:
        return Crossover(self.macd_col, self.signal_col)

    def exit(self) -> Crossunder:
        return Crossunder(self.macd_col, self.signal_col)


@dataclass(frozen=True, slots=True)
class MacdCrossoverShort:
    """MACD crossover strategy for short side.

    Entry when MACD crosses *below* signal (bearish), exit when it crosses
    *above* (bullish).  The entry condition carries ``TradeSide.SHORT`` so the
    engine automatically applies short-side return logic.
    """

    macd_col: str = "macd"
    signal_col: str = "macd_signal"

    def entry(self) -> Crossunder:
        return Crossunder(self.macd_col, self.signal_col, trade_side=TradeSide.SHORT)

    def exit(self) -> Crossover:
        return Crossover(self.macd_col, self.signal_col)
