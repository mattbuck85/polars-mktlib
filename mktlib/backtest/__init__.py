from __future__ import annotations

try:
    import polars_talib as _plta  # noqa: F401  # type: ignore[import-not-found]
except ModuleNotFoundError as _e:
    raise ModuleNotFoundError(
        "mktlib.backtest requires polars-talib. "
        "Install with: pip install mktlib[backtest]"
    ) from _e

from mktlib.backtest._conditions import (
    All,
    Any_,
    Condition,
    Crossover,
    Crossunder,
    IsFalling,
    IsRising,
    Not,
    PriceIsAbove,
    PriceIsBelow,
)
from mktlib.backtest._engine import run
from mktlib.backtest._sweep import sweep
from mktlib.backtest._types import BacktestResult, Strategy, SweepResult, TradeSide

__all__ = [
    "All",
    "Any_",
    "BacktestResult",
    "Condition",
    "Crossover",
    "Crossunder",
    "IsFalling",
    "IsRising",
    "Not",
    "PriceIsAbove",
    "PriceIsBelow",
    "Strategy",
    "SweepResult",
    "TradeSide",
    "run",
    "sweep",
]
