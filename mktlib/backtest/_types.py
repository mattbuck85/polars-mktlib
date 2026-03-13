from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable


class TradeSide(enum.IntEnum):
    """Trade direction: +1 for long, -1 for short.

    ``IntEnum`` so it works directly as a numeric multiplier.
    """

    LONG = 1
    SHORT = -1

if TYPE_CHECKING:
    from mktlib.backtest._conditions import Condition

import polars as pl


@runtime_checkable
class Strategy(Protocol):
    """Any object with ``entry()`` and ``exit()`` returning Conditions."""

    def entry(self) -> Condition | pl.Expr: ...
    def exit(self) -> Condition | pl.Expr: ...


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Result of a single backtest run."""

    returns: pl.DataFrame
    """``(date, return)`` daily strategy returns."""
    trades: pl.DataFrame
    """``(entry_date, exit_date, pnl, bars_held)`` per-trade log."""
    signals: pl.DataFrame
    """Full frame with ``_entry``, ``_exit``, ``_position`` columns."""
