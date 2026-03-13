from __future__ import annotations

import enum
import functools
from collections.abc import Iterator, ItemsView
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable


class TradeSide(enum.IntEnum):
    """Trade direction: +1 for long, -1 for short.

    ``IntEnum`` so it works directly as a numeric multiplier.
    """

    LONG = 1
    SHORT = -1

if TYPE_CHECKING:
    from typing import Literal

    from mktlib.backtest._conditions import Condition

import polars as pl


@runtime_checkable
class Strategy(Protocol):
    """Any object with ``entry()`` and ``exit()`` returning Conditions.

    Strategies may optionally define an ``init(self, df) -> pl.DataFrame``
    method to enrich the DataFrame with indicator columns before signal
    evaluation. This lets strategies encapsulate their own indicator setup,
    making them self-contained. The ``init`` method is **not** part of the
    Protocol to avoid breaking existing strategies that don't need it.
    """

    def entry(self) -> Condition | pl.Expr: ...
    def exit(self) -> Condition | pl.Expr: ...


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Result of a single-symbol backtest run."""

    returns: pl.DataFrame
    """``(date, return)`` daily strategy returns."""
    trades: pl.DataFrame
    """``(entry_date, exit_date, pnl, bars_held)`` per-trade log."""
    signals: pl.DataFrame
    """Full frame with ``_entry``, ``_exit``, ``_position`` columns."""


class MultiBacktestResult:
    """Result of a multi-symbol backtest run.

    Stores per-symbol :class:`BacktestResult` instances for O(1) access via
    ``result["AAPL"]``. Combined DataFrames with the symbol column prepended
    are available via :attr:`returns`, :attr:`trades`, and :attr:`signals`
    properties (lazy-cached on first access).
    """

    __slots__ = ("_by_instrument", "_instrument_col", "__dict__")

    def __init__(
        self,
        by_instrument: dict[str, BacktestResult],
        *,
        instrument_col: str,
    ) -> None:
        self._by_instrument = by_instrument
        self._instrument_col = instrument_col

    # -- dict-like access --------------------------------------------------

    def __getitem__(self, symbol: str) -> BacktestResult:
        return self._by_instrument[symbol]

    def __len__(self) -> int:
        return len(self._by_instrument)

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_instrument)

    def __contains__(self, symbol: object) -> bool:
        return symbol in self._by_instrument

    def items(self) -> ItemsView[str, BacktestResult]:
        return self._by_instrument.items()

    @property
    def symbols(self) -> list[str]:
        """Ordered list of symbol keys."""
        return list(self._by_instrument)

    # -- combined views (lazy-cached) --------------------------------------

    def _concat_field(self, field: Literal["returns", "trades", "signals"]) -> pl.DataFrame:
        """Concatenate a BacktestResult field across symbols, symbol col first."""
        frames = [
            cast(pl.DataFrame, getattr(result, field)).with_columns(
                pl.lit(symbol).alias(self._instrument_col)
            )
            for symbol, result in self._by_instrument.items()
        ]
        combined = pl.concat(frames)
        other_cols = [c for c in combined.columns if c != self._instrument_col]
        return combined.select(self._instrument_col, *other_cols)

    @functools.cached_property
    def returns(self) -> pl.DataFrame:
        """``(symbol, date, return)`` — all symbols concatenated."""
        return self._concat_field("returns")

    @functools.cached_property
    def trades(self) -> pl.DataFrame:
        """``(symbol, entry_date, exit_date, pnl, bars_held)`` — all symbols."""
        return self._concat_field("trades")

    @functools.cached_property
    def signals(self) -> pl.DataFrame:
        """``(symbol, ..., _entry, _exit, _position)`` — all symbols."""
        return self._concat_field("signals")
