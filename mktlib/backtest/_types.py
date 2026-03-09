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

    def entry(self) -> Condition: ...
    def exit(self) -> Condition: ...


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Result of a single backtest run."""

    returns: pl.DataFrame
    """``(date, return)`` daily strategy returns."""
    trades: pl.DataFrame
    """``(entry_date, exit_date, pnl, bars_held)`` per-trade log."""
    signals: pl.DataFrame
    """Full frame with ``_entry``, ``_exit``, ``_position`` columns."""


@dataclass(frozen=True, slots=True)
class SweepResult:
    """Result of a parameter sweep."""

    results: list[tuple[dict[str, object], BacktestResult]]

    def best(self, metric: str = "total_return") -> tuple[dict[str, object], BacktestResult]:
        """Return the params and result that maximise *metric*."""
        return max(self.results, key=lambda r: _compute_metric(r[1], metric))

    def to_frame(self) -> pl.DataFrame:
        """All parameter combos with summary stats as columns."""
        rows: list[dict[str, object]] = []
        for params, result in self.results:
            rets = result.returns["return"]
            cum = (1 + rets).product() - 1  # type: ignore[operator]
            equity = (1 + rets).cum_prod()
            running_max = equity.cum_max()
            dd = (equity - running_max) / running_max
            max_dd = dd.min()
            wins = result.trades.filter(pl.col("pnl") > 0).height
            total = result.trades.height
            rows.append(
                {
                    **params,
                    "total_return": cum,
                    "max_drawdown": max_dd,
                    "win_rate": wins / total if total > 0 else 0.0,
                    "num_trades": total,
                }
            )
        return pl.DataFrame(rows)


def _compute_metric(result: BacktestResult, metric: str) -> float:
    rets = result.returns["return"]
    if metric == "total_return":
        val = (1 + rets).product() - 1  # type: ignore[operator]
    elif metric == "max_drawdown":
        equity = (1 + rets).cum_prod()
        running_max = equity.cum_max()
        dd = (equity - running_max) / running_max
        val = dd.min()  # least negative = best
    else:
        msg = f"Unknown metric: {metric!r}"
        raise ValueError(msg)
    return float(val)  # type: ignore[arg-type]
