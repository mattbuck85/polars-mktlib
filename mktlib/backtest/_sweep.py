from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import polars as pl

from mktlib.backtest._engine import run
from mktlib.backtest._types import SweepResult, TradeSide

if TYPE_CHECKING:
    from mktlib.backtest._types import Strategy
    from mktlib.scheduling import ExchangeCalendar


def sweep(
    df: pl.DataFrame,
    strategy_cls: type[Strategy],
    params: dict[str, list[object]],
    *,
    metric: str = "total_return",
    trade_side: TradeSide = TradeSide.LONG,
    calendar: ExchangeCalendar | None = None,
    flatten_eod: bool = False,
) -> SweepResult:
    """Run a parameter grid sweep over *strategy_cls*.

    Parameters
    ----------
    df
        OHLCV + indicator DataFrame.
    strategy_cls
        Strategy class (not instance) — instantiated per combo.
    params
        ``{field_name: [values]}`` grid.
    metric
        Metric used by ``SweepResult.best()``.
    """
    keys = list(params.keys())
    values = list(params.values())
    results: list[tuple[dict[str, object], object]] = []

    for combo in itertools.product(*values):
        param_dict = dict(zip(keys, combo))
        strategy = strategy_cls(**param_dict)  # type: ignore[call-arg]
        result = run(
            df,
            strategy,
            trade_side=trade_side,
            calendar=calendar,
            flatten_eod=flatten_eod,
        )
        results.append((param_dict, result))

    return SweepResult(results=results)  # type: ignore[arg-type]
