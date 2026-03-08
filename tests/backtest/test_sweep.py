from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest

from mktlib.backtest._conditions import Crossover, Crossunder
from mktlib.backtest._sweep import sweep


@dataclass(frozen=True, slots=True)
class SweepableStrategy:
    fast: str = "fast_a"
    slow: str = "slow"

    def entry(self) -> Crossover:
        return Crossover(self.fast, self.slow)

    def exit(self) -> Crossunder:
        return Crossunder(self.fast, self.slow)


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 8), eager=True),
            "open": [100.0, 100.5, 101.5, 103.5, 105.5, 103.5, 99.5, 97.5],
            "close": [100.0, 101.0, 103.0, 105.0, 104.0, 100.0, 98.0, 97.0],
            # fast_a crosses above slow at index 2
            "fast_a": [1.0, 1.0, 3.0, 4.0, 3.5, 1.0, 0.5, 0.3],
            # fast_b never crosses above slow
            "fast_b": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "slow": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )


class TestSweep:
    def test_grid_size(self, df: pl.DataFrame) -> None:
        result = sweep(
            df,
            SweepableStrategy,
            {"fast": ["fast_a", "fast_b"]},
        )
        assert len(result.results) == 2

    def test_best_returns_best(self, df: pl.DataFrame) -> None:
        result = sweep(
            df,
            SweepableStrategy,
            {"fast": ["fast_a", "fast_b"]},
        )
        best_params, best_result = result.best("total_return")
        # fast_b never enters, so total_return = 0
        # fast_a enters and loses money, but it's the only one with trades
        # Actually fast_b has 0 return, fast_a has negative return from the data
        # So fast_b should be "best" with 0 return
        assert best_params["fast"] == "fast_b"

    def test_to_frame(self, df: pl.DataFrame) -> None:
        result = sweep(
            df,
            SweepableStrategy,
            {"fast": ["fast_a", "fast_b"]},
        )
        frame = result.to_frame()
        assert "total_return" in frame.columns
        assert "max_drawdown" in frame.columns
        assert "win_rate" in frame.columns
        assert "num_trades" in frame.columns
        assert frame.height == 2

    def test_multi_param_grid(self, df: pl.DataFrame) -> None:
        """2 x 1 = 2 combos (slow has only one option)."""
        result = sweep(
            df,
            SweepableStrategy,
            {"fast": ["fast_a", "fast_b"], "slow": ["slow"]},
        )
        assert len(result.results) == 2
