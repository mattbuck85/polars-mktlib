"""Unit tests for compute_trade_metrics()."""

from __future__ import annotations

import math

import polars as pl
import pytest

from mktlib.reports._stats import compute_trade_metrics, _max_consecutive
from mktlib.reports._types import TradeMetrics


def _make_trades(
    pnl: list[float],
    bars_held: list[int] | None = None,
    *,
    start_entry: str = "2024-01-02",
    days_apart: int = 5,
) -> pl.DataFrame:
    """Build a minimal trades DataFrame from PnL values."""
    n = len(pnl)
    if bars_held is None:
        bars_held = [days_apart] * n
    entry_dates = []
    exit_dates = []
    from datetime import date, timedelta
    base = date.fromisoformat(start_entry)
    for i in range(n):
        e = base + timedelta(days=i * days_apart)
        entry_dates.append(e)
        exit_dates.append(e + timedelta(days=days_apart - 1))
    return pl.DataFrame(
        {
            "entry_date": entry_dates,
            "exit_date": exit_dates,
            "side": pl.Series([1] * n, dtype=pl.Int8),
            "pnl": pnl,
            "bars_held": bars_held,
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestMaxConsecutive:
    def test_all_wins(self):
        assert _max_consecutive([0.1, 0.2, 0.3], positive=True) == 3

    def test_all_losses(self):
        assert _max_consecutive([-0.1, -0.2], positive=False) == 2

    def test_alternating(self):
        assert _max_consecutive([0.1, -0.1, 0.1, 0.1, -0.1], positive=True) == 2

    def test_empty(self):
        assert _max_consecutive([], positive=True) == 0

    def test_zero_not_a_win_or_loss(self):
        # zero pnl counts as neither win nor loss
        assert _max_consecutive([0.0, 0.1], positive=True) == 1
        assert _max_consecutive([0.0, -0.1], positive=False) == 1


# ---------------------------------------------------------------------------
# Empty trades
# ---------------------------------------------------------------------------


class TestEmptyTrades:
    def test_returns_zero_metrics(self):
        empty = pl.DataFrame(
            {
                "entry_date": pl.Series([], dtype=pl.Date),
                "exit_date": pl.Series([], dtype=pl.Date),
                "side": pl.Series([], dtype=pl.Int8),
                "pnl": pl.Series([], dtype=pl.Float64),
                "bars_held": pl.Series([], dtype=pl.Int64),
            }
        )
        tm = compute_trade_metrics(empty)
        assert tm.total_trades == 0
        assert tm.trade_win_rate == 0.0
        assert tm.trade_sharpe == 0.0


# ---------------------------------------------------------------------------
# Win rate
# ---------------------------------------------------------------------------


class TestWinRate:
    def test_all_winners(self):
        tm = compute_trade_metrics(_make_trades([0.01, 0.02, 0.03]))
        assert tm.trade_win_rate == pytest.approx(1.0)

    def test_all_losers(self):
        tm = compute_trade_metrics(_make_trades([-0.01, -0.02]))
        assert tm.trade_win_rate == pytest.approx(0.0)

    def test_half_half(self):
        tm = compute_trade_metrics(_make_trades([0.02, -0.01, 0.03, -0.01]))
        assert tm.trade_win_rate == pytest.approx(0.5)

    def test_total_trades(self):
        tm = compute_trade_metrics(_make_trades([0.01, -0.01, 0.02]))
        assert tm.total_trades == 3


# ---------------------------------------------------------------------------
# Payoff ratio / profit factor
# ---------------------------------------------------------------------------


class TestPayoffAndProfitFactor:
    def test_payoff_ratio(self):
        # avg_win = 0.025, avg_loss = 0.01 → 2.5
        tm = compute_trade_metrics(_make_trades([0.02, -0.01, 0.03, -0.01]))
        assert tm.payoff_ratio == pytest.approx(2.5, rel=1e-4)

    def test_profit_factor(self):
        # sum_win = 0.05, sum_loss = 0.02 → 2.5
        tm = compute_trade_metrics(_make_trades([0.02, -0.01, 0.03, -0.01]))
        assert tm.profit_factor == pytest.approx(2.5, rel=1e-4)

    def test_no_losers_profit_factor_inf(self):
        tm = compute_trade_metrics(_make_trades([0.01, 0.02]))
        assert math.isinf(tm.profit_factor)

    def test_no_winners_payoff_zero(self):
        # No winners → avg_win = 0.0 → payoff ratio = 0.0
        tm = compute_trade_metrics(_make_trades([-0.01, -0.02]))
        assert tm.payoff_ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Kelly criterion
# ---------------------------------------------------------------------------


class TestKellyCriterion:
    def test_known_values(self):
        # win_rate=0.5, payoff=2.5 → kelly = 0.5 - 0.5/2.5 = 0.3
        tm = compute_trade_metrics(_make_trades([0.02, -0.01, 0.03, -0.01]))
        assert tm.kelly_criterion == pytest.approx(0.3, rel=1e-4)


# ---------------------------------------------------------------------------
# Avg metrics
# ---------------------------------------------------------------------------


class TestAvgMetrics:
    def test_avg_trade_pnl(self):
        tm = compute_trade_metrics(_make_trades([0.02, -0.01, 0.03, -0.01]))
        assert tm.avg_trade_pnl == pytest.approx(0.0075, rel=1e-4)

    def test_avg_bars_held(self):
        tm = compute_trade_metrics(
            _make_trades([0.01, -0.01], bars_held=[4, 8])
        )
        assert tm.avg_bars_held == pytest.approx(6.0)

    def test_avg_winner_loser(self):
        tm = compute_trade_metrics(_make_trades([0.02, -0.01, 0.03, -0.01]))
        assert tm.avg_winner == pytest.approx(0.025, rel=1e-4)
        assert tm.avg_loser == pytest.approx(-0.01, rel=1e-4)

    def test_largest_winner_loser(self):
        tm = compute_trade_metrics(_make_trades([0.05, -0.02, 0.01, -0.10]))
        assert tm.largest_winner == pytest.approx(0.05)
        assert tm.largest_loser == pytest.approx(-0.10)


# ---------------------------------------------------------------------------
# Consecutive runs
# ---------------------------------------------------------------------------


class TestConsecutiveRuns:
    def test_max_consecutive_wins(self):
        # W W L W W W L
        tm = compute_trade_metrics(
            _make_trades([0.01, 0.02, -0.01, 0.01, 0.02, 0.03, -0.01])
        )
        assert tm.max_consecutive_wins == 3

    def test_max_consecutive_losses(self):
        # L L L W
        tm = compute_trade_metrics(
            _make_trades([-0.01, -0.02, -0.01, 0.05])
        )
        assert tm.max_consecutive_losses == 3


# ---------------------------------------------------------------------------
# Risk-adjusted
# ---------------------------------------------------------------------------


class TestRiskAdjusted:
    def test_trade_sharpe_positive(self):
        # Mostly winning trades → positive Sharpe
        tm = compute_trade_metrics(
            _make_trades([0.02, 0.01, 0.03, -0.005, 0.015])
        )
        assert tm.trade_sharpe > 0

    def test_trade_sortino_positive(self):
        # Two losers with different values → downside std > 0
        tm = compute_trade_metrics(
            _make_trades([0.02, 0.01, 0.03, -0.005, -0.015])
        )
        assert tm.trade_sortino > 0

    def test_trades_per_year(self):
        # 4 trades over ~1 year → ~4 trades/year
        from datetime import date
        trades = pl.DataFrame(
            {
                "entry_date": [
                    date(2024, 1, 1),
                    date(2024, 4, 1),
                    date(2024, 7, 1),
                    date(2024, 10, 1),
                ],
                "exit_date": [
                    date(2024, 2, 1),
                    date(2024, 5, 1),
                    date(2024, 8, 1),
                    date(2024, 11, 1),
                ],
                "side": pl.Series([1, 1, 1, 1], dtype=pl.Int8),
                "pnl": [0.01, -0.01, 0.02, -0.005],
                "bars_held": [30, 30, 30, 30],
            }
        )
        tm = compute_trade_metrics(trades)
        # span ≈ 9 months (0.75 years) → 4 / 0.75 ≈ 5.3; allow generous range
        assert 3.0 < tm.trades_per_year < 10.0


class TestSingleTrade:
    def test_single_winner(self):
        tm = compute_trade_metrics(_make_trades([0.05]))
        assert tm.total_trades == 1
        assert tm.trade_win_rate == 1.0
        assert tm.avg_winner == pytest.approx(0.05)
        assert tm.avg_loser == 0.0
        assert tm.payoff_ratio == float("inf")
        # Sharpe and Sortino: n <= 1, std = 0 → 0
        assert tm.trade_sharpe == 0.0
        assert tm.trade_sortino == 0.0

    def test_single_loser(self):
        tm = compute_trade_metrics(_make_trades([-0.03]))
        assert tm.total_trades == 1
        assert tm.trade_win_rate == 0.0
        assert tm.avg_winner == 0.0
        assert tm.avg_loser == pytest.approx(-0.03)
        assert tm.trade_sharpe == 0.0
        assert tm.trade_sortino == 0.0

    def test_constant_pnl_zero_sharpe(self):
        # All same pnl → std=0 → sharpe=0
        tm = compute_trade_metrics(_make_trades([0.01, 0.01, 0.01]))
        assert tm.trade_sharpe == pytest.approx(0.0)

    def test_no_downside_sortino_zero(self):
        # All positive → no downside → sortino=0
        tm = compute_trade_metrics(_make_trades([0.01, 0.02, 0.03]))
        assert tm.trade_sortino == pytest.approx(0.0)
