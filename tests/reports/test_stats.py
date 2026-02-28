from __future__ import annotations

import datetime as dt
import math

import polars as pl
import pytest

from mktlib.reports._stats import (
    compute_metrics,
    cumulative_returns,
    drawdown_series,
    monthly_returns,
    rolling_sharpe,
    rolling_volatility,
    yearly_returns,
)
from mktlib.reports._types import ReportConfig


def _make_returns(values: list[float], start: dt.date = dt.date(2024, 1, 2)) -> pl.DataFrame:
    """Build a returns DataFrame with business-day dates."""
    dates: list[dt.date] = []
    current = start
    for _ in values:
        while current.weekday() >= 5:
            current += dt.timedelta(days=1)
        dates.append(current)
        current += dt.timedelta(days=1)
    return pl.DataFrame({"date": dates, "return": values})


# ---------------------------------------------------------------------------
# Cumulative helpers
# ---------------------------------------------------------------------------

class TestCumulativeReturns:
    def test_compounded(self):
        ret = pl.Series("r", [0.10, 0.10, -0.05])
        cum = cumulative_returns(ret, compounded=True)
        # (1.1 * 1.1 * 0.95) - 1 = 0.1495
        assert cum[-1] == pytest.approx(0.1495, abs=1e-6)

    def test_simple(self):
        ret = pl.Series("r", [0.10, 0.10, -0.05])
        cum = cumulative_returns(ret, compounded=False)
        assert cum[-1] == pytest.approx(0.15, abs=1e-6)

    def test_empty(self):
        ret = pl.Series("r", [], dtype=pl.Float64)
        assert len(cumulative_returns(ret)) == 0


class TestDrawdownSeries:
    def test_basic(self):
        # Up 10%, then down 20% → wealth 1.1 → 0.88, running max 1.1
        ret = pl.Series("r", [0.10, -0.20])
        dd = drawdown_series(ret, compounded=True)
        assert dd[0] == pytest.approx(0.0)  # No drawdown at peak
        assert dd[1] == pytest.approx(0.88 / 1.1 - 1, abs=1e-6)

    def test_no_drawdown(self):
        ret = pl.Series("r", [0.01, 0.02, 0.01])
        dd = drawdown_series(ret)
        assert dd.min() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Grouped returns
# ---------------------------------------------------------------------------

class TestMonthlyReturns:
    def test_groups_by_month(self):
        returns = _make_returns([0.01, 0.02, -0.01], start=dt.date(2024, 1, 2))
        monthly = monthly_returns(returns, compounded=True)
        assert len(monthly) == 1  # All in Jan 2024
        expected = (1.01 * 1.02 * 0.99) - 1
        assert monthly["monthly_return"][0] == pytest.approx(expected, abs=1e-6)


class TestYearlyReturns:
    def test_groups_by_year(self):
        returns = _make_returns([0.01] * 5, start=dt.date(2024, 1, 2))
        yearly = yearly_returns(returns, compounded=True)
        assert len(yearly) == 1
        expected = 1.01**5 - 1
        assert yearly["yearly_return"][0] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Rolling
# ---------------------------------------------------------------------------

class TestRolling:
    def test_rolling_sharpe_short_series(self):
        ret = pl.Series("r", [0.01] * 10)
        rs = rolling_sharpe(ret, window=126)
        assert all(v is None for v in rs.to_list())

    def test_rolling_volatility_short_series(self):
        ret = pl.Series("r", [0.01] * 10)
        rv = rolling_volatility(ret, window=126)
        assert all(v is None for v in rv.to_list())

    def test_rolling_sharpe_long_enough(self):
        ret = pl.Series("r", [0.001] * 200)
        rs = rolling_sharpe(ret, window=126)
        # First 124 values are None, index 125 is the first computed value (126th element)
        assert rs[124] is None
        assert rs[125] is not None


# ---------------------------------------------------------------------------
# Full metrics computation
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    @pytest.fixture
    def daily_returns(self) -> pl.DataFrame:
        """252 days of returns with slight variation (mostly positive, some negative)."""
        import math
        return _make_returns([0.001 + 0.002 * math.sin(i * 0.5) for i in range(252)])

    @pytest.fixture
    def config(self) -> ReportConfig:
        return ReportConfig()

    def test_cumulative_return(self, daily_returns: pl.DataFrame, config: ReportConfig):
        result = compute_metrics(daily_returns, None, config)
        assert result.cumulative_return > 0  # Net positive returns

    def test_cagr_one_year(self, daily_returns: pl.DataFrame, config: ReportConfig):
        result = compute_metrics(daily_returns, None, config)
        # 1 year of data → CAGR ≈ cumulative return
        assert result.cagr == pytest.approx(result.cumulative_return, rel=1e-4)

    def test_sharpe_positive(self, daily_returns: pl.DataFrame, config: ReportConfig):
        result = compute_metrics(daily_returns, None, config)
        assert result.sharpe > 0

    def test_sortino_positive(self, daily_returns: pl.DataFrame, config: ReportConfig):
        result = compute_metrics(daily_returns, None, config)
        assert result.sortino > 0

    def test_max_drawdown_bounded(self, daily_returns: pl.DataFrame, config: ReportConfig):
        result = compute_metrics(daily_returns, None, config)
        assert -1.0 <= result.max_drawdown <= 0.0

    def test_win_rate_reasonable(self, daily_returns: pl.DataFrame, config: ReportConfig):
        result = compute_metrics(daily_returns, None, config)
        assert 0.0 < result.win_rate < 1.0

    def test_benchmark_metrics(self, daily_returns: pl.DataFrame, config: ReportConfig):
        bench = _make_returns([0.0005] * 252)
        result = compute_metrics(daily_returns, bench, config)
        assert result.alpha is not None
        assert result.beta is not None
        assert result.r_squared is not None
        assert result.information_ratio is not None

    def test_no_benchmark(self, daily_returns: pl.DataFrame, config: ReportConfig):
        result = compute_metrics(daily_returns, None, config)
        assert result.alpha is None
        assert result.beta is None

    def test_empty_returns(self, config: ReportConfig):
        empty = pl.DataFrame({"date": [], "return": []}, schema={"date": pl.Date, "return": pl.Float64})
        result = compute_metrics(empty, None, config)
        assert result.cumulative_return == 0.0
        assert result.sharpe == 0.0
        assert result.win_rate == 0.0


class TestMetricsWithDrawdowns:
    def test_drawdown_metrics(self):
        # Create returns with a clear drawdown: up 10%, then down 15%, then up 5%
        values = [0.05, 0.05, -0.08, -0.08, 0.03, 0.03]
        returns = _make_returns(values)
        config = ReportConfig()
        result = compute_metrics(returns, None, config)
        assert result.max_drawdown < 0
        assert result.longest_drawdown_days > 0
        assert result.avg_drawdown < 0

    def test_var_cvar(self):
        values = [0.02, -0.05, 0.01, -0.03, 0.015, -0.04, 0.005, -0.02, 0.01, -0.01]
        returns = _make_returns(values)
        config = ReportConfig()
        result = compute_metrics(returns, None, config)
        # VaR should be negative (worst quantile)
        assert result.var_95 < 0
        # CVaR should be <= VaR (further into the tail)
        assert result.cvar_95 <= result.var_95

    def test_profit_factor(self):
        values = [0.02, -0.01, 0.03, -0.01]
        returns = _make_returns(values)
        config = ReportConfig()
        result = compute_metrics(returns, None, config)
        # gains = 0.02 + 0.03 = 0.05, losses = 0.01 + 0.01 = 0.02
        assert result.profit_factor == pytest.approx(2.5, rel=1e-4)

    def test_kelly_criterion(self):
        values = [0.02, -0.01, 0.03, -0.01]
        returns = _make_returns(values)
        config = ReportConfig()
        result = compute_metrics(returns, None, config)
        # win_rate=0.5, payoff_ratio = avg_win/avg_loss = 0.025/0.01 = 2.5
        # kelly = 0.5 - 0.5/2.5 = 0.5 - 0.2 = 0.3
        assert result.kelly_criterion == pytest.approx(0.3, rel=1e-4)

    def test_omega_ratio(self):
        values = [0.02, -0.01, 0.03, -0.01]
        returns = _make_returns(values)
        config = ReportConfig(rf=0.0)
        result = compute_metrics(returns, None, config)
        # threshold=0, gains = 0.02+0.03 = 0.05, losses = 0.01+0.01 = 0.02
        assert result.omega == pytest.approx(2.5, rel=1e-4)
