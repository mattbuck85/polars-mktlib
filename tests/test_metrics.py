from __future__ import annotations

import datetime as dt
import math

import polars as pl
import pytest

from mktlib.metrics import (
    Metric,
    avg_drawdown,
    cagr,
    calculate_metric,
    cumulative_return,
    annualized_volatility,
    cvar,
    drawdown_series,
    kelly_criterion,
    longest_drawdown_days,
    omega,
    payoff_ratio,
    profit_factor,
    sharpe,
    sortino,
    var,
    win_rate,
)


@pytest.fixture
def daily_ret() -> pl.Series:
    """252 days of returns with slight variation."""
    return pl.Series("r", [0.001 + 0.002 * math.sin(i * 0.5) for i in range(252)])


@pytest.fixture
def dates_series() -> pl.Series:
    """Business-day dates for 252 bars."""
    dates: list[dt.date] = []
    current = dt.date(2024, 1, 2)
    for _ in range(252):
        while current.weekday() >= 5:
            current += dt.timedelta(days=1)
        dates.append(current)
        current += dt.timedelta(days=1)
    return pl.Series("date", dates)


@pytest.fixture
def empty_ret() -> pl.Series:
    return pl.Series("r", [], dtype=pl.Float64)


# ---------------------------------------------------------------------------
# Individual function tests
# ---------------------------------------------------------------------------


class TestCumulativeReturn:
    def test_compounded(self, daily_ret: pl.Series) -> None:
        result = cumulative_return(daily_ret, compounded=True)
        assert isinstance(result, float)
        assert result > 0

    def test_simple(self, daily_ret: pl.Series) -> None:
        result = cumulative_return(daily_ret, compounded=False)
        assert isinstance(result, float)
        assert result == pytest.approx(float(daily_ret.sum()), abs=1e-10)

    def test_empty(self, empty_ret: pl.Series) -> None:
        assert cumulative_return(empty_ret) == 0.0


class TestCAGR:
    def test_one_year(self, daily_ret: pl.Series) -> None:
        result = cagr(daily_ret)
        assert isinstance(result, float)
        # 252 bars / 252 ppy = 1 year → CAGR ≈ cumulative
        cum = cumulative_return(daily_ret)
        assert result == pytest.approx(cum, rel=1e-4)

    def test_empty(self, empty_ret: pl.Series) -> None:
        assert cagr(empty_ret) == 0.0


class TestAnnualizedVolatility:
    def test_positive(self, daily_ret: pl.Series) -> None:
        result = annualized_volatility(daily_ret)
        assert isinstance(result, float)
        assert result > 0

    def test_single_bar(self) -> None:
        assert annualized_volatility(pl.Series("r", [0.01])) == 0.0


class TestDrawdownSeries:
    def test_basic(self) -> None:
        ret = pl.Series("r", [0.10, -0.20])
        dd = drawdown_series(ret, compounded=True)
        assert dd[0] == pytest.approx(0.0)
        assert dd[1] == pytest.approx(0.88 / 1.1 - 1, abs=1e-6)

    def test_no_drawdown(self) -> None:
        ret = pl.Series("r", [0.01, 0.02, 0.01])
        dd = drawdown_series(ret)
        assert dd.min() == pytest.approx(0.0)

    def test_empty(self, empty_ret: pl.Series) -> None:
        dd = drawdown_series(empty_ret)
        assert len(dd) == 0


class TestAvgDrawdown:
    def test_with_drawdown(self) -> None:
        dd = pl.Series("dd", [0.0, -0.05, -0.10, 0.0])
        result = avg_drawdown(dd)
        assert isinstance(result, float)
        assert result == pytest.approx((-0.05 + -0.10) / 2, abs=1e-10)

    def test_no_drawdown(self) -> None:
        dd = pl.Series("dd", [0.0, 0.0, 0.0])
        assert avg_drawdown(dd) == 0.0


class TestLongestDrawdownDays:
    def test_basic(self) -> None:
        dd = pl.Series("dd", [0.0, -0.05, -0.10, -0.03, 0.0])
        dates = pl.Series("date", [
            dt.date(2024, 1, 1),
            dt.date(2024, 1, 2),
            dt.date(2024, 1, 3),
            dt.date(2024, 1, 4),
            dt.date(2024, 1, 5),
        ])
        result = longest_drawdown_days(dd, dates)
        assert isinstance(result, float)
        assert result == 2.0  # Jan 2 to Jan 4

    def test_empty(self) -> None:
        dd = pl.Series("dd", [], dtype=pl.Float64)
        dates = pl.Series("date", [], dtype=pl.Date)
        assert longest_drawdown_days(dd, dates) == 0.0


class TestSharpe:
    def test_positive(self, daily_ret: pl.Series) -> None:
        result = sharpe(daily_ret)
        assert isinstance(result, float)
        assert result > 0

    def test_empty(self, empty_ret: pl.Series) -> None:
        assert sharpe(empty_ret) == 0.0


class TestSortino:
    def test_positive(self, daily_ret: pl.Series) -> None:
        result = sortino(daily_ret)
        assert isinstance(result, float)
        assert result > 0


class TestOmega:
    def test_basic(self) -> None:
        ret = pl.Series("r", [0.02, -0.01, 0.03, -0.01])
        result = omega(ret, rf=0.0)
        # gains = 0.02+0.03 = 0.05, losses = 0.01+0.01 = 0.02
        assert result == pytest.approx(2.5, rel=1e-4)

    def test_empty(self, empty_ret: pl.Series) -> None:
        assert omega(empty_ret) == 0.0


class TestVaR:
    def test_negative(self) -> None:
        ret = pl.Series("r", [-0.05, -0.03, 0.01, 0.02, 0.015, -0.04, 0.005, -0.02, 0.01, -0.01])
        result = var(ret)
        assert isinstance(result, float)
        assert result < 0

    def test_empty(self, empty_ret: pl.Series) -> None:
        assert var(empty_ret) == 0.0


class TestCVaR:
    def test_lte_var(self) -> None:
        ret = pl.Series("r", [-0.05, -0.03, 0.01, 0.02, 0.015, -0.04, 0.005, -0.02, 0.01, -0.01])
        assert cvar(ret) <= var(ret)

    def test_empty(self, empty_ret: pl.Series) -> None:
        assert cvar(empty_ret) == 0.0


class TestWinRate:
    def test_basic(self) -> None:
        ret = pl.Series("r", [0.01, -0.01, 0.02, -0.02])
        assert win_rate(ret) == pytest.approx(0.5)

    def test_empty(self, empty_ret: pl.Series) -> None:
        assert win_rate(empty_ret) == 0.0


class TestPayoffRatio:
    def test_basic(self) -> None:
        ret = pl.Series("r", [0.02, -0.01, 0.03, -0.01])
        # avg win = 0.025, avg loss = 0.01 → 2.5
        assert payoff_ratio(ret) == pytest.approx(2.5, rel=1e-4)

    def test_no_losses(self) -> None:
        ret = pl.Series("r", [0.01, 0.02])
        assert payoff_ratio(ret) == 0.0


class TestProfitFactor:
    def test_basic(self) -> None:
        ret = pl.Series("r", [0.02, -0.01, 0.03, -0.01])
        # gains=0.05, losses=0.02 → 2.5
        assert profit_factor(ret) == pytest.approx(2.5, rel=1e-4)

    def test_no_losses(self) -> None:
        ret = pl.Series("r", [0.01, 0.02])
        assert profit_factor(ret) == float("inf")


class TestKellyCriterion:
    def test_basic(self) -> None:
        ret = pl.Series("r", [0.02, -0.01, 0.03, -0.01])
        # wr=0.5, payoff=2.5 → 0.5 - 0.5/2.5 = 0.3
        assert kelly_criterion(ret) == pytest.approx(0.3, rel=1e-4)


# ---------------------------------------------------------------------------
# calculate_metric dispatcher
# ---------------------------------------------------------------------------


class TestCalculateMetric:
    def test_all_metrics_return_float(self, daily_ret: pl.Series, dates_series: pl.Series) -> None:
        """Every Metric member returns a float."""
        dd = drawdown_series(daily_ret)
        for m in Metric:
            result = calculate_metric(
                m, daily_ret, dd=dd, dates=dates_series, ppy=252, rf=0.0, alpha=0.05,
            )
            assert isinstance(result, float), f"{m.name} returned {type(result)}"

    def test_drawdown_reuse(self, daily_ret: pl.Series) -> None:
        """Passing dd= produces identical results to auto-compute."""
        dd = drawdown_series(daily_ret)
        for m in (Metric.MAX_DRAWDOWN, Metric.AVG_DRAWDOWN, Metric.CALMAR, Metric.ROMAD):
            with_dd = calculate_metric(m, daily_ret, dd=dd)
            without_dd = calculate_metric(m, daily_ret)
            assert with_dd == pytest.approx(without_dd, abs=1e-12), f"{m.name} mismatch"

    def test_longest_drawdown_days_requires_dates(self, daily_ret: pl.Series) -> None:
        with pytest.raises(ValueError, match="dates="):
            calculate_metric(Metric.LONGEST_DRAWDOWN_DAYS, daily_ret)

    def test_empty_series(self, empty_ret: pl.Series) -> None:
        """All metrics return 0.0 for empty series."""
        for m in Metric:
            if m == Metric.LONGEST_DRAWDOWN_DAYS:
                result = calculate_metric(
                    m, empty_ret, dates=pl.Series("d", [], dtype=pl.Date),
                )
            else:
                result = calculate_metric(m, empty_ret)
            assert result == 0.0, f"{m.name} returned {result} for empty series"


class TestLongestDrawdownDaysFloat:
    """Verify longest_drawdown_days returns float, not int."""

    def test_return_type(self) -> None:
        ret = pl.Series("r", [0.05, 0.05, -0.08, -0.08, 0.03, 0.03])
        dates = pl.Series("date", [
            dt.date(2024, 1, i) for i in range(1, 7)
        ])
        dd = drawdown_series(ret)
        result = calculate_metric(
            Metric.LONGEST_DRAWDOWN_DAYS, ret, dd=dd, dates=dates,
        )
        assert isinstance(result, float)
