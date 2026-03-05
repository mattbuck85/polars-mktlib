from __future__ import annotations

import datetime as dt

import pytest

plotly = pytest.importorskip("plotly")

from mktlib.reports import _plots


@pytest.fixture
def sample_dates() -> list[dt.date]:
    return [dt.date(2024, 1, i) for i in range(2, 12)]


@pytest.fixture
def sample_returns() -> list[float]:
    return [
        0.01,
        -0.02,
        0.015,
        -0.005,
        0.008,
        -0.012,
        0.003,
        0.02,
        -0.01,
        0.005,
    ]


class TestCumulativeReturnsChart:
    def test_returns_html_div(self, sample_dates: list, sample_returns: list):
        html = _plots.cumulative_returns_chart(sample_dates, sample_returns)
        assert "<div" in html
        assert (
            "plotly" in html.lower()
            or "js-plotly" in html.lower()
            or "data" in html
        )

    def test_with_benchmark(self, sample_dates: list, sample_returns: list):
        bench = [r * 0.5 for r in sample_returns]
        html = _plots.cumulative_returns_chart(
            sample_dates, sample_returns, sample_dates, bench
        )
        assert "<div" in html


class TestDrawdownChart:
    def test_returns_html_div(self, sample_dates: list):
        drawdowns = [
            -0.01,
            -0.02,
            -0.015,
            -0.005,
            0.0,
            -0.01,
            -0.03,
            -0.02,
            -0.01,
            0.0,
        ]
        html = _plots.drawdown_chart(sample_dates, drawdowns)
        assert "<div" in html


class TestYearlyReturnsChart:
    def test_returns_html_div(self):
        html = _plots.yearly_returns_chart(
            [2022, 2023, 2024], [0.15, -0.05, 0.20]
        )
        assert "<div" in html


class TestMonthlyHeatmapChart:
    def test_returns_html_div(self):
        years = [2024] * 3
        months = [1, 2, 3]
        returns = [0.05, -0.02, 0.03]
        html = _plots.monthly_heatmap_chart(years, months, returns)
        assert "<div" in html

    def test_multiple_years(self):
        years = [2023, 2023, 2024, 2024]
        months = [11, 12, 1, 2]
        returns = [0.03, -0.01, 0.02, 0.04]
        html = _plots.monthly_heatmap_chart(years, months, returns)
        assert "<div" in html


class TestRollingCharts:
    def test_rolling_sharpe(self, sample_dates: list, sample_returns: list):
        html = _plots.rolling_sharpe_chart(sample_dates, sample_returns)
        assert "<div" in html

    def test_rolling_volatility(
        self, sample_dates: list, sample_returns: list
    ):
        html = _plots.rolling_volatility_chart(sample_dates, sample_returns)
        assert "<div" in html


class TestDailyReturnsScatter:
    def test_returns_html_div(self, sample_dates: list, sample_returns: list):
        html = _plots.daily_returns_scatter(sample_dates, sample_returns)
        assert "<div" in html


class TestReturnsDistribution:
    def test_returns_html_div(self, sample_returns: list):
        html = _plots.returns_distribution_chart(sample_returns)
        assert "<div" in html
