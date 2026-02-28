from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import polars as pl
import pytest

plotly = pytest.importorskip("plotly")
jinja2 = pytest.importorskip("jinja2")

from mktlib.reports import html, metrics


def _make_returns(n: int = 100, start: dt.date = dt.date(2024, 1, 2)) -> pl.DataFrame:
    """Generate *n* days of small positive returns."""
    import math
    dates: list[dt.date] = []
    current = start
    for _ in range(n):
        while current.weekday() >= 5:
            current += dt.timedelta(days=1)
        dates.append(current)
        current += dt.timedelta(days=1)
    # Alternating returns for realism
    values = [0.005 * math.sin(i * 0.3) + 0.001 for i in range(n)]
    return pl.DataFrame({"date": dates, "return": values})


class TestHtmlGeneration:
    def test_returns_html_string(self):
        returns = _make_returns()
        result = html(returns, title="Test Report")
        assert result is not None
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result
        assert "Test Report" in result

    def test_writes_to_file(self, tmp_path: Path):
        returns = _make_returns()
        output_path = str(tmp_path / "report.html")
        result = html(returns, output=output_path, title="File Report")
        assert result is None
        assert Path(output_path).exists()
        content = Path(output_path).read_text()
        assert "File Report" in content
        assert "<!DOCTYPE html>" in content

    def test_creates_parent_dirs(self, tmp_path: Path):
        returns = _make_returns()
        output_path = str(tmp_path / "sub" / "dir" / "report.html")
        html(returns, output=output_path)
        assert Path(output_path).exists()

    def test_with_benchmark(self):
        returns = _make_returns(100)
        bench = _make_returns(100)
        result = html(returns, benchmark=bench)
        assert result is not None
        assert "Benchmark" in result

    def test_contains_metrics_sections(self):
        returns = _make_returns()
        result = html(returns)
        assert result is not None
        for section in ["Returns", "Ratios", "Risk", "Win/Loss"]:
            assert section in result

    def test_contains_chart_divs(self):
        returns = _make_returns()
        result = html(returns)
        assert result is not None
        assert result.count("<div") > 10  # Multiple chart + metric divs

    def test_plotly_cdn_included(self):
        returns = _make_returns()
        result = html(returns)
        assert result is not None
        assert "plotly" in result
        assert "cdn.plot.ly" in result


class TestMetricsFunction:
    def test_returns_metrics_result(self):
        returns = _make_returns()
        result = metrics(returns)
        assert hasattr(result, "sharpe")
        assert hasattr(result, "cumulative_return")
        assert hasattr(result, "max_drawdown")

    def test_with_benchmark(self):
        returns = _make_returns(100)
        bench = _make_returns(100)
        result = metrics(returns, benchmark=bench)
        assert result.alpha is not None
        assert result.beta is not None

    def test_custom_rf(self):
        returns = _make_returns()
        r1 = metrics(returns, rf=0.0)
        r2 = metrics(returns, rf=0.05)
        # Different risk-free rates should produce different Sharpe
        assert r1.sharpe != r2.sharpe


class TestPandasIntegration:
    def test_html_with_pandas_series(self):
        pd = pytest.importorskip("pandas")
        idx = pd.DatetimeIndex([dt.datetime(2024, 1, i) for i in range(2, 32) if dt.date(2024, 1, i).weekday() < 5])
        values = [0.001 * (i % 5 - 2) for i in range(len(idx))]
        s = pd.Series(values, index=idx, name="returns")
        result = html(s, title="Pandas Test")
        assert result is not None
        assert "Pandas Test" in result
