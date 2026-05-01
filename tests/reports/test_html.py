from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import polars as pl
import pytest

plotly = pytest.importorskip("plotly")
jinja2 = pytest.importorskip("jinja2")

from mktlib.reports import html, metrics


def _make_returns(
    n: int = 100, start: dt.date = dt.date(2024, 1, 2)
) -> pl.DataFrame:
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


class TestExtensibility:
    def test_extra_metrics_appear_in_html(self):
        returns = _make_returns()
        result = html(returns, extra_metrics={"Custom": [("Foo", "42")]})
        assert result is not None
        assert "Custom" in result
        assert "Foo" in result
        assert "42" in result

    def test_extra_charts_rendered(self):
        go = pytest.importorskip("plotly.graph_objects")
        returns = _make_returns()
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
        result = html(returns, extra_charts={"pnl": fig})
        assert result is not None
        assert "plotly" in result.lower() or "<div" in result

    def test_custom_template_string(self):
        returns = _make_returns()
        tpl = "<html>{{ title }} {{ trading_days }}</html>"
        result = html(returns, template=tpl, title="Custom Title")
        assert result is not None
        assert "Custom Title" in result
        assert "<!DOCTYPE html>" not in result  # no built-in layout

    def test_custom_template_path(self, tmp_path: Path):
        returns = _make_returns()
        tpl_file = tmp_path / "my.j2"
        tpl_file.write_text("<html>{{ title }} {{ start_date }}</html>")
        result = html(returns, template=tpl_file, title="Path Title")
        assert result is not None
        assert "Path Title" in result
        assert "2024" in result

    def test_custom_template_receives_all_vars(self):
        go = pytest.importorskip("plotly.graph_objects")
        returns = _make_returns()
        fig = go.Figure()
        tpl = (
            "groups={{ metrics_groups | length }}"
            " charts={{ charts | length }}"
            " extra={{ extra_charts | length }}"
        )
        result = html(
            returns,
            template=tpl,
            extra_metrics={"X": [("a", "1")]},
            extra_charts={"c": fig},
        )
        assert result is not None
        assert "groups=7" in result  # 6 built-in + 1 extra
        assert "charts=8" in result
        assert "extra=1" in result

    def test_all_three_combined(self):
        go = pytest.importorskip("plotly.graph_objects")
        returns = _make_returns()
        fig = go.Figure(data=go.Scatter(x=[1], y=[1]))
        result = html(
            returns,
            extra_metrics={"Execution": [("Trades", "142")]},
            extra_charts={"pnl": fig},
        )
        assert result is not None
        assert "Execution" in result
        assert "Trades" in result
        assert "142" in result

    def test_defaults_unchanged(self):
        """html() with no new params produces the same output as before."""
        returns = _make_returns(50)
        result = html(returns, title="Regression")
        assert result is not None
        assert "<!DOCTYPE html>" in result
        assert "Regression" in result
        for section in ["Returns", "Ratios", "Risk", "Win/Loss", "Benchmark"]:
            assert section in result


class TestPandasIntegration:
    def test_html_with_pandas_series(self):
        pd = pytest.importorskip("pandas")
        idx = pd.DatetimeIndex(
            [
                dt.datetime(2024, 1, i)
                for i in range(2, 32)
                if dt.date(2024, 1, i).weekday() < 5
            ]
        )
        values = [0.001 * (i % 5 - 2) for i in range(len(idx))]
        s = pd.Series(values, index=idx, name="returns")
        result = html(s, title="Pandas Test")
        assert result is not None
        assert "Pandas Test" in result


class TestMonteCarloIntegration:
    """`mc_config=` opt-in: Forward-Looking Risk card + simulation paths chart."""

    def test_disabled_default_unchanged(self):
        """No mc_config → output identical to pre-feature behavior."""
        from mktlib.metrics import _mc_cache_clear

        _mc_cache_clear()
        df = _make_returns(n=80)
        baseline = html(df, title="No MC")
        assert baseline is not None
        # Forward-Looking card and MC chart absent
        assert "Forward-Looking Risk" not in baseline
        assert "MC VaR" not in baseline

    def test_disabled_explicit_no_mc_artifacts(self):
        """mc_config=MonteCarloConfig(enabled=False) introduces no MC card or chart.

        (Byte-identity is unattainable because Plotly embeds random div IDs;
        the contract is that the MC code path is fully gated.)
        """
        from mktlib.reports import MonteCarloConfig
        from mktlib.metrics import _mc_cache_clear

        _mc_cache_clear()
        df = _make_returns(n=80)
        out = html(df, mc_config=MonteCarloConfig(enabled=False))
        assert out is not None
        assert "Forward-Looking Risk" not in out
        assert "Monte Carlo Forward Paths" not in out
        assert "MC VaR" not in out

    def test_enabled_renders_card_and_chart(self):
        from mktlib.reports import MonteCarloConfig
        from mktlib.metrics import _mc_cache_clear

        _mc_cache_clear()
        df = _make_returns(n=120)
        out = html(
            df, title="With MC",
            mc_config=MonteCarloConfig(
                enabled=True, seed=42, n_simulations=1_000, horizon=21,
            ),
        )
        assert out is not None
        assert "Forward-Looking Risk" in out
        assert "MC VaR" in out and "MC CVaR" in out
        # Chart heading from monte_carlo_paths_chart
        assert "Monte Carlo Forward Paths" in out

    def test_metrics_function_populates_mc_fields(self):
        from mktlib.reports import MonteCarloConfig
        from mktlib.metrics import _mc_cache_clear

        _mc_cache_clear()
        df = _make_returns(n=120)
        result = metrics(
            df,
            mc_config=MonteCarloConfig(
                enabled=True, seed=42, n_simulations=1_000, horizon=21,
            ),
        )
        assert result.mc_var is not None
        assert result.mc_cvar is not None
        assert result.mc_cvar <= result.mc_var

    def test_metrics_default_mc_fields_none(self):
        df = _make_returns(n=80)
        result = metrics(df)
        assert result.mc_var is None
        assert result.mc_cvar is None

    def test_one_batch_shared_across_var_cvar(self):
        """Cache pre-populate: one MC batch services VaR + CVaR + chart."""
        from mktlib.reports import MonteCarloConfig
        from mktlib.metrics import _mc_cache, _mc_cache_clear

        _mc_cache_clear()
        df = _make_returns(n=80)
        html(
            df,
            mc_config=MonteCarloConfig(
                enabled=True, seed=42, n_simulations=1_000, horizon=21,
            ),
        )
        assert len(_mc_cache) == 1

    def test_students_t_innovations_round_trip(self):
        from mktlib.data import Innovations
        from mktlib.reports import MonteCarloConfig
        from mktlib.metrics import _mc_cache_clear

        _mc_cache_clear()
        df = _make_returns(n=120)
        result = metrics(
            df,
            mc_config=MonteCarloConfig(
                enabled=True, seed=42, n_simulations=1_000, horizon=21,
                innovations=Innovations.STUDENT_T, df=5,
            ),
        )
        assert result.mc_var is not None
        assert result.mc_cvar is not None

    def test_bootstrap_innovations_round_trip(self):
        from mktlib.data import Innovations
        from mktlib.reports import MonteCarloConfig
        from mktlib.metrics import _mc_cache_clear

        _mc_cache_clear()
        df = _make_returns(n=120)
        result = metrics(
            df,
            mc_config=MonteCarloConfig(
                enabled=True, seed=42, n_simulations=1_000, horizon=21,
                innovations=Innovations.BOOTSTRAP,
            ),
        )
        assert result.mc_var is not None
        assert result.mc_cvar is not None
