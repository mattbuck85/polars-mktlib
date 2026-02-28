"""Polars-native tearsheet generator — drop-in replacement for quantstats."""
from __future__ import annotations

from ._compat import PandasConvertible, ReturnsInput
from ._types import DrawdownInfo, MetricsResult, ReportConfig

__all__ = [
    "html", "metrics",
    "DrawdownInfo", "MetricsResult", "ReportConfig",
    "PandasConvertible", "ReturnsInput",
]


def html(
    returns: ReturnsInput,
    *,
    benchmark: ReturnsInput | None = None,
    output: str | None = None,
    title: str = "Strategy Tearsheet",
    rf: float | str = 0.0,
    periods_per_year: int = 252,
    compounded: bool = True,
) -> str | None:
    """Generate an interactive HTML tearsheet report.

    Parameters
    ----------
    returns
        Daily returns as ``pl.Series``, ``pl.DataFrame`` (with *date* and
        *return* columns), or ``pd.Series`` with a ``DatetimeIndex``.
    benchmark
        Optional benchmark returns (same types as *returns*).
    output
        File path to write the HTML to.  When *None*, returns the HTML string.
    title
        Report title shown in the header.
    rf
        Risk-free rate (annualised, e.g. ``0.05`` for 5 %).  Pass ``"auto"``
        to fetch the 3-month T-bill average for the returns date range.
    periods_per_year
        Trading days per year (default 252).
    compounded
        Whether to compute compounded returns (default *True*).

    Returns
    -------
    str | None
        The HTML string when *output* is *None*, otherwise *None*.
    """
    from . import _compat, _plots, _stats, _template

    # Coerce inputs
    ret_df = _compat.coerce_returns(returns)
    bench_df = _compat.coerce_benchmark(benchmark)

    rf_resolved: float = rf if isinstance(rf, (int, float)) else 0.0
    if rf == "auto":
        from datetime import date as _date

        from ..rates._treasury import fetch_average_rate

        start = ret_df["date"].min()
        end = ret_df["date"].max()
        assert isinstance(start, _date) and isinstance(end, _date)
        rf_resolved = fetch_average_rate(start, end)

    config = ReportConfig(rf=rf_resolved, periods_per_year=periods_per_year, compounded=compounded, title=title)

    # Compute metrics
    result = _stats.compute_metrics(ret_df, bench_df, config)

    # Build charts
    ret_series = ret_df["return"]
    dates_list = ret_df["date"].to_list()

    cum_ret = _stats.cumulative_returns(ret_series, compounded).to_list()
    bench_dates = bench_cum = None
    if bench_df is not None:
        bench_dates = bench_df["date"].to_list()
        bench_cum = _stats.cumulative_returns(bench_df["return"], compounded).to_list()

    dd = _stats.drawdown_series(ret_series, compounded).to_list()
    monthly = _stats.monthly_returns(ret_df, compounded)
    yearly = _stats.yearly_returns(ret_df, compounded)
    r_sharpe = _stats.rolling_sharpe(ret_series, ppy=periods_per_year).to_list()
    r_vol = _stats.rolling_volatility(ret_series, ppy=periods_per_year).to_list()

    charts = {
        "cumulative": _plots.cumulative_returns_chart(dates_list, cum_ret, bench_dates, bench_cum),
        "drawdown": _plots.drawdown_chart(dates_list, dd),
        "yearly_bar": _plots.yearly_returns_chart(yearly["year"].to_list(), yearly["yearly_return"].to_list()),
        "monthly_heatmap": _plots.monthly_heatmap_chart(
            years=monthly["year"].to_list(),
            months=monthly["month"].to_list(),
            returns=monthly["monthly_return"].to_list(),
        ),
        "rolling_sharpe": _plots.rolling_sharpe_chart(dates_list, r_sharpe),
        "rolling_vol": _plots.rolling_volatility_chart(dates_list, r_vol),
        "daily_scatter": _plots.daily_returns_scatter(dates_list, ret_series.to_list()),
        "distribution": _plots.returns_distribution_chart(ret_series.to_list()),
    }

    # Render
    start_date = str(ret_df["date"].min())
    end_date = str(ret_df["date"].max())
    html_str = _template.render(result, charts, title, start_date, end_date, len(ret_df))

    if output is not None:
        import pathlib
        pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(output).write_text(html_str, encoding="utf-8")
        return None
    return html_str


def metrics(
    returns: ReturnsInput,
    *,
    benchmark: ReturnsInput | None = None,
    rf: float | str = 0.0,
    periods_per_year: int = 252,
    compounded: bool = True,
) -> MetricsResult:
    """Compute performance metrics without generating an HTML report.

    Parameters
    ----------
    rf
        Risk-free rate (annualised).  Pass ``"auto"`` to fetch the 3-month
        T-bill average for the returns date range.
    """
    from . import _compat, _stats

    ret_df = _compat.coerce_returns(returns)
    bench_df = _compat.coerce_benchmark(benchmark)

    rf_resolved: float = rf if isinstance(rf, (int, float)) else 0.0
    if rf == "auto":
        from datetime import date as _date

        from ..rates._treasury import fetch_average_rate

        start = ret_df["date"].min()
        end = ret_df["date"].max()
        assert isinstance(start, _date) and isinstance(end, _date)
        rf_resolved = fetch_average_rate(start, end)

    config = ReportConfig(rf=rf_resolved, periods_per_year=periods_per_year, compounded=compounded)
    return _stats.compute_metrics(ret_df, bench_df, config)
