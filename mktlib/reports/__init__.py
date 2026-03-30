"""Polars-native tearsheet generator — drop-in replacement for quantstats."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl

from . import _compat, _plots, _stats, _template
from ._compat import PandasConvertible, ReturnsInput
from ._types import DrawdownInfo, MetricsResult, ReportConfig
from ..rates._treasury import fetch_average_rate


def _infer_ppy(ret_df: pl.DataFrame) -> int:
    """Infer periods-per-year from the returns DataFrame using row count bucketing.

    Buckets:
      - 0–52 rows    → weekly   (52)
      - 53–400 rows   → daily    (252)
      - 401+ rows     → minutely (252 × 390 = 98_280)

    The daily upper bound (400) provides leeway for non-US calendars
    and datasets that include some non-trading days.

    Only called when the caller passes ``periods_per_year=None``.
    """
    if ret_df.is_empty():
        return 252
    n = ret_df.height
    if n <= 52:
        return 52
    if n <= 400:
        return 252
    return 252 * 390

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = [
    "html",
    "metrics",
    "DrawdownInfo",
    "MetricsResult",
    "ReportConfig",
    "PandasConvertible",
    "ReturnsInput",
]


def html(
    returns: ReturnsInput,
    *,
    benchmark: ReturnsInput | None = None,
    output: str | None = None,
    title: str = "Strategy Tearsheet",
    rf: float | str = 0.0,
    periods_per_year: int | None = 252,
    compounded: bool = True,
    extra_metrics: dict[str, list[tuple[str, str]]] | None = None,
    extra_charts: dict[str, go.Figure] | None = None,
    template: str | Path | None = None,
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
        Periods per year for annualisation.  Default ``252`` (daily).  Pass
        ``None`` to auto-detect from row count (weekly/daily/minutely bucketing).
    compounded
        Whether to compute compounded returns (default *True*).
    extra_metrics
        Additional metric cards: ``{card_title: [(label, value), ...]}``.
        Appended to the built-in metrics grid.
    extra_charts
        Additional charts: ``{name: plotly.graph_objects.Figure}``.
        Converted to HTML divs and rendered after the built-in charts.
    template
        Custom Jinja2 template.  ``Path`` loads from file, ``str`` is treated
        as inline Jinja2 source, ``None`` uses the built-in template.

    Returns
    -------
    str | None
        The HTML string when *output* is *None*, otherwise *None*.
    """
    # Coerce inputs
    ret_df = _compat.coerce_returns(returns)
    bench_df = _compat.coerce_benchmark(benchmark)

    rf_resolved: float = rf if isinstance(rf, (int, float)) else 0.0
    if rf == "auto":
        start = cast(date, ret_df["date"].min())
        end = cast(date, ret_df["date"].max())
        rf_resolved = fetch_average_rate(start, end)

    effective_ppy = _infer_ppy(ret_df) if periods_per_year is None else periods_per_year

    config = ReportConfig(
        rf=rf_resolved,
        periods_per_year=effective_ppy,
        compounded=compounded,
        title=title,
    )

    # Compute metrics
    result = _stats.compute_metrics(ret_df, bench_df, config)

    # Build charts
    ret_series = ret_df["return"]
    dates_list = ret_df["date"].to_list()

    cum_ret = _stats.cumulative_returns(ret_series, compounded).to_list()
    bench_dates = bench_cum = None
    if bench_df is not None:
        bench_dates = bench_df["date"].to_list()
        bench_cum = _stats.cumulative_returns(
            bench_df["return"], compounded
        ).to_list()

    dd = _stats.drawdown_series(ret_series, compounded).to_list()
    monthly = _stats.monthly_returns(ret_df, compounded)
    yearly = _stats.yearly_returns(ret_df, compounded)
    r_sharpe = _stats.rolling_sharpe(
        ret_series, ppy=effective_ppy
    ).to_list()
    r_vol = _stats.rolling_volatility(
        ret_series, ppy=effective_ppy
    ).to_list()

    charts = {
        "cumulative": _plots.cumulative_returns_chart(
            dates_list, cum_ret, bench_dates, bench_cum
        ),
        "drawdown": _plots.drawdown_chart(dates_list, dd),
        "yearly_bar": _plots.yearly_returns_chart(
            yearly["year"].to_list(), yearly["yearly_return"].to_list()
        ),
        "monthly_heatmap": _plots.monthly_heatmap_chart(
            years=monthly["year"].to_list(),
            months=monthly["month"].to_list(),
            returns=monthly["monthly_return"].to_list(),
        ),
        "rolling_sharpe": _plots.rolling_sharpe_chart(dates_list, r_sharpe),
        "rolling_vol": _plots.rolling_volatility_chart(dates_list, r_vol),
        "daily_scatter": _plots.daily_returns_scatter(
            dates_list, ret_series.to_list()
        ),
        "distribution": _plots.returns_distribution_chart(
            ret_series.to_list()
        ),
    }

    # Convert extra plotly figures to HTML divs
    extra_chart_divs: dict[str, str] = {}
    if extra_charts:
        for name, fig in extra_charts.items():
            extra_chart_divs[name] = _plots._to_div(fig)

    # Render
    start_date = str(ret_df["date"].min())
    end_date = str(ret_df["date"].max())
    trading_days = ret_df["date"].dt.date().n_unique()
    html_str = _template.render(
        result,
        charts,
        title,
        start_date,
        end_date,
        trading_days,
        extra_metrics=extra_metrics,
        extra_charts=extra_chart_divs,
        template_override=template,
    )

    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(html_str, encoding="utf-8")
        return None
    return html_str


def metrics(
    returns: ReturnsInput,
    *,
    benchmark: ReturnsInput | None = None,
    rf: float | str = 0.0,
    periods_per_year: int | None = 252,
    compounded: bool = True,
) -> MetricsResult:
    """Compute performance metrics without generating an HTML report.

    Parameters
    ----------
    rf
        Risk-free rate (annualised).  Pass ``"auto"`` to fetch the 3-month
        T-bill average for the returns date range.
    periods_per_year
        Periods per year for annualisation.  Default ``252`` (daily).  Pass
        ``None`` to auto-detect from row count.
    """
    ret_df = _compat.coerce_returns(returns)
    bench_df = _compat.coerce_benchmark(benchmark)

    rf_resolved: float = rf if isinstance(rf, (int, float)) else 0.0
    if rf == "auto":
        start = cast(date, ret_df["date"].min())
        end = cast(date, ret_df["date"].max())
        rf_resolved = fetch_average_rate(start, end)

    effective_ppy = _infer_ppy(ret_df) if periods_per_year is None else periods_per_year

    config = ReportConfig(
        rf=rf_resolved,
        periods_per_year=effective_ppy,
        compounded=compounded,
    )
    return _stats.compute_metrics(ret_df, bench_df, config)
