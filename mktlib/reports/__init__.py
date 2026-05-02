"""Polars-native tearsheet generator — drop-in replacement for quantstats."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl

from . import _compat, _plots, _stats, _template
from ._compat import PandasConvertible, ReturnsInput
from ._types import (
    DrawdownInfo,
    MetricsResult,
    MonteCarloConfig,
    ReportConfig,
    TradeMetrics,
)
from ..rates._treasury import fetch_average_rate

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = [
    "html",
    "metrics",
    "DrawdownInfo",
    "MetricsResult",
    "MonteCarloConfig",
    "ReportConfig",
    "TradeMetrics",
    "PandasConvertible",
    "ReturnsInput",
]


def _run_monte_carlo_block(
    ret_series: pl.Series,
    mc_config: MonteCarloConfig,
    periods_per_year: int,
) -> tuple[float, float, pl.DataFrame]:
    """Run MC for the report; return (mc_var, mc_cvar, sims).

    Shared helper called from both :func:`html` and :func:`metrics` when
    ``mc_config.enabled``.  Three MC GBM batches run in series — one for
    the chart's full sims frame, two for the VaR / CVaR estimators —
    deliberately sharing a single fixed *seed* so they all produce
    statistically identical paths.  When ``mc_config.seed is None`` we
    mint one OS-derived seed up front and thread it through so the
    chart and the displayed numbers stay mutually consistent (otherwise
    they would draw three independent samples and disagree by ~1–5 bps
    of sampling noise).

    At the report-default workload (10 k × 22) each MC batch runs in
    10–15 ms under the perf path, so the full triplet adds 30–45 ms
    to the tearsheet — invisible inside the typical 250 ms render.
    """
    import random

    from ..metrics import Metric, monte_carlo_paths, simulate_metric

    dt_step = 1.0 / periods_per_year
    # Mint a seed once when caller passed None so all three MC calls
    # share sample paths.  Using ``random.Random()`` keeps the OS-time
    # entropy source intact while letting us pin a value for this run.
    effective_seed = (
        mc_config.seed
        if mc_config.seed is not None
        else random.Random().randrange(2**63)
    )
    sims = monte_carlo_paths(
        ret_series,
        horizon=mc_config.horizon,
        n_simulations=mc_config.n_simulations,
        dt=dt_step,
        innovations=mc_config.innovations,
        df=mc_config.df,
        seed=effective_seed,
    )
    common = dict(
        method="monte_carlo",
        horizon=mc_config.horizon,
        n_simulations=mc_config.n_simulations,
        dt=dt_step,
        innovations=mc_config.innovations,
        df=mc_config.df,
        seed=effective_seed,
    )
    mc_var_v = simulate_metric(
        Metric.VAR, ret_series, alpha=mc_config.alpha, **common,  # type: ignore[arg-type]
    )
    mc_cvar_v = simulate_metric(
        Metric.CVAR, ret_series, alpha=mc_config.alpha, **common,  # type: ignore[arg-type]
    )
    return mc_var_v, mc_cvar_v, sims


def html(
    returns: ReturnsInput,
    *,
    benchmark: ReturnsInput | None = None,
    trades: pl.DataFrame | None = None,
    output: str | None = None,
    title: str = "Strategy Tearsheet",
    rf: float | str = 0.0,
    periods_per_year: int = 252,
    compounded: bool = True,
    extra_metrics: dict[str, list[tuple[str, str]]] | None = None,
    extra_charts: dict[str, go.Figure] | None = None,
    template: str | Path | None = None,
    mc_config: MonteCarloConfig | None = None,
) -> str | None:
    """Generate an interactive HTML tearsheet report.

    Parameters
    ----------
    returns
        Daily returns as ``pl.Series``, ``pl.DataFrame`` (with *date* and
        *return* columns), or ``pd.Series`` with a ``DatetimeIndex``.
    benchmark
        Optional benchmark returns (same types as *returns*).
    trades
        Optional per-trade DataFrame with columns ``entry_date`` (Date),
        ``exit_date`` (Date), ``side`` (Int8), ``pnl`` (Float64), and
        ``bars_held`` (Int64).  When provided, per-trade metrics are computed
        and the Win/Loss card values are overridden with trade-based figures.
        Two extra metric cards (Trade Stats, Trade Risk-Adjusted) and two
        extra charts (PnL distribution, PnL over time) are added.
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

    config = ReportConfig(
        rf=rf_resolved,
        periods_per_year=periods_per_year,
        compounded=compounded,
        title=title,
    )

    # Compute metrics
    result = _stats.compute_metrics(ret_df, bench_df, config)

    # Per-trade metrics (when trades provided)
    trade_met: TradeMetrics | None = None
    if trades is not None and not trades.is_empty():
        trade_met = _stats.compute_trade_metrics(trades)
        # Patch trade_metrics into result (frozen dataclass — rebuild)
        import dataclasses
        result = dataclasses.replace(result, trade_metrics=trade_met)

    # Build charts
    ret_series = ret_df["return"]
    dates_list = ret_df["date"].to_list()

    # Monte Carlo block (opt-in).  Runs three MC batches under one
    # shared seed so the chart frame and the VaR / CVaR numbers all
    # come from the same sample paths.  The full sims frame feeds the
    # path chart further down.
    mc_sims: pl.DataFrame | None = None
    if mc_config is not None and mc_config.enabled:
        import dataclasses
        mc_var_v, mc_cvar_v, mc_sims = _run_monte_carlo_block(
            ret_series, mc_config, periods_per_year
        )
        result = dataclasses.replace(
            result, mc_var=mc_var_v, mc_cvar=mc_cvar_v
        )

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
        ret_series, ppy=periods_per_year
    ).to_list()
    r_vol = _stats.rolling_volatility(
        ret_series, ppy=periods_per_year
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

    # MC paths chart — anchors the simulation at the last historical equity
    # value so the user sees the forecast as a natural continuation.
    if mc_sims is not None and mc_config is not None:
        from ..scheduling import get_calendar

        last_date = dates_list[-1]
        last_value = float(cum_ret[-1]) + 1.0  # cumulative_returns is 0-based
        cal = get_calendar(mc_config.exchange)
        # session_offset raises if anchor is not a session — fall back to
        # the previous session in that case.
        try:
            forward_dates = [
                cal.session_offset(last_date, i)
                for i in range(1, mc_config.horizon + 1)
            ]
        except Exception:
            anchor = cal.date_to_session(last_date, "previous")
            forward_dates = [
                cal.session_offset(anchor, i)
                for i in range(1, mc_config.horizon + 1)
            ]
        # ~3 trading months of historical context, anchored at last_value
        lookback = min(60, len(dates_list))
        hist_dates = dates_list[-lookback:]
        hist_equity = [(c + 1.0) for c in cum_ret[-lookback:]]
        charts["monte_carlo_paths"] = _plots.monte_carlo_paths_chart(
            historical_dates=hist_dates,
            historical_equity=hist_equity,
            forward_dates=forward_dates,
            sims_frame=mc_sims,
            last_value=last_value,
            n_paths_displayed=mc_config.n_paths_displayed,
            alpha=mc_config.alpha,
        )

    # Trade charts and extra metric cards (when trades provided)
    merged_extra_metrics: dict[str, list[tuple[str, str]]] = dict(extra_metrics or {})
    if trade_met is not None:
        pnl_list = trades["pnl"].to_list()  # type: ignore[union-attr]
        trade_dates = trades["entry_date"].to_list()  # type: ignore[union-attr]
        charts["trade_distribution"] = _plots.trade_pnl_distribution_chart(pnl_list)
        charts["trade_scatter"] = _plots.trade_pnl_scatter_chart(trade_dates, pnl_list)

        def _pct(v: float) -> str:
            return f"{v * 100:.2f}%"

        def _ratio(v: float) -> str:
            if abs(v) == float("inf"):
                return "∞" if v > 0 else "-∞"
            return f"{v:.3f}"

        merged_extra_metrics["Trade Stats"] = [
            ("Avg Winner", _pct(trade_met.avg_winner)),
            ("Avg Loser", _pct(trade_met.avg_loser)),
            ("Largest Winner", _pct(trade_met.largest_winner)),
            ("Largest Loser", _pct(trade_met.largest_loser)),
            ("Max Consecutive Wins", str(trade_met.max_consecutive_wins)),
            ("Max Consecutive Losses", str(trade_met.max_consecutive_losses)),
        ]
        merged_extra_metrics["Trade Risk-Adjusted"] = [
            ("Trade Sharpe", _ratio(trade_met.trade_sharpe)),
            ("Trade Sortino", _ratio(trade_met.trade_sortino)),
            ("Trades/Year", f"{trade_met.trades_per_year:.1f}"),
        ]

    # Convert extra plotly figures to HTML divs
    extra_chart_divs: dict[str, str] = {}
    if extra_charts:
        for name, fig in extra_charts.items():
            extra_chart_divs[name] = _plots._to_div(fig)

    # Render
    start_date = str(ret_df["date"].min())
    end_date = str(ret_df["date"].max())
    html_str = _template.render(
        result,
        charts,
        title,
        start_date,
        end_date,
        len(ret_df),
        extra_metrics=merged_extra_metrics or None,
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
    trades: pl.DataFrame | None = None,
    rf: float | str = 0.0,
    periods_per_year: int = 252,
    compounded: bool = True,
    mc_config: MonteCarloConfig | None = None,
) -> MetricsResult:
    """Compute performance metrics without generating an HTML report.

    Parameters
    ----------
    rf
        Risk-free rate (annualised).  Pass ``"auto"`` to fetch the 3-month
        T-bill average for the returns date range.
    trades
        Optional per-trade DataFrame (same schema as ``html()``).  When
        provided, ``MetricsResult.trade_metrics`` is populated.
    mc_config
        Optional :class:`MonteCarloConfig`.  When ``mc_config.enabled``,
        populates ``MetricsResult.mc_var`` and ``mc_cvar`` with simulation-
        based forward-looking risk numbers; otherwise leaves them ``None``.
    """
    ret_df = _compat.coerce_returns(returns)
    bench_df = _compat.coerce_benchmark(benchmark)

    rf_resolved: float = rf if isinstance(rf, (int, float)) else 0.0
    if rf == "auto":
        start = cast(date, ret_df["date"].min())
        end = cast(date, ret_df["date"].max())
        rf_resolved = fetch_average_rate(start, end)

    config = ReportConfig(
        rf=rf_resolved,
        periods_per_year=periods_per_year,
        compounded=compounded,
    )
    result = _stats.compute_metrics(ret_df, bench_df, config)

    if trades is not None and not trades.is_empty():
        import dataclasses
        trade_met = _stats.compute_trade_metrics(trades)
        result = dataclasses.replace(result, trade_metrics=trade_met)

    if mc_config is not None and mc_config.enabled:
        import dataclasses
        ret_series = ret_df["return"]
        mc_var_v, mc_cvar_v, _ = _run_monte_carlo_block(
            ret_series, mc_config, periods_per_year
        )
        result = dataclasses.replace(
            result, mc_var=mc_var_v, mc_cvar=mc_cvar_v
        )

    return result
