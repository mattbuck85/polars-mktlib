from __future__ import annotations

import datetime as dt
import random
import warnings
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

STRATEGY_COLOR = "#2196F3"
BENCHMARK_COLOR = "#9E9E9E"
NEGATIVE_COLOR = "#EF5350"
POSITIVE_COLOR = "#66BB6A"


def _base_layout(
    title: str, height: int = 400, **kwargs: Any
) -> dict[str, Any]:
    return dict(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=50, b=40),
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        hovermode="x unified",
        **kwargs,
    )


def _to_div(fig: go.Figure) -> str:
    result: str = pio.to_html(fig, full_html=False, include_plotlyjs=False)
    return result


def cumulative_returns_chart(
    dates: list[dt.date],
    cum_ret: list[float],
    bench_dates: list[dt.date] | None = None,
    bench_cum_ret: list[float] | None = None,
) -> str:
    fig = go.Figure()
    fig = fig.add_trace(
        go.Scatter(
            x=dates,
            y=cum_ret,
            name="Strategy",
            line=dict(color=STRATEGY_COLOR, width=2),
        )
    )
    if bench_dates and bench_cum_ret:
        fig = fig.add_trace(
            go.Scatter(
                x=bench_dates,
                y=bench_cum_ret,
                name="Benchmark",
                line=dict(color=BENCHMARK_COLOR, width=1.5, dash="dot"),
            )
        )
    fig = fig.update_layout(
        _base_layout("Cumulative Returns"), yaxis_tickformat=".1%"
    )
    return _to_div(fig)


def drawdown_chart(dates: list[dt.date], drawdowns: list[float]) -> str:
    fig = go.Figure()
    fig = fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdowns,
            fill="tozeroy",
            fillcolor="rgba(239,83,80,0.3)",
            line=dict(color=NEGATIVE_COLOR, width=1),
            name="Drawdown",
        )
    )
    fig = fig.update_layout(_base_layout("Drawdown"), yaxis_tickformat=".1%")
    return _to_div(fig)


def yearly_returns_chart(years: list[int], returns: list[float]) -> str:
    colors = [POSITIVE_COLOR if r >= 0 else NEGATIVE_COLOR for r in returns]
    fig = go.Figure()
    fig = fig.add_trace(
        go.Bar(
            x=[str(y) for y in years],
            y=returns,
            marker_color=colors,
            name="Return",
        )
    )
    fig = fig.update_layout(
        _base_layout("Yearly Returns", height=350), yaxis_tickformat=".1%"
    )
    return _to_div(fig)


def monthly_heatmap_chart(
    years: list[int], months: list[int], returns: list[float]
) -> str:
    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    unique_years = sorted(set(years))

    lookup: dict[tuple[int, int], float] = {}
    for y, m, r in zip(years, months, returns):
        lookup[(y, m)] = r

    z: list[list[float | None]] = []
    text: list[list[str]] = []
    for y in unique_years:
        row = [lookup.get((y, m + 1)) for m in range(12)]
        z.append(row)
        text.append([f"{v * 100:.1f}%" if v is not None else "" for v in row])

    fig = go.Figure(
        data=go.Heatmap(
            z=z,  # type: ignore[arg-type]  # plotly stubs too narrow
            x=month_labels,  # type: ignore[arg-type]
            y=[str(y) for y in unique_years],  # type: ignore[arg-type]
            text=text,  # type: ignore[arg-type]
            texttemplate="%{text}",  # type: ignore[arg-type]
            colorscale="RdYlGn",
            zmid=0,
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{text}<extra></extra>",
        )
    )
    fig = fig.update_layout(
        _base_layout(
            "Monthly Returns", height=max(200, len(unique_years) * 35 + 100)
        )
    )
    return _to_div(fig)


def rolling_sharpe_chart(
    dates: list[dt.date], values: list[float | None]
) -> str:
    fig = go.Figure()
    fig = fig.add_trace(go.Scatter(x=dates, y=values, line=dict(color=STRATEGY_COLOR, width=1.5), name="Rolling Sharpe"))  # type: ignore[arg-type]  # stubs don't allow None in y
    fig = fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig = fig.update_layout(_base_layout("Rolling Sharpe (126d)"))
    return _to_div(fig)


def rolling_volatility_chart(
    dates: list[dt.date], values: list[float | None]
) -> str:
    fig = go.Figure()
    fig = fig.add_trace(go.Scatter(x=dates, y=values, line=dict(color="#FF9800", width=1.5), name="Rolling Volatility"))  # type: ignore[arg-type]  # stubs don't allow None in y
    fig = fig.update_layout(
        _base_layout("Rolling Volatility (126d)"), yaxis_tickformat=".1%"
    )
    return _to_div(fig)


def daily_returns_scatter(dates: list[dt.date], returns: list[float]) -> str:
    colors = [POSITIVE_COLOR if r >= 0 else NEGATIVE_COLOR for r in returns]
    fig = go.Figure()
    fig = fig.add_trace(
        go.Scatter(
            x=dates,
            y=returns,
            mode="markers",
            marker=dict(size=3, color=colors, opacity=0.6),
            name="Daily Return",
        )
    )
    fig = fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig = fig.update_layout(
        _base_layout("Daily Returns"), yaxis_tickformat=".2%"
    )
    return _to_div(fig)


def returns_distribution_chart(returns: list[float]) -> str:
    fig = go.Figure()
    fig = fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=50,
            marker_color=STRATEGY_COLOR,
            opacity=0.7,
            name="Returns",
        )
    )
    fig = fig.update_layout(
        _base_layout("Returns Distribution", height=350),
        xaxis_tickformat=".1%",
        bargap=0.05,
    )
    return _to_div(fig)


def trade_pnl_distribution_chart(pnl_list: list[float]) -> str:
    """Histogram of per-trade PnL, green for winners, red for losers."""
    winners = [v for v in pnl_list if v > 0]
    losers = [v for v in pnl_list if v < 0]
    fig = go.Figure()
    if winners:
        fig = fig.add_trace(
            go.Histogram(
                x=winners,
                nbinsx=30,
                marker_color=POSITIVE_COLOR,
                opacity=0.7,
                name="Winners",
            )
        )
    if losers:
        fig = fig.add_trace(
            go.Histogram(
                x=losers,
                nbinsx=30,
                marker_color=NEGATIVE_COLOR,
                opacity=0.7,
                name="Losers",
            )
        )
    fig = fig.update_layout(
        _base_layout("Trade PnL Distribution", height=350),
        barmode="overlay",
        bargap=0.05,
        xaxis_tickformat=".2%",
    )
    return _to_div(fig)


def trade_pnl_scatter_chart(
    dates_list: list[dt.date], pnl_list: list[float]
) -> str:
    """Scatter plot of per-trade PnL by entry date, green/red coloring."""
    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in pnl_list]
    fig = go.Figure()
    fig = fig.add_trace(
        go.Scatter(
            x=dates_list,
            y=pnl_list,
            mode="markers",
            marker=dict(size=6, color=colors, opacity=0.7),
            name="Trade PnL",
        )
    )
    fig = fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig = fig.update_layout(
        _base_layout("Trade PnL Over Time"), yaxis_tickformat=".2%"
    )
    return _to_div(fig)


_MC_PATHS_DOM_CAP = 500
"""Hard cap on individual MC paths rendered as discrete traces.

Plotly's DOM struggles past ~500 ``go.Scatter`` traces in a single
figure; rendering more silently degrades interactivity.  Callers can
still ask for more ``n_paths_displayed``; the chart helper clips
silently to keep the page responsive.
"""


def monte_carlo_paths_chart(
    historical_dates: list[dt.date],
    historical_equity: list[float],
    forward_dates: list[dt.date],
    sims_frame: Any,
    last_value: float,
    n_paths_displayed: int,
    alpha: float,
    title: str = "Monte Carlo Forward Paths",
) -> str:
    """Spaghetti + percentile-band fan chart anchored at *last_value*.

    Trace order is load-bearing for the ``fill="tonexty"`` band: the
    lower-percentile trace must precede the upper-percentile trace, and
    nothing renderable may sit between them.  Order:

    1. Historical equity line (dimmer continuation of the cumulative-returns chart).
    2. Spaghetti subset — up to ``min(n_paths_displayed, n_simulations, 500)`` paths
       at low opacity, no legend, no hover.
    3. Lower-percentile (α/2) line, transparent — the fill anchor.
    4. Upper-percentile (1−α/2) line with ``fill="tonexty"``.
    5. Median path on top.
    6. Vertical anchor line at the last historical date.

    Y values are absolute equity (= ``last_value × price``); the sims
    frame is produced by :func:`mktlib.metrics.monte_carlo_paths` with
    ``base_price=1.0`` so the rescale here is the only conversion.

    The forward-date list comes from ``mktlib.scheduling.get_calendar(...)
    .session_offset(last_date, i)`` in the caller; this helper takes it
    pre-built so the plotting code stays free of calendar concerns.
    """
    import polars as pl

    sims = sims_frame
    n_sims = int(sims["simulation"].max()) + 1  # type: ignore[operator]
    horizon_steps = int(sims["step"].max())  # type: ignore[operator]

    # Build forward step → date mapping (step 0 == last historical date,
    # step 1 == forward_dates[0], etc.).  The chart renders only steps 1+
    # of each path; step 0 is implicit at last_value on last_date.
    last_date = historical_dates[-1]
    step_to_date: list[dt.date] = [last_date] + list(forward_dates)
    step_to_date = step_to_date[: horizon_steps + 1]

    fig = go.Figure()

    # 1. Historical equity in dimmed Strategy color
    fig = fig.add_trace(
        go.Scatter(
            x=historical_dates,
            y=historical_equity,
            name="Historical",
            line=dict(color=STRATEGY_COLOR, width=2),
            hoverinfo="skip",
        )
    )

    # 2. Spaghetti subset — uniform random sample without replacement.
    # Selecting the prefix `[0, cap)` would lean on the underlying RNG's
    # i.i.d. property over consecutive samples (mostly fine for modern
    # PRNGs but a known MC-literature footgun re: RNG warm-up bias).
    # Drawing a uniform subset is unambiguously representative; we seed
    # the subset selection from the MC run's own parent seed so the
    # displayed paths are deterministic given identical sims input.
    cap = min(n_paths_displayed, n_sims, _MC_PATHS_DOM_CAP)
    if n_paths_displayed > _MC_PATHS_DOM_CAP:
        warnings.warn(
            f"n_paths_displayed={n_paths_displayed} exceeds the Plotly DOM "
            f"responsiveness cap (_MC_PATHS_DOM_CAP={_MC_PATHS_DOM_CAP}); "
            f"clipping to {cap} paths in the displayed subset.",
            stacklevel=2,
        )
    parent_seed = int(sims["seed"][0])
    selected = random.Random(parent_seed ^ 0xCC0FFEE).sample(
        range(n_sims), cap,
    )
    selected_set = pl.Series("simulation", selected)
    spaghetti = sims.filter(pl.col("simulation").is_in(selected_set))
    for sim_idx in selected:
        path = spaghetti.filter(pl.col("simulation") == sim_idx)
        prices = path["price"].to_numpy() * last_value
        fig = fig.add_trace(
            go.Scatter(
                x=step_to_date[: len(prices)],
                y=prices,
                mode="lines",
                line=dict(color="rgba(33, 150, 243, 0.12)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # 3-5. Percentile bands + median, computed across all simulations per step
    by_step = (
        sims.group_by("step")
        .agg(
            pl.col("price").quantile(alpha / 2.0).alias("lo"),
            pl.col("price").quantile(0.5).alias("med"),
            pl.col("price").quantile(1.0 - alpha / 2.0).alias("hi"),
        )
        .sort("step")
    )
    lo = (by_step["lo"].to_numpy() * last_value).tolist()
    med = (by_step["med"].to_numpy() * last_value).tolist()
    hi = (by_step["hi"].to_numpy() * last_value).tolist()

    band_pct = int(round((1.0 - alpha) * 100))

    # Lower percentile — transparent line, must come BEFORE upper for tonexty
    fig = fig.add_trace(
        go.Scatter(
            x=step_to_date,
            y=lo,
            mode="lines",
            line=dict(color="rgba(0, 0, 0, 0)", width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # Upper percentile — fills toward previous trace (the lower percentile)
    fig = fig.add_trace(
        go.Scatter(
            x=step_to_date,
            y=hi,
            mode="lines",
            line=dict(color="rgba(0, 0, 0, 0)", width=0),
            fill="tonexty",
            fillcolor="rgba(33, 150, 243, 0.15)",
            name=f"{band_pct}% band",
        )
    )
    # Median
    fig = fig.add_trace(
        go.Scatter(
            x=step_to_date,
            y=med,
            mode="lines",
            line=dict(color=STRATEGY_COLOR, width=2, dash="dash"),
            name="Median",
        )
    )

    # 6. Anchor line at last historical date
    # plotly accepts dates at runtime; the stub's int|float annotation is wrong.
    fig = fig.add_vline(
        x=last_date,  # type: ignore[arg-type]
        line_dash="dot",
        line_color="gray",
        opacity=0.5,
    )

    fig = fig.update_layout(
        _base_layout(
            f"{title} — {n_sims:,} sims, {horizon_steps}-bar horizon",
            height=420,
        ),
        yaxis_tickprefix="",
        yaxis_tickformat=",.2f",
    )
    return _to_div(fig)
