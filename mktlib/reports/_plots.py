# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go  # type: ignore[import-untyped]
import plotly.io as pio  # type: ignore[import-untyped]

STRATEGY_COLOR = "#2196F3"
BENCHMARK_COLOR = "#9E9E9E"
NEGATIVE_COLOR = "#EF5350"
POSITIVE_COLOR = "#66BB6A"


def _base_layout(title: str, height: int = 400, **kwargs: Any) -> dict[str, Any]:
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
    dates: list[Any], cum_ret: list[Any],
    bench_dates: list[Any] | None = None, bench_cum_ret: list[Any] | None = None,
) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=cum_ret, name="Strategy", line=dict(color=STRATEGY_COLOR, width=2)))
    if bench_dates and bench_cum_ret:
        fig.add_trace(go.Scatter(
            x=bench_dates, y=bench_cum_ret, name="Benchmark",
            line=dict(color=BENCHMARK_COLOR, width=1.5, dash="dot"),
        ))
    fig.update_layout(**_base_layout("Cumulative Returns"), yaxis_tickformat=".1%")
    return _to_div(fig)


def drawdown_chart(dates: list[Any], drawdowns: list[Any]) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=drawdowns, fill="tozeroy",
        fillcolor="rgba(239,83,80,0.3)", line=dict(color=NEGATIVE_COLOR, width=1),
        name="Drawdown",
    ))
    fig.update_layout(**_base_layout("Drawdown"), yaxis_tickformat=".1%")
    return _to_div(fig)


def yearly_returns_chart(years: list[Any], returns: list[float]) -> str:
    colors = [POSITIVE_COLOR if r >= 0 else NEGATIVE_COLOR for r in returns]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(y) for y in years], y=returns, marker_color=colors, name="Return"))
    fig.update_layout(**_base_layout("Yearly Returns", height=350), yaxis_tickformat=".1%")
    return _to_div(fig)


def monthly_heatmap_chart(years: list[int], months: list[int], returns: list[float]) -> str:
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
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

    fig = go.Figure(data=go.Heatmap(
        z=z, x=month_labels, y=[str(y) for y in unique_years],
        text=text, texttemplate="%{text}",
        colorscale="RdYlGn", zmid=0,
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{text}<extra></extra>",
    ))
    fig.update_layout(**_base_layout("Monthly Returns", height=max(200, len(unique_years) * 35 + 100)))
    return _to_div(fig)


def rolling_sharpe_chart(dates: list[Any], values: list[Any]) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, line=dict(color=STRATEGY_COLOR, width=1.5), name="Rolling Sharpe"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(**_base_layout("Rolling Sharpe (126d)"))
    return _to_div(fig)


def rolling_volatility_chart(dates: list[Any], values: list[Any]) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, line=dict(color="#FF9800", width=1.5), name="Rolling Volatility"))
    fig.update_layout(**_base_layout("Rolling Volatility (126d)"), yaxis_tickformat=".1%")
    return _to_div(fig)


def daily_returns_scatter(dates: list[Any], returns: list[float]) -> str:
    colors = [POSITIVE_COLOR if r >= 0 else NEGATIVE_COLOR for r in returns]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=returns, mode="markers",
        marker=dict(size=3, color=colors, opacity=0.6), name="Daily Return",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(**_base_layout("Daily Returns"), yaxis_tickformat=".2%")
    return _to_div(fig)


def returns_distribution_chart(returns: list[float]) -> str:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, marker_color=STRATEGY_COLOR, opacity=0.7, name="Returns"))
    fig.update_layout(**_base_layout("Returns Distribution", height=350), xaxis_tickformat=".1%", bargap=0.05)
    return _to_div(fig)
