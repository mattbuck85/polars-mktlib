from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

import jinja2

if TYPE_CHECKING:
    from ._types import MetricsResult

_cached_template: jinja2.Template | None = None


def _get_template() -> jinja2.Template:
    global _cached_template  # noqa: PLW0603
    if _cached_template is None:
        text = (files("mktlib.reports") / "templates" / "tearsheet.html.j2").read_text(encoding="utf-8")
        env = jinja2.Environment(autoescape=False)
        _cached_template = env.from_string(text)
    return _cached_template


def render(
    metrics: MetricsResult,
    charts: dict[str, str],
    title: str,
    start_date: str,
    end_date: str,
    trading_days: int,
) -> str:
    """Render the full tearsheet HTML."""
    template = _get_template()
    return template.render(
        title=title,
        start_date=start_date,
        end_date=end_date,
        trading_days=trading_days,
        metrics_groups=_format_metrics(metrics),
        charts=charts,
    )


def _format_metrics(m: MetricsResult) -> list[tuple[str, list[tuple[str, str]]]]:
    """Format metrics into display-ready groups."""
    def pct(v: float) -> str:
        return f"{v * 100:.2f}%"

    def ratio(v: float) -> str:
        if abs(v) == float("inf"):
            return "∞" if v > 0 else "-∞"
        return f"{v:.3f}"

    def opt_pct(v: float | None) -> str:
        return pct(v) if v is not None else "—"

    def opt_ratio(v: float | None) -> str:
        return ratio(v) if v is not None else "—"

    return [
        ("Returns", [
            ("Cumulative Return", pct(m.cumulative_return)),
            ("CAGR", pct(m.cagr)),
            ("MTD", pct(m.mtd)),
            ("YTD", pct(m.ytd)),
            ("1Y Return", pct(m.one_year)),
        ]),
        ("Ratios", [
            ("Sharpe Ratio", ratio(m.sharpe)),
            ("Sortino Ratio", ratio(m.sortino)),
            ("Calmar Ratio", ratio(m.calmar)),
            ("Omega Ratio", ratio(m.omega)),
            ("RoMaD", ratio(m.romad)),
        ]),
        ("Risk", [
            ("Max Drawdown", pct(m.max_drawdown)),
            ("Max DD Date", m.max_drawdown_date or "—"),
            ("Longest DD (days)", str(m.longest_drawdown_days)),
            ("Avg Drawdown", pct(m.avg_drawdown)),
            ("Volatility (ann.)", pct(m.volatility)),
        ]),
        ("Tail Risk", [
            ("VaR (95%)", pct(m.var_95)),
            ("CVaR (95%)", pct(m.cvar_95)),
        ]),
        ("Win/Loss", [
            ("Win Rate", pct(m.win_rate)),
            ("Payoff Ratio", ratio(m.payoff_ratio)),
            ("Profit Factor", ratio(m.profit_factor)),
            ("Kelly Criterion", pct(m.kelly_criterion)),
        ]),
        ("Benchmark", [
            ("Alpha", opt_pct(m.alpha)),
            ("Beta", opt_ratio(m.beta)),
            ("R-Squared", opt_ratio(m.r_squared)),
            ("Information Ratio", opt_ratio(m.information_ratio)),
        ]),
    ]
