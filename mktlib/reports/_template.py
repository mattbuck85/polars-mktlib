# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownParameterType=false
from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import jinja2  # type: ignore[import-untyped]
from markupsafe import Markup  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from ._types import MetricsResult

_cached_template: Any = None

type _Formatter = Callable[[float], str]


def _get_template() -> Any:
    global _cached_template  # noqa: PLW0603
    if _cached_template is None:
        text = (
            files("mktlib.reports") / "templates" / "tearsheet.html.j2"
        ).read_text(encoding="utf-8")
        env = jinja2.Environment(autoescape=True)
        _cached_template = env.from_string(text)
    return _cached_template


def _resolve_template(override: str | Path | None) -> Any:
    """Return a Jinja2 template from override or the built-in default."""
    if override is None:
        return _get_template()
    env = jinja2.Environment(autoescape=True)
    if isinstance(override, Path):
        return env.from_string(override.read_text(encoding="utf-8"))
    return env.from_string(override)


def render(
    metrics: MetricsResult,
    charts: dict[str, str],
    title: str,
    start_date: str,
    end_date: str,
    trading_days: int,
    *,
    extra_metrics: dict[str, list[tuple[str, str]]] | None = None,
    extra_charts: dict[str, str] | None = None,
    template_override: str | Path | None = None,
) -> str:
    """Render the full tearsheet HTML."""
    template = _resolve_template(template_override)
    groups = _format_metrics(metrics)
    if extra_metrics:
        for category, items in extra_metrics.items():
            groups.append((category, items))
    safe_charts = {k: Markup(v) for k, v in charts.items()}
    safe_extra = {k: Markup(v) for k, v in (extra_charts or {}).items()}
    result: str = template.render(
        title=title,
        start_date=start_date,
        end_date=end_date,
        trading_days=trading_days,
        metrics_groups=groups,
        charts=safe_charts,
        extra_charts=safe_extra,
    )
    return result


def _win_loss_items(
    m: MetricsResult,
    pct: _Formatter,
    ratio: _Formatter,
) -> list[tuple[str, str]]:
    """Return Win/Loss card items, preferring trade_metrics when available."""
    tm = m.trade_metrics
    if tm is not None:
        return [
            ("Trade Win Rate", pct(tm.trade_win_rate)),
            ("Payoff Ratio", ratio(tm.payoff_ratio)),
            ("Profit Factor", ratio(tm.profit_factor)),
            ("Kelly Criterion", pct(tm.kelly_criterion)),
            ("Avg Trade PnL", pct(tm.avg_trade_pnl)),
            ("Avg Bars Held", f"{tm.avg_bars_held:.1f}"),
            ("Total Trades", str(tm.total_trades)),
        ]
    return [
        ("Win Rate", pct(m.win_rate)),
        ("Payoff Ratio", ratio(m.payoff_ratio)),
        ("Profit Factor", ratio(m.profit_factor)),
        ("Kelly Criterion", pct(m.kelly_criterion)),
    ]


def _format_metrics(
    m: MetricsResult,
) -> list[tuple[str, list[tuple[str, str]]]]:
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
        (
            "Returns",
            [
                ("Cumulative Return", pct(m.cumulative_return)),
                ("CAGR", pct(m.cagr)),
                ("MTD", pct(m.mtd)),
                ("YTD", pct(m.ytd)),
                ("1Y Return", pct(m.one_year)),
            ],
        ),
        (
            "Ratios",
            [
                ("Sharpe Ratio", ratio(m.sharpe)),
                ("Sortino Ratio", ratio(m.sortino)),
                ("Calmar Ratio", ratio(m.calmar)),
                ("Omega Ratio", ratio(m.omega)),
                ("RoMaD", ratio(m.romad)),
            ],
        ),
        (
            "Risk",
            [
                ("Max Drawdown", pct(m.max_drawdown)),
                ("Max DD Date", m.max_drawdown_date or "—"),
                ("Longest DD (days)", str(m.longest_drawdown_days)),
                ("Avg Drawdown", pct(m.avg_drawdown)),
                ("Volatility (ann.)", pct(m.volatility)),
            ],
        ),
        (
            "Tail Risk",
            [
                ("VaR (95%)", pct(m.var_95)),
                ("CVaR (95%)", pct(m.cvar_95)),
            ],
        ),
        (
            "Win/Loss",
            _win_loss_items(m, pct, ratio),
        ),
        (
            "Benchmark",
            [
                ("Alpha", opt_pct(m.alpha)),
                ("Beta", opt_ratio(m.beta)),
                ("R-Squared", opt_ratio(m.r_squared)),
                ("Information Ratio", opt_ratio(m.information_ratio)),
            ],
        ),
    ]
