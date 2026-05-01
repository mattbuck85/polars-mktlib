from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mktlib.data import Innovations


@dataclass(frozen=True, slots=True)
class ReportConfig:
    """Configuration for report generation."""

    rf: float = 0.0
    periods_per_year: int = 252
    compounded: bool = True
    title: str = "Strategy Tearsheet"


@dataclass(slots=True)
class DrawdownInfo:
    """Drawdown analysis results."""

    max_drawdown: float
    max_drawdown_date: str | None
    longest_drawdown_days: float
    avg_drawdown: float


@dataclass(slots=True)
class MetricsResult:
    """Complete metrics computation result (25 metrics)."""

    # Returns
    cumulative_return: float
    cagr: float
    mtd: float
    ytd: float
    one_year: float

    # Ratios
    sharpe: float
    sortino: float
    calmar: float
    omega: float
    romad: float

    # Risk
    max_drawdown: float
    max_drawdown_date: str | None
    longest_drawdown_days: float
    avg_drawdown: float
    volatility: float

    # Tail risk
    var_95: float
    cvar_95: float

    # Win/Loss
    win_rate: float
    payoff_ratio: float
    profit_factor: float
    kelly_criterion: float

    # Benchmark (None when no benchmark provided)
    alpha: float | None = None
    beta: float | None = None
    r_squared: float | None = None
    information_ratio: float | None = None

    # Per-trade metrics (None when no trades provided)
    trade_metrics: TradeMetrics | None = None

    # Forward-looking risk (None unless MonteCarloConfig.enabled=True)
    mc_var: float | None = None
    mc_cvar: float | None = None


@dataclass(frozen=True, slots=True)
class TradeMetrics:
    """Per-trade performance metrics computed from a trades DataFrame."""

    # Win/Loss (replaces daily-based when trades are provided)
    trade_win_rate: float
    payoff_ratio: float
    profit_factor: float
    kelly_criterion: float
    avg_trade_pnl: float
    avg_bars_held: float
    avg_duration_minutes: float
    total_trades: int

    # Trade Stats card
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Trade Risk-Adjusted card
    trade_sharpe: float
    trade_sortino: float
    trades_per_year: float


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    """Configuration for opt-in Monte Carlo VaR / CVaR + path chart in reports.

    Pass to :func:`mktlib.reports.html` / :func:`mktlib.reports.metrics`
    via the ``mc_config=`` kwarg.  Default ``enabled=False`` preserves
    backwards-compatible report output exactly.

    When ``enabled=True``, one MC GBM batch is run (per report) using
    :func:`mktlib.metrics.monte_carlo_paths`.  Its horizon-end returns
    pre-populate the module-level MC cache so the subsequent VaR / CVaR
    calls reuse the same simulation.  The full simulation frame is also
    consumed by the Monte Carlo paths chart in :func:`mktlib.reports.html`.

    The dataclass intentionally has *no* "method" knob (closed-form
    Gaussian vs simulation): when MC is enabled we always run paths
    because the spaghetti chart requires actual paths.  Users who want
    closed-form numbers without paths should call
    ``mktlib.metrics.var(method="gaussian")`` directly.

    Parameters
    ----------
    enabled
        Master switch; ``False`` (default) disables the entire MC code
        path and the report renders identically to v0.10.x.
    horizon
        Forecast horizon in trading bars (default 21 ≈ 1 month at 252 ppy).
    n_simulations
        Number of simulated paths (default 10 000).  At default scales,
        Gaussian MC runs in ~10 ms / Student-t in ~15 ms.
    innovations
        :class:`mktlib.data.Innovations` member or ``None`` (Gaussian default).
        ``STUDENT_T`` requires ``df``; ``BOOTSTRAP`` resamples
        standardized empirical residuals from the input series.
    df
        Degrees of freedom for ``Innovations.STUDENT_T`` (>2).
    seed
        Reproducibility seed.  ``None`` uses OS entropy.
    alpha
        Tail probability for both the VaR / CVaR numbers and the
        ``α/2 / 1−α/2`` percentile band on the path chart.
    n_paths_displayed
        Number of individual MC paths to render in the spaghetti plot.
        Hard-capped at 500 inside the chart helper to keep the Plotly
        DOM responsive — pass any larger value and it is silently clipped.
    exchange
        ISO MIC for the calendar used to project forward dates on the
        path chart's x-axis (``"XNYS"`` / ``"XLON"`` / ``"XPAR"`` /
        ``"XCME"`` / ``"XNAS"`` / ``"XCBO"`` / ``"XTSE"`` / ``"XETR"`` /
        ``"XTKS"``).
    """

    enabled: bool = False
    horizon: int = 21
    n_simulations: int = 10_000
    innovations: "Innovations | None" = None
    df: float | None = None
    seed: int | None = None
    alpha: float = 0.05
    n_paths_displayed: int = 100
    exchange: str = "XNYS"
