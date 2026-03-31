from __future__ import annotations

from dataclasses import dataclass


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
