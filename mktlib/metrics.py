from __future__ import annotations

import enum
import math

import polars as pl


class Metric(enum.Enum):
    """Enumeration of standalone financial metrics."""

    CUMULATIVE_RETURN = "cumulative_return"
    CAGR = "cagr"
    ANNUALIZED_VOLATILITY = "annualized_volatility"
    MAX_DRAWDOWN = "max_drawdown"
    AVG_DRAWDOWN = "avg_drawdown"
    LONGEST_DRAWDOWN_DAYS = "longest_drawdown_days"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    CALMAR = "calmar"
    ROMAD = "romad"
    OMEGA = "omega"
    VAR = "var"
    CVAR = "cvar"
    WIN_RATE = "win_rate"
    PAYOFF_RATIO = "payoff_ratio"
    PROFIT_FACTOR = "profit_factor"
    KELLY_CRITERION = "kelly_criterion"


def drawdown_series(ret: pl.Series, compounded: bool = True) -> pl.Series:
    """Compute drawdown series from returns."""
    if len(ret) == 0:
        return pl.Series("drawdown", [], dtype=pl.Float64)
    if compounded:
        wealth = (1 + ret).cum_prod()
    else:
        wealth = 1 + ret.cum_sum()
    running_max = wealth.cum_max()
    return (wealth / running_max - 1).alias("drawdown")


# ---------------------------------------------------------------------------
# Standalone metric functions
# ---------------------------------------------------------------------------


def cumulative_return(ret: pl.Series, compounded: bool = True) -> float:
    """Total cumulative return."""
    if len(ret) == 0:
        return 0.0
    if compounded:
        return float((1 + ret).product()) - 1
    return float(ret.sum())


def cagr(ret: pl.Series, compounded: bool = True, ppy: int = 252) -> float:
    """Compound annual growth rate."""
    cum = cumulative_return(ret, compounded)
    n_years = len(ret) / ppy
    if n_years <= 0 or cum <= -1:
        return 0.0
    return (1 + cum) ** (1 / n_years) - 1


def annualized_volatility(ret: pl.Series, ppy: int = 252) -> float:
    """Annualized standard deviation of returns."""
    if len(ret) < 2:
        return 0.0
    return float(ret.std()) * math.sqrt(ppy)  # type: ignore[arg-type]


def avg_drawdown(dd: pl.Series) -> float:
    """Average of drawdown values during drawdown episodes."""
    if len(dd) == 0:
        return 0.0
    neg = dd.filter(dd < 0)
    if len(neg) == 0:
        return 0.0
    return float(neg.mean())  # type: ignore[arg-type]


def longest_drawdown_days(dd: pl.Series, dates: pl.Series) -> float:
    """Longest drawdown duration in calendar days.

    Returns
    -------
    float
        Duration in calendar days.

    Raises
    ------
    ValueError
        If *dates* is not provided (None passed via calculate_metric).
    """
    if len(dd) == 0:
        return 0.0
    df = (
        pl.DataFrame({"dd": dd, "date": dates})
        .with_columns(
            (pl.col("dd") < 0).alias("in_dd"),
        )
        .with_columns(
            (pl.col("in_dd") != pl.col("in_dd").shift(1))
            .fill_null(True)
            .cum_sum()
            .alias("group"),
        )
    )
    dd_groups = (
        df.filter(pl.col("in_dd"))
        .group_by("group")
        .agg(
            (pl.col("date").max() - pl.col("date").min())
            .dt.total_days()
            .alias("days")
        )
    )
    if dd_groups.is_empty():
        return 0.0
    return float(dd_groups["days"].max())  # type: ignore[arg-type]


def sharpe(ret: pl.Series, ppy: int = 252, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio."""
    if len(ret) < 2:
        return 0.0
    rf_daily = rf / ppy
    excess = ret - rf_daily
    std = float(excess.std())  # type: ignore[arg-type]
    if std == 0:
        return 0.0
    return float(excess.mean()) / std * math.sqrt(ppy)  # type: ignore[arg-type]


def sortino(ret: pl.Series, ppy: int = 252, rf: float = 0.0) -> float:
    """Annualized Sortino ratio."""
    if len(ret) < 2:
        return 0.0
    rf_daily = rf / ppy
    excess = ret - rf_daily
    diff = ret - rf_daily
    neg_sq = diff.clip(upper_bound=0.0) ** 2
    downside_dev = math.sqrt(float(neg_sq.mean()))  # type: ignore[arg-type]
    if downside_dev == 0:
        return 0.0
    return float(excess.mean()) / downside_dev * math.sqrt(ppy)  # type: ignore[arg-type]


def omega(ret: pl.Series, ppy: int = 252, rf: float = 0.0) -> float:
    """Omega ratio."""
    if len(ret) == 0:
        return 0.0
    threshold = rf / ppy
    diff = ret - threshold
    gains = float(diff.clip(lower_bound=0.0).sum())
    losses = float((-diff).clip(lower_bound=0.0).sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def var(ret: pl.Series, alpha: float = 0.05) -> float:
    """Value at Risk at *alpha* confidence level."""
    if len(ret) == 0:
        return 0.0
    return float(ret.quantile(alpha, interpolation="linear"))  # type: ignore[arg-type]


def cvar(ret: pl.Series, alpha: float = 0.05) -> float:
    """Conditional Value at Risk (Expected Shortfall) at *alpha*."""
    if len(ret) == 0:
        return 0.0
    threshold = ret.quantile(alpha, interpolation="linear")
    tail = ret.filter(ret <= threshold)
    if len(tail) == 0:
        return float(threshold)  # type: ignore[arg-type]
    return float(tail.mean())  # type: ignore[arg-type]


def win_rate(ret: pl.Series) -> float:
    """Fraction of positive-return bars."""
    if len(ret) == 0:
        return 0.0
    return float((ret > 0).sum()) / len(ret)


def payoff_ratio(ret: pl.Series) -> float:
    """Average win / average loss."""
    wins = ret.filter(ret > 0)
    losses = ret.filter(ret < 0)
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    avg_loss = abs(float(losses.mean()))  # type: ignore[arg-type]
    if avg_loss == 0:
        return 0.0
    return float(wins.mean()) / avg_loss  # type: ignore[arg-type]


def profit_factor(ret: pl.Series) -> float:
    """Sum of gains / sum of losses."""
    gains = float(ret.filter(ret > 0).sum())
    losses = abs(float(ret.filter(ret < 0).sum()))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def kelly_criterion(ret: pl.Series) -> float:
    """Kelly criterion from bar-level returns."""
    wr = win_rate(ret)
    pr = payoff_ratio(ret)
    if pr == 0:
        return 0.0
    return wr - (1 - wr) / pr


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def calculate_metric(
    metric: Metric,
    ret: pl.Series,
    *,
    dd: pl.Series | None = None,
    dates: pl.Series | None = None,
    compounded: bool = True,
    ppy: int = 252,
    rf: float = 0.0,
    alpha: float = 0.05,
) -> float:
    """Compute a single metric by enum value.

    Parameters
    ----------
    metric
        Which metric to compute.
    ret
        Bar-level return series.
    dd
        Pre-computed drawdown series. Lazily computed from *ret* when ``None``
        and the metric requires it.
    dates
        Date series, required for ``LONGEST_DRAWDOWN_DAYS``.
    compounded
        Whether returns compound (affects cumulative return and drawdown).
    ppy
        Periods per year for annualisation (default 252).
    rf
        Annual risk-free rate.
    alpha
        Tail-risk quantile for VaR/CVaR.
    """

    def _dd() -> pl.Series:
        return dd if dd is not None else drawdown_series(ret, compounded)

    match metric:
        case Metric.CUMULATIVE_RETURN:
            return cumulative_return(ret, compounded)
        case Metric.CAGR:
            return cagr(ret, compounded, ppy)
        case Metric.ANNUALIZED_VOLATILITY:
            return annualized_volatility(ret, ppy)
        case Metric.MAX_DRAWDOWN:
            d = _dd()
            return float(d.min()) if len(d) > 0 else 0.0  # type: ignore[arg-type]
        case Metric.AVG_DRAWDOWN:
            return avg_drawdown(_dd())
        case Metric.LONGEST_DRAWDOWN_DAYS:
            if dates is None:
                msg = "dates= is required for LONGEST_DRAWDOWN_DAYS"
                raise ValueError(msg)
            return longest_drawdown_days(_dd(), dates)
        case Metric.SHARPE:
            return sharpe(ret, ppy, rf)
        case Metric.SORTINO:
            return sortino(ret, ppy, rf)
        case Metric.CALMAR:
            c = cagr(ret, compounded, ppy)
            d = _dd()
            max_dd = float(d.min()) if len(d) > 0 else 0.0  # type: ignore[arg-type]
            return c / abs(max_dd) if max_dd != 0 else 0.0
        case Metric.ROMAD:
            cum = cumulative_return(ret, compounded)
            d = _dd()
            max_dd = float(d.min()) if len(d) > 0 else 0.0  # type: ignore[arg-type]
            return cum / abs(max_dd) if max_dd != 0 else 0.0
        case Metric.OMEGA:
            return omega(ret, ppy, rf)
        case Metric.VAR:
            return var(ret, alpha)
        case Metric.CVAR:
            return cvar(ret, alpha)
        case Metric.WIN_RATE:
            return win_rate(ret)
        case Metric.PAYOFF_RATIO:
            return payoff_ratio(ret)
        case Metric.PROFIT_FACTOR:
            return profit_factor(ret)
        case Metric.KELLY_CRITERION:
            return kelly_criterion(ret)
