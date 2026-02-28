from __future__ import annotations

import math
from datetime import date

import polars as pl

from ._types import MetricsResult, ReportConfig


def compute_metrics(
    returns: pl.DataFrame,
    benchmark: pl.DataFrame | None,
    config: ReportConfig,
) -> MetricsResult:
    """Compute all 25 metrics from a daily returns DataFrame."""
    ret = returns["return"]
    ppy = config.periods_per_year
    rf_daily = config.rf / ppy

    # --- Returns ---
    cumulative = _cumulative_return(ret, config.compounded)
    n_years = len(ret) / ppy
    cagr = _cagr(cumulative, n_years)
    mtd = _period_return(returns, "mtd", config.compounded)
    ytd = _period_return(returns, "ytd", config.compounded)
    one_year = _period_return(returns, "1y", config.compounded)

    # --- Risk ---
    vol = _annualized_volatility(ret, ppy)
    dd = drawdown_series(ret, config.compounded)
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0  # type: ignore[arg-type]
    max_dd_idx = dd.arg_min()
    dates = returns["date"]
    max_dd_date = str(dates[max_dd_idx]) if max_dd_idx is not None and max_dd_idx < len(dates) else None
    longest_dd = _longest_drawdown_days(dd, dates)
    avg_dd = _avg_drawdown(dd)

    # --- Ratios ---
    excess = ret - rf_daily
    sharpe = _sharpe(excess, ppy)
    sortino = _sortino(excess, ret, rf_daily, ppy)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    omega = _omega(ret, rf_daily)
    romad = cumulative / abs(max_dd) if max_dd != 0 else 0.0

    # --- Tail ---
    var_95 = _var(ret, 0.05)
    cvar_95 = _cvar(ret, 0.05)

    # --- Win/Loss ---
    wr = _win_rate(ret)
    payoff = _payoff_ratio(ret)
    pf = _profit_factor(ret)
    kelly = _kelly_criterion(wr, payoff)

    # --- Benchmark ---
    alpha = beta = r_sq = info_ratio = None
    if benchmark is not None and not benchmark.is_empty():
        bench_ret = benchmark["return"]
        min_len = min(len(ret), len(bench_ret))
        r, b = ret.head(min_len), bench_ret.head(min_len)
        beta = _beta(r, b)
        alpha = _alpha(r, b, beta, config.rf, ppy)
        r_sq = _r_squared(r, b)
        info_ratio = _information_ratio(r, b, ppy)

    return MetricsResult(
        cumulative_return=cumulative, cagr=cagr, mtd=mtd, ytd=ytd, one_year=one_year,
        sharpe=sharpe, sortino=sortino, calmar=calmar, omega=omega, romad=romad,
        max_drawdown=max_dd, max_drawdown_date=max_dd_date,
        longest_drawdown_days=longest_dd, avg_drawdown=avg_dd, volatility=vol,
        var_95=var_95, cvar_95=cvar_95,
        win_rate=wr, payoff_ratio=payoff, profit_factor=pf, kelly_criterion=kelly,
        alpha=alpha, beta=beta, r_squared=r_sq, information_ratio=info_ratio,
    )


# ---------------------------------------------------------------------------
# Cumulative / Growth
# ---------------------------------------------------------------------------

def _cumulative_return(ret: pl.Series, compounded: bool) -> float:
    if len(ret) == 0:
        return 0.0
    if compounded:
        return float((1 + ret).product()) - 1
    return float(ret.sum())


def _cagr(cumulative: float, n_years: float) -> float:
    if n_years <= 0 or cumulative <= -1:
        return 0.0
    return (1 + cumulative) ** (1 / n_years) - 1


def _period_return(returns: pl.DataFrame, period: str, compounded: bool) -> float:
    """Return for a sub-period: mtd, ytd, or 1y."""
    if returns.is_empty():
        return 0.0
    last_date: date = returns["date"].max()  # type: ignore[assignment]
    match period:
        case "mtd":
            mask = (pl.col("date").dt.year() == last_date.year) & (pl.col("date").dt.month() == last_date.month)
        case "ytd":
            mask = pl.col("date").dt.year() == last_date.year
        case "1y":
            try:
                cutoff = last_date.replace(year=last_date.year - 1)
            except ValueError:  # Feb 29 → Feb 28
                cutoff = last_date.replace(year=last_date.year - 1, day=28)
            mask = pl.col("date") > cutoff
        case _:
            return 0.0
    subset = returns.filter(mask)["return"]
    return _cumulative_return(subset, compounded)


# ---------------------------------------------------------------------------
# Volatility / Drawdown
# ---------------------------------------------------------------------------

def _annualized_volatility(ret: pl.Series, ppy: int) -> float:
    if len(ret) < 2:
        return 0.0
    return float(ret.std()) * math.sqrt(ppy)  # type: ignore[arg-type]


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


def _longest_drawdown_days(dd: pl.Series, dates: pl.Series) -> int:
    """Longest drawdown duration in calendar days (vectorised)."""
    if len(dd) == 0:
        return 0
    df = pl.DataFrame({"dd": dd, "date": dates}).with_columns(
        (pl.col("dd") < 0).alias("in_dd"),
    ).with_columns(
        (pl.col("in_dd") != pl.col("in_dd").shift(1)).fill_null(True).cum_sum().alias("group"),
    )
    dd_groups = (
        df.filter(pl.col("in_dd"))
        .group_by("group")
        .agg((pl.col("date").max() - pl.col("date").min()).dt.total_days().alias("days"))
    )
    if dd_groups.is_empty():
        return 0
    return int(dd_groups["days"].max())  # type: ignore[arg-type]


def _avg_drawdown(dd: pl.Series) -> float:
    """Average of drawdown values during drawdown episodes."""
    if len(dd) == 0:
        return 0.0
    neg = dd.filter(dd < 0)
    if len(neg) == 0:
        return 0.0
    return float(neg.mean())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Ratios
# ---------------------------------------------------------------------------

def _sharpe(excess: pl.Series, ppy: int) -> float:
    if len(excess) < 2:
        return 0.0
    std = float(excess.std())  # type: ignore[arg-type]
    if std == 0:
        return 0.0
    return float(excess.mean()) / std * math.sqrt(ppy)  # type: ignore[arg-type]


def _sortino(excess: pl.Series, ret: pl.Series, rf_daily: float, ppy: int) -> float:
    if len(ret) < 2:
        return 0.0
    diff = ret - rf_daily
    neg_sq = diff.clip(upper_bound=0.0) ** 2
    downside_dev = math.sqrt(float(neg_sq.mean()))  # type: ignore[arg-type]
    if downside_dev == 0:
        return 0.0
    return float(excess.mean()) / downside_dev * math.sqrt(ppy)  # type: ignore[arg-type]


def _omega(ret: pl.Series, threshold: float) -> float:
    if len(ret) == 0:
        return 0.0
    diff = ret - threshold
    gains = float(diff.clip(lower_bound=0.0).sum())
    losses = float((-diff).clip(lower_bound=0.0).sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


# ---------------------------------------------------------------------------
# Tail Risk
# ---------------------------------------------------------------------------

def _var(ret: pl.Series, alpha: float) -> float:
    if len(ret) == 0:
        return 0.0
    return float(ret.quantile(alpha, interpolation="linear"))  # type: ignore[arg-type]


def _cvar(ret: pl.Series, alpha: float) -> float:
    if len(ret) == 0:
        return 0.0
    threshold = ret.quantile(alpha, interpolation="linear")
    tail = ret.filter(ret <= threshold)
    if len(tail) == 0:
        return float(threshold)  # type: ignore[arg-type]
    return float(tail.mean())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Win / Loss
# ---------------------------------------------------------------------------

def _win_rate(ret: pl.Series) -> float:
    if len(ret) == 0:
        return 0.0
    return float((ret > 0).sum()) / len(ret)


def _payoff_ratio(ret: pl.Series) -> float:
    wins = ret.filter(ret > 0)
    losses = ret.filter(ret < 0)
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    avg_loss = abs(float(losses.mean()))  # type: ignore[arg-type]
    if avg_loss == 0:
        return 0.0
    return float(wins.mean()) / avg_loss  # type: ignore[arg-type]


def _profit_factor(ret: pl.Series) -> float:
    gains = float(ret.filter(ret > 0).sum())
    losses = abs(float(ret.filter(ret < 0).sum()))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _kelly_criterion(win_rate: float, payoff_ratio: float) -> float:
    if payoff_ratio == 0:
        return 0.0
    return win_rate - (1 - win_rate) / payoff_ratio


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _beta(ret: pl.Series, bench: pl.Series) -> float:
    if len(ret) < 2:
        return 0.0
    n = len(ret)
    mean_r = float(ret.mean())  # type: ignore[arg-type]
    mean_b = float(bench.mean())  # type: ignore[arg-type]
    cov = float(((ret - mean_r) * (bench - mean_b)).sum()) / (n - 1)
    var_b = float(bench.var())  # type: ignore[arg-type]
    if var_b == 0:
        return 0.0
    return cov / var_b


def _alpha(ret: pl.Series, bench: pl.Series, beta: float, rf: float, ppy: int) -> float:
    rf_daily = rf / ppy
    ann_ret = (float(ret.mean()) - rf_daily) * ppy  # type: ignore[arg-type]
    ann_bench = (float(bench.mean()) - rf_daily) * ppy  # type: ignore[arg-type]
    return ann_ret - beta * ann_bench


def _r_squared(ret: pl.Series, bench: pl.Series) -> float:
    if len(ret) < 2:
        return 0.0
    df = pl.DataFrame({"r": ret, "b": bench})
    corr = df.select(pl.corr("r", "b")).item()
    if corr is None:
        return 0.0
    return float(corr) ** 2


def _information_ratio(ret: pl.Series, bench: pl.Series, ppy: int) -> float:
    if len(ret) < 2:
        return 0.0
    active = ret - bench
    te = float(active.std()) * math.sqrt(ppy)  # type: ignore[arg-type]
    if te == 0:
        return 0.0
    return float(active.mean()) * ppy / te  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Grouping helpers (used by _plots and consumers)
# ---------------------------------------------------------------------------

def monthly_returns(returns: pl.DataFrame, compounded: bool = True) -> pl.DataFrame:
    """Group returns by year/month → DataFrame with ``year``, ``month``, ``monthly_return``."""
    agg_expr = ((1 + pl.col("return")).product() - 1) if compounded else pl.col("return").sum()
    return (
        returns
        .with_columns(pl.col("date").dt.year().alias("year"), pl.col("date").dt.month().alias("month"))
        .group_by("year", "month")
        .agg(agg_expr.alias("monthly_return"))
        .sort("year", "month")
    )


def yearly_returns(returns: pl.DataFrame, compounded: bool = True) -> pl.DataFrame:
    """Group returns by year → DataFrame with ``year``, ``yearly_return``."""
    agg_expr = ((1 + pl.col("return")).product() - 1) if compounded else pl.col("return").sum()
    return (
        returns
        .with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year")
        .agg(agg_expr.alias("yearly_return"))
        .sort("year")
    )


def cumulative_returns(ret: pl.Series, compounded: bool = True) -> pl.Series:
    """Compute cumulative returns series."""
    if len(ret) == 0:
        return pl.Series("cumulative", [], dtype=pl.Float64)
    if compounded:
        return ((1 + ret).cum_prod() - 1).alias("cumulative")
    return ret.cum_sum().alias("cumulative")


def rolling_sharpe(ret: pl.Series, window: int = 126, ppy: int = 252) -> pl.Series:
    """Rolling Sharpe ratio with *window*-day lookback."""
    if len(ret) < window:
        return pl.Series("rolling_sharpe", [None] * len(ret), dtype=pl.Float64)
    mean = ret.rolling_mean(window_size=window)
    std = ret.rolling_std(window_size=window)
    return (mean / std * math.sqrt(ppy)).alias("rolling_sharpe")


def rolling_volatility(ret: pl.Series, window: int = 126, ppy: int = 252) -> pl.Series:
    """Rolling annualised volatility with *window*-day lookback."""
    if len(ret) < window:
        return pl.Series("rolling_vol", [None] * len(ret), dtype=pl.Float64)
    return (ret.rolling_std(window_size=window) * math.sqrt(ppy)).alias("rolling_vol")
