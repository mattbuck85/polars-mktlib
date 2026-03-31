from __future__ import annotations

import math
from datetime import date

import polars as pl

from mktlib.metrics import (
    Metric,
    calculate_metric,
    drawdown_series,
)

from ._types import MetricsResult, ReportConfig, TradeMetrics


def compute_metrics(
    returns: pl.DataFrame,
    benchmark: pl.DataFrame | None,
    config: ReportConfig,
) -> MetricsResult:
    """Compute all 25 metrics from a daily returns DataFrame."""
    ret = returns["return"]
    ppy = config.periods_per_year
    dates = returns["date"] if not returns.is_empty() else None

    # Pre-compute drawdown once for reuse
    dd = drawdown_series(ret, config.compounded)

    # --- Returns ---
    cumulative = calculate_metric(Metric.CUMULATIVE_RETURN, ret, compounded=config.compounded)
    cagr_val = calculate_metric(Metric.CAGR, ret, compounded=config.compounded, ppy=ppy)
    mtd = _period_return(returns, "mtd", config.compounded)
    ytd = _period_return(returns, "ytd", config.compounded)
    one_year = _period_return(returns, "1y", config.compounded)

    # --- Risk ---
    vol = calculate_metric(Metric.ANNUALIZED_VOLATILITY, ret, ppy=ppy)
    max_dd = calculate_metric(Metric.MAX_DRAWDOWN, ret, dd=dd, compounded=config.compounded)
    max_dd_idx = dd.arg_min()
    max_dd_date = (
        str(dates[max_dd_idx])
        if dates is not None and max_dd_idx is not None and max_dd_idx < len(dates)
        else None
    )
    longest_dd = calculate_metric(
        Metric.LONGEST_DRAWDOWN_DAYS, ret, dd=dd, dates=dates, compounded=config.compounded,
    ) if dates is not None else 0.0
    avg_dd = calculate_metric(Metric.AVG_DRAWDOWN, ret, dd=dd, compounded=config.compounded)

    # --- Ratios ---
    sharpe_val = calculate_metric(Metric.SHARPE, ret, ppy=ppy, rf=config.rf)
    sortino_val = calculate_metric(Metric.SORTINO, ret, ppy=ppy, rf=config.rf)
    calmar_val = calculate_metric(Metric.CALMAR, ret, dd=dd, compounded=config.compounded, ppy=ppy)
    omega_val = calculate_metric(Metric.OMEGA, ret, ppy=ppy, rf=config.rf)
    romad_val = calculate_metric(Metric.ROMAD, ret, dd=dd, compounded=config.compounded)

    # --- Tail ---
    var_95 = calculate_metric(Metric.VAR, ret, alpha=0.05)
    cvar_95 = calculate_metric(Metric.CVAR, ret, alpha=0.05)

    # --- Win/Loss ---
    wr = calculate_metric(Metric.WIN_RATE, ret)
    payoff = calculate_metric(Metric.PAYOFF_RATIO, ret)
    pf = calculate_metric(Metric.PROFIT_FACTOR, ret)
    kelly = calculate_metric(Metric.KELLY_CRITERION, ret)

    # --- Benchmark ---
    alpha_val = beta = r_sq = info_ratio = None
    if benchmark is not None and not benchmark.is_empty():
        bench_ret = benchmark["return"]
        min_len = min(len(ret), len(bench_ret))
        r, b = ret.head(min_len), bench_ret.head(min_len)
        beta = _beta(r, b)
        alpha_val = _alpha(r, b, beta, config.rf, ppy)
        r_sq = _r_squared(r, b)
        info_ratio = _information_ratio(r, b, ppy)

    return MetricsResult(
        cumulative_return=cumulative,
        cagr=cagr_val,
        mtd=mtd,
        ytd=ytd,
        one_year=one_year,
        sharpe=sharpe_val,
        sortino=sortino_val,
        calmar=calmar_val,
        omega=omega_val,
        romad=romad_val,
        max_drawdown=max_dd,
        max_drawdown_date=max_dd_date,
        longest_drawdown_days=longest_dd,
        avg_drawdown=avg_dd,
        volatility=vol,
        var_95=var_95,
        cvar_95=cvar_95,
        win_rate=wr,
        payoff_ratio=payoff,
        profit_factor=pf,
        kelly_criterion=kelly,
        alpha=alpha_val,
        beta=beta,
        r_squared=r_sq,
        information_ratio=info_ratio,
    )


# ---------------------------------------------------------------------------
# Per-trade metrics
# ---------------------------------------------------------------------------


def compute_trade_metrics(trades: pl.DataFrame) -> TradeMetrics:
    """Compute per-trade metrics from a trades DataFrame.

    Parameters
    ----------
    trades
        DataFrame with columns: ``entry_date`` (Date), ``exit_date`` (Date),
        ``side`` (Int8), ``pnl`` (Float64), ``bars_held`` (Int64 or Int32).
    """
    pnl = trades["pnl"].drop_nulls().drop_nans()
    n = len(pnl)

    if n == 0:
        return TradeMetrics(
            trade_win_rate=0.0,
            payoff_ratio=0.0,
            profit_factor=0.0,
            kelly_criterion=0.0,
            avg_trade_pnl=0.0,
            avg_bars_held=0.0,
            total_trades=0,
            avg_winner=0.0,
            avg_loser=0.0,
            largest_winner=0.0,
            largest_loser=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            trade_sharpe=0.0,
            trade_sortino=0.0,
            trades_per_year=0.0,
        )

    winners = pnl.filter(pnl > 0)
    losers = pnl.filter(pnl < 0)

    win_count = len(winners)
    win_rate = win_count / n

    avg_win = float(winners.mean()) if len(winners) > 0 else 0.0  # type: ignore[arg-type]
    avg_loss = float(losers.mean()) if len(losers) > 0 else 0.0  # type: ignore[arg-type]

    payoff = avg_win / abs(avg_loss) if avg_loss != 0.0 else float("inf")
    pf_sum_win = float(winners.sum()) if len(winners) > 0 else 0.0  # type: ignore[arg-type]
    pf_sum_loss = abs(float(losers.sum())) if len(losers) > 0 else 0.0  # type: ignore[arg-type]
    profit_factor = pf_sum_win / pf_sum_loss if pf_sum_loss != 0.0 else float("inf")

    kelly = win_rate - (1.0 - win_rate) / payoff if payoff != 0.0 and not math.isinf(payoff) else win_rate

    avg_pnl = float(pnl.mean())  # type: ignore[arg-type]
    avg_bars = float(trades["bars_held"].mean())  # type: ignore[arg-type]

    # Consecutive wins / losses
    pnl_list = pnl.to_list()
    max_wins = _max_consecutive(pnl_list, positive=True)
    max_losses = _max_consecutive(pnl_list, positive=False)

    # Risk-adjusted — trades per year
    entry_min: date = trades["entry_date"].min()  # type: ignore[assignment]
    exit_max: date = trades["exit_date"].max()  # type: ignore[assignment]
    if entry_min is not None and exit_max is not None:
        span_days = float((exit_max - entry_min).days)  # type: ignore[union-attr]
        # Calendar days (not trading days) — trade frequency spans weekends/holidays
        span_years = span_days / 365.25
        trades_per_year = n / span_years if span_years > 0.0 else float(n)
    else:
        trades_per_year = float(n)

    pnl_std = float(pnl.std()) if n > 1 else 0.0  # type: ignore[arg-type]
    trade_sharpe = (
        avg_pnl / pnl_std * math.sqrt(trades_per_year)
        if pnl_std > 0.0
        else 0.0
    )

    downside = pnl.filter(pnl < 0)
    downside_std_raw = downside.std() if len(downside) > 1 else None  # type: ignore[arg-type]
    downside_std = abs(float(downside_std_raw)) if downside_std_raw is not None else 0.0  # type: ignore[arg-type]
    trade_sortino = (
        avg_pnl / downside_std * math.sqrt(trades_per_year)
        if downside_std != 0.0
        else 0.0
    )

    return TradeMetrics(
        trade_win_rate=win_rate,
        payoff_ratio=payoff,
        profit_factor=profit_factor,
        kelly_criterion=kelly,
        avg_trade_pnl=avg_pnl,
        avg_bars_held=avg_bars,
        total_trades=n,
        avg_winner=avg_win,
        avg_loser=avg_loss,
        largest_winner=float(pnl.max()),  # type: ignore[arg-type]
        largest_loser=float(pnl.min()),  # type: ignore[arg-type]
        max_consecutive_wins=max_wins,
        max_consecutive_losses=max_losses,
        trade_sharpe=trade_sharpe,
        trade_sortino=trade_sortino,
        trades_per_year=trades_per_year,
    )


def _max_consecutive(pnl_list: list[float], *, positive: bool) -> int:
    """Return the longest consecutive run of wins (positive=True) or losses."""
    max_run = 0
    current = 0
    for v in pnl_list:
        if (positive and v > 0) or (not positive and v < 0):
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


# ---------------------------------------------------------------------------
# Period returns (need DataFrame with date column — not in core metrics)
# ---------------------------------------------------------------------------


def _period_return(
    returns: pl.DataFrame, period: str, compounded: bool
) -> float:
    """Return for a sub-period: mtd, ytd, or 1y."""
    if returns.is_empty():
        return 0.0
    last_date: date = returns["date"].max()  # type: ignore[assignment]
    match period:
        case "mtd":
            mask = (pl.col("date").dt.year() == last_date.year) & (
                pl.col("date").dt.month() == last_date.month
            )
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
    from mktlib.metrics import cumulative_return
    subset = returns.filter(mask)["return"]
    return cumulative_return(subset, compounded)


# ---------------------------------------------------------------------------
# Benchmark (need benchmark series — not in core metrics)
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


def _alpha(
    ret: pl.Series, bench: pl.Series, beta: float, rf: float, ppy: int
) -> float:
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


def monthly_returns(
    returns: pl.DataFrame, compounded: bool = True
) -> pl.DataFrame:
    """Group returns by year/month → DataFrame with ``year``, ``month``, ``monthly_return``."""
    agg_expr = (
        ((1 + pl.col("return")).product() - 1)
        if compounded
        else pl.col("return").sum()
    )
    return (
        returns.with_columns(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        )
        .group_by("year", "month")
        .agg(agg_expr.alias("monthly_return"))
        .sort("year", "month")
    )


def yearly_returns(
    returns: pl.DataFrame, compounded: bool = True
) -> pl.DataFrame:
    """Group returns by year → DataFrame with ``year``, ``yearly_return``."""
    agg_expr = (
        ((1 + pl.col("return")).product() - 1)
        if compounded
        else pl.col("return").sum()
    )
    return (
        returns.with_columns(pl.col("date").dt.year().alias("year"))
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


def rolling_sharpe(
    ret: pl.Series, window: int = 126, ppy: int = 252
) -> pl.Series:
    """Rolling Sharpe ratio with *window*-day lookback."""
    if len(ret) < window:
        return pl.Series("rolling_sharpe", [None] * len(ret), dtype=pl.Float64)
    mean = ret.rolling_mean(window_size=window)
    std = ret.rolling_std(window_size=window)
    return (mean / std * math.sqrt(ppy)).alias("rolling_sharpe")


def rolling_volatility(
    ret: pl.Series, window: int = 126, ppy: int = 252
) -> pl.Series:
    """Rolling annualised volatility with *window*-day lookback."""
    if len(ret) < window:
        return pl.Series("rolling_vol", [None] * len(ret), dtype=pl.Float64)
    return (ret.rolling_std(window_size=window) * math.sqrt(ppy)).alias(
        "rolling_vol"
    )
