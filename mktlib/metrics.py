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
    """Compute drawdown series from returns.

    A drawdown measures the decline from the most recent equity peak at each
    point in time.  Values are zero when equity is at a new high and negative
    when below one.  The series is useful for visualizing underwater periods
    and as input to ``avg_drawdown`` and ``longest_drawdown_days``.

    When *compounded* is ``True`` (the default), equity is built via
    cumulative product; otherwise via cumulative sum (arithmetic returns).

    Parameters
    ----------
    ret
        Bar-level return series (e.g. daily close-to-close percent changes).
    compounded
        If ``True``, compound returns geometrically.

    Returns
    -------
    pl.Series
        Series named ``"drawdown"`` with values in (-1, 0].
    """
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
    """Total cumulative return over the full series.

    Answers "if I invested $1 at the start, how much did I gain or lose?"
    A value of 0.25 means a 25 % gain; −0.10 means a 10 % loss.

    When *compounded* is ``True``:
    :math:`R = \\prod(1 + r_i) - 1`.
    When ``False``, returns are summed (arithmetic).

    Parameters
    ----------
    ret
        Bar-level return series.
    compounded
        If ``True``, compound returns geometrically.

    Returns
    -------
    float
        Cumulative return as a decimal (not percent).
    """
    if len(ret) == 0:
        return 0.0
    if compounded:
        return float((1 + ret).product()) - 1
    return float(ret.sum())


def cagr(ret: pl.Series, compounded: bool = True, ppy: int = 252) -> float:
    """Compound annual growth rate.

    CAGR converts the total cumulative return into a smooth annualized rate,
    answering "what constant yearly return would produce the same final
    wealth?"

    :math:`\\text{CAGR} = (1 + R)^{1/n_{\\text{years}}} - 1`

    Higher is better.  A CAGR of 0.08 means 8 % per year.  Returns 0.0 when
    the series is empty or the cumulative return is −100 % or worse.

    Parameters
    ----------
    ret
        Bar-level return series.
    compounded
        If ``True``, compound returns geometrically.
    ppy
        Periods per year — must match the bar frequency of *ret*.
        Use 252 for daily, 52 for weekly, 12 for monthly,
        ``252 * 390`` for minute bars, etc.

    Returns
    -------
    float
        Annualized growth rate as a decimal.
    """
    cum = cumulative_return(ret, compounded)
    n_years = len(ret) / ppy
    if n_years <= 0 or cum <= -1:
        return 0.0
    return (1 + cum) ** (1 / n_years) - 1


def annualized_volatility(ret: pl.Series, ppy: int = 252) -> float:
    """Annualized standard deviation of returns.

    Volatility quantifies the dispersion of returns and is the most widely
    used measure of investment risk.  Higher values indicate larger typical
    swings, both up and down.

    :math:`\\sigma_{\\text{ann}} = \\text{std}(r) \\cdot \\sqrt{N}`

    Typical daily-return volatility for US equities is 0.15–0.25
    (15–25 % annualized).  The annualization assumes returns are
    approximately i.i.d., which breaks down for assets with strong serial
    correlation.

    Parameters
    ----------
    ret
        Bar-level return series.
    ppy
        Periods per year — must match the bar frequency of *ret*.
        Use 252 for daily, 52 for weekly, 12 for monthly,
        ``252 * 390`` for minute bars, etc.

    Returns
    -------
    float
        Annualized volatility as a decimal.
    """
    if len(ret) < 2:
        return 0.0
    return float(ret.std()) * math.sqrt(ppy)  # type: ignore[arg-type]


def avg_drawdown(dd: pl.Series) -> float:
    """Average of drawdown values during drawdown episodes.

    While max drawdown captures the worst single decline, average drawdown
    gives a sense of the *typical* underwater experience.  A strategy with a
    deep max drawdown but shallow average drawdown had one bad episode;
    one where both are close had persistently poor periods.

    Only bars where the drawdown is strictly negative are included.  Returns
    0.0 when the series is always at or above its high-water mark.

    Parameters
    ----------
    dd
        Pre-computed drawdown series (output of ``drawdown_series``).
        Values should be <= 0.

    Returns
    -------
    float
        Mean drawdown (a negative number, or 0.0 if no drawdown occurred).
    """
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
    """Annualized Sharpe ratio.

    The Sharpe ratio measures risk-adjusted return: how much excess return
    (above the risk-free rate) you earn per unit of total volatility.

    :math:`S = \\frac{\\bar{r} - r_f}{\\sigma} \\cdot \\sqrt{N}`

    A Sharpe above 1.0 is generally considered good; above 2.0 is excellent.
    Values near or below 0 indicate the strategy barely beats (or
    underperforms) the risk-free rate after accounting for volatility.

    The risk-free rate *rf* is specified as an **annual** rate and is
    converted internally to a per-bar rate by dividing by *ppy*.

    Parameters
    ----------
    ret
        Bar-level return series.
    ppy
        Periods per year — must match the bar frequency of *ret*.
        Use 252 for daily, 52 for weekly, 12 for monthly,
        ``252 * 390`` for minute bars, etc.
    rf
        Annual risk-free rate (e.g. 0.05 for 5 %).

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    if len(ret) < 2:
        return 0.0
    rf_daily = rf / ppy
    excess = ret - rf_daily
    std = float(excess.std())  # type: ignore[arg-type]
    if std == 0:
        return 0.0
    return float(excess.mean()) / std * math.sqrt(ppy)  # type: ignore[arg-type]


def sortino(ret: pl.Series, ppy: int = 252, rf: float = 0.0) -> float:
    """Annualized Sortino ratio.

    Like the Sharpe ratio, but penalizes only *downside* volatility instead
    of total volatility.  This is more appropriate when the return
    distribution is skewed, because upside surprises should not count
    against a strategy.

    :math:`\\text{Sortino} = \\frac{\\bar{r} - r_f}{\\sigma_{\\text{down}}} \\cdot \\sqrt{N}`

    where :math:`\\sigma_{\\text{down}} = \\sqrt{\\text{mean}(\\min(r - r_f, 0)^2)}`.

    A Sortino of 2.0+ is strong.  Because it ignores upside variance, the
    Sortino is typically higher than the Sharpe for positively skewed
    strategies.

    Parameters
    ----------
    ret
        Bar-level return series.
    ppy
        Periods per year — must match the bar frequency of *ret*.
        Use 252 for daily, 52 for weekly, 12 for monthly,
        ``252 * 390`` for minute bars, etc.
    rf
        Annual risk-free rate (e.g. 0.05 for 5 %).

    Returns
    -------
    float
        Annualized Sortino ratio.
    """
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
    """Omega ratio.

    The Omega ratio is the probability-weighted ratio of gains over losses
    relative to a threshold (the per-bar risk-free rate).  Unlike Sharpe and
    Sortino, it captures the *entire* return distribution — all moments,
    not just mean and variance.

    :math:`\\Omega = \\frac{\\sum \\max(r_i - \\tau, 0)}{\\sum \\max(\\tau - r_i, 0)}`

    An Omega of 1.0 means gains and losses are balanced.  Values above 1
    indicate net positive excess returns; higher is better.  Returns
    ``inf`` when there are gains but zero losses, and 0.0 for an empty
    series.

    Parameters
    ----------
    ret
        Bar-level return series.
    ppy
        Periods per year — must match the bar frequency of *ret*.
        Used only to convert the annual *rf* to a per-bar threshold.
    rf
        Annual risk-free rate (e.g. 0.05 for 5 %).

    Returns
    -------
    float
        Omega ratio (>= 0; ``inf`` when losses are zero).
    """
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
    """Value at Risk at the *alpha* confidence level.

    VaR answers "what is the worst return I can expect in all but the worst
    *alpha* fraction of bars?"  At ``alpha=0.05``, this is the 5th-percentile
    return — on 95 % of bars, the return was at least this good.

    :math:`\\text{VaR}_{\\alpha} = Q_{\\alpha}(r)`

    The result is typically negative.  A VaR of −0.02 at ``alpha=0.05`` means
    daily losses exceeded 2 % only about 5 % of the time.

    This implementation uses historical (non-parametric) quantile estimation
    with linear interpolation.

    Parameters
    ----------
    ret
        Bar-level return series.
    alpha
        Tail probability (default 0.05 = 5th percentile).

    Returns
    -------
    float
        The return at the *alpha* quantile (usually negative).
    """
    if len(ret) == 0:
        return 0.0
    return float(ret.quantile(alpha, interpolation="linear"))  # type: ignore[arg-type]


def cvar(ret: pl.Series, alpha: float = 0.05) -> float:
    """Conditional Value at Risk (Expected Shortfall) at *alpha*.

    CVaR answers "when losses do exceed VaR, how bad are they on average?"
    It is the mean of all returns at or below the VaR threshold and is
    always at least as extreme as VaR itself.

    :math:`\\text{CVaR}_{\\alpha} = E[r \\mid r \\leq \\text{VaR}_{\\alpha}]`

    CVaR is preferred over VaR by many risk frameworks (including Basel III)
    because it is *coherent* — it does not underestimate the risk of
    concentrated tail events.

    A CVaR of −0.035 at ``alpha=0.05`` means that in the worst 5 % of bars,
    the average loss was 3.5 %.

    Parameters
    ----------
    ret
        Bar-level return series.
    alpha
        Tail probability (default 0.05 = 5th percentile).

    Returns
    -------
    float
        Mean return in the worst *alpha* fraction of bars (usually negative).
    """
    if len(ret) == 0:
        return 0.0
    threshold = ret.quantile(alpha, interpolation="linear")
    tail = ret.filter(ret <= threshold)
    if len(tail) == 0:
        return float(threshold)  # type: ignore[arg-type]
    return float(tail.mean())  # type: ignore[arg-type]


def win_rate(ret: pl.Series) -> float:
    """Fraction of positive-return bars.

    Win rate alone says little about profitability — a strategy can win on
    90 % of bars but still lose money if the average loss far exceeds the
    average gain.  Always pair with ``payoff_ratio`` or ``profit_factor``
    for a complete picture.

    Returns a value in [0, 1].  A win rate of 0.55 means 55 % of bars had
    a positive return.
    """
    if len(ret) == 0:
        return 0.0
    return float((ret > 0).sum()) / len(ret)


def payoff_ratio(ret: pl.Series) -> float:
    """Average win divided by average loss.

    Measures the magnitude of the typical winning bar relative to the
    typical losing bar.

    :math:`\\text{payoff} = \\frac{\\text{mean}(r^+)}{|\\text{mean}(r^-)|}`

    A payoff ratio above 1.0 means winners are larger than losers on
    average.  Combined with ``win_rate``, it determines whether a strategy
    is profitable: a low win rate can still be profitable if the payoff
    ratio is high enough (trend-following), and vice versa
    (mean-reversion).

    Returns 0.0 when there are no wins or no losses.
    """
    wins = ret.filter(ret > 0)
    losses = ret.filter(ret < 0)
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    avg_loss = abs(float(losses.mean()))  # type: ignore[arg-type]
    if avg_loss == 0:
        return 0.0
    return float(wins.mean()) / avg_loss  # type: ignore[arg-type]


def profit_factor(ret: pl.Series) -> float:
    """Sum of gains divided by sum of losses.

    While ``payoff_ratio`` compares averages, profit factor compares
    *totals* — it answers "for every dollar lost, how many dollars were
    gained?"

    :math:`\\text{PF} = \\frac{\\sum r^+}{|\\sum r^-|}`

    A profit factor above 1.0 means the strategy is net profitable.  Values
    above 2.0 are strong.  Returns ``inf`` when there are gains but no
    losses, and 0.0 for an empty or all-negative series.
    """
    gains = float(ret.filter(ret > 0).sum())
    losses = abs(float(ret.filter(ret < 0).sum()))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def kelly_criterion(ret: pl.Series) -> float:
    """Kelly criterion from bar-level returns.

    The Kelly criterion gives the theoretically optimal fraction of capital
    to risk per bar in order to maximize the long-run geometric growth rate,
    assuming i.i.d. returns.

    :math:`K = w - \\frac{1 - w}{P}`

    where *w* is the win rate and *P* is the payoff ratio.

    A positive Kelly means the edge is positive; the magnitude suggests how
    aggressively to size positions (though practitioners typically use a
    fraction of full Kelly — "half Kelly" — to reduce variance).  A
    negative value means the strategy has negative expected geometric growth.

    Returns 0.0 when ``payoff_ratio`` is zero (no wins or no losses).
    """
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
