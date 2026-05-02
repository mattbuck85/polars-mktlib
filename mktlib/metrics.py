from __future__ import annotations

import collections
import enum
import math
import statistics
from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from mktlib.data import Innovations


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


_VarMethod = Literal["historical", "gaussian", "monte_carlo"]


def _norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF via :class:`statistics.NormalDist`."""
    return statistics.NormalDist(0.0, 1.0).inv_cdf(p)


def _gbm_log_return_moments(ret: pl.Series, dt: float) -> tuple[float, float]:
    """Fit GBM log-return drift / volatility from a simple-return series.

    Returns ``(mu_hat, sigma_hat)`` such that
    ``log(1 + r_t) ~ N(mu_hat * dt, sigma_hat**2 * dt)`` empirically.
    """
    log_r = (ret + 1.0).log()
    mu = float(log_r.mean()) / dt  # type: ignore[arg-type]
    sigma = float(log_r.std()) / math.sqrt(dt)  # type: ignore[arg-type]
    return mu, sigma


def _standardized_residuals(ret: pl.Series) -> pl.Series:
    """Centre + scale to unit variance.  Used as bootstrap noise."""
    log_r = (ret + 1.0).log()
    mean = float(log_r.mean())  # type: ignore[arg-type]
    std = float(log_r.std())  # type: ignore[arg-type]
    if std == 0.0:
        return pl.Series("residuals", [0.0] * len(log_r), dtype=pl.Float64)
    return ((log_r - mean) / std).alias("residuals")


def _make_mc_cache_key(
    ret: pl.Series,
    *,
    mu: float,
    sigma: float,
    horizon: int,
    n_simulations: int,
    dt: float,
    innovations: "Innovations",
    df: float | None,
    seed: int | None,
) -> tuple:
    """Build the canonical cache key for an MC GBM batch.

    Content-based fingerprint of *ret* — robust to wrapper-object
    identity, so two distinct ``pl.Series`` objects over the same data
    hit the same cache entry.  Both ``_monte_carlo_horizon_returns``
    and ``monte_carlo_paths`` call this so the keys can never drift apart.

    The fingerprint includes ``(len, sum)`` plus four positional
    samples — head, 25th, 50th, 75th, tail — so any pair of series with
    different middle data produce different keys (collision probability
    ~zero for any realistic financial series).  Each ``.item()`` is
    O(1) on a polars Series, so the seven-field fingerprint adds
    microseconds vs. the 10–100ms it gates.
    """
    n = len(ret)
    if n == 0:
        fingerprint = (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    elif n < 4:
        # Series too short for distinct mid-samples — use head/tail only.
        head = float(ret.head(1).item())
        tail = float(ret.tail(1).item())
        fingerprint = (
            n, float(ret.sum()),  # type: ignore[arg-type]
            head, head, head, head, tail,
        )
    else:
        fingerprint = (
            n,
            float(ret.sum()),  # type: ignore[arg-type]
            float(ret.head(1).item()),
            float(ret[n // 4]),
            float(ret[n // 2]),
            float(ret[3 * n // 4]),
            float(ret.tail(1).item()),
        )
    return (
        fingerprint, mu, sigma, horizon, n_simulations, dt,
        innovations.value, df, seed,
    )


def _monte_carlo_horizon_returns(
    ret: pl.Series,
    *,
    horizon: int,
    n_simulations: int,
    dt: float,
    innovations: "Innovations | None",
    df: float | None,
    seed: int | None,
    cache: bool = False,
) -> pl.Series:
    """Compute per-simulation horizon returns under fitted GBM.

    Optional FIFO cache (``_MC_CACHE_MAX = 8``) keyed on a fingerprint
    of ``ret`` plus the MC parameters; enabled only when ``cache=True``.
    Disabled by default — single-shot ``var`` / ``cvar`` calls don't pay
    the cache bookkeeping; reporting code that loops over both metrics
    with identical args opts in to share one batch.

    Bootstrap residuals are derived from ``ret`` directly when
    ``innovations=Innovations.BOOTSTRAP``.
    """
    from mktlib.data import Innovations as _Innovations
    from mktlib.data import Process, monte_carlo

    inn = innovations if innovations is not None else _Innovations.GAUSSIAN
    mu, sigma = _gbm_log_return_moments(ret, dt) if len(ret) > 0 else (0.0, 0.0)

    residuals: pl.Series | None = None
    if inn is _Innovations.BOOTSTRAP:
        residuals = _standardized_residuals(ret)

    cache_key: tuple | None = None
    if cache:
        key = _make_mc_cache_key(
            ret, mu=mu, sigma=sigma, horizon=horizon,
            n_simulations=n_simulations, dt=dt,
            innovations=inn, df=df, seed=seed,
        )
        cache_key = key
        cached = _mc_cache.get(key)
        if cached is not None:
            return cached

    sims = monte_carlo(
        Process.GBM,
        n_simulations=n_simulations,
        # n is the number of price points; horizon SDE steps require horizon+1
        # points (including the base_price at step 0).
        n=horizon + 1,
        base_price=1.0,
        drift=mu,
        volatility=sigma,
        dt=dt,
        seed=seed,
        innovations=inn,
        df=df,
        residuals=residuals,
        # Single-batch sampling is statistically identical (i.i.d. by
        # construction) and 5–200× faster.  The metrics layer never
        # introspects per-simulation seeds, so the only cosmetic
        # property lost (the seed column reporting one shared parent
        # seed) doesn't affect any downstream metric.
        independent_streams=False,
    )
    horizon_returns = (
        sims.group_by("simulation")
        .agg((pl.col("price").last() - 1.0).alias("R"))
        .get_column("R")
    )
    if cache_key is not None:
        _mc_cache_insert(cache_key, horizon_returns)
    return horizon_returns


_MC_CACHE_MAX = 8
_mc_cache: dict[tuple, pl.Series] = {}
_mc_cache_order: collections.deque[tuple] = collections.deque(maxlen=_MC_CACHE_MAX)


def _mc_cache_insert(key: tuple, value: pl.Series) -> None:
    if key in _mc_cache:
        return
    if len(_mc_cache_order) == _MC_CACHE_MAX:
        # deque is full — appending would silently discard the oldest;
        # mirror that on the dict side before extending.
        _mc_cache.pop(_mc_cache_order[0], None)
    _mc_cache_order.append(key)
    _mc_cache[key] = value


def _mc_cache_clear() -> None:
    _mc_cache.clear()
    _mc_cache_order.clear()


def monte_carlo_paths(
    ret: pl.Series,
    *,
    horizon: int,
    n_simulations: int = 10_000,
    dt: float = 1 / 252,
    innovations: "Innovations | None" = None,
    df: float | None = None,
    seed: int | None = None,
) -> pl.DataFrame:
    """Run one MC GBM batch fitted to *ret*; return the full simulation frame.

    Companion entry point for callers — typically reporting code — that need
    both the per-simulation horizon-end returns (for VaR / CVaR) and the
    full price paths (for charting).  This function:

    1. Fits ``mu_hat, sigma_hat`` from the log-returns of *ret*.
    2. Derives standardized residuals when ``innovations=Innovations.BOOTSTRAP``.
    3. Runs ``monte_carlo(Process.GBM, ..., independent_streams=False)``
       — the perf path; statistically i.i.d. by construction.
    4. Computes the per-simulation horizon return and **inserts it into
       the module-level MC cache** under the canonical key
       (``_make_mc_cache_key``).  Subsequent calls to
       :func:`var` / :func:`cvar` / :func:`simulate_metric` with
       ``method="monte_carlo"``, ``cache=True``, and matching kwargs
       hit that cache and skip a second simulation.

    Parameters
    ----------
    ret
        Historical bar-level return series used to fit the GBM model.
    horizon
        Forecast horizon in bars (number of SDE steps; the returned
        frame has ``horizon + 1`` price points per simulation including
        the base value at step 0).
    n_simulations, dt, innovations, df, seed
        Forwarded to :func:`mktlib.data.monte_carlo`.  See
        :class:`mktlib.data.Innovations` for the innovation contract.

    Returns
    -------
    pl.DataFrame
        Long-form frame with columns ``simulation``, ``seed``, ``step``,
        ``price`` (base = 1.0; multiply by initial portfolio equity for
        absolute units).

    Notes
    -----
    The cache key uses a content-based fingerprint of *ret* so two
    distinct ``pl.Series`` wrappers over the same data hit the same
    entry — important for callers that re-bind ``df["return"]`` between
    this call and a downstream :func:`var` call.
    """
    from mktlib.data import Innovations as _Innovations
    from mktlib.data import Process, monte_carlo

    inn = innovations if innovations is not None else _Innovations.GAUSSIAN
    mu, sigma = (
        _gbm_log_return_moments(ret, dt) if len(ret) > 0 else (0.0, 0.0)
    )

    residuals: pl.Series | None = None
    if inn is _Innovations.BOOTSTRAP:
        residuals = _standardized_residuals(ret)

    sims = monte_carlo(
        Process.GBM,
        n_simulations=n_simulations,
        n=horizon + 1,
        base_price=1.0,
        drift=mu,
        volatility=sigma,
        dt=dt,
        seed=seed,
        innovations=inn,
        df=df,
        residuals=residuals,
        independent_streams=False,
    )

    horizon_returns = (
        sims.group_by("simulation")
        .agg((pl.col("price").last() - 1.0).alias("R"))
        .get_column("R")
    )
    cache_key = _make_mc_cache_key(
        ret, mu=mu, sigma=sigma, horizon=horizon,
        n_simulations=n_simulations, dt=dt,
        innovations=inn, df=df, seed=seed,
    )
    _mc_cache_insert(cache_key, horizon_returns)
    return sims


def var(
    ret: pl.Series,
    alpha: float = 0.05,
    *,
    method: _VarMethod = "historical",
    horizon: int = 1,
    n_simulations: int = 10_000,
    dt: float = 1 / 252,
    innovations: "Innovations | None" = None,
    df: float | None = None,
    seed: int | None = None,
    cache: bool = False,
) -> float:
    """Value at Risk at the *alpha* confidence level.

    VaR answers "what is the worst return I can expect in all but the worst
    *alpha* fraction of bars?"  At ``alpha=0.05``, this is the 5th-percentile
    return — on 95 % of bars, the return was at least this good.

    :math:`\\text{VaR}_{\\alpha} = Q_{\\alpha}(r)`

    The result is typically negative.  A VaR of −0.02 at ``alpha=0.05`` means
    losses exceeded 2 % only about 5 % of the time.

    Parameters
    ----------
    ret
        Bar-level return series (simple returns; e.g. close-to-close pct
        changes).
    alpha
        Tail probability (default 0.05 = 5th percentile).
    method
        - ``"historical"`` (default) — empirical quantile of *ret*.
        - ``"gaussian"`` — closed-form parametric VaR under GBM, fitted from
          *ret*.  Uses :math:`\\mu \\cdot H \\Delta t + \\sigma \\sqrt{H \\Delta t}\\,\\Phi^{-1}(\\alpha)`.
        - ``"monte_carlo"`` — simulation-based; required when
          *innovations* is non-Gaussian or when path-dependent
          extensions are added.  Under ``Innovations.GAUSSIAN`` and
          ``horizon=1`` the result matches ``"gaussian"`` modulo
          sampling noise — *the simulation pays compute for nothing
          there*.
    horizon
        Forecast horizon in bars.  Only used by ``"gaussian"`` /
        ``"monte_carlo"``.  At ``horizon=1`` MC under Gaussian
        innovations returns the closed-form Gaussian VaR.
    n_simulations
        Monte Carlo path count.  Rule of thumb: ``n_simulations >= 200/alpha``
        keeps the CVaR tail well-populated.
    dt
        Time step for the SDE (default ``1/252`` ≈ daily under the standard
        252-trading-day year).
    innovations
        Noise distribution for ``method="monte_carlo"``.  Defaults to
        ``Innovations.GAUSSIAN``.  See :class:`mktlib.data.Innovations`.
    df
        Degrees of freedom for ``Innovations.STUDENT_T`` (must be > 2).
    seed
        RNG seed for reproducibility (Monte Carlo path only).
    cache
        If ``True``, share the Monte Carlo simulation batch across
        repeated calls with identical arguments via a module-level FIFO
        cache (``_MC_CACHE_MAX = 8``).  Default ``False`` — single-shot
        ad-hoc calls don't pay the bookkeeping.  Reporting code that
        loops over both :func:`var` and :func:`cvar` (e.g. tearsheet
        generation) should opt in so the simulation runs once.

    Returns
    -------
    float
        The α-quantile return (usually negative).
    """
    if len(ret) == 0:
        return 0.0
    if method == "historical":
        return float(ret.quantile(alpha, interpolation="linear"))  # type: ignore[arg-type]
    if method == "gaussian":
        mu, sigma = _gbm_log_return_moments(ret, dt)
        h_dt = horizon * dt
        log_q = mu * h_dt + sigma * math.sqrt(h_dt) * _norm_ppf(alpha)
        return math.expm1(log_q)
    if method == "monte_carlo":
        horizon_returns = _monte_carlo_horizon_returns(
            ret,
            horizon=horizon,
            n_simulations=n_simulations,
            dt=dt,
            innovations=innovations,
            df=df,
            seed=seed,
            cache=cache,
        )
        return float(horizon_returns.quantile(alpha, interpolation="linear"))  # type: ignore[arg-type]
    raise ValueError(f"unknown var method: {method!r}")  # pyright: ignore


def cvar(
    ret: pl.Series,
    alpha: float = 0.05,
    *,
    method: _VarMethod = "historical",
    horizon: int = 1,
    n_simulations: int = 10_000,
    dt: float = 1 / 252,
    innovations: "Innovations | None" = None,
    df: float | None = None,
    seed: int | None = None,
    cache: bool = False,
) -> float:
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
    Same as :func:`var`.

    Returns
    -------
    float
        Mean return in the worst *alpha* fraction (usually negative).
    """
    if len(ret) == 0:
        return 0.0
    if method == "historical":
        threshold = ret.quantile(alpha, interpolation="linear")
        tail = ret.filter(ret <= threshold)
        if len(tail) == 0:
            return float(threshold)  # type: ignore[arg-type]
        return float(tail.mean())  # type: ignore[arg-type]
    if method == "gaussian":
        mu, sigma = _gbm_log_return_moments(ret, dt)
        h_dt = horizon * dt
        z_alpha = _norm_ppf(alpha)
        phi_z = math.exp(-0.5 * z_alpha * z_alpha) / math.sqrt(2.0 * math.pi)
        log_es = mu * h_dt - sigma * math.sqrt(h_dt) * (phi_z / alpha)
        return math.expm1(log_es)
    if method == "monte_carlo":
        horizon_returns = _monte_carlo_horizon_returns(
            ret,
            horizon=horizon,
            n_simulations=n_simulations,
            dt=dt,
            innovations=innovations,
            df=df,
            seed=seed,
            cache=cache,
        )
        threshold = horizon_returns.quantile(alpha, interpolation="linear")
        tail = horizon_returns.filter(horizon_returns <= threshold)
        if len(tail) == 0:
            return float(threshold)  # type: ignore[arg-type]
        return float(tail.mean())  # type: ignore[arg-type]
    raise ValueError(f"unknown cvar method: {method!r}")  # pyright: ignore


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
    """Compute a single metric by enum value (historical / empirical only).

    For forward-looking parametric estimators of VaR / CVaR
    (``"gaussian"``, ``"monte_carlo"``), use :func:`simulate_metric`
    instead.  Splitting the two keeps this dispatcher cheap and free
    of simulation kwargs.

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


_SIMULATABLE_METRICS = frozenset({Metric.VAR, Metric.CVAR})


def simulate_metric(
    metric: Metric,
    ret: pl.Series,
    *,
    alpha: float = 0.05,
    method: _VarMethod = "monte_carlo",
    horizon: int = 1,
    n_simulations: int = 10_000,
    dt: float = 1 / 252,
    innovations: "Innovations | None" = None,
    df: float | None = None,
    seed: int | None = None,
    cache: bool = False,
) -> float:
    """Forward-looking parametric estimator for VaR / CVaR.

    Companion to :func:`calculate_metric` — this dispatcher hosts the
    simulation-aware estimators (``"gaussian"`` closed-form, ``"monte_carlo"``)
    so that :func:`calculate_metric` stays lean for the historical /
    empirical metric set.

    Currently supports :data:`Metric.VAR` and :data:`Metric.CVAR`.  Other
    members raise :class:`ValueError` — no simulatable definition exists
    for them in this release.

    Parameters
    ----------
    metric
        Which metric to simulate.  Must be one of ``Metric.VAR`` or
        ``Metric.CVAR``.
    ret
        Historical bar-level return series used to fit the parametric
        model (μ̂, σ̂ via :func:`_gbm_log_return_moments`; standardized
        residuals for ``Innovations.BOOTSTRAP``).
    alpha
        Tail probability (default ``0.05``).
    method
        ``"gaussian"`` for the closed-form parametric estimator or
        ``"monte_carlo"`` (default) for the simulation-based path.
        ``"historical"`` is rejected — call :func:`calculate_metric`
        for that.
    horizon, n_simulations, dt, innovations, df, seed
        Forwarded to :func:`var` / :func:`cvar`.
    cache
        Forwarded to :func:`var` / :func:`cvar`.  Default ``False``;
        set to ``True`` from reporting code that calls both VaR and
        CVaR with identical args so the underlying simulation batch is
        reused (``_MC_CACHE_MAX = 8``, FIFO-evicted).

    Returns
    -------
    float
        VaR or CVaR at the requested confidence and horizon.
    """
    if metric not in _SIMULATABLE_METRICS:
        raise ValueError(
            f"simulate_metric supports only {', '.join(m.name for m in _SIMULATABLE_METRICS)};"
            f" got {metric.name}.  Use calculate_metric for empirical metrics."
        )
    if method == "historical":
        raise ValueError(
            'simulate_metric does not accept method="historical"; '
            "use calculate_metric for empirical estimates."
        )
    kwargs: dict[str, object] = {
        "method": method,
        "horizon": horizon,
        "n_simulations": n_simulations,
        "dt": dt,
        "innovations": innovations,
        "df": df,
        "seed": seed,
        "cache": cache,
    }
    if metric is Metric.VAR:
        return var(ret, alpha, **kwargs)  # type: ignore[arg-type]
    return cvar(ret, alpha, **kwargs)  # type: ignore[arg-type]
