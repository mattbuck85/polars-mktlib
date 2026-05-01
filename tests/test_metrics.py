from __future__ import annotations

import datetime as dt
import math

import polars as pl
import pytest

from mktlib.metrics import (
    Metric,
    avg_drawdown,
    cagr,
    calculate_metric,
    cumulative_return,
    annualized_volatility,
    cvar,
    drawdown_series,
    kelly_criterion,
    longest_drawdown_days,
    omega,
    payoff_ratio,
    profit_factor,
    sharpe,
    simulate_metric,
    sortino,
    var,
    win_rate,
)


@pytest.fixture
def daily_ret() -> pl.Series:
    """252 days of returns with slight variation."""
    return pl.Series("r", [0.001 + 0.002 * math.sin(i * 0.5) for i in range(252)])


@pytest.fixture
def dates_series() -> pl.Series:
    """Business-day dates for 252 bars."""
    dates: list[dt.date] = []
    current = dt.date(2024, 1, 2)
    for _ in range(252):
        while current.weekday() >= 5:
            current += dt.timedelta(days=1)
        dates.append(current)
        current += dt.timedelta(days=1)
    return pl.Series("date", dates)


@pytest.fixture
def empty_ret() -> pl.Series:
    return pl.Series("r", [], dtype=pl.Float64)


# ---------------------------------------------------------------------------
# Individual function tests
# ---------------------------------------------------------------------------


class TestCumulativeReturn:
    def test_compounded(self, daily_ret: pl.Series) -> None:
        result = cumulative_return(daily_ret, compounded=True)
        assert isinstance(result, float)
        assert result > 0

    def test_simple(self, daily_ret: pl.Series) -> None:
        result = cumulative_return(daily_ret, compounded=False)
        assert isinstance(result, float)
        assert result == pytest.approx(float(daily_ret.sum()), abs=1e-10)

    def test_empty(self, empty_ret: pl.Series) -> None:
        assert cumulative_return(empty_ret) == 0.0


class TestCAGR:
    def test_one_year(self, daily_ret: pl.Series) -> None:
        result = cagr(daily_ret)
        assert isinstance(result, float)
        # 252 bars / 252 ppy = 1 year → CAGR ≈ cumulative
        cum = cumulative_return(daily_ret)
        assert result == pytest.approx(cum, rel=1e-4)

    def test_empty(self, empty_ret: pl.Series) -> None:
        assert cagr(empty_ret) == 0.0


class TestAnnualizedVolatility:
    def test_positive(self, daily_ret: pl.Series) -> None:
        result = annualized_volatility(daily_ret)
        assert isinstance(result, float)
        assert result > 0

    def test_single_bar(self) -> None:
        assert annualized_volatility(pl.Series("r", [0.01])) == 0.0


class TestDrawdownSeries:
    def test_basic(self) -> None:
        ret = pl.Series("r", [0.10, -0.20])
        dd = drawdown_series(ret, compounded=True)
        assert dd[0] == pytest.approx(0.0)
        assert dd[1] == pytest.approx(0.88 / 1.1 - 1, abs=1e-6)

    def test_no_drawdown(self) -> None:
        ret = pl.Series("r", [0.01, 0.02, 0.01])
        dd = drawdown_series(ret)
        assert dd.min() == pytest.approx(0.0)

    def test_empty(self, empty_ret: pl.Series) -> None:
        dd = drawdown_series(empty_ret)
        assert len(dd) == 0


class TestAvgDrawdown:
    def test_with_drawdown(self) -> None:
        dd = pl.Series("dd", [0.0, -0.05, -0.10, 0.0])
        result = avg_drawdown(dd)
        assert isinstance(result, float)
        assert result == pytest.approx((-0.05 + -0.10) / 2, abs=1e-10)

    def test_no_drawdown(self) -> None:
        dd = pl.Series("dd", [0.0, 0.0, 0.0])
        assert avg_drawdown(dd) == 0.0


class TestLongestDrawdownDays:
    def test_basic(self) -> None:
        dd = pl.Series("dd", [0.0, -0.05, -0.10, -0.03, 0.0])
        dates = pl.Series("date", [
            dt.date(2024, 1, 1),
            dt.date(2024, 1, 2),
            dt.date(2024, 1, 3),
            dt.date(2024, 1, 4),
            dt.date(2024, 1, 5),
        ])
        result = longest_drawdown_days(dd, dates)
        assert isinstance(result, float)
        assert result == 2.0  # Jan 2 to Jan 4

    def test_empty(self) -> None:
        dd = pl.Series("dd", [], dtype=pl.Float64)
        dates = pl.Series("date", [], dtype=pl.Date)
        assert longest_drawdown_days(dd, dates) == 0.0


class TestSharpe:
    def test_positive(self, daily_ret: pl.Series) -> None:
        result = sharpe(daily_ret)
        assert isinstance(result, float)
        assert result > 0

    def test_empty(self, empty_ret: pl.Series) -> None:
        assert sharpe(empty_ret) == 0.0


class TestSortino:
    def test_positive(self, daily_ret: pl.Series) -> None:
        result = sortino(daily_ret)
        assert isinstance(result, float)
        assert result > 0


class TestOmega:
    def test_basic(self) -> None:
        ret = pl.Series("r", [0.02, -0.01, 0.03, -0.01])
        result = omega(ret, rf=0.0)
        # gains = 0.02+0.03 = 0.05, losses = 0.01+0.01 = 0.02
        assert result == pytest.approx(2.5, rel=1e-4)

    def test_empty(self, empty_ret: pl.Series) -> None:
        assert omega(empty_ret) == 0.0


class TestVaR:
    def test_negative(self) -> None:
        ret = pl.Series("r", [-0.05, -0.03, 0.01, 0.02, 0.015, -0.04, 0.005, -0.02, 0.01, -0.01])
        result = var(ret)
        assert isinstance(result, float)
        assert result < 0

    def test_empty(self, empty_ret: pl.Series) -> None:
        assert var(empty_ret) == 0.0


class TestCVaR:
    def test_lte_var(self) -> None:
        ret = pl.Series("r", [-0.05, -0.03, 0.01, 0.02, 0.015, -0.04, 0.005, -0.02, 0.01, -0.01])
        assert cvar(ret) <= var(ret)

    def test_empty(self, empty_ret: pl.Series) -> None:
        assert cvar(empty_ret) == 0.0


class TestVaRMethods:
    """Method-flag dispatch for var()."""

    def test_historical_default_unchanged(self) -> None:
        """Calling without method= must yield the old historical number."""
        ret = pl.Series(
            "r",
            [-0.05, -0.03, 0.01, 0.02, 0.015, -0.04, 0.005, -0.02, 0.01, -0.01],
        )
        # Historical 5th percentile via linear interpolation
        expected = float(ret.quantile(0.05, interpolation="linear"))  # type: ignore[arg-type]
        assert var(ret) == pytest.approx(expected, rel=1e-12)
        assert var(ret, method="historical") == pytest.approx(expected, rel=1e-12)

    def test_gaussian_closed_form(self) -> None:
        """Gaussian VaR matches the analytic μ·Hdt + σ·√(Hdt)·Φ⁻¹(α) form."""
        import statistics

        ret = pl.Series("r", [0.001 + 0.002 * math.sin(i * 0.5) for i in range(252)])
        dt = 1 / 252
        log_r = (ret + 1.0).log()
        mu = float(log_r.mean()) / dt  # type: ignore[arg-type]
        sigma = float(log_r.std()) / math.sqrt(dt)  # type: ignore[arg-type]
        z = statistics.NormalDist(0.0, 1.0).inv_cdf(0.05)
        expected = math.expm1(mu * dt + sigma * math.sqrt(dt) * z)
        assert var(ret, method="gaussian", horizon=1, dt=dt) == pytest.approx(
            expected, rel=1e-12
        )

    def test_gaussian_horizon_scaling(self) -> None:
        """At zero drift, log-VaR's volatility term scales with √H."""
        # Synthetic zero-drift series with measurable vol so the √H scaling
        # is visible without drift-domination at long H.
        ret = pl.Series(
            "r", [0.02 * math.sin(i * 0.7) - 0.02 * math.sin(i * 0.3) for i in range(252)]
        )
        v_h1 = var(ret, method="gaussian", horizon=1)
        v_h10 = var(ret, method="gaussian", horizon=10)
        # Convert simple-return VaR back to log-VaR for a clean √H check
        log_v1 = math.log1p(v_h1)
        log_v10 = math.log1p(v_h10)
        # log-VaR_H ≈ μ·H·dt + σ·√(H·dt)·z  → at near-zero drift, ratio ≈ √10
        ratio = log_v10 / log_v1
        assert ratio == pytest.approx(math.sqrt(10), rel=0.10)

    def test_unknown_method_raises(self) -> None:
        ret = pl.Series("r", [0.01, -0.01])
        with pytest.raises(Exception):
            var(ret, method="bogus")  # type: ignore[arg-type]


class TestCVaRMethods:
    def test_historical_default_unchanged(self) -> None:
        ret = pl.Series(
            "r",
            [-0.05, -0.03, 0.01, 0.02, 0.015, -0.04, 0.005, -0.02, 0.01, -0.01],
        )
        # Compare against the pre-refactor implementation
        threshold = ret.quantile(0.05, interpolation="linear")
        tail = ret.filter(ret <= threshold)
        expected = float(tail.mean()) if len(tail) > 0 else float(threshold)  # type: ignore[arg-type]
        assert cvar(ret) == pytest.approx(expected, rel=1e-12)

    def test_gaussian_more_extreme_than_var(self) -> None:
        """Gaussian CVaR ≤ Gaussian VaR (coherence)."""
        ret = pl.Series("r", [0.001 + 0.002 * math.sin(i * 0.5) for i in range(252)])
        v = var(ret, method="gaussian")
        c = cvar(ret, method="gaussian")
        assert c <= v


class TestMonteCarloVaRConvergence:
    """MC under Gaussian innovations should converge to the closed-form Gaussian."""

    def test_gaussian_mc_matches_closed_form(self) -> None:
        ret = pl.Series("r", [0.001 + 0.002 * math.sin(i * 0.5) for i in range(252)])
        from mktlib.metrics import _mc_cache_clear

        _mc_cache_clear()
        n_sims = 50_000
        v_closed = var(ret, method="gaussian", horizon=1)
        v_mc = var(
            ret, method="monte_carlo", horizon=1, n_simulations=n_sims, seed=42
        )
        # Sample SE on the 5%-quantile of n iid normals ~ sigma / (alpha * sqrt(n))
        # which at alpha=0.05, n=50k is ~3-5 bps.  Use a generous 30 bp tolerance.
        assert v_mc == pytest.approx(v_closed, abs=3e-3)

    def test_gaussian_mc_cvar_matches_closed_form(self) -> None:
        ret = pl.Series("r", [0.001 + 0.002 * math.sin(i * 0.5) for i in range(252)])
        from mktlib.metrics import _mc_cache_clear

        _mc_cache_clear()
        c_closed = cvar(ret, method="gaussian", horizon=1)
        c_mc = cvar(
            ret, method="monte_carlo", horizon=1, n_simulations=50_000, seed=42
        )
        assert c_mc == pytest.approx(c_closed, abs=3e-3)

    def test_cache_disabled_by_default(self) -> None:
        """Default cache=False — back-to-back MC calls leave the cache empty."""
        from mktlib.metrics import _mc_cache, _mc_cache_clear

        _mc_cache_clear()
        ret = pl.Series("r", [0.001 + 0.002 * math.sin(i * 0.5) for i in range(252)])
        var(ret, method="monte_carlo", n_simulations=2_000, seed=7)
        cvar(ret, method="monte_carlo", n_simulations=2_000, seed=7)
        assert len(_mc_cache) == 0, (
            f"cache should stay empty when cache=False, found {len(_mc_cache)}"
        )

    def test_var_cvar_share_cache_when_opted_in(self) -> None:
        """One MC batch feeds both var() and cvar() when cache=True."""
        from mktlib.metrics import _mc_cache, _mc_cache_clear

        _mc_cache_clear()
        ret = pl.Series("r", [0.001 + 0.002 * math.sin(i * 0.5) for i in range(252)])
        var(ret, method="monte_carlo", n_simulations=5_000, seed=7, cache=True)
        cvar(ret, method="monte_carlo", n_simulations=5_000, seed=7, cache=True)
        assert len(_mc_cache) == 1, (
            f"expected one cached batch, found {len(_mc_cache)}"
        )

    def test_cache_evicts_beyond_max(self) -> None:
        from mktlib.metrics import _MC_CACHE_MAX, _mc_cache, _mc_cache_clear

        _mc_cache_clear()
        ret = pl.Series("r", [0.001 + 0.002 * math.sin(i * 0.5) for i in range(252)])
        for s in range(_MC_CACHE_MAX + 3):
            var(ret, method="monte_carlo", n_simulations=1_000, seed=s, cache=True)
        assert len(_mc_cache) == _MC_CACHE_MAX

    def test_mc_reproducibility(self) -> None:
        from mktlib.metrics import _mc_cache_clear

        _mc_cache_clear()
        ret = pl.Series("r", [0.001 + 0.002 * math.sin(i * 0.5) for i in range(252)])
        v1 = var(ret, method="monte_carlo", seed=42, n_simulations=1_000)
        _mc_cache_clear()
        v2 = var(ret, method="monte_carlo", seed=42, n_simulations=1_000)
        assert v1 == v2


class TestBootstrapResiduals:
    """Bootstrap path standardizes the user's returns to unit-variance noise."""

    def test_standardized_residuals_centered_unit_variance(self) -> None:
        from mktlib.metrics import _standardized_residuals

        ret = pl.Series("r", [-0.05, 0.0, 0.05, 0.02, -0.03])
        res = _standardized_residuals(ret)
        assert float(res.mean()) == pytest.approx(0.0, abs=1e-12)  # type: ignore[arg-type]
        assert float(res.std()) == pytest.approx(1.0, rel=1e-9)  # type: ignore[arg-type]

    def test_zero_variance_input_returns_zeros(self) -> None:
        from mktlib.metrics import _standardized_residuals

        ret = pl.Series("r", [0.0, 0.0, 0.0])
        res = _standardized_residuals(ret)
        assert (res == 0.0).all()

    def test_bootstrap_var_runs(self) -> None:
        from mktlib.data import Innovations
        from mktlib.metrics import _mc_cache_clear

        _mc_cache_clear()
        ret = pl.Series(
            "r", [-0.05, -0.03, 0.01, 0.02, 0.015, -0.04, 0.005, -0.02, 0.01, -0.01] * 25
        )
        v = var(
            ret, method="monte_carlo", innovations=Innovations.BOOTSTRAP,
            horizon=1, n_simulations=5_000, seed=42,
        )
        assert v < 0
        assert math.isfinite(v)


class TestMonteCarloPathsHelper:
    """`monte_carlo_paths()` returns the full sims frame and pre-populates the cache."""

    def test_returns_full_sims_frame(self) -> None:
        from mktlib.metrics import _mc_cache_clear, monte_carlo_paths

        _mc_cache_clear()
        ret = pl.Series("r", [0.001 * (i % 7 - 3) for i in range(100)])
        sims = monte_carlo_paths(ret, horizon=21, n_simulations=500, seed=42)
        assert sims.shape == (500 * 22, 4)
        assert sims.columns == ["simulation", "seed", "step", "price"]
        assert (sims["price"] > 0).all()

    def test_populates_cache_for_var_cvar(self) -> None:
        """Running monte_carlo_paths means the next var/cvar with matching args is a cache hit."""
        from mktlib.metrics import _mc_cache, _mc_cache_clear, monte_carlo_paths

        _mc_cache_clear()
        ret = pl.Series("r", [0.001 * (i % 7 - 3) for i in range(100)])
        monte_carlo_paths(ret, horizon=21, n_simulations=500, seed=42)
        assert len(_mc_cache) == 1
        # var with matching args must NOT grow the cache
        var(
            ret, method="monte_carlo", horizon=21, n_simulations=500,
            seed=42, cache=True,
        )
        cvar(
            ret, method="monte_carlo", horizon=21, n_simulations=500,
            seed=42, cache=True,
        )
        assert len(_mc_cache) == 1

    def test_cache_hit_across_distinct_series_objects(self) -> None:
        """Content-based fingerprint: distinct Series wrappers over same data hit the cache."""
        from mktlib.metrics import _mc_cache, _mc_cache_clear, monte_carlo_paths

        _mc_cache_clear()
        data = [0.001 * (i % 7 - 3) for i in range(100)]
        ret_a = pl.Series("r", data)
        ret_b = pl.Series("r", data)  # distinct wrapper, same content
        assert id(ret_a) != id(ret_b)
        monte_carlo_paths(ret_a, horizon=21, n_simulations=500, seed=42)
        var(
            ret_b, method="monte_carlo", horizon=21, n_simulations=500,
            seed=42, cache=True,
        )
        assert len(_mc_cache) == 1

    def test_bootstrap_parity_driver_vs_metric(self) -> None:
        """Bootstrap residuals derived in monte_carlo_paths == those used by var()."""
        from mktlib.data import Innovations
        from mktlib.metrics import _mc_cache_clear, monte_carlo_paths

        _mc_cache_clear()
        ret = pl.Series("r", [0.001 * (i % 7 - 3) for i in range(100)])
        monte_carlo_paths(
            ret, horizon=21, n_simulations=500, seed=42,
            innovations=Innovations.BOOTSTRAP,
        )
        cached_var = var(
            ret, method="monte_carlo", horizon=21, n_simulations=500,
            seed=42, innovations=Innovations.BOOTSTRAP, cache=True,
        )
        # Direct call without cache pre-populate
        _mc_cache_clear()
        direct_var = var(
            ret, method="monte_carlo", horizon=21, n_simulations=500,
            seed=42, innovations=Innovations.BOOTSTRAP, cache=True,
        )
        assert cached_var == pytest.approx(direct_var, rel=1e-12)


class TestWinRate:
    def test_basic(self) -> None:
        ret = pl.Series("r", [0.01, -0.01, 0.02, -0.02])
        assert win_rate(ret) == pytest.approx(0.5)

    def test_empty(self, empty_ret: pl.Series) -> None:
        assert win_rate(empty_ret) == 0.0


class TestPayoffRatio:
    def test_basic(self) -> None:
        ret = pl.Series("r", [0.02, -0.01, 0.03, -0.01])
        # avg win = 0.025, avg loss = 0.01 → 2.5
        assert payoff_ratio(ret) == pytest.approx(2.5, rel=1e-4)

    def test_no_losses(self) -> None:
        ret = pl.Series("r", [0.01, 0.02])
        assert payoff_ratio(ret) == 0.0


class TestProfitFactor:
    def test_basic(self) -> None:
        ret = pl.Series("r", [0.02, -0.01, 0.03, -0.01])
        # gains=0.05, losses=0.02 → 2.5
        assert profit_factor(ret) == pytest.approx(2.5, rel=1e-4)

    def test_no_losses(self) -> None:
        ret = pl.Series("r", [0.01, 0.02])
        assert profit_factor(ret) == float("inf")


class TestKellyCriterion:
    def test_basic(self) -> None:
        ret = pl.Series("r", [0.02, -0.01, 0.03, -0.01])
        # wr=0.5, payoff=2.5 → 0.5 - 0.5/2.5 = 0.3
        assert kelly_criterion(ret) == pytest.approx(0.3, rel=1e-4)


# ---------------------------------------------------------------------------
# calculate_metric dispatcher
# ---------------------------------------------------------------------------


class TestCalculateMetric:
    def test_all_metrics_return_float(self, daily_ret: pl.Series, dates_series: pl.Series) -> None:
        """Every Metric member returns a float."""
        dd = drawdown_series(daily_ret)
        for m in Metric:
            result = calculate_metric(
                m, daily_ret, dd=dd, dates=dates_series, ppy=252, rf=0.0, alpha=0.05,
            )
            assert isinstance(result, float), f"{m.name} returned {type(result)}"

    def test_drawdown_reuse(self, daily_ret: pl.Series) -> None:
        """Passing dd= produces identical results to auto-compute."""
        dd = drawdown_series(daily_ret)
        for m in (Metric.MAX_DRAWDOWN, Metric.AVG_DRAWDOWN, Metric.CALMAR, Metric.ROMAD):
            with_dd = calculate_metric(m, daily_ret, dd=dd)
            without_dd = calculate_metric(m, daily_ret)
            assert with_dd == pytest.approx(without_dd, abs=1e-12), f"{m.name} mismatch"

    def test_longest_drawdown_days_requires_dates(self, daily_ret: pl.Series) -> None:
        with pytest.raises(ValueError, match="dates="):
            calculate_metric(Metric.LONGEST_DRAWDOWN_DAYS, daily_ret)

    def test_empty_series(self, empty_ret: pl.Series) -> None:
        """All metrics return 0.0 for empty series."""
        for m in Metric:
            if m == Metric.LONGEST_DRAWDOWN_DAYS:
                result = calculate_metric(
                    m, empty_ret, dates=pl.Series("d", [], dtype=pl.Date),
                )
            else:
                result = calculate_metric(m, empty_ret)
            assert result == 0.0, f"{m.name} returned {result} for empty series"


class TestLongestDrawdownDaysFloat:
    """Verify longest_drawdown_days returns float, not int."""

    def test_return_type(self) -> None:
        ret = pl.Series("r", [0.05, 0.05, -0.08, -0.08, 0.03, 0.03])
        dates = pl.Series("date", [
            dt.date(2024, 1, i) for i in range(1, 7)
        ])
        dd = drawdown_series(ret)
        result = calculate_metric(
            Metric.LONGEST_DRAWDOWN_DAYS, ret, dd=dd, dates=dates,
        )
        assert isinstance(result, float)


class TestSimulateMetric:
    """Forward-looking dispatcher for VaR / CVaR."""

    def test_var_gaussian_matches_var_function(self, daily_ret: pl.Series) -> None:
        from mktlib.metrics import _mc_cache_clear

        _mc_cache_clear()
        a = simulate_metric(Metric.VAR, daily_ret, method="gaussian")
        b = var(daily_ret, method="gaussian")
        assert a == pytest.approx(b, rel=1e-12)

    def test_cvar_gaussian_matches_cvar_function(self, daily_ret: pl.Series) -> None:
        from mktlib.metrics import _mc_cache_clear

        _mc_cache_clear()
        a = simulate_metric(Metric.CVAR, daily_ret, method="gaussian")
        b = cvar(daily_ret, method="gaussian")
        assert a == pytest.approx(b, rel=1e-12)

    def test_var_mc_matches_var_function(self, daily_ret: pl.Series) -> None:
        from mktlib.metrics import _mc_cache_clear

        _mc_cache_clear()
        a = simulate_metric(
            Metric.VAR, daily_ret, method="monte_carlo",
            n_simulations=2_000, seed=42,
        )
        _mc_cache_clear()
        b = var(daily_ret, method="monte_carlo", n_simulations=2_000, seed=42)
        assert a == pytest.approx(b, rel=1e-12)

    def test_cache_disabled_by_default(self, daily_ret: pl.Series) -> None:
        """simulate_metric default cache=False keeps the cache empty."""
        from mktlib.metrics import _mc_cache, _mc_cache_clear

        _mc_cache_clear()
        simulate_metric(
            Metric.VAR, daily_ret, method="monte_carlo",
            n_simulations=2_000, seed=42,
        )
        simulate_metric(
            Metric.CVAR, daily_ret, method="monte_carlo",
            n_simulations=2_000, seed=42,
        )
        assert len(_mc_cache) == 0

    def test_var_cvar_share_cache_when_opted_in(self, daily_ret: pl.Series) -> None:
        """VaR + CVaR via simulate_metric reuse a single MC batch when cache=True."""
        from mktlib.metrics import _mc_cache, _mc_cache_clear

        _mc_cache_clear()
        simulate_metric(
            Metric.VAR, daily_ret, method="monte_carlo",
            n_simulations=2_000, seed=42, cache=True,
        )
        simulate_metric(
            Metric.CVAR, daily_ret, method="monte_carlo",
            n_simulations=2_000, seed=42, cache=True,
        )
        assert len(_mc_cache) == 1

    def test_rejects_non_simulatable_metric(self, daily_ret: pl.Series) -> None:
        with pytest.raises(ValueError, match="VAR, CVAR|simulate_metric supports"):
            simulate_metric(Metric.SHARPE, daily_ret)

    def test_rejects_historical_method(self, daily_ret: pl.Series) -> None:
        with pytest.raises(ValueError, match='method="historical"|calculate_metric'):
            simulate_metric(Metric.VAR, daily_ret, method="historical")

    def test_innovations_passthrough(self, daily_ret: pl.Series) -> None:
        from mktlib.data import Innovations
        from mktlib.metrics import _mc_cache_clear

        _mc_cache_clear()
        v = simulate_metric(
            Metric.VAR, daily_ret, method="monte_carlo",
            innovations=Innovations.STUDENT_T, df=5,
            n_simulations=2_000, seed=42,
        )
        assert math.isfinite(v)
