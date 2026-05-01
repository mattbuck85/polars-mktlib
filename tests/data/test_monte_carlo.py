from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mktlib.data import (
    Innovations,
    Process,
    fractional_random_walk,
    geometric_brownian_motion,
    monte_carlo,
    ornstein_uhlenbeck,
)


class TestMonteCarlo:
    def test_output_shape(self):
        df = monte_carlo(
            geometric_brownian_motion, n_simulations=10, n=50, seed=42
        )
        assert df.shape == (500, 4)  # 10 sims * 50 steps

    def test_simulation_column_values(self):
        df = monte_carlo(
            geometric_brownian_motion, n_simulations=5, n=20, seed=42
        )
        sims = df["simulation"].unique().sort().to_list()
        assert sims == [0, 1, 2, 3, 4]

    def test_reproducibility(self):
        df1 = monte_carlo(
            geometric_brownian_motion, n_simulations=5, n=50, seed=99
        )
        df2 = monte_carlo(
            geometric_brownian_motion, n_simulations=5, n=50, seed=99
        )
        assert df1.equals(df2)

    def test_different_seeds_differ(self):
        df1 = monte_carlo(
            geometric_brownian_motion, n_simulations=3, n=50, seed=1
        )
        df2 = monte_carlo(
            geometric_brownian_motion, n_simulations=3, n=50, seed=2
        )
        assert not df1.equals(df2)

    def test_works_with_ou(self):
        df = monte_carlo(ornstein_uhlenbeck, n_simulations=3, n=100, seed=42)
        assert df.shape == (300, 4)

    def test_passes_kwargs(self):
        df = monte_carlo(
            geometric_brownian_motion,
            n_simulations=2,
            n=10,
            base_price=500.0,
            seed=42,
        )
        # First price of each sim should be base_price
        first_prices = df.filter(step=0)["price"].to_list()
        assert all(p == pytest.approx(500.0) for p in first_prices)

    def test_invalid_n_simulations(self):
        with pytest.raises(ValueError, match="n_simulations must be >= 1"):
            monte_carlo(geometric_brownian_motion, n_simulations=0, n=10)


class TestMonteCarloEnum:
    def test_gbm_enum_shape(self):
        df = monte_carlo(Process.GBM, n_simulations=10, n=50, seed=42)
        assert df.shape == (500, 4)

    def test_gbm_enum_simulation_values(self):
        df = monte_carlo(Process.GBM, n_simulations=5, n=20, seed=42)
        sims = df["simulation"].unique().sort().to_list()
        assert sims == [0, 1, 2, 3, 4]

    def test_gbm_enum_reproducibility(self):
        df1 = monte_carlo(Process.GBM, n_simulations=5, n=50, seed=99)
        df2 = monte_carlo(Process.GBM, n_simulations=5, n=50, seed=99)
        assert df1.equals(df2)

    def test_gbm_enum_different_seeds_differ(self):
        df1 = monte_carlo(Process.GBM, n_simulations=3, n=50, seed=1)
        df2 = monte_carlo(Process.GBM, n_simulations=3, n=50, seed=2)
        assert not df1.equals(df2)

    def test_gbm_enum_base_price(self):
        df = monte_carlo(
            Process.GBM, n_simulations=2, n=10, base_price=500.0, seed=42
        )
        first_prices = df.filter(step=0)["price"].to_list()
        assert all(p == pytest.approx(500.0) for p in first_prices)

    def test_gbm_enum_prices_positive(self):
        df = monte_carlo(
            Process.GBM, n_simulations=10, n=100, volatility=0.1, seed=42
        )
        assert (df["price"] > 0).all()

    def test_ou_enum(self):
        df = monte_carlo(Process.OU, n_simulations=3, n=100, seed=42)
        assert df.shape == (300, 4)

    def test_frw_enum(self):
        df = monte_carlo(Process.FRW, n_simulations=3, n=100, seed=42)
        assert df.shape == (300, 4)

    def test_gbm_enum_single_point(self):
        df = monte_carlo(Process.GBM, n_simulations=5, n=1, seed=42)
        assert df.shape == (5, 4)
        assert all(
            p == pytest.approx(100.0) for p in df["price"].to_list()
        )


class TestFrwVectorized:
    def test_reproducibility(self):
        df1 = monte_carlo(
            Process.FRW, n_simulations=5, n=50, hurst=0.7, seed=99
        )
        df2 = monte_carlo(
            Process.FRW, n_simulations=5, n=50, hurst=0.7, seed=99
        )
        assert df1.equals(df2)

    def test_hurst_above_05_positive_autocorrelation(self):
        df = monte_carlo(
            Process.FRW,
            n_simulations=20,
            n=500,
            hurst=0.8,
            step_size=1.0,
            seed=42,
        )
        autocorrs = []
        for sim in df["simulation"].unique().to_list():
            prices = df.filter(simulation=sim)["price"].to_numpy()
            inc = np.diff(prices)
            if len(inc) > 1:
                autocorrs.append(np.corrcoef(inc[:-1], inc[1:])[0, 1])
        mean_ac = np.mean(autocorrs)
        assert mean_ac > 0.1

    def test_hurst_below_05_negative_autocorrelation(self):
        df = monte_carlo(
            Process.FRW,
            n_simulations=20,
            n=500,
            hurst=0.2,
            step_size=1.0,
            seed=42,
        )
        autocorrs = []
        for sim in df["simulation"].unique().to_list():
            prices = df.filter(simulation=sim)["price"].to_numpy()
            inc = np.diff(prices)
            if len(inc) > 1:
                autocorrs.append(np.corrcoef(inc[:-1], inc[1:])[0, 1])
        mean_ac = np.mean(autocorrs)
        assert mean_ac < -0.1

    def test_base_price(self):
        df = monte_carlo(
            Process.FRW,
            n_simulations=5,
            n=20,
            base_price=250.0,
            seed=42,
        )
        first_prices = df.filter(step=0)["price"].to_list()
        assert all(p == pytest.approx(250.0) for p in first_prices)


_SEED_CASES = [
    pytest.param(Process.GBM, {}, id="gbm"),
    pytest.param(Process.OU, {}, id="ou"),
    pytest.param(Process.FRW, {"hurst": 0.5}, id="frw_h05"),
    pytest.param(Process.FRW, {"hurst": 0.7}, id="frw_h07"),
    pytest.param(geometric_brownian_motion, {}, id="loop"),
]


class TestSeedColumn:
    """Every simulation must carry exactly one child seed."""

    @pytest.mark.parametrize("process, kwargs", _SEED_CASES)
    def test_one_seed_per_simulation(self, process, kwargs):
        df = monte_carlo(process, n_simulations=5, n=20, seed=42, **kwargs)
        seeds_per_sim = df.group_by("simulation").agg(
            pl.col("seed").n_unique().alias("n_seeds")
        )
        assert (seeds_per_sim["n_seeds"] == 1).all()

    @pytest.mark.parametrize("process, kwargs", _SEED_CASES)
    def test_different_simulations_have_different_seeds(self, process, kwargs):
        df = monte_carlo(process, n_simulations=5, n=20, seed=42, **kwargs)
        unique_seeds = df["seed"].unique()
        assert len(unique_seeds) == 5


class TestVectorizedMatchesLoop:
    """Verify vectorized (Process enum) path matches serial (callable) path."""

    def test_gbm_vectorized_matches_loop(self):
        vec = monte_carlo(
            Process.GBM, n_simulations=5, n=100, seed=42, drift=0.05, volatility=0.1,
        )
        loop = monte_carlo(
            geometric_brownian_motion, n_simulations=5, n=100, seed=42, drift=0.05, volatility=0.1,
        )
        for sim in range(5):
            v = vec.filter(simulation=sim)["price"].to_list()
            l = loop.filter(simulation=sim)["price"].to_list()
            for i, (a, b) in enumerate(zip(v, l)):
                assert a == pytest.approx(b, rel=1e-10), (
                    f"GBM sim {sim} step {i}: vec={a}, loop={b}"
                )

    def test_ou_vectorized_matches_loop(self):
        vec = monte_carlo(
            Process.OU, n_simulations=5, n=100, seed=42, theta=0.5, mu=50.0, sigma=2.0,
        )
        loop = monte_carlo(
            ornstein_uhlenbeck, n_simulations=5, n=100, seed=42, theta=0.5, mu=50.0, sigma=2.0,
        )
        for sim in range(5):
            v = vec.filter(simulation=sim)["value"].to_list()
            l = loop.filter(simulation=sim)["value"].to_list()
            for i, (a, b) in enumerate(zip(v, l)):
                assert a == pytest.approx(b, rel=1e-10), (
                    f"OU sim {sim} step {i}: vec={a}, loop={b}"
                )


class TestGeometricMonteCarlo:
    def test_frw_geometric_positive(self):
        df = monte_carlo(Process.FRW, n_simulations=10, n=100, seed=42, geometric=True, step_size=0.3)
        assert (df["price"] > 0).all()

    def test_ou_geometric_positive(self):
        df = monte_carlo(Process.OU, n_simulations=10, n=100, seed=42, geometric=True, mu=0.0, sigma=0.5)
        assert "price" in df.columns
        assert (df["price"] > 0).all()

    def test_frw_geometric_davies_harte_positive(self):
        df = monte_carlo(Process.FRW, n_simulations=5, n=100, seed=42, geometric=True, hurst=0.7, step_size=0.2)
        assert (df["price"] > 0).all()

    @pytest.mark.parametrize(
        "hurst, step_size",
        [(0.5, 1.0), (0.7, 2.0), (0.3, 0.5)],
        ids=["h05-standard", "h07-persistent", "h03-antipersistent"],
    )
    def test_frw_vectorized_matches_loop(self, hurst, step_size):
        n_sims, n = 5, 100
        vec = monte_carlo(
            Process.FRW, n_simulations=n_sims, n=n, seed=42,
            hurst=hurst, step_size=step_size, base_price=100.0,
        )
        loop = monte_carlo(
            fractional_random_walk, n_simulations=n_sims, n=n, seed=42,
            hurst=hurst, step_size=step_size, base_price=100.0,
        )
        for sim in range(n_sims):
            v = vec.filter(simulation=sim)["price"].to_list()
            l = loop.filter(simulation=sim)["price"].to_list()
            for i, (a, b) in enumerate(zip(v, l)):
                assert a == pytest.approx(b, rel=1e-10), (
                    f"FRW(H={hurst}) sim {sim} step {i}: vec={a}, loop={b}"
                )


class TestInnovations:
    """Pluggable innovation distributions for Process.GBM."""

    def test_gaussian_default_matches_explicit(self):
        """Default == Innovations.GAUSSIAN — locks the default."""
        df_default = monte_carlo(Process.GBM, n_simulations=5, n=20, seed=42)
        df_explicit = monte_carlo(
            Process.GBM, n_simulations=5, n=20, seed=42,
            innovations=Innovations.GAUSSIAN,
        )
        assert df_default.equals(df_explicit)

    def test_students_t_unit_variance(self):
        """Student-t innovations realize ~unit-variance log-returns at fixed sigma."""
        # Gaussian baseline log-return std under volatility=0.2, dt=1/252
        gauss = monte_carlo(
            Process.GBM, n_simulations=2000, n=200, seed=42,
            volatility=0.2, dt=1 / 252,
            innovations=Innovations.GAUSSIAN,
        )
        t_df = monte_carlo(
            Process.GBM, n_simulations=2000, n=200, seed=42,
            volatility=0.2, dt=1 / 252,
            innovations=Innovations.STUDENT_T, df=5,
        )
        # Per-bar log-return std should match within ~5% (sample SE of std ~ 1/sqrt(2N))
        gauss_std = (
            gauss.with_columns(
                pl.col("price").log().diff().over("simulation").alias("logret")
            )
            .drop_nulls("logret")["logret"]
            .std()
        )
        t_std = (
            t_df.with_columns(
                pl.col("price").log().diff().over("simulation").alias("logret")
            )
            .drop_nulls("logret")["logret"]
            .std()
        )
        assert t_std == pytest.approx(gauss_std, rel=0.10)

    def test_students_t_kurtosis_higher(self):
        """Student-t increments must have heavier tails than Gaussian."""
        gauss = monte_carlo(
            Process.GBM, n_simulations=200, n=500, seed=42,
            volatility=0.2, innovations=Innovations.GAUSSIAN,
        )
        t_df = monte_carlo(
            Process.GBM, n_simulations=200, n=500, seed=42,
            volatility=0.2, innovations=Innovations.STUDENT_T, df=4,
        )
        gauss_logret = (
            gauss.with_columns(
                pl.col("price").log().diff().over("simulation").alias("logret")
            )["logret"].drop_nulls().to_numpy()
        )
        t_logret = (
            t_df.with_columns(
                pl.col("price").log().diff().over("simulation").alias("logret")
            )["logret"].drop_nulls().to_numpy()
        )
        # Excess kurtosis of t with df=4 = 6/(df-4) → infinite; in finite samples
        # at least clearly above Gaussian (~0).  Use a generous margin.
        from scipy.stats import kurtosis  # type: ignore[import-untyped]
        assert kurtosis(t_logret) > kurtosis(gauss_logret) + 1.0

    def test_students_t_requires_df(self):
        with pytest.raises(ValueError, match="df="):
            monte_carlo(
                Process.GBM, n_simulations=2, n=5, seed=1,
                innovations=Innovations.STUDENT_T,
            )

    def test_students_t_rejects_df_le_2(self):
        with pytest.raises(ValueError, match="df must be > 2"):
            monte_carlo(
                Process.GBM, n_simulations=2, n=5, seed=1,
                innovations=Innovations.STUDENT_T, df=2.0,
            )

    def test_bootstrap_uses_residuals(self):
        """Bootstrap path consumes the supplied residual series."""
        # Two-valued residuals → only two possible log-increments per step
        residuals = pl.Series("r", [-1.0, 1.0])
        out = monte_carlo(
            Process.GBM, n_simulations=10, n=10, seed=42,
            volatility=0.05, drift=0.0, dt=1.0,
            innovations=Innovations.BOOTSTRAP, residuals=residuals,
        )
        # Per-bar log-increment must be ±0.05 + small drift correction
        log_increments = (
            out.with_columns(
                pl.col("price").log().diff().over("simulation").alias("logret")
            )["logret"].drop_nulls().to_list()
        )
        # mu_dt = -0.5*0.05^2*1 = -0.00125; sigma*sqrt(dt) = 0.05
        # so logret ∈ {-0.00125 - 0.05, -0.00125 + 0.05}
        for v in log_increments:
            assert v == pytest.approx(-0.05125, abs=1e-9) or v == pytest.approx(
                0.04875, abs=1e-9
            )

    def test_bootstrap_requires_residuals(self):
        with pytest.raises(ValueError, match="residuals="):
            monte_carlo(
                Process.GBM, n_simulations=2, n=5, seed=1,
                innovations=Innovations.BOOTSTRAP,
            )

    def test_callable_innovations(self):
        """Callable noise source — same result as GAUSSIAN when wrapping sample_normal."""
        from polars_sdist import sample_normal

        def gauss_callable(n: int, seed: int | None) -> pl.Series:
            return sample_normal(n, seed=seed)

        df_callable = monte_carlo(
            Process.GBM, n_simulations=3, n=20, seed=42,
            innovations=gauss_callable,
        )
        df_enum = monte_carlo(
            Process.GBM, n_simulations=3, n=20, seed=42,
            innovations=Innovations.GAUSSIAN,
        )
        assert df_callable.equals(df_enum)

    def test_innovations_only_supported_for_gbm(self):
        with pytest.raises(NotImplementedError, match="Process.GBM"):
            monte_carlo(
                Process.OU, n_simulations=2, n=5, seed=1,
                innovations=Innovations.STUDENT_T, df=5,
            )
        with pytest.raises(NotImplementedError, match="Process.GBM"):
            monte_carlo(
                Process.FRW, n_simulations=2, n=5, seed=1,
                innovations=Innovations.STUDENT_T, df=5,
            )

    def test_loop_callable_rejects_innovations(self):
        with pytest.raises(NotImplementedError, match="callable processes"):
            monte_carlo(
                geometric_brownian_motion, n_simulations=2, n=5, seed=1,
                innovations=Innovations.STUDENT_T, df=5,
            )

    def test_innovations_reproducibility(self):
        df1 = monte_carlo(
            Process.GBM, n_simulations=5, n=50, seed=99,
            innovations=Innovations.STUDENT_T, df=5,
        )
        df2 = monte_carlo(
            Process.GBM, n_simulations=5, n=50, seed=99,
            innovations=Innovations.STUDENT_T, df=5,
        )
        assert df1.equals(df2)

    @pytest.mark.parametrize(
        "innovations, kwargs",
        [
            pytest.param(Innovations.GAUSSIAN, {}, id="gaussian"),
            pytest.param(Innovations.STUDENT_T, {"df": 5.0}, id="students_t"),
            pytest.param(
                Innovations.BOOTSTRAP,
                {"residuals": pl.Series("r", [-1.0, 0.0, 1.0])},
                id="bootstrap",
            ),
        ],
    )
    def test_prices_positive(self, innovations, kwargs):
        df = monte_carlo(
            Process.GBM, n_simulations=10, n=100, seed=42,
            volatility=0.1, innovations=innovations, **kwargs,
        )
        assert (df["price"] > 0).all()


class TestIndependentStreams:
    """Single-batch fast path (independent_streams=False) for GBM."""

    def test_default_is_independent_streams(self):
        """Default behavior (no flag) matches independent_streams=True byte-for-byte."""
        df_default = monte_carlo(
            Process.GBM, n_simulations=8, n=20, seed=42, volatility=0.2,
        )
        df_explicit = monte_carlo(
            Process.GBM, n_simulations=8, n=20, seed=42, volatility=0.2,
            independent_streams=True,
        )
        assert df_default.equals(df_explicit)

    def test_single_batch_reproducible(self):
        df1 = monte_carlo(
            Process.GBM, n_simulations=10, n=50, seed=99, volatility=0.2,
            independent_streams=False,
        )
        df2 = monte_carlo(
            Process.GBM, n_simulations=10, n=50, seed=99, volatility=0.2,
            independent_streams=False,
        )
        assert df1.equals(df2)

    def test_single_batch_seed_column_uniform(self):
        """All simulations share one parent-derived seed in single-batch mode."""
        df = monte_carlo(
            Process.GBM, n_simulations=10, n=20, seed=42,
            independent_streams=False,
        )
        assert df["seed"].n_unique() == 1

    def test_single_batch_differs_from_streams(self):
        """The two modes produce structurally different frames."""
        streams = monte_carlo(
            Process.GBM, n_simulations=5, n=20, seed=42,
            independent_streams=True,
        )
        single = monte_carlo(
            Process.GBM, n_simulations=5, n=20, seed=42,
            independent_streams=False,
        )
        assert not streams.equals(single)

    @pytest.mark.parametrize(
        "innovations, kwargs",
        [
            pytest.param(Innovations.GAUSSIAN, {}, id="gaussian"),
            pytest.param(Innovations.STUDENT_T, {"df": 5.0}, id="students_t"),
            pytest.param(
                Innovations.BOOTSTRAP,
                {"residuals": pl.Series("r", [-1.0, 0.0, 1.0, 0.5, -0.5])},
                id="bootstrap",
            ),
        ],
    )
    def test_single_batch_prices_positive(self, innovations, kwargs):
        df = monte_carlo(
            Process.GBM, n_simulations=20, n=50, seed=42, volatility=0.1,
            innovations=innovations, independent_streams=False, **kwargs,
        )
        assert (df["price"] > 0).all()

    @pytest.mark.parametrize(
        "innovations, kwargs",
        [
            pytest.param(Innovations.GAUSSIAN, {}, id="gaussian"),
            pytest.param(Innovations.STUDENT_T, {"df": 5.0}, id="students_t"),
        ],
    )
    def test_single_batch_distribution_matches(self, innovations, kwargs):
        """Empirical mean/std of single-batch ≈ multi-stream at large n_sims."""
        n_sims, n = 5000, 50
        streams = monte_carlo(
            Process.GBM, n_simulations=n_sims, n=n, seed=42,
            volatility=0.2, innovations=innovations,
            independent_streams=True, **kwargs,
        )
        single = monte_carlo(
            Process.GBM, n_simulations=n_sims, n=n, seed=42,
            volatility=0.2, innovations=innovations,
            independent_streams=False, **kwargs,
        )
        s_mean = float(streams["price"].mean())  # type: ignore[arg-type]
        b_mean = float(single["price"].mean())  # type: ignore[arg-type]
        s_std = float(streams["price"].std())  # type: ignore[arg-type]
        b_std = float(single["price"].std())  # type: ignore[arg-type]
        # Loose: same population, sample-size-dependent SE
        assert b_mean == pytest.approx(s_mean, rel=0.05)
        assert b_std == pytest.approx(s_std, rel=0.10)

    def test_ou_supports_single_batch(self):
        df = monte_carlo(
            Process.OU, n_simulations=10, n=50, seed=42,
            independent_streams=False,
        )
        assert df.shape == (500, 4)
        assert df["seed"].n_unique() == 1

    @pytest.mark.parametrize("hurst", [0.3, 0.5, 0.7])
    def test_frw_supports_single_batch(self, hurst):
        df = monte_carlo(
            Process.FRW, n_simulations=10, n=50, seed=42, hurst=hurst,
            independent_streams=False,
        )
        assert df.shape == (500, 4)
        assert df["seed"].n_unique() == 1

    def test_ou_single_batch_reproducible(self):
        df1 = monte_carlo(
            Process.OU, n_simulations=5, n=50, seed=99,
            independent_streams=False,
        )
        df2 = monte_carlo(
            Process.OU, n_simulations=5, n=50, seed=99,
            independent_streams=False,
        )
        assert df1.equals(df2)

    @pytest.mark.parametrize("hurst", [0.5, 0.7])
    def test_frw_single_batch_reproducible(self, hurst):
        df1 = monte_carlo(
            Process.FRW, n_simulations=5, n=50, seed=99, hurst=hurst,
            independent_streams=False,
        )
        df2 = monte_carlo(
            Process.FRW, n_simulations=5, n=50, seed=99, hurst=hurst,
            independent_streams=False,
        )
        assert df1.equals(df2)

    def test_callable_process_rejects_single_batch(self):
        from mktlib.data import geometric_brownian_motion

        with pytest.raises(NotImplementedError, match="callable processes"):
            monte_carlo(
                geometric_brownian_motion, n_simulations=2, n=5, seed=1,
                independent_streams=False,
            )


class TestStreamModeStatisticalEquivalence:
    """Two-sample Kolmogorov–Smirnov tests confirming the two stream modes
    draw from the same population at large *n_simulations*.

    These are the *integration* checks that back the math: any future
    refactor that breaks the i.i.d. property of the single-batch path
    will trip these long before subtle convergence bugs reach reports.
    """

    @staticmethod
    def _ks_pvalue(a, b) -> float:
        from scipy.stats import ks_2samp  # type: ignore[import-untyped]
        return float(ks_2samp(a, b).pvalue)  # pyright: ignore[reportAttributeAccessIssue]

    def test_gbm_gaussian_horizon_returns_match(self):
        n_sims, n = 20_000, 22
        streams = monte_carlo(
            Process.GBM, n_simulations=n_sims, n=n, seed=42,
            volatility=0.2, drift=0.05, independent_streams=True,
        )
        single = monte_carlo(
            Process.GBM, n_simulations=n_sims, n=n, seed=43,  # different seed
            volatility=0.2, drift=0.05, independent_streams=False,
        )
        # Compare the cross-simulation distribution of horizon-end prices
        s_end = streams.filter(pl.col("step") == n - 1)["price"].to_numpy()
        b_end = single.filter(pl.col("step") == n - 1)["price"].to_numpy()
        p = self._ks_pvalue(s_end, b_end)
        assert p > 0.001, f"GBM Gaussian KS p-value too low: {p:.4f}"

    def test_gbm_students_t_horizon_returns_match(self):
        n_sims, n = 20_000, 22
        streams = monte_carlo(
            Process.GBM, n_simulations=n_sims, n=n, seed=42,
            volatility=0.2, drift=0.05,
            innovations=Innovations.STUDENT_T, df=5,
            independent_streams=True,
        )
        single = monte_carlo(
            Process.GBM, n_simulations=n_sims, n=n, seed=43,
            volatility=0.2, drift=0.05,
            innovations=Innovations.STUDENT_T, df=5,
            independent_streams=False,
        )
        s_end = streams.filter(pl.col("step") == n - 1)["price"].to_numpy()
        b_end = single.filter(pl.col("step") == n - 1)["price"].to_numpy()
        p = self._ks_pvalue(s_end, b_end)
        assert p > 0.001, f"GBM Student-t KS p-value too low: {p:.4f}"

    def test_ou_horizon_values_match(self):
        n_sims, n = 10_000, 50
        streams = monte_carlo(
            Process.OU, n_simulations=n_sims, n=n, seed=42,
            theta=0.5, mu=0.0, sigma=1.0,
            independent_streams=True,
        )
        single = monte_carlo(
            Process.OU, n_simulations=n_sims, n=n, seed=43,
            theta=0.5, mu=0.0, sigma=1.0,
            independent_streams=False,
        )
        s_end = streams.filter(pl.col("step") == n - 1)["value"].to_numpy()
        b_end = single.filter(pl.col("step") == n - 1)["value"].to_numpy()
        p = self._ks_pvalue(s_end, b_end)
        assert p > 0.001, f"OU KS p-value too low: {p:.4f}"

    def test_frw_h05_horizon_prices_match(self):
        n_sims, n = 10_000, 50
        streams = monte_carlo(
            Process.FRW, n_simulations=n_sims, n=n, seed=42, hurst=0.5,
            independent_streams=True,
        )
        single = monte_carlo(
            Process.FRW, n_simulations=n_sims, n=n, seed=43, hurst=0.5,
            independent_streams=False,
        )
        s_end = streams.filter(pl.col("step") == n - 1)["price"].to_numpy()
        b_end = single.filter(pl.col("step") == n - 1)["price"].to_numpy()
        p = self._ks_pvalue(s_end, b_end)
        assert p > 0.001, f"FRW H=0.5 KS p-value too low: {p:.4f}"

    def test_frw_h07_horizon_prices_match(self):
        n_sims, n = 10_000, 50
        streams = monte_carlo(
            Process.FRW, n_simulations=n_sims, n=n, seed=42, hurst=0.7,
            step_size=0.5, independent_streams=True,
        )
        single = monte_carlo(
            Process.FRW, n_simulations=n_sims, n=n, seed=43, hurst=0.7,
            step_size=0.5, independent_streams=False,
        )
        s_end = streams.filter(pl.col("step") == n - 1)["price"].to_numpy()
        b_end = single.filter(pl.col("step") == n - 1)["price"].to_numpy()
        p = self._ks_pvalue(s_end, b_end)
        assert p > 0.001, f"FRW H=0.7 KS p-value too low: {p:.4f}"

    def test_gbm_bootstrap_horizon_returns_match(self):
        n_sims, n = 20_000, 22
        residuals = pl.Series(
            "r",
            [-2.0, -1.5, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0],
        )
        # Standardize to unit variance (BOOTSTRAP contract)
        residuals = (residuals - residuals.mean()) / residuals.std()  # type: ignore[operator]
        streams = monte_carlo(
            Process.GBM, n_simulations=n_sims, n=n, seed=42,
            volatility=0.2, innovations=Innovations.BOOTSTRAP,
            residuals=residuals, independent_streams=True,
        )
        single = monte_carlo(
            Process.GBM, n_simulations=n_sims, n=n, seed=43,
            volatility=0.2, innovations=Innovations.BOOTSTRAP,
            residuals=residuals, independent_streams=False,
        )
        s_end = streams.filter(pl.col("step") == n - 1)["price"].to_numpy()
        b_end = single.filter(pl.col("step") == n - 1)["price"].to_numpy()
        p = self._ks_pvalue(s_end, b_end)
        assert p > 0.001, f"GBM Bootstrap KS p-value too low: {p:.4f}"
