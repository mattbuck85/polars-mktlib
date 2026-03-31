from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mktlib.data import (
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
