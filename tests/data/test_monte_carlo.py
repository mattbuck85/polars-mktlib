from __future__ import annotations

import pytest

from mktlib.data import geometric_brownian_motion, monte_carlo, ornstein_uhlenbeck


class TestMonteCarlo:
    def test_output_shape(self):
        df = monte_carlo(geometric_brownian_motion, n_simulations=10, n=50, seed=42)
        assert df.shape == (500, 3)  # 10 sims * 50 steps
        assert df.columns == ["simulation", "step", "price"]

    def test_simulation_column_values(self):
        df = monte_carlo(geometric_brownian_motion, n_simulations=5, n=20, seed=42)
        sims = df["simulation"].unique().sort().to_list()
        assert sims == [0, 1, 2, 3, 4]

    def test_reproducibility(self):
        df1 = monte_carlo(geometric_brownian_motion, n_simulations=5, n=50, seed=99)
        df2 = monte_carlo(geometric_brownian_motion, n_simulations=5, n=50, seed=99)
        assert df1.equals(df2)

    def test_different_seeds_differ(self):
        df1 = monte_carlo(geometric_brownian_motion, n_simulations=3, n=50, seed=1)
        df2 = monte_carlo(geometric_brownian_motion, n_simulations=3, n=50, seed=2)
        assert not df1.equals(df2)

    def test_works_with_ou(self):
        df = monte_carlo(ornstein_uhlenbeck, n_simulations=3, n=100, seed=42)
        assert df.columns == ["simulation", "step", "value"]
        assert df.shape == (300, 3)

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
