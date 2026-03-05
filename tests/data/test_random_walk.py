from __future__ import annotations

import numpy as np
import pytest

from mktlib.data import fractional_random_walk


class TestFractionalRandomWalk:
    def test_output_shape_and_columns(self):
        df = fractional_random_walk(100, seed=42)
        assert df.shape == (100, 2)
        assert df.columns == ["step", "price"]

    def test_first_value_is_base_price(self):
        df = fractional_random_walk(50, base_price=200.0, seed=1)
        assert df["price"][0] == pytest.approx(200.0)

    def test_reproducibility(self):
        df1 = fractional_random_walk(100, seed=123)
        df2 = fractional_random_walk(100, seed=123)
        assert df1.equals(df2)

    def test_different_seeds_differ(self):
        df1 = fractional_random_walk(100, seed=1)
        df2 = fractional_random_walk(100, seed=2)
        assert not df1.equals(df2)

    def test_hurst_05_is_standard_walk(self):
        df = fractional_random_walk(1000, hurst=0.5, seed=42)
        prices = df["price"].to_numpy()
        increments = np.diff(prices)
        # Standard walk increments should have near-zero autocorrelation
        autocorr = np.corrcoef(increments[:-1], increments[1:])[0, 1]
        assert abs(autocorr) < 0.1

    def test_hurst_above_05_positive_autocorrelation(self):
        df = fractional_random_walk(2000, hurst=0.8, step_size=1.0, seed=42)
        prices = df["price"].to_numpy()
        increments = np.diff(prices)
        autocorr = np.corrcoef(increments[:-1], increments[1:])[0, 1]
        assert autocorr > 0.1  # trending: positive autocorrelation

    def test_hurst_below_05_negative_autocorrelation(self):
        df = fractional_random_walk(2000, hurst=0.2, step_size=1.0, seed=42)
        prices = df["price"].to_numpy()
        increments = np.diff(prices)
        autocorr = np.corrcoef(increments[:-1], increments[1:])[0, 1]
        assert autocorr < -0.1  # mean-reverting: negative autocorrelation

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            fractional_random_walk(0)

    def test_invalid_hurst(self):
        with pytest.raises(ValueError, match="hurst must be in"):
            fractional_random_walk(10, hurst=0.0)
        with pytest.raises(ValueError, match="hurst must be in"):
            fractional_random_walk(10, hurst=1.0)

    def test_step_size_scales_variance(self):
        df_small = fractional_random_walk(500, step_size=0.1, seed=42)
        df_large = fractional_random_walk(500, step_size=10.0, seed=42)
        var_small = np.var(np.diff(df_small["price"].to_numpy()))
        var_large = np.var(np.diff(df_large["price"].to_numpy()))
        # Ratio should be approximately (10/0.1)^2 = 10000
        assert var_large / var_small == pytest.approx(10000, rel=0.1)
