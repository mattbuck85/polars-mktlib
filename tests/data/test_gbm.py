from __future__ import annotations

import numpy as np
import pytest

from mktlib.data import geometric_brownian_motion


class TestGeometricBrownianMotion:
    def test_output_shape_and_columns(self):
        df = geometric_brownian_motion(100, seed=42)
        assert df.shape == (100, 2)
        assert df.columns == ["step", "price"]

    def test_first_value_is_base_price(self):
        df = geometric_brownian_motion(50, base_price=200.0, seed=1)
        assert df["price"][0] == pytest.approx(200.0)

    def test_prices_always_positive(self):
        df = geometric_brownian_motion(1000, volatility=0.1, seed=42)
        assert (df["price"] > 0).all()

    def test_reproducibility(self):
        df1 = geometric_brownian_motion(100, seed=99)
        df2 = geometric_brownian_motion(100, seed=99)
        assert df1.equals(df2)

    def test_different_seeds_differ(self):
        df1 = geometric_brownian_motion(100, seed=1)
        df2 = geometric_brownian_motion(100, seed=2)
        assert not df1.equals(df2)

    def test_zero_volatility_pure_drift(self):
        df = geometric_brownian_motion(10, base_price=100.0, drift=0.05, volatility=0.0, dt=1.0, seed=42)
        prices = df["price"].to_numpy()
        # With zero vol: S_t = S_0 * exp(drift * t)
        expected = 100.0 * np.exp(0.05 * np.arange(10))
        np.testing.assert_allclose(prices, expected, rtol=1e-10)

    def test_drift_affects_mean(self):
        n = 5000
        pos_drift = geometric_brownian_motion(n, drift=0.1, volatility=0.01, seed=42)
        neg_drift = geometric_brownian_motion(n, drift=-0.1, volatility=0.01, seed=42)
        assert pos_drift["price"][-1] > neg_drift["price"][-1]

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            geometric_brownian_motion(0)

    def test_single_point(self):
        df = geometric_brownian_motion(1, base_price=50.0, seed=1)
        assert df.shape == (1, 2)
        assert df["price"][0] == pytest.approx(50.0)
