from __future__ import annotations

import numpy as np
import pytest

from mktlib.data import ornstein_uhlenbeck


class TestOrnsteinUhlenbeck:
    def test_output_shape_and_columns(self):
        df = ornstein_uhlenbeck(100, seed=42)
        assert df.shape == (100, 2)
        assert df.columns == ["step", "value"]

    def test_starts_at_mu_by_default(self):
        df = ornstein_uhlenbeck(10, mu=50.0, seed=1)
        assert df["value"][0] == pytest.approx(50.0)

    def test_starts_at_x0_when_given(self):
        df = ornstein_uhlenbeck(10, mu=100.0, x0=50.0, seed=1)
        assert df["value"][0] == pytest.approx(50.0)

    def test_reproducibility(self):
        df1 = ornstein_uhlenbeck(100, seed=42)
        df2 = ornstein_uhlenbeck(100, seed=42)
        assert df1.equals(df2)

    def test_mean_reversion(self):
        # Start far from mu, high theta — should revert
        df = ornstein_uhlenbeck(
            5000, theta=0.5, mu=100.0, sigma=0.5, x0=200.0, dt=0.1, seed=42
        )
        values = df["value"].to_numpy()
        # Last 1000 values should be centered near mu
        tail_mean = np.mean(values[-1000:])
        assert abs(tail_mean - 100.0) < 10.0

    def test_high_theta_faster_reversion(self):
        kwargs = {
            "n": 500,
            "mu": 100.0,
            "sigma": 0.5,
            "x0": 200.0,
            "dt": 0.1,
            "seed": 42,
        }
        slow = ornstein_uhlenbeck(theta=0.1, **kwargs)
        fast = ornstein_uhlenbeck(theta=2.0, **kwargs)
        # Fast reversion should be closer to mu at step 100
        assert abs(fast["value"][100] - 100.0) < abs(
            slow["value"][100] - 100.0
        )

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            ornstein_uhlenbeck(0)
