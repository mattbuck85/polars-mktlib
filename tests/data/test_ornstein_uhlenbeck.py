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

    @pytest.mark.parametrize(
        "theta, mu, sigma, x0, dt",
        [
            (0.7, 100.0, 1.0, None, 1.0),
            (0.5, 50.0, 2.0, 200.0, 0.1),
            (2.0, 0.0, 0.5, -10.0, 0.01),
        ],
    )
    def test_equivalence_with_iterative(self, theta, mu, sigma, x0, dt):
        """Vectorized cumsum formulation must match the Euler-Maruyama loop."""
        n = 500
        seed = 99
        df = ornstein_uhlenbeck(
            n, theta=theta, mu=mu, sigma=sigma, x0=x0, dt=dt, seed=seed
        )

        # Rebuild iteratively from the same noise vector
        from polars_sdist import sample_normal

        z = sample_normal(n, seed=seed).to_list()
        start = mu if x0 is None else x0
        noise_scale = sigma * np.sqrt(dt)

        x = [start]
        for i in range(1, n):
            x_prev = x[-1]
            x.append(x_prev + theta * (mu - x_prev) * dt + noise_scale * z[i])

        np.testing.assert_allclose(
            df["value"].to_numpy(), np.array(x), rtol=1e-12
        )
