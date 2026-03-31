from __future__ import annotations

import math

import numpy as np
import pytest

from mktlib.data import ornstein_uhlenbeck


class TestOrnsteinUhlenbeck:
    def test_output_shape_and_columns(self):
        df = ornstein_uhlenbeck(100, seed=42)
        assert df.shape == (100, 2)

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


class TestOUParity:
    """Verify Polars OU matches a pure-Python reference via algebraic Z reconstruction.

    Reconstructs Z from the OU value path by inverting Euler-Maruyama,
    then feeds those Z values into a Python loop to verify round-trip.
    """

    @staticmethod
    def _extract_z(
        values: list[float], theta: float, mu: float, sigma: float, dt: float,
    ) -> list[float]:
        """Reconstruct Z from the OU value path by inverting Euler-Maruyama.

            z[i] = (x[i] - x[i-1] - theta*(mu - x[i-1])*dt) / (sigma*sqrt(dt))
        """
        noise_scale = sigma * math.sqrt(dt)
        z = [0.0]  # z[0] unused (step 0 is deterministic)
        for i in range(1, len(values)):
            z.append(
                (values[i] - values[i - 1] - theta * (mu - values[i - 1]) * dt)
                / noise_scale
            )
        return z

    @staticmethod
    def _reference_ou(
        z: list[float], theta: float, mu: float, sigma: float, x0: float, dt: float,
    ) -> list[float]:
        """Pure-Python Euler-Maruyama loop for OU."""
        noise_scale = sigma * math.sqrt(dt)
        x = [x0]
        for i in range(1, len(z)):
            x_prev = x[-1]
            x.append(x_prev + theta * (mu - x_prev) * dt + noise_scale * z[i])
        return x

    @pytest.mark.parametrize(
        "theta, mu, sigma, x0, dt, seed, n",
        [
            (0.7, 100.0, 1.0, 100.0, 1.0, 42, 500),
            (0.5, 50.0, 2.0, 200.0, 0.1, 99, 300),
            (2.0, 0.0, 0.5, -10.0, 0.01, 7, 200),
            (0.1, 75.0, 3.0, 75.0, 0.5, 1, 400),
        ],
        ids=["default-params", "far-from-mean", "negative-start", "at-equilibrium"],
    )
    def test_polars_matches_python_reference(
        self, theta, mu, sigma, x0, dt, seed, n,
    ):
        """Reconstruct Z from Polars values, feed into Python reference, verify round-trip."""
        df = ornstein_uhlenbeck(
            n, theta=theta, mu=mu, sigma=sigma, x0=x0, dt=dt, seed=seed,
        )
        polars_values = df["value"].to_list()

        z_values = self._extract_z(polars_values, theta, mu, sigma, dt)
        python_values = self._reference_ou(z_values, theta, mu, sigma, x0, dt)

        assert len(polars_values) == len(python_values)
        for i, (p, r) in enumerate(zip(polars_values, python_values)):
            assert p == pytest.approx(r, rel=1e-10), (
                f"Mismatch at step {i}: polars={p}, python={r}"
            )

    @pytest.mark.parametrize(
        "theta, mu, sigma, x0, dt, seed, n",
        [
            (0.7, 100.0, 1.0, 100.0, 1.0, 42, 500),
            (0.5, 50.0, 2.0, 200.0, 0.1, 99, 300),
            (2.0, 0.0, 0.5, -10.0, 0.01, 7, 200),
        ],
        ids=["default-params", "far-from-mean", "negative-start"],
    )
    def test_reconstructed_z_matches_rng(
        self, theta, mu, sigma, x0, dt, seed, n,
    ):
        """Algebraically reconstructed Z matches the original RNG draws."""
        from polars_sdist import sample_normal

        df = ornstein_uhlenbeck(
            n, theta=theta, mu=mu, sigma=sigma, x0=x0, dt=dt, seed=seed,
        )
        polars_values = df["value"].to_list()
        reconstructed_z = self._extract_z(polars_values, theta, mu, sigma, dt)
        original_z = sample_normal(n, seed=seed).to_list()

        for i in range(1, n):  # skip z[0] (unused)
            assert reconstructed_z[i] == pytest.approx(original_z[i], rel=1e-10), (
                f"Z mismatch at step {i}: reconstructed={reconstructed_z[i]}, original={original_z[i]}"
            )


class TestGeometricOU:
    def test_geometric_prices_always_positive(self):
        df = ornstein_uhlenbeck(500, geometric=True, mu=0.0, sigma=0.5, seed=42)
        assert "price" in df.columns
        assert (df["price"] > 0).all()

    def test_geometric_equals_exp_of_additive(self):
        """geometric=True output should be exp() of the additive output."""
        kwargs = dict(n=200, theta=0.7, mu=0.0, sigma=0.5, x0=0.0, seed=42)
        additive = ornstein_uhlenbeck(**kwargs, geometric=False)
        geometric = ornstein_uhlenbeck(**kwargs, geometric=True)
        expected = np.exp(additive["value"].to_numpy())
        np.testing.assert_allclose(geometric["price"].to_numpy(), expected, rtol=1e-12)

    def test_geometric_default_false_unchanged(self):
        df_old = ornstein_uhlenbeck(100, seed=42)
        df_new = ornstein_uhlenbeck(100, geometric=False, seed=42)
        assert df_old.equals(df_new)

    def test_geometric_column_name_is_price(self):
        df = ornstein_uhlenbeck(50, geometric=True, seed=1)
        assert "price" in df.columns
        assert "value" not in df.columns
