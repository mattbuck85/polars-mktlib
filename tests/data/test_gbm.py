from __future__ import annotations

import math

import numpy as np
import pytest

from mktlib.data import geometric_brownian_motion


class TestGeometricBrownianMotion:
    def test_output_shape_and_columns(self):
        df = geometric_brownian_motion(100, seed=42)
        assert df.shape == (100, 2)

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
        df = geometric_brownian_motion(
            10, base_price=100.0, drift=0.05, volatility=0.0, dt=1.0, seed=42
        )
        prices = df["price"].to_numpy()
        # With zero vol: S_t = S_0 * exp(drift * t)
        expected = 100.0 * np.exp(0.05 * np.arange(10))
        np.testing.assert_allclose(prices, expected, rtol=1e-10)

    def test_drift_affects_mean(self):
        n = 5000
        pos_drift = geometric_brownian_motion(
            n, drift=0.1, volatility=0.01, seed=42
        )
        neg_drift = geometric_brownian_motion(
            n, drift=-0.1, volatility=0.01, seed=42
        )
        assert pos_drift["price"][-1] > neg_drift["price"][-1]

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            geometric_brownian_motion(0)

    def test_single_point(self):
        df = geometric_brownian_motion(1, base_price=50.0, seed=1)
        assert df.shape == (1, 2)
        assert df["price"][0] == pytest.approx(50.0)


class TestGBMParity:
    """Verify Polars GBM matches a pure-Python reference implementation.

    The Polars implementation uses polars_sdist (Rust) for random draws and
    vectorized cum_sum + exp for prices. This test extracts the Z values
    from the Polars result and feeds them through a Python loop to verify
    the formula produces identical prices.

    The pure-Python reference implements the exact log-space GBM solution:
        ln S[i] = ln S[i-1] + (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z[i]
    """

    @staticmethod
    def _reference_gbm(
        z_values: list[float],
        base_price: float,
        drift: float,
        volatility: float,
        dt: float,
    ) -> list[float]:
        """Pure-Python GBM using stdlib math and a for loop."""
        n = len(z_values)
        mu_dt = (drift - 0.5 * volatility ** 2) * dt
        sigma_sqrt_dt = volatility * math.sqrt(dt)
        log_s = math.log(base_price)

        prices = []
        for i in range(n):
            if i > 0:
                log_s += z_values[i] * sigma_sqrt_dt + mu_dt
            prices.append(math.exp(log_s))
        return prices

    @staticmethod
    def _extract_z(
        prices: list[float], drift: float, volatility: float, dt: float,
    ) -> list[float]:
        """Reconstruct Z values from the price path by inverting the log-space formula.

            Z[i] = (ln S[i] - ln S[i-1] - mu_dt) / sigma_sqrt_dt
        """
        mu_dt = (drift - 0.5 * volatility ** 2) * dt
        sigma_sqrt_dt = volatility * math.sqrt(dt)
        z = [0.0]  # Z[0] unused (step 0 is deterministic)
        for i in range(1, len(prices)):
            z.append((math.log(prices[i]) - math.log(prices[i - 1]) - mu_dt) / sigma_sqrt_dt)
        return z

    @pytest.mark.parametrize(
        "base_price, drift, volatility, dt, seed, n",
        [
            (100.0, 0.0, 0.01, 1 / 252, 42, 500),
            (200.0, 0.05, 0.16, 1 / 252, 99, 300),
            (50.0, -0.03, 0.30, 1 / 52, 7, 200),
            (1000.0, 0.10, 0.005, 1.0, 1, 100),
        ],
        ids=["default-params", "positive-drift", "weekly-high-vol", "annual-low-vol"],
    )
    def test_polars_matches_python_reference(
        self, base_price, drift, volatility, dt, seed, n,
    ):
        """Reconstruct Z from Polars prices, feed into Python reference, verify round-trip."""
        df = geometric_brownian_motion(
            n, base_price=base_price, drift=drift, volatility=volatility, dt=dt, seed=seed,
        )
        polars_prices = df["price"].to_list()

        z_values = self._extract_z(polars_prices, drift, volatility, dt)
        python_prices = self._reference_gbm(z_values, base_price, drift, volatility, dt)

        assert len(polars_prices) == len(python_prices)
        for i, (p, r) in enumerate(zip(polars_prices, python_prices)):
            assert p == pytest.approx(r, rel=1e-10), (
                f"Mismatch at step {i}: polars={p}, python={r}"
            )

    @pytest.mark.parametrize(
        "base_price, drift, volatility, dt, seed, n",
        [
            (100.0, 0.0, 0.01, 1 / 252, 42, 500),
            (200.0, 0.05, 0.16, 1 / 252, 99, 300),
            (50.0, -0.03, 0.30, 1 / 52, 7, 200),
            (1000.0, 0.10, 0.005, 1.0, 1, 100),
        ],
        ids=["default-params", "positive-drift", "weekly-high-vol", "annual-low-vol"],
    )
    def test_reconstructed_z_matches_rng(
        self, base_price, drift, volatility, dt, seed, n,
    ):
        """Algebraically reconstructed Z matches the original RNG draws."""
        from polars_sdist import sample_normal

        df = geometric_brownian_motion(
            n, base_price=base_price, drift=drift, volatility=volatility, dt=dt, seed=seed,
        )
        polars_prices = df["price"].to_list()
        reconstructed_z = self._extract_z(polars_prices, drift, volatility, dt)
        original_z = sample_normal(n, seed=seed).to_list()

        for i in range(1, n):  # skip Z[0] (unused)
            assert reconstructed_z[i] == pytest.approx(original_z[i], rel=1e-10), (
                f"Z mismatch at step {i}: reconstructed={reconstructed_z[i]}, original={original_z[i]}"
            )
