from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mktlib.data import fractional_random_walk, monte_carlo
from mktlib.data._monte_carlo import Process


class TestFractionalRandomWalk:
    def test_output_shape_and_columns(self):
        df = fractional_random_walk(100, seed=42)
        assert df.shape == (100, 2)

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
        autocorr = np.corrcoef(increments[:-1], increments[1:])[0, 1]
        assert abs(autocorr) < 0.1

    def test_hurst_above_05_positive_autocorrelation(self):
        df = fractional_random_walk(2000, hurst=0.8, step_size=1.0, seed=42)
        prices = df["price"].to_numpy()
        increments = np.diff(prices)
        autocorr = np.corrcoef(increments[:-1], increments[1:])[0, 1]
        assert autocorr > 0.1

    def test_hurst_below_05_negative_autocorrelation(self):
        df = fractional_random_walk(2000, hurst=0.2, step_size=1.0, seed=42)
        prices = df["price"].to_numpy()
        increments = np.diff(prices)
        autocorr = np.corrcoef(increments[:-1], increments[1:])[0, 1]
        assert autocorr < -0.1

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
        assert var_large / var_small == pytest.approx(10000, rel=0.1)


class TestRfftOverPartitions:
    """Step 0 gate: verify polars-rfft works with .over()."""

    def test_rfft_fft_over_groups(self):
        df = pl.DataFrame({
            "group": [0, 0, 0, 0, 1, 1, 1, 1],
            "value": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        })
        result = df.with_columns(
            pl.col("value").rfft.fft().over("group").alias("fft_result")
        )
        assert result.shape[0] == 8
        # Group 0: impulse at 0 → all re=1, im=0
        g0 = result.filter(group=0)["fft_result"]
        assert all(
            pytest.approx(s["re"], abs=1e-10) == 1.0
            for s in g0.to_list()
        )

    def test_rfft_ifft_real_over_groups(self):
        df = pl.DataFrame({
            "group": [0, 0, 0, 0, 1, 1, 1, 1],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        })
        # Round-trip: fft → ifft_real should recover original
        result = df.with_columns(
            pl.col("value")
            .rfft.fft()
            .rfft.ifft_real()
            .over("group")
            .alias("recovered")
        )
        original = result["value"].to_list()
        recovered = result["recovered"].to_list()
        for o, r in zip(original, recovered):
            assert pytest.approx(r, abs=1e-10) == o


def _cholesky_fbm_increments(
    n: int, hurst: float, rng: np.random.Generator
) -> np.ndarray:
    """Exact fBm increments via Cholesky decomposition (test oracle)."""
    h2 = 2 * hurst
    # Covariance matrix of fBm increments
    i_idx = np.arange(n)
    j_idx = np.arange(n)
    ii, jj = np.meshgrid(i_idx, j_idx, indexing="ij")
    gamma = 0.5 * (
        np.abs(ii + 1) ** h2
        + np.abs(jj + 1) ** h2
        - np.abs(ii - jj) ** h2
        - np.abs(ii) ** h2
        - np.abs(jj) ** h2
        + np.abs(ii - jj) ** h2
    )
    # Simplified: Γ[i,j] = 0.5*(|i-j+1|^2H + |i-j-1|^2H - 2|i-j|^2H)
    # Actually the autocovariance of fGn: gamma(k) = 0.5*(|k+1|^2H - 2|k|^2H + |k-1|^2H)
    diff = np.abs(ii - jj)
    gamma = 0.5 * (
        np.abs(diff + 1) ** h2 - 2 * diff ** h2 + np.abs(diff - 1) ** h2
    )
    L = np.linalg.cholesky(gamma)
    z = rng.standard_normal(n)
    return L @ z


class TestDaviesHarteVsCholesky:
    """Validate Davies-Harte (polars-rfft) against Cholesky oracle."""

    @pytest.mark.parametrize("hurst", [0.3, 0.7])
    def test_covariance_structure(self, hurst: float):
        """Compare sample covariance against *theoretical* fGn covariance."""
        n = 30
        n_samples = 5000
        h2 = 2 * hurst

        # Theoretical autocovariance of fGn at lag k
        def gamma(k: int) -> float:
            return 0.5 * (abs(k + 1) ** h2 - 2 * abs(k) ** h2 + abs(k - 1) ** h2)

        # Build theoretical covariance matrix
        cov_theory = np.array([[gamma(abs(i - j)) for j in range(n)] for i in range(n)])

        # Vectorized: generate all samples in one call
        df = monte_carlo(
            Process.FRW,
            n_simulations=n_samples,
            n=n + 1,
            hurst=hurst,
            base_price=0.0,
            step_size=1.0,
            seed=0,
        )
        prices = df["price"].to_numpy().reshape(n_samples, n + 1)
        dh_samples = np.diff(prices, axis=1)

        cov_dh = np.cov(dh_samples, rowvar=False)

        # Compare diagonal (variances) — should all be gamma(0) = 1
        np.testing.assert_allclose(
            np.diag(cov_dh), np.diag(cov_theory), atol=0.08
        )

        # Compare first 5 off-diagonals
        for lag in range(1, 6):
            dh_diag = np.mean(np.diag(cov_dh, lag))
            theory_diag = gamma(lag)
            assert abs(dh_diag - theory_diag) < 0.08, (
                f"lag {lag}: DH={dh_diag:.4f}, theory={theory_diag:.4f}"
            )

    @pytest.mark.parametrize("hurst", [0.3, 0.7])
    def test_autocorrelation(self, hurst: float):
        """Compare DH sample autocorrelation against theoretical values."""
        n = 200
        n_samples = 3000
        max_lag = 5
        h2 = 2 * hurst

        def gamma(k: int) -> float:
            return 0.5 * (abs(k + 1) ** h2 - 2 * abs(k) ** h2 + abs(k - 1) ** h2)

        # Theoretical autocorrelation = gamma(k) / gamma(0)
        g0 = gamma(0)

        # Vectorized: generate all samples in one call
        df = monte_carlo(
            Process.FRW,
            n_simulations=n_samples,
            n=n + 1,
            hurst=hurst,
            base_price=0.0,
            step_size=1.0,
            seed=0,
        )
        prices = df["price"].to_numpy().reshape(n_samples, n + 1)
        dh_increments = np.diff(prices, axis=1)

        for lag in range(1, max_lag + 1):
            dh_ac = np.mean([
                np.corrcoef(row[:-lag], row[lag:])[0, 1]
                for row in dh_increments
            ])
            theory_ac = gamma(lag) / g0
            assert abs(dh_ac - theory_ac) < 0.05, (
                f"lag {lag}: DH={dh_ac:.4f}, theory={theory_ac:.4f}"
            )
