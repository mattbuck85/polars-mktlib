"""Schema validation tests for mktlib.data generators."""

from __future__ import annotations

import polars as pl
import pytest

from mktlib.data import (
    Process,
    fractional_random_walk,
    geometric_brownian_motion,
    monte_carlo,
    ornstein_uhlenbeck,
    ticks_to_ohlcv,
)

from tests.schemas.data import (
    FRWSchema,
    GBMSchema,
    MonteCarloFRWSchema,
    MonteCarloGBMSchema,
    MonteCarloOUSchema,
    OHLCVSchema,
    OHLCVWithVolumeSchema,
    OUSchema,
)


class TestGBMSchema:
    def test_default_params(self):
        df = geometric_brownian_motion(100, seed=42)
        GBMSchema.validate(df)

    def test_custom_params(self):
        df = geometric_brownian_motion(
            50, base_price=200.0, drift=0.05, volatility=0.2, seed=42
        )
        GBMSchema.validate(df)


class TestFRWSchema:
    @pytest.mark.parametrize("hurst", [0.3, 0.5, 0.7])
    def test_hurst_values(self, hurst: float):
        df = fractional_random_walk(100, hurst=hurst, seed=42)
        FRWSchema.validate(df)


class TestOUSchema:
    def test_default_params(self):
        df = ornstein_uhlenbeck(100, seed=42)
        OUSchema.validate(df)

    def test_custom_params(self):
        df = ornstein_uhlenbeck(100, theta=1.5, mu=50.0, sigma=2.0, seed=42)
        OUSchema.validate(df)


class TestOHLCVSchema:
    def test_with_volume(self):
        ticks = geometric_brownian_motion(500, seed=42)
        df = ticks_to_ohlcv(ticks, 10, volume=True, seed=42)
        OHLCVWithVolumeSchema.validate(df)
        # Cross-column invariants
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

    def test_without_volume(self):
        ticks = geometric_brownian_motion(500, seed=42)
        df = ticks_to_ohlcv(ticks, 10, volume=False)
        OHLCVSchema.validate(df)
        assert "volume" not in df.columns

    def test_large_bar_size(self):
        ticks = geometric_brownian_motion(1000, seed=42)
        df = ticks_to_ohlcv(ticks, 100, volume=True, seed=42)
        OHLCVWithVolumeSchema.validate(df)
        assert df.shape[0] == 10


class TestMonteCarloSchema:
    def test_gbm(self):
        df = monte_carlo(Process.GBM, n_simulations=3, seed=42, n=50)
        MonteCarloGBMSchema.validate(df)
        assert df.shape == (150, 4)

    def test_ou(self):
        df = monte_carlo(Process.OU, n_simulations=3, seed=42, n=50)
        MonteCarloOUSchema.validate(df)
        assert df.shape == (150, 4)

    def test_frw(self):
        df = monte_carlo(Process.FRW, n_simulations=3, seed=42, n=50)
        MonteCarloFRWSchema.validate(df)
        assert df.shape == (150, 4)

    @pytest.mark.parametrize("hurst", [0.3, 0.7])
    def test_frw_hurst(self, hurst: float):
        df = monte_carlo(
            Process.FRW, n_simulations=2, seed=42, n=30, hurst=hurst
        )
        MonteCarloFRWSchema.validate(df)

    def test_callable(self):
        df = monte_carlo(
            geometric_brownian_motion, n_simulations=3, seed=42, n=50
        )
        # Callable (serial loop) path uses pl.lit(i) → Int32 for simulation,
        # vs vectorized path's Int64 from pl.arange.  Validate structurally.
        assert df.columns == ["simulation", "seed", "step", "price"]
        assert df["price"].dtype == pl.Float64
        assert (df["price"] > 0).all()
        assert (df["step"] >= 0).all()
        assert (df["simulation"] >= 0).all()
