"""pandera schemas for mktlib.data output DataFrames."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class GBMSchema(pa.DataFrameModel):
    """geometric_brownian_motion() output."""

    step: int = pa.Field(ge=0)
    price: float = pa.Field(gt=0)


class OUSchema(pa.DataFrameModel):
    """ornstein_uhlenbeck() output."""

    step: int = pa.Field(ge=0)
    value: float


class FRWSchema(pa.DataFrameModel):
    """fractional_random_walk() output."""

    step: int = pa.Field(ge=0)
    price: float


class OHLCVSchema(pa.DataFrameModel):
    """ticks_to_ohlcv() output (volume optional)."""

    bar: pl.UInt32 = pa.Field(ge=0)
    open: float  # noqa: A003
    high: float
    low: float
    close: float

    class Config:
        strict = False


class OHLCVWithVolumeSchema(OHLCVSchema):
    """ticks_to_ohlcv(volume=True) output."""

    volume: int = pa.Field(ge=0)

    class Config:
        strict = True


class MonteCarloGBMSchema(pa.DataFrameModel):
    """monte_carlo(Process.GBM) output."""

    simulation: int = pa.Field(ge=0)
    seed: int
    step: int = pa.Field(ge=0)
    price: float = pa.Field(gt=0)


class MonteCarloOUSchema(pa.DataFrameModel):
    """monte_carlo(Process.OU) output."""

    simulation: int = pa.Field(ge=0)
    seed: int
    step: int = pa.Field(ge=0)
    value: float


class MonteCarloFRWSchema(pa.DataFrameModel):
    """monte_carlo(Process.FRW) output."""

    simulation: int = pa.Field(ge=0)
    seed: int
    step: int = pa.Field(ge=0)
    price: float
