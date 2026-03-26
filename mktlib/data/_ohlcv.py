from __future__ import annotations

import polars as pl
from polars_sdist import sample_lognormal


def ticks_to_ohlcv(
    ticks: pl.DataFrame,
    bar_size: int,
    *,
    column: str = "price",
    volume: bool = True,
    seed: int | None = None,
) -> pl.DataFrame:
    """Aggregate a tick-level numeric series into OHLCV bars.

    Parameters
    ----------
    ticks
        DataFrame with a column named *column* (output of any generator).
    bar_size
        Number of ticks per bar.  The last incomplete bar is dropped.
    column
        Column to aggregate. Defaults to ``"price"`` (GBM / fRW output).
        Pass ``"value"`` for Ornstein-Uhlenbeck output.
    volume
        Generate synthetic lognormal volume. False → no volume column.
    seed
        RNG seed for volume generation (ignored when volume=False).

    Returns
    -------
    pl.DataFrame
        Columns: ``bar``, ``open``, ``high``, ``low``, ``close``
        [, ``volume``].
    """
    if bar_size < 1:
        raise ValueError("bar_size must be >= 1")
    if column not in ticks.columns:
        raise ValueError(f"ticks DataFrame must contain a {column!r} column")

    raw = ticks.select(pl.col(column)).with_row_index("_idx")
    n_bars = len(raw) // bar_size
    if n_bars < 1:
        raise ValueError(
            f"Not enough rows for even 1 bar: got {len(raw)} rows with bar_size={bar_size}"
        )

    # Truncate to exact coverage (drop truly incomplete tail)
    raw = raw.head(n_bars * bar_size)

    result = (
        raw.with_columns((pl.col("_idx") // bar_size).alias("bar"))
        .group_by("bar", maintain_order=True)
        .agg(
            pl.col(column).first().alias("open"),
            pl.col(column).max().alias("high"),
            pl.col(column).min().alias("low"),
            pl.col(column).last().alias("close"),
        )
    )

    if volume:
        vol = sample_lognormal(n_bars, mu=10.0, sigma=1.0, seed=seed)
        result = result.with_columns(vol.cast(pl.Int64).alias("volume"))

    return result
