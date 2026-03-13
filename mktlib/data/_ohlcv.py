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

    raw = ticks[column]
    n_bars = (len(raw) - 1) // bar_size
    if n_bars < 1:
        raise ValueError(
            f"Not enough rows for even 1 bar: got {len(raw)} rows with bar_size={bar_size}"
        )

    # Truncate to exact coverage
    raw = raw.slice(0, n_bars * bar_size + 1)

    # Open/close at bar boundaries
    open_idx = list(range(0, n_bars * bar_size, bar_size))
    close_idx = [i + bar_size for i in open_idx]
    open_arr = raw.gather(open_idx)
    close_arr = raw.gather(close_idx)

    # High/low per bar
    highs: list[float] = []
    lows: list[float] = []
    for i in range(n_bars):
        bar_slice = raw.slice(i * bar_size, bar_size + 1)
        highs.append(bar_slice.max())  # type: ignore[arg-type]
        lows.append(bar_slice.min())  # type: ignore[arg-type]

    data: dict[str, object] = {
        "bar": range(n_bars),
        "open": open_arr,
        "high": highs,
        "low": lows,
        "close": close_arr,
    }

    if volume:
        vol = sample_lognormal(n_bars, mu=10.0, sigma=1.0, seed=seed)
        data["volume"] = vol.cast(pl.Int64)

    return pl.DataFrame(data)
