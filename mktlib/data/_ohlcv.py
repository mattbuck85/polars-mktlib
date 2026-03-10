from __future__ import annotations

import numpy as np
import polars as pl


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

    raw = ticks[column].to_numpy()
    n_bars = (len(raw) - 1) // bar_size
    if n_bars < 1:
        raise ValueError(
            f"Not enough rows for even 1 bar: got {len(raw)} rows with bar_size={bar_size}"
        )

    # Truncate to exact coverage
    raw = raw[: n_bars * bar_size + 1]

    idx = np.arange(n_bars) * bar_size
    open_arr = raw[idx]
    close_arr = raw[idx + bar_size]

    # Build (n_bars, bar_size+1) index array for high/low
    sub_idx = idx[:, None] + np.arange(bar_size + 1)
    sub_prices = raw[sub_idx]
    high_arr = sub_prices.max(axis=1)
    low_arr = sub_prices.min(axis=1)

    data: dict[str, object] = {
        "bar": range(n_bars),
        "open": open_arr,
        "high": high_arr,
        "low": low_arr,
        "close": close_arr,
    }

    if volume:
        rng = np.random.default_rng(seed)
        data["volume"] = rng.lognormal(mean=10.0, sigma=1.0, size=n_bars).astype(
            np.int64
        )

    return pl.DataFrame(data)
