from __future__ import annotations

import datetime as dt
from typing import Any, Protocol

import polars as pl


class PandasConvertible(Protocol):
    """Duck-type protocol for objects convertible via ``to_frame()`` (e.g. ``pd.Series``)."""

    def to_frame(self) -> Any: ...


ReturnsInput = pl.DataFrame | pl.Series | PandasConvertible
"""Accepted input types for returns / benchmark data."""


def coerce_returns(data: ReturnsInput) -> pl.DataFrame:
    """Coerce returns input to a pl.DataFrame with ``date`` and ``return`` columns.

    Accepts ``pl.Series``, ``pl.DataFrame``, or ``pd.Series`` (duck-typed via
    :class:`PandasConvertible`).
    """
    if isinstance(data, pl.DataFrame):
        return _validate_returns_df(data)
    if isinstance(data, pl.Series):
        return _series_to_df(data)
    return _pandas_to_df(data)


def coerce_benchmark(data: ReturnsInput | None) -> pl.DataFrame | None:
    """Coerce benchmark input, returning *None* if input is *None*."""
    if data is None:
        return None
    return coerce_returns(data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pandas_to_df(series: PandasConvertible) -> pl.DataFrame:
    """Convert a pandas Series with DatetimeIndex to pl.DataFrame."""
    frame = series.to_frame()
    df = pl.from_pandas(frame, include_index=True)  # type: ignore[arg-type]
    cols = df.columns
    if len(cols) < 2:
        msg = "pandas Series must have a DatetimeIndex"
        raise ValueError(msg)
    df = df.rename({cols[0]: "date", cols[1]: "return"})
    return _normalize_date_col(df.select("date", "return"))


def _series_to_df(series: pl.Series) -> pl.DataFrame:
    """Convert a bare ``pl.Series`` to a DataFrame with synthetic business-day dates."""
    n = len(series)
    dates: list[dt.date] = []
    current = dt.date(2000, 1, 3)  # a Monday
    while len(dates) < n:
        if current.weekday() < 5:
            dates.append(current)
        current += dt.timedelta(days=1)
    return pl.DataFrame({"date": dates, "return": series})


def _validate_returns_df(df: pl.DataFrame) -> pl.DataFrame:
    """Validate and normalise a ``pl.DataFrame`` to have ``date`` and ``return`` columns."""
    if "date" in df.columns and "return" in df.columns:
        return _normalize_date_col(df.select("date", "return"))
    # Try to infer: first date-like column → date, first float column → return
    date_col = return_col = None
    for col in df.columns:
        dtype = df[col].dtype
        if date_col is None and (
            dtype == pl.Date or isinstance(dtype, pl.Datetime)
        ):
            date_col = col
        elif return_col is None and dtype.is_float():
            return_col = col
    if date_col is None or return_col is None:
        msg = f"Cannot infer 'date' and 'return' columns from: {df.columns}"
        raise ValueError(msg)
    return _normalize_date_col(
        df.select(
            pl.col(date_col).alias("date"), pl.col(return_col).alias("return")
        )
    )


def _normalize_date_col(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure the ``date`` column is ``pl.Date``."""
    dtype = df["date"].dtype
    if dtype == pl.Date:
        return df
    if isinstance(dtype, pl.Datetime):
        return df.with_columns(
            pl.col("date").dt.replace_time_zone(None).cast(pl.Date)
        )
    return df.with_columns(pl.col("date").cast(pl.Date))
