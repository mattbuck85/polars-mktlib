"""Treasury yield curve rates for risk-free rate estimation."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "MeanMethod",
    "TreasuryRate",
    "get_mean_treasury_rate",
    "get_risk_free_rate",
    "get_treasury_rates",
    "get_treasury_spread",
]


class MeanMethod(StrEnum):
    """Averaging method for rate aggregation."""

    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


class TreasuryRate(StrEnum):
    """Treasury yield curve instruments available from Treasury.gov."""

    ONE_MONTH = "BC_1MONTH"
    TWO_MONTH = "BC_2MONTH"
    THREE_MONTH = "BC_3MONTH"
    FOUR_MONTH = "BC_4MONTH"
    SIX_MONTH = "BC_6MONTH"
    ONE_YEAR = "BC_1YEAR"
    TWO_YEAR = "BC_2YEAR"
    THREE_YEAR = "BC_3YEAR"
    FIVE_YEAR = "BC_5YEAR"
    SEVEN_YEAR = "BC_7YEAR"
    TEN_YEAR = "BC_10YEAR"
    TWENTY_YEAR = "BC_20YEAR"
    THIRTY_YEAR = "BC_30YEAR"
    THIRTY_YEAR_DISPLAY = "BC_30YEARDISPLAY"


def _parse_date(d: date | str) -> date:
    return date.fromisoformat(d) if isinstance(d, str) else d


def get_risk_free_rate(
    start: date | str,
    end: date | str,
    instrument: TreasuryRate = TreasuryRate.THREE_MONTH,
) -> float:
    """Fetch the average annualised risk-free rate for a date range.

    Returns the arithmetic mean of daily Treasury yields as a decimal
    (e.g., 0.0436 for 4.36%).

    Parameters
    ----------
    start, end
        Date range (inclusive). Accepts ``date`` objects or ISO strings
        (``"2024-01-01"``).
    instrument
        Which Treasury yield to use. Defaults to the 3-month T-bill,
        the standard academic proxy for the risk-free rate.
    """
    from . import _treasury

    return _treasury.fetch_average_rate(_parse_date(start), _parse_date(end), instrument.value)


def get_mean_treasury_rate(
    start: date | str,
    end: date | str,
    instrument: TreasuryRate = TreasuryRate.THREE_MONTH,
    method: MeanMethod = MeanMethod.ARITHMETIC,
) -> float:
    """Fetch the mean annualised Treasury rate for a date range.

    Parameters
    ----------
    start, end
        Date range (inclusive). Accepts ``date`` objects or ISO strings.
    instrument
        Which Treasury yield to use.
    method
        Averaging method — arithmetic (default) or geometric.
    """
    from . import _treasury

    return _treasury.fetch_mean_rate(
        _parse_date(start), _parse_date(end), instrument.value, method.value
    )


def get_treasury_rates(
    start: date | str,
    end: date | str,
    instrument: TreasuryRate | Sequence[TreasuryRate] | None = None,
) -> pl.DataFrame:
    """Fetch daily Treasury rates as a Polars DataFrame.

    Parameters
    ----------
    start, end
        Date range (inclusive). Accepts ``date`` objects or ISO strings.
    instrument
        Single instrument → 2-column DataFrame (``date``, ``rate``).
        Sequence or ``None`` (all) → wide DataFrame with one column per
        instrument, named by the enum member in lowercase
        (e.g. ``"three_month"``).
    """
    import polars as pl

    from . import _treasury

    start, end = _parse_date(start), _parse_date(end)

    # Single instrument — delegate to existing fetch_daily_rates
    if isinstance(instrument, TreasuryRate):
        rows = _treasury.fetch_daily_rates(start, end, instrument.value)
        return pl.DataFrame(
            [{"date": d, "rate": r} for d, r in rows],
            schema={"date": pl.Date, "rate": pl.Float64},
        )

    # Multi-instrument or all
    if instrument is not None:
        keys = [i.value for i in instrument]
    else:
        keys = None

    raw = _treasury.fetch_daily_rates_multi(start, end, keys)

    # One-pass DataFrame from row-dicts
    rows = [{"date": row_date, **rates} for row_date, rates in raw]

    # Determine rename mapping and desired column order
    value_to_name = {m.value: m.name.lower() for m in TreasuryRate}
    if keys is not None:
        rename = {k: value_to_name[k] for k in keys}
    else:
        rename = {m.value: m.name.lower() for m in TreasuryRate}

    if not rows:
        schema = {"date": pl.Date} | {n: pl.Float64 for n in rename.values()}
        return pl.DataFrame(schema=schema)

    df = pl.DataFrame(rows)
    # Only rename columns that exist in the data
    actual_rename = {k: v for k, v in rename.items() if k in df.columns}
    df = df.rename(actual_rename)
    # Build select list: date + all requested columns (missing ones as null)
    rate_cols = list(rename.values())
    select_exprs: list[pl.Expr] = [pl.col("date").cast(pl.Date)]
    for c in rate_cols:
        if c in df.columns:
            select_exprs.append(pl.col(c).cast(pl.Float64))
        else:
            select_exprs.append(pl.lit(None, dtype=pl.Float64).alias(c))
    return df.select(select_exprs)


def get_treasury_spread(
    start: date | str,
    end: date | str,
    long: TreasuryRate = TreasuryRate.TEN_YEAR,
    short: TreasuryRate = TreasuryRate.TWO_YEAR,
) -> pl.DataFrame:
    """Fetch the daily spread between two Treasury instruments.

    Returns a DataFrame with ``date`` and ``spread`` columns.
    Only includes days where both instruments have data.

    Parameters
    ----------
    start, end
        Date range (inclusive). Accepts ``date`` objects or ISO strings.
    long
        Longer-maturity instrument (default: 10-year).
    short
        Shorter-maturity instrument (default: 2-year).
    """
    import polars as pl

    long_name = long.name.lower()
    short_name = short.name.lower()
    df = get_treasury_rates(start, end, [long, short])
    return df.select(
        "date",
        (pl.col(long_name) - pl.col(short_name)).alias("spread"),
    ).drop_nulls("spread")
