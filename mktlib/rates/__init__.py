"""Treasury yield curve rates for risk-free rate estimation."""

from __future__ import annotations

import itertools
import operator
from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from functools import partial, reduce
from typing import TYPE_CHECKING

from . import _treasury

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "MeanMethod",
    "TreasuryRate",
    "get_mean_treasury_rate",
    "get_risk_free_rate",
    "get_treasury_rates",
    "get_treasury_spread",
    "get_treasury_spread_matrix",
]


class MeanMethod(StrEnum):
    """Averaging method for rate aggregation."""

    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


class TreasuryRate(StrEnum):
    """Treasury yield curve instruments available from Treasury.gov."""

    ONE_MONTH = "BC_1MONTH"
    ONE_AND_HALF_MONTH = "BC_1_5MONTH"
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
    return _treasury.fetch_average_rate(
        _parse_date(start), _parse_date(end), instrument.value
    )


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
    return _treasury.fetch_mean_rate(
        _parse_date(start), _parse_date(end), instrument.value, method.value
    )


_fetch_years = partial(map, _treasury.fetch_year)


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

    start, end = _parse_date(start), _parse_date(end)

    df = pl.DataFrame(
        reduce(
            operator.iadd, _fetch_years(range(start.year, end.year + 1)), []
        )
    ).filter(pl.col("date").is_between(start, end))

    # Single instrument → 2-column DataFrame (date, rate)
    if isinstance(instrument, TreasuryRate):
        key = instrument.value
        if df.is_empty():
            return pl.DataFrame(schema={"date": pl.Date, "rate": pl.Float64})
        if key not in df.columns:
            return pl.DataFrame(schema={"date": pl.Date, "rate": pl.Float64})
        return df.select(
            pl.col("date").cast(pl.Date),
            pl.col(key).cast(pl.Float64).alias("rate"),
        ).drop_nulls("rate")

    # Multi-instrument or all
    if instrument is not None:
        keys = [i.value for i in instrument]
    else:
        keys = None

    # Determine rename mapping and desired column order
    value_to_name = {m.value: m.name.lower() for m in TreasuryRate}
    if keys is not None:
        rename = {k: value_to_name[k] for k in keys}
    else:
        rename = {m.value: m.name.lower() for m in TreasuryRate}

    if df.is_empty():
        schema = {"date": pl.Date} | {n: pl.Float64 for n in rename.values()}
        return pl.DataFrame(schema=schema)

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


def get_treasury_spread_matrix(
    start: date | str,
    end: date | str,
    instruments: Sequence[TreasuryRate] | None = None,
) -> pl.DataFrame:
    """Daily spreads for every pair of Treasury instruments (cross join).

    Returns a wide DataFrame with ``date`` plus one ``Float64`` column per
    unique tenor pair, named ``spread_{long}_{short}`` (e.g.
    ``spread_ten_year_two_year``) and computed as ``long - short`` where
    *long* is the longer-maturity instrument. Nulls propagate per-column on
    days where either leg is missing; rows are **not** dropped (unlike the
    single-pair :func:`get_treasury_spread`).

    Parameters
    ----------
    start, end
        Date range (inclusive). Accepts ``date`` objects or ISO strings.
    instruments
        Instruments to pair. ``None`` (default) uses every tenor except
        ``TreasuryRate.THIRTY_YEAR_DISPLAY`` (a duplicate of ``THIRTY_YEAR``).
        Pairing follows maturity order regardless of the argument order, so
        the ``long - short`` orientation and column names are stable.
    """
    import polars as pl

    requested = (
        set(TreasuryRate) - {TreasuryRate.THIRTY_YEAR_DISPLAY}
        if instruments is None
        else set(instruments)
    )
    # Iterate in enum declaration order (ascending maturity) so each pair is
    # (short, long) irrespective of how the caller ordered ``instruments``.
    ordered = [m for m in TreasuryRate if m in requested]
    pairs = list(itertools.combinations(ordered, 2))

    df = get_treasury_rates(start, end, ordered)

    spread_exprs = [
        (pl.col(long.name.lower()) - pl.col(short.name.lower())).alias(
            f"spread_{long.name.lower()}_{short.name.lower()}"
        )
        for short, long in pairs
    ]

    if df.is_empty():
        schema = {"date": pl.Date} | {
            f"spread_{long.name.lower()}_{short.name.lower()}": pl.Float64
            for short, long in pairs
        }
        return pl.DataFrame(schema=schema)

    return df.select(pl.col("date"), *spread_exprs)
