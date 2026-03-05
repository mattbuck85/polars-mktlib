"""Treasury.gov daily yield curve XML feed — fetch and parse."""

from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET
from datetime import date, datetime
from statistics import mean
from urllib.request import urlopen

from . import _bundled, _disk_cache
from ._disk_cache import RateRow

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
    "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
}

_BASE_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/pages/xml?data=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}"
)

# Module-level cache: year -> list of flat rate dicts
_cache: dict[int, list[RateRow]] = {}


def fetch_year(year: int) -> list[RateRow]:
    """Fetch and parse one year of Treasury yield data. Cached per-year."""
    if year in _cache:
        return _cache[year]

    cached = _disk_cache.load_year(year)
    if cached is not None:
        _cache[year] = cached
        return cached

    # Past years are finalized — seed disk cache from bundled data to avoid
    # unnecessary network requests.
    if year < date.today().year:
        bundled = _bundled.load_year(year)
        if bundled:
            _disk_cache.save_year(year, bundled)
            _cache[year] = bundled
            return bundled

    url = _BASE_URL.format(year=year)
    try:
        with urlopen(url, timeout=30) as resp:  # noqa: S310
            if resp.status != 200:
                raise ConnectionError(f"HTTP {resp.status}")  # noqa: TRY301
            data = resp.read()

        root = ET.fromstring(data)  # noqa: S314
        rows: list[RateRow] = []

        for entry in root.findall("atom:entry", _NS):
            props = entry.find("atom:content/m:properties", _NS)
            if props is None:
                continue

            date_el = props.find("d:NEW_DATE", _NS)
            if date_el is None or not date_el.text:
                continue

            row_date = datetime.fromisoformat(date_el.text).date()
            rates: RateRow = {"date": row_date}

            for child in props:
                tag = child.tag.split("}")[-1]  # strip namespace
                if tag.startswith("BC_") and child.text:
                    try:
                        rates[tag] = float(child.text) / 100.0
                    except ValueError:
                        continue

            if len(rates) > 1:  # has rate keys beyond "date"
                rows.append(rates)

        rows.sort(key=lambda x: x["date"])  # type: ignore[type-var]
    except Exception as exc:
        stale = _disk_cache.load_year(year, ignore_stale=True)
        bundled = _bundled.load_year(year)
        fallback = max(filter(None, [stale, bundled]), key=len, default=None)
        if fallback is not None:
            source = "disk cache" if fallback is stale else "bundled data"
            warnings.warn(
                f"Treasury.gov unreachable for {year}, using {source} ({len(fallback)} days)",
                stacklevel=2,
            )
            _disk_cache.save_year(year, fallback)
            _cache[year] = fallback
            return fallback
        raise type(exc)(
            f"Failed to fetch Treasury yield data for {year} from {url}: {exc}"
        ) from exc

    existing = _disk_cache.load_year(year, ignore_stale=True)
    if existing is None or len(rows) >= len(existing):
        _disk_cache.save_year(year, rows)
    _cache[year] = rows
    return rows


def _daily_rates(
    start: date,
    end: date,
    instrument: str,
) -> list[float]:
    """Collect daily rate values for *instrument* within [start, end]."""
    years = range(start.year, end.year + 1)
    return [
        row[instrument]  # type: ignore[index]
        for year in years
        for row in fetch_year(year)
        if start <= row["date"] <= end and instrument in row  # type: ignore[operator]
    ]


def fetch_average_rate(
    start: date,
    end: date,
    instrument: str = "BC_3MONTH",
) -> float:
    """Return the arithmetic mean of daily rates for the period."""
    daily = _daily_rates(start, end, instrument)
    if not daily:
        return 0.0
    return mean(daily)


def fetch_mean_rate(
    start: date,
    end: date,
    instrument: str = "BC_3MONTH",
    method: str = "arithmetic",
) -> float:
    """Return the mean of daily rates using the specified method.

    ``"arithmetic"`` — standard arithmetic mean.
    ``"geometric"`` — geometric mean of gross returns, converted back to a rate:
    ``exp(mean(log(1 + r))) - 1``.
    """
    rates = _daily_rates(start, end, instrument)
    if not rates:
        return 0.0
    if method == "geometric":
        for r in rates:
            if r <= -1.0:
                msg = f"Rate {r} <= -1.0; cannot compute geometric mean (log undefined)"
                raise ValueError(msg)
        return math.exp(mean(math.log(1 + r) for r in rates)) - 1
    return mean(rates)


def clear_cache(*, disk: bool = False) -> None:
    """Clear the module-level year cache (useful for testing).

    If *disk* is True, also delete all persistent disk-cached CSVs.
    """
    _cache.clear()
    if disk:
        _disk_cache.clear()
