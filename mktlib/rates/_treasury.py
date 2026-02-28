"""Treasury.gov daily yield curve XML feed — fetch and parse."""
from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from datetime import date, datetime
from statistics import mean
from urllib.request import urlopen

from . import _bundled

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

# Module-level cache: year -> list of (date, {field: rate_decimal})
_cache: dict[int, list[tuple[date, dict[str, float]]]] = {}


def _fetch_year(year: int) -> list[tuple[date, dict[str, float]]]:
    """Fetch and parse one year of Treasury yield data. Cached per-year."""
    if year in _cache:
        return _cache[year]

    url = _BASE_URL.format(year=year)
    try:
        with urlopen(url, timeout=30) as resp:  # noqa: S310
            data = resp.read()
    except Exception as exc:
        bundled = _bundled.load_year(year)
        if bundled:
            warnings.warn(
                f"Treasury.gov unreachable for {year}, using bundled data ({len(bundled)} days)",
                stacklevel=2,
            )
            _cache[year] = bundled
            return bundled
        raise ConnectionError(
            f"Failed to fetch Treasury yield data for {year} from {url}: {exc}"
        ) from exc

    root = ET.fromstring(data)  # noqa: S314
    rows: list[tuple[date, dict[str, float]]] = []

    for entry in root.findall("atom:entry", _NS):
        props = entry.find("atom:content/m:properties", _NS)
        if props is None:
            continue

        date_el = props.find("d:NEW_DATE", _NS)
        if date_el is None or not date_el.text:
            continue

        row_date = datetime.fromisoformat(date_el.text).date()
        rates: dict[str, float] = {}

        for child in props:
            tag = child.tag.split("}")[-1]  # strip namespace
            if tag.startswith("BC_") and child.text:
                try:
                    rates[tag] = float(child.text) / 100.0
                except ValueError:
                    continue

        rows.append((row_date, rates))

    _cache[year] = rows
    return rows


def fetch_daily_rates(
    start: date,
    end: date,
    instrument: str = "BC_3MONTH",
) -> list[tuple[date, float]]:
    """Return daily rates for *instrument* within [start, end]."""
    years = range(start.year, end.year + 1)
    result: list[tuple[date, float]] = []

    for year in years:
        for row_date, rates in _fetch_year(year):
            if start <= row_date <= end and instrument in rates:
                result.append((row_date, rates[instrument]))

    result.sort(key=lambda x: x[0])
    return result


def fetch_average_rate(
    start: date,
    end: date,
    instrument: str = "BC_3MONTH",
) -> float:
    """Return the arithmetic mean of daily rates for the period."""
    daily = fetch_daily_rates(start, end, instrument)
    if not daily:
        return 0.0
    return mean(r for _, r in daily)


def clear_cache() -> None:
    """Clear the module-level year cache (useful for testing)."""
    _cache.clear()
