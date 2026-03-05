#!/usr/bin/env python3
"""Fetch Treasury yield curve data and write bundled CSVs.

Standalone script — no package install required. Uses only stdlib.
Writes one CSV per year to mktlib/rates/_data/.
"""
from __future__ import annotations

import csv
import sys
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from urllib.request import urlopen

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

# All BC_ fields that may appear in Treasury data
_FIELDS = [
    "BC_1MONTH",
    "BC_2MONTH",
    "BC_3MONTH",
    "BC_4MONTH",
    "BC_6MONTH",
    "BC_1YEAR",
    "BC_2YEAR",
    "BC_3YEAR",
    "BC_5YEAR",
    "BC_7YEAR",
    "BC_10YEAR",
    "BC_20YEAR",
    "BC_30YEAR",
    "BC_30YEARDISPLAY",
]

_DATA_DIR = (
    Path(__file__).resolve().parent.parent / "mktlib" / "rates" / "_data"
)

_START_YEAR = 2000


def _fetch_and_parse(year: int) -> list[dict[str, str]]:
    """Fetch one year of data from Treasury.gov and return rows as dicts."""
    url = _BASE_URL.format(year=year)
    print(f"  Fetching {year}...", end=" ", flush=True)
    with urlopen(url, timeout=60) as resp:  # noqa: S310
        data = resp.read()

    root = ET.fromstring(data)  # noqa: S314
    rows: list[dict[str, str]] = []

    for entry in root.findall("atom:entry", _NS):
        props = entry.find("atom:content/m:properties", _NS)
        if props is None:
            continue

        date_el = props.find("d:NEW_DATE", _NS)
        if date_el is None or not date_el.text:
            continue

        row_date = datetime.fromisoformat(date_el.text).date()
        row: dict[str, str] = {"date": row_date.isoformat()}

        for child in props:
            tag = child.tag.split("}")[-1]
            if tag.startswith("BC_") and child.text:
                try:
                    float(child.text)  # validate it's numeric
                    row[tag] = child.text
                except ValueError:
                    continue

        rows.append(row)

    rows.sort(key=lambda r: r["date"])
    print(f"{len(rows)} days")
    return rows


def _write_csv(year: int, rows: list[dict[str, str]]) -> Path:
    """Write rows to a CSV file, returning the path."""
    # Determine which fields actually have data this year
    used_fields = [f for f in _FIELDS if any(f in row for row in rows)]
    fieldnames = ["date", *used_fields]

    path = _DATA_DIR / f"{year}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    current_year = date.today().year
    years = range(_START_YEAR, current_year + 1)

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Refreshing Treasury data for {_START_YEAR}–{current_year}")

    errors: list[int] = []
    for year in years:
        try:
            rows = _fetch_and_parse(year)
            if rows:
                _write_csv(year, rows)
        except Exception as exc:
            print(f"  ERROR for {year}: {exc}")
            errors.append(year)

    if errors:
        print(f"\nFailed years: {errors}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nDone. {len(years)} years written to {_DATA_DIR}")


if __name__ == "__main__":
    main()
