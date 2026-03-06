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

_DATA_DIR = (
    Path(__file__).resolve().parent.parent / "mktlib" / "rates" / "_data"
)
_SCHEMA_PATH = _DATA_DIR / "schema.csv"

_START_YEAR = 2000


def _load_known_fields() -> list[str]:
    """Load known BC_* fields from schema.csv, or return empty list."""
    if not _SCHEMA_PATH.exists():
        return []
    with _SCHEMA_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [f for f in (reader.fieldnames or []) if f != "year"]


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


def _write_csv(
    year: int, rows: list[dict[str, str]], all_fields: list[str]
) -> Path:
    """Write rows to a CSV file, returning the path."""
    used_fields = [f for f in all_fields if any(f in row for row in rows)]
    fieldnames = ["date", *used_fields]

    path = _DATA_DIR / f"{year}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_schema(
    year_fields: dict[int, list[str]], all_fields: list[str]
) -> None:
    """Write schema.csv with per-year field presence flags."""
    with _SCHEMA_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", *all_fields])
        for year in sorted(year_fields):
            present = set(year_fields[year])
            writer.writerow([year, *(1 if fld in present else 0 for fld in all_fields)])


def main() -> None:
    current_year = date.today().year
    years = range(_START_YEAR, current_year + 1)

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Refreshing Treasury data for {_START_YEAR}–{current_year}")

    # Start with known fields from existing schema.csv (preserves order)
    known = _load_known_fields()
    all_fields_set = set(known)
    all_fields_ordered = list(known)

    errors: list[int] = []
    year_data: dict[int, list[dict[str, str]]] = {}
    year_fields: dict[int, list[str]] = {}

    for year in years:
        try:
            rows = _fetch_and_parse(year)
            if rows:
                year_data[year] = rows
                # Discover BC_* fields present this year
                present: set[str] = set()
                for row in rows:
                    present.update(k for k in row if k.startswith("BC_"))
                year_fields[year] = sorted(present)
                # Track any new fields (append to preserve existing order)
                for field in sorted(present - all_fields_set):
                    all_fields_ordered.append(field)
                    all_fields_set.add(field)
        except Exception as exc:
            print(f"  ERROR for {year}: {exc}")
            errors.append(year)

    # Write per-year CSVs
    for year, rows in year_data.items():
        _write_csv(year, rows, all_fields_ordered)

    # Write updated schema.csv
    _write_schema(year_fields, all_fields_ordered)
    print(f"  Updated schema.csv ({len(all_fields_ordered)} fields, {len(year_fields)} years)")

    if errors:
        print(f"\nFailed years: {errors}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nDone. {len(year_data)} years written to {_DATA_DIR}")


if __name__ == "__main__":
    main()
