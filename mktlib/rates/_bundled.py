"""Load bundled Treasury yield CSV data shipped with the package."""

from __future__ import annotations

import csv
from importlib.resources import files
from pathlib import Path

from ._disk_cache import RateRow

YEAR_RANGE = range(2000, 2027)  # update when new years are bundled


def bundled_path(year: int) -> Path | None:
    """Return filesystem path to bundled CSV for *year*, or None."""
    ref = files("mktlib.rates") / "_data" / f"{year}.csv"
    p = Path(str(ref))
    return p if p.exists() else None


def load_year(year: int) -> list[RateRow]:
    """Load bundled CSV data for *year*, returning [] if unavailable."""
    data_file = files("mktlib.rates") / "_data" / f"{year}.csv"
    try:
        text = data_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []

    from datetime import date

    rows: list[RateRow] = []
    for row in csv.DictReader(text.splitlines()):
        row_date = date.fromisoformat(row["date"])
        rates: RateRow = {"date": row_date}
        for key, val in row.items():
            if key.startswith("BC_") and val:
                try:
                    rates[key] = float(val) / 100.0
                except ValueError:
                    continue
        rows.append(rates)
    return rows
