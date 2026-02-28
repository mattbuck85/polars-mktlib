"""Load bundled Treasury yield CSV data shipped with the package."""
from __future__ import annotations

import csv
from datetime import date
from importlib.resources import files


def load_year(year: int) -> list[tuple[date, dict[str, float]]]:
    """Load bundled CSV data for *year*, returning [] if unavailable."""
    data_file = files("mktlib.rates") / "_data" / f"{year}.csv"
    try:
        text = data_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []

    rows: list[tuple[date, dict[str, float]]] = []
    for row in csv.DictReader(text.splitlines()):
        row_date = date.fromisoformat(row["date"])
        rates: dict[str, float] = {}
        for key, val in row.items():
            if key.startswith("BC_") and val:
                try:
                    rates[key] = float(val) / 100.0
                except ValueError:
                    continue
        rows.append((row_date, rates))
    return rows
