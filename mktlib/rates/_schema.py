"""Read the bundled schema.csv to discover known Treasury yield fields."""
from __future__ import annotations

import csv
from functools import cache
from importlib.resources import files


@cache
def all_fields() -> list[str]:
    """Return the canonical ordered list of all known BC_* fields.

    Derived from column headers of schema.csv (everything after 'year').
    Replaces the former hardcoded _FIELDS list in _disk_cache.
    """
    text = (files("mktlib.rates") / "_data" / "schema.csv").read_text(encoding="utf-8")
    reader = csv.DictReader(text.splitlines())
    return [f for f in (reader.fieldnames or []) if f != "year"]


@cache
def load_schema() -> dict[int, list[str]]:
    """Return {year: [fields present that year]} from bundled schema.csv."""
    text = (files("mktlib.rates") / "_data" / "schema.csv").read_text(encoding="utf-8")
    result: dict[int, list[str]] = {}
    for row in csv.DictReader(text.splitlines()):
        year = int(row["year"])
        result[year] = [k for k, v in row.items() if k != "year" and v == "1"]
    return result
