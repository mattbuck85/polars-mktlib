"""Persistent disk cache for Treasury yield data (~/.cache/mktlib/rates/)."""

from __future__ import annotations

import csv
import time
from datetime import date
from pathlib import Path

from ._schema import all_fields

type RateRow = dict[str, date | float]

_CACHE_DIR = Path.home() / ".cache" / "mktlib" / "rates"
_MAX_AGE = 7 * 86400  # 1 week in seconds


def _is_stale(path: Path, year: int) -> bool:
    """Return True if *path* is stale and should be re-fetched."""
    if year < date.today().year:
        return False  # past years are finalized
    return path.stat().st_mtime < time.time() - _MAX_AGE


def parse_csv(text: str) -> list[RateRow]:
    """Parse a Treasury yield CSV string into rate rows."""
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


def load_year(
    year: int, *, ignore_stale: bool = False
) -> list[RateRow] | None:
    """Load year data from disk cache. Returns None if missing or stale.

    If *ignore_stale* is True, skip the staleness check and return data
    whenever the file exists — useful as a last-resort fallback.
    """
    path = _CACHE_DIR / f"{year}.csv"
    if not path.exists():
        return None
    if not ignore_stale and _is_stale(path, year):
        return None

    return parse_csv(path.read_text(encoding="utf-8"))


def save_year(year: int, rows: list[RateRow]) -> None:
    """Write year data to disk cache as CSV (values stored as percentages)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which fields actually have data
    used_fields = [f for f in all_fields() if any(f in row for row in rows)]
    fieldnames = ["date", *used_fields]

    path = _CACHE_DIR / f"{year}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            csv_row: dict[str, str] = {"date": row["date"].isoformat()}  # type: ignore[union-attr]
            for field in used_fields:
                if field in row:
                    csv_row[field] = f"{row[field] * 100:.2f}"  # type: ignore[str-bytes-safe]
            writer.writerow(csv_row)


def clear() -> None:
    """Delete all cached CSV files."""
    if _CACHE_DIR.exists():
        for f in _CACHE_DIR.glob("*.csv"):
            f.unlink()
