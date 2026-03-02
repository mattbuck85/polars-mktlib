from __future__ import annotations

from datetime import time

from mktlib.scheduling.exchanges._us_holidays import (
    US_ADHOC_CLOSURES,
    US_EARLY_CLOSES,
    US_RECURRING_HOLIDAYS,
)

# NASDAQ follows the NYSE calendar exactly — same holidays, early closes, and adhoc closures.

RECURRING_HOLIDAYS = US_RECURRING_HOLIDAYS
ADHOC_CLOSURES = US_ADHOC_CLOSURES
EARLY_CLOSES = US_EARLY_CLOSES

# --- Exchange constants ---

NASDAQ_OPEN = time(9, 30)
NASDAQ_CLOSE = time(16, 0)
NASDAQ_TZ = "America/New_York"
