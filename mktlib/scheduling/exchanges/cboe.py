from __future__ import annotations

from datetime import time

from mktlib.scheduling.exchanges._us_holidays import (
    THANKSGIVING,
    US_ADHOC_CLOSURES,
    US_EARLY_CLOSE_TIME,
    US_RECURRING_HOLIDAYS,
)
from mktlib.scheduling.rules import EarlyClose, day_after

# CBOE Options follows NYSE holidays. Normal close is 16:15.
# Early close schedule differs from NYSE — CBOE Futures (XCBF in exchange_calendars)
# only has Black Friday as an early close (no Independence Day Eve, no Christmas Eve).

RECURRING_HOLIDAYS = US_RECURRING_HOLIDAYS
ADHOC_CLOSURES = US_ADHOC_CLOSURES

EARLY_CLOSES: list[EarlyClose] = [
    EarlyClose(
        "Black Friday", US_EARLY_CLOSE_TIME, compute_fn=day_after(THANKSGIVING)
    ),
]

# --- Exchange constants ---

CBOE_OPEN = time(9, 30)
CBOE_CLOSE = time(16, 15)
CBOE_TZ = "America/New_York"
