from __future__ import annotations

from datetime import time

# CME shares all NYSE holidays and early-close generators.
# Re-export the relevant symbols for use by calendar.py factory functions.
from mktlib.scheduling.exchanges.nyse import (
    ADHOC_CLOSURES as ADHOC_CLOSURES,
    EARLY_CLOSES as EARLY_CLOSES,
    RECURRING_HOLIDAYS as RECURRING_HOLIDAYS,
    black_friday_early_closes as black_friday_early_closes,
    christmas_eve_early_closes as christmas_eve_early_closes,
    good_friday_closures as good_friday_closures,
    independence_day_early_closes as independence_day_early_closes,
)

# --- CME-specific constants ---

CME_RTH_OPEN = time(9, 30)
CME_RTH_CLOSE = time(16, 15)

CME_GLOBEX_OPEN = time(18, 0)
CME_GLOBEX_CLOSE = time(17, 0)

CME_TZ = "America/New_York"
CME_EARLY_CLOSE_TIME = time(13, 0)
