from __future__ import annotations

from datetime import time

from mktlib.scheduling.exchanges._us_holidays import (
    CHRISTMAS as CHRISTMAS,
    INDEPENDENCE_DAY as INDEPENDENCE_DAY,
    LABOR_DAY as LABOR_DAY,
    MEMORIAL_DAY as MEMORIAL_DAY,
    MLK_DAY as MLK_DAY,
    NEW_YEARS_DAY as NEW_YEARS_DAY,
    PRESIDENTS_DAY as PRESIDENTS_DAY,
    THANKSGIVING as THANKSGIVING,
    US_ADHOC_CLOSURES,
    US_EARLY_CLOSE_TIME,
    US_EARLY_CLOSES,
    US_RECURRING_HOLIDAYS,
    us_good_friday_closures,
)

# Re-export for backward compatibility
RECURRING_HOLIDAYS = US_RECURRING_HOLIDAYS
ADHOC_CLOSURES = US_ADHOC_CLOSURES
EARLY_CLOSES = US_EARLY_CLOSES
EARLY_CLOSE_TIME = US_EARLY_CLOSE_TIME
good_friday_closures = us_good_friday_closures

# --- Exchange constants ---

NYSE_OPEN = time(9, 30)
NYSE_CLOSE = time(16, 0)
NYSE_TZ = "America/New_York"
