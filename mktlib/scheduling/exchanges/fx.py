from __future__ import annotations

from datetime import time

from mktlib.scheduling.rules import AdhocClosure, EarlyClose, HolidayRule

# FX spot market: 24 hours, 5 days a week, no holiday closures.
# Sessions run 17:00 ET (previous day) → 17:00 ET (current day).

FX_OPEN = time(17, 0)   # 5:00 PM ET (previous day via open_offset=-1)
FX_CLOSE = time(17, 0)  # 5:00 PM ET
FX_TZ = "America/New_York"

RECURRING_HOLIDAYS: list[HolidayRule] = []
ADHOC_CLOSURES: list[AdhocClosure] = []
EARLY_CLOSES: list[EarlyClose] = []
