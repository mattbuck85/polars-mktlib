from __future__ import annotations

from datetime import date, time

from mktlib.scheduling.exchanges.nyse import (
    CHRISTMAS,
    INDEPENDENCE_DAY,
    LABOR_DAY,
    MEMORIAL_DAY,
    MLK_DAY,
    NEW_YEARS_DAY,
    PRESIDENTS_DAY,
    THANKSGIVING,
)
from mktlib.scheduling.rules import (
    AdhocClosure,
    EarlyClose,
    day_after,
    fixed_date_if_weekday,
)

# --- CME-specific holidays ---
# CME only fully closes for New Year's and Christmas (plus Good Friday via special_closures_fn).
# The other US holidays are early-close days at 13:00.

# CME-specific adhoc closures — same as NYSE except Hurricane Sandy (NYSE floor only).
ADHOC_CLOSURES: list[AdhocClosure] = [
    AdhocClosure(
        name="September 11, 2001",
        dates=[
            date(2001, 9, 11),
            date(2001, 9, 12),
            date(2001, 9, 13),
            date(2001, 9, 14),
        ],
    ),
    AdhocClosure(
        name="President Ford National Day of Mourning",
        dates=[date(2007, 1, 2)],
    ),
    AdhocClosure(
        name="President Reagan National Day of Mourning",
        dates=[date(2004, 6, 11)],
    ),
    AdhocClosure(
        name="President Nixon National Day of Mourning",
        dates=[date(1994, 4, 27)],
    ),
    AdhocClosure(
        name="President H.W. Bush National Day of Mourning",
        dates=[date(2018, 12, 5)],
    ),
    AdhocClosure(
        name="President Carter National Day of Mourning",
        dates=[date(2025, 1, 9)],
    ),
]

RECURRING_HOLIDAYS = [NEW_YEARS_DAY, CHRISTMAS]

CME_EARLY_CLOSE_TIME = time(13, 0)

EARLY_CLOSES = [
    EarlyClose("MLK Day", CME_EARLY_CLOSE_TIME, rule=MLK_DAY),
    EarlyClose("Presidents Day", CME_EARLY_CLOSE_TIME, rule=PRESIDENTS_DAY),
    EarlyClose("Memorial Day", CME_EARLY_CLOSE_TIME, rule=MEMORIAL_DAY),
    EarlyClose(
        "Independence Day", CME_EARLY_CLOSE_TIME, rule=INDEPENDENCE_DAY
    ),
    EarlyClose("Labor Day", CME_EARLY_CLOSE_TIME, rule=LABOR_DAY),
    EarlyClose("Thanksgiving", CME_EARLY_CLOSE_TIME, rule=THANKSGIVING),
    EarlyClose(
        "Black Friday",
        CME_EARLY_CLOSE_TIME,
        compute_fn=day_after(THANKSGIVING),
    ),
    EarlyClose(
        "Christmas Eve",
        CME_EARLY_CLOSE_TIME,
        compute_fn=fixed_date_if_weekday(12, 24, start_year=1993),
    ),
]

# --- CME-specific constants ---

CME_RTH_OPEN = time(9, 30)
CME_RTH_CLOSE = time(16, 15)

CME_GLOBEX_OPEN = time(18, 0)
CME_GLOBEX_CLOSE = time(17, 0)

CME_TZ = "America/New_York"
