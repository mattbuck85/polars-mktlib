from __future__ import annotations


def test_version():
    import mktlib

    assert mktlib.__version__ == "0.2.0"


def test_scheduling_import():
    from mktlib.scheduling import ExchangeCalendar, get_calendar

    assert ExchangeCalendar is not None
    assert get_calendar is not None


def test_get_calendar_nyse():
    from mktlib.scheduling import get_calendar

    cal = get_calendar("XNYS")
    assert cal.name == "XNYS"


def test_get_calendar_alias():
    from mktlib.scheduling import get_calendar

    cal = get_calendar("NYSE")
    assert cal.name == "XNYS"


def test_get_calendar_lse():
    from mktlib.scheduling import get_calendar

    cal = get_calendar("XLON")
    assert cal.name == "XLON"


def test_lse_alias():
    from mktlib.scheduling import get_calendar

    assert get_calendar("LSE").name == "XLON"


def test_get_calendar_euronext():
    from mktlib.scheduling import get_calendar

    cal = get_calendar("XPAR")
    assert cal.name == "XPAR"


def test_euronext_alias():
    from mktlib.scheduling import get_calendar

    assert get_calendar("Euronext").name == "XPAR"
