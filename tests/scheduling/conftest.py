from __future__ import annotations

import pytest

from mktlib.scheduling import get_calendar


@pytest.fixture
def nyse():
    return get_calendar("XNYS")


@pytest.fixture
def lse():
    return get_calendar("XLON")


@pytest.fixture
def euronext():
    return get_calendar("XPAR")
