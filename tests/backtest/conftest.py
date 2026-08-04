"""Run the anchoring gates against every resolver backend that is available.

`mktlib/backtest/_scan.py` and the optional `mktlib-scan` accelerator implement
the same contract, and the whole reason it is safe to pick the faster one
silently is that their output is required to be **bit-identical** — not close.
That claim is only worth anything if it is checked, so the existing corpus is
parameterized over backends rather than a second corpus being written for the
new one. Two copies would drift; this cannot.

The fixture skips `native` when the accelerator is not installed, so a plain
`pip install -e ".[dev]"` without it still runs a full, meaningful suite.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from mktlib.backtest import _backend


def _native_available() -> bool:
    previous = _backend.get_scan_backend()
    try:
        _backend.set_scan_backend("auto")
        return _backend.active_scan_backend() == "native"
    finally:
        _backend.set_scan_backend(previous)


NATIVE_AVAILABLE = _native_available()


@pytest.fixture(params=["python", "native"])
def scan_backend(request: pytest.FixtureRequest) -> Iterator[str]:
    """Pin the chain resolver, and restore whatever was configured before.

    Restoration matters more than it looks: these tests run in the same process
    as everything else, and a leaked backend would silently change which
    implementation the *rest* of the suite exercised.
    """
    name: str = request.param
    if name == "native" and not NATIVE_AVAILABLE:
        pytest.skip("mktlib-scan is not installed")
    _backend.set_scan_backend(name)  # type: ignore[arg-type]
    try:
        yield name
    finally:
        _backend._reset_for_tests()
