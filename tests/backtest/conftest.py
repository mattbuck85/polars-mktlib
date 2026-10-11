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

from tests._scan_backend import NATIVE_AVAILABLE, scan_backend

__all__ = ["NATIVE_AVAILABLE", "scan_backend"]
