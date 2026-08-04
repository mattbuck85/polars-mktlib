"""Which implementation resolves the entry/exit chain.

``_scan.py`` is the reference implementation and always works. ``mktlib-scan``
is an optional compiled accelerator implementing the same contract, installed
via ``pip install mktlib[fast]``. Selecting between them is this module's whole
job.

**The default is "use it if it is there".** That is only defensible because the
two are held to *bit-identical* output rather than "close enough": the outputs
are indices and booleans, the only float operations are comparisons and
``nextafter``, and mktlib's own equivalence corpus plus all of its frozen golden
baselines run against both backends in CI. A backtester whose numbers depended on
which wheels happened to install would be a correctness hazard; one whose numbers
are provably identical is just faster.

Two escape hatches exist anyway, because "provably identical" is a claim about
today's code and you may want to check it yourself:

* :func:`set_scan_backend`, or the ``MKTLIB_SCAN_BACKEND`` environment variable,
  to pin ``"python"`` or ``"native"``.
* :func:`active_scan_backend`, to ask which one *would* run.

Asking for ``"native"`` explicitly and not having it **raises**. Asking for
``"auto"`` and not having it falls back silently — the difference matters,
because a benchmark or a CI job that meant to exercise the accelerator and
quietly did not is worse than one that fails.

The import happens inside a cached lookup, never at ``mktlib`` import time, so
``import mktlib`` costs nothing whether or not the accelerator is installed.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Literal, cast

#: What a caller may ask for.
BackendName = Literal["auto", "python", "native"]

#: What can actually run.
ResolvedBackend = Literal["python", "native"]

#: Read once per resolution, so setting it in a subprocess works as expected.
ENV_VAR = "MKTLIB_SCAN_BACKEND"

#: The contract version of ``mktlib-scan`` this release speaks. A mismatch means
#: the two disagree about semantics — value ids, the NaN-null convention, the
#: entry-wins rule — so it refuses rather than resolving trades under rules it
#: does not recognize. See the accelerator's ``CONTRACT.md``.
REQUIRED_CONTRACT_VERSION = 1

_VALID: tuple[BackendName, ...] = ("auto", "python", "native")

_override: BackendName | None = None

#: ``None`` = not yet looked up. ``False`` = looked up and unavailable, with the
#: reason in ``_unavailable_reason``. Cached because the answer cannot change
#: within a process and the import is not free.
_module: ModuleType | None | Literal[False] = None
_unavailable_reason = ""


def _load() -> ModuleType | None:
    """Import the accelerator once, or record why it is unusable."""
    global _module, _unavailable_reason  # noqa: PLW0603
    if _module is not None:
        return _module or None

    # `importlib.import_module` rather than a plain `import mktlib_scan`, and
    # not because of style: mktlib type-checks under pyright strict, and a
    # static import of a package that is *meant* to be absent is an unresolved
    # import in every environment without the extra. Suppressing that with an
    # ignore comment would also suppress a genuinely misspelled module name.
    # This is an optional dependency resolved at runtime, so saying so directly
    # leaves the type checker with nothing to be wrong about.
    try:
        module = importlib.import_module("mktlib_scan")
    except ImportError as exc:
        _module = False
        _unavailable_reason = (
            f"mktlib-scan is not installed ({exc}). "
            f"Install it with: pip install 'mktlib[fast]'"
        )
        return None

    found = getattr(module, "CONTRACT_VERSION", None)
    if found != REQUIRED_CONTRACT_VERSION:
        _module = False
        _unavailable_reason = (
            f"mktlib-scan speaks contract version {found!r}, but this release of "
            f"mktlib requires {REQUIRED_CONTRACT_VERSION!r}. Install a matching "
            f"version: pip install 'mktlib[fast]'"
        )
        return None

    _module = module
    return _module


def set_scan_backend(name: BackendName) -> None:
    """Pin the chain resolver to ``"python"``, ``"native"``, or ``"auto"``.

    Overrides :data:`ENV_VAR` for the rest of the process. ``"auto"`` restores
    the default (use the accelerator when it is importable).

    Setting ``"native"`` does not itself raise if the accelerator is missing —
    the failure surfaces at the next resolution, where it can name the reason.
    """
    if name not in _VALID:
        msg = f"unknown scan backend {name!r}; expected one of {_VALID}"
        raise ValueError(msg)
    global _override  # noqa: PLW0603
    _override = name


def get_scan_backend() -> BackendName:
    """What is configured — not necessarily what will run.

    ``"auto"`` here means "decide at resolution time"; call
    :func:`active_scan_backend` for the decision.
    """
    if _override is not None:
        return _override
    env = os.environ.get(ENV_VAR, "auto").strip().lower()
    if env not in _VALID:
        msg = (
            f"{ENV_VAR}={env!r} is not a valid backend; expected one of {_VALID}"
        )
        raise ValueError(msg)
    return cast("BackendName", env)


def active_scan_backend() -> ResolvedBackend:
    """Which implementation would actually resolve a chain right now.

    Useful in a benchmark or a CI job that means to exercise a specific backend:
    assert on this rather than assuming, because "the accelerator was not
    installed" and "the accelerator is no faster" look identical from the
    outside.

    Raises if ``"native"`` was requested explicitly and is unavailable.
    """
    requested = get_scan_backend()
    if requested == "python":
        return "python"
    if _load() is not None:
        return "native"
    if requested == "native":
        raise RuntimeError(_unavailable_reason)
    return "python"


def native_module() -> ModuleType:
    """The loaded accelerator. Call only when the backend resolved to native."""
    module = _load()
    if module is None:  # pragma: no cover - guarded by active_scan_backend
        raise RuntimeError(_unavailable_reason)
    return module


def _reset_for_tests() -> None:
    """Drop the cached override and import result."""
    global _override, _module, _unavailable_reason  # noqa: PLW0603
    _override = None
    _module = None
    _unavailable_reason = ""
