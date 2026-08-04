"""Selecting a chain resolver, and failing loudly when the choice is impossible.

The distinction this file exists to protect: **asking for the accelerator and
silently not getting it must be impossible.** A benchmark or a CI job that meant
to exercise compiled code and quietly ran the Python kernel produces numbers that
look fine and mean nothing — and that is not hypothetical, it happened while
benchmarking this very feature, where both arms measured Python and agreed
perfectly, which read as correctness.

So ``"auto"`` falls back silently and ``"native"`` raises, and
:func:`active_scan_backend` exists so a caller can assert on the answer instead
of assuming it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import polars as pl
import pytest

from mktlib.backtest import (
    Condition,
    Crossover,
    EntryRef,
    Pct,
    ValueGT,
    ValueLT,
    active_scan_backend,
    get_scan_backend,
    run,
    set_scan_backend,
)
from mktlib.backtest import _backend
from tests.backtest.conftest import NATIVE_AVAILABLE


@pytest.fixture(autouse=True)
def _clean_backend_state() -> None:
    """Every test here starts from an unconfigured backend."""
    _backend._reset_for_tests()


def test_default_is_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing configured at all — no override, no environment.

    The env var must be cleared explicitly: CI legs set it, and a test asserting
    a default while the environment names a different one is asserting nothing
    about the default.
    """
    monkeypatch.delenv(_backend.ENV_VAR, raising=False)
    assert get_scan_backend() == "auto"


def test_env_var_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_backend.ENV_VAR, "python")
    assert get_scan_backend() == "python"
    assert active_scan_backend() == "python"


def test_env_var_is_case_and_space_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_backend.ENV_VAR, "  PYTHON ")
    assert get_scan_backend() == "python"


def test_a_bad_env_var_is_rejected_not_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo must not silently mean 'auto'."""
    monkeypatch.setenv(_backend.ENV_VAR, "rust")
    with pytest.raises(ValueError, match="not a valid backend"):
        get_scan_backend()


def test_set_scan_backend_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="unknown scan backend"):
        set_scan_backend("fastest")  # type: ignore[arg-type]


def test_set_scan_backend_overrides_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_backend.ENV_VAR, "python")
    set_scan_backend("auto")
    assert get_scan_backend() == "auto"


def test_python_is_always_available() -> None:
    """The fallback has no dependencies and cannot be unavailable."""
    set_scan_backend("python")
    assert active_scan_backend() == "python"


def test_auto_falls_back_silently_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_backend, "_module", False)
    monkeypatch.setattr(_backend, "_unavailable_reason", "simulated absence")
    set_scan_backend("auto")
    assert active_scan_backend() == "python"


def test_explicit_native_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point: asking for it and not getting it must be loud."""
    monkeypatch.setattr(_backend, "_module", False)
    monkeypatch.setattr(
        _backend, "_unavailable_reason", "mktlib-scan is not installed"
    )
    set_scan_backend("native")
    with pytest.raises(RuntimeError, match="not installed"):
        active_scan_backend()


def test_a_contract_mismatch_refuses_the_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same version, different rules, is the dangerous case.

    A mismatched contract means the two disagree about semantics — value ids,
    the NaN-null convention, the entry-wins rule. Running anyway would resolve
    trades under rules this release does not recognize, so it refuses.
    """

    class _WrongContract:
        CONTRACT_VERSION = _backend.REQUIRED_CONTRACT_VERSION + 1

    monkeypatch.setitem(__import__("sys").modules, "mktlib_scan", _WrongContract)
    monkeypatch.setattr(_backend, "_module", None)
    set_scan_backend("native")
    with pytest.raises(RuntimeError, match="contract version"):
        active_scan_backend()


def test_importing_mktlib_does_not_import_the_accelerator() -> None:
    """The guard must live behind a call, not at module import.

    `import mktlib` has to cost the same whether or not the accelerator is
    installed, so the import happens inside a cached lookup.
    """
    _backend._reset_for_tests()
    assert _backend._module is None, "the module cache must start empty"
    set_scan_backend("python")
    active_scan_backend()
    assert _backend._module is None, "resolving to python must not import it"


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="mktlib-scan is not installed")
def test_auto_prefers_the_accelerator_when_present() -> None:
    set_scan_backend("auto")
    assert active_scan_backend() == "native"


def test_ci_actually_got_the_backend_it_asked_for() -> None:
    """The guard that stops a CI leg from silently testing Python twice.

    If ``MKTLIB_SCAN_BACKEND=native`` is set — as the native CI leg does — and
    the wheel did not install, every other test in the suite would skip
    ``native`` and pass, reporting a green run that exercised nothing. This
    fails instead.
    """
    requested = os.environ.get(_backend.ENV_VAR, "").strip().lower()
    if requested != "native":
        pytest.skip("this environment did not ask for the native backend")
    assert NATIVE_AVAILABLE, (
        f"{_backend.ENV_VAR}=native but mktlib-scan is not importable — this run "
        "would have silently tested the Python backend twice"
    )
    _backend._reset_for_tests()
    assert active_scan_backend() == "native"


# --- end to end ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Anchored:
    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Condition:
        return ValueGT("close", Pct(EntryRef("close"), 4.0)) | ValueLT(
            "close", Pct(EntryRef("close"), -3.0)
        )


def _frame(n: int = 400) -> pl.DataFrame:
    state = 11
    close: list[float] = []
    px = 100.0
    for _ in range(n):
        state = (1_664_525 * state + 1_013_904_223) % (2**32)
        px *= 1.0 + ((state / 2**32) - 0.5) * 0.03
        close.append(px)
    return pl.DataFrame(
        {
            "date": pl.datetime_range(
                pl.datetime(2024, 1, 1),
                pl.datetime(2024, 1, 1) + pl.duration(minutes=n - 1),
                interval="1m",
                eager=True,
            ),
            "close": close,
        }
    ).with_columns(
        pl.col("close").alias("open"),
        (pl.col("close") * 1.002).alias("high"),
        (pl.col("close") * 0.998).alias("low"),
        pl.col("close").rolling_mean(4).fill_null(strategy="backward").alias("fast"),
        pl.col("close").rolling_mean(16).fill_null(strategy="backward").alias("slow"),
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="mktlib-scan is not installed")
def test_a_whole_backtest_is_identical_across_backends() -> None:
    """Not just the resolver — every artifact `run()` returns.

    The resolver feeds the snapshot columns, which feed the exits, which feed
    the trades and the returns. Comparing only the resolver's own output would
    miss a divergence that shows up downstream.
    """
    frame, strategy = _frame(), _Anchored()

    set_scan_backend("python")
    py = run(frame, strategy)
    set_scan_backend("native")
    native = run(frame, strategy)

    assert py.trades.height > 1, "fixture must trade, or this asserts nothing"
    assert py.trades.equals(native.trades)
    assert py.returns.equals(native.returns)
    assert py.signals.equals(native.signals)
