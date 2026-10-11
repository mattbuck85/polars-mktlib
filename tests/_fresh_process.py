"""Run a snippet in fresh interpreters that differ in their environment.

Some results could depend on state that is fixed when a process starts:
the string hash seed (the iteration order of a ``set`` of names) or the
size of the polars thread pool (how polars splits a column into chunks).
Neither can be varied inside a running test, so the snippet is run once
per environment in a new interpreter and the printed results are compared.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def run_fresh(
    script: str,
    environments: Sequence[Mapping[str, str]],
    *,
    extra_path: Sequence[Path] = (),
) -> list[str]:
    """The stdout of *script* under each of *environments*, in order.

    Each environment is laid over the current one.  *extra_path* is put
    ahead of the repository root on ``PYTHONPATH``.
    """
    path = os.pathsep.join([*map(str, extra_path), str(_REPO_ROOT)])
    procs = []
    for overrides in environments:
        env = {**os.environ, "PYTHONPATH": path, **overrides}
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", script],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    outputs: list[str] = []
    for overrides, proc in zip(environments, procs, strict=True):
        out, err = proc.communicate(timeout=300)
        assert proc.returncode == 0, f"{dict(overrides)}: {err}"
        outputs.append(out)
    return outputs


def hash_seeds(n: int) -> list[dict[str, str]]:
    """Environments with ``PYTHONHASHSEED`` 0 to *n* - 1."""
    return [{"PYTHONHASHSEED": str(seed)} for seed in range(n)]


def thread_pools(*sizes: int) -> list[dict[str, str]]:
    """Environments with the polars thread pool at each of *sizes*."""
    return [{"POLARS_MAX_THREADS": str(size)} for size in sizes]
