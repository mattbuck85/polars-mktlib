"""A deterministic stand-in for the row orders polars leaves unspecified.

polars does not promise the row order of

* a ``DataFrame.join`` called without ``maintain_order`` (or with
  ``maintain_order="none"``), and
* a ``group_by`` built without ``maintain_order=True``.

The order such a call returns today is an accident of the engine that ran
it, so a test that passes on it proves nothing about another engine or
another release.  :func:`perturb_unspecified_order` turns the freedom into
a perturbation that is the same on every polars version: while it is
active, each such call returns its rows **reversed**.  A call that pins its
order is returned untouched.  ``join_asof`` and sorts are not perturbed.

Code whose result is the same with and without the perturbation does not
depend on an order polars does not promise, at the calls the run reached.
The calls reached are recorded on the object the context manager yields.

:func:`perturb_tie_order` is the same idea for the one freedom a sort has:
the relative order of rows that are equal on the sort key.  While it is
active, a ``DataFrame.sort`` or an ``Expr.sort_by`` that does not pass
``maintain_order=True`` returns such rows in the **reverse** of their input
order.  Rows with distinct keys are where any sort puts them.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
from polars.dataframe.group_by import GroupBy

import mktlib

_LIBRARY_ROOT = Path(mktlib.__file__).resolve().parent

#: Every ``GroupBy`` method that returns the grouped frame.  The names a
#: polars version does not have are skipped.  Several of them are written in
#: terms of ``agg``; the depth guard below keeps such a call from being
#: reversed twice.
GROUP_BY_TERMINALS = (
    "agg",
    "all",
    "count",
    "first",
    "head",
    "last",
    "len",
    "map_groups",
    "max",
    "mean",
    "median",
    "min",
    "n_unique",
    "quantile",
    "sum",
    "tail",
)


@dataclass(frozen=True, order=True)
class CallSite:
    """One join or group_by terminal reached while the perturbation was on."""

    file: str
    line: int
    kind: str
    pinned: bool


@dataclass
class Perturbation:
    """What the perturbation saw: every call site, and how many it reversed."""

    sites: set[CallSite] = field(default_factory=set)
    reversed_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, site: CallSite) -> None:
        with self._lock:
            self.sites.add(site)
            if not site.pinned:
                self.reversed_calls += 1

    def library_sites(self, *, pinned: bool) -> list[CallSite]:
        """The recorded sites under ``mktlib/`` with the given pinning."""
        return sorted(
            s for s in self.sites if s.pinned is pinned and s.file != "<caller>"
        )


def _calling_site(kind: str, pinned: bool) -> CallSite:
    """The innermost frame under ``mktlib/``, or ``<caller>`` if none is."""
    frame = sys._getframe(2)
    while frame is not None:
        path = Path(frame.f_code.co_filename)
        if path.is_absolute() and path.is_relative_to(_LIBRARY_ROOT):
            rel = path.relative_to(_LIBRARY_ROOT.parent).as_posix()
            return CallSite(rel, frame.f_lineno, kind, pinned)
        frame = frame.f_back  # type: ignore[assignment]
    return CallSite("<caller>", 0, kind, pinned)


@contextmanager
def perturb_unspecified_order() -> Iterator[Perturbation]:
    """Reverse the rows of every join and group_by that does not pin order."""
    seen = Perturbation()
    # A terminal written in terms of another terminal (``len`` calls ``agg``)
    # is one call, reversed once.  The depth is per thread: the library runs
    # some backtests on a worker pool.
    local = threading.local()

    original_join = pl.DataFrame.join

    def join(self: pl.DataFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        out = original_join(self, *args, **kwargs)
        pinned = kwargs.get("maintain_order") not in (None, "none")
        seen.record(_calling_site("join", pinned))
        return out if pinned else out.reverse()

    def wrap(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def terminal(self: GroupBy, *args: Any, **kwargs: Any) -> Any:
            depth = getattr(local, "depth", 0)
            local.depth = depth + 1
            try:
                out = original(self, *args, **kwargs)
            finally:
                local.depth = depth
            if depth or not isinstance(out, pl.DataFrame):
                return out
            pinned = bool(self.maintain_order)
            seen.record(_calling_site(f"group_by.{name}", pinned))
            return out if pinned else out.reverse()

        terminal.__name__ = name
        return terminal

    originals = {
        name: getattr(GroupBy, name)
        for name in GROUP_BY_TERMINALS
        if hasattr(GroupBy, name)
    }
    pl.DataFrame.join = join  # type: ignore[method-assign, assignment]
    for name, original in originals.items():
        setattr(GroupBy, name, wrap(name, original))
    try:
        yield seen
    finally:
        pl.DataFrame.join = original_join  # type: ignore[method-assign]
        for name, original in originals.items():
            setattr(GroupBy, name, original)


def _reversed_keys(keys: tuple[Any, ...]) -> list[pl.Expr] | None:
    """Each sort key as an expression over the reversed frame, if it can be."""
    out: list[pl.Expr] = []
    for key in keys:
        if isinstance(key, str):
            out.append(pl.col(key).reverse())
        elif isinstance(key, pl.Expr):
            out.append(key.reverse())
        elif isinstance(key, (list, tuple)):
            inner = _reversed_keys(tuple(key))
            if inner is None:
                return None
            out.extend(inner)
        else:
            return None
    return out


@contextmanager
def perturb_tie_order() -> Iterator[Perturbation]:
    """Reverse the input order of tied rows in every sort that does not pin it."""
    seen = Perturbation()
    original_sort = pl.DataFrame.sort
    original_sort_by = pl.Expr.sort_by

    def sort(self: pl.DataFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        pinned = bool(kwargs.get("maintain_order", False))
        seen.record(_calling_site("sort", pinned))
        if pinned:
            return original_sort(self, *args, **kwargs)
        # A stable sort of the reversed frame: ties come out reversed.
        return original_sort(self.reverse(), *args, **{**kwargs, "maintain_order": True})

    def sort_by(self: pl.Expr, *keys: Any, **kwargs: Any) -> pl.Expr:
        pinned = bool(kwargs.get("maintain_order", False))
        reversed_keys = None if pinned else _reversed_keys(keys)
        seen.record(_calling_site("sort_by", pinned or reversed_keys is None))
        if reversed_keys is None:
            return original_sort_by(self, *keys, **kwargs)
        return original_sort_by(
            self.reverse(), *reversed_keys, **{**kwargs, "maintain_order": True}
        )

    pl.DataFrame.sort = sort  # type: ignore[method-assign, assignment]
    pl.Expr.sort_by = sort_by  # type: ignore[method-assign, assignment]
    try:
        yield seen
    finally:
        pl.DataFrame.sort = original_sort  # type: ignore[method-assign]
        pl.Expr.sort_by = original_sort_by  # type: ignore[method-assign]
