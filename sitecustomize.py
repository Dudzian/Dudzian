"""Umożliwia import lokalnych modułów bez instalacji pakietu."""
from __future__ import annotations

from pathlib import Path

import importlib
import sys
from contextlib import contextmanager
from typing import Iterable, Iterator

from pathbootstrap import ensure_repo_root_on_sys_path


@contextmanager
def _without_conflicting_packages() -> "Iterator[None]":
    """Tymczasowo usuń z ``sys.path`` katalogi kolidujące z `packaging`."""

    repo_root = Path(__file__).resolve().parent
    conflict_dirs = {
        repo_root / "scripts",
        repo_root / "deploy",
        repo_root / "tests",
    }
    original_path = list(sys.path)
    filtered = []
    modified = False
    for entry in original_path:
        try:
            entry_path = Path(entry).resolve()
        except Exception:
            filtered.append(entry)
            continue
        if entry_path in conflict_dirs:
            modified = True
            continue
        filtered.append(entry)
    if not modified:
        yield
        return
    sys.path = filtered
    try:
        yield
    finally:
        sys.path = original_path


def _ensure_stdlib_packaging() -> None:
    """Załaduj `packaging` zanim lokalne moduły zdążą je przesłonić."""

    if "packaging.version" in sys.modules:
        return
    with _without_conflicting_packages():
        try:
            importlib.import_module("packaging.version")
        except ModuleNotFoundError:
            # Pozwól środowisku kontynuować bez zależności – fallback
            # wykorzystujący stuby załaduje się później w konkretnych skryptach.
            return


def _stabilize_numpy_no_value() -> None:
    """Zapewnij spójność sentinela NumPy `_NoValue` po ewentualnych przeładowaniach."""

    try:
        import numpy as _np
        from numpy.core import _multiarray_umath as _umath  # type: ignore
        from numpy.core import _methods as _methods  # type: ignore
    except Exception:
        return

    sentinel = getattr(_umath, "_NoValue", None)
    if sentinel is None:
        return
    try:
        _np._NoValue = sentinel  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        if getattr(_methods, "_NoValue", None) is not sentinel:
            _methods._NoValue = sentinel  # type: ignore[attr-defined]
    except Exception:
        pass


def _find_repo_root(start: Path, sentinels: Iterable[str]) -> Path | None:
    """Znajdź katalog repozytorium zawierający jeden z plików-wskaźników.

    W środowiskach, gdzie ``sitecustomize`` jest importowane z ``site-packages``,
    nie chcemy przerywać działania interpretera, jeśli repozytorium źródłowe nie
    jest dostępne.  Funkcja zwraca pierwszy katalog nadrzędny zawierający
    sentinel lub ``None`` gdy nie uda się go odnaleźć.
    """

    current = start
    while True:
        if any((current / sentinel).exists() for sentinel in sentinels):
            return current
        if current.parent == current:
            return None
        current = current.parent


# Repozytorium musi być widoczne na ścieżce importu nawet wtedy, gdy
# skrypty są uruchamiane spoza katalogu głównego.  Jednocześnie chcemy, by
# standardowe pakiety (np. `packaging`) pozostawały nadrzędne względem
# lokalnych modułów o tych samych nazwach (`scripts/packaging`,
# `deploy/packaging`).  Dlatego katalog repozytorium dołączamy na koniec
# `sys.path`, dzięki czemu nie przesłania on bibliotek zainstalowanych w
# środowisku wykonawczym.  Jeżeli nie znajdziemy żadnego sentinela, po prostu
# pomijamy modyfikację ścieżki, co pozwala uruchomić Pythona również w
# środowiskach bez repozytorium źródłowego.
repo_root = _find_repo_root(Path(__file__).resolve().parent, sentinels=("pyproject.toml",))
if repo_root:
    ensure_repo_root_on_sys_path(repo_root, position="append")
_ensure_stdlib_packaging()
_stabilize_numpy_no_value()


_original_reload = importlib.reload


def _patched_reload(module):  # type: ignore[override]
    result = _original_reload(module)
    if getattr(module, "__name__", "") == "numpy" or getattr(module, "__name__", "").startswith("numpy."):
        _stabilize_numpy_no_value()
    return result


importlib.reload = _patched_reload
