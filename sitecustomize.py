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


def _looks_like_site_packages(path: Path) -> bool:
    """Sprawdź, czy ścieżka należy do katalogu site-packages/dist-packages."""

    lowered_parts = {part.lower() for part in path.parts}
    return "site-packages" in lowered_parts or "dist-packages" in lowered_parts


def _looks_like_numpy_source_tree(path: Path) -> bool:
    """Wykryj lokalne checkouty NumPy, które mogą zacieniać zainstalowane koło."""

    try:
        resolved = path if path.is_absolute() else path.resolve(strict=False)
    except Exception:
        return False

    if _looks_like_site_packages(resolved):
        return False

    markers = ("setup.py", "pyproject.toml", "meson.build")
    candidates: list[tuple[Path, Path]] = []

    # Standard przypadek: wpis sys.path wskazuje na katalog repozytorium,
    # który zawiera podkatalog ``numpy``.
    candidates.append((resolved, resolved / "numpy"))

    # Dodatkowy przypadek: wpis sys.path wskazuje bezpośrednio na katalog
    # pakietu ``numpy`` w checkoutcie źródeł.
    if resolved.name.lower() == "numpy":
        candidates.append((resolved.parent, resolved))

    for repo_root, numpy_dir in candidates:
        try:
            if not numpy_dir.is_dir():
                continue
            init_file = numpy_dir / "__init__.py"
            if not init_file.exists():
                continue
            if any((repo_root / marker).exists() for marker in markers):
                return True
        except Exception:
            continue

    return False


def _demote_numpy_source_shadows() -> None:
    """Przenieś lokalne checkouty NumPy na koniec sys.path, by nie zasłaniały wheel'a."""

    original = list(sys.path)
    sanitized: list[str] = []
    demoted: list[str] = []

    for entry in original:
        if entry == "":
            # Pusty wpis odpowiada bieżącemu katalogowi roboczemu.
            # Jeżeli cwd wygląda jak checkout NumPy, traktujemy go tak samo jak
            # inne ścieżki źródłowe i przenosimy na koniec sys.path, żeby nie
            # zasłaniał zainstalowanego koła w site-packages.
            try:
                cwd = Path.cwd()
            except Exception:
                sanitized.append(entry)
                continue

            try:
                if _looks_like_numpy_source_tree(cwd):
                    demoted.append(entry)
                else:
                    sanitized.append(entry)
            except Exception:
                sanitized.append(entry)
            continue

        try:
            entry_path = Path(entry)
        except Exception:
            sanitized.append(entry)
            continue

        if _looks_like_numpy_source_tree(entry_path):
            demoted.append(entry)
        else:
            sanitized.append(entry)

    if not demoted:
        return

    seen: set[str] = set()
    reordered: list[str] = []
    for entry in (*sanitized, *demoted):
        if entry in seen:
            continue
        reordered.append(entry)
        seen.add(entry)
    sys.path = reordered


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


def _iter_repo_root_candidates() -> Iterator[Path]:
    """Podaj możliwe miejsca startu do wyszukania repozytorium.

    Podczas importu z ``site-packages`` bardziej wiarygodnym wskaźnikiem
    dostępności źródeł bywa bieżący katalog roboczy, dlatego sprawdzamy zarówno
    lokalizację pliku, jak i ``Path.cwd()`` (jeśli jest dostępne).
    """

    yield Path(__file__).resolve().parent
    try:
        cwd = Path.cwd()
    except Exception:
        return

    try:
        repo_file_dir = Path(__file__).resolve().parent
        # Nie dubluj tej samej ścieżki, jeśli np. importujemy z repo.
        if cwd.resolve() == repo_file_dir:
            return
    except Exception:
        pass

    yield cwd


# Repozytorium musi być widoczne na ścieżce importu nawet wtedy, gdy
# skrypty są uruchamiane spoza katalogu głównego.  Jednocześnie chcemy, by
# standardowe pakiety (np. `packaging`) pozostawały nadrzędne względem
# lokalnych modułów o tych samych nazwach (`scripts/packaging`,
# `deploy/packaging`).  Dlatego katalog repozytorium dołączamy na koniec
# `sys.path`, dzięki czemu nie przesłania on bibliotek zainstalowanych w
# środowisku wykonawczym.  Jeżeli nie znajdziemy żadnego sentinela, po prostu
# pomijamy modyfikację ścieżki, co pozwala uruchomić Pythona również w
# środowiskach bez repozytorium źródłowego.
def _bootstrap_repo_path() -> None:
    """Spróbuj dodać repozytorium na ``sys.path`` bez blokowania startu Pythona."""

    try:
        repo_root = next(
            (
                root
                for root in (
                    _find_repo_root(candidate, sentinels=("pyproject.toml",))
                    for candidate in _iter_repo_root_candidates()
                )
                if root is not None
            ),
            None,
        )
    except FileNotFoundError:
        # W niefortunnym scenariuszu, gdy `_find_repo_root` sam podniesie wyjątek,
        # przechodzimy do trybu degradacji i nie modyfikujemy ``sys.path``.
        return

    if not repo_root:
        return

    try:
        ensure_repo_root_on_sys_path(repo_root, position="append")
    except FileNotFoundError:
        # W środowiskach, gdzie skrypt jest zainstalowany bez pełnego repozytorium
        # źródłowego, pomijamy modyfikację ścieżki, by nie blokować startu Pythona.
        return


_bootstrap_repo_path()
_demote_numpy_source_shadows()
_ensure_stdlib_packaging()
_stabilize_numpy_no_value()


_original_reload = importlib.reload


def _patched_reload(module):  # type: ignore[override]
    result = _original_reload(module)
    if getattr(module, "__name__", "") == "numpy" or getattr(module, "__name__", "").startswith(
        "numpy."
    ):
        _stabilize_numpy_no_value()
    return result


importlib.reload = _patched_reload
