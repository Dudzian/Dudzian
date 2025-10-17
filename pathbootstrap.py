"""Narzędzia do zapewnienia, że katalog repozytorium znajduje się na ``sys.path``.

Moduł można również uruchomić jako skrypt, aby wypisać wykrytą ścieżkę lub
tymczasowo dodać ją do ``sys.path``. Domyślne zachowanie można rozszerzyć poprzez
zmienne środowiskowe, m.in. :envvar:`PATHBOOTSTRAP_ADD_PATHS` pozwalającą wskazać
dodatkowe katalogi do umieszczenia na ``sys.path``.
"""
from __future__ import annotations

import argparse
import functools
import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from os import PathLike
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple
from typing import Literal

DEFAULT_SENTINELS: Tuple[str, ...] = ("pyproject.toml",)
ENV_ROOT_HINT = "PATHBOOTSTRAP_ROOT_HINT"
ENV_SENTINELS = "PATHBOOTSTRAP_SENTINELS"
ENV_ADDITIONAL_PATHS = "PATHBOOTSTRAP_ADD_PATHS"


def _discover_repo_root(start: Path, sentinels: Sequence[str]) -> Path:
    """Znajdź katalog repozytorium na podstawie listy plików-wskaźników."""

    for candidate in (start, *start.parents):
        if any((candidate / sentinel).exists() for sentinel in sentinels):
            return candidate
    sentinel_list = ", ".join(sentinels)
    raise FileNotFoundError(
        f"Nie znaleziono żadnego z plików-wskaźników ({sentinel_list}) począwszy od {start}."
    )


def _normalize_hint(root_hint: Optional[PathLike[str] | str]) -> Path:
    if root_hint is None:
        env_hint = os.environ.get(ENV_ROOT_HINT)
        base: Optional[PathLike[str] | str]
        if env_hint:
            base = env_hint
        else:
            base = Path(__file__).resolve().parent
    else:
        base = root_hint
    path = Path(base)
    try:
        return path.resolve()
    except FileNotFoundError:  # pragma: no cover - ścieżki tymczasowe mogą znikać w trakcie testów
        return path


def _resolve_sentinels(sentinels: Iterable[str]) -> Tuple[str, ...]:
    env_sentinels = os.environ.get(ENV_SENTINELS)
    if env_sentinels is not None:
        candidates = [part.strip() for part in env_sentinels.split(os.pathsep)]
        sentinel_list = tuple(filter(None, candidates))
        if not sentinel_list:
            raise ValueError(
                "Zmiennej środowiskowej PATHBOOTSTRAP_SENTINELS nie można ustawiać na pustą wartość."
            )
        return sentinel_list

    sentinel_list = tuple(sentinels)
    if not sentinel_list:
        raise ValueError("Lista sentinelów nie może być pusta.")

    return sentinel_list


@functools.lru_cache(maxsize=None)
def _discover_repo_root_cached(start: str, sentinels: Tuple[str, ...]) -> Path:
    return _discover_repo_root(Path(start), sentinels)


def get_repo_root(
    root_hint: Optional[PathLike[str] | str] = None,
    *,
    sentinels: Iterable[str] = DEFAULT_SENTINELS,
) -> Path:
    """Zwróć katalog repozytorium bez modyfikowania ``sys.path``.

    Kolejność priorytetów dla wskazówki początkowej jest następująca:

    1. argument ``root_hint`` (jeśli przekazany),
    2. zmienna środowiskowa :envvar:`PATHBOOTSTRAP_ROOT_HINT`,
    3. katalog zawierający plik :mod:`pathbootstrap`.

    Listę sentinelów można nadpisać globalnie poprzez zmienną
    :envvar:`PATHBOOTSTRAP_SENTINELS` (oddzielaną ``os.pathsep``), która ma
    pierwszeństwo przed argumentem ``sentinels``.
    """

    sentinel_list = _resolve_sentinels(sentinels)

    hint_path = _normalize_hint(root_hint)
    return _discover_repo_root_cached(str(hint_path), sentinel_list)


def _unique_entries(entries: Iterable[str]) -> Tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in entries:
        if entry not in seen:
            ordered.append(entry)
            seen.add(entry)
    return tuple(ordered)


def _get_env_additional_paths() -> Tuple[str, ...]:
    value = os.environ.get(ENV_ADDITIONAL_PATHS)
    if not value:
        return ()
    return tuple(part for part in value.split(os.pathsep) if part)


def _normalize_additional_paths(
    repo_root: Path, additional_paths: Iterable[PathLike[str] | str]
) -> Tuple[str, ...]:
    normalized: list[str] = []
    for candidate in additional_paths:
        path_obj = Path(candidate)
        if not path_obj.is_absolute():
            path_obj = repo_root / path_obj
        resolved = path_obj.resolve(strict=False)
        path_str = str(resolved)
        if path_str not in normalized:
            normalized.append(path_str)
    return tuple(normalized)


def _resolve_additional_paths(
    repo_root: Path,
    additional_paths: Iterable[PathLike[str] | str],
    *,
    include_env: bool = True,
) -> Tuple[str, ...]:
    provided = tuple(additional_paths)
    if include_env:
        env_entries = _get_env_additional_paths()
        if env_entries:
            combined: Tuple[PathLike[str] | str, ...] = (*env_entries, *provided)
        else:
            combined = provided
    else:
        combined = provided
    return _normalize_additional_paths(repo_root, combined)


def _apply_entries(
    container: list[str], entries: Iterable[str], position: Literal["prepend", "append"]
) -> list[str]:
    unique_entries = _unique_entries(entries)
    inserted: list[str] = []
    if position == "append":
        for entry in unique_entries:
            if entry not in container:
                container.append(entry)
                inserted.append(entry)
    elif position == "prepend":
        for entry in reversed(unique_entries):
            if entry not in container:
                container.insert(0, entry)
                inserted.insert(0, entry)
    else:  # pragma: no cover - zabezpieczenie przed błędami wywołań
        raise ValueError(f"Nieznana pozycja wstawiania: {position}")
    return inserted


def ensure_repo_root_on_sys_path(
    root_hint: Optional[PathLike[str] | str] = None,
    *,
    sentinels: Iterable[str] = DEFAULT_SENTINELS,
    position: Literal["prepend", "append"] = "prepend",
    additional_paths: Iterable[PathLike[str] | str] = (),
    use_env_additional_paths: bool = True,
) -> str:
    """Dodaj katalog repozytorium na ``sys.path`` i zwróć go jako string.

    Parametr ``position`` pozwala określić, czy ścieżka ma zostać wstawiona na
    początek listy ``sys.path`` (``prepend``), czy też na jej koniec
    (``append``). Gdy katalog znajduje się już na ścieżce, jego położenie nie
    jest modyfikowane. Opcjonalny parametr ``additional_paths`` pozwala dodać
    dodatkowe katalogi (względne lub bezwzględne), które zostaną wstawione na
    ``sys.path`` obok katalogu repozytorium. Dodatkowe ścieżki można również
    zdefiniować globalnie poprzez zmienną środowiskową
    :envvar:`PATHBOOTSTRAP_ADD_PATHS` (oddzielając wpisy separatorem
    :data:`os.pathsep`). Jeśli ``use_env_additional_paths`` ustawiono na
    ``False``, zmienna środowiskowa nie jest uwzględniana.
    """

    repo_root = get_repo_root(root_hint, sentinels=sentinels)
    repo_str = str(repo_root)
    additional_entries = _resolve_additional_paths(
        repo_root, additional_paths, include_env=use_env_additional_paths
    )
    _apply_entries(sys.path, (repo_str, *additional_entries), position)
    return repo_str


@contextmanager
def repo_on_sys_path(
    root_hint: Optional[PathLike[str] | str] = None,
    *,
    sentinels: Iterable[str] = DEFAULT_SENTINELS,
    position: Literal["prepend", "append"] = "prepend",
    additional_paths: Iterable[PathLike[str] | str] = (),
    use_env_additional_paths: bool = True,
) -> Iterator[Path]:
    """Kontekstowy manager dodający katalog repozytorium na ``sys.path``.

    Jeśli repozytorium było już obecne na ścieżce, pozostaje tam po wyjściu z
    kontekstu. W przeciwnym wypadku wpis jest usuwany przy wyjściu. Parametr
    ``position`` kontroluje, czy nowy wpis trafia na początek, czy na koniec
    ``sys.path``. Podobnie jak w :func:`ensure_repo_root_on_sys_path`, można
    przekazać dodatkowe katalogi przez ``additional_paths`` lub zdefiniować je
    globalnie przy użyciu zmiennej środowiskowej
    :envvar:`PATHBOOTSTRAP_ADD_PATHS`. Parametr ``use_env_additional_paths``
    pozwala pominąć wpisy zdefiniowane w zmiennej środowiskowej.
    """

    repo_root = get_repo_root(root_hint, sentinels=sentinels)
    repo_str = str(repo_root)
    additional_entries = _resolve_additional_paths(
        repo_root, additional_paths, include_env=use_env_additional_paths
    )
    inserted_entries = _apply_entries(sys.path, (repo_str, *additional_entries), position)
    try:
        yield repo_root
    finally:
        for entry in reversed(inserted_entries):
            try:
                sys.path.remove(entry)
            except ValueError:  # pragma: no cover - wpis mógł zostać usunięty niezależnie
                pass


@contextmanager
def chdir_repo_root(
    root_hint: Optional[PathLike[str] | str] = None,
    *,
    sentinels: Iterable[str] = DEFAULT_SENTINELS,
) -> Iterator[Path]:
    """Tymczasowo zmień bieżący katalog roboczy na katalog repozytorium."""

    repo_root = get_repo_root(root_hint, sentinels=sentinels)
    original_cwd = Path.cwd()
    if original_cwd == repo_root:
        yield repo_root
        return

    os.chdir(repo_root)
    try:
        yield repo_root
    finally:
        os.chdir(original_cwd)


def clear_cache() -> None:
    """Wyczyść cache lokalizowania katalogu repozytorium."""

    _discover_repo_root_cached.cache_clear()

__all__ = [
    "ensure_repo_root_on_sys_path",
    "get_repo_root",
    "clear_cache",
    "repo_on_sys_path",
    "chdir_repo_root",
    "main",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Wykryj katalog repozytorium i opcjonalnie dodaj go na sys.path. "
            "Domyślne zachowanie to wypisanie wykrytego katalogu."
        )
    )
    parser.add_argument(
        "--root-hint",
        help=(
            "Punkt startowy wyszukiwania katalogu repozytorium. Domyślnie używany "
            "jest katalog zawierający pathbootstrap.py lub wartość zmiennej "
            f"środowiskowej {ENV_ROOT_HINT}."
        ),
    )
    parser.add_argument(
        "--sentinel",
        dest="sentinels",
        action="append",
        default=None,
        help=(
            "Nazwa pliku-wskaźnika. Można podać wiele razy. Jeżeli nie zostanie "
            "określone, użyte będą wartości domyślne lub zmienna środowiskowa "
            f"{ENV_SENTINELS}."
        ),
    )
    parser.add_argument(
        "--ensure",
        action="store_true",
        help="Dodaj katalog repozytorium na sys.path przed zakończeniem programu.",
    )
    parser.add_argument(
        "--chdir",
        action="store_true",
        help="Tymczasowo przełącz bieżący katalog roboczy na katalog repozytorium.",
    )
    parser.add_argument(
        "--set-env",
        dest="set_env",
        metavar="NAZWA",
        help="Zamiast ścieżki wypisz przypisanie zmiennej środowiskowej do katalogu repozytorium.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Użyte z --set-env wypisze polecenie eksportu w składni POSIX.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Wyczyść pamięć podręczną wykrywania repozytorium przed uruchomieniem.",
    )
    parser.add_argument(
        "--position",
        choices=("prepend", "append"),
        default="prepend",
        help="Określ, gdzie dodać repozytorium na sys.path (domyślnie na początku).",
    )
    parser.add_argument(
        "--add-path",
        dest="additional_paths",
        action="append",
        default=None,
        help=(
            "Dodatkowa ścieżka do dodania na sys.path. Ścieżki względne są interpretowane "
            "względem katalogu repozytorium. Opcję można podać wielokrotnie. "
            f"Domyślne wartości można określić w zmiennej środowiskowej {ENV_ADDITIONAL_PATHS}, "
            "oddzielając ścieżki separatorem os.pathsep."
        ),
    )
    parser.add_argument(
        "--no-env-add-paths",
        action="store_true",
        help=(
            "Ignoruj ścieżki zdefiniowane w zmiennej środowiskowej "
            f"{ENV_ADDITIONAL_PATHS}."
        ),
    )
    parser.add_argument(
        "--pythonpath-var",
        default="PYTHONPATH",
        help=(
            "Nazwa zmiennej środowiskowej, która zostanie zaktualizowana przy uruchamianiu "
            "polecenia. Domyślnie używana jest PYTHONPATH."
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help=(
            "Polecenie do uruchomienia po dodaniu katalogu repozytorium na sys.path. "
            "Polecenie należy poprzedzić separatorem '--'."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Uruchom moduł w trybie skryptu."""

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.clear_cache:
        clear_cache()

    if args.export and not args.set_env:
        parser.error("opcja --export wymaga jednoczesnego użycia z --set-env")

    sentinel_arg = tuple(args.sentinels) if args.sentinels else DEFAULT_SENTINELS
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if command == [""]:
        command = []
    if args.command and not command:
        parser.error("należy podać polecenie po separatorze '--'")

    if not args.pythonpath_var:
        parser.error("opcja --pythonpath-var nie może być pusta")

    repo_root_hint = get_repo_root(args.root_hint, sentinels=sentinel_arg)
    additional_paths = tuple(args.additional_paths) if args.additional_paths else ()
    include_env_paths = not args.no_env_add_paths
    context = (
        chdir_repo_root(repo_root_hint, sentinels=sentinel_arg)
        if args.chdir
        else nullcontext(repo_root_hint)
    )

    with context as repo_root:
        normalized_additional = _resolve_additional_paths(
            repo_root, additional_paths, include_env=include_env_paths
        )
        if args.ensure or command:
            repo_str = ensure_repo_root_on_sys_path(
                repo_root,
                sentinels=sentinel_arg,
                position=args.position,
                additional_paths=additional_paths,
                use_env_additional_paths=include_env_paths,
            )
        else:
            repo_str = str(repo_root)

        if command:
            env = os.environ.copy()
            pythonpath_var = args.pythonpath_var
            existing_pythonpath = env.get(pythonpath_var, "")
            path_entries = [entry for entry in existing_pythonpath.split(os.pathsep) if entry]
            _apply_entries(path_entries, (repo_str, *normalized_additional), args.position)
            env[pythonpath_var] = os.pathsep.join(path_entries) if path_entries else repo_str

            if args.set_env:
                env[args.set_env] = repo_str

            completed = subprocess.run(command, check=False, env=env)
            return completed.returncode

        if args.set_env:
            assignment = f"{args.set_env}={repo_str}"
            if args.export:
                assignment = f"export {assignment}"
            print(assignment)
        else:
            print(repo_str)
        return 0


if __name__ == "__main__":  # pragma: no cover - testowane poprzez main()
    raise SystemExit(main())
