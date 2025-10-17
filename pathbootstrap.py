"""Narzędzia do zapewnienia, że katalog repozytorium znajduje się na ``sys.path``.

Moduł można również uruchomić jako skrypt, aby wypisać wykrytą ścieżkę lub
tymczasowo dodać ją do ``sys.path``. Domyślne zachowanie można rozszerzyć poprzez
zmienne środowiskowe, m.in. :envvar:`PATHBOOTSTRAP_ADD_PATHS` pozwalającą wskazać
dodatkowe katalogi do umieszczenia na ``sys.path``.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import shlex
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
SET_ENV_FORMATS: Tuple[str, ...] = ("plain", "posix", "powershell", "cmd")


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


def _load_sentinels_from_file(path: Path) -> Tuple[str, ...]:
    """Wczytaj listę sentinelów z pliku tekstowego."""

    content = path.read_text(encoding="utf-8")
    sentinels: list[str] = []
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        sentinels.append(stripped)

    if not sentinels:
        raise ValueError(f"Plik {path} nie zawiera żadnych nazw sentinelów.")

    return tuple(sentinels)


def _load_additional_paths_from_file(path: Path) -> Tuple[str, ...]:
    """Wczytaj dodatkowe ścieżki z pliku tekstowego."""

    content = path.read_text(encoding="utf-8")
    paths: list[str] = []
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        paths.append(stripped)

    if not paths:
        raise ValueError(f"Plik {path} nie zawiera żadnych ścieżek.")

    return tuple(paths)


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


def _format_env_assignment(variable: str, value: str, fmt: str) -> str:
    if fmt == "plain":
        return f"{variable}={value}"
    if fmt == "posix":
        return f"export {variable}={value}"
    if fmt == "powershell":
        escaped = value.replace("'", "''")
        return f"$Env:{variable} = '{escaped}'"
    if fmt == "cmd":
        return f"set {variable}={value}"
    raise ValueError(f"Nieznany format przypisania zmiennej środowiskowej: {fmt}")


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
        "--sentinel-file",
        help=(
            "Ścieżka pliku zawierającego nazwy plików-wskaźników (po jednej w linii). "
            "Linie zaczynające się od '#' lub puste są ignorowane."
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
        "--set-env-format",
        choices=SET_ENV_FORMATS,
        default="plain",
        help=(
            "Format wypisywania przypisania dla --set-env. Dostępne wartości to plain "
            "(domyślne), posix (export VAR=...), powershell ($Env:VAR = '...') oraz "
            "cmd (set VAR=...)."
        ),
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
        "--add-path-file",
        dest="additional_path_files",
        action="append",
        default=None,
        help=(
            "Plik zawierający dodatkowe ścieżki (jedna na linię, linie rozpoczynające się "
            "od '#' są ignorowane). Opcję można podać wielokrotnie."
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
        "--format",
        choices=("text", "json"),
        default="text",
        help="Format wyjścia przy wypisywaniu katalogu repozytorium (domyślnie text).",
    )
    parser.add_argument(
        "--print-pythonpath",
        action="store_true",
        help=(
            "Zamiast ścieżki repozytorium wypisz wartość PYTHONPATH zawierającą "
            "repozytorium i dodatkowe ścieżki."
        ),
    )
    parser.add_argument(
        "--print-sys-path",
        action="store_true",
        help=(
            "Zamiast ścieżki repozytorium wypisz wynikową listę sys.path po dodaniu "
            "repozytorium i dodatkowych ścieżek."
        ),
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help=(
            "Wypisz szczegółowe informacje o konfiguracji pathbootstrap, w tym sentinele "
            "i dodatkowe ścieżki."
        ),
    )
    parser.add_argument(
        "--output",
        metavar="PLIK",
        help=(
            "Ścieżka pliku, do którego należy zapisać wynik działania pathbootstrap. "
            "Jeżeli zostanie podana, wynik nie jest wypisywany na stdout."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Wypisz diagnostyczne informacje o działaniach pathbootstrap na stderr.",
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

    if args.format == "json" and args.set_env:
        parser.error("opcja --format=json nie jest dostępna razem z --set-env")

    sentinel_cli = tuple(args.sentinels) if args.sentinels else ()
    sentinel_from_file: Tuple[str, ...] = ()
    if args.sentinel_file:
        sentinel_path = Path(args.sentinel_file)
        try:
            sentinel_from_file = _load_sentinels_from_file(sentinel_path)
        except FileNotFoundError:
            parser.error(f"plik sentinel {sentinel_path} nie istnieje")
        except OSError as exc:
            parser.error(f"nie można odczytać pliku sentinel {sentinel_path}: {exc}")
        except ValueError as exc:
            parser.error(str(exc))

    if sentinel_from_file or sentinel_cli:
        sentinel_candidates: Tuple[str, ...] = (*sentinel_from_file, *sentinel_cli)
    else:
        sentinel_candidates = DEFAULT_SENTINELS
    env_sentinels_raw = os.environ.get(ENV_SENTINELS)
    if env_sentinels_raw:
        env_sentinels = tuple(
            part.strip() for part in env_sentinels_raw.split(os.pathsep) if part.strip()
        )
    else:
        env_sentinels = ()
    sentinel_arg = _resolve_sentinels(sentinel_candidates)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if command == [""]:
        command = []
    if args.command and not command:
        parser.error("należy podać polecenie po separatorze '--'")

    if not args.pythonpath_var:
        parser.error("opcja --pythonpath-var nie może być pusta")

    if args.output and command:
        parser.error("opcja --output nie może być używana jednocześnie z poleceniem")

    if args.print_pythonpath and args.set_env:
        parser.error("opcja --print-pythonpath nie może być używana razem z --set-env")

    if args.print_pythonpath and command:
        parser.error("opcja --print-pythonpath nie może być używana jednocześnie z poleceniem")

    if args.print_sys_path and args.set_env:
        parser.error("opcja --print-sys-path nie może być używana razem z --set-env")

    if args.print_sys_path and command:
        parser.error("opcja --print-sys-path nie może być używana jednocześnie z poleceniem")

    if args.print_sys_path and args.print_pythonpath:
        parser.error(
            "opcja --print-sys-path nie może być używana jednocześnie z --print-pythonpath"
        )

    if args.print_config and args.set_env:
        parser.error("opcja --print-config nie może być używana razem z --set-env")

    if args.print_config and command:
        parser.error("opcja --print-config nie może być używana jednocześnie z poleceniem")

    if args.print_config and args.print_pythonpath:
        parser.error(
            "opcja --print-config nie może być używana jednocześnie z --print-pythonpath"
        )

    if args.print_config and args.print_sys_path:
        parser.error("opcja --print-config nie może być używana jednocześnie z --print-sys-path")

    if args.set_env_format != "plain" and not args.set_env:
        parser.error("opcja --set-env-format wymaga jednoczesnego użycia z --set-env")

    set_env_format = args.set_env_format
    if args.export:
        if set_env_format not in ("plain", "posix"):
            parser.error(
                "opcja --export nie może być łączona z --set-env-format innym niż posix"
            )
        set_env_format = "posix"

    repo_root_hint = get_repo_root(args.root_hint, sentinels=sentinel_arg)
    additional_paths = tuple(args.additional_paths) if args.additional_paths else ()
    additional_paths_from_files: Tuple[str, ...] = ()
    if args.additional_path_files:
        collected: list[str] = []
        for file_entry in args.additional_path_files:
            file_path = Path(file_entry)
            try:
                collected.extend(_load_additional_paths_from_file(file_path))
            except FileNotFoundError:
                parser.error(f"plik ścieżek {file_path} nie istnieje")
            except OSError as exc:
                parser.error(f"nie można odczytać pliku ścieżek {file_path}: {exc}")
            except ValueError as exc:
                parser.error(str(exc))
        additional_paths_from_files = tuple(collected)
    include_env_paths = not args.no_env_add_paths
    env_additional: Tuple[str, ...] = _get_env_additional_paths() if include_env_paths else ()
    context = (
        chdir_repo_root(repo_root_hint, sentinels=sentinel_arg)
        if args.chdir
        else nullcontext(repo_root_hint)
    )

    prefix = "[pathbootstrap]"

    with context as repo_root:
        combined_additional = (*additional_paths_from_files, *additional_paths)
        normalized_additional = _resolve_additional_paths(
            repo_root, combined_additional, include_env=include_env_paths
        )
        repo_str = str(repo_root)
        additional_cli_display = tuple(str(Path(entry)) for entry in additional_paths)
        additional_files_display = tuple(
            str(Path(entry)) for entry in additional_paths_from_files
        )
        pythonpath_entries: Tuple[str, ...] | None = None
        pythonpath_value: Optional[str] = None
        sys_path_entries: Tuple[str, ...] | None = None
        sys_path_value: Optional[str] = None
        sys_path_snapshot = list(sys.path)

        if args.verbose:
            def emit(message: str) -> None:
                print(f"{prefix} {message}", file=sys.stderr)

            emit(f"katalog repozytorium: {repo_str}")
            emit(
                "sentinele: " + (", ".join(sentinel_arg) if sentinel_arg else "(brak)")
            )
            emit(f"pozycja na sys.path: {args.position}")
            emit(
                "PATHBOOTSTRAP_SENTINELS: "
                + (", ".join(env_sentinels) if env_sentinels else "(brak)")
            )
            if include_env_paths:
                emit(
                    "PATHBOOTSTRAP_ADD_PATHS: "
                    + (", ".join(env_additional) if env_additional else "(brak)")
                )
            else:
                emit("PATHBOOTSTRAP_ADD_PATHS: (wyłączone)")
            emit(
                "dodatkowe ścieżki CLI: "
                + (
                    ", ".join(additional_cli_display)
                    if additional_cli_display
                    else "(brak)"
                )
            )
            emit(
                "dodatkowe ścieżki z plików: "
                + (
                    ", ".join(additional_files_display)
                    if additional_files_display
                    else "(brak)"
                )
            )
            emit(
                "znormalizowane dodatkowe ścieżki: "
                + (
                    ", ".join(normalized_additional)
                    if normalized_additional
                    else "(brak)"
                )
            )

        if args.ensure or command:
            if args.verbose:
                print(
                    f"{prefix} dodawanie katalogu repozytorium do sys.path",
                    file=sys.stderr,
                )
            repo_str = ensure_repo_root_on_sys_path(
                repo_root,
                sentinels=sentinel_arg,
                position=args.position,
                additional_paths=combined_additional,
                use_env_additional_paths=include_env_paths,
            )

        if args.print_pythonpath:
            pythonpath_buffer: list[str] = []
            _apply_entries(
                pythonpath_buffer, (repo_str, *normalized_additional), args.position
            )
            pythonpath_entries = tuple(pythonpath_buffer)
            pythonpath_value = os.pathsep.join(pythonpath_buffer)
            if args.verbose:
                print(
                    f"{prefix} symulowana wartość {args.pythonpath_var}: {pythonpath_value}",
                    file=sys.stderr,
                )

        if args.print_sys_path:
            if args.ensure or command:
                base = list(sys.path)
            else:
                base = list(sys_path_snapshot)
            preview = list(base)
            _apply_entries(preview, (repo_str, *normalized_additional), args.position)
            sys_path_entries = tuple(preview)
            sys_path_value = "\n".join(preview)
            if args.verbose:
                print(
                    f"{prefix} symulowany sys.path: {sys_path_value}",
                    file=sys.stderr,
                )

        if command:
            env = os.environ.copy()
            pythonpath_var = args.pythonpath_var
            existing_pythonpath = env.get(pythonpath_var, "")
            path_entries = [entry for entry in existing_pythonpath.split(os.pathsep) if entry]
            inserted = _apply_entries(
                path_entries, (repo_str, *normalized_additional), args.position
            )
            env[pythonpath_var] = os.pathsep.join(path_entries) if path_entries else repo_str

            if args.verbose:
                print(
                    f"{prefix} aktualizacja {pythonpath_var}: "
                    + (
                        os.pathsep.join(path_entries)
                        if path_entries
                        else repo_str
                    ),
                    file=sys.stderr,
                )
                if inserted:
                    print(
                        f"{prefix} dodane wpisy do {pythonpath_var}: "
                        + ", ".join(inserted),
                        file=sys.stderr,
                    )

            if args.set_env:
                env[args.set_env] = repo_str
                if args.verbose:
                    print(
                        f"{prefix} ustawienie zmiennej {args.set_env} na {repo_str}",
                        file=sys.stderr,
                    )

            if args.verbose:
                print(
                    f"{prefix} uruchamianie polecenia: {shlex.join(command)}",
                    file=sys.stderr,
                )
            completed = subprocess.run(command, check=False, env=env)
            return completed.returncode

        output_text: Optional[str]
        if args.set_env:
            assignment = _format_env_assignment(args.set_env, repo_str, set_env_format)
            output_text = assignment
            if args.verbose:
                print(
                    f"{prefix} ustawienie zmiennej {args.set_env} na {repo_str}",
                    file=sys.stderr,
                )
        else:
            if args.print_pythonpath:
                assert pythonpath_value is not None
                assert pythonpath_entries is not None
                if args.format == "json":
                    payload = {
                        "repo_root": repo_str,
                        "additional_paths": list(normalized_additional),
                        "pythonpath": pythonpath_value,
                        "pythonpath_entries": list(pythonpath_entries),
                    }
                    output_text = json.dumps(payload)
                else:
                    output_text = pythonpath_value
            elif args.print_sys_path:
                assert sys_path_value is not None
                assert sys_path_entries is not None
                if args.format == "json":
                    payload = {
                        "repo_root": repo_str,
                        "additional_paths": list(normalized_additional),
                        "sys_path": sys_path_value,
                        "sys_path_entries": list(sys_path_entries),
                    }
                    output_text = json.dumps(payload)
                else:
                    output_text = sys_path_value
            elif args.print_config:
                if args.format == "json":
                    payload = {
                        "repo_root": repo_str,
                        "sentinels": list(sentinel_arg),
                        "sentinels_cli": list(sentinel_cli),
                        "sentinels_file": list(sentinel_from_file),
                        "sentinels_env": list(env_sentinels),
                        "position": args.position,
                        "include_env_additional_paths": include_env_paths,
                        "set_env_format": set_env_format,
                        "additional_paths": {
                            "normalized": list(normalized_additional),
                            "cli": list(additional_cli_display),
                            "files": list(additional_files_display),
                            "env": list(env_additional),
                        },
                        "pythonpath_var": args.pythonpath_var,
                        "ensure": bool(args.ensure),
                        "chdir": bool(args.chdir),
                    }
                    output_text = json.dumps(payload)
                else:
                    def format_section(name: str, entries: Iterable[str]) -> list[str]:
                        sequence = list(entries)
                        lines = [f"{name}:"]
                        if sequence:
                            lines.extend(f"  - {item}" for item in sequence)
                        else:
                            lines.append("  (brak)")
                        return lines

                    lines = [
                        f"repo_root: {repo_str}",
                        "sentinels: " + (", ".join(sentinel_arg) if sentinel_arg else "(brak)"),
                        "sentinels_cli: "
                        + (", ".join(sentinel_cli) if sentinel_cli else "(brak)"),
                        "sentinels_file: "
                        + (", ".join(sentinel_from_file) if sentinel_from_file else "(brak)"),
                        "sentinels_env: "
                        + (", ".join(env_sentinels) if env_sentinels else "(brak)"),
                        f"position: {args.position}",
                        f"include_env_additional_paths: {include_env_paths}",
                        f"set_env_format: {set_env_format}",
                    ]
                    lines.extend(
                        format_section(
                            "additional_paths_normalized", normalized_additional
                        )
                    )
                    lines.extend(
                        format_section("additional_paths_cli", additional_cli_display)
                    )
                    lines.extend(
                        format_section("additional_paths_files", additional_files_display)
                    )
                    lines.extend(
                        format_section("additional_paths_env", env_additional)
                    )
                    lines.append(f"pythonpath_var: {args.pythonpath_var}")
                    lines.append(f"ensure: {bool(args.ensure)}")
                    lines.append(f"chdir: {bool(args.chdir)}")
                    output_text = "\n".join(lines)
            else:
                if args.format == "json":
                    payload = {
                        "repo_root": repo_str,
                        "additional_paths": list(normalized_additional),
                    }
                    output_text = json.dumps(payload)
                else:
                    output_text = repo_str

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text + "\n", encoding="utf-8")
            if args.verbose:
                print(
                    f"{prefix} zapisano wynik do pliku {output_path}",
                    file=sys.stderr,
                )
        else:
            print(output_text)
        return 0


if __name__ == "__main__":  # pragma: no cover - testowane poprzez main()
    raise SystemExit(main())
