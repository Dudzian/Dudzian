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
from typing import Literal, NamedTuple

# tomllib is built-in since Python 3.11; provide a safe fallback for older versions
try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_SENTINELS: Tuple[str, ...] = ("pyproject.toml",)
ENV_ROOT_HINT = "PATHBOOTSTRAP_ROOT_HINT"
ENV_SENTINELS = "PATHBOOTSTRAP_SENTINELS"
ENV_ADDITIONAL_PATHS = "PATHBOOTSTRAP_ADD_PATHS"
ENV_PROFILES = "PATHBOOTSTRAP_PROFILES"
ENV_ALLOW_GIT = "PATHBOOTSTRAP_ALLOW_GIT"
ENV_CONFIG_FILES = "PATHBOOTSTRAP_CONFIG"
ENV_CONFIG_INLINE = "PATHBOOTSTRAP_CONFIG_INLINE"
ENV_MAX_DEPTH = "PATHBOOTSTRAP_MAX_DEPTH"
_CONFIG_KEYS = {
    "root_hint",
    "sentinels",
    "additional_paths",
    "additional_path_files",
    "allow_git",
    "max_depth",
    "use_env_additional_paths",
    "position",
    "path_style",
    "pythonpath_var",
    "profiles",
    "default_profiles",
}
_PROFILE_ONLY_KEYS = {"extends"}
SET_ENV_FORMATS: Tuple[str, ...] = ("plain", "posix", "powershell", "cmd")
PATH_STYLES: Tuple[str, ...] = ("auto", "posix", "windows")
PATH_STYLE_SEPARATORS = {
    "auto": os.pathsep,
    "posix": ":",
    "windows": ";",
}
SHELL_KIND_POSIX = "posix"
SHELL_KIND_CMD = "cmd"
SHELL_KIND_POWERSHELL = "powershell"


class ProfileConfig(NamedTuple):
    extends: Tuple[str, ...]
    values: dict[str, object]


class ProfileToken(NamedTuple):
    name: str
    action: Literal["add", "remove"]


class RepoDiscovery(NamedTuple):
    """Szczegóły wykrycia katalogu repozytorium."""

    root: Path
    method: Literal["sentinel", "git"]
    sentinel: Optional[str]
    depth: Optional[int]
    start: Path


def _expand_pathlike(value: PathLike[str] | str) -> Path:
    """Znormalizuj wartość ścieżki, obsługując ``~`` oraz zmienne środowiskowe."""

    raw = os.fspath(value)
    expanded_user = os.path.expanduser(raw)
    expanded_env = os.path.expandvars(expanded_user)
    return Path(expanded_env)


def _interpret_env_flag(value: str) -> bool:
    """Zinterpretuj wartość logiczną ze zmiennej środowiskowej."""

    normalized = value.strip().lower()
    if not normalized:
        return False
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return True


def _discover_git_repo_root(start: Path) -> Optional[Path]:
    """Wykryj katalog repozytorium przy pomocy `git rev-parse`."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    output = completed.stdout.strip()
    if not output:
        return None

    path = Path(output)
    try:
        return path.resolve()
    except FileNotFoundError:  # pragma: no cover - katalog mógł zostać usunięty w trakcie testów
        return path


def _discover_repo_root(
    start: Path,
    sentinels: Sequence[str],
    *,
    allow_git: bool = False,
    max_depth: Optional[int] = None,
) -> RepoDiscovery:
    """Znajdź katalog repozytorium na podstawie listy plików-wskaźników."""

    for depth, candidate in enumerate((start, *start.parents)):
        if max_depth is not None and depth > max_depth:
            break
        for sentinel in sentinels:
            if (candidate / sentinel).exists():
                return RepoDiscovery(candidate, "sentinel", sentinel, depth, start)
    sentinel_list = ", ".join(sentinels)
    if allow_git:
        git_root = _discover_git_repo_root(start)
        if git_root is not None:
            return RepoDiscovery(git_root, "git", None, None, start)

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
    path = _expand_pathlike(base)
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


def _resolve_allow_git_flag(allow_git: Optional[bool]) -> bool:
    if allow_git is not None:
        return allow_git

    env_allow_git = os.environ.get(ENV_ALLOW_GIT)
    if env_allow_git is None:
        return False

    return _interpret_env_flag(env_allow_git)


def _coerce_str_sequence(value: object, *, key: str) -> Tuple[str, ...]:
    if isinstance(value, str):
        candidates = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        candidates = tuple(value)
    else:
        raise TypeError(f"Wartość '{key}' w konfiguracji musi być łańcuchem lub listą łańcuchów.")

    normalized: list[str] = []
    for index, item in enumerate(candidates):
        if not isinstance(item, str):
            raise TypeError(
                f"Element {index} wartości '{key}' w konfiguracji musi być łańcuchem znaków."
            )
        stripped = item.strip()
        if not stripped:
            raise ValueError(
                f"Element {index} wartości '{key}' w konfiguracji nie może być pusty."
            )
        normalized.append(stripped)

    return tuple(normalized)


def _coerce_optional_path(value: object, *, key: str, base_dir: Path) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Wartość '{key}' w konfiguracji musi być łańcuchem znaków.")

    candidate = value.strip()
    if not candidate:
        raise ValueError(f"Wartość '{key}' w konfiguracji nie może być pusta.")

    path = _expand_pathlike(candidate)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def _coerce_optional_bool(value: object, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _interpret_env_flag(value)
    raise TypeError(f"Wartość '{key}' w konfiguracji musi być typu bool lub string.")


def _coerce_optional_int(
    value: object, *, key: str, min_value: int = 0, allow_none: bool = True
) -> Optional[int]:
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"Wartość '{key}' w konfiguracji nie może być pusta.")
    candidate: int
    if isinstance(value, int):
        candidate = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"Wartość '{key}' w konfiguracji nie może być pusta.")
        try:
            candidate = int(stripped)
        except ValueError as exc:  # pragma: no cover - normalizacja komunikatu
            raise ValueError(
                f"Wartość '{key}' w konfiguracji musi być liczbą całkowitą."
            ) from exc
    else:
        raise TypeError(
            f"Wartość '{key}' w konfiguracji musi być liczbą całkowitą lub łańcuchem znaków."
        )

    if candidate < min_value:
        raise ValueError(
            f"Wartość '{key}' w konfiguracji nie może być mniejsza niż {min_value}."
        )
    return candidate


def _format_config_value(value: object) -> str:
    """Zwróć reprezentację tekstową wartości konfiguracji."""

    if isinstance(value, tuple):
        try:
            return json.dumps(list(value))
        except TypeError:
            return repr(list(value))
    try:
        return json.dumps(value)
    except TypeError:
        return repr(value)


def _load_config_file(path: Path) -> dict[str, object]:
    suffix = path.suffix.lower()
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise OSError(f"nie można odczytać pliku konfiguracji {path}: {exc}") from exc

    if suffix == ".json":
        data = json.loads(content)
    elif suffix in {".toml", ".tml"}:
        data = tomllib.loads(content)
    else:
        raise ValueError(
            f"Nieobsługiwany format pliku konfiguracji {path}. Obsługiwane rozszerzenia to .json oraz .toml."
        )

    if not isinstance(data, dict):
        raise ValueError(f"Plik konfiguracji {path} musi zawierać mapę klucz-wartość na najwyższym poziomie.")

    return data


def _normalize_config_mapping(
    mapping: dict[str, object], *, source: Path, allow_profiles: bool = True
) -> dict[str, object]:
    allowed_keys = _CONFIG_KEYS | (_PROFILE_ONLY_KEYS if not allow_profiles else set())
    unknown_keys = set(mapping) - allowed_keys
    if unknown_keys:
        unknown_list = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Plik konfiguracji {source} zawiera nieobsługiwane opcje: {unknown_list}")

    base_dir = source.parent
    normalized: dict[str, object] = {}

    if not allow_profiles:
        if "profiles" in mapping:
            raise ValueError(
                f"Profil w konfiguracji {source} nie może zawierać zagnieżdżonej sekcji 'profiles'."
            )
        if "default_profiles" in mapping:
            raise ValueError(
                f"Profil w konfiguracji {source} nie może określać 'default_profiles'."
            )
    elif "extends" in mapping:
        raise ValueError(
            f"Opcja 'extends' może być używana wyłącznie wewnątrz profili konfiguracji (plik {source})."
        )

    if "root_hint" in mapping:
        normalized["root_hint"] = _coerce_optional_path(
            mapping["root_hint"], key="root_hint", base_dir=base_dir
        )

    if "sentinels" in mapping:
        normalized["sentinels"] = _coerce_str_sequence(mapping["sentinels"], key="sentinels")

    if "additional_paths" in mapping:
        normalized["additional_paths"] = _coerce_str_sequence(
            mapping["additional_paths"], key="additional_paths"
        )

    if "additional_path_files" in mapping:
        files = _coerce_str_sequence(mapping["additional_path_files"], key="additional_path_files")
        normalized["additional_path_files"] = tuple(
            _coerce_optional_path(entry, key="additional_path_files", base_dir=base_dir)
            for entry in files
        )

    if "allow_git" in mapping:
        normalized["allow_git"] = _coerce_optional_bool(
            mapping["allow_git"], key="allow_git"
        )

    if "max_depth" in mapping:
        normalized["max_depth"] = _coerce_optional_int(
            mapping["max_depth"], key="max_depth"
        )

    if "use_env_additional_paths" in mapping:
        normalized["use_env_additional_paths"] = _coerce_optional_bool(
            mapping["use_env_additional_paths"], key="use_env_additional_paths"
        )

    if "position" in mapping:
        position = mapping["position"]
        if not isinstance(position, str):
            raise TypeError("Wartość 'position' w konfiguracji musi być łańcuchem znaków.")
        position_normalized = position.strip()
        if position_normalized not in {"prepend", "append"}:
            raise ValueError("Wartość 'position' w konfiguracji musi być 'prepend' lub 'append'.")
        normalized["position"] = position_normalized

    if "path_style" in mapping:
        path_style = mapping["path_style"]
        if not isinstance(path_style, str):
            raise TypeError("Wartość 'path_style' w konfiguracji musi być łańcuchem znaków.")
        style_normalized = path_style.strip()
        if style_normalized not in PATH_STYLES:
            allowed = ", ".join(PATH_STYLES)
            raise ValueError(
                f"Wartość 'path_style' w konfiguracji musi być jedną z: {allowed}."
            )
        normalized["path_style"] = style_normalized

    if "pythonpath_var" in mapping:
        pythonpath_var = mapping["pythonpath_var"]
        if not isinstance(pythonpath_var, str):
            raise TypeError("Wartość 'pythonpath_var' w konfiguracji musi być łańcuchem znaków.")
        var_normalized = pythonpath_var.strip()
        if not var_normalized:
            raise ValueError("Wartość 'pythonpath_var' w konfiguracji nie może być pusta.")
        normalized["pythonpath_var"] = var_normalized

    if allow_profiles and "default_profiles" in mapping:
        normalized["default_profiles"] = _coerce_str_sequence(
            mapping["default_profiles"], key="default_profiles"
        )

    if not allow_profiles and "extends" in mapping:
        normalized["extends"] = _coerce_str_sequence(mapping["extends"], key="extends")

    if allow_profiles and "profiles" in mapping:
        raw_profiles = mapping["profiles"]
        if not isinstance(raw_profiles, dict):
            raise TypeError("Sekcja 'profiles' w konfiguracji musi być mapą profili.")
        normalized_profiles: dict[str, ProfileConfig] = {}
        for raw_name, profile_mapping in raw_profiles.items():
            if not isinstance(raw_name, str):
                raise TypeError("Nazwy profili w konfiguracji muszą być łańcuchami znaków.")
            name = raw_name.strip()
            if not name:
                raise ValueError("Nazwa profilu w konfiguracji nie może być pusta.")
            if not isinstance(profile_mapping, dict):
                raise TypeError(
                    f"Profil '{raw_name}' w konfiguracji {source} musi być mapą klucz-wartość."
                )
            normalized_profile = _normalize_config_mapping(
                profile_mapping, source=source, allow_profiles=False
            )
            extends = tuple(normalized_profile.pop("extends", ()))
            normalized_profiles[name] = ProfileConfig(extends, dict(normalized_profile))
        normalized["profiles"] = normalized_profiles

    return normalized


def _merge_config_data(
    accumulated: dict[str, object], mapping: dict[str, object]
) -> None:
    """Scal przekazaną konfigurację z akumulowaną mapą."""

    payload = dict(mapping)
    if "profiles" in payload:
        profiles_mapping = dict(payload.pop("profiles"))
        existing_profiles = accumulated.get("profiles")
        if existing_profiles is not None:
            merged_profiles = dict(existing_profiles)
            merged_profiles.update(profiles_mapping)
        else:
            merged_profiles = profiles_mapping
        accumulated["profiles"] = merged_profiles
    accumulated.update(payload)


def _serialize_profile_values(values: dict[str, object]) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in values.items():
        if isinstance(value, tuple):
            serialized[key] = list(value)
        else:
            serialized[key] = value
    return serialized


def _serialize_config_definition(mapping: dict[str, object]) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in mapping.items():
        if key == "profiles" and isinstance(value, dict):
            serialized[key] = {
                name: {
                    "extends": list(profile.extends),
                    "values": _serialize_profile_values(profile.values),
                }
                for name, profile in value.items()
            }
            continue
        if isinstance(value, tuple):
            serialized[key] = list(value)
        else:
            serialized[key] = value
    return serialized


def _resolve_include_paths(base: Path, includes: Tuple[str, ...]) -> Tuple[Path, ...]:
    resolved: list[Path] = []
    for entry in includes:
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = (base / candidate).resolve()
        else:
            candidate = candidate.resolve()
        resolved.append(candidate)
    return tuple(resolved)


def _load_configurations(
    paths: Sequence[Path],
) -> Tuple[dict[str, object], Tuple[str, ...], Tuple[Tuple[str, str], ...]]:
    accumulated: dict[str, object] = {}
    used: list[str] = []
    include_edges: list[Tuple[str, str]] = []

    def process(path: Path, stack: Tuple[Path, ...]) -> None:
        resolved = path if path.is_absolute() else path.resolve()
        data = _load_config_file(resolved)
        raw_includes = data.pop("includes", None)
        include_paths: Tuple[Path, ...] = ()
        if raw_includes is not None:
            include_entries = _coerce_str_sequence(raw_includes, key="includes")
            include_paths = _resolve_include_paths(resolved.parent, include_entries)
        for include_path in include_paths:
            if include_path in stack or include_path == resolved:
                cycle_chain = [*stack, resolved, include_path]
                raise ValueError(
                    "wykryto cykliczne dołączanie konfiguracji: "
                    + " -> ".join(str(item) for item in cycle_chain)
                )
            include_edges.append((str(resolved), str(include_path)))
            process(include_path, (*stack, resolved))
        normalized = _normalize_config_mapping(data, source=resolved)
        _merge_config_data(accumulated, normalized)
        used.append(str(resolved))

    for raw_path in paths:
        path = raw_path if isinstance(raw_path, Path) else Path(raw_path)
        process(path, tuple())

    return accumulated, tuple(used), tuple(include_edges)


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
def _discover_repo_root_cached(
    start: str, sentinels: Tuple[str, ...], allow_git: bool, max_depth: Optional[int]
) -> RepoDiscovery:
    return _discover_repo_root(
        Path(start), sentinels, allow_git=allow_git, max_depth=max_depth
    )


def _parse_inline_config(value: str, *, source: str) -> dict[str, object]:
    """Przetwórz konfigurację przekazaną wprost jako łańcuch."""

    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{source} nie może być pusty.")

    errors: list[str] = []
    for label, loader in (("JSON", json.loads), ("TOML", tomllib.loads)):
        try:
            data = loader(stripped)
        except Exception as exc:  # noqa: BLE001 - normalizujemy komunikat
            errors.append(f"{label}: {exc}")
            continue
        if not isinstance(data, dict):
            raise TypeError(
                f"{source} musi reprezentować mapę klucz-wartość (otrzymano {type(data).__name__})."
            )
        source_path = Path(f"<inline:{source}>")
        return _normalize_config_mapping(data, source=source_path)

    joined = ", ".join(errors) if errors else "brak szczegółów"
    raise ValueError(f"Nie udało się sparsować konfiguracji inline z {source}: {joined}")


def get_repo_info(
    root_hint: Optional[PathLike[str] | str] = None,
    *,
    sentinels: Iterable[str] = DEFAULT_SENTINELS,
    allow_git: Optional[bool] = None,
    max_depth: Optional[int] = None,
) -> RepoDiscovery:
    """Zwróć szczegóły wykrycia katalogu repozytorium."""

    sentinel_list = _resolve_sentinels(sentinels)

    hint_path = _normalize_hint(root_hint)
    effective_allow_git = _resolve_allow_git_flag(allow_git)

    if max_depth is not None and max_depth < 0:
        raise ValueError("Parametr max_depth nie może być ujemny.")

    cached = _discover_repo_root_cached(
        str(hint_path), sentinel_list, effective_allow_git, max_depth
    )
    return RepoDiscovery(cached.root, cached.method, cached.sentinel, cached.depth, hint_path)


def get_repo_root(
    root_hint: Optional[PathLike[str] | str] = None,
    *,
    sentinels: Iterable[str] = DEFAULT_SENTINELS,
    allow_git: Optional[bool] = None,
    max_depth: Optional[int] = None,
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

    return get_repo_info(
        root_hint, sentinels=sentinels, allow_git=allow_git, max_depth=max_depth
    ).root


def _unique_entries(entries: Iterable[str]) -> Tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in entries:
        if entry not in seen:
            ordered.append(entry)
            seen.add(entry)
    return tuple(ordered)


def _parse_profile_tokens(
    entries: Sequence[str], *, allow_remove: bool, source: str
) -> Tuple[ProfileToken, ...]:
    tokens: list[ProfileToken] = []
    for raw in entries:
        stripped = raw.strip()
        if not stripped:
            continue
        action: Literal["add", "remove"] = "add"
        name = stripped
        if stripped[0] in {"-", "!"}:
            if not allow_remove:
                raise ValueError(
                    f"{source} nie może usuwać profili ({stripped!r})."
                )
            action = "remove"
            name = stripped[1:].strip()
        if not name:
            raise ValueError(f"Pusta nazwa profilu w {source}.")
        tokens.append(ProfileToken(name, action))
    return tuple(tokens)


def _resolve_profile_values(
    name: str,
    profiles: dict[str, ProfileConfig],
    cache: dict[str, dict[str, object]],
    stack: Tuple[str, ...],
) -> dict[str, object]:
    if name in cache:
        return cache[name]

    if name in stack:
        cycle = " -> ".join((*stack, name))
        raise ValueError(f"wykryto cykliczne dziedziczenie profili: {cycle}")

    profile = profiles[name]
    merged: dict[str, object] = {}
    for parent in profile.extends:
        parent_values = _resolve_profile_values(parent, profiles, cache, (*stack, name))
        merged.update(parent_values)
    merged.update(profile.values)
    cache[name] = merged
    return merged


def _get_env_additional_paths() -> Tuple[str, ...]:
    value = os.environ.get(ENV_ADDITIONAL_PATHS)
    if not value:
        return ()
    entries: list[str] = []
    for raw_part in value.split(os.pathsep):
        stripped = raw_part.strip()
        if not stripped:
            continue
        expanded = _expand_pathlike(stripped)
        entries.append(str(expanded))
    return tuple(entries)


def _normalize_additional_paths(
    repo_root: Path, additional_paths: Iterable[PathLike[str] | str]
) -> Tuple[str, ...]:
    normalized: list[str] = []
    for candidate in additional_paths:
        path_obj = _expand_pathlike(candidate)
        if not path_obj.is_absolute():
            resolved = (repo_root / path_obj).resolve(strict=False)
        else:
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


def _detect_shell_kind(shell_path: str) -> str:
    """Rozpoznaj typ powłoki, aby dobrać sposób uruchomienia poleceń."""

    lowered = shell_path.lower()
    if "powershell" in lowered or lowered.endswith("pwsh"):
        return SHELL_KIND_POWERSHELL
    if os.name == "nt":
        return SHELL_KIND_CMD
    return SHELL_KIND_POSIX


def _get_default_shell() -> str:
    """Zwróć domyślną powłokę dla bieżącego systemu."""

    if os.name == "nt":
        return os.environ.get("COMSPEC", "cmd.exe")
    return os.environ.get("SHELL", "/bin/sh")


def _build_shell_command(shell_path: str, command: Sequence[str]) -> Tuple[str, ...]:
    """Zbuduj polecenie wywołujące powłokę z opcjonalnym poleceniem."""

    kind = _detect_shell_kind(shell_path)
    if not command:
        return (shell_path,)

    if kind == SHELL_KIND_POWERSHELL:
        joined = " ".join(shlex.quote(part) for part in command)
        return (shell_path, "-Command", joined)

    if kind == SHELL_KIND_CMD:
        joined = subprocess.list2cmdline(list(command))
        return (shell_path, "/C", joined)

    # Domyślnie zakładamy semantykę POSIX.
    joined = shlex.join(command)
    return (shell_path, "-c", joined)


def _format_path_value(path: str, style: str) -> str:
    """Sformatuj ścieżkę według zadanego stylu."""

    if style == "auto":
        return path
    if style == "posix":
        return path.replace("\\", "/")
    if style == "windows":
        return path.replace("/", "\\")
    raise ValueError(f"Nieznany styl ścieżki: {style}")


def _format_path_sequence(paths: Iterable[str], style: str) -> Tuple[str, ...]:
    return tuple(_format_path_value(path, style) for path in paths)


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
    allow_git: Optional[bool] = None,
    max_depth: Optional[int] = None,
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
    ``False``, zmienna środowiskowa nie jest uwzględniana. Parametr
    ``allow_git`` umożliwia korzystanie z zapasowego wykrywania katalogu
    repozytorium opartego na poleceniu ``git rev-parse`` – domyślnie jest ono
    wyłączone, chyba że ustawiono :envvar:`PATHBOOTSTRAP_ALLOW_GIT`. Opcjonalny
    parametr ``max_depth`` pozwala ograniczyć liczbę katalogów nadrzędnych,
    które zostaną sprawdzone w poszukiwaniu sentinelów (licząc od katalogu
    startowego).
    """

    repo_root = get_repo_info(
        root_hint, sentinels=sentinels, allow_git=allow_git, max_depth=max_depth
    ).root
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
    allow_git: Optional[bool] = None,
    max_depth: Optional[int] = None,
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
    pozwala pominąć wpisy zdefiniowane w zmiennej środowiskowej. Parametr
    ``allow_git`` kontroluje użycie zapasowego wykrywania katalogu repozytorium
    przy pomocy ``git rev-parse`` (można go także włączyć przez
    :envvar:`PATHBOOTSTRAP_ALLOW_GIT`). Parametr ``max_depth`` ogranicza liczbę
    poziomów katalogów nadrzędnych, które zostaną przejrzane podczas
    wyszukiwania sentinelów.
    """

    repo_root = get_repo_info(
        root_hint, sentinels=sentinels, allow_git=allow_git, max_depth=max_depth
    ).root
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
    allow_git: Optional[bool] = None,
    max_depth: Optional[int] = None,
) -> Iterator[Path]:
    """Tymczasowo zmień bieżący katalog roboczy na katalog repozytorium.

    Parametr ``allow_git`` pozwala korzystać z zapasowego wykrywania katalogu
    repozytorium przy pomocy polecenia ``git rev-parse``; funkcję można także
    aktywować globalnie zmienną :envvar:`PATHBOOTSTRAP_ALLOW_GIT`. Parametr
    ``max_depth`` ogranicza liczbę poziomów katalogów nadrzędnych sprawdzanych
    przy wyszukiwaniu sentinelów.
    """

    repo_root = get_repo_info(
        root_hint, sentinels=sentinels, allow_git=allow_git, max_depth=max_depth
    ).root
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
    "get_repo_info",
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
        "--config-file",
        dest="config_files",
        action="append",
        default=None,
        help=(
            "Plik konfiguracji pathbootstrap w formacie JSON lub TOML. Opcję można podać "
            "wielokrotnie; wartości z późniejszych plików nadpisują wcześniejsze. "
            f"Ścieżki można również określić w zmiennej środowiskowej {ENV_CONFIG_FILES}, "
            "oddzielając je separatorem os.pathsep."
        ),
    )
    parser.add_argument(
        "--config-inline",
        dest="config_inline",
        action="append",
        default=None,
        help=(
            "Fragment konfiguracji w formacie JSON lub TOML przekazany bezpośrednio w "
            "linii poleceń. Opcję można wskazać wielokrotnie; wartości z późniejszych wpisów "
            "nadpisują wcześniejsze. Konfigurację inline można również dostarczyć poprzez "
            f"zmienną środowiskową {ENV_CONFIG_INLINE}."
        ),
    )
    parser.add_argument(
        "--profile",
        dest="profiles",
        action="append",
        default=None,
        help=(
            "Nazwa profilu konfiguracji do zastosowania. Można podać wielokrotnie; kolejność "
            "ma znaczenie. Poprzedź nazwę znakiem '!' lub '-' aby usunąć profil. Profile "
            "mogą być również określone przez zmienną środowiskową "
            f"{ENV_PROFILES} (oddzielone separatorem os.pathsep) lub wpis 'default_profiles' w pliku."
        ),
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help=(
            "Wypisz dostępne profile konfiguracji wraz z kluczami oraz wskazaniem profili "
            "domyślnych i aktualnie wybranych."
        ),
    )
    parser.add_argument(
        "--list-sentinels",
        action="store_true",
        help=(
            "Wypisz listę plików-wskaźników (sentineli) pochodzących z konfiguracji, "
            "zmiennych środowiskowych oraz argumentów CLI."
        ),
    )
    parser.add_argument(
        "--list-add-paths",
        action="store_true",
        help=(
            "Wypisz dodatkowe ścieżki wykorzystywane przy bootstrapie sys.path, "
            "wraz ze źródłami w konfiguracji, środowisku i argumentach CLI."
        ),
    )
    parser.add_argument(
        "--list-config-files",
        action="store_true",
        help=(
            "Wypisz pliki konfiguracyjne pathbootstrap oraz źródła, z których pochodzą, "
            "uwzględniając konfigurację inline i relacje include."
        ),
    )
    parser.add_argument(
        "--list-root-hints",
        action="store_true",
        help=(
            "Wypisz źródła wartości root_hint wykorzystywanych do wyszukiwania "
            "repozytorium, w tym wartości z konfiguracji, zmiennych środowiskowych, "
            "argumentów CLI oraz wartość domyślną."
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
        "--max-depth",
        type=int,
        default=None,
        help=(
            "Maksymalna liczba poziomów katalogów nadrzędnych sprawdzanych przy szukaniu "
            "sentinelów. Domyślnie brak limitu. Można również ustawić zmienną środowiskową "
            f"{ENV_MAX_DEPTH}."
        ),
    )
    allow_git_group = parser.add_mutually_exclusive_group()
    allow_git_group.add_argument(
        "--allow-git",
        dest="allow_git",
        action="store_true",
        help=(
            "Użyj 'git rev-parse --show-toplevel' jako zapasowego sposobu wykrywania "
            "repozytorium, jeżeli sentinele nie zostaną znalezione. Opcję można także "
            f"włączyć przez ustawienie zmiennej {ENV_ALLOW_GIT}."
        ),
    )
    allow_git_group.add_argument(
        "--no-allow-git",
        dest="allow_git",
        action="store_false",
        help="Wyłącz użycie zapasowego wykrywania repozytorium przy pomocy git.",
    )
    parser.set_defaults(allow_git=None)
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
        default=None,
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
        default=None,
        help=(
            "Nazwa zmiennej środowiskowej, która zostanie zaktualizowana przy uruchamianiu "
            "polecenia. Domyślnie używana jest PYTHONPATH."
        ),
    )
    parser.add_argument(
        "--module",
        help=(
            "Uruchom moduł Pythona przy pomocy 'python -m <moduł>' po dodaniu repozytorium "
            "na sys.path. Argumenty modułu można przekazać po separatorze '--'."
        ),
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help=(
            "Ścieżka do interpretera Pythona używanego przy uruchamianiu modułu (--module). "
            "Domyślnie używany jest bieżący interpreter."
        ),
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help=(
            "Uruchom powłokę systemową po dodaniu repozytorium na sys.path. Argumenty "
            "powłoki można przekazać po separatorze '--'."
        ),
    )
    parser.add_argument(
        "--shell-path",
        help=(
            "Ścieżka do programu powłoki używanego z --shell. Domyślnie używana jest wartość "
            "$SHELL (na systemach POSIX) lub $COMSPEC (na Windows)."
        ),
    )
    parser.add_argument(
        "--path-style",
        choices=PATH_STYLES,
        default=None,
        help=(
            "Styl formatowania ścieżek w wyjściu. 'auto' pozostawia oryginalne separatory, "
            "'posix' wymusza ukośniki, a 'windows' używa odwrotnych ukośników oraz ';' jako "
            "separatora PYTHONPATH."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Format wyjścia przy wypisywaniu katalogu repozytorium (domyślnie text).",
    )
    parser.add_argument(
        "--json-indent",
        type=int,
        default=None,
        help=(
            "Wielkość wcięcia używana przy formacie JSON. Wymaga jednoczesnego użycia z "
            "--format=json."
        ),
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
        "--print-discovery",
        action="store_true",
        help=(
            "Wypisz szczegóły procesu wykrywania katalogu repozytorium, w tym metodę, "
            "zastosowany sentinel oraz głębokość wyszukiwania."
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

    if args.json_indent is not None and args.format != "json":
        parser.error("opcja --json-indent wymaga jednoczesnego użycia z --format=json")
    if args.json_indent is not None and args.json_indent < 0:
        parser.error("wartość --json-indent musi być nieujemna")
    if args.max_depth is not None and args.max_depth < 0:
        parser.error("wartość --max-depth musi być nieujemna")

    if args.json_indent is not None:
        json_kwargs: dict[str, object] = {"indent": args.json_indent}
    else:
        json_kwargs = {}

    config_cli = tuple(args.config_files) if args.config_files else ()
    env_config_raw = os.environ.get(ENV_CONFIG_FILES)
    env_config_files: Tuple[str, ...]
    if env_config_raw:
        env_config_files = tuple(
            part.strip() for part in env_config_raw.split(os.pathsep) if part.strip()
        )
    else:
        env_config_files = ()

    env_config_entries = tuple(_expand_pathlike(entry) for entry in env_config_files)
    cli_config_entries = tuple(_expand_pathlike(entry) for entry in config_cli)
    config_entries = [*env_config_entries, *cli_config_entries]
    env_config_entries_display_raw = tuple(str(path) for path in env_config_entries)
    cli_config_entries_display_raw = tuple(str(path) for path in cli_config_entries)
    config_root_entries_display_raw = tuple(str(path) for path in config_entries)
    config_data: dict[str, object] = {}
    config_files_used: Tuple[str, ...] = ()
    config_include_edges: Tuple[Tuple[str, str], ...] = ()

    if config_entries:
        try:
            config_data, config_files_used, config_include_edges = _load_configurations(
                config_entries
            )
        except FileNotFoundError as exc:
            missing = exc.filename if exc.filename else str(config_entries[0])
            parser.error(f"plik konfiguracji {missing} nie istnieje")
        except ValueError as exc:
            parser.error(str(exc))
        except OSError as exc:
            parser.error(str(exc))

    config_inline_cli = tuple(args.config_inline) if args.config_inline else ()
    env_config_inline_raw = os.environ.get(ENV_CONFIG_INLINE)
    inline_inputs: list[Tuple[str, str]] = []
    if env_config_inline_raw is not None:
        inline_inputs.append((f"ENV:{ENV_CONFIG_INLINE}", env_config_inline_raw))
    for index, snippet in enumerate(config_inline_cli):
        inline_inputs.append((f"CLI[{index}]", snippet))

    config_inline_definitions: list[Tuple[str, dict[str, object]]] = []
    for source_label, raw_value in inline_inputs:
        try:
            parsed_inline = _parse_inline_config(raw_value, source=source_label)
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        config_inline_definitions.append((source_label, parsed_inline))
        _merge_config_data(config_data, parsed_inline)

    config_inline_sources = tuple(source for source, _ in config_inline_definitions)
    config_inline_serialized = tuple(
        (source, _serialize_config_definition(definition))
        for source, definition in config_inline_definitions
    )

    raw_profiles = config_data.get("profiles", {})
    config_profiles: dict[str, ProfileConfig] = {}
    if isinstance(raw_profiles, dict):
        for name, profile in raw_profiles.items():
            if isinstance(profile, ProfileConfig):
                config_profiles[name] = ProfileConfig(
                    tuple(profile.extends), dict(profile.values)
                )
            elif isinstance(profile, dict):
                config_profiles[name] = ProfileConfig(tuple(), dict(profile))
            else:
                parser.error(
                    "nieprawidłowa definicja profilu w konfiguracji: oczekiwano mapy wartości"
                )
    else:
        parser.error("sekcja profiles w konfiguracji musi być mapą profili")

    config_profiles_defined = tuple(config_profiles.keys())
    config_default_raw = tuple(config_data.get("default_profiles", ()))
    try:
        config_default_tokens = _parse_profile_tokens(
            config_default_raw,
            allow_remove=False,
            source="sekcja default_profiles konfiguracji",
        )
    except ValueError as exc:
        parser.error(str(exc))
    config_default_profiles = _unique_entries(token.name for token in config_default_tokens)

    missing_profile_bases: dict[str, Tuple[str, ...]] = {}
    for profile_name, profile in config_profiles.items():
        missing = tuple(parent for parent in profile.extends if parent not in config_profiles)
        if missing:
            missing_profile_bases[profile_name] = missing

    if missing_profile_bases:
        details = "; ".join(
            f"{name}: {', '.join(missing)}" for name, missing in missing_profile_bases.items()
        )
        parser.error(
            "profile konfiguracji odwołują się do nieistniejących baz: "
            f"{details}"
        )

    if config_default_profiles:
        missing_defaults = [
            name for name in config_default_profiles if name not in config_profiles
        ]
        if missing_defaults:
            available = (
                ", ".join(config_profiles_defined) if config_profiles_defined else "(brak)"
            )
            missing_display = ", ".join(missing_defaults)
            parser.error(
                "konfiguracja odwołuje się do nieistniejących profili: "
                f"{missing_display} (dostępne: {available})"
            )

    env_profiles_raw = os.environ.get(ENV_PROFILES)
    if env_profiles_raw:
        env_entries = tuple(
            part.strip() for part in env_profiles_raw.split(os.pathsep) if part.strip()
        )
    else:
        env_entries = ()

    try:
        env_profile_tokens = _parse_profile_tokens(
            env_entries,
            allow_remove=True,
            source=f"zmienna środowiskowa {ENV_PROFILES}",
        )
    except ValueError as exc:
        parser.error(str(exc))
    env_profiles_added = tuple(
        token.name for token in env_profile_tokens if token.action == "add"
    )
    env_profiles_removed = tuple(
        token.name for token in env_profile_tokens if token.action == "remove"
    )

    cli_profiles_input = tuple(args.profiles) if args.profiles else ()
    for profile_name in cli_profiles_input:
        if not profile_name or not profile_name.strip():
            parser.error("opcja --profile nie może być pusta")
    try:
        cli_profile_tokens = _parse_profile_tokens(
            cli_profiles_input,
            allow_remove=True,
            source="opcji --profile",
        )
    except ValueError as exc:
        parser.error(str(exc))
    cli_profiles_added = tuple(
        token.name for token in cli_profile_tokens if token.action == "add"
    )
    cli_profiles_removed = tuple(
        token.name for token in cli_profile_tokens if token.action == "remove"
    )

    referenced_env = {token.name for token in env_profile_tokens}
    missing_env = [name for name in referenced_env if name not in config_profiles]
    if missing_env:
        available = (
            ", ".join(config_profiles_defined) if config_profiles_defined else "(brak)"
        )
        missing_display = ", ".join(sorted(missing_env))
        parser.error(
            f"zmienna środowiskowa {ENV_PROFILES} odwołuje się do nieistniejących profili: "
            f"{missing_display} (dostępne: {available})"
        )

    referenced_cli = {token.name for token in cli_profile_tokens}
    missing_cli = [name for name in referenced_cli if name not in config_profiles]
    if missing_cli:
        available = (
            ", ".join(config_profiles_defined) if config_profiles_defined else "(brak)"
        )
        missing_display = ", ".join(sorted(missing_cli))
        parser.error(
            "opcja --profile odwołuje się do nieistniejących profili: "
            f"{missing_display} (dostępne: {available})"
        )

    selected_profiles_list = list(config_default_profiles)
    for token in env_profile_tokens:
        if token.action == "add":
            if token.name not in selected_profiles_list:
                selected_profiles_list.append(token.name)
        else:
            selected_profiles_list = [
                name for name in selected_profiles_list if name != token.name
            ]
    for token in cli_profile_tokens:
        if token.action == "add":
            if token.name not in selected_profiles_list:
                selected_profiles_list.append(token.name)
        else:
            selected_profiles_list = [
                name for name in selected_profiles_list if name != token.name
            ]

    selected_profiles = tuple(selected_profiles_list)

    profile_resolution_cache: dict[str, dict[str, object]] = {}
    try:
        for profile_name in config_profiles:
            _resolve_profile_values(profile_name, config_profiles, profile_resolution_cache, tuple())
    except ValueError as exc:
        parser.error(str(exc))

    base_config_values = {
        key: value
        for key, value in config_data.items()
        if key not in {"profiles", "default_profiles"}
    }
    effective_config = dict(base_config_values)
    for profile_name in selected_profiles:
        effective_config.update(profile_resolution_cache[profile_name])

    if args.export and not args.set_env:
        parser.error("opcja --export wymaga jednoczesnego użycia z --set-env")

    sentinel_cli = tuple(args.sentinels) if args.sentinels else ()
    sentinel_from_file: Tuple[str, ...] = ()
    sentinel_file_path: Optional[Path] = None
    if args.sentinel_file:
        sentinel_path = _expand_pathlike(args.sentinel_file)
        sentinel_file_path = sentinel_path
        try:
            sentinel_from_file = _load_sentinels_from_file(sentinel_path)
        except FileNotFoundError:
            parser.error(f"plik sentinel {sentinel_path} nie istnieje")
        except OSError as exc:
            parser.error(f"nie można odczytać pliku sentinel {sentinel_path}: {exc}")
        except ValueError as exc:
            parser.error(str(exc))

    if args.export and not args.set_env:
        parser.error("opcja --export wymaga jednoczesnego użycia z --set-env")

    if args.format == "json" and args.set_env:
        parser.error("opcja --format=json nie jest dostępna razem z --set-env")

    if sentinel_from_file or sentinel_cli or effective_config.get("sentinels"):
        sentinel_candidates: Tuple[str, ...] = (
            *tuple(effective_config.get("sentinels", ())),
            *sentinel_from_file,
            *sentinel_cli,
        )
    else:
        sentinel_candidates = DEFAULT_SENTINELS
    env_sentinels_raw = os.environ.get(ENV_SENTINELS)
    if env_sentinels_raw:
        env_sentinels = tuple(
            part.strip() for part in env_sentinels_raw.split(os.pathsep) if part.strip()
        )
    else:
        env_sentinels = ()

    env_allow_git_raw = os.environ.get(ENV_ALLOW_GIT)
    env_allow_git = (
        _interpret_env_flag(env_allow_git_raw) if env_allow_git_raw is not None else None
    )
    config_allow_git_value = effective_config.get("allow_git")
    allow_git_param: Optional[bool]
    if args.allow_git is not None:
        allow_git_param = args.allow_git
    elif config_allow_git_value is not None:
        allow_git_param = bool(config_allow_git_value)
    else:
        allow_git_param = env_allow_git
    effective_allow_git = allow_git_param if allow_git_param is not None else False

    env_max_depth_raw = os.environ.get(ENV_MAX_DEPTH)
    env_max_depth: Optional[int] = None
    if env_max_depth_raw is not None:
        stripped = env_max_depth_raw.strip()
        if not stripped:
            parser.error(
                f"zmienna środowiskowa {ENV_MAX_DEPTH} nie może być pusta"
            )
        try:
            env_max_depth_candidate = int(stripped)
        except ValueError:
            parser.error(
                f"zmienna środowiskowa {ENV_MAX_DEPTH} musi być liczbą całkowitą"
            )
        if env_max_depth_candidate < 0:
            parser.error(
                f"zmienna środowiskowa {ENV_MAX_DEPTH} nie może być ujemna"
            )
        env_max_depth = env_max_depth_candidate

    config_max_depth_value = effective_config.get("max_depth")
    if config_max_depth_value is not None and not isinstance(config_max_depth_value, int):
        parser.error("wartość 'max_depth' w konfiguracji musi być liczbą całkowitą nieujemną")

    position = (
        args.position
        if args.position is not None
        else effective_config.get("position") if effective_config.get("position") is not None else "prepend"
    )
    path_style = (
        args.path_style
        if args.path_style is not None
        else effective_config.get("path_style") if effective_config.get("path_style") is not None else "auto"
    )
    pythonpath_var = (
        args.pythonpath_var
        if args.pythonpath_var is not None
        else effective_config.get("pythonpath_var") if effective_config.get("pythonpath_var") is not None else "PYTHONPATH"
    )

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if command == [""]:
        command = []
    module_name = args.module
    shell_requested = bool(args.shell)
    if args.command and not command and not module_name and not shell_requested:
        parser.error("należy podać polecenie po separatorze '--'")

    should_run_command = bool(command or module_name or shell_requested)

    # Early "list" style commands
    sentinel_arg = _resolve_sentinels(sentinel_candidates)
    prefix = "[pathbootstrap]"

    # Helpers for output path formatting
    def _format_path(p: str) -> str:
        return _format_path_value(p, path_style)

    def _fmt_seq(seq: Sequence[str]) -> Tuple[str, ...]:
        return _format_path_sequence(seq, path_style)

    # Root-hint listing
    if args.list_root_hints:
        if any([args.set_env, args.export, args.ensure, should_run_command,
                args.print_pythonpath, args.print_sys_path, args.print_config,
                args.print_discovery, args.list_profiles, args.list_sentinels,
                args.list_add_paths, args.list_config_files]):
            parser.error("--list-root-hints nie może być łączone z innymi trybami wypisywania/uruchamiania")

        default_root_hint_path = Path(__file__).resolve().parent
        default_root_hint_raw = str(default_root_hint_path)

        def format_optional_hint(value: Optional[PathLike[str] | str]) -> Optional[str]:
            if value is None:
                return None
            path_obj = _expand_pathlike(value)
            try:
                resolved = path_obj.resolve()
            except FileNotFoundError:
                resolved = path_obj
            return _format_path(str(resolved))

        cli_root_hint_raw = args.root_hint
        config_root_hint_raw = effective_config.get("root_hint")
        env_root_hint_raw = os.environ.get(ENV_ROOT_HINT)
        env_root_hint_display = format_optional_hint(env_root_hint_raw)
        cli_root_hint_display = format_optional_hint(cli_root_hint_raw)
        config_root_hint_display = format_optional_hint(config_root_hint_raw)
        default_root_hint_display = format_optional_hint(default_root_hint_raw)

        if args.root_hint is not None:
            effective_source = "cli"
            effective_raw = args.root_hint
            normalize_input: Optional[PathLike[str] | str] = args.root_hint
        elif config_root_hint_raw is not None:
            effective_source = "config"
            effective_raw = config_root_hint_raw
            normalize_input = config_root_hint_raw
        elif env_root_hint_raw is not None:
            effective_source = "env"
            effective_raw = env_root_hint_raw
            normalize_input = None
        else:
            effective_source = "default"
            effective_raw = default_root_hint_raw
            normalize_input = None

        effective_hint_path = _normalize_hint(normalize_input)
        effective_path_str = str(effective_hint_path)
        effective_display = _format_path(effective_path_str)

        if args.format == "json":
            payload = {
                "effective_source": effective_source,
                "effective_raw": effective_raw,
                "effective_path": effective_path_str,
                "effective_display": effective_display,
                "cli_raw": cli_root_hint_raw,
                "cli_display": cli_root_hint_display,
                "config_raw": config_root_hint_raw,
                "config_display": config_root_hint_display,
                "env_var": ENV_ROOT_HINT,
                "env_raw": env_root_hint_raw,
                "env_display": env_root_hint_display,
                "default_raw": default_root_hint_raw,
                "default_display": default_root_hint_display,
            }
            output_text = json.dumps(payload, **json_kwargs)
        else:
            lines: list[str] = []

            def append_line(label: str, value: Optional[str]) -> None:
                if value is None:
                    lines.append(f"{label}: (brak)")
                else:
                    lines.append(f"{label}: {value}")

            append_line("root_hint_effective_source", effective_source)
            append_line("root_hint_effective_raw", effective_raw)
            append_line("root_hint_effective_path", effective_path_str)
            append_line("root_hint_effective_display", effective_display)
            append_line("root_hint_cli", cli_root_hint_raw)
            append_line("root_hint_cli_display", cli_root_hint_display)
            append_line("root_hint_config", config_root_hint_raw)
            append_line("root_hint_config_display", config_root_hint_display)
            lines.append(f"root_hint_env_var: {ENV_ROOT_HINT}")
            append_line("root_hint_env", env_root_hint_raw)
            append_line("root_hint_env_display", env_root_hint_display)
            append_line("root_hint_default", default_root_hint_raw)
            append_line("root_hint_default_display", default_root_hint_display)

            output_text = "\n".join(lines)

        if args.output:
            output_path = _expand_pathlike(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text + "\n", encoding="utf-8")
        else:
            print(output_text)
        return 0

    # Sentinels listing
    if args.list_sentinels:
        if any([args.set_env, args.export, args.ensure, should_run_command,
                args.print_pythonpath, args.print_sys_path, args.print_config,
                args.print_discovery, args.list_profiles, args.list_config_files,
                args.list_add_paths, args.list_root_hints]):
            parser.error("--list-sentinels nie może być łączone z innymi trybami wypisywania/uruchamiania")

        sentinel_file_cli = (
            (str(_expand_pathlike(args.sentinel_file)),)
            if args.sentinel_file is not None
            else ()
        )
        if args.format == "json":
            payload = {
                "effective": list(sentinel_arg),
                "candidates": list(sentinel_candidates),
                "default": list(DEFAULT_SENTINELS),
                "config": list(tuple(effective_config.get("sentinels", ()) )),
                "env": list(env_sentinels),
                "cli": list(tuple(args.sentinels) if args.sentinels else ()),
                "file": list(tuple(_load_sentinels_from_file(_expand_pathlike(args.sentinel_file))) if args.sentinel_file else ()),
                "sentinel_file": sentinel_file_cli[0] if sentinel_file_cli else None,
            }
            output_text = json.dumps(payload, **json_kwargs)
        else:
            lines: list[str] = []

            def append_section(title: str, entries: Sequence[str]) -> None:
                if entries:
                    lines.append(f"{title}:")
                    for item in entries:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"{title}: (brak)")

            append_section("sentinels_effective", sentinel_arg)
            append_section("sentinels_candidates", sentinel_candidates)
            append_section("sentinels_default", DEFAULT_SENTINELS)
            append_section("sentinels_config", tuple(effective_config.get("sentinels", ()) ))
            append_section("sentinels_env", env_sentinels)
            append_section("sentinels_cli", tuple(args.sentinels) if args.sentinels else ())
            append_section("sentinels_file", tuple(_load_sentinels_from_file(_expand_pathlike(args.sentinel_file))) if args.sentinel_file else ())
            append_section("sentinel_file_cli", sentinel_file_cli)

            output_text = "\n".join(lines)

        if args.output:
            output_path = _expand_pathlike(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text + "\n", encoding="utf-8")
        else:
            print(output_text)

        return 0

    # Config files listing
    if args.list_config_files:
        if any([args.set_env, args.export, args.ensure, should_run_command,
                args.print_pythonpath, args.print_sys_path, args.print_config,
                args.print_discovery, args.list_profiles, args.list_sentinels,
                args.list_add_paths, args.list_root_hints]):
            parser.error("--list-config-files nie może być łączone z innymi trybami wypisywania/uruchamiania")

        env_config_display = _fmt_seq(env_config_entries_display_raw)
        cli_config_display = _fmt_seq(cli_config_entries_display_raw)
        config_roots_display = _fmt_seq(config_root_entries_display_raw)
        config_loaded_display = _fmt_seq(config_files_used)
        include_edges_display = tuple(
            f"{_format_path(parent)} -> {_format_path(child)}"
            for parent, child in config_include_edges
        )
        include_edges_payload = [
            {
                "parent": parent,
                "child": child,
                "parent_display": _format_path(parent),
                "child_display": _format_path(child),
            }
            for parent, child in config_include_edges
        ]
        inline_definitions_payload = [
            {"source": source, "definition": definition}
            for source, definition in config_inline_serialized
        ]

        if args.format == "json":
            payload = {
                "env": list(env_config_files),
                "env_display": list(env_config_display),
                "env_var": ENV_CONFIG_FILES,
                "cli": list(config_cli),
                "cli_display": list(cli_config_display),
                "roots": list(config_root_entries_display_raw),
                "roots_display": list(config_roots_display),
                "loaded": list(config_files_used),
                "loaded_display": list(config_loaded_display),
                "inline_sources": list(config_inline_sources),
                "inline_definitions": inline_definitions_payload,
                "include_edges": include_edges_payload,
                "include_edges_display": list(include_edges_display),
            }
            output_text = json.dumps(payload, **json_kwargs)
        else:
            lines: list[str] = []

            def append_section(title: str, entries: Sequence[str]) -> None:
                if entries:
                    lines.append(f"{title}:")
                    for item in entries:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"{title}: (brak)")

            append_section("config_files_env", env_config_files)
            append_section("config_files_env_display", env_config_display)
            append_section("config_files_cli", config_cli)
            append_section("config_files_cli_display", cli_config_display)
            append_section("config_files_roots", config_root_entries_display_raw)
            append_section("config_files_roots_display", config_roots_display)
            append_section("config_files_loaded", config_files_used)
            append_section("config_files_loaded_display", config_loaded_display)
            append_section("config_inline_sources", config_inline_sources)
            for source, definition in config_inline_serialized:
                keys = sorted(definition.keys())
                keys_display = ", ".join(keys) if keys else "(brak)"
                lines.append(f"config_inline_definition_{source}_keys: {keys_display}")
            append_section("config_include_edges", tuple(
                f"{parent} -> {child}" for parent, child in config_include_edges
            ))
            append_section("config_include_edges_display", include_edges_display)
            lines.append(f"config_files_env_var: {ENV_CONFIG_FILES}")

            output_text = "\n".join(lines)

        if args.output:
            output_path = _expand_pathlike(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text + "\n", encoding="utf-8")
        else:
            print(output_text)

        return 0

    if args.export and not args.set_env:
        parser.error("opcja --export wymaga jednoczesnego użycia z --set-env")

    if args.format == "json" and args.set_env:
        parser.error("opcja --format=json nie jest dostępna razem z --set-env")

    # Compute repo discovery
    root_hint_candidate = args.root_hint if args.root_hint is not None else effective_config.get("root_hint")
    max_depth_param: Optional[int]
    if args.max_depth is not None:
        max_depth_param = args.max_depth
    elif config_max_depth_value is not None:
        max_depth_param = int(config_max_depth_value)
    else:
        max_depth_param = env_max_depth
    repo_discovery = get_repo_info(
        root_hint_candidate,
        sentinels=sentinel_arg,
        allow_git=allow_git_param,
        max_depth=max_depth_param,
    )
    repo_root_path = repo_discovery.root
    additional_paths_cli = tuple(args.additional_paths) if args.additional_paths else ()
    config_additional_paths = tuple(effective_config.get("additional_paths", ()))
    config_additional_path_files = tuple(effective_config.get("additional_path_files", ()))
    additional_paths_from_config_files: Tuple[str, ...] = ()
    if config_additional_path_files:
        collected_config: list[str] = []
        for file_entry in config_additional_path_files:
            file_path = _expand_pathlike(file_entry)
            try:
                collected_config.extend(_load_additional_paths_from_file(file_path))
            except FileNotFoundError:
                parser.error(f"plik ścieżek {file_path} z konfiguracji nie istnieje")
            except OSError as exc:
                parser.error(f"nie można odczytać pliku ścieżek {file_path}: {exc}")
            except ValueError as exc:
                parser.error(str(exc))
        additional_paths_from_config_files = tuple(collected_config)
    additional_paths_from_cli_files: Tuple[str, ...] = ()
    additional_cli_file_paths: Tuple[str, ...] = ()
    if args.additional_path_files:
        collected_cli: list[str] = []
        cli_files_resolved: list[str] = []
        for file_entry in args.additional_path_files:
            file_path = _expand_pathlike(file_entry)
            cli_files_resolved.append(str(file_path))
            try:
                collected_cli.extend(_load_additional_paths_from_file(file_path))
            except FileNotFoundError:
                parser.error(f"plik ścieżek {file_path} nie istnieje")
            except OSError as exc:
                parser.error(f"nie można odczytać pliku ścieżek {file_path}: {exc}")
            except ValueError as exc:
                parser.error(str(exc))
        additional_paths_from_cli_files = tuple(collected_cli)
        additional_cli_file_paths = tuple(cli_files_resolved)
    include_env_paths = not args.no_env_add_paths
    config_use_env_paths = effective_config.get("use_env_additional_paths")
    if not args.no_env_add_paths and config_use_env_paths is not None:
        include_env_paths = bool(config_use_env_paths)
    env_additional: Tuple[str, ...] = _get_env_additional_paths() if include_env_paths else ()
    combined_additional: Tuple[str, ...] = (
        *additional_paths_from_config_files,
        *config_additional_paths,
        *additional_paths_from_cli_files,
        *additional_paths_cli,
    )
    normalized_additional = _resolve_additional_paths(
        repo_root_path, combined_additional, include_env=include_env_paths
    )
    normalized_additional_display = _fmt_seq(normalized_additional)
    additional_cli_display_raw = tuple(str(Path(entry)) for entry in additional_paths_cli)
    additional_cli_display = _fmt_seq(additional_cli_display_raw)
    additional_config_display_raw = tuple(str(Path(entry)) for entry in config_additional_paths)
    additional_config_display = _fmt_seq(additional_config_display_raw)
    additional_cli_path_files_display_raw = tuple(
        str(Path(entry)) for entry in additional_cli_file_paths
    )
    additional_cli_path_files_display = _fmt_seq(
        additional_cli_path_files_display_raw
    )
    additional_config_files_display_raw = tuple(
        str(Path(entry)) for entry in config_additional_path_files
    )
    additional_config_files_display = _fmt_seq(
        additional_config_files_display_raw
    )
    additional_cli_file_entries_display_raw = tuple(
        str(Path(entry)) for entry in additional_paths_from_cli_files
    )
    additional_cli_file_entries_display = _fmt_seq(
        additional_cli_file_entries_display_raw
    )
    additional_cli_files_display = additional_cli_file_entries_display
    additional_config_file_entries_display_raw = tuple(
        str(Path(entry)) for entry in additional_paths_from_config_files
    )
    additional_config_file_entries_display = _fmt_seq(
        additional_config_file_entries_display_raw
    )
    env_additional_display = _fmt_seq(env_additional)

    # list-add-paths
    if args.list_add_paths:
        if any([args.set_env, args.export, args.ensure, should_run_command,
                args.print_pythonpath, args.print_sys_path, args.print_config,
                args.print_discovery, args.list_profiles, args.list_sentinels,
                args.list_config_files, args.list_root_hints]):
            parser.error("--list-add-paths nie może być łączone z innymi trybami wypisywania/uruchamiania")

        if args.format == "json":
            payload = {
                "effective": list(normalized_additional),
                "effective_display": list(normalized_additional_display),
                "config": list(config_additional_paths),
                "config_display": list(additional_config_display),
                "config_file_entries": list(additional_paths_from_config_files),
                "config_file_entries_display": list(
                    additional_config_file_entries_display
                ),
                "config_files": list(config_additional_path_files),
                "config_files_display": list(additional_config_files_display),
                "env_included": include_env_paths,
                "env": list(env_additional),
                "env_display": list(env_additional_display),
                "env_var": ENV_ADDITIONAL_PATHS,
                "cli": list(additional_paths_cli),
                "cli_display": list(additional_cli_display),
                "cli_file_entries": list(additional_paths_from_cli_files),
                "cli_file_entries_display": list(additional_cli_file_entries_display),
                "cli_files": list(additional_cli_file_paths),
                "cli_files_display": list(additional_cli_path_files_display),
            }
            output_text = json.dumps(payload, **json_kwargs)
        else:
            lines: list[str] = []

            def append_section(title: str, entries: Sequence[str]) -> None:
                if entries:
                    lines.append(f"{title}:")
                    for item in entries:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"{title}: (brak)")

            append_section("additional_paths_effective", normalized_additional_display)
            append_section("additional_paths_config", additional_config_display)
            append_section(
                "additional_paths_config_file_entries", additional_config_file_entries_display
            )
            append_section("additional_paths_config_files", additional_config_files_display)
            lines.append(
                "additional_paths_env_included: "
                + ("true" if include_env_paths else "false")
            )
            if include_env_paths:
                append_section("additional_paths_env", env_additional_display)
            else:
                lines.append("additional_paths_env: (wyłączone)")
            lines.append(f"additional_paths_env_var: {ENV_ADDITIONAL_PATHS}")
            append_section("additional_paths_cli", additional_cli_display)
            append_section(
                "additional_paths_cli_file_entries", additional_cli_file_entries_display
            )
            append_section("additional_paths_cli_files", additional_cli_path_files_display)

            output_text = "\n".join(lines)

        if args.output:
            output_path = _expand_pathlike(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text + "\n", encoding="utf-8")
        else:
            print(output_text)

        return 0

    # Profiles listing
    if args.list_profiles:
        if any([args.set_env, args.export, args.ensure, should_run_command,
                args.print_pythonpath, args.print_sys_path, args.print_config,
                args.print_discovery, args.list_sentinels,
                args.list_config_files, args.list_add_paths, args.list_root_hints]):
            parser.error("--list-profiles nie może być łączone z innymi trybami wypisywania/uruchamiania")

        defined_profiles = []
        for name in sorted(config_profiles):
            profile_entry = config_profiles[name]
            resolved_values = profile_resolution_cache.get(name, {})
            defined_profiles.append(
                {
                    "name": name,
                    "extends": list(profile_entry.extends),
                    "defined_keys": sorted(profile_entry.values.keys()),
                    "resolved_keys": sorted(resolved_values.keys()),
                }
            )

        if args.format == "json":
            payload = {
                "profiles": defined_profiles,
                "default": list(config_default_profiles),
                "selected": list(selected_profiles),
                "env_added": list(env_profiles_added),
                "env_removed": list(env_profiles_removed),
                "cli_added": list(cli_profiles_added),
                "cli_removed": list(cli_profiles_removed),
            }
            output_text = json.dumps(payload, **json_kwargs)
        else:
            lines: list[str] = []

            if defined_profiles:
                lines.append("profiles_defined:")
                for entry in defined_profiles:
                    extends_display = (
                        ", ".join(entry["extends"]) if entry["extends"] else "(brak)"
                    )
                    defined_keys_display = (
                        ", ".join(entry["defined_keys"]) if entry["defined_keys"] else "(brak)"
                    )
                    resolved_keys_display = (
                        ", ".join(entry["resolved_keys"]) if entry["resolved_keys"] else "(brak)"
                    )
                    lines.append(
                        "  - {name} (extends: {extends}; defined_keys: {defined_keys}; "
                        "resolved_keys: {resolved})".format(
                            name=entry["name"],
                            extends=extends_display,
                            defined_keys=defined_keys_display,
                            resolved=resolved_keys_display,
                        )
                    )
            else:
                lines.append("profiles_defined: (brak)")

            def append_section(title: str, entries: Sequence[str]) -> None:
                if entries:
                    lines.append(f"{title}:")
                    for item in entries:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"{title}: (brak)")

            append_section("default_profiles", config_default_profiles)
            append_section("env_profiles_added", env_profiles_added)
            append_section("env_profiles_removed", env_profiles_removed)
            append_section("cli_profiles_added", cli_profiles_added)
            append_section("cli_profiles_removed", cli_profiles_removed)
            append_section("selected_profiles", selected_profiles)

            output_text = "\n".join(lines)

        if args.output:
            output_path = _expand_pathlike(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text + "\n", encoding="utf-8")
        else:
            print(output_text)

        return 0

    # Discovery details
    if args.print_discovery:
        start_str = str(repo_discovery.start)
        start_display = _format_path(start_str)
        sentinel_value = repo_discovery.sentinel
        depth_value = repo_discovery.depth
        allow_git_details = {
            "effective": effective_allow_git,
            "cli": args.allow_git,
            "config": config_allow_git_value,
            "env": env_allow_git,
            "env_raw": env_allow_git_raw,
        }
        max_depth_details = {
            "effective": max_depth_param,
            "cli": args.max_depth,
            "config": config_max_depth_value,
            "env": env_max_depth,
            "env_raw": env_max_depth_raw,
        }
        if args.format == "json":
            payload = {
                "root": _format_path(str(repo_root_path)),
                "root_raw": str(repo_root_path),
                "method": repo_discovery.method,
                "sentinel": sentinel_value,
                "depth": depth_value,
                "start": start_str,
                "start_display": start_display,
                "allow_git": allow_git_details,
                "max_depth": max_depth_details,
                "sentinels": list(sentinel_arg),
                "sentinel_candidates": list(sentinel_candidates),
            }
            output_text = json.dumps(payload, **json_kwargs)
        else:
            lines = [
                f"discovery_root: {_format_path(str(repo_root_path))}",
                f"discovery_root_raw: {str(repo_root_path)}",
                f"discovery_method: {repo_discovery.method}",
            ]
            lines.append(
                "discovery_sentinel: "
                + (sentinel_value if sentinel_value is not None else "(brak)")
            )
            lines.append(
                "discovery_depth: "
                + (str(depth_value) if depth_value is not None else "(brak)")
            )
            lines.append(f"discovery_start: {start_str}")
            lines.append(f"discovery_start_display: {start_display}")
            lines.append(f"discovery_allow_git_effective: {effective_allow_git}")
            lines.append(
                "discovery_allow_git_cli: "
                + (str(args.allow_git) if args.allow_git is not None else "(brak)")
            )
            lines.append(
                "discovery_allow_git_config: "
                + (
                    str(config_allow_git_value)
                    if config_allow_git_value is not None
                    else "(brak)"
                )
            )
            lines.append(
                "discovery_allow_git_env: "
                + (str(env_allow_git) if env_allow_git is not None else "(brak)")
            )
            lines.append(
                "discovery_allow_git_env_raw: "
                + (env_allow_git_raw if env_allow_git_raw is not None else "(brak)")
            )
            lines.append(
                "discovery_max_depth_effective: "
                + (str(max_depth_param) if max_depth_param is not None else "(brak)")
            )
            lines.append(
                "discovery_max_depth_cli: "
                + (str(args.max_depth) if args.max_depth is not None else "(brak)")
            )
            lines.append(
                "discovery_max_depth_config: "
                + (
                    str(config_max_depth_value)
                    if config_max_depth_value is not None
                    else "(brak)"
                )
            )
            lines.append(
                "discovery_max_depth_env: "
                + (
                    str(env_max_depth) if env_max_depth is not None else "(brak)"
                )
            )
            lines.append(
                "discovery_max_depth_env_raw: "
                + (env_max_depth_raw if env_max_depth_raw is not None else "(brak)")
            )
            lines.append(
                "discovery_sentinels_effective: "
                + (", ".join(sentinel_arg) if sentinel_arg else "(brak)")
            )
            lines.append(
                "discovery_sentinels_candidates: "
                + (", ".join(sentinel_candidates) if sentinel_candidates else "(brak)")
            )
            output_text = "\n".join(lines)
        if args.output:
            output_path = _expand_pathlike(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text + "\n", encoding="utf-8")
        else:
            print(output_text)

        return 0

    # Print-config
    if args.print_config:
        if args.format == "json":
            payload = {
                "repo_root": _format_path(str(repo_root_path)),
                "sentinels": list(sentinel_arg),
                "sentinels_cli": list(tuple(args.sentinels) if args.sentinels else ()),
                "sentinels_file": list(tuple(_load_sentinels_from_file(_expand_pathlike(args.sentinel_file))) if args.sentinel_file else ()),
                "sentinels_env": list(env_sentinels),
                "allow_git": {
                            "effective": effective_allow_git,
                            "cli": args.allow_git,
                            "env": env_allow_git,
                            "env_raw": env_allow_git_raw,
                        },
                "max_depth": {
                            "effective": max_depth_param,
                            "cli": args.max_depth,
                            "config": config_max_depth_value,
                            "env": env_max_depth,
                            "env_raw": env_max_depth_raw,
                        },
                "position": position,
                "include_env_additional_paths": include_env_paths,
                "set_env_format": args.set_env_format,
                "additional_paths": {
                    "normalized": list(normalized_additional_display),
                    "cli": list(additional_cli_display),
                    "cli_files": list(additional_cli_files_display),
                    "config": list(additional_config_display),
                    "config_files": list(additional_config_files_display),
                    "env": list(env_additional),
                },
                "pythonpath_var": pythonpath_var,
                "ensure": bool(args.ensure),
                "chdir": bool(args.chdir),
                "path_style": path_style,
                "discovery": {
                    "method": repo_discovery.method,
                    "sentinel": repo_discovery.sentinel,
                    "depth": repo_discovery.depth,
                    "start": str(repo_discovery.start),
                    "start_display": _format_path(str(repo_discovery.start)),
                },
                "config": {
                    "files": list(config_files_used),
                    "includes": [
                        {"parent": parent, "child": child}
                        for parent, child in config_include_edges
                    ],
                    "inline": {
                        "sources": list(config_inline_sources),
                        "entries": [
                            {"source": source, "values": definition}
                            for source, definition in config_inline_serialized
                        ],
                    },
                    "values": {
                        "root_hint": effective_config.get("root_hint"),
                        "sentinels": list(tuple(effective_config.get("sentinels", ()) )),
                        "additional_paths": list(tuple(effective_config.get("additional_paths", ()) )),
                        "additional_path_files": list(tuple(effective_config.get("additional_path_files", ()) )),
                        "allow_git": config_allow_git_value,
                        "max_depth": config_max_depth_value,
                        "use_env_additional_paths": effective_config.get("use_env_additional_paths"),
                        "position": effective_config.get("position"),
                        "path_style": effective_config.get("path_style"),
                        "pythonpath_var": effective_config.get("pythonpath_var"),
                    },
                    "profiles": {
                        "defined": list(config_profiles_defined),
                        "default": list(config_default_profiles),
                        "env": list(env_profiles_added),
                        "env_removed": list(env_profiles_removed),
                        "cli": list(cli_profiles_added),
                        "cli_removed": list(cli_profiles_removed),
                        "selected": list(selected_profiles),
                        "definitions": {
                            name: {
                                "extends": list(config_profiles[name].extends),
                                "values": _serialize_profile_values(
                                    config_profiles[name].values
                                ),
                                "resolved": _serialize_profile_values(
                                    profile_resolution_cache[name]
                                ),
                            }
                            for name in config_profiles_defined
                        },
                    },
                },
                "shell": {
                    "enabled": shell_requested,
                    "path": args.shell_path if shell_requested else None,
                    "args": list(command) if shell_requested else [],
                },
            }
            output_text = json.dumps(payload, **json_kwargs)
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
                f"repo_root: {_format_path(str(repo_root_path))}",
                "sentinels: " + (", ".join(sentinel_arg) if sentinel_arg else "(brak)"),
                "sentinels_cli: "
                + (", ".join(args.sentinels) if args.sentinels else "(brak)"),
                "sentinels_file: "
                + (", ".join(_load_sentinels_from_file(_expand_pathlike(args.sentinel_file))) if args.sentinel_file else "(brak)"),
                "sentinels_env: "
                + (", ".join(env_sentinels) if env_sentinels else "(brak)"),
                f"allow_git_effective: {effective_allow_git}",
                "allow_git_cli: "
                + (str(args.allow_git) if args.allow_git is not None else "(brak)"),
                "allow_git_env: "
                + (str(env_allow_git) if env_allow_git is not None else "(brak)"),
                "allow_git_env_raw: "
                + (env_allow_git_raw if env_allow_git_raw is not None else "(brak)"),
                "max_depth_effective: "
                + (
                    str(max_depth_param)
                    if max_depth_param is not None
                    else "(brak)"
                ),
                "max_depth_cli: "
                + (str(args.max_depth) if args.max_depth is not None else "(brak)"),
                "max_depth_config: "
                + (
                    str(config_max_depth_value)
                    if config_max_depth_value is not None
                    else "(brak)"
                ),
                "max_depth_env: "
                + (
                    str(env_max_depth) if env_max_depth is not None else "(brak)"
                ),
                "max_depth_env_raw: "
                + (env_max_depth_raw if env_max_depth_raw is not None else "(brak)"),
                f"position: {position}",
                f"include_env_additional_paths: {include_env_paths}",
                f"set_env_format: {args.set_env_format}",
                f"path_style: {path_style}",
            ]
            lines.extend(
                format_section(
                    "additional_paths_normalized",
                    normalized_additional_display,
                )
            )
            lines.extend(
                format_section(
                    "additional_paths_config", additional_config_display
                )
            )
            lines.extend(
                format_section("additional_paths_cli", additional_cli_display)
            )
            lines.extend(
                format_section(
                    "additional_paths_config_files",
                    additional_config_files_display,
                )
            )
            lines.extend(
                format_section(
                    "additional_paths_cli_files", additional_cli_files_display
                )
            )
            lines.extend(
                format_section("additional_paths_env", env_additional_display)
            )
            lines.append(f"pythonpath_var: {pythonpath_var}")
            lines.append(f"ensure: {bool(args.ensure)}")
            lines.append(f"chdir: {bool(args.chdir)}")
            lines.append(f"shell_enabled: {shell_requested}")
            lines.append(f"discovery_method: {repo_discovery.method}")
            lines.append(
                "discovery_sentinel: "
                + (repo_discovery.sentinel if repo_discovery.sentinel else "(brak)")
            )
            lines.append(
                "discovery_depth: "
                + (str(repo_discovery.depth) if repo_discovery.depth is not None else "(brak)")
            )
            lines.append(f"discovery_start: {str(repo_discovery.start)}")
            lines.append(f"discovery_start_display: {_format_path(str(repo_discovery.start))}")
            lines.extend(format_section("config_files", config_files_used))
            lines.extend(
                format_section("config_inline_sources", config_inline_sources)
            )
            if config_inline_serialized:
                lines.append("config_inline_definitions:")
                for source, definition in config_inline_serialized:
                    lines.append(f"  - {source}:")
                    if not definition:
                        lines.append("      (brak)")
                        continue
                    for key, value in definition.items():
                        if key == "profiles" and isinstance(value, dict):
                            lines.append("      profiles:")
                            if value:
                                for profile_name, profile_data in value.items():
                                    lines.append(f"        {profile_name}:")
                                    extends = profile_data.get("extends", [])
                                    extends_display = (
                                        ", ".join(extends)
                                        if extends
                                        else "(brak)"
                                    )
                                    lines.append(
                                        f"          extends: {extends_display}"
                                    )
                                    profile_values = profile_data.get("values", {})
                                    if profile_values:
                                        lines.append("          values:")
                                        for sub_key, sub_value in profile_values.items():
                                            lines.append(
                                                "            "
                                                f"{sub_key}: {_format_config_value(sub_value)}"
                                            )
                                    else:
                                        lines.append("          values: (brak)")
                            else:
                                lines.append("        (brak)")
                        else:
                            lines.append(
                                f"      {key}: {_format_config_value(value)}"
                            )
            else:
                lines.append("config_inline_definitions: (brak)")
            if config_include_edges:
                def edges():
                    for parent, child in config_include_edges:
                        yield f"{parent} -> {child}"
                lines.extend(format_section("config_includes", tuple(edges())))
            lines.append(
                "config_root_hint: "
                + (str(effective_config.get("root_hint")) if effective_config.get("root_hint") else "(brak)")
            )
            lines.extend(
                format_section("config_sentinels", tuple(effective_config.get("sentinels", ()) ))
            )
            lines.extend(
                format_section("config_additional_paths", tuple(effective_config.get("additional_paths", ()) ))
            )
            lines.extend(
                format_section(
                    "config_additional_path_files", tuple(effective_config.get("additional_path_files", ()) )
                )
            )
            lines.append(
                "config_allow_git: "
                + (
                    str(config_allow_git_value)
                    if config_allow_git_value is not None
                    else "(brak)"
                )
            )
            lines.append(
                "config_max_depth: "
                + (
                    str(config_max_depth_value)
                    if config_max_depth_value is not None
                    else "(brak)"
                )
            )
            lines.append(
                "config_use_env_additional_paths: "
                + (
                    str(effective_config.get("use_env_additional_paths"))
                    if effective_config.get("use_env_additional_paths") is not None
                    else "(brak)"
                )
            )
            lines.append(
                "config_position: "
                + (str(effective_config.get("position")) if effective_config.get("position") else "(brak)")
            )
            lines.append(
                "config_path_style: "
                + (str(effective_config.get("path_style")) if effective_config.get("path_style") else "(brak)")
            )
            lines.append(
                "config_pythonpath_var: "
                + (
                    str(effective_config.get("pythonpath_var")) if effective_config.get("pythonpath_var") else "(brak)"
                )
            )
            lines.extend(
                format_section(
                    "config_profiles_defined", config_profiles_defined
                )
            )
            lines.extend(
                format_section(
                    "config_default_profiles", config_default_profiles
                )
            )
            lines.extend(
                format_section("profiles_env", env_profiles_added)
            )
            lines.extend(
                format_section("profiles_env_removed", env_profiles_removed)
            )
            lines.extend(
                format_section("profiles_cli", cli_profiles_added)
            )
            lines.extend(
                format_section("profiles_cli_removed", cli_profiles_removed)
            )
            lines.extend(
                format_section("profiles_selected", selected_profiles)
            )
            if config_profiles_defined:
                lines.append("config_profiles_definitions:")
                for name in config_profiles_defined:
                    profile = config_profiles[name]
                    lines.append(f"  - {name}:")
                    extends_display = (
                        ", ".join(profile.extends)
                        if profile.extends
                        else "(brak)"
                    )
                    lines.append(f"      extends: {extends_display}")
                    if profile.values:
                        lines.append("      values:")
                        for key, value in profile.values.items():
                            lines.append(
                                f"        {key}: {_format_config_value(value)}"
                            )
                    else:
                        lines.append("      values: (brak)")
                    resolved_values = profile_resolution_cache[name]
                    if resolved_values:
                        lines.append("      resolved:")
                        for key, value in resolved_values.items():
                            lines.append(
                                f"        {key}: {_format_config_value(value)}"
                            )
                    else:
                        lines.append("      resolved: (brak)")
            else:
                lines.append("config_profiles_definitions: (brak)")
            output_text = "\n".join(lines)

        if args.output:
            output_path = _expand_pathlike(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text + "\n", encoding="utf-8")
        else:
            print(output_text)
        return 0

    # If we got here, we may need to add paths and/or run a command
    context = (
        chdir_repo_root(
            repo_root_path,
            sentinels=sentinel_arg,
            allow_git=allow_git_param,
            max_depth=max_depth_param,
        )
        if args.chdir
        else nullcontext(repo_root_path)
    )

    with context as repo_root:
        repo_str = str(repo_root)
        display_separator = PATH_STYLE_SEPARATORS[path_style]
        repo_display = _format_path(repo_str)
        discovery_start_str = str(repo_discovery.start)
        discovery_start_display = _format_path(discovery_start_str)
        python_executable = args.python_executable
        command_to_run = list(command)
        if module_name:
            command_to_run = [python_executable, "-m", module_name, *command_to_run]
        shell_path_used: Optional[str] = None
        shell_command_display: Optional[str] = None
        if shell_requested:
            shell_path_used = args.shell_path or _get_default_shell()
            shell_command = _build_shell_command(shell_path_used, command_to_run)
            if command:
                shell_command_display = shlex.join(command)
            command_to_run = list(shell_command)
        pythonpath_entries: Tuple[str, ...] | None = None
        pythonpath_value: Optional[str] = None
        pythonpath_display_entries: Tuple[str, ...] | None = None
        pythonpath_display_value: Optional[str] = None
        sys_path_entries: Tuple[str, ...] | None = None
        sys_path_value: Optional[str] = None
        sys_path_display_entries: Tuple[str, ...] | None = None
        sys_path_display_value: Optional[str] = None
        sys_path_snapshot = list(sys.path)

        if args.print_discovery and args.verbose:
            print(f"{prefix} punkt startowy wykrycia: {discovery_start_display}", file=sys.stderr)

        if args.print_pythonpath:
            pythonpath_buffer: list[str] = []
            _apply_entries(
                pythonpath_buffer, (repo_str, *normalized_additional), position
            )
            pythonpath_entries = tuple(pythonpath_buffer)
            pythonpath_value = os.pathsep.join(pythonpath_buffer)
            pythonpath_display_entries = _format_path_sequence(
                pythonpath_entries, path_style
            )
            pythonpath_display_value = display_separator.join(
                pythonpath_display_entries
            )

        if args.print_sys_path:
            if args.ensure or should_run_command:
                base = list(sys.path)
            else:
                base = list(sys_path_snapshot)
            preview = list(base)
            _apply_entries(preview, (repo_str, *normalized_additional), position)
            sys_path_entries = tuple(preview)
            sys_path_value = "\n".join(preview)
            sys_path_display_entries = _format_path_sequence(
                sys_path_entries, path_style
            )
            sys_path_display_value = "\n".join(sys_path_display_entries)

        if should_run_command:
            env = os.environ.copy()
            existing_pythonpath = env.get(pythonpath_var, "")
            path_entries = [entry for entry in existing_pythonpath.split(os.pathsep) if entry]
            inserted = _apply_entries(
                path_entries, (repo_str, *normalized_additional), position
            )
            env[pythonpath_var] = os.pathsep.join(path_entries) if path_entries else repo_str

            if args.set_env:
                env[args.set_env] = repo_str

            completed = subprocess.run(command_to_run, check=False, env=env)
            return completed.returncode

        output_text: Optional[str]
        if args.set_env:
            assignment = _format_env_assignment(
                args.set_env, repo_display, args.set_env_format
            )
            output_text = assignment
        else:
            if args.print_pythonpath:
                assert pythonpath_value is not None
                assert pythonpath_entries is not None
                assert pythonpath_display_entries is not None
                assert pythonpath_display_value is not None
                if args.format == "json":
                    payload = {
                        "repo_root": repo_display,
                        "additional_paths": list(normalized_additional_display),
                        "pythonpath": pythonpath_display_value,
                        "pythonpath_entries": list(pythonpath_display_entries),
                    }
                    output_text = json.dumps(payload, **json_kwargs)
                else:
                    output_text = pythonpath_display_value
            elif args.print_sys_path:
                assert sys_path_value is not None
                assert sys_path_entries is not None
                assert sys_path_display_entries is not None
                assert sys_path_display_value is not None
                if args.format == "json":
                    payload = {
                        "repo_root": repo_display,
                        "additional_paths": list(normalized_additional_display),
                        "sys_path": sys_path_display_value,
                        "sys_path_entries": list(sys_path_display_entries),
                    }
                    output_text = json.dumps(payload, **json_kwargs)
                else:
                    output_text = sys_path_display_value
            elif args.print_config:
                # Already handled earlier; keeping branch for completeness
                output_text = ""
            else:
                if args.format == "json":
                    payload = {
                        "repo_root": repo_display,
                        "additional_paths": list(normalized_additional_display),
                    }
                    output_text = json.dumps(payload, **json_kwargs)
                else:
                    output_text = repo_display

        if args.output:
            output_path = _expand_pathlike(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text + "\n", encoding="utf-8")
        else:
            print(output_text)
        return 0


if __name__ == "__main__":  # pragma: no cover - testowane poprzez main()
    raise SystemExit(main())
