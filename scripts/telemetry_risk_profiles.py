"""Kompatybilna warstwa eksportująca presety profili ryzyka oraz proste CLI."""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

try:  # pragma: no cover - moduł konfiguracyjny może być niedostępny w starszych gałęziach
    from bot_core.config.loader import load_core_config
except Exception:  # pragma: no cover - środowisko bez warstwy konfiguracji
    load_core_config = None  # type: ignore[assignment]

from bot_core.runtime.telemetry_risk_profiles import (
    MetricsRiskProfileResolver,
    get_metrics_service_config_overrides,
    get_metrics_service_env_overrides,
    get_metrics_service_overrides,
    get_risk_profile,
    list_risk_profile_files,
    list_risk_profile_names,
    load_risk_profiles_from_file,
    load_risk_profiles_with_metadata,
    register_risk_profiles,
    risk_profile_metadata,
    summarize_risk_profile,
)

__all__ = [
    "MetricsRiskProfileResolver",
    "get_metrics_service_config_overrides",
    "get_metrics_service_env_overrides",
    "get_metrics_service_overrides",
    "get_risk_profile",
    "list_risk_profile_files",
    "load_risk_profiles_with_metadata",
    "list_risk_profile_names",
    "load_risk_profiles_from_file",
    "register_risk_profiles",
    "risk_profile_metadata",
]


RENDER_SECTION_CHOICES: tuple[str, ...] = (
    "metrics_service_overrides",
    "metrics_service_config_overrides",
    "metrics_service_env_overrides",
    "cli_flags",
    "env_assignments",
    "env_assignments_format",
    "sources",
    "risk_profile",
    "summary",
    "core_config",
)


# ---------------------------------------------------------------------------
# Pomocnicze funkcje CLI


def _load_core_metadata(path: str | None) -> Mapping[str, Any] | None:
    if not path:
        return None
    if load_core_config is None:  # pragma: no cover - defensywne
        raise RuntimeError("Obsługa --core-config wymaga modułu bot_core.config")

    target = Path(path).expanduser()
    core_config = load_core_config(target)

    metadata: dict[str, Any] = {"path": str(target)}
    metrics_cfg = getattr(core_config, "metrics_service", None)
    if metrics_cfg is None:
        metadata["warning"] = "metrics_service_missing"
        return metadata

    metrics_meta: dict[str, Any] = {
        "host": getattr(metrics_cfg, "host", None),
        "port": getattr(metrics_cfg, "port", None),
        "risk_profile": getattr(metrics_cfg, "ui_alerts_risk_profile", None),
        "risk_profiles_file": getattr(
            metrics_cfg, "ui_alerts_risk_profiles_file", None
        ),
    }

    tls_cfg = getattr(metrics_cfg, "tls", None)
    if tls_cfg is not None:
        metrics_meta["tls_enabled"] = bool(getattr(tls_cfg, "enabled", False))
        metrics_meta["client_auth"] = bool(
            getattr(tls_cfg, "require_client_auth", False)
        )
        metrics_meta["client_cert_configured"] = bool(
            getattr(tls_cfg, "certificate_path", None)
        )
        metrics_meta["client_key_configured"] = bool(
            getattr(tls_cfg, "private_key_path", None)
        )
        metrics_meta["root_cert_configured"] = bool(
            getattr(tls_cfg, "client_ca_path", None)
        )

    if getattr(metrics_cfg, "auth_token", None):
        metrics_meta["auth_token_configured"] = True

    metadata["metrics_service"] = {
        key: value for key, value in metrics_meta.items() if value not in (None, "")
    }
    return metadata


def _load_additional_profiles(
    files: Iterable[str] | None,
    directories: Iterable[str] | None,
) -> list[Mapping[str, Any]]:
    metadata_entries: list[Mapping[str, Any]] = []
    for raw in files or []:
        _, metadata = load_risk_profiles_with_metadata(
            raw, origin_label="cli:file"
        )
        metadata_entries.append(metadata)
    for raw in directories or []:
        _, metadata = load_risk_profiles_with_metadata(
            raw, origin_label="cli:dir"
        )
        metadata_entries.append(metadata)
    return metadata_entries


def _dump_payload(
    payload: Mapping[str, Any], *, output: str | None, fmt: str = "json"
) -> None:
    if fmt not in {"json", "yaml"}:
        raise ValueError(f"Nieobsługiwany format serializacji: {fmt}")

    if fmt == "yaml":
        rendered = yaml.safe_dump(
            payload,
            allow_unicode=True,
            sort_keys=False,
        )
    else:
        rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    _write_text_output(rendered, output=output)


def _infer_format_from_output(path: str | None) -> str | None:
    if not path:
        return None

    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix in {".json", ".jsonl"}:
        return "json"
    return None


def _resolve_yaml_json_format(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    output_path: str | None,
    default: str = "json",
) -> str:
    allowed = {"json", "yaml"}
    inferred = _infer_format_from_output(output_path)
    if inferred is not None and inferred not in allowed:
        parser.error("Rozszerzenie pliku wyjściowego nie obsługuje formatu json/yaml")

    explicit = getattr(args, "format", None)
    if explicit is None:
        final = inferred or default
    else:
        if explicit not in allowed:
            parser.error("Format musi być jednym z: json, yaml")
        if inferred is not None and inferred != explicit:
            parser.error(
                "Rozszerzenie pliku wyjściowego nie zgadza się z wymuszonym formatem"
            )
        final = explicit

    if final not in allowed:
        parser.error("Format musi być jednym z: json, yaml")

    setattr(args, "format", final)
    return final


def _add_shared_arguments(target: argparse.ArgumentParser) -> None:
    target.add_argument(
        "--risk-profiles-file",
        action="append",
        dest="risk_profiles_files",
        metavar="PATH",
        help="Dodatkowe pliki JSON/YAML z profilami ryzyka",
    )
    target.add_argument(
        "--risk-profiles-dir",
        action="append",
        dest="risk_profiles_dirs",
        metavar="PATH",
        help="Katalog zawierający pliki z profilami ryzyka",
    )
    target.add_argument(
        "--core-config",
        metavar="PATH",
        help="Opcjonalny plik core.yaml w celu dołączenia metadanych runtime",
    )
    target.add_argument(
        "--output",
        metavar="PATH",
        help=(
            "Ścieżka wyjściowa dla raportu (domyślnie STDOUT); format może być "
            "wywnioskowany z rozszerzenia pliku"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="telemetry_risk_profiles",
        description=(
            "Zarządzanie presetami profili ryzyka telemetryjnego oraz audyt konfiguracji"
        ),
        conflict_handler="resolve",
    )
    _add_shared_arguments(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list", help="Wyświetl listę dostępnych profili ryzyka"
    )
    _add_shared_arguments(list_parser)
    list_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Dołącz szczegóły każdego profilu",
    )
    list_parser.add_argument(
        "--format",
        choices=("json", "yaml"),
        help=(
            "Format wyjścia raportu (json lub yaml). Domyślnie json lub zgodnie z rozszerzeniem "
            "pliku wyjściowego"
        ),
    )

    show_parser = subparsers.add_parser(
        "show", help="Wyświetl szczegóły wybranego profilu"
    )
    _add_shared_arguments(show_parser)
    show_parser.add_argument("name", help="Nazwa profilu ryzyka do wyświetlenia")
    show_parser.add_argument(
        "--format",
        choices=("json", "yaml"),
        help=(
            "Format wyjścia raportu (json lub yaml). Domyślnie json lub zgodnie z rozszerzeniem "
            "pliku wyjściowego"
        ),
    )

    render_parser = subparsers.add_parser(
        "render",
        help=(
            "Wygeneruj nadpisania konfiguracji MetricsService dla wskazanego profilu"
        ),
    )
    _add_shared_arguments(render_parser)
    render_parser.add_argument("name", help="Nazwa profilu ryzyka")
    render_parser.add_argument(
        "--format",
        choices=("json", "yaml", "cli", "env"),
        help=(
            "Format wyjścia (json, yaml, lista flag CLI lub przypisania zmiennych środowiskowych). "
            "Domyślnie json lub zgodny z rozszerzeniem pliku wyjściowego"
        ),
    )
    render_parser.add_argument(
        "--include-profile",
        action="store_true",
        help="Dołącz pełną definicję profilu do wyniku JSON/YAML (niedostępne dla formatów CLI/env)",
    )
    render_parser.add_argument(
        "--cli-style",
        choices=("equals", "space"),
        default="equals",
        help=(
            "Sposób formatowania wartości flag CLI (equals: --key=value, space: --key value)"
        ),
    )
    render_parser.add_argument(
        "--env-style",
        choices=("dotenv", "export"),
        default="dotenv",
        help=(
            "Sposób formatowania zmiennych środowiskowych (dotenv: KEY=value, export: export KEY=value)"
        ),
    )
    render_parser.add_argument(
        "--section",
        dest="sections",
        action="append",
        choices=RENDER_SECTION_CHOICES,
        metavar="NAME",
        help=(
            "Ogranicz wynik JSON/YAML do wskazanych sekcji (można podać wielokrotnie). "
            "Dostępne: " + ", ".join(RENDER_SECTION_CHOICES)
        ),
    )

    diff_parser = subparsers.add_parser(
        "diff",
        help="Porównaj dwa profile ryzyka i wypisz różnice w nadpisaniach",
    )
    _add_shared_arguments(diff_parser)
    diff_parser.add_argument("base", help="Nazwa profilu bazowego")
    diff_parser.add_argument("target", help="Profil, z którym porównujemy")
    diff_parser.add_argument(
        "--format",
        choices=("json", "yaml"),
        help=(
            "Format raportu różnic (json lub yaml). Domyślnie json lub zgodnie z rozszerzeniem "
            "pliku wyjściowego"
        ),
    )
    diff_parser.add_argument(
        "--include-profiles",
        action="store_true",
        help="Dołącz pełne definicje profili do wyniku JSON/YAML",
    )
    diff_parser.add_argument(
        "--hide-unchanged",
        action="store_true",
        help="Ukryj sekcje oznaczone jako niezmienione, aby uprościć raport",
    )
    diff_parser.add_argument(
        "--section",
        dest="sections",
        action="append",
        choices=(
            "diff",
            "summary",
            "cli",
            "env",
            "profiles",
            "core_config",
            "sources",
        ),
        metavar="NAME",
        help=(
            "Ogranicz wynik do wskazanych sekcji (można podać wielokrotnie). "
            "Domyślnie raport zawiera wszystkie sekcje."
        ),
    )
    diff_parser.add_argument(
        "--fail-on-diff",
        action="store_true",
        help=(
            "Zakończ działanie kodem 1, jeżeli wykryto różnice pomiędzy profilami"
        ),
    )
    diff_parser.add_argument(
        "--cli-style",
        choices=("equals", "space"),
        default="equals",
        help=(
            "Format flag CLI w sekcji porównania (equals: --key=value, space: --key value)"
        ),
    )
    diff_parser.add_argument(
        "--env-style",
        choices=("dotenv", "export"),
        default="dotenv",
        help=(
            "Format przypisań środowiskowych w sekcji porównania (dotenv lub export)"
        ),
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Zweryfikuj dostępność zadanych profili"
    )
    _add_shared_arguments(validate_parser)
    validate_parser.add_argument(
        "--require",
        action="append",
        dest="required_profiles",
        metavar="NAME",
        help="Profil, który musi istnieć (można podać wielokrotnie)",
    )
    validate_parser.add_argument(
        "--format",
        choices=("json", "yaml"),
        help=(
            "Format raportu (json lub yaml). Domyślnie json lub zgodnie z rozszerzeniem pliku"
        ),
    )

    return parser


def _handle_list(
    *,
    verbose: bool,
    selected: str | None,
    sources: list[Mapping[str, Any]],
    core_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "risk_profiles": list(list_risk_profile_names()),
        "sources": sources,
    }
    if verbose:
        payload["profiles"] = {
            name: risk_profile_metadata(name)
            for name in list_risk_profile_names()
        }
    if selected:
        payload["selected"] = selected.strip().lower()
    if core_metadata:
        payload["core_config"] = dict(core_metadata)
    return payload


def _handle_show(
    name: str,
    *,
    sources: list[Mapping[str, Any]],
    core_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized = name.strip().lower()
    metadata = risk_profile_metadata(normalized)
    payload: dict[str, Any] = {
        "risk_profile": metadata,
        "name": normalized,
        "sources": sources,
    }
    payload["metrics_service_overrides"] = get_metrics_service_overrides(normalized)
    if core_metadata:
        payload["core_config"] = dict(core_metadata)
    return payload


def _build_cli_flags(
    overrides: Mapping[str, Any], *, style: str = "equals"
) -> list[str]:
    flags: list[str] = []
    for option, value in sorted(overrides.items()):
        flag = "--" + option.replace("_", "-")
        if isinstance(value, bool):
            value_repr = "true" if value else "false"
        elif isinstance(value, (int, float)):
            value_repr = repr(value)
        else:
            value_repr = str(value)
        if style == "space":
            flags.append(f"{flag} {value_repr}")
        else:
            flags.append(f"{flag}={value_repr}")
    return flags


_DOTENV_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_./:@-]+$")


def _format_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if value is None:
        return ""
    return str(value)


def _quote_for_dotenv(value: str) -> str:
    if value == "":
        return '""'
    if _DOTENV_SAFE_VALUE.match(value):
        return value
    escaped = (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace('"', '\\"')
    )
    return f'"{escaped}"'


def _build_env_assignments(
    overrides: Mapping[str, Any], *, style: str = "dotenv"
) -> list[str]:
    assignments: list[str] = []
    for env_name, raw_value in sorted(overrides.items()):
        value_text = _format_env_value(raw_value)
        if style == "dotenv":
            value_repr = _quote_for_dotenv(value_text)
            assignments.append(f"{env_name}={value_repr}")
        elif style == "export":
            value_repr = shlex.quote(value_text)
            assignments.append(f"export {env_name}={value_repr}")
        else:  # pragma: no cover - zabezpieczenie przed nieznanym stylem
            raise ValueError(f"Nieobsługiwany styl env: {style}")
    return assignments


def _write_text_output(payload: str, *, output: str | None) -> None:
    if output:
        target = Path(output).expanduser()
        target.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
        if not payload.endswith("\n"):
            sys.stdout.write("\n")


def _handle_render(
    name: str,
    *,
    sources: list[Mapping[str, Any]],
    core_metadata: Mapping[str, Any] | None,
    fmt: str,
    include_profile: bool,
    cli_style: str,
    env_style: str,
    sections: Iterable[str] | None,
) -> tuple[Mapping[str, Any] | None, list[str] | None]:
    normalized = name.strip().lower()
    metadata = risk_profile_metadata(normalized)
    cli_overrides = dict(get_metrics_service_overrides(normalized))
    config_overrides = dict(get_metrics_service_config_overrides(normalized))
    env_overrides = dict(get_metrics_service_env_overrides(normalized))

    cli_flags = _build_cli_flags(cli_overrides, style=cli_style)
    env_assignments = _build_env_assignments(env_overrides, style=env_style)

    selected_sections = tuple(section.strip() for section in sections or [])

    if fmt in {"cli", "env"} and include_profile:
        raise ValueError(
            "Opcja --include-profile jest dostępna wyłącznie dla formatu json/yaml"
        )

    if fmt in {"cli", "env"} and selected_sections:
        raise ValueError(
            "Opcja --section jest dostępna wyłącznie dla formatów json oraz yaml"
        )

    if fmt == "cli":
        return None, cli_flags

    if fmt == "env":
        return None, env_assignments

    include_profile_section = include_profile or ("risk_profile" in selected_sections)
    include_summary_section = "summary" in selected_sections
    if not selected_sections and include_profile_section:
        include_summary_section = True

    payload: dict[str, Any] = {
        "name": normalized,
        "metrics_service_overrides": cli_overrides,
        "metrics_service_config_overrides": config_overrides,
        "metrics_service_env_overrides": env_overrides,
        "cli_flags": cli_flags,
        "env_assignments": env_assignments,
        "env_assignments_format": env_style,
        "sources": sources,
    }
    if include_profile_section:
        payload["risk_profile"] = metadata
    if include_summary_section:
        payload["summary"] = summarize_risk_profile(metadata)
    if core_metadata:
        payload["core_config"] = dict(core_metadata)

    if selected_sections:
        filtered: dict[str, Any] = {"name": payload["name"]}
        for key in selected_sections:
            if key == "env_assignments" and key in payload:
                filtered[key] = payload[key]
                if (
                    "env_assignments_format" in payload
                    and "env_assignments_format" not in selected_sections
                ):
                    filtered["env_assignments_format"] = payload[
                        "env_assignments_format"
                    ]
            elif key in payload:
                filtered[key] = payload[key]
        payload = filtered
    return payload, None


def _diff_mapping(
    base_map: Mapping[str, Any], target_map: Mapping[str, Any]
) -> dict[str, Any]:
    base_data = dict(base_map)
    target_data = dict(target_map)

    added = {key: deepcopy(value) for key, value in target_data.items() if key not in base_data}
    removed = sorted(key for key in base_data if key not in target_data)
    changed = {
        key: {"from": deepcopy(base_data[key]), "to": deepcopy(target_data[key])}
        for key in sorted(set(base_data) & set(target_data))
        if base_data[key] != target_data[key]
    }
    unchanged = {
        key: deepcopy(target_data[key])
        for key in sorted(set(base_data) & set(target_data))
        if base_data[key] == target_data[key]
    }

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": unchanged,
    }


def _diff_mapping_has_changes(diff: Mapping[str, Any]) -> bool:
    return bool(
        diff.get("added") or diff.get("removed") or diff.get("changed")
    )


def _collect_added_or_changed(diff: Mapping[str, Any]) -> Mapping[str, Any]:
    combined: dict[str, Any] = {}
    for key, value in diff.get("added", {}).items():
        combined[key] = value
    for key, payload in diff.get("changed", {}).items():
        combined[key] = payload.get("to")
    return combined


def _diff_scalar(base_value: Any, target_value: Any) -> Mapping[str, Any]:
    if base_value == target_value:
        return {"unchanged": deepcopy(base_value)}
    return {"from": deepcopy(base_value), "to": deepcopy(target_value)}


def _scalar_diff_has_changes(diff: Mapping[str, Any]) -> bool:
    return "unchanged" not in diff


def _handle_diff(
    base: str,
    target: str,
    *,
    sources: list[Mapping[str, Any]],
    core_metadata: Mapping[str, Any] | None,
    include_profiles: bool,
    cli_style: str,
    env_style: str,
    include_unchanged: bool,
    sections: Iterable[str] | None,
) -> tuple[Mapping[str, Any], bool]:
    base_name = base.strip().lower()
    target_name = target.strip().lower()

    base_metadata = risk_profile_metadata(base_name)
    target_metadata = risk_profile_metadata(target_name)

    base_cli = dict(get_metrics_service_overrides(base_name))
    target_cli = dict(get_metrics_service_overrides(target_name))
    base_cfg = dict(get_metrics_service_config_overrides(base_name))
    target_cfg = dict(get_metrics_service_config_overrides(target_name))
    base_env = dict(get_metrics_service_env_overrides(base_name))
    target_env = dict(get_metrics_service_env_overrides(target_name))

    cli_diff = _diff_mapping(base_cli, target_cli)
    cfg_diff = _diff_mapping(base_cfg, target_cfg)
    env_diff = _diff_mapping(base_env, target_env)

    max_counts_diff = _diff_mapping(
        dict(base_metadata.get("max_event_counts") or {}),
        dict(target_metadata.get("max_event_counts") or {}),
    )
    min_counts_diff = _diff_mapping(
        dict(base_metadata.get("min_event_counts") or {}),
        dict(target_metadata.get("min_event_counts") or {}),
    )

    selected_sections = {
        section.strip().lower() for section in sections or [] if section
    }

    severity_diff = _diff_scalar(
        base_metadata.get("severity_min"), target_metadata.get("severity_min")
    )
    extends_diff = _diff_scalar(
        base_metadata.get("extends"), target_metadata.get("extends")
    )
    extends_chain_diff = _diff_scalar(
        base_metadata.get("extends_chain"),
        target_metadata.get("extends_chain"),
    )
    expect_summary_diff = _diff_scalar(
        base_metadata.get("expect_summary_enabled"),
        target_metadata.get("expect_summary_enabled"),
    )
    require_screen_diff = _diff_scalar(
        base_metadata.get("require_screen_info"),
        target_metadata.get("require_screen_info"),
    )

    mapping_diffs = [
        cli_diff,
        cfg_diff,
        env_diff,
        max_counts_diff,
        min_counts_diff,
    ]
    has_changes = any(_diff_mapping_has_changes(item) for item in mapping_diffs) or any(
        _scalar_diff_has_changes(item)
        for item in (
            severity_diff,
            expect_summary_diff,
            require_screen_diff,
            extends_diff,
            extends_chain_diff,
        )
    )

    payload: dict[str, Any] = {
        "base": base_name,
        "target": target_name,
        "sources": sources,
        "diff": {
            "metrics_service_overrides": cli_diff,
            "metrics_service_config_overrides": cfg_diff,
            "metrics_service_env_overrides": env_diff,
            "max_event_counts": max_counts_diff,
            "min_event_counts": min_counts_diff,
            "severity_min": severity_diff,
            "extends": extends_diff,
            "extends_chain": extends_chain_diff,
            "expect_summary_enabled": expect_summary_diff,
            "require_screen_info": require_screen_diff,
        },
            "summary": {
                "base": summarize_risk_profile(base_metadata),
                "target": summarize_risk_profile(target_metadata),
            },
        "cli": {
            "base": _build_cli_flags(base_cli, style=cli_style),
            "target": _build_cli_flags(target_cli, style=cli_style),
            "added_or_changed": _build_cli_flags(
                _collect_added_or_changed(cli_diff), style=cli_style
            ),
            "removed": cli_diff["removed"],
        },
        "env": {
            "base": _build_env_assignments(base_env, style=env_style),
            "target": _build_env_assignments(target_env, style=env_style),
            "added_or_changed": _build_env_assignments(
                _collect_added_or_changed(env_diff), style=env_style
            ),
            "removed": env_diff["removed"],
            "format": env_style,
        },
    }

    if not include_unchanged:
        for section in (
            cli_diff,
            cfg_diff,
            env_diff,
            max_counts_diff,
            min_counts_diff,
        ):
            section.pop("unchanged", None)

        for scalar_key in (
            "severity_min",
            "extends",
            "extends_chain",
            "expect_summary_enabled",
            "require_screen_info",
        ):
            scalar_section = payload["diff"].get(scalar_key, {})
            if "unchanged" in scalar_section:
                payload["diff"][scalar_key] = {}

    include_profiles_section = include_profiles or (
        "profiles" in selected_sections if selected_sections else False
    )

    if include_profiles_section:
        payload["profiles"] = {"base": base_metadata, "target": target_metadata}

    if core_metadata:
        payload["core_config"] = dict(core_metadata)

    if selected_sections:
        allowed = set(selected_sections)
        allowed.update({"base", "target"})
        for key in list(payload.keys()):
            if key in {"base", "target"}:
                continue
            if key not in allowed:
                payload.pop(key, None)

    return payload, has_changes


def _handle_validate(
    required: Iterable[str] | None,
    *,
    sources: list[Mapping[str, Any]],
    core_metadata: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], int]:
    available = set(list_risk_profile_names())
    required_set = {item.strip().lower() for item in required or [] if item}
    missing = sorted(required_set - available)
    payload: dict[str, Any] = {
        "risk_profiles": sorted(available),
        "missing": missing,
        "sources": sources,
    }
    if core_metadata:
        payload["core_config"] = dict(core_metadata)
    if missing:
        return payload, 1
    return payload, 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        sources = _load_additional_profiles(
            args.risk_profiles_files, args.risk_profiles_dirs
        )
    except Exception as exc:  # noqa: BLE001 - błędy raportujemy operatorowi
        parser.error(str(exc))

    core_metadata: Mapping[str, Any] | None = None
    if getattr(args, "core_config", None):
        try:
            core_metadata = _load_core_metadata(args.core_config)
        except Exception as exc:  # noqa: BLE001 - przekazujemy błąd użytkownikowi
            parser.error(f"Nie udało się wczytać --core-config: {exc}")

    command = args.command
    output_path = getattr(args, "output", None)

    if command in {"list", "show", "validate"}:
        _resolve_yaml_json_format(parser, args, output_path=output_path)
    elif command == "render":
        inferred_format = _infer_format_from_output(output_path)
        if getattr(args, "format", None) is None:
            args.format = inferred_format or "json"
        elif (
            getattr(args, "format") in {"json", "yaml"}
            and inferred_format is not None
            and inferred_format != getattr(args, "format")
        ):
            parser.error(
                "Rozszerzenie pliku wyjściowego nie zgadza się z wymuszonym formatem"
            )
    elif command == "diff":
        inferred_format = _infer_format_from_output(output_path)
        if getattr(args, "format", None) is None:
            args.format = inferred_format or "json"
        elif (
            inferred_format is not None and inferred_format != getattr(args, "format")
        ):
            parser.error(
                "Rozszerzenie pliku wyjściowego nie zgadza się z wymuszonym formatem"
            )

    if command == "list":
        payload = _handle_list(
            verbose=bool(getattr(args, "verbose", False)),
            selected=getattr(args, "core_config", None)
            and core_metadata
            and core_metadata.get("metrics_service", {}).get("risk_profile"),
            sources=sources,
            core_metadata=core_metadata,
        )
        _dump_payload(
            payload,
            output=getattr(args, "output", None),
            fmt=str(getattr(args, "format", "json")),
        )
        return 0

    if command == "show":
        try:
            payload = _handle_show(
                args.name,
                sources=sources,
                core_metadata=core_metadata,
            )
        except (KeyError, ValueError) as exc:
            parser.error(str(exc))
        _dump_payload(
            payload,
            output=getattr(args, "output", None),
            fmt=str(getattr(args, "format", "json")),
        )
        return 0

    if command == "render":
        try:
            payload, cli_flags = _handle_render(
                args.name,
                sources=sources,
                core_metadata=core_metadata,
                fmt=args.format,
                include_profile=bool(getattr(args, "include_profile", False)),
                cli_style=str(getattr(args, "cli_style", "equals")),
                env_style=str(getattr(args, "env_style", "dotenv")),
                sections=getattr(args, "sections", None),
            )
        except (KeyError, ValueError) as exc:
            parser.error(str(exc))
        if cli_flags is not None:
            output_lines = "\n".join(cli_flags)
            if output_lines:
                output_lines += "\n"
            _write_text_output(output_lines, output=getattr(args, "output", None))
            return 0
        if args.format == "yaml":
            yaml_payload = payload or {}
            yaml_text = yaml.safe_dump(
                yaml_payload,
                allow_unicode=True,
                sort_keys=False,
            )
            _write_text_output(yaml_text, output=getattr(args, "output", None))
            return 0
        _dump_payload(payload or {}, output=getattr(args, "output", None))
        return 0

    if command == "diff":
        try:
            payload, has_changes = _handle_diff(
                args.base,
                args.target,
                sources=sources,
                core_metadata=core_metadata,
                include_profiles=bool(getattr(args, "include_profiles", False)),
                cli_style=str(getattr(args, "cli_style", "equals")),
                env_style=str(getattr(args, "env_style", "dotenv")),
                include_unchanged=not bool(
                    getattr(args, "hide_unchanged", False)
                ),
                sections=getattr(args, "sections", None),
            )
        except (KeyError, ValueError) as exc:
            parser.error(str(exc))
        if args.format == "yaml":
            yaml_text = yaml.safe_dump(
                payload,
                allow_unicode=True,
                sort_keys=False,
            )
            _write_text_output(yaml_text, output=getattr(args, "output", None))
        else:
            _dump_payload(payload, output=getattr(args, "output", None))
        if getattr(args, "fail_on_diff", False) and has_changes:
            return 1
        return 0

    if command == "validate":
        payload, exit_code = _handle_validate(
            getattr(args, "required_profiles", None),
            sources=sources,
            core_metadata=core_metadata,
        )
        _dump_payload(
            payload,
            output=getattr(args, "output", None),
            fmt=str(getattr(args, "format", "json")),
        )
        return exit_code

    parser.error(f"Nieobsługiwane polecenie: {command}")
    return 2


if __name__ == "__main__":  # pragma: no cover - obsługa uruchomień z CLI
    raise SystemExit(main())
