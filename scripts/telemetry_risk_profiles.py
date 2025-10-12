"""Kompatybilna warstwa eksportująca presety profili ryzyka oraz proste CLI.

- Jeśli dostępny jest moduł `bot_core.runtime.telemetry_risk_profiles`, re-eksportujemy jego API.
- W przeciwnym razie używamy lokalnych presetów i minimalnych implementacji (fallback), tak aby
  skrypty mogły korzystać m.in. z:
    * get_metrics_service_overrides(profile_name)
    * get_metrics_service_config_overrides(profile_name)
    * get_metrics_service_env_overrides(profile_name)
    * list_risk_profile_names()
    * load_risk_profiles_with_metadata(path, origin_label=...)
    * risk_profile_metadata(name)
    * summarize_risk_profile(metadata)
  oraz pokrewnych funkcji pomocniczych.

Moduł może być współdzielony przez watcher (`watch_metrics_stream`) i weryfikator
logów decyzji (`verify_decision_log`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Tuple

# PyYAML jest opcjonalny – używamy go tylko gdy użytkownik wybierze format YAML
try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover - środowisko bez PyYAML
    yaml = None  # type: ignore

# Warstwa core-config jest opcjonalna (starsze gałęzie mogą jej nie mieć)
try:  # pragma: no cover - moduł konfiguracyjny może być niedostępny
    from bot_core.config.loader import load_core_config  # type: ignore
except Exception:  # pragma: no cover - środowisko bez warstwy konfiguracji
    load_core_config = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# PRÓBA UŻYCIA PEŁNEGO API Z bot_core (preferowane)
try:  # pragma: no cover
    from bot_core.runtime.telemetry_risk_profiles import (  # type: ignore
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
        reset_risk_profile_store,
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
        "list_risk_profile_names",
        "load_risk_profiles_from_file",
        "load_risk_profiles_with_metadata",
        "register_risk_profiles",
        "reset_risk_profile_store",
        "risk_profile_metadata",
        "summarize_risk_profile",
    ]

    _FALLBACK = False

except Exception:  # pragma: no cover - fallback lokalny
    _FALLBACK = True

    # --- PRESETY LOKALNE -----------------------------------------------------
    _RISK_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
        "conservative": {
            "expect_summary_enabled": True,
            "require_screen_info": True,
            "severity_min": "warning",
            "max_event_counts": {
                "overlay_budget": 0,
                "jank": 0,
                "reduce_motion": 3,
            },
            "min_event_counts": {"reduce_motion": 1},
            "metrics_service_overrides": {
                "ui_alerts_reduce_mode": "enable",
                "ui_alerts_overlay_mode": "enable",
                "ui_alerts_jank_mode": "enable",
                "ui_alerts_reduce_active_severity": "critical",
                "ui_alerts_reduce_recovered_severity": "notice",
                "ui_alerts_overlay_exceeded_severity": "critical",
                "ui_alerts_overlay_recovered_severity": "notice",
                "ui_alerts_overlay_critical_severity": "critical",
                "ui_alerts_overlay_critical_threshold": 1,
                "ui_alerts_jank_spike_severity": "warning",
                "ui_alerts_jank_critical_severity": "error",
                "ui_alerts_jank_critical_over_ms": 12.0,
            },
        },
        "balanced": {
            "expect_summary_enabled": True,
            "require_screen_info": True,
            "severity_min": "notice",
            "max_event_counts": {
                "overlay_budget": 2,
                "jank": 1,
                "reduce_motion": 5,
            },
            "metrics_service_overrides": {
                "ui_alerts_reduce_mode": "enable",
                "ui_alerts_overlay_mode": "enable",
                "ui_alerts_jank_mode": "enable",
                "ui_alerts_reduce_active_severity": "warning",
                "ui_alerts_reduce_recovered_severity": "info",
                "ui_alerts_overlay_exceeded_severity": "warning",
                "ui_alerts_overlay_recovered_severity": "info",
                "ui_alerts_overlay_critical_severity": "error",
                "ui_alerts_overlay_critical_threshold": 2,
                "ui_alerts_jank_spike_severity": "notice",
                "ui_alerts_jank_critical_severity": "warning",
                "ui_alerts_jank_critical_over_ms": 18.0,
            },
        },
        "aggressive": {
            "expect_summary_enabled": True,
            "require_screen_info": True,
            "severity_min": "info",
            "max_event_counts": {
                "overlay_budget": 4,
                "jank": 2,
                "reduce_motion": 8,
            },
            "metrics_service_overrides": {
                "ui_alerts_reduce_mode": "enable",
                "ui_alerts_overlay_mode": "enable",
                "ui_alerts_jank_mode": "enable",
                "ui_alerts_reduce_active_severity": "info",
                "ui_alerts_reduce_recovered_severity": "info",
                "ui_alerts_overlay_exceeded_severity": "info",
                "ui_alerts_overlay_recovered_severity": "info",
                "ui_alerts_overlay_critical_severity": "warning",
                "ui_alerts_overlay_critical_threshold": 3,
                "ui_alerts_jank_spike_severity": "info",
                "ui_alerts_jank_critical_severity": "warning",
                "ui_alerts_jank_critical_over_ms": 22.0,
            },
        },
        "manual": {},
    }

    _PROFILE_STORE: dict[str, dict[str, Any]] = {}
    _PROFILE_ORIGINS: dict[str, str] = {}
    _REGISTERED_SOURCES: list[str] = []
    _SUPPORTED_SUFFIXES = {".json", ".yaml", ".yml"}

    _METRICS_CLI_TO_ENV: Mapping[str, str] = {
        "ui_alerts_reduce_mode": "RUN_METRICS_SERVICE_UI_ALERTS_REDUCE_MODE",
        "ui_alerts_overlay_mode": "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_MODE",
        "ui_alerts_jank_mode": "RUN_METRICS_SERVICE_UI_ALERTS_JANK_MODE",
        "ui_alerts_reduce_category": "RUN_METRICS_SERVICE_UI_ALERTS_REDUCE_CATEGORY",
        "ui_alerts_reduce_active_severity": "RUN_METRICS_SERVICE_UI_ALERTS_REDUCE_ACTIVE_SEVERITY",
        "ui_alerts_reduce_recovered_severity": "RUN_METRICS_SERVICE_UI_ALERTS_REDUCE_RECOVERED_SEVERITY",
        "ui_alerts_overlay_category": "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_CATEGORY",
        "ui_alerts_overlay_exceeded_severity": "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_EXCEEDED_SEVERITY",
        "ui_alerts_overlay_recovered_severity": "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_RECOVERED_SEVERITY",
        "ui_alerts_overlay_critical_severity": "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_CRITICAL_SEVERITY",
        "ui_alerts_overlay_critical_threshold": "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_CRITICAL_THRESHOLD",
        "ui_alerts_jank_category": "RUN_METRICS_SERVICE_UI_ALERTS_JANK_CATEGORY",
        "ui_alerts_jank_spike_severity": "RUN_METRICS_SERVICE_UI_ALERTS_JANK_SPIKE_SEVERITY",
        "ui_alerts_jank_critical_severity": "RUN_METRICS_SERVICE_UI_ALERTS_JANK_CRITICAL_SEVERITY",
        "ui_alerts_jank_critical_over_ms": "RUN_METRICS_SERVICE_UI_ALERTS_JANK_CRITICAL_OVER_MS",
        "ui_alerts_audit_dir": "RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_DIR",
        "ui_alerts_audit_backend": "RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_BACKEND",
        "ui_alerts_audit_pattern": "RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_PATTERN",
        "ui_alerts_audit_retention_days": "RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_RETENTION_DAYS",
        "ui_alerts_audit_fsync": "RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_FSYNC",
    }

    def _initialize_store() -> None:
        if _PROFILE_STORE:
            return
        for name, data in _RISK_PROFILE_PRESETS.items():
            normalized = name.strip().lower()
            _PROFILE_STORE[normalized] = deepcopy(data)
            _PROFILE_ORIGINS[normalized] = "builtin"

    def reset_risk_profile_store() -> None:
        _PROFILE_STORE.clear()
        _PROFILE_ORIGINS.clear()
        _REGISTERED_SOURCES.clear()
        _initialize_store()

    def list_risk_profile_names() -> list[str]:
        _initialize_store()
        return sorted(_PROFILE_STORE)

    def get_risk_profile(name: str) -> Mapping[str, Any]:
        _initialize_store()
        normalized = name.strip().lower()
        try:
            preset = _PROFILE_STORE[normalized]
        except KeyError as exc:
            raise KeyError(f"Nieznany profil ryzyka: {name!r}") from exc
        return deepcopy(preset)

    def risk_profile_metadata(name: str) -> dict[str, Any]:
        profile = get_risk_profile(name)
        metadata = dict(profile)
        normalized = name.strip().lower()
        metadata["name"] = normalized
        origin = _PROFILE_ORIGINS.get(normalized)
        if origin:
            metadata.setdefault("origin", origin)
        return metadata

    def _merge_profile_dicts(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
        result = deepcopy(base)
        for key, value in overrides.items():
            if key == "extends":
                continue
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = _merge_profile_dicts(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    def register_risk_profiles(
        profiles: Mapping[str, Mapping[str, Any]]
        | Iterable[tuple[str, Mapping[str, Any]]]
        | None,
        *,
        origin: str = "external",
    ) -> list[str]:
        if not profiles:
            return []

        _initialize_store()

        items = profiles.items() if isinstance(profiles, Mapping) else list(profiles)
        normalized_profiles: dict[str, dict[str, Any]] = {}
        for name, cfg in items:
            normalized = str(name).strip().lower()
            if not normalized:
                continue
            if not isinstance(cfg, Mapping):
                raise ValueError(
                    f"Profil ryzyka '{name}' musi być mapą, otrzymano {type(cfg)!r}"
                )
            normalized_profiles[normalized] = deepcopy(dict(cfg))

        resolved: dict[str, dict[str, Any]] = {}
        visiting: set[str] = set()

        def resolve(target: str) -> dict[str, Any]:
            if target in resolved:
                return resolved[target]
            if target in visiting:
                raise ValueError(
                    f"Wykryto cykliczne dziedziczenie profili ryzyka przy '{target}'"
                )
            try:
                entry = normalized_profiles[target]
            except KeyError as exc:
                raise KeyError(f"Nieznany profil do zarejestrowania: {target}") from exc

            visiting.add(target)
            extends_raw = entry.get("extends")
            if extends_raw:
                base_name = str(extends_raw).strip().lower()
                if not base_name:
                    raise ValueError(
                        f"Profil ryzyka '{target}' posiada nieprawidłowe pole extends"
                    )
                if base_name == target:
                    raise ValueError(
                        f"Profil ryzyka '{target}' nie może dziedziczyć z samego siebie"
                    )
                if base_name in normalized_profiles:
                    base_profile = resolve(base_name)
                else:
                    try:
                        base_profile = get_risk_profile(base_name)
                    except KeyError as exc:
                        raise ValueError(
                            f"Profil ryzyka '{target}' dziedziczy z nieznanego profilu '{extends_raw}'"
                        ) from exc
                merged = _merge_profile_dicts(base_profile, entry)
                chain: list[str] = []
                base_chain = base_profile.get("extends_chain")
                if isinstance(base_chain, list):
                    chain.extend(base_chain)
                elif base_profile.get("extends"):
                    chain.append(str(base_profile["extends"]))
                chain.append(base_name)
                merged["extends"] = base_name
                if chain:
                    merged["extends_chain"] = chain
            else:
                base_profile = _PROFILE_STORE.get(target)
                if base_profile:
                    merged = _merge_profile_dicts(base_profile, entry)
                    if "extends" not in merged and base_profile.get("extends"):
                        merged["extends"] = base_profile.get("extends")
                    base_chain = base_profile.get("extends_chain")
                    if (
                        base_chain
                        and "extends_chain" not in merged
                        and isinstance(base_chain, list)
                    ):
                        merged["extends_chain"] = list(base_chain)
                else:
                    merged = deepcopy(entry)
                    if "extends_chain" in merged and not isinstance(
                        merged.get("extends_chain"), list
                    ):
                        merged["extends_chain"] = list(merged.get("extends_chain") or [])

            visiting.remove(target)
            resolved[target] = merged
            return merged

        registered: list[str] = []
        for normalized in normalized_profiles:
            profile_data = resolve(normalized)
            _PROFILE_STORE[normalized] = deepcopy(profile_data)
            _PROFILE_ORIGINS[normalized] = origin
            registered.append(normalized)
        return registered

    def list_risk_profile_files(directory: Path) -> list[Path]:
        files = [
            entry
            for entry in directory.iterdir()
            if entry.is_file() and entry.suffix.lower() in _SUPPORTED_SUFFIXES
        ]
        files.sort()
        return files

    def _load_risk_profiles_from_single_file(
        source: Path, *, origin: str | None = None
    ) -> list[str]:
        if not source.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku profili ryzyka: {source}")
        if source.is_dir():
            raise IsADirectoryError(
                f"Ścieżka {source} wskazuje katalog – użyj load_risk_profiles_from_file do obsługi katalogów"
            )

        text = source.read_text(encoding="utf-8")
        suffix = source.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError(
                    "Do wczytania profili w YAML wymagany jest PyYAML (pip install pyyaml)."
                )
            data: Any = yaml.safe_load(text) or {}
        else:
            data = json.loads(text or "{}")

        if isinstance(data, Mapping) and "risk_profiles" in data:
            data = data["risk_profiles"]
        if not isinstance(data, Mapping):
            raise ValueError("Plik profili ryzyka musi zawierać mapę risk_profiles")

        origin_label = f"file:{source}" if origin is None else origin
        registered = register_risk_profiles(data, origin=origin_label)
        _REGISTERED_SOURCES.append(str(source))
        return registered

    def load_risk_profiles_from_file(
        path: str | Path, *, origin: str | None = None
    ) -> list[str]:
        target = Path(path).expanduser()
        if target.is_dir():
            files = list_risk_profile_files(target)
            if not files:
                raise ValueError(
                    f"Katalog {target} nie zawiera żadnych plików JSON/YAML z profilami ryzyka"
                )
            registered: list[str] = []
            base_origin = origin or f"dir:{target}"
            for entry in files:
                entry_origin = f"{base_origin}#{entry.name}" if base_origin else None
                registered.extend(
                    _load_risk_profiles_from_single_file(entry, origin=entry_origin)
                )
            return registered

        return _load_risk_profiles_from_single_file(target, origin=origin)

    def load_risk_profiles_with_metadata(
        path: str | Path, *, origin_label: str | None = None
    ) -> Tuple[list[str], Mapping[str, Any]]:
        target = Path(path).expanduser()
        if target.is_dir():
            files = list_risk_profile_files(target)
            if not files:
                raise ValueError(
                    f"Katalog {target} nie zawiera żadnych plików JSON/YAML z profilami ryzyka"
                )
            registered: list[str] = []
            files_meta: list[dict[str, Any]] = []
            base_origin = origin_label or f"dir:{target}"
            for entry in files:
                entry_origin = f"{base_origin}#{entry.name}" if base_origin else None
                entry_registered = _load_risk_profiles_from_single_file(
                    entry, origin=entry_origin
                )
                files_meta.append(
                    {
                        "path": str(entry),
                        "registered_profiles": list(entry_registered),
                    }
                )
                registered.extend(entry_registered)

            metadata: dict[str, Any] = {
                "path": str(target),
                "type": "directory",
                "files": files_meta,
                "registered_profiles": list(registered),
            }
            if origin_label:
                metadata["origin"] = origin_label
            return registered, metadata

        registered = _load_risk_profiles_from_single_file(
            target, origin=origin_label or f"file:{target}"
        )
        metadata = {
            "path": str(target),
            "type": "file",
            "registered_profiles": list(registered),
        }
        if origin_label:
            metadata["origin"] = origin_label
        return registered, metadata

    def _severity_for(profile: Mapping[str, Any]) -> str | None:
        value = profile.get("severity_min")
        return str(value).lower() if value is not None else None

    def get_metrics_service_overrides(profile_name: str) -> dict[str, Any]:
        profile = get_risk_profile(profile_name)
        overrides = dict(profile.get("metrics_service_overrides", {}))
        if overrides:
            return overrides

        severity = _severity_for(profile) or "warning"
        max_counts = dict(profile.get("max_event_counts", {}))
        overlay_raw = max_counts.get("overlay_budget", 0)
        overlay_threshold = int(overlay_raw) if isinstance(overlay_raw, (int, float)) else 0
        dynamic: dict[str, Any] = {
            "ui_alerts_reduce_mode": "enable",
            "ui_alerts_overlay_mode": "enable",
            "ui_alerts_jank_mode": "enable",
            "ui_alerts_reduce_active_severity": severity,
            "ui_alerts_overlay_exceeded_severity": severity,
            "ui_alerts_jank_spike_severity": severity,
            "ui_alerts_overlay_critical_threshold": max(0, overlay_threshold),
        }
        jank_cap = max_counts.get("jank")
        if isinstance(jank_cap, (int, float)) and jank_cap <= 0:
            dynamic["ui_alerts_jank_critical_severity"] = "critical"
        return dynamic

    def get_metrics_service_config_overrides(profile_name: str) -> dict[str, Any]:
        return get_metrics_service_overrides(profile_name)

    def get_metrics_service_env_overrides(profile_name: str) -> dict[str, Any]:
        overrides = get_metrics_service_overrides(profile_name)
        env: dict[str, Any] = {}
        for option, value in overrides.items():
            env_name = _METRICS_CLI_TO_ENV.get(option)
            if env_name is None:
                continue
            env[env_name] = value
        return env

    def summarize_risk_profile(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        summary: dict[str, Any] = {}

        for key in (
            "name",
            "origin",
            "extends",
            "severity_min",
            "expect_summary_enabled",
            "require_screen_info",
        ):
            value = metadata.get(key)
            if value is not None:
                summary[key] = deepcopy(value)

        extends_chain = metadata.get("extends_chain")
        if extends_chain is None:
            if metadata.get("extends"):
                extends_chain = [metadata.get("extends")]  # type: ignore[list-item]
            else:
                extends_chain = []
        summary["extends_chain"] = deepcopy(extends_chain)

        for section in ("max_event_counts", "min_event_counts"):
            value = metadata.get(section)
            if value:
                summary[section] = deepcopy(value)

        name = metadata.get("name")
        if name:
            try:
                summary["recommended_overrides"] = dict(
                    get_metrics_service_overrides(str(name))
                )
            except Exception:
                pass
        return summary

    class MetricsRiskProfileResolver:
        """Minimalny resolver profili ryzyka (fallback)."""

        def __init__(self) -> None:
            self._names = set(list_risk_profile_names())

        def names(self) -> list[str]:
            return list_risk_profile_names()

        def resolve_overrides(self, name: str) -> dict[str, Any]:
            return get_metrics_service_overrides(name)

        def register_file(self, path: Path | str) -> list[str]:
            regs, _meta = load_risk_profiles_with_metadata(path, origin_label=f"resolver:{path}")
            self._names.update(regs)
            return regs

    __all__ = [
        "MetricsRiskProfileResolver",
        "get_metrics_service_config_overrides",
        "get_metrics_service_env_overrides",
        "get_metrics_service_overrides",
        "get_risk_profile",
        "list_risk_profile_files",
        "list_risk_profile_names",
        "load_risk_profiles_from_file",
        "load_risk_profiles_with_metadata",
        "register_risk_profiles",
        "reset_risk_profile_store",
        "risk_profile_metadata",
        "summarize_risk_profile",
    ]

# ---------------------------------------------------------------------------
# CLI – wspólne dla trybu pełnego i fallback

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
        "risk_profiles_file": getattr(metrics_cfg, "ui_alerts_risk_profiles_file", None),
    }

    tls_cfg = getattr(metrics_cfg, "tls", None)
    if tls_cfg is not None:
        metrics_meta["tls_enabled"] = bool(getattr(tls_cfg, "enabled", False))
        metrics_meta["client_auth"] = bool(getattr(tls_cfg, "require_client_auth", False))
        metrics_meta["client_cert_configured"] = bool(getattr(tls_cfg, "certificate_path", None))
        metrics_meta["client_key_configured"] = bool(getattr(tls_cfg, "private_key_path", None))
        metrics_meta["root_cert_configured"] = bool(getattr(tls_cfg, "client_ca_path", None))

    if getattr(metrics_cfg, "auth_token", None):
        metrics_meta["auth_token_configured"] = True

    metadata["metrics_service"] = {key: value for key, value in metrics_meta.items() if value not in (None, "")}
    return metadata


def _load_additional_profiles(
    files: Iterable[str] | None,
    directories: Iterable[str] | None,
) -> list[Mapping[str, Any]]:
    metadata_entries: list[Mapping[str, Any]] = []
    for raw in files or []:
        _, metadata = load_risk_profiles_with_metadata(raw, origin_label="cli:file")
        metadata_entries.append(metadata)
    for raw in directories or []:
        _, metadata = load_risk_profiles_with_metadata(raw, origin_label="cli:dir")
        metadata_entries.append(metadata)
    return metadata_entries


def _write_text_output(payload: str, *, output: str | None) -> None:
    if output:
        target = Path(output).expanduser()
        target.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
        if not payload.endswith("\n"):
            sys.stdout.write("\n")


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
            parser.error("Rozszerzenie pliku wyjściowego nie zgadza się z wymuszonym formatem")
        final = explicit

    if final not in allowed:
        parser.error("Format musi być jednym z: json, yaml")

    setattr(args, "format", final)
    return final


def _dump_payload(payload: Mapping[str, Any], *, output: str | None, fmt: str = "json") -> None:
    if fmt not in {"json", "yaml"}:
        raise ValueError(f"Nieobsługiwany format serializacji: {fmt}")

    if fmt == "yaml":
        if yaml is None:
            raise RuntimeError("Wymagany PyYAML (pip install pyyaml), aby użyć formatu YAML")
        rendered = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    else:
        rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    _write_text_output(rendered, output=output)


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
            "Ścieżka wyjściowa dla raportu (domyślnie STDOUT); format może być wywnioskowany z rozszerzenia pliku"
        ),
    )


# ---------------------------------------------------------------------------
# Budowa parsera i podkomend

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="telemetry_risk_profiles",
        description=("Zarządzanie presetami profili ryzyka telemetryjnego oraz audyt konfiguracji"),
        conflict_handler="resolve",
    )
    _add_shared_arguments(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Wyświetl listę dostępnych profili ryzyka")
    _add_shared_arguments(list_parser)
    list_parser.add_argument("--verbose", action="store_true", help="Dołącz szczegóły każdego profilu")
    list_parser.add_argument(
        "--format",
        choices=("json", "yaml"),
        help=(
            "Format wyjścia raportu (json lub yaml). Domyślnie json lub zgodnie z rozszerzeniem pliku wyjściowego"
        ),
    )

    show_parser = subparsers.add_parser("show", help="Wyświetl szczegóły wybranego profilu")
    _add_shared_arguments(show_parser)
    show_parser.add_argument("name", help="Nazwa profilu ryzyka do wyświetlenia")
    show_parser.add_argument(
        "--format",
        choices=("json", "yaml"),
        help=(
            "Format wyjścia raportu (json lub yaml). Domyślnie json lub zgodnie z rozszerzeniem pliku wyjściowego"
        ),
    )

    render_parser = subparsers.add_parser(
        "render",
        help=("Wygeneruj nadpisania konfiguracji MetricsService dla wskazanego profilu"),
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
        help=("Dołącz pełną definicję profilu do wyniku JSON/YAML (niedostępne dla formatów CLI/env)"),
    )
    render_parser.add_argument(
        "--cli-style",
        choices=("equals", "space"),
        default="equals",
        help=("Sposób formatowania wartości flag CLI (equals: --key=value, space: --key value)"),
    )
    render_parser.add_argument(
        "--env-style",
        choices=("dotenv", "export"),
        default="dotenv",
        help=("Sposób formatowania zmiennych środowiskowych (dotenv: KEY=value, export: export KEY=value)"),
    )
    render_parser.add_argument(
        "--section",
        dest="sections",
        action="append",
        choices=RENDER_SECTION_CHOICES,
        metavar="NAME",
        help=(
            "Ogranicz wynik JSON/YAML do wskazanych sekcji (można podać wielokrotnie). "
            "Dostępne: "
            + ", ".join(RENDER_SECTION_CHOICES)
        ),
    )

    bundle_parser = subparsers.add_parser(
        "bundle",
        help=(
            "Wygeneruj pakiet szablonów .env/.yaml dla etapów demo→paper→live na podstawie presetów profili ryzyka"
        ),
    )
    _add_shared_arguments(bundle_parser)
    bundle_parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        metavar="NAME=PROFILE",
        help=(
            "Przypisz profil ryzyka do etapu (np. --stage demo=conservative). "
            "Domyślnie generowane są etapy demo/paper/live zgodnie z mapowaniem konserwatywny/zbalansowany/manualny"
        ),
    )
    bundle_parser.add_argument(
        "--output-dir",
        required=True,
        help="Katalog docelowy, w którym zostaną zapisane szablony i manifest",
    )
    bundle_parser.add_argument(
        "--env-style",
        choices=("dotenv", "export"),
        default="dotenv",
        help="Styl formatowania plików środowiskowych (dotenv lub export)",
    )
    bundle_parser.add_argument(
        "--config-format",
        choices=("yaml", "json"),
        default="yaml",
        help="Format pliku konfiguracyjnego MetricsService (yaml lub json)",
    )

    diff_parser = subparsers.add_parser("diff", help="Porównaj dwa profile ryzyka i wypisz różnice w nadpisaniach")
    _add_shared_arguments(diff_parser)
    diff_parser.add_argument("base", help="Nazwa profilu bazowego")
    diff_parser.add_argument("target", help="Profil, z którym porównujemy")
    diff_parser.add_argument(
        "--format",
        choices=("json", "yaml"),
        help=(
            "Format raportu różnic (json lub yaml). Domyślnie json lub zgodnie z rozszerzeniem pliku wyjściowego"
        ),
    )
    diff_parser.add_argument("--include-profiles", action="store_true", help="Dołącz pełne definicje profili do wyniku JSON/YAML")
    diff_parser.add_argument("--hide-unchanged", action="store_true", help="Ukryj sekcje niezmienione, aby uprościć raport")
    diff_parser.add_argument(
        "--section",
        dest="sections",
        action="append",
        choices=("diff", "summary", "cli", "env", "profiles", "core_config", "sources"),
        metavar="NAME",
        help=(
            "Ogranicz wynik do wskazanych sekcji (można podać wielokrotnie). "
            "Domyślnie raport zawiera wszystkie sekcje."
        ),
    )
    diff_parser.add_argument(
        "--fail-on-diff",
        action="store_true",
        help=("Zakończ działanie kodem 1, jeżeli wykryto różnice pomiędzy profilami"),
    )
    diff_parser.add_argument(
        "--cli-style",
        choices=("equals", "space"),
        default="equals",
        help=("Format flag CLI w sekcji porównania (equals: --key=value, space: --key value)"),
    )
    diff_parser.add_argument(
        "--env-style",
        choices=("dotenv", "export"),
        default="dotenv",
        help=("Format przypisań środowiskowych w sekcji porównania (dotenv lub export)"),
    )

    validate_parser = subparsers.add_parser("validate", help="Zweryfikuj dostępność zadanych profili")
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
        help=("Format raportu (json lub yaml). Domyślnie json lub zgodnie z rozszerzeniem pliku"),
    )

    return parser


# ---------------------------------------------------------------------------
# Pomocnicze: rendering CLI/env i sekcji

def _build_cli_flags(overrides: Mapping[str, Any], *, style: str = "equals") -> list[str]:
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
    escaped = value.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace('"', '\\"')
    return f'"{escaped}"'


def _build_env_assignments(overrides: Mapping[str, Any], *, style: str = "dotenv") -> list[str]:
    assignments: list[str] = []
    for env_name, raw_value in sorted(overrides.items()):
        value_text = _format_env_value(raw_value)
        if style == "dotenv":
            value_repr = _quote_for_dotenv(value_text)
            assignments.append(f"{env_name}={value_repr}")
        elif style == "export":
            value_repr = shlex.quote(value_text)
            assignments.append(f"export {env_name}={value_repr}")
        else:  # pragma: no cover - zabezpieczenie
            raise ValueError(f"Nieobsługiwany styl env: {style}")
    return assignments


DEFAULT_BUNDLE_STAGE_MAP: Mapping[str, str] = {
    "demo": "conservative",
    "paper": "balanced",
    "live": "manual",
}


def _parse_stage_mapping(stage_args: Iterable[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in stage_args or []:
        if "=" not in entry:
            raise ValueError("Opcja --stage wymaga formatu etap=profil (np. --stage demo=conservative)")
        stage, profile = entry.split("=", 1)
        normalized_stage = stage.strip().lower()
        normalized_profile = profile.strip().lower()
        if not normalized_stage or not normalized_profile:
            raise ValueError("Opcja --stage wymaga niepustych nazw etapu oraz profilu")
        mapping[normalized_stage] = normalized_profile
    final_mapping: dict[str, str] = dict(DEFAULT_BUNDLE_STAGE_MAP)
    if mapping:
        for stage, profile in mapping.items():
            final_mapping[stage] = profile
    return final_mapping


def _handle_bundle(
    *,
    stage_mapping: Mapping[str, str],
    output_dir: str,
    env_style: str,
    config_format: str,
    sources: list[Mapping[str, Any]],
    core_metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if config_format not in {"yaml", "json"}:
        raise ValueError("Nieobsługiwany format konfiguracji (dozwolone: yaml, json)")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(target_dir),
        "env_style": env_style,
        "config_format": config_format,
        "stages": [],
    }
    if sources:
        manifest["sources"] = [dict(item) for item in sources]
    if core_metadata:
        manifest["core_config"] = dict(core_metadata)

    for stage, profile in stage_mapping.items():
        metadata = risk_profile_metadata(profile)
        summary = summarize_risk_profile(metadata)
        env_overrides = dict(get_metrics_service_env_overrides(profile))
        config_overrides = dict(get_metrics_service_config_overrides(profile))
        env_lines = _build_env_assignments(env_overrides, style=env_style)

        stage_dir = target_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        env_filename = "metrics.env" if env_style == "dotenv" else "metrics.env.sh"
        env_path = stage_dir / env_filename
        summary_dump = json.dumps(summary, ensure_ascii=False, sort_keys=True)
        header_lines = [
            "# telemetry risk profile bundle",
            f"# stage: {stage}",
            f"# risk_profile: {profile}",
            "# risk_profile_summary:",
            f"#   {summary_dump}",
            "",
        ]
        env_content = "\n".join(header_lines + env_lines)
        if env_content and not env_content.endswith("\n"):
            env_content += "\n"
        env_path.write_text(env_content, encoding="utf-8")

        config_filename = "metrics.yaml" if config_format == "yaml" else "metrics.json"
        config_path = stage_dir / config_filename
        config_payload: dict[str, Any] = {
            "stage": stage,
            "risk_profile": profile,
            "risk_profile_summary": summary,
            "metrics_service": {
                "env_overrides": env_overrides,
                "config_overrides": config_overrides,
            },
        }
        if core_metadata:
            config_payload["core_config"] = dict(core_metadata)

        if config_format == "yaml":
            if yaml is None:
                raise RuntimeError("Wymagany PyYAML (pip install pyyaml), aby użyć formatu YAML")
            config_text = yaml.safe_dump(config_payload, allow_unicode=True, sort_keys=False)
        else:
            config_text = json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n"
        config_path.write_text(config_text, encoding="utf-8")

        manifest["stages"].append(
            {
                "stage": stage,
                "risk_profile": profile,
                "risk_profile_summary": summary,
                "paths": {"env": str(env_path), "config": str(config_path)},
            }
        )

    manifest_path = target_dir / "manifest.json"
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


# ---------------------------------------------------------------------------
# Handlery podkomend

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
        payload["profiles"] = {name: risk_profile_metadata(name) for name in list_risk_profile_names()}
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


def _collect_added_or_changed(diff: Mapping[str, Any]) -> Mapping[str, Any]:
    combined: dict[str, Any] = {}
    for key, value in diff.get("added", {}).items():
        combined[key] = value
    for key, payload in diff.get("changed", {}).items():
        combined[key] = payload.get("to")
    return combined


def _diff_mapping(base_map: Mapping[str, Any], target_map: Mapping[str, Any]) -> dict[str, Any]:
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

    return {"added": added, "removed": removed, "changed": changed, "unchanged": unchanged}


def _diff_mapping_has_changes(diff: Mapping[str, Any]) -> bool:
    return bool(diff.get("added") or diff.get("removed") or diff.get("changed"))


def _diff_scalar(base_value: Any, target_value: Any) -> Mapping[str, Any]:
    if base_value == target_value:
        return {"unchanged": deepcopy(base_value)}
    return {"from": deepcopy(base_value), "to": deepcopy(target_value)}


def _scalar_diff_has_changes(diff: Mapping[str, Any]) -> bool:
    return "unchanged" not in diff


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
        raise ValueError("Opcja --include-profile jest dostępna wyłącznie dla formatu json/yaml")

    if fmt in {"cli", "env"} and selected_sections:
        raise ValueError("Opcja --section jest dostępna wyłącznie dla formatów json oraz yaml")

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
                if "env_assignments_format" in payload and "env_assignments_format" not in selected_sections:
                    filtered["env_assignments_format"] = payload["env_assignments_format"]
            elif key in payload:
                filtered[key] = payload[key]
        payload = filtered
    return payload, None


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

    selected_sections = {section.strip().lower() for section in sections or [] if section}

    severity_diff = _diff_scalar(base_metadata.get("severity_min"), target_metadata.get("severity_min"))
    extends_diff = _diff_scalar(base_metadata.get("extends"), target_metadata.get("extends"))
    extends_chain_diff = _diff_scalar(base_metadata.get("extends_chain"), target_metadata.get("extends_chain"))
    expect_summary_diff = _diff_scalar(
        base_metadata.get("expect_summary_enabled"), target_metadata.get("expect_summary_enabled")
    )
    require_screen_diff = _diff_scalar(
        base_metadata.get("require_screen_info"), target_metadata.get("require_screen_info")
    )

    mapping_diffs = [cli_diff, cfg_diff, env_diff, max_counts_diff, min_counts_diff]
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
            "added_or_changed": _build_cli_flags(_collect_added_or_changed(cli_diff), style=cli_style),
            "removed": cli_diff["removed"],
        },
        "env": {
            "base": _build_env_assignments(base_env, style=env_style),
            "target": _build_env_assignments(target_env, style=env_style),
            "added_or_changed": _build_env_assignments(_collect_added_or_changed(env_diff), style=env_style),
            "removed": env_diff["removed"],
            "format": env_style,
        },
    }

    if not include_unchanged:
        for section in (cli_diff, cfg_diff, env_diff, max_counts_diff, min_counts_diff):
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

    include_profiles_section = include_profiles or ("profiles" in selected_sections if selected_sections else False)
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


# ---------------------------------------------------------------------------
# Główne wejście CLI

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Upewnij się, że każda sesja CLI startuje od czystego zestawu builtinów.
    reset_risk_profile_store()

    # Zarejestruj dodatkowe profile z plików/katalogów
    try:
        sources = _load_additional_profiles(args.risk_profiles_files, args.risk_profiles_dirs)
    except Exception as exc:  # noqa: BLE001
        parser.error(str(exc))

    core_metadata: Mapping[str, Any] | None = None
    if getattr(args, "core_config", None):
        try:
            core_metadata = _load_core_metadata(args.core_config)
        except Exception as exc:  # noqa: BLE001
            parser.error(f"Nie udało się wczytać --core-config: {exc}")

    command = args.command
    output_path = getattr(args, "output", None)

    if command in {"list", "show", "validate"}:
        _resolve_yaml_json_format(parser, args, output_path=output_path)
    elif command == "render":
        inferred_format = _infer_format_from_output(output_path)
        if getattr(args, "format", None) is None:
            args.format = inferred_format or "json"
        elif getattr(args, "format") in {"json", "yaml"} and inferred_format is not None and inferred_format != getattr(args, "format"):
            parser.error("Rozszerzenie pliku wyjściowego nie zgadza się z wymuszonym formatem")
    elif command == "diff":
        inferred_format = _infer_format_from_output(output_path)
        if getattr(args, "format", None) is None:
            args.format = inferred_format or "json"
        elif inferred_format is not None and inferred_format != getattr(args, "format"):
            parser.error("Rozszerzenie pliku wyjściowego nie zgadza się z wymuszonym formatem")

    if command == "list":
        payload = _handle_list(
            verbose=bool(getattr(args, "verbose", False)),
            selected=getattr(args, "core_config", None)
            and core_metadata
            and core_metadata.get("metrics_service", {}).get("risk_profile"),
            sources=sources,
            core_metadata=core_metadata,
        )
        _dump_payload(payload, output=getattr(args, "output", None), fmt=str(getattr(args, "format", "json")))
        return 0

    if command == "show":
        try:
            payload = _handle_show(args.name, sources=sources, core_metadata=core_metadata)
        except (KeyError, ValueError) as exc:
            parser.error(str(exc))
        _dump_payload(payload, output=getattr(args, "output", None), fmt=str(getattr(args, "format", "json")))
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
            if yaml is None:
                parser.error("Wymagany PyYAML (pip install pyyaml), aby użyć formatu YAML")
            yaml_payload = payload or {}
            yaml_text = yaml.safe_dump(yaml_payload, allow_unicode=True, sort_keys=False)
            _write_text_output(yaml_text, output=getattr(args, "output", None))
            return 0
        _dump_payload(payload or {}, output=getattr(args, "output", None))
        return 0

    if command == "bundle":
        try:
            stage_mapping = _parse_stage_mapping(getattr(args, "stages", None))
            manifest_payload = _handle_bundle(
                stage_mapping=stage_mapping,
                output_dir=str(getattr(args, "output_dir")),
                env_style=str(getattr(args, "env_style", "dotenv")),
                config_format=str(getattr(args, "config_format", "yaml")),
                sources=sources,
                core_metadata=core_metadata,
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        _dump_payload(manifest_payload, output=getattr(args, "output", None))
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
                include_unchanged=not bool(getattr(args, "hide_unchanged", False)),
                sections=getattr(args, "sections", None),
            )
        except (KeyError, ValueError) as exc:
            parser.error(str(exc))
        if args.format == "yaml":
            if yaml is None:
                parser.error("Wymagany PyYAML (pip install pyyaml), aby użyć formatu YAML")
            yaml_text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
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
        _dump_payload(payload, output=getattr(args, "output", None), fmt=str(getattr(args, "format", "json")))
        return exit_code

    parser.error(f"Nieobsługiwane polecenie: {command}")
    return 2


if __name__ == "__main__":  # pragma: no cover - obsługa uruchomień z CLI
    raise SystemExit(main())
