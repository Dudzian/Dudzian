"""Modele konfiguracji modułu cloudowego."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from bot_core.security.fingerprint import decode_secret


class CloudConfigError(RuntimeError):
    """Błąd walidacji konfiguracji modułu cloudowego."""


@dataclass(slots=True)
class CloudRuntimeConfig:
    """Sekcja określająca źródła konfiguracji runtime Stage6."""

    config_path: Path = Path("config/runtime.yaml")
    entrypoint: str | None = None


@dataclass(slots=True)
class CloudLicenseConfig:
    """Ustawienia kontroli licencji / HWID po stronie serwera."""

    enabled: bool = False
    bundle_path: Path | None = None
    expected_hwid: str | None = None


@dataclass(slots=True)
class CloudMarketplaceConfig:
    """Parametryzacja synchronizacji katalogu presetów."""

    refresh_interval_seconds: int = 900
    auto_reload: bool = True


@dataclass(slots=True)
class CloudAllowedClientConfig:
    """Definicja klienta dopuszczonego do trybu cloud."""

    license_id: str
    fingerprint: str
    shared_secret: bytes
    note: str | None = None
    secret_source: str | None = None
    license_bundle_path: Path | None = None


@dataclass(slots=True)
class CloudSecurityConfig:
    """Ustawienia kontroli HWID/licencji w module cloudowym."""

    require_handshake: bool = False
    session_ttl_seconds: int = 900
    audit_log_path: Path = Path("logs/security_admin.log")
    allowed_clients: tuple[CloudAllowedClientConfig, ...] = ()


@dataclass(slots=True)
class CloudServerConfig:
    """Zbiorcza konfiguracja usługi cloudowej."""

    host: str = "0.0.0.0"
    port: int = 50052
    max_workers: int = 32
    log_level: str = "INFO"
    runtime: CloudRuntimeConfig = field(default_factory=CloudRuntimeConfig)
    license: CloudLicenseConfig = field(default_factory=CloudLicenseConfig)
    marketplace: CloudMarketplaceConfig = field(default_factory=CloudMarketplaceConfig)
    security: CloudSecurityConfig = field(default_factory=CloudSecurityConfig)


def _resolve_path(base: Path, value: str | Path | None) -> Path:
    path = Path(value or "").expanduser()
    if path.is_absolute():
        return path
    candidate = (base / path).resolve()
    if candidate.exists():
        return candidate
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / path).resolve()


def _resolve_optional_path(base: Path, value: str | Path | None) -> Path | None:
    if not value:
        return None
    return _resolve_path(base, value)


def _normalize_interval(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(30, number)


def _load_shared_secret(entry: Mapping[str, Any], base: Path) -> tuple[bytes, str]:
    env_name = str(entry.get("shared_secret_env") or "").strip() or None
    inline_value = entry.get("shared_secret")
    path_value = entry.get("shared_secret_path")
    if env_name:
        value = os.environ.get(env_name)
        if not value:
            raise CloudConfigError(
                f"Brak wartości w zmiennej środowiskowej '{env_name}' dla wpisu allowlisty cloud."
            )
        return decode_secret(value), f"env:{env_name}"
    if path_value:
        resolved = _resolve_path(base, path_value)
        if not resolved.exists():
            raise CloudConfigError(f"Plik z sekretem klienta cloud nie istnieje: {resolved}")
        content = resolved.read_text(encoding="utf-8").strip()
        return decode_secret(content), str(resolved)
    if inline_value:
        text = str(inline_value)
        return decode_secret(text), "inline"
    raise CloudConfigError("Każdy klient cloud musi określać shared_secret/shared_secret_env/shared_secret_path")


def _parse_allowed_clients(entries: Sequence[Any], base: Path) -> tuple[CloudAllowedClientConfig, ...]:
    allowed: list[CloudAllowedClientConfig] = []
    for raw in entries:
        if not isinstance(raw, Mapping):
            continue
        license_id = str(raw.get("license_id") or "").strip()
        fingerprint = str(raw.get("fingerprint") or "").strip()
        if not license_id or not fingerprint:
            raise CloudConfigError("Definicja klienta cloud musi zawierać license_id oraz fingerprint")
        secret_bytes, source = _load_shared_secret(raw, base)
        bundle_path = _resolve_optional_path(base, raw.get("license_bundle_path"))
        allowed.append(
            CloudAllowedClientConfig(
                license_id=license_id,
                fingerprint=fingerprint,
                shared_secret=secret_bytes,
                note=str(raw.get("note") or "") or None,
                secret_source=source,
                license_bundle_path=bundle_path,
            )
        )
    return tuple(allowed)


def load_cloud_server_config(path: str | Path) -> CloudServerConfig:
    """Ładuje konfigurację cloud z pliku YAML."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise CloudConfigError(f"Plik konfiguracji cloud nie istnieje: {config_path!s}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise CloudConfigError("Plik konfiguracji cloud musi zawierać obiekt mapujący")

    runtime_section = payload.get("runtime", {}) or {}
    if not isinstance(runtime_section, Mapping):
        runtime_section = {}
    runtime_cfg = CloudRuntimeConfig(
        config_path=_resolve_path(
            config_path.parent,
            runtime_section.get("config_path", "config/runtime.yaml"),
        ),
        entrypoint=runtime_section.get("entrypoint"),
    )

    license_section = payload.get("license", {}) or {}
    if not isinstance(license_section, Mapping):
        license_section = {}
    license_cfg = CloudLicenseConfig(
        enabled=bool(license_section.get("enabled", False)),
        bundle_path=_resolve_optional_path(config_path.parent, license_section.get("bundle_path")),
        expected_hwid=license_section.get("expected_hwid"),
    )

    marketplace_section = payload.get("marketplace", {}) or {}
    if not isinstance(marketplace_section, Mapping):
        marketplace_section = {}
    marketplace_cfg = CloudMarketplaceConfig(
        refresh_interval_seconds=_normalize_interval(
            marketplace_section.get("refresh_interval_seconds"),
            default=900,
        ),
        auto_reload=bool(marketplace_section.get("auto_reload", True)),
    )

    security_section = payload.get("security", {}) or {}
    if not isinstance(security_section, Mapping):
        security_section = {}
    audit_path = _resolve_path(config_path.parent, security_section.get("audit_log_path", "logs/security_admin.log"))
    allowed_entries = security_section.get("allowed_clients") or ()
    if not isinstance(allowed_entries, Sequence):
        allowed_entries = []
    security_cfg = CloudSecurityConfig(
        require_handshake=bool(security_section.get("require_handshake", False)),
        session_ttl_seconds=max(60, int(security_section.get("session_ttl_seconds", 900) or 900)),
        audit_log_path=audit_path,
        allowed_clients=_parse_allowed_clients(allowed_entries, config_path.parent),
    )

    return CloudServerConfig(
        host=str(payload.get("host") or "0.0.0.0"),
        port=int(payload.get("port", 50052) or 0),
        max_workers=int(payload.get("max_workers", 32) or 32),
        log_level=str(payload.get("log_level") or "INFO"),
        runtime=runtime_cfg,
        license=license_cfg,
        marketplace=marketplace_cfg,
        security=security_cfg,
    )


__all__ = [
    "CloudConfigError",
    "CloudRuntimeConfig",
    "CloudLicenseConfig",
    "CloudMarketplaceConfig",
    "CloudAllowedClientConfig",
    "CloudSecurityConfig",
    "CloudServerConfig",
    "load_cloud_server_config",
]
