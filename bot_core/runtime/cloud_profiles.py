"""Pomocnicze funkcje do obsługi profili cloudowych."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from bot_core.config.loader import load_cloud_client_config, load_runtime_app_config
from bot_core.config.models import (
    CloudClientConfig,
    RuntimeAppConfig,
    RuntimeCloudProfileConfig,
    RuntimeCloudSettings,
)


@dataclass(slots=True)
class RuntimeCloudClientSelection:
    """Zwraca spójny zestaw danych potrzebnych do uruchomienia trybu cloud."""

    runtime_config: RuntimeAppConfig
    profile_name: str
    profile: RuntimeCloudProfileConfig
    client: CloudClientConfig


def resolve_runtime_cloud_client(
    config_path: str | Path,
    *,
    profile_name: str | None = None,
) -> RuntimeCloudClientSelection | None:
    """Zwraca konfigurację klienta cloudowego gdy profil remote jest dostępny."""

    runtime_config = load_runtime_app_config(config_path)
    cloud_section: RuntimeCloudSettings | None = getattr(runtime_config, "cloud", None)
    if cloud_section is None:
        return None

    profiles = dict(getattr(cloud_section, "profiles", {}) or {})
    if not profiles:
        return None

    selected_name = profile_name or cloud_section.default_profile
    profile = profiles.get(selected_name) if selected_name else None
    if profile is None:
        for name, candidate in profiles.items():
            if str(getattr(candidate, "mode", "local")).lower() == "remote":
                profile = candidate
                selected_name = name
                break

    if profile is None:
        return None

    if str(getattr(profile, "mode", "local")).lower() != "remote":
        return None

    client_manifest = getattr(profile, "client_config_path", None)
    if not client_manifest:
        return None

    client_config = load_cloud_client_config(client_manifest)
    resolved_name = selected_name or "remote"
    return RuntimeCloudClientSelection(
        runtime_config=runtime_config,
        profile_name=resolved_name,
        profile=profile,
        client=client_config,
    )


__all__ = [
    "RuntimeCloudClientSelection",
    "resolve_runtime_cloud_client",
]
