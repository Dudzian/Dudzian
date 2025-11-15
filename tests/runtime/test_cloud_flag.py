from __future__ import annotations

from types import SimpleNamespace

import bot_core.runtime.cloud_profiles as cloud_profiles


def test_resolve_runtime_cloud_client_none_without_profiles(monkeypatch) -> None:
    runtime_cfg = SimpleNamespace(cloud=None)
    monkeypatch.setattr(cloud_profiles, "load_runtime_app_config", lambda path: runtime_cfg)
    selection = cloud_profiles.resolve_runtime_cloud_client("config/runtime.yaml")
    assert selection is None


def test_resolve_runtime_cloud_client_loads_manifest(monkeypatch) -> None:
    cloud_section = SimpleNamespace(
        enabled=True,
        default_profile="remote",
        profiles={
            "remote": SimpleNamespace(
                mode="remote",
                client_config_path="config/cloud/client.yaml",
                entrypoint="cloud-entry",
            )
        },
    )
    runtime_cfg = SimpleNamespace(cloud=cloud_section)
    client_cfg = SimpleNamespace(
        address="cloud.example:50052",
        metadata={},
        metadata_env={},
        metadata_files={},
        fallback_entrypoint="cloud-entry",
        allow_local_fallback=True,
        auto_connect=True,
        use_tls=False,
        tls=None,
    )
    monkeypatch.setattr(cloud_profiles, "load_runtime_app_config", lambda path: runtime_cfg)
    monkeypatch.setattr(cloud_profiles, "load_cloud_client_config", lambda path: client_cfg)

    selection = cloud_profiles.resolve_runtime_cloud_client("config/runtime.yaml")

    assert selection is not None
    assert selection.profile_name == "remote"
    assert selection.client.address == "cloud.example:50052"
