"""Narzędzia do ładowania i zapisywania manifestów pluginów."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .manifest import SignedStrategyPlugin, StrategyPluginManifest


def load_manifest(path: str | Path) -> StrategyPluginManifest:
    """Ładuje manifest strategii z pliku JSON lub YAML."""

    data = _read_file(path)
    if path_as_path(path).suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(data) or {}
    else:
        payload = json.loads(data)
    if not isinstance(payload, dict):
        raise TypeError("Manifest musi być obiektem JSON/YAML")
    return StrategyPluginManifest.from_dict(payload)


def dump_manifest(manifest: StrategyPluginManifest, path: str | Path) -> None:
    destination = path_as_path(path)
    if destination.suffix.lower() in {".yaml", ".yml"}:
        payload = manifest.to_dict()
        destination.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        destination.write_text(manifest.to_json(), encoding="utf-8")


def load_package(path: str | Path) -> SignedStrategyPlugin:
    data = _read_file(path)
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise TypeError("Package pluginu musi być obiektem JSON")
    return SignedStrategyPlugin.from_dict(payload)


def dump_package(package: SignedStrategyPlugin, path: str | Path) -> None:
    destination = path_as_path(path)
    destination.write_text(package.to_json(), encoding="utf-8")


def path_as_path(path: str | Path) -> Path:
    if isinstance(path, Path):
        return path
    return Path(path)


def _read_file(path: str | Path) -> str:
    candidate = path_as_path(path)
    return candidate.read_text(encoding="utf-8")


__all__ = [
    "load_manifest",
    "dump_manifest",
    "load_package",
    "dump_package",
]

