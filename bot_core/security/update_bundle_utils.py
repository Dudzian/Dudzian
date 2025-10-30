"""Wspólne funkcje dla obsługi pakietów aktualizacji offline."""
"""Wspólne funkcje dla obsługi pakietów aktualizacji offline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from bot_core.security.update import UpdateManifest


class UpdateDescriptionError(RuntimeError):
    """Wyjątek zgłaszany przy problemach z opisem pakietu."""


def hash_file(path: Path, algorithm: str) -> str:
    """Zwraca skrót pliku dla wskazanego algorytmu (sha256/sha384)."""

    normalized = algorithm.lower()
    if normalized not in {"sha256", "sha384"}:
        raise ValueError(f"Nieobsługiwany algorytm haszujący: {algorithm}")
    digest = hashlib.new(normalized)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> tuple[UpdateManifest, Mapping[str, Any]]:
    """Ładuje ``manifest.json`` i zwraca jego strukturę wraz z surowym payloadem."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensywne logowanie
        raise UpdateDescriptionError(f"Manifest {path} zawiera niepoprawny JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise UpdateDescriptionError("Manifest aktualizacji musi być obiektem JSON")
    try:
        manifest = UpdateManifest.from_mapping(payload)
    except ValueError as exc:
        raise UpdateDescriptionError(str(exc)) from exc
    return manifest, payload


def describe_update_directory(package_dir: Path) -> dict[str, Any]:
    """Buduje ujednolicony opis pakietu aktualizacji."""

    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        raise UpdateDescriptionError(f"Brak manifestu w katalogu {package_dir}")

    manifest, raw_payload = load_manifest(manifest_path)

    fingerprint = raw_payload.get("fingerprint")
    if fingerprint is None:
        fingerprint = raw_payload.get("metadata", {}).get("fingerprint") if isinstance(raw_payload.get("metadata"), Mapping) else None

    signature = raw_payload.get("signature")
    signature_str: str | None
    signature_obj: Mapping[str, Any] | None
    if isinstance(signature, Mapping):
        signature_obj = signature
        signature_str = signature.get("value") if isinstance(signature.get("value"), str) else None
    else:
        signature_str = signature if isinstance(signature, str) else None
        signature_obj = None

    diff_artifact = next((artifact for artifact in manifest.artifacts if artifact.kind == "diff"), None)
    full_artifact = next((artifact for artifact in manifest.artifacts if artifact.kind != "diff"), None)

    integrity_manifest = raw_payload.get("integrity_manifest")
    if integrity_manifest is None:
        integrity_manifest = raw_payload.get("integrity")
    if integrity_manifest is not None and not isinstance(integrity_manifest, Mapping):
        raise UpdateDescriptionError("Pole integrity_manifest musi być mapą")

    metadata = raw_payload.get("metadata") if isinstance(raw_payload.get("metadata"), Mapping) else {}

    base_id = None
    if diff_artifact is not None:
        base_id = diff_artifact.base_id
    if base_id is None:
        candidate = raw_payload.get("base_id") or raw_payload.get("baseId")
        if isinstance(candidate, str):
            base_id = candidate

    return {
        "id": str(raw_payload.get("id") or package_dir.name),
        "version": manifest.version,
        "fingerprint": fingerprint,
        "signature": signature_str,
        "signature_object": dict(signature_obj) if signature_obj is not None else {},
        "differential": diff_artifact is not None,
        "base_id": base_id,
        "payload_file": full_artifact.path if full_artifact is not None else None,
        "diff_file": diff_artifact.path if diff_artifact is not None else None,
        "integrity": dict(integrity_manifest) if integrity_manifest else {},
        "metadata": dict(metadata),
        "manifest_path": str(manifest_path),
    }


__all__ = [
    "hash_file",
    "load_manifest",
    "describe_update_directory",
    "UpdateDescriptionError",
]
