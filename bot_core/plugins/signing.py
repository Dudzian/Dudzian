"""Obsługa podpisów HMAC dla pluginów strategii."""

from __future__ import annotations

from typing import Iterable

from bot_core.security.signing import build_hmac_signature, verify_hmac_signature

from .manifest import PluginSignature, SignedStrategyPlugin, StrategyPluginManifest


class PluginSigner:
    """Buduje podpisy HMAC dla manifestów pluginów."""

    def __init__(self, key: bytes, *, key_id: str | None = None, algorithm: str = "HMAC-SHA256") -> None:
        if not isinstance(key, (bytes, bytearray)):
            raise TypeError("key must be bytes")
        self._key = bytes(key)
        self._key_id = key_id
        self._algorithm = algorithm

    def sign_manifest(self, manifest: StrategyPluginManifest) -> PluginSignature:
        signature_doc = build_hmac_signature(
            manifest.to_dict(),
            key=self._key,
            key_id=self._key_id,
            algorithm=self._algorithm,
        )
        return PluginSignature(
            algorithm=signature_doc.get("algorithm", self._algorithm),
            key_id=signature_doc.get("key_id"),
            value=signature_doc.get("value", ""),
        )

    def build_package(
        self,
        manifest: StrategyPluginManifest,
        *,
        review_notes: Iterable[str] | None = None,
    ) -> SignedStrategyPlugin:
        signature = self.sign_manifest(manifest)
        notes = tuple(str(note).strip() for note in (review_notes or ()) if str(note).strip())
        return SignedStrategyPlugin(manifest=manifest, signature=signature, review_notes=notes)


class PluginVerifier:
    """Weryfikuje podpis manifestu dostawcy."""

    def __init__(self, key: bytes, *, algorithm: str = "HMAC-SHA256") -> None:
        if not isinstance(key, (bytes, bytearray)):
            raise TypeError("key must be bytes")
        self._key = bytes(key)
        self._algorithm = algorithm

    def verify(self, manifest: StrategyPluginManifest, signature: PluginSignature) -> bool:
        signature_payload = {
            "algorithm": signature.algorithm,
            "value": signature.value,
        }
        if signature.key_id:
            signature_payload["key_id"] = signature.key_id
        return verify_hmac_signature(
            manifest.to_dict(),
            signature_payload,
            key=self._key,
            algorithm=self._algorithm,
        )


__all__ = ["PluginSigner", "PluginVerifier"]

