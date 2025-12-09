from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from cryptography.hazmat.primitives.asymmetric import ed25519

from bot_core.marketplace.presets import parse_preset_document
from bot_core.marketplace.signatures import SignatureProvider, sign_preset_payload

from .models import PresetDocument, PresetSignature
from .presets import PresetRepository, serialize_preset_document

if TYPE_CHECKING:
    from bot_core.marketplace.signed import MarketplaceSyncResult


class MarketplaceService:
    """Centralny serwis do obsługi presetów Marketplace."""

    def __init__(
        self,
        *,
        signing_keys: Mapping[str, bytes | str] | None = None,
        providers: tuple[SignatureProvider, ...] | None = None,
        repository_root: str | Path | None = None,
        repository: PresetRepository | None = None,
    ) -> None:
        self._signing_keys = dict(signing_keys) if signing_keys else None
        self._providers = providers
        self._repository = repository or (
            PresetRepository(repository_root) if repository_root is not None else None
        )

    @property
    def repository(self) -> PresetRepository | None:
        return self._repository

    @property
    def signing_keys(self) -> Mapping[str, bytes | str] | None:
        return self._signing_keys

    def load(
        self,
        payload: bytes | str | Mapping[str, object] | Path,
        *,
        source: Path | None = None,
        require_signature: bool = False,
    ) -> PresetDocument:
        if isinstance(payload, Path):
            source = payload
            raw = payload.read_bytes()
        elif isinstance(payload, Mapping):
            raw = json.dumps(payload, ensure_ascii=False)
        else:
            raw = payload

        document = parse_preset_document(
            raw,
            source=source,
            signing_keys=self._signing_keys,
            providers=self._providers,
        )
        if require_signature and not document.verification.verified:
            raise ValueError("Preset musi zawierać zweryfikowany podpis.")
        return document

    def validate(
        self,
        document_or_payload: PresetDocument | bytes | str | Mapping[str, object],
        *,
        source: Path | None = None,
        require_signature: bool = False,
    ) -> PresetDocument:
        document = (
            document_or_payload
            if isinstance(document_or_payload, PresetDocument)
            else self.load(document_or_payload, source=source, require_signature=False)
        )

        if require_signature and not document.verification.verified:
            raise ValueError("Preset musi zawierać zweryfikowany podpis.")
        return document

    def sign(
        self,
        payload: Mapping[str, object] | PresetDocument,
        *,
        private_key: ed25519.Ed25519PrivateKey,
        key_id: str,
        issuer: str | None = None,
        include_public_key: bool = True,
        signed_at=None,
    ) -> PresetSignature:
        raw_payload = payload.payload if isinstance(payload, PresetDocument) else payload
        return sign_preset_payload(
            raw_payload,
            private_key=private_key,
            key_id=key_id,
            issuer=issuer,
            include_public_key=include_public_key,
            signed_at=signed_at,
        )

    def sync(
        self,
        catalog,
        *,
        hwid_provider=None,
    ) -> "MarketplaceSyncResult":
        if self._repository is None:
            raise ValueError("Brak repozytorium presetów do synchronizacji.")

        from bot_core.marketplace.signed import SignedPresetMarketplace

        marketplace = SignedPresetMarketplace(
            self._repository.root,
            signing_keys=self._signing_keys or {},
        )
        return marketplace.sync(catalog, hwid_provider=hwid_provider)

    def export(self, document: PresetDocument, *, format: str = "json") -> bytes:
        return serialize_preset_document(document, format=format)


__all__ = ["MarketplaceService"]
