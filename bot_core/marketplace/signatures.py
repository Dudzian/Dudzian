from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from bot_core.security.signing import canonical_json_bytes, verify_hmac_signature

from .models import (
    PresetSignature,
    PresetSignatureVerification,
    canonical_preset_bytes,
)

DEFAULT_SIGNATURE_ALGORITHM = "ed25519"


def decode_key_material(value: bytes | str) -> bytes:
    """Dekoduje materiał klucza zakodowany w base64/hex/UTF-8."""

    def _to_text(raw: bytes | str) -> str:
        if isinstance(raw, bytes):
            try:
                return raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                return base64.b64encode(raw).decode("ascii")
        return raw.strip()

    text = _to_text(value)
    if not text:
        raise ValueError("pusty materiał klucza")

    try:
        return base64.b64decode(text, validate=True)
    except Exception:
        pass

    try:
        return bytes.fromhex(text)
    except ValueError:
        pass

    return text.encode("utf-8")


def _load_ed25519_public_key(payload: bytes) -> ed25519.Ed25519PublicKey:
    try:
        return ed25519.Ed25519PublicKey.from_public_bytes(payload)
    except ValueError:
        try:
            public_key = serialization.load_pem_public_key(payload)
        except ValueError as exc:  # pragma: no cover - defensywne logowanie
            raise ValueError("niepoprawny klucz publiczny ed25519") from exc
        if not isinstance(public_key, ed25519.Ed25519PublicKey):
            raise ValueError("klucz publiczny nie jest typu Ed25519")
        return public_key


def _load_ed25519_private_key(data: bytes) -> ed25519.Ed25519PrivateKey:
    try:
        return ed25519.Ed25519PrivateKey.from_private_bytes(data)
    except ValueError:
        try:
            key = serialization.load_pem_private_key(data, password=None)
        except ValueError as exc:  # pragma: no cover - defensywne logowanie
            raise ValueError("niepoprawny klucz prywatny Ed25519") from exc
        if not isinstance(key, ed25519.Ed25519PrivateKey):
            raise ValueError("klucz prywatny nie jest typu Ed25519")
        return key


class SignatureProvider(ABC):
    """Interfejs adaptera do weryfikacji i podpisywania presetów."""

    algorithm: str

    @abstractmethod
    def verify(
        self,
        payload: Mapping[str, Any],
        signature_doc: Mapping[str, Any] | None,
        *,
        signing_keys: Mapping[str, bytes | str] | None = None,
    ) -> tuple[PresetSignatureVerification, PresetSignature | None]:
        """Weryfikuje podpis dla podanego payloadu."""

    def supports(self, algorithm: str | None) -> bool:
        normalized = (algorithm or "").strip().lower()
        return normalized in {self.algorithm, self.algorithm.replace("_", "-")}


@dataclass
class Ed25519SignatureProvider(SignatureProvider):
    algorithm: str = DEFAULT_SIGNATURE_ALGORITHM

    def verify(
        self,
        payload: Mapping[str, Any],
        signature_doc: Mapping[str, Any] | None,
        *,
        signing_keys: Mapping[str, bytes | str] | None = None,
    ) -> tuple[PresetSignatureVerification, PresetSignature | None]:
        if not signature_doc:
            return PresetSignatureVerification(False, ("missing-signature",)), None

        key_id = signature_doc.get("key_id") or signature_doc.get("kid")
        key_id_text = str(key_id).strip() if key_id not in (None, "") else None

        signature_value = signature_doc.get("value") or signature_doc.get("signature")
        if not isinstance(signature_value, str):
            return PresetSignatureVerification(
                False, ("signature-missing",), self.algorithm, key_id_text
            ), None

        try:
            signature_bytes = decode_key_material(signature_value)
        except ValueError as exc:
            return PresetSignatureVerification(
                False,
                (f"signature-invalid:{exc}",),
                self.algorithm,
                key_id_text,
            ), None

        if len(signature_bytes) != 64:
            return PresetSignatureVerification(
                False,
                ("signature-invalid-length",),
                self.algorithm,
                key_id_text,
            ), None

        public_key_bytes: bytes | None = None
        raw_public_key = signature_doc.get("public_key")
        if raw_public_key not in (None, ""):
            try:
                public_key_bytes = decode_key_material(raw_public_key)
            except ValueError as exc:
                return PresetSignatureVerification(
                    False,
                    (f"public-key-invalid:{exc}",),
                    self.algorithm,
                    key_id_text,
                ), None
        elif signing_keys and key_id_text:
            candidate = signing_keys.get(key_id_text)
            if candidate is not None:
                try:
                    public_key_bytes = decode_key_material(candidate)
                except ValueError as exc:
                    return PresetSignatureVerification(
                        False,
                        (f"signing-key-invalid:{exc}",),
                        self.algorithm,
                        key_id_text,
                    ), None

        if public_key_bytes is None:
            return PresetSignatureVerification(
                False,
                ("missing-public-key",),
                self.algorithm,
                key_id_text,
            ), None

        try:
            public_key = _load_ed25519_public_key(public_key_bytes)
        except ValueError as exc:
            return PresetSignatureVerification(
                False,
                (f"public-key-invalid:{exc}",),
                self.algorithm,
                key_id_text,
            ), None

        try:
            public_key.verify(signature_bytes, canonical_preset_bytes(payload))
        except InvalidSignature:
            return PresetSignatureVerification(
                False,
                ("signature-mismatch",),
                self.algorithm,
                key_id_text,
            ), None

        signature = PresetSignature(
            algorithm=self.algorithm,
            value=base64.b64encode(signature_bytes).decode("ascii"),
            key_id=key_id_text,
            public_key=base64.b64encode(public_key_bytes).decode("ascii"),
            signed_at=str(signature_doc.get("signed_at") or "").strip() or None,
            issuer=str(signature_doc.get("issuer") or "").strip() or None,
        )
        return PresetSignatureVerification(True, (), self.algorithm, key_id_text), signature

    def sign(
        self,
        payload: Mapping[str, Any],
        *,
        private_key: ed25519.Ed25519PrivateKey,
        key_id: str,
        issuer: str | None = None,
        include_public_key: bool = True,
        signed_at: datetime | None = None,
    ) -> PresetSignature:
        canonical = canonical_json_bytes(payload)
        signature_bytes = private_key.sign(canonical)
        public_key_bytes = (
            private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            if include_public_key
            else None
        )
        timestamp = signed_at or datetime.now(timezone.utc)
        return PresetSignature(
            algorithm=self.algorithm,
            value=base64.b64encode(signature_bytes).decode("ascii"),
            key_id=str(key_id).strip() or None,
            public_key=(
                base64.b64encode(public_key_bytes).decode("ascii") if public_key_bytes else None
            ),
            signed_at=timestamp.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            issuer=issuer.strip() if isinstance(issuer, str) and issuer.strip() else None,
        )


@dataclass
class HmacSha256SignatureProvider(SignatureProvider):
    algorithm: str = "hmac-sha256"

    def verify(
        self,
        payload: Mapping[str, Any],
        signature_doc: Mapping[str, Any] | None,
        *,
        signing_keys: Mapping[str, bytes | str] | None = None,
    ) -> tuple[PresetSignatureVerification, PresetSignature | None]:
        if not signature_doc:
            return PresetSignatureVerification(False, ("missing-signature",)), None

        key_id = signature_doc.get("key_id")
        key_id_text = str(key_id).strip() if key_id not in (None, "") else None
        key_bytes: bytes | None = None
        if signing_keys and key_id_text:
            candidate = signing_keys.get(key_id_text)
            if candidate is not None:
                try:
                    key_bytes = decode_key_material(candidate)
                except ValueError:
                    key_bytes = None

        verified = verify_hmac_signature(
            payload, signature_doc, key=key_bytes, algorithm="HMAC-SHA256"
        )
        signature = PresetSignature(
            algorithm="HMAC-SHA256",
            value=str(signature_doc.get("value") or signature_doc.get("signature")),
            key_id=key_id_text,
        )
        issues: tuple[str, ...] = () if verified else ("signature-mismatch",)
        return PresetSignatureVerification(verified, issues, "HMAC-SHA256", key_id_text), signature


def verify_preset_signature(
    payload: Mapping[str, Any],
    signature_doc: Mapping[str, Any] | None,
    *,
    signing_keys: Mapping[str, bytes | str] | None = None,
    providers: tuple[SignatureProvider, ...] | None = None,
) -> tuple[PresetSignatureVerification, PresetSignature | None]:
    """Weryfikuje podpis z użyciem zadanego adaptera."""

    algorithm = (
        (signature_doc or {}).get("algorithm") if isinstance(signature_doc, Mapping) else None
    )
    normalized_algorithm = str(algorithm or "").strip().lower() or DEFAULT_SIGNATURE_ALGORITHM
    available = providers or (
        Ed25519SignatureProvider(),
        HmacSha256SignatureProvider(),
    )

    for provider in available:
        if provider.supports(normalized_algorithm):
            return provider.verify(payload, signature_doc, signing_keys=signing_keys)

    return PresetSignatureVerification(
        False,
        (f"unsupported-algorithm:{normalized_algorithm}",),
        normalized_algorithm,
        None,
    ), None


def sign_preset_payload(
    payload: Mapping[str, Any],
    *,
    private_key: ed25519.Ed25519PrivateKey,
    key_id: str,
    issuer: str | None = None,
    include_public_key: bool = True,
    signed_at: datetime | None = None,
) -> PresetSignature:
    provider = Ed25519SignatureProvider()
    return provider.sign(
        payload,
        private_key=private_key,
        key_id=key_id,
        issuer=issuer,
        include_public_key=include_public_key,
        signed_at=signed_at,
    )


def load_private_key(path) -> ed25519.Ed25519PrivateKey:
    data = path.read_bytes()
    try:
        material = decode_key_material(data)
    except ValueError:
        material = data
    return _load_ed25519_private_key(material)


__all__ = [
    "DEFAULT_SIGNATURE_ALGORITHM",
    "Ed25519SignatureProvider",
    "HmacSha256SignatureProvider",
    "SignatureProvider",
    "decode_key_material",
    "load_private_key",
    "sign_preset_payload",
    "verify_preset_signature",
]
