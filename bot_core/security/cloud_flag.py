"""Walidacja podpisanej flagi aktywującej tryb cloud."""

from __future__ import annotations

import base64
import binascii
import json
import os
from pathlib import Path
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

from bot_core.config.models import RuntimeCloudSignedFlagConfig
from bot_core.security.fingerprint import decode_secret
from bot_core.security.signing import canonical_json_bytes, verify_hmac_signature

try:  # pragma: no cover - PyYAML może nie być dostępny w dystrybucjach light
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


class CloudFlagValidationError(RuntimeError):
    """Wyjątek zgłaszany, gdy podpis flagi cloudowej jest niepoprawny."""


def _as_optional_str(value: object | None) -> str | None:
    if value in (None, "", False):
        return None
    text = str(value).strip()
    return text or None


def _resolve_runtime_path(base: Path, value: object | None) -> str | None:
    text = _as_optional_str(value)
    if text is None:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    candidate = (base / path).expanduser()
    try:
        candidate = candidate.resolve(strict=False)
    except Exception:  # pragma: no cover - defensywne
        candidate = candidate.absolute()
    if candidate.exists() or candidate.parent.exists():
        return str(candidate)
    fallback = (base.parent / path).expanduser()
    try:
        fallback = fallback.resolve(strict=False)
    except Exception:  # pragma: no cover - defensywne
        fallback = fallback.absolute()
    return str(fallback)


def _load_flag_payload(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise CloudFlagValidationError(f"Plik flagi cloud nie istnieje: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise CloudFlagValidationError(f"Nie udało się zdekodować JSON flagi cloud: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CloudFlagValidationError("Plik flagi cloud musi zawierać obiekt JSON")
    if not payload.get("enabled"):
        raise CloudFlagValidationError("Flaga cloudowa nie zawiera 'enabled: true'.")
    return dict(payload)


def _load_signature(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise CloudFlagValidationError(f"Plik podpisu flagi cloud nie istnieje: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise CloudFlagValidationError(f"Nie udało się odczytać podpisu JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise CloudFlagValidationError("Dokument podpisu flagi cloud musi być obiektem JSON")
    signature = document.get("signature") if "signature" in document else document
    if not isinstance(signature, Mapping):
        raise CloudFlagValidationError("Struktura podpisu jest nieprawidłowa")
    return signature


def _load_key_material(config: RuntimeCloudSignedFlagConfig) -> bytes:
    # 1. Jawne źródła z konfiguracji
    if config.key_value:
        return decode_secret(str(config.key_value))

    if config.key_env:
        value = os.environ.get(config.key_env)
        if not value:
            raise CloudFlagValidationError(
                f"Zmienna środowiskowa '{config.key_env}' nie zawiera klucza "
                "do weryfikacji flagi cloud."
            )
        return decode_secret(value)

    if config.key_path:
        path = Path(config.key_path).expanduser()
        if not path.exists():
            raise CloudFlagValidationError(f"Plik klucza flagi cloud nie istnieje: {path}")
        return decode_secret(path.read_text(encoding="utf-8"))

    # 2. 🔥 PATCH: fallback dla testów / CI
    fallback_env = "CLOUD_RUNTIME_FLAG_SECRET"
    fallback_value = os.environ.get(fallback_env)
    if fallback_value:
        return decode_secret(fallback_value)

    raise CloudFlagValidationError(
        "Sekcja cloud.enabled_signed nie określa źródła klucza "
        "(key_value/key_env/key_path) ani nie ustawiono "
        "CLOUD_RUNTIME_FLAG_SECRET."
    )


def _verify_ed25519(
    payload: Mapping[str, Any],
    signature: Mapping[str, Any],
    key_bytes: bytes,
) -> bool:
    value = signature.get("value")
    if not isinstance(value, str):
        return False
    try:
        sig_bytes = base64.b64decode(value)
    except (ValueError, binascii.Error):
        return False
    try:
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(key_bytes)
    except ValueError as exc:  # pragma: no cover
        raise CloudFlagValidationError(
            "Nie udało się zainicjalizować klucza publicznego Ed25519"
        ) from exc
    try:
        public_key.verify(sig_bytes, canonical_json_bytes(payload))
        return True
    except InvalidSignature:
        return False


def _verify_signature(
    payload: Mapping[str, Any],
    signature: Mapping[str, Any],
    config: RuntimeCloudSignedFlagConfig,
    key_bytes: bytes,
) -> bool:
    algorithm = (config.algorithm or "HMAC-SHA256").strip()
    if algorithm.upper().startswith("HMAC"):
        return verify_hmac_signature(payload, signature, key=key_bytes, algorithm=algorithm)
    if algorithm.lower() == "ed25519":
        declared = str(signature.get("algorithm") or "").lower() or "ed25519"
        if declared != "ed25519":
            return False
        return _verify_ed25519(payload, signature, key_bytes)
    raise CloudFlagValidationError(f"Nieobsługiwany algorytm flagi cloud: {algorithm}")


def validate_signed_cloud_flag(
    config: RuntimeCloudSignedFlagConfig,
) -> Mapping[str, Any]:
    """Sprawdza podpis i zwraca payload flagi gdy jest poprawny."""

    flag_path = Path(config.flag_path).expanduser()
    signature_path = Path(config.signature_path).expanduser()
    payload = _load_flag_payload(flag_path)
    signature = _load_signature(signature_path)
    key_bytes = _load_key_material(config)

    if not _verify_signature(payload, signature, config, key_bytes):
        raise CloudFlagValidationError("Podpis flagi cloudowej jest niepoprawny.")

    return payload


def validate_runtime_cloud_flag(config_path: str | Path) -> Mapping[str, Any]:
    """Ładuje runtime.yaml i wymusza poprawny podpis cloud.enabled_signed."""

    config_path = Path(config_path).expanduser()
    signed_config = _load_runtime_signed_flag_config(config_path)
    return validate_signed_cloud_flag(signed_config)


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError(
            "Do walidacji podpisanej flagi cloud wymagany jest pakiet PyYAML; "
            "zainstaluj go poleceniem `pip install pyyaml`."
        )


def _load_runtime_signed_flag_config(
    config_path: Path,
) -> RuntimeCloudSignedFlagConfig:
    if not config_path.exists():
        raise CloudFlagValidationError(f"Plik konfiguracji runtime nie istnieje: {config_path}")
    _require_yaml()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover
        raise CloudFlagValidationError(f"Nie udało się zdekodować {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CloudFlagValidationError("config/runtime.yaml musi zawierać mapę kluczy")

    cloud_section = payload.get("cloud") or {}
    if not isinstance(cloud_section, Mapping):
        raise CloudFlagValidationError("config/runtime.yaml nie zawiera sekcji cloud")

    signed_section = cloud_section.get("enabled_signed") or {}
    if not isinstance(signed_section, Mapping) or not signed_section:
        raise CloudFlagValidationError(
            "config/runtime.yaml nie definiuje sekcji "
            "cloud.enabled_signed wymaganej do aktywacji trybu cloud."
        )

    flag_path = _resolve_runtime_path(
        config_path.parent,
        signed_section.get("flag_path") or signed_section.get("flag"),
    )
    signature_path = _resolve_runtime_path(
        config_path.parent,
        signed_section.get("signature_path") or signed_section.get("signature"),
    )

    if not flag_path or not signature_path:
        raise CloudFlagValidationError(
            "Sekcja cloud.enabled_signed musi wskazywać flag_path oraz signature_path"
        )

    key_path = _resolve_runtime_path(config_path.parent, signed_section.get("key_path"))

    algorithm = _as_optional_str(signed_section.get("algorithm")) or "HMAC-SHA256"

    return RuntimeCloudSignedFlagConfig(
        flag_path=flag_path,
        signature_path=signature_path,
        algorithm=algorithm,
        key_env=_as_optional_str(signed_section.get("key_env")),
        key_path=key_path,
        key_value=_as_optional_str(signed_section.get("key_value")),
    )


__all__ = [
    "CloudFlagValidationError",
    "validate_signed_cloud_flag",
    "validate_runtime_cloud_flag",
]
