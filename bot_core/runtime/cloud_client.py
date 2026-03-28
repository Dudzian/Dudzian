"""Wspólne helpery do obsługi klienta cloudowego i handshake'u HWID."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import secrets
from typing import Iterable, Mapping, Sequence

import grpc

from bot_core.config.loader import load_cloud_client_config
from bot_core.config.models import CloudClientConfig
from bot_core.security.fingerprint import FingerprintError, sign_license_payload
from bot_core.security.hwid import HwIdProvider, HwIdProviderError

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - podczas testów moduł gRPC może być nieobecny
    from bot_core.generated import trading_pb2, trading_pb2_grpc
except Exception:  # pragma: no cover - środowiska light
    trading_pb2 = trading_pb2_grpc = None  # type: ignore[assignment]


DEFAULT_CLIENT_CONFIG = Path("config/cloud/client.yaml")
DEFAULT_LICENSE_STATUS = Path("var/security/license_status.json")
DEFAULT_LICENSE_SECRET = Path("var/security/license_secret.key")


@dataclass(slots=True)
class CloudClientOptions:
    """Zwarta struktura konfiguracji klienta cloudowego."""

    config_path: Path
    client: CloudClientConfig
    metadata: list[tuple[str, str]]
    tls_credentials: grpc.ChannelCredentials | None
    authority_override: str | None


@dataclass(slots=True)
class LicenseIdentity:
    """Tożsamość licencji wykorzystywana w handshake'u cloudowym."""

    license_id: str
    fingerprint: str
    source: str | None = None


@dataclass(slots=True)
class CloudHandshakeResult:
    """Szczegóły ostatniego handshake'u z CloudAuthService."""

    status: str
    message: str | None = None
    license_id: str | None = None
    fingerprint: str | None = None
    session_token: str | None = None
    expires_at: datetime | None = None


def load_cloud_client_options(
    path: str | os.PathLike[str] | None = None,
    *,
    base_metadata: Iterable[tuple[str, str]] | None = None,
) -> CloudClientOptions:
    """Ładuje config/cloud/client.yaml i przygotowuje metadane/TLS."""

    config_path = Path(path or DEFAULT_CLIENT_CONFIG).expanduser().resolve()
    client_cfg = load_cloud_client_config(config_path)
    metadata = list(base_metadata or ())
    metadata.extend(_build_metadata_from_config(client_cfg))
    credentials, authority = _configure_tls(client_cfg)
    return CloudClientOptions(config_path, client_cfg, metadata, credentials, authority)


def load_license_identity(
    status_path: str | os.PathLike[str] | None = None,
    *,
    hwid_provider: HwIdProvider | None = None,
) -> LicenseIdentity | None:
    """Odczytuje identyfikator licencji i fingerprint urządzenia."""

    source_path = Path(status_path or DEFAULT_LICENSE_STATUS).expanduser()
    license_id: str | None = None
    fingerprint: str | None = None
    source: str | None = None
    if source_path.exists():
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - diagnostyka
            LOGGER.warning("Niepoprawny JSON w statusie licencji: %s", exc)
            payload = {}
        if isinstance(payload, Mapping):
            license_id = (
                str(payload.get("license_id") or payload.get("licenseId") or "").strip() or None
            )
            fingerprint = (
                str(payload.get("local_hwid") or payload.get("fingerprint") or "").strip() or None
            )
            source = str(payload.get("bundle_path") or source_path)
    provider = hwid_provider or HwIdProvider()
    if not fingerprint:
        try:
            fingerprint = provider.read()
            source = "hwid-provider"
        except HwIdProviderError as exc:
            LOGGER.warning("Nie udało się odczytać fingerprintu urządzenia: %s", exc)
            fingerprint = None
    if not license_id or not fingerprint:
        return None
    return LicenseIdentity(license_id=license_id, fingerprint=fingerprint, source=source)


def perform_cloud_handshake(
    options: CloudClientOptions,
    identity: LicenseIdentity,
    *,
    metadata: Sequence[tuple[str, str]] | None = None,
    license_secret: bytes | None = None,
    secret_path: str | os.PathLike[str] | None = DEFAULT_LICENSE_SECRET,
    timeout: float = 5.0,
    now: datetime | None = None,
) -> CloudHandshakeResult:
    """Przeprowadza handshake CloudAuthService.AuthorizeClient."""

    if trading_pb2 is None or trading_pb2_grpc is None:
        return CloudHandshakeResult(
            status="unavailable",
            message="Brak modułów trading_pb2 w tej dystrybucji",
            license_id=identity.license_id,
            fingerprint=identity.fingerprint,
        )

    payload = {
        "license_id": identity.license_id,
        "fingerprint": identity.fingerprint,
        "nonce": secrets.token_hex(12),
    }
    try:
        signature = sign_license_payload(
            payload,
            fingerprint=identity.fingerprint,
            secret=license_secret,
            secret_path=secret_path,
        )
    except FingerprintError as exc:
        return CloudHandshakeResult(
            status="missing_secret",
            message=str(exc),
            license_id=identity.license_id,
            fingerprint=identity.fingerprint,
        )

    request = trading_pb2.CloudAuthRequest(
        fingerprint=identity.fingerprint,
        license_id=identity.license_id,
        nonce=str(payload["nonce"]),
        signature=trading_pb2.CloudAuthSignature(
            algorithm=signature.get("algorithm", ""),
            value=signature.get("value", ""),
            key_id=signature.get("key_id", ""),
        ),
    )

    channel_options: list[tuple[str, str]] = []
    if options.authority_override:
        channel_options.append(("grpc.ssl_target_name_override", options.authority_override))
    channel = _build_channel(options, tuple(channel_options))
    metadata_seq = tuple(options.metadata if metadata is None else metadata)

    try:
        stub = trading_pb2_grpc.CloudAuthServiceStub(channel)
        response = stub.AuthorizeClient(request, metadata=metadata_seq, timeout=timeout)
    except grpc.RpcError as exc:
        status_code = exc.code()
        if status_code == grpc.StatusCode.UNIMPLEMENTED:
            return CloudHandshakeResult(
                status="not_required",
                message="CloudAuthService nie jest aktywny na serwerze",
                license_id=identity.license_id,
                fingerprint=identity.fingerprint,
            )
        message = exc.details() or str(exc)
        return CloudHandshakeResult(
            status="error",
            message=message,
            license_id=identity.license_id,
            fingerprint=identity.fingerprint,
        )
    finally:
        try:
            channel.close()
        except Exception:  # pragma: no cover - defensywne
            pass

    expires_at = None
    if response.HasField("expires_at"):
        expires_at = response.expires_at.ToDatetime().replace(tzinfo=timezone.utc)
    elif now is not None and response.authorized:
        expires_at = now + timedelta(minutes=15)
    status = "ok" if response.authorized else "denied"
    return CloudHandshakeResult(
        status=status,
        message=response.message or None,
        license_id=identity.license_id,
        fingerprint=identity.fingerprint,
        session_token=response.session_token or None,
        expires_at=expires_at,
    )


def _build_metadata_from_config(client_cfg: CloudClientConfig) -> list[tuple[str, str]]:
    metadata: list[tuple[str, str]] = []
    for key, value in (client_cfg.metadata or {}).items():
        header = str(key).strip().lower()
        if header and value:
            metadata.append((header, str(value).strip()))
    for key, env_name in (client_cfg.metadata_env or {}).items():
        header = str(key).strip().lower()
        env_key = str(env_name).strip()
        if not header or not env_key:
            continue
        value = os.environ.get(env_key)
        if value:
            metadata.append((header, value))
    for key, path in (client_cfg.metadata_files or {}).items():
        header = str(key).strip().lower()
        if not header or not path:
            continue
        try:
            content = Path(path).expanduser().read_text(encoding="utf-8").strip()
        except OSError as exc:  # pragma: no cover - diagnostyka środowiska
            LOGGER.warning("Nie udało się odczytać pliku metadanych %s: %s", path, exc)
            continue
        if content:
            metadata.append((header, content))
    return metadata


def _configure_tls(
    client_cfg: CloudClientConfig,
) -> tuple[grpc.ChannelCredentials | None, str | None]:
    tls_cfg = getattr(client_cfg, "tls", None)
    use_tls = bool(getattr(client_cfg, "use_tls", False) or (tls_cfg and tls_cfg.enabled))
    if not use_tls:
        return None, None
    if tls_cfg is None:
        return grpc.ssl_channel_credentials(), None
    root_cert = (
        Path(tls_cfg.ca_certificate).expanduser().read_bytes() if tls_cfg.ca_certificate else None
    )
    client_cert = (
        Path(tls_cfg.client_certificate).expanduser().read_bytes()
        if tls_cfg.client_certificate
        else None
    )
    client_key = Path(tls_cfg.client_key).expanduser().read_bytes() if tls_cfg.client_key else None
    credentials = grpc.ssl_channel_credentials(
        root_certificates=root_cert,
        private_key=client_key,
        certificate_chain=client_cert,
    )
    return credentials, tls_cfg.override_authority


def _build_channel(options: CloudClientOptions, channel_options: tuple[tuple[str, str], ...]):
    if options.tls_credentials is not None:
        return grpc.secure_channel(
            options.client.address, options.tls_credentials, options=channel_options
        )
    return grpc.insecure_channel(options.client.address, options=channel_options)


__all__ = [
    "CloudClientOptions",
    "CloudHandshakeResult",
    "LicenseIdentity",
    "load_cloud_client_options",
    "load_license_identity",
    "perform_cloud_handshake",
]
