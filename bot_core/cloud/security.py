"""Obsługa autoryzacji cloudowej (HWID/licencje + sesje)."""

from __future__ import annotations

import json
import logging
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import grpc
from google.protobuf import timestamp_pb2

from bot_core.cloud.config import CloudAllowedClientConfig, CloudSecurityConfig
from bot_core.generated import trading_pb2, trading_pb2_grpc
from bot_core.security.fingerprint import verify_license_payload_signature
from bot_core.security.hwid import HwIdProvider, HwIdProviderError
from bot_core.security.license_service import LicenseService, LicenseServiceError

LOGGER = logging.getLogger(__name__)


class CloudSecurityError(RuntimeError):
    """Ogólny błąd warstwy bezpieczeństwa cloud."""


class CloudAuthorizationError(CloudSecurityError):
    """Żądanie nie spełnia wymogów autoryzacyjnych."""


@dataclass(slots=True)
class _AuthorizedSession:
    token: str
    license_id: str
    fingerprint: str
    expires_at: datetime


def _signature_to_mapping(signature: trading_pb2.CloudAuthSignature | None) -> Mapping[str, str]:
    if signature is None:
        return {}
    payload: dict[str, str] = {}
    if signature.algorithm:
        payload["algorithm"] = signature.algorithm
    if signature.value:
        payload["value"] = signature.value
    if signature.key_id:
        payload["key_id"] = signature.key_id
    return payload


class CloudSecurityManager:
    """Zarządza whitelistą klientów i sesjami CloudSession."""

    def __init__(
        self,
        config: CloudSecurityConfig,
        *,
        license_service: LicenseService | None = None,
    ) -> None:
        self._config = config
        self._license_service = license_service
        self._audit_log_path = Path(config.audit_log_path).expanduser()
        self._session_ttl = max(60, int(config.session_ttl_seconds or 900))
        self._require_handshake = bool(config.require_handshake)
        self._lock = threading.RLock()
        self._sessions: dict[str, _AuthorizedSession] = {}
        self._allowed: dict[tuple[str, str], CloudAllowedClientConfig] = {}
        self._verified_licenses: set[tuple[str, Path]] = set()
        self.refresh_allowed_clients(config.allowed_clients)

    @property
    def requires_handshake(self) -> bool:
        return self._require_handshake

    def refresh_allowed_clients(self, entries: Sequence[CloudAllowedClientConfig]) -> None:
        with self._lock:
            updated = {
                (entry.license_id, entry.fingerprint): entry
                for entry in entries
                if entry.license_id and entry.fingerprint
            }
            self._allowed = updated
            self._sessions = {
                token: session
                for token, session in self._sessions.items()
                if (session.license_id, session.fingerprint) in updated
            }
            allowed_bundles = {
                (entry.license_id, entry.license_bundle_path)
                for entry in updated.values()
                if entry.license_bundle_path is not None
            }
            self._verified_licenses = {
                key for key in self._verified_licenses if key in allowed_bundles
            }

    def authorize(self, request: trading_pb2.CloudAuthRequest) -> _AuthorizedSession:
        if not self.requires_handshake:
            raise CloudAuthorizationError("Tryb cloud nie wymaga handshake'u – nic do autoryzacji.")
        normalized_fingerprint = self._normalize_fingerprint(request.fingerprint)
        license_id = (request.license_id or "").strip()
        if not license_id:
            raise CloudAuthorizationError("Brak identyfikatora licencji w żądaniu autoryzacyjnym.")
        key = (license_id, normalized_fingerprint)
        with self._lock:
            entry = self._allowed.get(key)
        if entry is None:
            self._append_audit_log(
                "cloud_auth_rejected",
                {
                    "reason": "unknown_client",
                    "license_id": license_id,
                    "fingerprint": normalized_fingerprint,
                },
            )
            raise CloudAuthorizationError("Licencja/HWID nie znajdują się na allowliście serwera cloud.")
        self._verify_license_bundle(entry, normalized_fingerprint)
        signature_mapping = _signature_to_mapping(request.signature)
        payload: dict[str, object] = {
            "license_id": license_id,
            "fingerprint": normalized_fingerprint,
        }
        if request.nonce:
            payload["nonce"] = request.nonce
        if not verify_license_payload_signature(
            payload,
            signature_mapping,
            fingerprint=normalized_fingerprint,
            secret=entry.shared_secret,
        ):
            self._append_audit_log(
                "cloud_auth_rejected",
                {
                    "reason": "invalid_signature",
                    "license_id": license_id,
                    "fingerprint": normalized_fingerprint,
                },
            )
            raise CloudAuthorizationError("Podpis autoryzacyjny jest niepoprawny lub brakujący.")
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self._session_ttl)
        token = secrets.token_urlsafe(32)
        session = _AuthorizedSession(
            token=token,
            license_id=license_id,
            fingerprint=normalized_fingerprint,
            expires_at=expires_at,
        )
        with self._lock:
            self._sessions[token] = session
        self._append_audit_log(
            "cloud_auth_granted",
            {
                "license_id": license_id,
                "fingerprint": normalized_fingerprint,
                "expires_at": expires_at.isoformat(),
            },
        )
        return session

    def require_session(self, metadata: Iterable[tuple[str, str]]) -> _AuthorizedSession:
        if not self.requires_handshake:
            raise CloudSecurityError("Warstwa cloud nie wymaga autoryzacji – interceptor nie powinien być aktywny.")
        token = self._extract_token(metadata)
        if not token:
            self._append_audit_log(
                "cloud_session_missing",
                {"event_type": "metadata", "reason": "missing_header"},
            )
            raise CloudAuthorizationError("Brak nagłówka Authorization: CloudSession <token>.")
        with self._lock:
            session = self._sessions.get(token)
        if session is None:
            self._append_audit_log(
                "cloud_session_invalid",
                {"event_type": "metadata", "reason": "unknown_token"},
            )
            raise CloudAuthorizationError("Sesja cloudowa nie istnieje lub wygasła.")
        if session.expires_at <= datetime.now(timezone.utc):
            with self._lock:
                self._sessions.pop(token, None)
            self._append_audit_log(
                "cloud_session_invalid",
                {
                    "event_type": "metadata",
                    "reason": "expired",
                    "license_id": session.license_id,
                },
            )
            raise CloudAuthorizationError("Sesja cloudowa wygasła.")
        key = (session.license_id, session.fingerprint)
        with self._lock:
            if key not in self._allowed:
                self._append_audit_log(
                    "cloud_session_invalid",
                    {
                        "event_type": "metadata",
                        "reason": "license_revoked",
                        "license_id": session.license_id,
                    },
                )
                raise CloudAuthorizationError("Licencja została usunięta z allowlisty cloud.")
        return session

    def _extract_token(self, metadata: Iterable[tuple[str, str]]) -> str | None:
        for key, value in metadata:
            if key.lower() != "authorization":
                continue
            text = value.strip()
            if not text:
                continue
            if text.lower().startswith("cloudsession "):
                return text.split(" ", 1)[1].strip()
        return None

    def _normalize_fingerprint(self, fingerprint: str) -> str:
        provider = HwIdProvider(fingerprint_reader=lambda: fingerprint)
        try:
            return provider.read()
        except HwIdProviderError as exc:  # pragma: no cover - walidacja wejścia
            raise CloudAuthorizationError(str(exc)) from exc

    def _verify_license_bundle(self, entry: CloudAllowedClientConfig, fingerprint: str) -> None:
        if not self._license_service or not entry.license_bundle_path:
            return
        cache_key = (entry.license_id, entry.license_bundle_path)
        if cache_key in self._verified_licenses:
            return
        try:
            snapshot = self._license_service.load_from_file(
                entry.license_bundle_path,
                expected_hwid=fingerprint,
            )
        except FileNotFoundError:
            LOGGER.warning("Pakiet licencyjny %s nie istnieje – pomijam dodatkową walidację", entry.license_bundle_path)
            return
        except LicenseServiceError as exc:  # pragma: no cover - walidacja konfiguracji
            LOGGER.error("Nie udało się zweryfikować pakietu licencji dla %s: %s", entry.license_id, exc)
            raise CloudAuthorizationError("Pakiet licencyjny klienta jest nieprawidłowy.") from exc
        expected_id = getattr(getattr(snapshot, "capabilities", None), "license_id", None)
        if expected_id and expected_id != entry.license_id:
            raise CloudAuthorizationError(
                "Identyfikator licencji w pakiecie nie zgadza się z allowlistą cloud."
            )
        self._verified_licenses.add(cache_key)

    def _append_audit_log(self, event: str, details: Mapping[str, object]) -> None:
        entry = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        try:
            self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_log_path.open("a", encoding="utf-8") as handle:
                json.dump(entry, handle, ensure_ascii=False)
                handle.write("\n")
        except OSError:  # pragma: no cover - log opcjonalny
            LOGGER.debug("Nie udało się zapisać wpisu audytowego cloud", exc_info=True)


class CloudAuthInterceptor(grpc.ServerInterceptor):
    """Interceptor wymuszający metadane CloudSession."""

    def __init__(self, security_manager: CloudSecurityManager) -> None:
        self._manager = security_manager

    def intercept_service(self, continuation, handler_call_details):  # type: ignore[override]
        handler = continuation(handler_call_details)
        if handler is None:
            return None
        method = handler_call_details.method or ""
        if method.endswith("/AuthorizeClient"):
            return handler
        if not self._manager.requires_handshake:
            return handler
        return self._wrap_handler(handler)

    def _wrap_handler(self, handler):
        def unary_unary(request, context):
            self._enforce_metadata(context)
            return handler.unary_unary(request, context)

        def unary_stream(request, context):
            self._enforce_metadata(context)
            return handler.unary_stream(request, context)

        def stream_unary(request_iterator, context):
            self._enforce_metadata(context)
            return handler.stream_unary(request_iterator, context)

        def stream_stream(request_iterator, context):
            self._enforce_metadata(context)
            return handler.stream_stream(request_iterator, context)

        if handler.unary_unary:
            return grpc.unary_unary_rpc_method_handler(
                unary_unary,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.unary_stream:
            return grpc.unary_stream_rpc_method_handler(
                unary_stream,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.stream_unary:
            return grpc.stream_unary_rpc_method_handler(
                stream_unary,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.stream_stream:
            return grpc.stream_stream_rpc_method_handler(
                stream_stream,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        return handler

    def _enforce_metadata(self, context: grpc.ServicerContext) -> None:
        try:
            self._manager.require_session(context.invocation_metadata())
        except CloudAuthorizationError as exc:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, str(exc))


class CloudAuthServicer(trading_pb2_grpc.CloudAuthServiceServicer):
    """Udostępnia RPC AuthorizeClient dla klientów UI."""

    def __init__(self, security_manager: CloudSecurityManager) -> None:
        self._manager = security_manager

    def AuthorizeClient(self, request, context):  # type: ignore[override]
        try:
            session = self._manager.authorize(request)
        except CloudAuthorizationError as exc:
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(str(exc))
            return trading_pb2.CloudAuthResponse(authorized=False, message=str(exc))
        expires_pb = timestamp_pb2.Timestamp()
        expires_pb.FromDatetime(session.expires_at)
        return trading_pb2.CloudAuthResponse(
            authorized=True,
            session_token=session.token,
            expires_at=expires_pb,
            message="authorized",
        )


__all__ = [
    "CloudAuthInterceptor",
    "CloudAuthServicer",
    "CloudAuthorizationError",
    "CloudSecurityError",
    "CloudSecurityManager",
]
