"""Obsługa autoryzacji cloudowej (HWID/licencje + sesje)."""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

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
    expires_monotonic: float


@dataclass(slots=True)
class _Challenge:
    client: tuple[str, str]
    expires_monotonic: float


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
        self._active = True
        self._sessions: dict[str, _AuthorizedSession] = {}
        self._allowed: dict[tuple[str, str], CloudAllowedClientConfig] = {}
        # Process-local challenges make all pre-restart proofs invalid. Never accept
        # client-generated nonces or fall back to the old one-message handshake.
        self._challenges: dict[str, _Challenge] = {}
        self._bindings: dict[tuple[str, str], tuple[bytes, Path | None]] = {}
        self._global_attempts: tuple[float, int] = (0.0, 0)
        self._source_attempts: dict[str, tuple[float, int]] = {}
        self._client_attempts: dict[tuple[str, str], tuple[float, int]] = {}
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
            bindings = {
                key: (entry.shared_secret, entry.license_bundle_path)
                for key, entry in updated.items()
            }
            retained = {key for key in bindings if self._bindings.get(key) == bindings[key]}
            self._allowed = updated
            self._bindings = bindings
            self._sessions = {
                token: session
                for token, session in self._sessions.items()
                if (session.license_id, session.fingerprint) in retained
            }
            self._challenges = {
                nonce: challenge
                for nonce, challenge in self._challenges.items()
                if challenge.client in retained
            }
            self._client_attempts = {
                key: window for key, window in self._client_attempts.items() if key in retained
            }

    def invalidate_sessions(self) -> None:
        """Invalidate current proofs while keeping this manager available."""
        with self._lock:
            self._sessions.clear()
            self._challenges.clear()

    def close(self) -> None:
        """Permanently reject admission from a stopped runtime's RPC handlers."""
        with self._lock:
            self._active = False
            self._sessions.clear()
            self._challenges.clear()

    @staticmethod
    def _take_attempt(window: tuple[float, int], now: float, limit: int) -> tuple[float, int]:
        deadline, count = window
        if now >= deadline:
            deadline, count = now + 60.0, 0
        if count >= limit:
            raise CloudAuthorizationError("Przekroczono limit prób uwierzytelniania cloud.")
        return deadline, count + 1

    def _admit_attempt(self, peer: str | None, now: float) -> None:
        # Ignore the ephemeral TCP port, so reconnecting cannot reset the limit.
        source = peer.rsplit(":", 1)[0] if peer and peer.startswith(("ipv4:", "ipv6:")) else "local"
        if len(source) > 128:
            source = "unknown"
        self._global_attempts = self._take_attempt(self._global_attempts, now, 120)
        self._source_attempts = {
            key: window for key, window in self._source_attempts.items() if window[0] > now
        }
        if source not in self._source_attempts and len(self._source_attempts) >= 1024:
            raise CloudAuthorizationError("Limit źródeł uwierzytelniania cloud.")
        self._source_attempts[source] = self._take_attempt(
            self._source_attempts.get(source, (0.0, 0)), now, 30
        )

    def _client_key(self, license_id: str, fingerprint: str, now: float) -> tuple[str, str]:
        if not self._active:
            raise CloudAuthorizationError("Uwierzytelnianie cloud zostało zatrzymane.")
        if not self.requires_handshake:
            raise CloudAuthorizationError("Handshake cloud jest nieaktywny.")
        if not license_id or len(license_id) > 256 or not fingerprint or len(fingerprint) > 256:
            raise CloudAuthorizationError("Niepoprawna tożsamość klienta cloud.")
        key = (license_id.strip(), self._normalize_fingerprint(fingerprint))
        if key not in self._allowed:
            raise CloudAuthorizationError("Klient nie jest uprawniony do handshake cloud.")
        # Only allowlisted identities allocate state, independent of TCP source.
        self._client_attempts[key] = self._take_attempt(
            self._client_attempts.get(key, (0.0, 0)), now, 10
        )
        return key

    def _prune_expired(self, now: float) -> None:
        self._challenges = {
            nonce: challenge
            for nonce, challenge in self._challenges.items()
            if challenge.expires_monotonic > now
        }
        self._sessions = {
            token: session
            for token, session in self._sessions.items()
            if session.expires_monotonic > now
        }

    def issue_challenge(
        self, request: trading_pb2.CloudChallengeRequest, *, peer: str | None = None
    ) -> str:
        with self._lock:
            now = time.monotonic()
            self._admit_attempt(peer, now)
            key = self._client_key(request.license_id, request.fingerprint, now)
            self._prune_expired(now)
            if len(self._challenges) >= 1024:
                raise CloudAuthorizationError("Limit oczekujących challenge cloud.")
            nonce = secrets.token_urlsafe(32)
            if nonce in self._challenges:
                raise CloudSecurityError("Nie udało się utworzyć unikalnego challenge cloud.")
            self._challenges[nonce] = _Challenge(key, now + 60.0)
            return nonce

    def authorize(
        self, request: trading_pb2.CloudAuthRequest, *, peer: str | None = None
    ) -> _AuthorizedSession:
        # Consumption, signature verification, allowlist checks and granting the
        # session share one lock, including races with client revocation.
        with self._lock:
            now = time.monotonic()
            self._admit_attempt(peer, now)
            key = self._client_key(request.license_id, request.fingerprint, now)
            self._prune_expired(now)
            if not request.nonce or len(request.nonce) > 128:
                raise CloudAuthorizationError("Wymagany jest świeży challenge serwera cloud.")
            challenge = self._challenges.pop(request.nonce, None)
            if challenge is None or challenge.client != key:
                raise CloudAuthorizationError("Challenge cloud jest nieprawidłowy lub zużyty.")
            entry = self._allowed[key]
            license_id, fingerprint = key
            self._verify_license_bundle(entry, fingerprint)
            payload: dict[str, object] = {
                "license_id": license_id,
                "fingerprint": fingerprint,
                "nonce": request.nonce,
            }
            if not verify_license_payload_signature(
                payload,
                _signature_to_mapping(request.signature),
                fingerprint=fingerprint,
                secret=entry.shared_secret,
            ):
                self._append_audit_log("cloud_auth_rejected", {"reason": "invalid_signature"})
                raise CloudAuthorizationError(
                    "Podpis autoryzacyjny jest niepoprawny lub brakujący."
                )
            if time.monotonic() >= challenge.expires_monotonic:
                raise CloudAuthorizationError("Challenge cloud wygasł podczas uwierzytelniania.")
            if len(self._sessions) >= 1024:
                raise CloudAuthorizationError("Limit aktywnych sesji cloud.")
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=self._session_ttl)
            token = secrets.token_urlsafe(32)
            if token in self._sessions:
                raise CloudSecurityError("Nie udało się utworzyć unikalnej sesji cloud.")
            session = _AuthorizedSession(
                token=token,
                license_id=license_id,
                fingerprint=fingerprint,
                expires_at=expires_at,
                expires_monotonic=time.monotonic() + self._session_ttl,
            )
            self._sessions[token] = session
        self._append_audit_log(
            "cloud_auth_granted",
            {
                "license_id": license_id,
                "fingerprint": fingerprint,
                "expires_at": expires_at.isoformat(),
            },
        )
        return session

    def require_session(self, metadata: Iterable[tuple[str, str]]) -> _AuthorizedSession:
        if not self.requires_handshake:
            raise CloudSecurityError("Handshake cloud jest nieaktywny.")
        token = self._extract_token(metadata)
        if not token or len(token) > 128:
            raise CloudAuthorizationError("Brak poprawnej sesji cloud.")
        # Linearize admission with refresh/invalidation as well as issuance.
        with self._lock:
            if not self._active:
                raise CloudAuthorizationError("Uwierzytelnianie cloud zostało zatrzymane.")
            session = self._sessions.get(token)
            if session is None:
                raise CloudAuthorizationError("Sesja cloudowa nie istnieje lub wygasła.")
            if (
                session.expires_at <= datetime.now(timezone.utc)
                or session.expires_monotonic <= time.monotonic()
            ):
                self._sessions.pop(token, None)
                raise CloudAuthorizationError("Sesja cloudowa wygasła.")
            if (session.license_id, session.fingerprint) not in self._allowed:
                raise CloudAuthorizationError("Klient cloud został odwołany.")
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
        if entry.license_bundle_path is None:
            return
        if self._license_service is None:
            raise CloudAuthorizationError("Brak weryfikatora skonfigurowanej licencji cloud.")
        try:
            snapshot = self._license_service.load_from_file(
                entry.license_bundle_path,
                expected_hwid=fingerprint,
            )
        except OSError as exc:
            raise CloudAuthorizationError(
                "Nie można odczytać pakietu licencji klienta cloud."
            ) from exc
        except LicenseServiceError as exc:  # pragma: no cover - walidacja konfiguracji
            LOGGER.error(
                "Nie udało się zweryfikować pakietu licencji dla %s: %s", entry.license_id, exc
            )
            raise CloudAuthorizationError("Pakiet licencyjny klienta jest nieprawidłowy.") from exc
        expected_id = getattr(getattr(snapshot, "capabilities", None), "license_id", None)
        if expected_id and expected_id != entry.license_id:
            raise CloudAuthorizationError(
                "Identyfikator licencji w pakiecie nie zgadza się z allowlistą cloud."
            )

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
        if method in {
            "/botcore.trading.v1.CloudAuthService/IssueChallenge",
            "/botcore.trading.v1.CloudAuthService/AuthorizeClient",
        }:
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

    def __init__(
        self,
        security_manager: CloudSecurityManager,
        *,
        health_headers_provider: Callable[[], Mapping[str, str]] | None = None,
    ) -> None:
        self._manager = security_manager
        self._health_headers_provider = health_headers_provider or (lambda: {})

    def _cloud_health_metadata(self) -> list[tuple[str, str]]:
        try:
            headers = self._health_headers_provider() or {}
        except Exception:  # pragma: no cover - defensywne logowanie
            LOGGER.debug("Health provider dla CloudAuth zgłosił wyjątek", exc_info=True)
            headers = {}
        return [(str(key), str(value)) for key, value in headers.items()]

    def _apply_cloud_health_metadata(self, rpc_context) -> None:
        metadata = self._cloud_health_metadata()
        if not metadata or rpc_context is None:
            return
        for method_name in ("send_initial_metadata", "set_trailing_metadata"):
            sender = getattr(rpc_context, method_name, None)
            if callable(sender):
                try:
                    sender(metadata)
                except Exception:  # pragma: no cover - defensywne logowanie
                    LOGGER.debug("Nie udało się przesłać metadanych health (auth)", exc_info=True)

    def IssueChallenge(self, request, context):
        try:
            nonce = self._manager.issue_challenge(request, peer=context.peer())
        except CloudAuthorizationError as exc:
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(str(exc))
            self._apply_cloud_health_metadata(context)
            return trading_pb2.CloudChallengeResponse()
        self._apply_cloud_health_metadata(context)
        return trading_pb2.CloudChallengeResponse(nonce=nonce, ttl_seconds=60)

    def AuthorizeClient(self, request, context):  # type: ignore[override]
        try:
            session = self._manager.authorize(request, peer=context.peer())
        except CloudAuthorizationError as exc:
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(str(exc))
            self._apply_cloud_health_metadata(context)
            return trading_pb2.CloudAuthResponse(authorized=False, message=str(exc))
        expires_pb = timestamp_pb2.Timestamp()
        expires_pb.FromDatetime(session.expires_at)
        response = trading_pb2.CloudAuthResponse(
            authorized=True,
            session_token=session.token,
            expires_at=expires_pb,
            message="authorized",
        )
        self._apply_cloud_health_metadata(context)
        return response


__all__ = [
    "CloudAuthInterceptor",
    "CloudAuthServicer",
    "CloudAuthorizationError",
    "CloudSecurityError",
    "CloudSecurityManager",
]
