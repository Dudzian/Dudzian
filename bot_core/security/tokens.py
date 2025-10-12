"""Walidacja tokenów usługowych i proste RBAC dla serwisów runtime."""
from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

from bot_core.config.models import ServiceTokenConfig


_DEFAULT_HASH_ALGORITHM = "sha256"
_SUPPORTED_HASH_ALGORITHMS = {
    "sha224",
    "sha256",
    "sha384",
    "sha512",
    "sha3_224",
    "sha3_256",
    "sha3_384",
    "sha3_512",
}


def _parse_hash_expression(expression: str) -> tuple[str, str]:
    normalized = str(expression or "").strip().lower()
    if not normalized:
        raise ValueError("token_hash nie może być puste")
    if ":" in normalized:
        algorithm, digest = normalized.split(":", 1)
    else:
        algorithm, digest = _DEFAULT_HASH_ALGORITHM, normalized
    algorithm = algorithm.strip() or _DEFAULT_HASH_ALGORITHM
    digest = digest.strip()
    if algorithm not in _SUPPORTED_HASH_ALGORITHMS:
        raise ValueError(f"Nieobsługiwany algorytm haszujący tokenu: {algorithm}")
    if not digest:
        raise ValueError("token_hash wymaga wartości skrótu w formacie hex")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:  # pragma: no cover - walidacja wejścia
        raise ValueError("token_hash musi zawierać poprawny zapis hex") from exc
    return algorithm, digest


def _normalize_scope(scope: str) -> str:
    return scope.strip().lower()


@dataclass(slots=True, frozen=True)
class ServiceToken:
    """Pojedynczy token uprawnień wraz z zakresem (scopes)."""

    token_id: str
    scopes: frozenset[str]
    secret: str | None
    hash_algorithm: str | None
    hashed_value: str | None

    @classmethod
    def from_config(
        cls,
        config: ServiceTokenConfig,
        *,
        env: Mapping[str, str] | None = None,
    ) -> "ServiceToken":
        env_mapping = env or os.environ
        token_id = str(config.token_id).strip()
        token_value = config.token_value
        if token_value is None and config.token_env:
            env_name = str(config.token_env).strip()
            if env_name:
                token_value = env_mapping.get(env_name)
        if token_value is not None:
            token_value = str(token_value)
            if not token_value:
                token_value = None
        hash_algorithm = None
        hashed_value = None
        if config.token_hash:
            hash_algorithm, hashed_value = _parse_hash_expression(config.token_hash)
        scopes = frozenset(
            _normalize_scope(scope)
            for scope in config.scopes
            if isinstance(scope, str) and scope.strip()
        )
        return cls(
            token_id=token_id,
            scopes=scopes,
            secret=token_value,
            hash_algorithm=hash_algorithm,
            hashed_value=hashed_value,
        )

    def allows_scope(self, scope: str | None) -> bool:
        if scope is None or not scope.strip():
            return True
        if not self.scopes:
            return True
        return _normalize_scope(scope) in self.scopes

    def matches(self, candidate: str, *, scope: str | None = None) -> bool:
        if not self.allows_scope(scope):
            return False
        if self.secret is not None and hmac.compare_digest(candidate, self.secret):
            return True
        if self.hash_algorithm and self.hashed_value:
            digest = hashlib.new(self.hash_algorithm)
            digest.update(candidate.encode("utf-8"))
            computed = digest.hexdigest().lower()
            if hmac.compare_digest(computed, self.hashed_value):
                return True
        return False

    def metadata(self) -> Mapping[str, object]:
        return {
            "token_id": self.token_id,
            "scopes": sorted(self.scopes) if self.scopes else (),
            "has_plain": self.secret is not None,
            "has_hash": self.hashed_value is not None,
        }


class ServiceTokenValidator:
    """Walidator tokenów usługowych z prostą obsługą scopes."""

    def __init__(
        self,
        tokens: Sequence[ServiceToken],
        *,
        default_scope: str | None = None,
    ) -> None:
        self._tokens = tuple(tokens)
        self._default_scope = default_scope

    @property
    def tokens(self) -> tuple[ServiceToken, ...]:
        return self._tokens

    @property
    def requires_token(self) -> bool:
        return bool(self._tokens)

    @property
    def default_scope(self) -> str | None:
        return self._default_scope

    def validate(self, candidate: str | None, *, scope: str | None = None) -> ServiceToken | None:
        if not candidate:
            return None
        requested_scope = scope or self._default_scope
        for token in self._tokens:
            if token.matches(candidate, scope=requested_scope):
                return token
        return None

    def metadata(self) -> Mapping[str, object]:
        return {
            "tokens": [token.metadata() for token in self._tokens],
            "default_scope": self._default_scope,
        }


def build_service_token_validator(
    configs: Sequence[ServiceTokenConfig],
    *,
    default_scope: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ServiceTokenValidator | None:
    tokens: list[ServiceToken] = []
    for config in configs:
        token_id = str(config.token_id).strip()
        if not token_id:
            continue
        tokens.append(ServiceToken.from_config(config, env=env))
    if not tokens:
        return None
    return ServiceTokenValidator(tokens, default_scope=default_scope)


def resolve_service_token_secret(
    configs: Sequence[ServiceTokenConfig],
    *,
    scope: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Zwraca jawny sekret tokenu usługi dostępnego dla podanego scope."""

    env_mapping = env or os.environ
    for config in configs:
        token = ServiceToken.from_config(config, env=env_mapping)
        if not token.allows_scope(scope):
            continue
        if token.secret:
            return token.secret
    return None


__all__ = [
    "ServiceToken",
    "ServiceTokenValidator",
    "build_service_token_validator",
    "resolve_service_token_secret",
]
