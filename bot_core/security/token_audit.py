"""Audyt konfiguracji tokenów usługowych i RBAC."""
from __future__ import annotations

import os
import stat
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from bot_core.config.models import (
    CoreConfig,
    MetricsServiceConfig,
    RiskServiceConfig,
    ServiceTokenConfig,
)
from bot_core.security.tokens import ServiceToken

__all__ = [
    "TokenAuditReport",
    "TokenAuditServiceReport",
    "TokenAuditFinding",
    "audit_service_token_configs",
    "audit_service_tokens",
]


@dataclass(slots=True)
class TokenAuditFinding:
    level: str
    message: str
    details: Mapping[str, Any] | None = None


@dataclass(slots=True)
class TokenAuditServiceReport:
    service: str
    enabled: bool
    configured: bool
    legacy_token: bool
    token_count: int
    required_scopes: Mapping[str, Sequence[str]]
    coverage: Mapping[str, Sequence[str]]
    findings: Sequence[TokenAuditFinding]
    tokens: Sequence[Mapping[str, Any]]

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "service": self.service,
            "enabled": self.enabled,
            "configured": self.configured,
            "legacy_token": self.legacy_token,
            "token_count": self.token_count,
            "required_scopes": {k: list(v) for k, v in self.required_scopes.items()},
            "coverage": {k: list(v) for k, v in self.coverage.items()},
            "findings": [
                {
                    "level": finding.level,
                    "message": finding.message,
                    "details": dict(finding.details) if finding.details else None,
                }
                for finding in self.findings
            ],
            "tokens": [dict(token) for token in self.tokens],
        }


@dataclass(slots=True)
class TokenAuditReport:
    services: Sequence[TokenAuditServiceReport]

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "services": [service.as_dict() for service in self.services],
            "warnings": [
                finding.message
                for service in self.services
                for finding in service.findings
                if finding.level == "warning"
            ],
            "errors": [
                finding.message
                for service in self.services
                for finding in service.findings
                if finding.level == "error"
            ],
        }


def _normalize_required_scopes(required_scopes: Iterable[str]) -> Mapping[str, Sequence[str]]:
    normalized: dict[str, set[str]] = {}
    for scope in required_scopes:
        normalized_scope = str(scope).strip().lower()
        if not normalized_scope:
            continue
        normalized.setdefault(normalized_scope, set()).add(normalized_scope)
    return {scope: tuple(sorted(values)) for scope, values in normalized.items()}


def _token_metadata(config: ServiceTokenConfig, token: ServiceToken, *, env: Mapping[str, str]) -> Mapping[str, Any]:
    source: str | None = None
    if config.token_env:
        source = "env"
    elif config.token_value:
        source = "inline"
    elif config.token_hash:
        source = "hash"
    env_present = None
    if config.token_env:
        env_present = config.token_env in env and bool(env.get(config.token_env))
    return {
        "token_id": token.token_id,
        "scopes": sorted(token.scopes) if token.scopes else (),
        "has_plain": token.secret is not None,
        "has_hash": token.hashed_value is not None,
        "hash_algorithm": token.hash_algorithm,
        "source": source,
        "token_env": config.token_env,
        "env_present": env_present,
    }


def audit_service_token_configs(
    service_name: str,
    *,
    config: MetricsServiceConfig | RiskServiceConfig | None,
    required_scopes: Sequence[str],
    env: Mapping[str, str],
    warn_on_legacy: bool = True,
) -> TokenAuditServiceReport:
    findings: list[TokenAuditFinding] = []
    coverage: MutableMapping[str, list[str]] = defaultdict(list)
    normalized_required = _normalize_required_scopes(required_scopes)

    if config is None:
        return TokenAuditServiceReport(
            service=service_name,
            enabled=False,
            configured=False,
            legacy_token=False,
            token_count=0,
            required_scopes=normalized_required,
            coverage={scope: [] for scope in normalized_required},
            findings=findings,
            tokens=(),
        )

    enabled = bool(getattr(config, "enabled", True))
    tokens_cfg = tuple(getattr(config, "rbac_tokens", ()) or ())
    auth_token = getattr(config, "auth_token", None)
    auth_token_env = getattr(config, "auth_token_env", None)
    auth_token_env_present = bool(
        auth_token_env and env.get(str(auth_token_env))
    )
    auth_token_file = getattr(config, "auth_token_file", None)
    auth_token_file_exists = False
    auth_token_file_mode: str | None = None
    auth_token_file_over_permissive = False
    if auth_token_file:
        try:
            auth_token_path = Path(str(auth_token_file)).expanduser()
        except (OSError, TypeError, ValueError):
            auth_token_path = None
        else:
            try:
                auth_token_file_exists = auth_token_path.is_file()
            except OSError:
                auth_token_file_exists = False
            if auth_token_file_exists:
                try:
                    file_mode = stat.S_IMODE(auth_token_path.stat().st_mode)
                    auth_token_file_mode = format(file_mode, "#04o")
                    if os.name != "nt" and file_mode & 0o077:
                        auth_token_file_over_permissive = True
                except OSError:
                    auth_token_file_mode = None

    tokens: list[Mapping[str, Any]] = []
    seen_ids: Counter[str] = Counter()
    if tokens_cfg:
        for cfg in tokens_cfg:
            token = ServiceToken.from_config(cfg, env=env)
            meta = _token_metadata(cfg, token, env=env)
            tokens.append(meta)
            identifier = token.token_id or "<unnamed>"
            seen_ids[identifier] += 1
            if not token.secret and not token.hashed_value:
                findings.append(
                    TokenAuditFinding(
                        level="warning",
                        message="Token RBAC nie posiada ani jawnego sekretu ani skrótu",
                        details={"service": service_name, "token_id": identifier},
                    )
                )
            if cfg.token_env and not meta["env_present"]:
                findings.append(
                    TokenAuditFinding(
                        level="warning",
                        message="Zmienna środowiskowa tokenu RBAC nie jest ustawiona",
                        details={"service": service_name, "token_id": identifier, "token_env": cfg.token_env},
                    )
                )
            for scope in normalized_required:
                if token.allows_scope(scope) and (token.secret or token.hashed_value):
                    coverage[scope].append(identifier)
        for token_id, count in seen_ids.items():
            if count > 1:
                findings.append(
                    TokenAuditFinding(
                        level="warning",
                        message="Duplikat token_id w konfiguracji RBAC",
                        details={"service": service_name, "token_id": token_id, "count": count},
                    )
                )
    else:
        if enabled:
            findings.append(
                TokenAuditFinding(
                    level="warning" if auth_token else "error",
                    message="Brak skonfigurowanych tokenów RBAC dla włączonej usługi",
                    details={"service": service_name},
                )
            )
    if auth_token_env and not auth_token_env_present:
        findings.append(
            TokenAuditFinding(
                level="error" if enabled and not tokens_cfg else "warning",
                message="Zmienna środowiskowa auth_token_env nie jest ustawiona",
                details={
                    "service": service_name,
                    "token_env": str(auth_token_env),
                },
            )
        )
    if auth_token_file and not auth_token_file_exists:
        findings.append(
            TokenAuditFinding(
                level="warning",
                message="Plik legacy auth_token wskazany w konfiguracji nie istnieje",
                details={
                    "service": service_name,
                    "token_file": str(auth_token_file),
                },
            )
        )
    if auth_token_file_exists and auth_token_file_over_permissive:
        findings.append(
            TokenAuditFinding(
                level="warning",
                message="Plik legacy auth_token ma zbyt szerokie uprawnienia",
                details={
                    "service": service_name,
                    "token_file": str(auth_token_file),
                    "mode": auth_token_file_mode or "<unknown>",
                    "expected_max_mode": "0o600",
                },
            )
        )
    if auth_token or auth_token_env_present or auth_token_file_exists:
        if warn_on_legacy:
            findings.append(
                TokenAuditFinding(
                    level="warning",
                    message="Usługa używa legacy auth_token – rozważ migrację na RBAC",
                    details={"service": service_name},
                )
            )
        for scope in normalized_required:
            coverage[scope].append("<legacy-auth-token>")

    for scope in normalized_required:
        if enabled and not coverage.get(scope):
            findings.append(
                TokenAuditFinding(
                    level="error",
                    message="Brak tokenu zapewniającego wymagany scope",
                    details={"service": service_name, "scope": scope},
                )
            )
        coverage.setdefault(scope, [])

    return TokenAuditServiceReport(
        service=service_name,
        enabled=enabled,
        configured=bool(tokens_cfg) or bool(auth_token) or auth_token_env_present or auth_token_file_exists,
        legacy_token=bool(auth_token or auth_token_env or auth_token_file),
        token_count=len(tokens_cfg),
        required_scopes=normalized_required,
        coverage={scope: tuple(ids) for scope, ids in coverage.items()},
        findings=tuple(findings),
        tokens=tuple(tokens),
    )


def audit_service_tokens(
    core_config: CoreConfig,
    *,
    env: Mapping[str, str] | None = None,
    metrics_required_scopes: Sequence[str] | None = None,
    risk_required_scopes: Sequence[str] | None = None,
    scheduler_required_scopes: Mapping[str, Sequence[str]]
    | Sequence[str]
    | None = None,
    warn_on_legacy: bool = True,
) -> TokenAuditReport:
    env_map = env or {}
    metrics_scopes = metrics_required_scopes or ("metrics.read",)
    risk_scopes = risk_required_scopes or ("risk.read",)

    scheduler_default_scopes: Sequence[str]
    scheduler_overrides: Mapping[str, Sequence[str]]
    scheduler_default_scopes = ("runtime.schedule.read", "runtime.schedule.write")
    scheduler_overrides = {}
    if scheduler_required_scopes:
        if isinstance(scheduler_required_scopes, Mapping):
            scheduler_overrides = {
                str(name): tuple(str(scope).strip().lower() for scope in values if scope)
                for name, values in scheduler_required_scopes.items()
                if name not in {"*", "__default__"}
            }
            default_candidate = None
            for key in ("*", "__default__"):
                if key in scheduler_required_scopes:
                    default_candidate = scheduler_required_scopes[key]
                    break
            if default_candidate:
                scheduler_default_scopes = tuple(
                    str(scope).strip().lower()
                    for scope in default_candidate
                    if str(scope).strip()
                )
        else:
            scheduler_default_scopes = tuple(
                str(scope).strip().lower()
                for scope in scheduler_required_scopes
                if str(scope).strip()
            ) or scheduler_default_scopes

    services: list[TokenAuditServiceReport] = []
    services.append(
        audit_service_token_configs(
            "metrics_service",
            config=getattr(core_config, "metrics_service", None),
            required_scopes=metrics_scopes,
            env=env_map,
            warn_on_legacy=warn_on_legacy,
        )
    )
    services.append(
        audit_service_token_configs(
            "risk_service",
            config=getattr(core_config, "risk_service", None),
            required_scopes=risk_scopes,
            env=env_map,
            warn_on_legacy=warn_on_legacy,
        )
    )

    scheduler_configs = getattr(core_config, "multi_strategy_schedulers", {}) or {}
    for scheduler_name, scheduler_config in scheduler_configs.items():
        tokens = tuple(getattr(scheduler_config, "rbac_tokens", ()) or ())
        required_scopes = scheduler_overrides.get(
            scheduler_name,
            scheduler_default_scopes,
        )
        services.append(
            audit_service_token_configs(
                f"multi_strategy_scheduler:{scheduler_name}",
                config=SimpleNamespace(
                    enabled=True,
                    auth_token=None,
                    rbac_tokens=tokens,
                ),
                required_scopes=required_scopes,
                env=env_map,
                warn_on_legacy=False,
            )
        )

    return TokenAuditReport(tuple(services))
