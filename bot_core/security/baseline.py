"""Zbiorczy raport bezpieczeństwa łączący audyty TLS oraz RBAC."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from bot_core.config.models import CoreConfig
from bot_core.security.tls_audit import audit_tls_assets
from bot_core.security.token_audit import audit_service_tokens

__all__ = [
    "SecurityBaselineReport",
    "generate_security_baseline_report",
]


@dataclass(slots=True)
class SecurityBaselineReport:
    """Wynik agregacji audytów bezpieczeństwa usług runtime."""

    tls: Mapping[str, Any]
    tokens: Mapping[str, Any]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    status: str

    def as_dict(self) -> Mapping[str, Any]:
        """Serializuje raport do struktury słownikowej."""

        return {
            "status": self.status,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "tls": self.tls,
            "tokens": self.tokens,
        }

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)


def _deduplicate(sequence: Sequence[str]) -> tuple[str, ...]:
    if not sequence:
        return ()
    deduped = list(dict.fromkeys(str(item) for item in sequence if item))
    return tuple(deduped)


def generate_security_baseline_report(
    core_config: CoreConfig,
    *,
    env: Mapping[str, str] | None = None,
    warn_expiring_within_days: float = 30.0,
    metrics_required_scopes: Sequence[str] | None = None,
    risk_required_scopes: Sequence[str] | None = None,
    scheduler_required_scopes: Mapping[str, Sequence[str]]
    | Sequence[str]
    | None = None,
    warn_on_shared_secret_tokens: bool = True,
) -> SecurityBaselineReport:
    """Generuje zbiorczy raport bezpieczeństwa dla runtime.

    Raport łączy wynik audytu TLS oraz RBAC, zapewniając jeden wskaźnik
    ``status`` wykorzystywany w pipeline'ach demo→paper→live.
    """

    env_map = env or os.environ

    tls_report = audit_tls_assets(
        core_config,
        warn_expiring_within_days=warn_expiring_within_days,
        env=env_map,
    )

    token_report = audit_service_tokens(
        core_config,
        env=env_map,
        metrics_required_scopes=metrics_required_scopes,
        risk_required_scopes=risk_required_scopes,
        scheduler_required_scopes=scheduler_required_scopes,
        warn_on_shared_secret=warn_on_shared_secret_tokens,
    ).as_dict()

    combined_warnings = _deduplicate(
        list(tls_report.get("warnings", ()))
        + list(token_report.get("warnings", ()))
    )
    combined_errors = _deduplicate(
        list(tls_report.get("errors", ()))
        + list(token_report.get("errors", ()))
    )

    if combined_errors:
        status = "error"
    elif combined_warnings:
        status = "warning"
    else:
        status = "ok"

    return SecurityBaselineReport(
        tls=tls_report,
        tokens=token_report,
        warnings=combined_warnings,
        errors=combined_errors,
        status=status,
    )

