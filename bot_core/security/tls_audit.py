"""Audyt konfiguracji TLS dla usług runtime."""
from __future__ import annotations

import os
import ssl
from pathlib import Path
from typing import Any, Mapping, MutableSequence

from bot_core.runtime.file_metadata import (
    collect_security_warnings,
    file_reference_metadata,
)
from bot_core.security.certificates import certificate_reference_metadata

__all__ = [
    "verify_certificate_key_pair",
    "audit_tls_entry",
    "audit_tls_assets",
]


def verify_certificate_key_pair(
    certificate_path: str | Path,
    private_key_path: str | Path,
    *,
    password: str | None = None,
) -> tuple[bool, str | None]:
    """Sprawdza, czy klucz prywatny pasuje do certyfikatu X.509.

    Zwraca krotkę ``(result, message)``.  W przypadku sukcesu ``result`` jest
    ``True``, a komunikat ma wartość ``None``.  Przy błędzie ``result`` jest
    ``False``, natomiast ``message`` zawiera krótki opis problemu.
    """

    cert = Path(certificate_path).expanduser()
    key = Path(private_key_path).expanduser()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    try:
        context.load_cert_chain(str(cert), str(key), password=password)
    except FileNotFoundError as exc:
        return False, f"Nie udało się załadować materiału TLS: {exc}"
    except ssl.SSLError as exc:
        return False, f"Klucz prywatny nie pasuje do certyfikatu ({exc})"
    except Exception as exc:  # noqa: BLE001 - diagnostyka nieoczekiwanych błędów
        return False, f"Weryfikacja pary certyfikat/klucz nie powiodła się ({exc})"
    return True, None


def audit_tls_entry(
    tls_config: Mapping[str, Any] | object | None,
    *,
    role_prefix: str,
    warn_expiring_within_days: float = 30.0,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Buduje raport audytu TLS dla pojedynczej sekcji konfiguracji."""

    env_map = env or os.environ
    report: dict[str, Any] = {
        "configured": tls_config is not None,
        "enabled": False,
        "certificate": None,
        "private_key": None,
        "client_ca": None,
        "warnings": [],
        "errors": [],
        "key_matches_certificate": None,
        "pinned_fingerprint_match": None,
        "pinned_fingerprints": (),
        "password_env": None,
    }

    if tls_config is None:
        return report

    warnings: MutableSequence[str] = []
    errors: MutableSequence[str] = []

    enabled = bool(getattr(tls_config, "enabled", False))
    report["enabled"] = enabled

    certificate_path = getattr(tls_config, "certificate_path", None)
    private_key_path = getattr(tls_config, "private_key_path", None)
    client_ca_path = getattr(tls_config, "client_ca_path", None)
    pinned = tuple(getattr(tls_config, "pinned_fingerprints", ()) or ())
    password_env = getattr(tls_config, "private_key_password_env", None)
    if password_env:
        report["password_env"] = str(password_env)

    certificate_metadata = None
    if certificate_path:
        try:
            certificate_metadata = certificate_reference_metadata(
                certificate_path,
                role=f"{role_prefix}_certificate",
                warn_expiring_within_days=warn_expiring_within_days,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostyka ścieżek TLS
            errors.append(f"Nie udało się odczytać certyfikatu TLS ({exc})")
        else:
            report["certificate"] = certificate_metadata
            for entry in collect_security_warnings(certificate_metadata):
                warnings.extend(str(item) for item in entry.get("warnings", ()))

    private_key_metadata = None
    if private_key_path:
        try:
            private_key_metadata = file_reference_metadata(
                private_key_path,
                role="tls_key",
            )
        except Exception as exc:  # noqa: BLE001 - diagnostyka ścieżek TLS
            errors.append(f"Nie udało się odczytać klucza prywatnego TLS ({exc})")
        else:
            private_key_metadata.setdefault("role_label", f"{role_prefix}_key")
            report["private_key"] = private_key_metadata
            for entry in collect_security_warnings(private_key_metadata):
                warnings.extend(str(item) for item in entry.get("warnings", ()))

    if client_ca_path:
        try:
            client_ca_metadata = certificate_reference_metadata(
                client_ca_path,
                role=f"{role_prefix}_client_ca",
                warn_expiring_within_days=warn_expiring_within_days,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostyka ścieżek TLS
            errors.append(f"Nie udało się odczytać klientskiego CA ({exc})")
        else:
            report["client_ca"] = client_ca_metadata
            for entry in collect_security_warnings(client_ca_metadata):
                warnings.extend(str(item) for item in entry.get("warnings", ()))

    if certificate_metadata and private_key_metadata:
        cert_exists = bool(certificate_metadata.get("exists"))
        key_exists = bool(private_key_metadata.get("exists"))
        if cert_exists and key_exists:
            password_value: str | None = None
            if password_env:
                password_value = env_map.get(str(password_env))
                if password_value is None:
                    warnings.append(
                        "Zmienna środowiskowa prywatnego klucza TLS nie jest ustawiona – klucz może być zaszyfrowany."
                    )
            success, message = verify_certificate_key_pair(
                certificate_metadata["path"],
                private_key_metadata["path"],
                password=password_value,
            )
            report["key_matches_certificate"] = success
            if not success and message:
                errors.append(message)

    if pinned:
        report["pinned_fingerprints"] = pinned
        available: dict[str, set[str]] = {}
        if certificate_metadata:
            certificates = certificate_metadata.get("certificates") or []
            for entry in certificates:
                if not isinstance(entry, Mapping):
                    continue
                sha256 = entry.get("fingerprint_sha256")
                if sha256:
                    available.setdefault("sha256", set()).add(str(sha256).lower())
        matched = False
        unsupported_algorithms: set[str] = set()
        for pin in pinned:
            normalized = str(pin).lower()
            if ":" not in normalized:
                errors.append(
                    f"Wpis pinningu '{pin}' ma nieprawidłowy format (brak algorytmu)"
                )
                continue
            algorithm, fingerprint = normalized.split(":", 1)
            if not fingerprint:
                errors.append(
                    f"Wpis pinningu '{pin}' ma pusty fingerprint"
                )
                continue
            fingerprints = available.get(algorithm)
            if fingerprints is None:
                unsupported_algorithms.add(algorithm)
                continue
            if fingerprint in fingerprints:
                matched = True
        if unsupported_algorithms:
            warnings.append(
                "Brak fingerprintów certyfikatu dla algorytmów: "
                + ", ".join(sorted(unsupported_algorithms))
            )
        report["pinned_fingerprint_match"] = matched if certificate_metadata else None
        if certificate_metadata and not matched:
            errors.append(
                "Żaden fingerprint certyfikatu nie pasuje do konfiguracji pinningu"
            )

    report["warnings"] = list(dict.fromkeys(warnings)) if warnings else []
    report["errors"] = list(dict.fromkeys(errors)) if errors else []
    return report


def audit_tls_assets(
    core_config: object,
    *,
    warn_expiring_within_days: float = 30.0,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Generuje raport TLS dla usług MetricsService i RiskService."""

    env_map = env or os.environ
    result: dict[str, Any] = {"services": {}, "warnings": [], "errors": []}

    metrics_config = getattr(core_config, "metrics_service", None)
    if metrics_config is not None:
        rbac_tokens_configured = bool(getattr(metrics_config, "rbac_tokens", ()) or ())
        service_report = {
            "enabled": bool(getattr(metrics_config, "enabled", False)),
            "auth_token_configured": bool(
                str(getattr(metrics_config, "auth_token", "") or "").strip()
            ),
            "rbac_tokens_configured": rbac_tokens_configured,
            "tls": audit_tls_entry(
                getattr(metrics_config, "tls", None),
                role_prefix="metrics_tls",
                warn_expiring_within_days=warn_expiring_within_days,
                env=env_map,
            ),
            "warnings": [],
            "errors": [],
        }
        if (
            service_report["enabled"]
            and not service_report["auth_token_configured"]
            and not rbac_tokens_configured
        ):
            service_report["warnings"].append(
                "MetricsService ma włączone API bez tokenu autoryzacyjnego"
            )
        if service_report["enabled"] and not service_report["tls"]["enabled"]:
            service_report["warnings"].append(
                "MetricsService działa bez TLS – rozważ włączenie szyfrowania"
            )
        service_report["warnings"].extend(service_report["tls"].get("warnings", ()))
        service_report["errors"].extend(service_report["tls"].get("errors", ()))
        service_report["warnings"] = list(dict.fromkeys(service_report["warnings"]))
        service_report["errors"] = list(dict.fromkeys(service_report["errors"]))
        result["services"]["metrics_service"] = service_report
        result["warnings"].extend(service_report["warnings"])
        result["errors"].extend(service_report["errors"])

    risk_config = getattr(core_config, "risk_service", None)
    if risk_config is not None:
        rbac_tokens_configured = bool(getattr(risk_config, "rbac_tokens", ()) or ())
        service_report = {
            "enabled": bool(getattr(risk_config, "enabled", False)),
            "auth_token_configured": bool(
                str(getattr(risk_config, "auth_token", "") or "").strip()
            ),
            "rbac_tokens_configured": rbac_tokens_configured,
            "tls": audit_tls_entry(
                getattr(risk_config, "tls", None),
                role_prefix="risk_tls",
                warn_expiring_within_days=warn_expiring_within_days,
                env=env_map,
            ),
            "warnings": [],
            "errors": [],
        }
        if (
            service_report["enabled"]
            and not service_report["auth_token_configured"]
            and not rbac_tokens_configured
        ):
            service_report["warnings"].append(
                "RiskService ma włączone API bez tokenu autoryzacyjnego"
            )
        if service_report["enabled"] and not service_report["tls"]["enabled"]:
            service_report["warnings"].append(
                "RiskService działa bez TLS – rozważ włączenie szyfrowania"
            )
        service_report["warnings"].extend(service_report["tls"].get("warnings", ()))
        service_report["errors"].extend(service_report["tls"].get("errors", ()))
        service_report["warnings"] = list(dict.fromkeys(service_report["warnings"]))
        service_report["errors"] = list(dict.fromkeys(service_report["errors"]))
        result["services"]["risk_service"] = service_report
        result["warnings"].extend(service_report["warnings"])
        result["errors"].extend(service_report["errors"])

    result["warnings"] = list(dict.fromkeys(result["warnings"]))
    result["errors"] = list(dict.fromkeys(result["errors"]))
    return result
