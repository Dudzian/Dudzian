"""Pakiet logiki backendowej interfejsu Qt."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .logging import (
    get_onboarding_logger,
    get_runbook_logger,
    get_support_logger,
    get_update_logger,
)

__all__ = [
    "ComplianceController",
    "DashboardSettingsController",
    "DiagnosticsController",
    "LicenseAuditController",
    "LicensingController",
    "OnboardingService",
    "PrivacySettingsController",
    "RunbookController",
    "RuntimeService",
    "SupportCenterController",
    "TelemetryProvider",
    "OfflineUpdateController",
    "get_onboarding_logger",
    "get_update_logger",
    "get_runbook_logger",
    "get_support_logger",
]

_MODULE_BY_EXPORT = {
    "ComplianceController": ".compliance_controller",
    "DashboardSettingsController": ".dashboard_settings",
    "DiagnosticsController": ".diagnostics_controller",
    "LicenseAuditController": ".license_audit_controller",
    "LicensingController": ".licensing_controller",
    "OnboardingService": ".onboarding_service",
    "PrivacySettingsController": ".privacy_settings",
    "RunbookController": ".runbook_controller",
    "RuntimeService": ".runtime_service",
    "SupportCenterController": ".support_center",
    "TelemetryProvider": ".telemetry_provider",
    "OfflineUpdateController": ".update_controller",
}


def __getattr__(name: str) -> Any:
    module_path = _MODULE_BY_EXPORT.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
