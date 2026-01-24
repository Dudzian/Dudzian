"""Pakiet logiki backendowej interfejsu Qt."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .compliance_controller import ComplianceController
    from .dashboard_settings import DashboardSettingsController
    from .diagnostics_controller import DiagnosticsController
    from .license_audit_controller import LicenseAuditController
    from .licensing_controller import LicensingController
    from .logging import (
        get_onboarding_logger,
        get_runbook_logger,
        get_support_logger,
        get_update_logger,
    )
    from .onboarding_service import OnboardingService
    from .privacy_settings import PrivacySettingsController
    from .runbook_controller import RunbookController
    from .runtime_service import RuntimeService
    from .support_center import SupportCenterController
    from .telemetry_provider import TelemetryProvider
    from .update_controller import OfflineUpdateController

_LAZY_IMPORTS = {
    "ComplianceController": "ui.backend.compliance_controller",
    "DashboardSettingsController": "ui.backend.dashboard_settings",
    "DiagnosticsController": "ui.backend.diagnostics_controller",
    "LicenseAuditController": "ui.backend.license_audit_controller",
    "LicensingController": "ui.backend.licensing_controller",
    "OnboardingService": "ui.backend.onboarding_service",
    "PrivacySettingsController": "ui.backend.privacy_settings",
    "RunbookController": "ui.backend.runbook_controller",
    "RuntimeService": "ui.backend.runtime_service",
    "SupportCenterController": "ui.backend.support_center",
    "TelemetryProvider": "ui.backend.telemetry_provider",
    "OfflineUpdateController": "ui.backend.update_controller",
    "get_onboarding_logger": "ui.backend.logging",
    "get_update_logger": "ui.backend.logging",
    "get_runbook_logger": "ui.backend.logging",
    "get_support_logger": "ui.backend.logging",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_IMPORTS.keys())
