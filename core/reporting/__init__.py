"""Moduły raportujące scenariusze E2E oraz guardrail'e."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .compliance_reporter import ComplianceFindingRow, ComplianceReport
    from .e2e_reporter import DemoPaperReport, StepStatus
    from .guardrails_reporter import (
        GuardrailLogRecord,
        GuardrailQueueSummary,
        GuardrailReport,
        GuardrailReportEndpoint,
    )
    from .retraining_reporter import RetrainingEventRow, RetrainingReport

_LAZY_IMPORTS = {
    "ComplianceFindingRow": ".compliance_reporter",
    "ComplianceReport": ".compliance_reporter",
    "DemoPaperReport": ".e2e_reporter",
    "StepStatus": ".e2e_reporter",
    "GuardrailReport": ".guardrails_reporter",
    "GuardrailLogRecord": ".guardrails_reporter",
    "GuardrailQueueSummary": ".guardrails_reporter",
    "GuardrailReportEndpoint": ".guardrails_reporter",
    "RetrainingReport": ".retraining_reporter",
    "RetrainingEventRow": ".retraining_reporter",
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_IMPORTS.keys())
