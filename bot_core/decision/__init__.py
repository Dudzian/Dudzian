"""Decision engine Etapu 5."""

from __future__ import annotations

import os

if os.environ.get("BOT_CORE_MINIMAL_DECISION") == "1":  # pragma: no cover - tryb testowy
    from .orchestrator import DecisionOrchestrator  # noqa: WPS433
    __all__ = ["DecisionOrchestrator"]
else:
    from .ai_connector import AIManagerDecisionConnector  # noqa: WPS433
    from .models import DecisionCandidate, DecisionEvaluation, RiskSnapshot  # noqa: WPS433
    from .orchestrator import DecisionOrchestrator  # noqa: WPS433
    from .summary import summarize_evaluation_payloads  # noqa: WPS433

    __all__ = [
        "DecisionCandidate",
        "DecisionEvaluation",
        "DecisionOrchestrator",
        "AIManagerDecisionConnector",
        "RiskSnapshot",
        "summarize_evaluation_payloads",
    ]
try:  # pragma: no cover - lekkie importy w środowisku testowym
    from .ai_connector import AIManagerDecisionConnector
    from .models import DecisionCandidate, DecisionEvaluation, RiskSnapshot
    from .orchestrator import DecisionOrchestrator
except Exception:  # pragma: no cover - brak zależności dla testów
    AIManagerDecisionConnector = None  # type: ignore[assignment]
    DecisionCandidate = DecisionEvaluation = RiskSnapshot = None  # type: ignore[assignment]
    DecisionOrchestrator = None  # type: ignore[assignment]

from .summary import DecisionSummaryAggregator, summarize_evaluation_payloads
from .ai_connector import AIManagerDecisionConnector
from .models import (
    DecisionCandidate,
    DecisionEngineSummary,
    DecisionEvaluation,
    RiskSnapshot,
)
from .orchestrator import DecisionOrchestrator
from .summary import DecisionEngineSummary, summarize_evaluation_payloads
from .utils import coerce_float

__all__ = [
    "DecisionCandidate",
    "DecisionEvaluation",
    "DecisionOrchestrator",
    "AIManagerDecisionConnector",
    "RiskSnapshot",
    "DecisionSummaryAggregator",
    "DecisionEngineSummary",
    "summarize_evaluation_payloads",
    "coerce_float",
]
