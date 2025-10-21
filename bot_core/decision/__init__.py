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
