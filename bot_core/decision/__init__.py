"""Decision engine Etapu 5."""

from __future__ import annotations

import os
from typing import Any

try:  # pragma: no cover - podsystem podsumowań nie ma zależności opcjonalnych
    from .summary import (
        DecisionEngineSummary,
        DecisionSummaryAggregator,
        summarize_evaluation_payloads,
    )
except Exception:  # pragma: no cover - defensywnie zapewniamy stabilne API
    DecisionEngineSummary = None  # type: ignore[assignment]
    DecisionSummaryAggregator = None  # type: ignore[assignment]

    def summarize_evaluation_payloads(*_: Any, **__: Any) -> Any:  # type: ignore[misc]
        raise RuntimeError("Decision summary utilities are not available")

try:  # pragma: no cover - utils mogą być odchudzone w środowisku testowym
    from .utils import coerce_float
except Exception:  # pragma: no cover - zachowujemy podpis funkcji
    def coerce_float(*_: Any, **__: Any) -> Any:  # type: ignore[misc]
        raise RuntimeError("Decision utilities are not available")

_MINIMAL_MODE = os.environ.get("BOT_CORE_MINIMAL_DECISION") == "1"

AIManagerDecisionConnector = None  # type: ignore[assignment]
DecisionCandidate = None  # type: ignore[assignment]
DecisionEvaluation = None  # type: ignore[assignment]
DecisionOrchestrator = None  # type: ignore[assignment]
RiskSnapshot = None  # type: ignore[assignment]

if _MINIMAL_MODE:
    try:  # pragma: no cover - orchestrator może wymagać cięższych zależności
        from .orchestrator import DecisionOrchestrator
    except Exception:  # pragma: no cover - zachowujemy stabilny import
        DecisionOrchestrator = None  # type: ignore[assignment]

    __all__ = [
        "DecisionOrchestrator",
        "DecisionSummaryAggregator",
        "DecisionEngineSummary",
        "summarize_evaluation_payloads",
        "coerce_float",
    ]
else:
    try:  # pragma: no cover - konektor AI wymaga opcjonalnych pakietów
        from .ai_connector import AIManagerDecisionConnector
    except Exception:  # pragma: no cover - zapewniamy kompatybilność API
        AIManagerDecisionConnector = None  # type: ignore[assignment]

    try:  # pragma: no cover - modele decyzji mogą nie być zainstalowane
        from .models import (
            DecisionCandidate,
            DecisionEvaluation,
            RiskSnapshot,
        )
    except Exception:  # pragma: no cover - odchudzone środowisko
        DecisionCandidate = DecisionEvaluation = RiskSnapshot = None  # type: ignore[assignment]

    try:  # pragma: no cover - orchestrator może zależeć od innych modułów
        from .orchestrator import DecisionOrchestrator
    except Exception:  # pragma: no cover
        DecisionOrchestrator = None  # type: ignore[assignment]

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

