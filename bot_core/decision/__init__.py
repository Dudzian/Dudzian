"""Decision engine Etapu 5."""
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
