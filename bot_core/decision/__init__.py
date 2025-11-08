"""Decision engine Etapu 5."""

from __future__ import annotations

import os
from importlib import import_module
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

if _MINIMAL_MODE:
    _LAZY_EXPORTS: dict[str, tuple[str, str]] = {
        "DecisionOrchestrator": (".orchestrator", "DecisionOrchestrator"),
    }
else:
    _LAZY_EXPORTS = {
        "DecisionCandidate": (".models", "DecisionCandidate"),
        "DecisionEvaluation": (".models", "DecisionEvaluation"),
        "RiskSnapshot": (".models", "RiskSnapshot"),
        "DecisionOrchestrator": (".orchestrator", "DecisionOrchestrator"),
        "DecisionMetricSet": (".metrics", "DecisionMetricSet"),
        "AIManagerDecisionConnector": (".ai_connector", "AIManagerDecisionConnector"),
    }

__all__ = [
    *sorted(_LAZY_EXPORTS.keys()),
    "DecisionSummaryAggregator",
    "DecisionEngineSummary",
    "summarize_evaluation_payloads",
    "coerce_float",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - leniwa inicjalizacja
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        try:
            module = import_module(module_name, __name__)
        except Exception as exc:  # pragma: no cover - raportuj brak zależności
            raise RuntimeError(
                f"Nie udało się załadować '{name}' z modułu '{module_name}'. "
                "Zainstaluj wymagane zależności lub ustaw BOT_CORE_MINIMAL_DECISION=1."
            ) from exc
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover - kosmetyka dla introspekcji
    return sorted(__all__)
