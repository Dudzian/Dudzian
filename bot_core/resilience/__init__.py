"""Pakiet odporności operacyjnej Stage6 – scalony HEAD + main.

Eksportuje zarówno moduły bundling/walidacja/scenariusze (HEAD), jak i
klasy drillów failover (main), jeśli są dostępne.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

# --- Część z HEAD: podpakiety narzędziowe -----------------------------------
# Zachowujemy oryginalną strukturę, aby 'from bot_core.resilience import audit' itd. działało.
try:
    from . import audit, bundle, drill, hypercare, policy, self_healing  # type: ignore
    _HAS_HEAD_SUBMODULES = True
except Exception:  # pragma: no cover – w razie użycia poza pakietem
    _HAS_HEAD_SUBMODULES = False

# --- Część z main: klasy failover drill --------------------------------------
_FAILOVER_SYMBOLS = (
    "FailoverDrillMetrics",
    "FailoverDrillReport",
    "FailoverDrillResult",
    "ResilienceFailoverDrill",
)
_FAILOVER_CACHE: Dict[str, Any] = {}
_FAILOVER_IMPORT_ERROR: Exception | None = None

try:
    from bot_core.resilience.failover import (  # type: ignore
        FailoverDrillMetrics,
        FailoverDrillReport,
        FailoverDrillResult,
        ResilienceFailoverDrill,
    )

    for name, value in (
        ("FailoverDrillMetrics", FailoverDrillMetrics),
        ("FailoverDrillReport", FailoverDrillReport),
        ("FailoverDrillResult", FailoverDrillResult),
        ("ResilienceFailoverDrill", ResilienceFailoverDrill),
    ):
        _FAILOVER_CACHE[name] = value
    _HAS_FAILOVER = True
except Exception as exc:  # pragma: no cover – brak modułu failover nie blokuje pakietu
    _FAILOVER_IMPORT_ERROR = exc
    _HAS_FAILOVER = False

# --- Publiczny interfejs -----------------------------------------------------
__all__ = []

if _HAS_HEAD_SUBMODULES:
    __all__.extend(["audit", "bundle", "drill", "hypercare", "policy", "self_healing"])

if _HAS_FAILOVER:
    __all__.extend(
        [
            "FailoverDrillMetrics",
            "FailoverDrillResult",
            "FailoverDrillReport",
            "ResilienceFailoverDrill",
        ]
    )


def __getattr__(name: str):  # pragma: no cover - mechanizm awaryjny
    """Lazy import failover symbols when optional dependency becomes available."""

    global _HAS_FAILOVER

    if name in {
        "FailoverDrillMetrics",
        "FailoverDrillResult",
        "FailoverDrillReport",
        "ResilienceFailoverDrill",
    }:
        try:
            from bot_core.resilience.failover import (  # type: ignore
                FailoverDrillMetrics as _FailoverDrillMetrics,
                FailoverDrillReport as _FailoverDrillReport,
                FailoverDrillResult as _FailoverDrillResult,
                ResilienceFailoverDrill as _ResilienceFailoverDrill,
            )
        except Exception as exc:  # pragma: no cover - propagate the root cause
            raise ImportError(
                "Nie można zaimportować komponentów failover resilience"
            ) from (_FAILOVER_IMPORT_ERROR or exc)

        globals().update(
            {
                "FailoverDrillMetrics": _FailoverDrillMetrics,
                "FailoverDrillReport": _FailoverDrillReport,
                "FailoverDrillResult": _FailoverDrillResult,
                "ResilienceFailoverDrill": _ResilienceFailoverDrill,
            }
        )
        if not _HAS_FAILOVER:
            __all__.extend(
                [
                    "FailoverDrillMetrics",
                    "FailoverDrillResult",
                    "FailoverDrillReport",
                    "ResilienceFailoverDrill",
                ]
            )
            _HAS_FAILOVER = True
        return globals()[name]
    raise AttributeError(name)
