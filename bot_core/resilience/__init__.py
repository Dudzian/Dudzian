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
except Exception:  # pragma: no cover – brak modułu failover nie blokuje pakietu
    _HAS_FAILOVER = False

# --- Publiczny interfejs -----------------------------------------------------
__all__ = []

if _HAS_HEAD_SUBMODULES:
    __all__.extend(["audit", "bundle", "drill", "hypercare", "policy", "self_healing"])

if _HAS_FAILOVER:
    __all__.extend(_FAILOVER_SYMBOLS)


def __getattr__(name: str) -> Any:
    if name in _FAILOVER_CACHE:
        return _FAILOVER_CACHE[name]
    if name in _FAILOVER_SYMBOLS:
        module = importlib.import_module("bot_core.resilience.failover")
        value = getattr(module, name)
        _FAILOVER_CACHE[name] = value
        globals()[name] = value
        if name not in __all__:
            __all__.append(name)
        return value
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(_FAILOVER_SYMBOLS)))
