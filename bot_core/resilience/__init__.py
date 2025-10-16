"""Pakiet odporności operacyjnej Stage6 – scalony HEAD + main.

Eksportuje zarówno moduły bundling/walidacja/scenariusze (HEAD), jak i
klasy drillów failover (main), jeśli są dostępne.
"""

from __future__ import annotations

# --- Część z HEAD: podpakiety narzędziowe -----------------------------------
# Zachowujemy oryginalną strukturę, aby 'from bot_core.resilience import audit' itd. działało.
try:
    from . import audit, bundle, drill, hypercare, policy, self_healing  # type: ignore
    _HAS_HEAD_SUBMODULES = True
except Exception:  # pragma: no cover – w razie użycia poza pakietem
    _HAS_HEAD_SUBMODULES = False

# --- Część z main: klasy failover drill --------------------------------------
# Te importy są opcjonalne – jeśli moduł istnieje, eksportujemy jego symbole.
_HAS_FAILOVER = False
try:
    from bot_core.resilience.failover import (  # type: ignore
        FailoverDrillMetrics,
        FailoverDrillReport,
        FailoverDrillResult,
        ResilienceFailoverDrill,
    )
    _HAS_FAILOVER = True
except Exception:  # pragma: no cover – brak modułu failover nie blokuje pakietu
    pass

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
