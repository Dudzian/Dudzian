"""Auto-trading helpers exposed by the bot_core namespace."""
from __future__ import annotations

from .app import AutoTrader, DecisionCycleReport, EmitterLike
from .paper_app import PaperAutoTradeApp
from .audit import DecisionAuditLog, DecisionAuditRecord
from .decision_scheduler import AutoTraderDecisionScheduler
from .risk_bridge import GuardrailTrigger, RiskDecision
from .schedule import ScheduleOverride, ScheduleState, ScheduleWindow, TradingSchedule

__all__ = [
    "AutoTrader",
    "DecisionCycleReport",
    "AutoTraderDecisionScheduler",
    "EmitterLike",
    "RiskDecision",
    "GuardrailTrigger",
    "DecisionAuditLog",
    "DecisionAuditRecord",
    "ScheduleOverride",
    "TradingSchedule",
    "ScheduleWindow",
    "ScheduleState",
    "PaperAutoTradeApp",
]
