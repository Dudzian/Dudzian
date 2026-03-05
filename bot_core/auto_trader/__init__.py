"""Auto-trading helpers exposed by the bot_core namespace."""

from __future__ import annotations

from .ai_governor import AIGovernorDecision, AutoTraderAIGovernor, AutoTraderAIGovernorRunner
from .app import (
    AutoTrader,
    DecisionCycleReport,
    DecisionCycleRequest,
    DecisionCycleRunner,
    DecisionLifecycleSnapshot,
    EmitterLike,
)
from .contracts import (
    DecisionCycleDecision,
    DecisionCycleGuardrails,
    DecisionCycleLatency,
    DecisionCycleMetadata,
    DecisionCycleMetrics,
    DecisionCycleModeTransition,
    DecisionCycleTelemetry,
    DecisionJournalEntry,
    normalize_decision_journal_entry,
)
from .paper_app import PaperAutoTradeApp
from .audit import DecisionAuditLog, DecisionAuditRecord
from .decision_scheduler import AutoTraderDecisionScheduler, AutoTraderSchedulerHooks
from .lifecycle import AutoTraderLifecycleManager
from .risk_bridge import GuardrailTrigger, RiskDecision
from .schedule import ScheduleOverride, ScheduleState, ScheduleWindow, TradingSchedule

__all__ = [
    "AutoTrader",
    "DecisionCycleReport",
    "DecisionCycleRequest",
    "DecisionLifecycleSnapshot",
    "DecisionCycleRunner",
    "DecisionCycleDecision",
    "DecisionCycleGuardrails",
    "DecisionCycleLatency",
    "DecisionCycleMetadata",
    "DecisionCycleMetrics",
    "DecisionCycleModeTransition",
    "DecisionCycleTelemetry",
    "DecisionJournalEntry",
    "normalize_decision_journal_entry",
    "AutoTraderDecisionScheduler",
    "AutoTraderLifecycleManager",
    "AutoTraderSchedulerHooks",
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
    "AutoTraderAIGovernor",
    "AutoTraderAIGovernorRunner",
    "AIGovernorDecision",
]
