"""Schematy kontraktowe raportów Decision Engine."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class DecisionEngineSummary(BaseModel):
    """Waliduje finalny payload raportu ``decision_engine_summary``."""

    type: Literal["decision_engine_summary"]
    generated_at: str | None = None
    total: int
    accepted: int
    rejected: int
    acceptance_rate: float
    history_limit: int
    history_window: int
    full_total: int
    full_accepted: int | None = None
    full_rejected: int | None = None
    full_acceptance_rate: float | None = None
    rejection_reasons: dict[str, int]
    unique_rejection_reasons: int
    unique_risk_flags: int
    risk_flags_with_accepts: int
    unique_stress_failures: int
    stress_failures_with_accepts: int
    unique_models: int
    models_with_accepts: int
    unique_actions: int
    actions_with_accepts: int
    unique_strategies: int
    strategies_with_accepts: int
    unique_symbols: int
    symbols_with_accepts: int
    current_acceptance_streak: int
    current_rejection_streak: int
    longest_acceptance_streak: int
    longest_rejection_streak: int
    history_start_generated_at: str | None = None
    latest_model: str | None = None
    latest_status: str | None = None
    latest_thresholds: dict[str, float | None] | None = None
    latest_reasons: list[str] | None = None
    latest_risk_flags: list[str] | None = None
    latest_stress_failures: list[str] | None = None
    latest_model_selection: dict[str, Any] | None = None
    latest_candidate: dict[str, Any] | None = None
    latest_generated_at: str | None = None
    filters: dict[str, Any] | None = None
    history: list[dict[str, Any]] | None = None
    risk_flag_counts: dict[str, int] | None = None
    stress_failure_counts: dict[str, int] | None = None
    model_usage: dict[str, int] | None = None
    action_usage: dict[str, int] | None = None
    strategy_usage: dict[str, int] | None = None
    symbol_usage: dict[str, int] | None = None
    risk_flag_breakdown: dict[str, dict[str, Any]] | None = None
    stress_failure_breakdown: dict[str, dict[str, Any]] | None = None
    model_breakdown: dict[str, dict[str, Any]] | None = None
    action_breakdown: dict[str, dict[str, Any]] | None = None
    strategy_breakdown: dict[str, dict[str, Any]] | None = None
    symbol_breakdown: dict[str, dict[str, Any]] | None = None

    model_config = ConfigDict(extra="allow")


__all__ = ["DecisionEngineSummary"]
