"""Schematy Pydantic dla podsumowań decision engine."""
from __future__ import annotations

from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field


class DecisionEngineSummary(BaseModel):
    """Ustrukturyzowany wynik działania agregatora ewaluacji decision engine."""

    total: int = Field(ge=0)
    accepted: int = Field(ge=0)
    rejected: int = Field(ge=0)
    acceptance_rate: float
    history_limit: int = Field(ge=0)
    history_window: int = Field(ge=0)
    full_total: int = Field(ge=0)
    full_accepted: int | None = Field(default=None, ge=0)
    full_rejected: int | None = Field(default=None, ge=0)
    full_acceptance_rate: float | None = None
    history_start_generated_at: str | None = None
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    unique_rejection_reasons: int = Field(default=0, ge=0)
    risk_flag_counts: dict[str, int] = Field(default_factory=dict)
    risk_flags_with_accepts: int = Field(default=0, ge=0)
    unique_risk_flags: int = Field(default=0, ge=0)
    risk_flag_breakdown: dict[str, Mapping[str, object]] = Field(default_factory=dict)
    stress_failure_counts: dict[str, int] = Field(default_factory=dict)
    stress_failures_with_accepts: int = Field(default=0, ge=0)
    unique_stress_failures: int = Field(default=0, ge=0)
    stress_failure_breakdown: dict[str, Mapping[str, object]] = Field(default_factory=dict)
    model_usage: dict[str, int] = Field(default_factory=dict)
    unique_models: int = Field(default=0, ge=0)
    models_with_accepts: int = Field(default=0, ge=0)
    model_breakdown: dict[str, Mapping[str, object]] = Field(default_factory=dict)
    action_usage: dict[str, int] = Field(default_factory=dict)
    unique_actions: int = Field(default=0, ge=0)
    actions_with_accepts: int = Field(default=0, ge=0)
    action_breakdown: dict[str, Mapping[str, object]] = Field(default_factory=dict)
    strategy_usage: dict[str, int] = Field(default_factory=dict)
    unique_strategies: int = Field(default=0, ge=0)
    strategies_with_accepts: int = Field(default=0, ge=0)
    strategy_breakdown: dict[str, Mapping[str, object]] = Field(default_factory=dict)
    symbol_usage: dict[str, int] = Field(default_factory=dict)
    unique_symbols: int = Field(default=0, ge=0)
    symbols_with_accepts: int = Field(default=0, ge=0)
    symbol_breakdown: dict[str, Mapping[str, object]] = Field(default_factory=dict)
    current_acceptance_streak: int = Field(default=0, ge=0)
    current_rejection_streak: int = Field(default=0, ge=0)
    longest_acceptance_streak: int = Field(default=0, ge=0)
    longest_rejection_streak: int = Field(default=0, ge=0)
    latest_status: str | None = None
    latest_model: str | None = None
    latest_generated_at: str | None = None
    latest_thresholds: dict[str, float | None] | None = None
    latest_risk_flags: Sequence[str] | None = None
    latest_stress_failures: Sequence[str] | None = None
    latest_reasons: Sequence[str] | None = None
    latest_candidate: Mapping[str, object] | None = None
    latest_model_selection: Mapping[str, object] | None = None

    model_config = ConfigDict(extra="allow")


__all__ = ["DecisionEngineSummary"]
