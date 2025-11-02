"""Bridges between auto-trader and risk engine payloads."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(slots=True)
class RiskDecision:
    """Serializable snapshot describing the outcome of a risk engine check."""

    should_trade: bool
    fraction: float
    state: str
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    mode: str = "demo"
    cooldown_active: bool = False
    cooldown_remaining_s: Optional[float] = None
    cooldown_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "should_trade": self.should_trade,
            "fraction": float(self.fraction),
            "state": self.state,
            "reason": self.reason,
            "details": dict(self.details),
            "mode": self.mode,
        }
        if self.stop_loss_pct is not None:
            payload["stop_loss_pct"] = float(self.stop_loss_pct)
        if self.take_profit_pct is not None:
            payload["take_profit_pct"] = float(self.take_profit_pct)
        payload["cooldown_active"] = self.cooldown_active
        if self.cooldown_remaining_s is not None:
            payload["cooldown_remaining_s"] = float(self.cooldown_remaining_s)
        if self.cooldown_reason is not None:
            payload["cooldown_reason"] = self.cooldown_reason
        return payload


@dataclass(slots=True)
class GuardrailTrigger:
    """Structured details about a guardrail that forced a HOLD signal."""

    name: str
    label: str
    comparator: str
    threshold: float
    unit: Optional[str] = None
    value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "comparator": self.comparator,
            "threshold": float(self.threshold),
        }
        if self.unit is not None:
            payload["unit"] = self.unit
        if self.value is not None:
            payload["value"] = float(self.value)
        return payload


__all__ = ["RiskDecision", "GuardrailTrigger"]
