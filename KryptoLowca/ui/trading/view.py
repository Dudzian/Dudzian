"""Tkinter view helpers used by the trading GUI."""
from __future__ import annotations

from dataclasses import dataclass
from tkinter import StringVar, ttk
from typing import Any, Mapping, Protocol

from .risk_helpers import format_key_limits, format_profile_name


class _StateWithRisk(Protocol):
    risk_profile_name: Any | None


@dataclass
class RiskProfileSection:
    """Simple container describing widgets used for risk profile presentation."""

    container: ttk.Frame
    profile_var: StringVar
    limits_var: StringVar

    def update(self, *, profile_name: Any | None = None, settings: Mapping[str, Any] | None = None) -> None:
        if profile_name is not None:
            self.profile_var.set(format_profile_name(profile_name))
        if settings is not None:
            self.limits_var.set(format_key_limits(settings))


def build_risk_profile_section(
    parent: ttk.Frame,
    state: _StateWithRisk,
    risk_manager_settings: Mapping[str, Any] | None,
) -> RiskProfileSection:
    """Create a header section with the current risk profile summary."""

    frame = ttk.Frame(parent)
    profile_var = StringVar(value=format_profile_name(getattr(state, "risk_profile_name", None)))
    limits_var = StringVar(value=format_key_limits(risk_manager_settings))

    ttk.Label(frame, text="Risk profile:").pack(side="left", padx=(12, 4))
    ttk.Label(frame, textvariable=profile_var).pack(side="left")
    ttk.Label(frame, text="Limits:").pack(side="left", padx=(12, 4))
    ttk.Label(frame, textvariable=limits_var).pack(side="left")

    return RiskProfileSection(frame, profile_var, limits_var)
