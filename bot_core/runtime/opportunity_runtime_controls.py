"""Runtime-hot operator control plane dla polityki Opportunity AI."""

from __future__ import annotations

from dataclasses import dataclass
import threading


@dataclass(slots=True, frozen=True)
class OpportunityRuntimeControlsSnapshot:
    """Snapshot aktualnego runtime state dla Opportunity AI."""

    opportunity_ai_enabled: bool
    manual_kill_switch: bool
    execution_disabled: bool
    policy_mode: str
    revision: int


class OpportunityRuntimeControls:
    """Wąski, współdzielony state container dla runtime-hot settings."""

    _VALID_POLICY_MODES: frozenset[str] = frozenset({"shadow", "assist", "live"})

    def __init__(
        self,
        *,
        opportunity_ai_enabled: bool = True,
        manual_kill_switch: bool = False,
        execution_disabled: bool = False,
        policy_mode: str = "shadow",
    ) -> None:
        self._lock = threading.RLock()
        self._opportunity_ai_enabled = bool(opportunity_ai_enabled)
        self._manual_kill_switch = bool(manual_kill_switch)
        self._execution_disabled = bool(execution_disabled)
        self._policy_mode = self._normalize_policy_mode(policy_mode)
        self._revision = 0

    @classmethod
    def _normalize_policy_mode(cls, value: object) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in cls._VALID_POLICY_MODES:
            return normalized
        return "shadow"

    def snapshot(self) -> OpportunityRuntimeControlsSnapshot:
        with self._lock:
            return OpportunityRuntimeControlsSnapshot(
                opportunity_ai_enabled=self._opportunity_ai_enabled,
                manual_kill_switch=self._manual_kill_switch,
                execution_disabled=self._execution_disabled,
                policy_mode=self._policy_mode,
                revision=self._revision,
            )

    def update(
        self,
        *,
        opportunity_ai_enabled: bool | None = None,
        manual_kill_switch: bool | None = None,
        execution_disabled: bool | None = None,
        policy_mode: str | None = None,
    ) -> OpportunityRuntimeControlsSnapshot:
        with self._lock:
            changed = False
            if opportunity_ai_enabled is not None:
                next_enabled = bool(opportunity_ai_enabled)
                if next_enabled != self._opportunity_ai_enabled:
                    self._opportunity_ai_enabled = next_enabled
                    changed = True
            if manual_kill_switch is not None:
                next_kill = bool(manual_kill_switch)
                if next_kill != self._manual_kill_switch:
                    self._manual_kill_switch = next_kill
                    changed = True
            if execution_disabled is not None:
                next_execution_disabled = bool(execution_disabled)
                if next_execution_disabled != self._execution_disabled:
                    self._execution_disabled = next_execution_disabled
                    changed = True
            if policy_mode is not None:
                next_mode = self._normalize_policy_mode(policy_mode)
                if next_mode != self._policy_mode:
                    self._policy_mode = next_mode
                    changed = True
            if changed:
                self._revision += 1
            return self.snapshot()


_GLOBAL_RUNTIME_CONTROLS = OpportunityRuntimeControls()


def get_opportunity_runtime_controls() -> OpportunityRuntimeControls:
    """Zwraca współdzielony runtime settings control plane."""

    return _GLOBAL_RUNTIME_CONTROLS
