from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class FailoverState:
    """Stan alertingu wykorzystywany do porównania failoveru."""

    last_error: str | None
    rules_digest: str | None = None
    firing_alerts: int | None = None


@dataclass(frozen=True)
class FailoverComparison:
    before: FailoverState
    after: FailoverState
    rehydrated: bool
    alerts_missing: bool
    regression: bool
    cleared: bool
    unchanged: bool
    rules_in_sync: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "before": self.before.__dict__,
            "after": self.after.__dict__,
            "rehydrated": self.rehydrated,
            "alertsMissing": self.alerts_missing,
            "regression": self.regression,
            "cleared": self.cleared,
            "unchanged": self.unchanged,
            "rulesInSync": self.rules_in_sync,
        }


def evaluate_outcome(
    comparison: FailoverComparison, *, before_ok: bool, after_ok: bool
) -> tuple[str, list[str]]:
    """Zwraca werdykt porównania failoveru wraz z listą przyczyn niepowodzenia."""

    reasons: list[str] = []
    if comparison.regression:
        reasons.append("regression_last_error")
    if not comparison.rules_in_sync:
        reasons.append("rules_drift")
    if comparison.alerts_missing:
        reasons.append("alerts_missing_after_rehydration")
    if not before_ok:
        reasons.append("before_probe_failed")
    if not after_ok:
        reasons.append("after_probe_failed")

    status = "pass" if not reasons else "fail"
    return status, reasons


def _normalize(state: Mapping[str, object] | FailoverState) -> FailoverState:
    if isinstance(state, FailoverState):
        return state
    return FailoverState(
        last_error=state.get("last_error") if isinstance(state, Mapping) else None,
        rules_digest=state.get("rules_digest") if isinstance(state, Mapping) else None,
        firing_alerts=state.get("firing_alerts") if isinstance(state, Mapping) else None,
    )


def compare_last_error(before: Mapping[str, object] | FailoverState, after: Mapping[str, object] | FailoverState) -> FailoverComparison:
    before_state = _normalize(before)
    after_state = _normalize(after)

    rehydrated = bool(before_state.last_error and after_state.last_error == before_state.last_error)
    alerts_missing = bool(
        (before_state.firing_alerts or 0) > 0
        and (after_state.firing_alerts or 0) == 0
        and rehydrated
    )
    regression = bool(
        after_state.last_error
        and (
            before_state.last_error is None
            or after_state.last_error != before_state.last_error
        )
    )
    cleared = bool(before_state.last_error and after_state.last_error is None)
    unchanged = before_state.last_error == after_state.last_error
    rules_in_sync = bool(
        before_state.rules_digest
        and after_state.rules_digest
        and before_state.rules_digest == after_state.rules_digest
    )

    return FailoverComparison(
        before=before_state,
        after=after_state,
        rehydrated=rehydrated,
        alerts_missing=alerts_missing,
        regression=regression,
        cleared=cleared,
        unchanged=unchanged,
        rules_in_sync=rules_in_sync,
    )


__all__ = [
    "FailoverState",
    "FailoverComparison",
    "compare_last_error",
    "evaluate_outcome",
]
