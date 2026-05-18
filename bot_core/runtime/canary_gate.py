"""Pure helper contract for future runtime canary-gate enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import Any, Mapping

LIVE_AUTONOMOUS_MODE = "live_autonomous"

REASON_CANARY_MISSING = "accepted_autonomous_canary_contract_missing"
REASON_CANARY_NOT_APPROVED = "accepted_autonomous_canary_contract_not_approved"
REASON_CANARY_REPORT_ONLY = "accepted_autonomous_canary_contract_report_only"
REASON_EXCHANGE_NOT_ALLOWED = "accepted_autonomous_canary_exchange_not_allowed"
REASON_SYMBOL_NOT_ALLOWED = "accepted_autonomous_canary_symbol_not_allowed"
REASON_ORDER_NOTIONAL_EXCEEDED = "accepted_autonomous_canary_order_notional_exceeded"
REASON_POSITION_NOTIONAL_EXCEEDED = "accepted_autonomous_canary_position_notional_exceeded"
REASON_CANARY_EXPIRED = "accepted_autonomous_canary_expired"

_CANARY_SNAPSHOT_FIELDS = (
    "report_only",
    "canary_status",
    "canary_profile_id",
    "allowed_exchanges",
    "allowed_symbols",
    "max_order_notional",
    "max_position_notional",
    "review_required_at",
    "expires_at",
)


@dataclass(frozen=True, slots=True)
class CanaryGateDecision:
    """Normalized allow/block result for a runtime canary contract check."""

    allowed: bool
    blocking_reason: str | None
    canary_status: str | None
    canary_profile_id: str | None
    snapshot: dict[str, object]


def evaluate_runtime_canary_gate(
    *,
    canary_contract: Mapping[str, object] | None,
    mode: str,
    exchange: str | None,
    symbol: str | None,
    order_notional: float | None,
    position_notional: float | None = None,
    is_reduce_risk: bool = False,
    now: datetime | None = None,
) -> CanaryGateDecision:
    """Evaluate the future runtime canary gate without mutating caller input.

    The helper intentionally does not wire itself into live order flow.  It only
    defines the fail-closed contract that a future TradingController hook can use
    after building an order request and before pre-trade risk checks.
    """

    snapshot = _build_snapshot(canary_contract)
    canary_status = _as_optional_string(snapshot.get("canary_status"))
    canary_profile_id = _as_optional_string(snapshot.get("canary_profile_id"))

    if _normalize_mode(mode) != LIVE_AUTONOMOUS_MODE or is_reduce_risk:
        return _decision(
            allowed=True,
            blocking_reason=None,
            canary_status=canary_status,
            canary_profile_id=canary_profile_id,
            snapshot=snapshot,
        )

    if not snapshot.get("present"):
        return _blocked(REASON_CANARY_MISSING, canary_status, canary_profile_id, snapshot)

    if snapshot.get("report_only") is True:
        return _blocked(REASON_CANARY_REPORT_ONLY, canary_status, canary_profile_id, snapshot)

    if _normalize_status(canary_status) != "approved":
        return _blocked(REASON_CANARY_NOT_APPROVED, canary_status, canary_profile_id, snapshot)

    allowed_exchanges = _normalize_string_list(snapshot.get("allowed_exchanges"), case="lower")
    if allowed_exchanges and _normalize_exchange(exchange) not in allowed_exchanges:
        return _blocked(REASON_EXCHANGE_NOT_ALLOWED, canary_status, canary_profile_id, snapshot)

    allowed_symbols = _normalize_string_list(snapshot.get("allowed_symbols"), case="upper")
    if allowed_symbols and _normalize_symbol(symbol) not in allowed_symbols:
        return _blocked(REASON_SYMBOL_NOT_ALLOWED, canary_status, canary_profile_id, snapshot)

    max_order_notional = _as_float(snapshot.get("max_order_notional"))
    if _exceeds_limit(order_notional, max_order_notional):
        return _blocked(REASON_ORDER_NOTIONAL_EXCEEDED, canary_status, canary_profile_id, snapshot)

    max_position_notional = _as_float(snapshot.get("max_position_notional"))
    if _exceeds_limit(position_notional, max_position_notional):
        return _blocked(
            REASON_POSITION_NOTIONAL_EXCEEDED, canary_status, canary_profile_id, snapshot
        )

    evaluation_time = _normalize_datetime(now or datetime.now(UTC))
    if _is_past_due(snapshot.get("review_required_at"), evaluation_time) or _is_past_due(
        snapshot.get("expires_at"), evaluation_time
    ):
        return _blocked(REASON_CANARY_EXPIRED, canary_status, canary_profile_id, snapshot)

    return _decision(
        allowed=True,
        blocking_reason=None,
        canary_status=canary_status,
        canary_profile_id=canary_profile_id,
        snapshot=snapshot,
    )


def _decision(
    *,
    allowed: bool,
    blocking_reason: str | None,
    canary_status: str | None,
    canary_profile_id: str | None,
    snapshot: Mapping[str, object],
) -> CanaryGateDecision:
    return CanaryGateDecision(
        allowed=allowed,
        blocking_reason=blocking_reason,
        canary_status=canary_status,
        canary_profile_id=canary_profile_id,
        snapshot=dict(snapshot),
    )


def _blocked(
    reason: str,
    canary_status: str | None,
    canary_profile_id: str | None,
    snapshot: Mapping[str, object],
) -> CanaryGateDecision:
    return _decision(
        allowed=False,
        blocking_reason=reason,
        canary_status=canary_status,
        canary_profile_id=canary_profile_id,
        snapshot=snapshot,
    )


def _build_snapshot(canary_contract: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(canary_contract, Mapping):
        return {"present": False}

    snapshot: dict[str, object] = {"present": True}
    for field in _CANARY_SNAPSHOT_FIELDS:
        if field not in canary_contract:
            continue
        snapshot[field] = _snapshot_value(canary_contract[field])
    return snapshot


def _snapshot_value(value: object) -> object:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, tuple | list):
        return [_snapshot_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _snapshot_value(item) for key, item in value.items()}
    return str(value)


def _normalize_mode(value: object) -> str:
    return str(value or "").strip().lower()


def _normalize_status(value: object) -> str:
    return str(value or "").strip().lower()


def _normalize_exchange(value: object) -> str:
    return str(value or "").strip().lower()


def _normalize_symbol(value: object) -> str:
    return str(value or "").strip().upper()


def _normalize_string_list(value: object, *, case: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_items = (value,)
    elif isinstance(value, tuple | list | set | frozenset):
        raw_items = tuple(value)
    else:
        return ()

    normalized: list[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if not text:
            continue
        normalized.append(text.lower() if case == "lower" else text.upper())
    return tuple(normalized)


def _as_optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _exceeds_limit(value: float | None, limit: float | None) -> bool:
    if value is None or limit is None:
        return False
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return False
    return value_float > limit


def _is_past_due(value: object, now: datetime) -> bool:
    due_at = _parse_datetime(value)
    if due_at is None:
        return False
    return due_at < now


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    if isinstance(value, date):
        return _normalize_datetime(datetime.combine(value, time.min))
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return _normalize_datetime(parsed)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
