"""Obsługa dynamicznych limitów sygnałów dla schedulera strategii."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Callable, Mapping, Sequence


@dataclass(slots=True)
class SignalLimitOverride:
    """Reprezentuje czasowe nadpisanie limitu sygnałów."""

    limit: int
    reason: str | None = None
    expires_at: datetime | None = None
    created_at: datetime | None = None

    def is_expired(self, now: datetime) -> bool:
        if self.expires_at is None:
            return False
        return now >= self.expires_at

    def remaining_seconds(self, now: datetime) -> float | None:
        if self.expires_at is None:
            return None
        return max(0.0, (self.expires_at - now).total_seconds())

    def to_snapshot(self, now: datetime) -> Mapping[str, object]:
        payload: dict[str, object] = {"limit": int(self.limit)}
        if self.reason:
            payload["reason"] = self.reason
        if self.created_at:
            payload["created_at"] = self.created_at.isoformat()
        if self.expires_at:
            payload["expires_at"] = self.expires_at.isoformat()
            remaining = self.remaining_seconds(now)
            if remaining is not None:
                payload["remaining_seconds"] = remaining
        payload["active"] = not self.is_expired(now)
        return payload


class SignalLimitManager:
    """Zarządza nadpisaniami limitów sygnałów."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime],
        logger: logging.Logger | None = None,
    ) -> None:
        self._clock = clock
        self._logger = logger or logging.getLogger(__name__)
        self._overrides: dict[tuple[str, str], SignalLimitOverride] = {}
        self._lock = RLock()

    def configure_limit(
        self,
        strategy_name: str,
        risk_profile: str,
        limit: object | None,
        *,
        reason: str | None = None,
        until: datetime | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        key = (strategy_name, risk_profile)
        with self._lock:
            if limit in (None, ""):
                self._overrides.pop(key, None)
                return
            override = self._normalize_signal_limit_override(
                limit,
                reason=reason,
                until=until,
                duration_seconds=duration_seconds,
            )
            if override is None:
                return
            if override.created_at is None:
                override.created_at = self._clock()
            self._overrides[key] = override

    def configure_limits(
        self,
        limits: Mapping[str, Mapping[str, object]],
        *,
        reason: str | None = None,
        until: datetime | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        for strategy, profiles in limits.items():
            for profile, limit in profiles.items():
                self.configure_limit(
                    strategy,
                    profile,
                    limit,
                    reason=reason,
                    until=until,
                    duration_seconds=duration_seconds,
                )

    def resolve_override(
        self,
        strategy_name: str,
        risk_profile: str,
        *,
        now: datetime | None = None,
    ) -> tuple[SignalLimitOverride | None, Mapping[tuple[str, str], SignalLimitOverride]]:
        moment = now or self._clock()
        with self._lock:
            expired = self._purge_expired(moment)
            override = self._overrides.get((strategy_name, risk_profile))
        if expired:
            self._log_expired(expired)
        return override, expired

    def active_overrides(
        self,
        *,
        now: datetime | None = None,
    ) -> tuple[Mapping[tuple[str, str], SignalLimitOverride], Mapping[tuple[str, str], SignalLimitOverride]]:
        moment = now or self._clock()
        with self._lock:
            expired = self._purge_expired(moment)
            active = dict(self._overrides)
        if expired:
            self._log_expired(expired)
        return active, expired

    def snapshot(
        self,
        *,
        now: datetime | None = None,
    ) -> Mapping[str, Mapping[str, Mapping[str, object]]]:
        moment = now or self._clock()
        snapshot: dict[str, dict[str, Mapping[str, object]]] = {}
        with self._lock:
            expired = self._purge_expired(moment)
            for (strategy, profile), override in self._overrides.items():
                strategy_entry = snapshot.setdefault(strategy, {})
                strategy_entry[profile] = dict(override.to_snapshot(moment))
        if expired:
            self._log_expired(expired)
        return snapshot

    def _normalize_signal_limit_override(
        self,
        limit: object,
        *,
        reason: str | None = None,
        until: datetime | None = None,
        duration_seconds: float | None = None,
    ) -> SignalLimitOverride | None:
        now = self._clock()
        resolved_reason: str | None = (reason or None)
        resolved_until: datetime | None = until
        resolved_duration: float | None = duration_seconds
        created_at: datetime | None = None
        if isinstance(limit, SignalLimitOverride):
            limit_value = limit.limit
            if resolved_reason is None and limit.reason:
                resolved_reason = limit.reason
            if resolved_until is None:
                resolved_until = limit.expires_at
            created_at = limit.created_at
        elif hasattr(limit, "limit") and not isinstance(limit, Mapping):
            limit_value = getattr(limit, "limit", None)
            try:
                limit_value = int(float(limit_value))
            except (TypeError, ValueError):
                return None
            if resolved_reason is None:
                reason_value = getattr(limit, "reason", None)
                if isinstance(reason_value, str):
                    resolved_reason = reason_value.strip() or None
                elif reason_value not in (None, ""):
                    resolved_reason = str(reason_value)
            if resolved_until is None:
                resolved_until = self._coerce_datetime(
                    getattr(limit, "until", None)
                    or getattr(limit, "expires_at", None)
                )
            if resolved_duration is None:
                resolved_duration = self._coerce_duration(
                    getattr(limit, "duration_seconds", None)
                    or getattr(limit, "duration", None)
                )
            created_at = self._coerce_datetime(getattr(limit, "created_at", None))
        elif isinstance(limit, Mapping):
            raw_limit = limit.get("limit", limit.get("value"))
            try:
                limit_value = int(float(raw_limit))
            except (TypeError, ValueError):
                return None
            if resolved_reason is None:
                reason_value = limit.get("reason")
                if isinstance(reason_value, str):
                    resolved_reason = reason_value.strip() or None
                elif reason_value not in (None, ""):
                    resolved_reason = str(reason_value)
            if resolved_until is None:
                resolved_until = self._coerce_datetime(
                    limit.get("until") or limit.get("expires_at")
                )
            if resolved_duration is None:
                resolved_duration = self._coerce_duration(
                    limit.get("duration_seconds") or limit.get("duration")
                )
            created_at = self._coerce_datetime(limit.get("created_at"))
        else:
            try:
                limit_value = int(limit)
            except (TypeError, ValueError):
                return None

        limit_value = max(0, int(limit_value))
        expiry = self._coerce_datetime(resolved_until)
        if expiry is None and resolved_duration not in (None, 0.0):
            try:
                seconds = float(resolved_duration)
            except (TypeError, ValueError):
                seconds = None
            if seconds is not None and math.isfinite(seconds) and seconds > 0.0:
                expiry = now + timedelta(seconds=seconds)

        reason_text = None
        if resolved_reason is not None:
            candidate = str(resolved_reason).strip()
            if candidate:
                reason_text = candidate

        created = self._coerce_datetime(created_at) or now
        return SignalLimitOverride(
            limit=limit_value,
            reason=reason_text,
            expires_at=expiry,
            created_at=created,
        )

    @staticmethod
    def _coerce_datetime(value: object | None) -> datetime | None:
        if value in (None, ""):
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                return None
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        return None

    @staticmethod
    def _coerce_duration(value: object | None) -> float | None:
        if value in (None, ""):
            return None
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(seconds) or seconds <= 0.0:
            return None
        return seconds

    def _purge_expired(
        self,
        now: datetime,
    ) -> Mapping[tuple[str, str], SignalLimitOverride]:
        expired: dict[tuple[str, str], SignalLimitOverride] = {}
        for key, override in list(self._overrides.items()):
            if override.is_expired(now):
                removed = self._overrides.pop(key, None)
                if removed is not None:
                    expired[key] = removed
        return expired

    def _log_expired(
        self,
        expired: Mapping[tuple[str, str], SignalLimitOverride],
    ) -> None:
        for (strategy, profile), override in expired.items():
            reason_part = f", powód: {override.reason}" if override.reason else ""
            expiry_part = (
                f", wygasło o {override.expires_at.isoformat()}"
                if override.expires_at
                else ""
            )
            self._logger.info(
                "Wygasło nadpisanie limitu sygnałów %s/%s (limit=%s%s%s)",
                strategy,
                profile,
                override.limit,
                reason_part,
                expiry_part,
            )

    def reapply_expired(
        self,
        expired: Mapping[tuple[str, str], SignalLimitOverride],
        schedules: Sequence[object],
        apply_callback: Callable[[object], None],
    ) -> None:
        if not expired:
            return
        affected = {key for key in expired}
        for schedule in schedules:
            strategy = getattr(schedule, "strategy_name", None)
            profile = getattr(schedule, "risk_profile", None)
            if (strategy, profile) in affected:
                apply_callback(schedule)

