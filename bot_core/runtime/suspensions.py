"""Obsługa zawieszeń harmonogramów i tagów strategii."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Callable, Mapping, Sequence


@dataclass(slots=True)
class SuspensionRecord:
    reason: str
    applied_at: datetime
    until: datetime | None = None
    origin: str = "schedule"
    tag: str | None = None

    def is_active(self, now: datetime) -> bool:
        if self.until is None:
            return True
        return now < self.until

    def remaining_seconds(self, now: datetime) -> float | None:
        if self.until is None:
            return None
        return max(0.0, (self.until - now).total_seconds())

    def clone_for_tag(self, tag: str) -> "SuspensionRecord":
        return SuspensionRecord(
            reason=self.reason,
            applied_at=self.applied_at,
            until=self.until,
            origin="tag",
            tag=tag,
        )

    def as_dict(self, now: datetime) -> dict[str, object]:
        payload: dict[str, object] = {
            "reason": self.reason,
            "applied_at": self.applied_at.isoformat(),
            "origin": self.origin,
        }
        if self.until is not None:
            payload["until"] = self.until.isoformat()
            payload["remaining_seconds"] = self.remaining_seconds(now)
        if self.tag:
            payload["tag"] = self.tag
        return payload


class SuspensionManager:
    """Zarządza zawieszeniami harmonogramów oraz tagów."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime],
        logger: logging.Logger | None = None,
    ) -> None:
        self._clock = clock
        self._logger = logger or logging.getLogger(__name__)
        self._schedule_suspensions: dict[str, SuspensionRecord] = {}
        self._tag_suspensions: dict[str, SuspensionRecord] = {}
        self._active_reasons: dict[str, str] = {}
        self._lock = RLock()

    def suspend_schedule(
        self,
        schedule_name: str,
        *,
        reason: str | None = None,
        until: datetime | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        name = (schedule_name or "").strip()
        if not name:
            return
        reason_text = (reason or "manual").strip() or "manual"
        now = self._clock()
        expiry = self._resolve_suspension_expiry(now, until, duration_seconds)
        record = SuspensionRecord(reason=reason_text, applied_at=now, until=expiry)
        with self._lock:
            self._schedule_suspensions[name] = record
        self._logger.warning(
            "Zawieszono harmonogram %s z powodu: %s%s",
            name,
            reason_text,
            f" (do {expiry.isoformat()})" if expiry else "",
        )

    def resume_schedule(self, schedule_name: str) -> bool:
        name = (schedule_name or "").strip()
        if not name:
            return False
        with self._lock:
            removed = self._schedule_suspensions.pop(name, None) is not None
        if removed:
            self._logger.info("Wznowiono harmonogram %s", name)
            self._active_reasons.pop(name, None)
        return removed

    def suspend_tag(
        self,
        tag: str,
        *,
        reason: str | None = None,
        until: datetime | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        normalized = (tag or "").strip()
        if not normalized:
            return
        reason_text = (reason or "manual").strip() or "manual"
        now = self._clock()
        expiry = self._resolve_suspension_expiry(now, until, duration_seconds)
        record = SuspensionRecord(
            reason=reason_text,
            applied_at=now,
            until=expiry,
            origin="tag",
            tag=normalized,
        )
        with self._lock:
            self._tag_suspensions[normalized] = record
        self._logger.warning(
            "Zawieszono tag strategii %s z powodu: %s%s",
            normalized,
            reason_text,
            f" (do {expiry.isoformat()})" if expiry else "",
        )

    def resume_tag(self, tag: str) -> bool:
        normalized = (tag or "").strip()
        if not normalized:
            return False
        with self._lock:
            removed = self._tag_suspensions.pop(normalized, None) is not None
        if removed:
            self._logger.info("Wznowiono tag strategii %s", normalized)
        return removed

    def snapshot(self) -> Mapping[str, Mapping[str, object]]:
        now = self._clock()
        self._purge_expired(now)
        schedules: dict[str, dict[str, object]] = {}
        tags: dict[str, dict[str, object]] = {}
        schedule_reasons: dict[str, str] = {}
        tag_reasons: dict[str, str] = {}
        next_expiration_info: tuple[str, str, float] | None = None
        with self._lock:
            for name, record in self._schedule_suspensions.items():
                schedules[name] = record.as_dict(now)
                descriptor = self._active_reasons.get(name, record.reason)
                schedule_reasons[name] = descriptor
                remaining = record.remaining_seconds(now)
                if remaining is not None:
                    if (
                        next_expiration_info is None
                        or remaining < next_expiration_info[2]
                    ):
                        next_expiration_info = ("schedule", name, remaining)
            for tag_name, record in self._tag_suspensions.items():
                tags[tag_name] = record.as_dict(now)
                tag_reasons[tag_name] = record.reason
                remaining = record.remaining_seconds(now)
                if remaining is not None:
                    if (
                        next_expiration_info is None
                        or remaining < next_expiration_info[2]
                    ):
                        next_expiration_info = ("tag", tag_name, remaining)
        counts = {
            "schedules": len(schedules),
            "tags": len(tags),
            "total": len(schedules) + len(tags),
        }
        reasons = {"schedules": schedule_reasons, "tags": tag_reasons}
        payload = {
            "schedules": schedules,
            "tags": tags,
            "counts": counts,
            "reasons": reasons,
        }
        if next_expiration_info is not None:
            scope, name, remaining = next_expiration_info
            payload["next_expiration"] = {
                "scope": scope,
                "name": name,
                "remaining_seconds": remaining,
            }
        return payload

    def resolve(
        self,
        schedule_name: str,
        tags: Sequence[str],
        primary_tag: str | None,
        timestamp: datetime,
    ) -> SuspensionRecord | None:
        self._purge_expired(timestamp)
        record: SuspensionRecord | None = None
        with self._lock:
            record = self._schedule_suspensions.get(schedule_name)
            if record is None:
                for tag in tags:
                    tag_record = self._tag_suspensions.get(tag)
                    if tag_record is not None:
                        record = tag_record.clone_for_tag(tag)
                        break
                else:
                    if primary_tag and primary_tag not in tags:
                        tag_record = self._tag_suspensions.get(primary_tag)
                        if tag_record is not None:
                            record = tag_record.clone_for_tag(primary_tag)
        self._update_state(schedule_name, record)
        return record

    def _purge_expired(self, now: datetime) -> None:
        expired_schedules: list[tuple[str, SuspensionRecord]] = []
        expired_tags: list[tuple[str, SuspensionRecord]] = []
        with self._lock:
            for name, record in list(self._schedule_suspensions.items()):
                if not record.is_active(now):
                    expired_schedules.append((name, record))
                    self._schedule_suspensions.pop(name, None)
            for tag_name, record in list(self._tag_suspensions.items()):
                if not record.is_active(now):
                    expired_tags.append((tag_name, record))
                    self._tag_suspensions.pop(tag_name, None)
        for name, record in expired_schedules:
            descriptor = self._active_reasons.pop(name, None) or record.reason
            self._logger.info(
                "Harmonogram %s automatycznie wznowiony po wygaśnięciu zawieszenia (%s)",
                name,
                descriptor,
            )
        for tag_name, record in expired_tags:
            descriptor = record.reason
            self._logger.info(
                "Tag strategii %s automatycznie wznowiony po wygaśnięciu zawieszenia (%s)",
                tag_name,
                descriptor,
            )

    def _update_state(self, schedule_name: str, record: SuspensionRecord | None) -> None:
        descriptor: str | None = None
        if record is not None:
            descriptor = record.reason
            if record.origin == "tag" and record.tag:
                descriptor = f"{descriptor} [tag={record.tag}]"

        previous = self._active_reasons.get(schedule_name)
        if record is None:
            if previous is not None:
                self._logger.info(
                    "Harmonogram %s wznowiony po zawieszeniu (%s)",
                    schedule_name,
                    previous,
                )
                self._active_reasons.pop(schedule_name, None)
            return

        if descriptor is None:
            descriptor = "manual"

        if previous != descriptor:
            if previous is None:
                self._logger.warning(
                    "Harmonogram %s przechodzi w stan zawieszenia: %s",
                    schedule_name,
                    descriptor,
                )
            else:
                self._logger.warning(
                    "Harmonogram %s zmienia powód zawieszenia: %s -> %s",
                    schedule_name,
                    previous,
                    descriptor,
                )
            self._active_reasons[schedule_name] = descriptor

    def _resolve_suspension_expiry(
        self,
        now: datetime,
        until: datetime | None,
        duration_seconds: float | None,
    ) -> datetime | None:
        if until is not None:
            if until.tzinfo is None:
                return until.replace(tzinfo=timezone.utc)
            return until.astimezone(timezone.utc)
        if duration_seconds is None:
            return None
        try:
            seconds = float(duration_seconds)
        except (TypeError, ValueError):
            return None
        if seconds <= 0:
            return None
        return now + timedelta(seconds=seconds)

