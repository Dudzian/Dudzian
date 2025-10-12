"""Obsługa zdarzeń decyzyjnych silnika ryzyka oraz logging JSONL."""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

from bot_core.security.signing import build_hmac_signature


def _to_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class RiskDecisionEvent:
    """Pojedyncza decyzja podjęta przez silnik ryzyka."""

    profile: str
    symbol: str
    side: str
    quantity: float
    price: float | None
    notional: float | None
    allowed: bool
    reason: str | None
    timestamp: datetime
    adjustments: Mapping[str, float] | None = None
    metadata: Mapping[str, object] | None = None
    signature: Mapping[str, str] | None = None

    def to_mapping(self) -> MutableMapping[str, object]:
        payload: MutableMapping[str, object] = {
            "profile": self.profile,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": float(self.quantity),
            "allowed": bool(self.allowed),
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
        }
        if self.price is not None:
            payload["price"] = float(self.price)
        if self.notional is not None:
            payload["notional"] = float(self.notional)
        if self.reason:
            payload["reason"] = self.reason
        if self.adjustments:
            payload["adjustments"] = {str(k): float(v) for k, v in self.adjustments.items()}
        if self.metadata:
            payload["metadata"] = {str(key): value for key, value in self.metadata.items()}
        if self.signature:
            payload["signature"] = dict(self.signature)
        return payload


class RiskDecisionLog:
    """Bufor decyzji silnika ryzyka z możliwością zapisu do JSONL."""

    def __init__(
        self,
        *,
        max_entries: int = 1_000,
        jsonl_path: str | Path | None = None,
        clock: Callable[[], datetime] = _utc_now,
        signing_key: bytes | None = None,
        signing_key_id: str | None = None,
        jsonl_fsync: bool = False,
    ) -> None:
        self._buffer: deque[RiskDecisionEvent] = deque(maxlen=max(1, int(max_entries)))
        self._lock = threading.Lock()
        self._path = Path(jsonl_path) if jsonl_path else None
        self._clock = clock
        self._signing_key = signing_key
        self._signing_key_id = signing_key_id
        self._jsonl_fsync = bool(jsonl_fsync)

    def append(self, event: RiskDecisionEvent) -> None:
        with self._lock:
            self._buffer.append(event)
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                payload = event.to_mapping()
                serialized = json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                )
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(serialized)
                    handle.write("\n")
                    handle.flush()
                    if self._jsonl_fsync:
                        try:
                            os.fsync(handle.fileno())
                        except OSError:
                            logging.getLogger(__name__).debug(
                                "Nie udało się zsynchronizować pliku risk decision logu",
                                exc_info=True,
                            )

    def record(
        self,
        *,
        profile: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None,
        notional: float | None,
        allowed: bool,
        reason: str | None,
        adjustments: Mapping[str, float] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> RiskDecisionEvent:
        event = RiskDecisionEvent(
            profile=profile,
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            price=_to_float(price),
            notional=_to_float(notional),
            allowed=bool(allowed),
            reason=reason,
            timestamp=self._clock(),
            adjustments=adjustments,
            metadata=metadata,
        )
        payload = event.to_mapping()
        if self._signing_key:
            signature = build_hmac_signature(
                payload,
                key=self._signing_key,
                key_id=self._signing_key_id,
            )
            event.signature = signature
        self.append(event)
        return event

    @property
    def path(self) -> Path | None:
        return self._path

    def tail(
        self,
        *,
        limit: int = 50,
        profile: str | None = None,
    ) -> Sequence[Mapping[str, object]]:
        if limit <= 0:
            return ()
        normalized_profile = profile.strip().lower() if isinstance(profile, str) else None
        with self._lock:
            candidates: Iterable[RiskDecisionEvent] = reversed(self._buffer)
            items: list[Mapping[str, object]] = []
            for event in candidates:
                if normalized_profile and event.profile.lower() != normalized_profile:
                    continue
                items.append(event.to_mapping())
                if len(items) >= limit:
                    break
        return tuple(reversed(items))

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


__all__ = ["RiskDecisionEvent", "RiskDecisionLog"]
