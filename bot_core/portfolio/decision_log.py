"""Dziennik decyzji PortfolioGovernora z podpisami HMAC."""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

from bot_core.security.signing import build_hmac_signature

if TYPE_CHECKING:
    from .governor import PortfolioDecision


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_value(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Mapping):
        return {str(k): _normalize_value(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_value(item) for item in value]
    return str(value)


def _normalize_metadata(metadata: Mapping[str, object] | None) -> Mapping[str, object]:
    if not metadata:
        return {}
    return {str(key): _normalize_value(value) for key, value in metadata.items()}


class PortfolioDecisionLog:
    """Bufor decyzji portfelowych zapisujący wpisy JSONL z podpisem HMAC."""

    def __init__(
        self,
        *,
        max_entries: int = 512,
        jsonl_path: str | Path | None = None,
        clock: Callable[[], datetime] = _utc_now,
        signing_key: bytes | None = None,
        signing_key_id: str | None = None,
        jsonl_fsync: bool = False,
    ) -> None:
        self._buffer: deque[Mapping[str, object]] = deque(maxlen=max(1, int(max_entries)))
        self._lock = threading.Lock()
        self._path = Path(jsonl_path) if jsonl_path else None
        self._clock = clock
        self._signing_key = signing_key
        self._signing_key_id = signing_key_id
        self._jsonl_fsync = bool(jsonl_fsync)

    @property
    def path(self) -> Path | None:
        return self._path

    def record(
        self,
        decision: "PortfolioDecision",
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        payload = decision.to_dict()
        payload.setdefault("type", "portfolio_decision")
        normalized_metadata = _normalize_metadata(metadata)
        if normalized_metadata:
            payload["metadata"] = normalized_metadata
        payload.setdefault("logged_at", self._clock().isoformat())
        if self._signing_key:
            payload["signature"] = build_hmac_signature(
                payload, key=self._signing_key, key_id=self._signing_key_id
            )
        self._append(payload)
        return payload

    def tail(self, *, limit: int = 20) -> Sequence[Mapping[str, object]]:
        if limit <= 0:
            return ()
        with self._lock:
            items = list(self._buffer)
        return items[-limit:]

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def _append(self, payload: Mapping[str, object]) -> None:
        with self._lock:
            self._buffer.append(dict(payload))
            if self._path is None:
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.write("\n")
                handle.flush()
                if self._jsonl_fsync:
                    try:
                        os.fsync(handle.fileno())
                    except OSError:  # pragma: no cover - diagnostyka IO
                        logging.getLogger(__name__).debug(
                            "Nie udało się zsynchronizować pliku decision logu portfela",
                            exc_info=True,
                        )


__all__ = ["PortfolioDecisionLog"]

