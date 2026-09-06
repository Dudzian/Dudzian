"""Process-local RuntimeSession identity owned by CoreHost."""

from __future__ import annotations

import secrets
import time
import uuid
from dataclasses import dataclass, field


def new_runtime_session_id() -> str:
    """Mint a lowercase ``run_<uuidv7>`` identity under the M0.2 policy."""

    timestamp = int(time.time_ns() // 1_000_000) & ((1 << 48) - 1)
    random_bits = secrets.randbits(74)
    value = (
        (timestamp << 80)
        | (0x7 << 76)
        | (((random_bits >> 62) & 0xFFF) << 64)
        | (0b10 << 62)
        | (random_bits & ((1 << 62) - 1))
    )
    return f"run_{uuid.UUID(int=value)}"


@dataclass(slots=True)
class RuntimeSession:
    """The active process-local session; durable history is not restored here."""

    runtime_session_id: str
    device_installation_id: str
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True


def create_runtime_session(device_installation_id: str) -> RuntimeSession:
    return RuntimeSession(new_runtime_session_id(), device_installation_id)


__all__ = ["RuntimeSession", "create_runtime_session", "new_runtime_session_id"]
