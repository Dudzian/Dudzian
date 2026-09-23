"""Core-owned UTC authority used at durable admission boundaries."""

from __future__ import annotations

from datetime import datetime, timezone


class ProductionCoreClock:
    """Sealed production clock with no caller-configurable provider."""

    __slots__ = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ProductionCoreClock is sealed")

    def now_utc(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


PRODUCTION_CORE_CLOCK = ProductionCoreClock()

__all__ = ["ProductionCoreClock", "PRODUCTION_CORE_CLOCK"]
