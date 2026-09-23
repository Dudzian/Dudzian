"""Deterministic Core time dependency available only through the explicit test path."""

from __future__ import annotations

from datetime import datetime

from bot_core.instruments.source_producer_membership import (
    _SQLiteMembershipCarrierBase,
    _SourceProducerMembershipAuthorityBase,
)


class TestSQLiteMembershipCarrier(_SQLiteMembershipCarrierBase):
    """Test-only durable carrier with a non-production digest domain."""

    __test__ = False
    _AUTHORITY_DOMAIN = "cryptohunter.source_producer_membership.test.v1"


class TestCoreClock:
    __test__ = False
    __slots__ = ("_now",)

    def __init__(self, now_utc: str) -> None:
        self.set_utc(now_utc)

    def set_utc(self, now_utc: str) -> None:
        if not isinstance(now_utc, str):
            raise ValueError("canonical UTC timestamp required")
        try:
            datetime.strptime(
                now_utc, "%Y-%m-%dT%H:%M:%SZ" if "." not in now_utc else "%Y-%m-%dT%H:%M:%S.%fZ"
            )
        except ValueError as exc:
            raise ValueError("canonical UTC timestamp required") from exc
        self._now = now_utc

    def now_utc(self) -> str:
        return self._now


class TestSourceProducerMembershipAuthority(_SourceProducerMembershipAuthorityBase):
    """Test-only authority; it is never a production authority runtime type."""

    __test__ = False
    __slots__ = ("_test_clock",)

    def __init__(self, carrier: TestSQLiteMembershipCarrier, clock: TestCoreClock) -> None:
        if type(carrier) is not TestSQLiteMembershipCarrier:
            raise TypeError("exact TestSQLiteMembershipCarrier required")
        if type(clock) is not TestCoreClock:
            raise TypeError("genuine TestCoreClock required")
        super().__init__(carrier)
        self._test_clock = clock

    def _now_utc(self) -> str:
        return self._test_clock.now_utc()


__all__ = [
    "TestCoreClock",
    "TestSQLiteMembershipCarrier",
    "TestSourceProducerMembershipAuthority",
]
