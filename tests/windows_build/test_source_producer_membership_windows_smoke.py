"""Cross-platform smoke exercised by the official windows-latest build job."""

import pytest

from bot_core.instruments.source_producer_membership import (
    SQLiteMembershipCarrier,
    SourceProducerMembershipAuthority,
)
from bot_core.instruments.testing_core_time import (
    TestCoreClock,
    TestSQLiteMembershipCarrier,
    TestSourceProducerMembershipAuthority,
)


def test_membership_grant_revoke_and_restart_on_windows_runner(tmp_path):
    production = SourceProducerMembershipAuthority(
        SQLiteMembershipCarrier(tmp_path / "production-membership.sqlite3")
    )
    assert production is not None
    path = tmp_path / "source-producer-membership.sqlite3"
    clock = TestCoreClock("2026-09-14T00:00:00Z")
    authority = TestSourceProducerMembershipAuthority(TestSQLiteMembershipCarrier(path), clock)
    grant = authority.admit_release_grant("core_release_1_45_binance_spot")
    assert grant is not None
    identity = {
        name: getattr(grant, name)
        for name in (
            "source_exchange_id",
            "market_type",
            "source_adapter_family_id",
            "source_adapter_implementation_id",
            "source_adapter_release_id",
            "source_adapter_version",
        )
    }
    assert authority.resolve_current(identity, 1, "2026-09-15T01:00:00Z") == grant
    clock.set_utc("2026-09-15T01:00:00Z")
    assert authority.admit_release_event("core_release_1_45_revoke_binance_spot")
    restored = TestSourceProducerMembershipAuthority(TestSQLiteMembershipCarrier(path), clock)
    assert restored.resolve_current(identity, 1, "2026-09-15T03:00:00Z") is None
    with pytest.raises(ValueError, match="MEMBERSHIP_AUTHORITY_DOMAIN_MISMATCH"):
        SQLiteMembershipCarrier(path)
