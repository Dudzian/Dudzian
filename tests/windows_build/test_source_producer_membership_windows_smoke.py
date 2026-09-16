"""Cross-platform smoke exercised by the official windows-latest build job."""

from bot_core.instruments.source_producer_membership import (
    SQLiteMembershipCarrier,
    SourceProducerMembershipAuthority,
)


def test_membership_grant_revoke_and_restart_on_windows_runner(tmp_path):
    path = tmp_path / "source-producer-membership.sqlite3"
    authority = SourceProducerMembershipAuthority(SQLiteMembershipCarrier(path))
    grant = authority.admit_release_grant("core_release_1_45_binance_spot", trusted_core_now_utc="2026-09-14T00:00:00Z")
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
    assert authority.admit_release_event("core_release_1_45_revoke_binance_spot", trusted_core_now_utc="2026-09-15T01:00:00Z")
    restored = SourceProducerMembershipAuthority(SQLiteMembershipCarrier(path))
    assert restored.resolve_current(identity, 1, "2026-09-15T03:00:00Z") is None
