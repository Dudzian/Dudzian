"""Red-team regressions for the genuine M0.12 producer authority boundary."""
from copy import deepcopy
import inspect
import json
import sqlite3
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

import pytest

from bot_core.instruments.catalog_projection_oracle import (
    SOURCE_FINGERPRINT_FIELDS, _fingerprint,
    validate_accepted_source_snapshot_membership_evidence,
    validate_canonical_catalog_context_graph,
)
from bot_core.instruments.source_producer_membership import (
    JsonlMembershipCarrier, SourceProducerMembershipAuthority as _Authority,
    membership_fingerprint, validate_source_producer_membership,
)
from bot_core.instruments.core_time import ProductionCoreClock
from bot_core.instruments.testing_core_time import (
    TestCoreClock,
    TestSQLiteMembershipCarrier,
    TestSourceProducerMembershipAuthority,
)

_TEST_CLOCK = TestCoreClock("2026-09-14T00:00:00Z")


@pytest.fixture(autouse=True)
def _deterministic_production_clock(monkeypatch):
    _TEST_CLOCK.set_utc("2026-09-14T00:00:00Z")
    monkeypatch.setattr(ProductionCoreClock, "now_utc", lambda self: _TEST_CLOCK.now_utc())


def SourceProducerMembershipAuthority(carrier):
    return _Authority(carrier)


def make_authority(tmp_path, name="authority"):
    authority = SourceProducerMembershipAuthority(JsonlMembershipCarrier(tmp_path / f"{name}.jsonl"))
    grant = authority.admit_release_grant("core_release_1_45_binance_spot")
    assert grant is not None
    return authority, grant


def _process_admit_same_grant(path, barrier, results):
    authority = SourceProducerMembershipAuthority(JsonlMembershipCarrier(path))
    barrier.wait()
    results.put(authority.admit_release_grant("core_release_1_45_binance_spot") is not None)


def identity(grant):
    return {name: getattr(grant, name) for name in (
        "source_exchange_id", "market_type", "source_adapter_family_id",
        "source_adapter_implementation_id", "source_adapter_release_id",
        "source_adapter_version",
    )}


def snapshot(grant, accepted="2026-09-15T01:00:00Z"):
    value = {
        "accepted_source_catalog_snapshot_id": "ascat_1", **identity(grant),
        "accepted_source_producer_membership_id": grant.accepted_source_producer_membership_id,
        "source_producer_generation": grant.producer_generation,
        "source_producer_membership_fingerprint": grant.content_fingerprint,
        "upstream_snapshot_or_retrieval_id": "exchangeInfo:42",
        "observed_at_utc": accepted, "effective_at_utc": accepted,
        "stale_after_utc": "2026-09-16T00:00:00Z", "previous_snapshot_id": None,
        "completeness_status": "COMPLETE", "completeness_evidence": {"pages": 1},
        "acceptance_status": "VALID",
        "member_source_product_metadata_versions": [{"source_exchange_id": "binance", "market_type": "SPOT", "venue_symbol": "BTCUSDT", "source_metadata_version_id": "meta_1"}],
        "content_fingerprint": "",
    }
    value["content_fingerprint"] = _fingerprint("cryptohunter.m0.5.accepted_source_catalog_snapshot.v2", SOURCE_FINGERPRINT_FIELDS, value)
    return value


def test_admission_api_has_no_caller_time_parameter():
    for method in (_Authority.admit_release_grant, _Authority.admit_release_event):
        parameters = inspect.signature(method).parameters
        assert "trusted_core_now_utc" not in parameters
        assert "now_utc" not in parameters
        assert "admitted_at_utc" not in parameters


def test_production_authority_has_non_swappable_clock_provenance(tmp_path):
    clock = TestCoreClock("2000-01-01T00:00:00Z")
    with pytest.raises(TypeError):
        _Authority(clock)
    production = _Authority(JsonlMembershipCarrier(tmp_path / "production.sqlite3"))
    assert type(production) is _Authority
    assert type(TestSourceProducerMembershipAuthority(
        TestSQLiteMembershipCarrier(tmp_path / "test.sqlite3"), clock
    )) is not _Authority
    with pytest.raises(AttributeError):
        production._trusted_clock = clock


@pytest.mark.parametrize(
    "untrusted",
    [
        {"now_utc": "2026-09-15T02:00:00Z"},
        {"admission_time": "2026-09-15T02:00:00Z"},
        {"trusted_core_now_utc": "2026-09-15T02:00:00Z"},
        {"time": "2026-09-15T02:00:00Z"},
    ],
)
def test_operation_config_plugin_and_adapter_time_inputs_cannot_enter_api(tmp_path, untrusted):
    authority = SourceProducerMembershipAuthority(JsonlMembershipCarrier(tmp_path / "isolation.sqlite3"))
    _TEST_CLOCK.set_utc("2026-09-15T04:00:00Z")
    with pytest.raises(TypeError):
        authority.admit_release_grant("core_release_1_45_binance_spot", **untrusted)
    grant = authority.admit_release_grant("core_release_1_45_binance_spot")
    assert grant is not None and grant.authority_admitted_at_utc == "2026-09-15T04:00:00Z"


def test_backdated_grant_and_event_are_api_impossible_and_clock_wins(tmp_path):
    authority = SourceProducerMembershipAuthority(JsonlMembershipCarrier(tmp_path / "backdate.sqlite3"))
    _TEST_CLOCK.set_utc("2026-09-15T04:00:00Z")
    with pytest.raises(TypeError):
        authority.admit_release_grant(
            "core_release_1_45_binance_spot", trusted_core_now_utc="2026-09-15T02:00:00Z"
        )
    grant = authority.admit_release_grant("core_release_1_45_binance_spot")
    assert grant is not None and grant.authority_admitted_at_utc == "2026-09-15T04:00:00Z"
    with pytest.raises(TypeError):
        authority.admit_release_event(
            "core_release_1_45_revoke_binance_spot", admitted_at_utc="2026-09-15T03:00:00Z"
        )
    event = authority.admit_release_event("core_release_1_45_revoke_binance_spot")
    assert event is not None and event.authority_admitted_at_utc == "2026-09-15T04:00:00Z"

    _TEST_CLOCK.set_utc("2026-09-16T00:00:00Z")
    assert authority.admit_release_grant("core_release_1_45_binance_spot") == grant


def test_public_resolver_boundary_does_not_accept_history_mapping(tmp_path):
    authority, grant = make_authority(tmp_path)
    assert "history" not in inspect.signature(authority.resolve_current).parameters
    assert "history" not in inspect.signature(authority.resolve_historical).parameters
    fake = grant.to_mapping()
    fake.update(accepted_source_producer_membership_id="aspm_evil", core_admission_evidence="evil")
    fake["content_fingerprint"] = membership_fingerprint(fake)
    assert validate_source_producer_membership(fake)  # integrity is deliberately public
    assert authority.resolve_current(identity(grant), 1, "2026-09-15T01:00:00Z") == grant
    assert authority.resolve_historical("aspm_evil", 1, fake["content_fingerprint"], identity(grant), "2026-09-15T01:00:00Z") is None
    assert authority.admit_release_grant(fake) is None


def test_append_only_revoke_preserves_grant_and_old_snapshot(tmp_path):
    authority, grant = make_authority(tmp_path, "revoke")
    old = snapshot(grant)
    with sqlite3.connect(authority._carrier.path) as connection:
        before = connection.execute("SELECT COUNT(*) FROM membership_authority_records").fetchone()[0]
    assert validate_accepted_source_snapshot_membership_evidence(old, authority)
    assert authority.admit_release_event("core_release_1_45_revoke_binance_spot") is not None
    with sqlite3.connect(authority._carrier.path) as connection:
        after = connection.execute("SELECT COUNT(*) FROM membership_authority_records").fetchone()[0]
    assert after == before + 1
    restored = SourceProducerMembershipAuthority(JsonlMembershipCarrier(authority._carrier.path))
    assert restored.resolve_current(identity(grant), 1, "2026-09-15T02:30:00Z") is None
    assert validate_accepted_source_snapshot_membership_evidence(old, restored)
    assert old["source_producer_membership_fingerprint"] == grant.content_fingerprint
    late = snapshot(grant, "2026-09-15T02:30:00Z")
    assert not validate_accepted_source_snapshot_membership_evidence(late, restored)


def test_append_only_supersession_derives_generation_two_current(tmp_path):
    authority, grant1 = make_authority(tmp_path, "supersede")
    old = snapshot(grant1)
    grant2 = authority.admit_release_grant("core_release_1_45_binance_spot_generation_2")
    assert grant2 is not None
    assert authority.resolve_current(identity(grant1), 1, "2026-09-15T02:00:00Z") is None  # ambiguity fails closed until terminal event
    assert authority.admit_release_event("core_release_1_45_supersede_binance_spot") is not None
    restored = SourceProducerMembershipAuthority(JsonlMembershipCarrier(authority._carrier.path))
    assert restored.resolve_current(identity(grant1), 1, "2026-09-15T02:30:00Z") is None
    assert restored.resolve_current(identity(grant2), 2, "2026-09-15T02:30:00Z") == grant2
    assert validate_accepted_source_snapshot_membership_evidence(old, restored)


@pytest.mark.parametrize("claim", [{"trusted": True}, {"factory": "fake"}, {"plugin_manifest": {}}, {"implementation_id": "impl_ccxt_binance"}, {"alias": "binance_adapter"}])
def test_runtime_plugin_config_factory_and_self_description_cannot_admit(tmp_path, claim):
    authority, _ = make_authority(tmp_path)
    assert authority.admit_release_grant(claim) is None
    assert authority.admit_release_event(claim) is None


@pytest.mark.parametrize("field,wrong", [("source_exchange_id", "kraken"), ("market_type", "PERPETUAL"), ("source_adapter_family_id", "other"), ("source_adapter_implementation_id", "impl_B"), ("source_adapter_release_id", "release_B"), ("source_adapter_version", "4.5.2")])
def test_exact_identity_claim_mismatch_denied(tmp_path, field, wrong):
    authority, grant = make_authority(tmp_path)
    claimant = identity(grant); claimant[field] = wrong
    assert authority.resolve_current(claimant, 1, "2026-09-15T01:00:00Z") is None


@pytest.mark.parametrize("bad", [None, [], {}, {"extra": True}, float("nan"), True, 0, -1])
def test_public_entrypoints_are_total(tmp_path, bad):
    authority, grant = make_authority(tmp_path)
    assert validate_source_producer_membership(bad) is False
    assert authority.resolve_current(bad, bad, bad) is None
    assert authority.resolve_historical(bad, bad, bad, bad, bad) is None
    assert validate_accepted_source_snapshot_membership_evidence(bad, authority) is False
    malformed = grant.to_mapping(); malformed["producer_generation"] = bad
    assert validate_source_producer_membership(malformed) is False


def corrupt(path, statement, parameters=()):
    with sqlite3.connect(path) as connection:
        connection.execute(statement, parameters)
        connection.commit()


def assert_corrupt(path):
    with pytest.raises((ValueError, sqlite3.DatabaseError), match="CORRUPT_MEMBERSHIP_JOURNAL|malformed|database"):
        SourceProducerMembershipAuthority(JsonlMembershipCarrier(path))


def test_clean_acknowledged_revoke_tail_deletion_is_rollback(tmp_path):
    authority, _ = make_authority(tmp_path, "revoke_rollback")
    authority.admit_release_event("core_release_1_45_revoke_binance_spot")
    corrupt(authority._carrier.path, "DELETE FROM membership_authority_records WHERE journal_sequence = 2")
    assert_corrupt(authority._carrier.path)


def test_clean_acknowledged_supersession_tail_deletion_is_rollback(tmp_path):
    authority, _ = make_authority(tmp_path, "supersede_rollback")
    authority.admit_release_grant("core_release_1_45_binance_spot_generation_2")
    authority.admit_release_event("core_release_1_45_supersede_binance_spot")
    corrupt(authority._carrier.path, "DELETE FROM membership_authority_records WHERE journal_sequence = 3")
    assert_corrupt(authority._carrier.path)


@pytest.mark.parametrize(
    "statement,parameters",
    [
        ("DELETE FROM membership_authority_records WHERE journal_sequence = 2", ()),
        ("UPDATE membership_authority_records SET journal_sequence = 9 WHERE journal_sequence = 2", ()),
        ("UPDATE membership_authority_records SET record_digest = ? WHERE journal_sequence = 2", ("0" * 64,)),
        ("UPDATE membership_authority_records SET previous_record_digest = ? WHERE journal_sequence = 2", ("f" * 64,)),
        ("UPDATE membership_authority_head SET committed_head_digest = ? WHERE singleton = 1", ("0" * 64,)),
        ("UPDATE membership_authority_head SET committed_sequence = 2 WHERE singleton = 1", ()),
    ],
)
def test_missing_middle_reorder_duplicate_sequence_and_digest_corruption_fail_closed(
    tmp_path, statement, parameters
):
    authority, _ = make_authority(tmp_path, "row_corruption")
    authority.admit_release_grant("core_release_1_45_binance_spot_generation_2")
    authority.admit_release_event("core_release_1_45_supersede_binance_spot")
    corrupt(authority._carrier.path, statement, parameters)
    assert_corrupt(authority._carrier.path)


@pytest.mark.parametrize("mutation", ["generation", "event_target", "malformed"])
def test_canonical_record_mutation_and_malformed_content_fail_closed(tmp_path, mutation):
    authority, _ = make_authority(tmp_path, "content_corruption")
    authority.admit_release_grant("core_release_1_45_binance_spot_generation_2")
    authority.admit_release_event("core_release_1_45_supersede_binance_spot")
    with sqlite3.connect(authority._carrier.path) as connection:
        sequence = 2 if mutation == "generation" else 3
        raw = connection.execute(
            "SELECT canonical_record FROM membership_authority_records WHERE journal_sequence = ?",
            (sequence,),
        ).fetchone()[0]
        if mutation == "malformed":
            changed = "{"
        else:
            changed_record = json.loads(raw)
            if mutation == "generation":
                changed_record["producer_generation"] = 1
            else:
                changed_record["membership_id"] = "aspm_evil"
            changed = json.dumps(changed_record, sort_keys=True, separators=(",", ":"))
        connection.execute(
            "UPDATE membership_authority_records SET canonical_record = ? WHERE journal_sequence = ?",
            (changed, sequence),
        )
        connection.commit()
    assert_corrupt(authority._carrier.path)


@pytest.mark.parametrize(
    "field,wrong",
    [
        ("accepted_source_producer_membership_id", "aspm_missing"),
        ("source_producer_membership_fingerprint", "f" * 64),
        ("source_producer_generation", 2),
        ("source_adapter_family_id", "evil_family"),
        ("source_adapter_implementation_id", "evil_impl"),
        ("source_adapter_release_id", "evil_release"),
        ("source_adapter_version", "999.0"),
    ],
)
def test_canonical_global_graph_requires_exact_genuine_membership(tmp_path, field, wrong):
    from tests.architecture.test_m06_workspace_catalog_source_chain import canonical_graph

    authority, _ = make_authority(tmp_path)
    source, projection, instrument, _, _ = canonical_graph()
    source[field] = wrong
    source["content_fingerprint"] = _fingerprint(
        "cryptohunter.m0.5.accepted_source_catalog_snapshot.v2",
        SOURCE_FINGERPRINT_FIELDS,
        source,
    )
    assert not validate_canonical_catalog_context_graph(
        workspace_catalog_projections_by_id={"wcat_1": projection},
        accepted_source_catalog_snapshots_by_id={"ascat_1": source},
        instruments_by_id={"instr_1": instrument},
        instrument_history_by_id={},
        source_producer_membership_authority=authority,
    )


def test_every_unrelated_snapshot_is_membership_validated(tmp_path):
    from tests.architecture.test_m06_workspace_catalog_source_chain import canonical_graph

    authority, _ = make_authority(tmp_path)
    source, projection, instrument, _, _ = canonical_graph()
    unrelated = deepcopy(source)
    unrelated["accepted_source_catalog_snapshot_id"] = "ascat_unrelated"
    unrelated["accepted_source_producer_membership_id"] = "aspm_evil"
    unrelated["content_fingerprint"] = _fingerprint(
        "cryptohunter.m0.5.accepted_source_catalog_snapshot.v2",
        SOURCE_FINGERPRINT_FIELDS,
        unrelated,
    )
    assert not validate_canonical_catalog_context_graph(
        workspace_catalog_projections_by_id={"wcat_1": projection},
        accepted_source_catalog_snapshots_by_id={"ascat_1": source, "ascat_unrelated": unrelated},
        instruments_by_id={"instr_1": instrument},
        instrument_history_by_id={},
        source_producer_membership_authority=authority,
    )


@pytest.mark.parametrize("fake", [None, object(), {}, [], "fake"])
def test_empty_canonical_graph_still_requires_genuine_authority(tmp_path, fake):
    with patch.object(ProductionCoreClock, "now_utc", return_value="2026-09-14T00:00:00Z"):
        authority = _Authority(JsonlMembershipCarrier(tmp_path / "production-empty.sqlite3"))
    kwargs = {
        "workspace_catalog_projections_by_id": {},
        "accepted_source_catalog_snapshots_by_id": {},
        "instruments_by_id": {},
        "instrument_history_by_id": {},
    }
    assert validate_canonical_catalog_context_graph(
        **kwargs, source_producer_membership_authority=authority
    )
    test_authority = TestSourceProducerMembershipAuthority(
        TestSQLiteMembershipCarrier(tmp_path / "test-empty.sqlite3"),
        TestCoreClock("2000-01-01T00:00:00Z"),
    )
    assert not validate_canonical_catalog_context_graph(
        **kwargs, source_producer_membership_authority=test_authority
    )
    assert not validate_canonical_catalog_context_graph(
        **kwargs, source_producer_membership_authority=fake
    )


def test_canonical_graph_rejects_legal_test_clock_authority(tmp_path):
    from tests.architecture.test_m06_workspace_catalog_source_chain import canonical_graph

    clock = TestCoreClock("2000-01-01T00:00:00Z")
    authority = TestSourceProducerMembershipAuthority(
        TestSQLiteMembershipCarrier(tmp_path / "test-provenance.sqlite3"), clock
    )
    assert authority.admit_release_grant("core_release_1_45_binance_spot") is not None
    source, projection, instrument, _, _ = canonical_graph()
    assert not validate_canonical_catalog_context_graph(
        workspace_catalog_projections_by_id={"wcat_1": projection},
        accepted_source_catalog_snapshots_by_id={"ascat_1": source},
        instruments_by_id={"instr_1": instrument},
        instrument_history_by_id={},
        source_producer_membership_authority=authority,
    )


def test_test_store_grant_and_event_cannot_be_opened_as_production(tmp_path):
    path = tmp_path / "cross-domain.sqlite3"
    clock = TestCoreClock("2000-01-01T00:00:00Z")
    authority = TestSourceProducerMembershipAuthority(
        TestSQLiteMembershipCarrier(path), clock
    )
    assert authority.admit_release_grant("core_release_1_45_binance_spot") is not None
    assert authority.admit_release_event("core_release_1_45_revoke_binance_spot") is not None
    del authority
    with pytest.raises(ValueError, match="MEMBERSHIP_AUTHORITY_DOMAIN_MISMATCH"):
        JsonlMembershipCarrier(path)


def test_production_store_cannot_be_opened_or_mutated_by_test_carrier(tmp_path):
    path = tmp_path / "reverse-cross-domain.sqlite3"
    authority = SourceProducerMembershipAuthority(JsonlMembershipCarrier(path))
    assert authority.admit_release_grant("core_release_1_45_binance_spot") is not None
    with pytest.raises(ValueError, match="MEMBERSHIP_AUTHORITY_DOMAIN_MISMATCH"):
        TestSQLiteMembershipCarrier(path)
    restored = SourceProducerMembershipAuthority(JsonlMembershipCarrier(path))
    assert restored.resolve_current(identity(authority._state().grants["aspm_core_1_45_binance_spot_1"]), 1, "2026-09-15T00:00:00Z") is not None


@pytest.mark.parametrize("mutation", ["test", "unknown", "missing_row", "missing_table"])
def test_production_store_domain_mutation_fails_closed(tmp_path, mutation):
    path = tmp_path / f"domain-{mutation}.sqlite3"
    authority = SourceProducerMembershipAuthority(JsonlMembershipCarrier(path))
    assert authority.admit_release_grant("core_release_1_45_binance_spot") is not None
    with sqlite3.connect(path) as connection:
        if mutation == "test":
            connection.execute(
                "UPDATE membership_authority_store_metadata SET authority_domain = ?",
                ("cryptohunter.source_producer_membership.test.v1",),
            )
        elif mutation == "unknown":
            connection.execute(
                "UPDATE membership_authority_store_metadata SET authority_domain = 'unknown'"
            )
        elif mutation == "missing_row":
            connection.execute("DELETE FROM membership_authority_store_metadata")
        else:
            connection.execute("DROP TABLE membership_authority_store_metadata")
        connection.commit()
    expected = "MEMBERSHIP_AUTHORITY_DOMAIN_MISSING" if mutation == "missing_table" else "MEMBERSHIP_AUTHORITY_DOMAIN_MISMATCH"
    with pytest.raises(ValueError, match=expected):
        JsonlMembershipCarrier(path)


def test_stale_test_domain_digest_cannot_be_copied_into_production_store(tmp_path):
    test_path = tmp_path / "digest-test.sqlite3"
    prod_path = tmp_path / "digest-production.sqlite3"
    test_authority = TestSourceProducerMembershipAuthority(
        TestSQLiteMembershipCarrier(test_path), TestCoreClock("2000-01-01T00:00:00Z")
    )
    assert test_authority.admit_release_grant("core_release_1_45_binance_spot") is not None
    JsonlMembershipCarrier(prod_path)
    with sqlite3.connect(test_path) as source, sqlite3.connect(prod_path) as target:
        row = source.execute(
            "SELECT journal_sequence, canonical_record, previous_record_digest, record_digest "
            "FROM membership_authority_records"
        ).fetchone()
        target.execute("INSERT INTO membership_authority_records VALUES (?, ?, ?, ?)", row)
        target.execute(
            "UPDATE membership_authority_head SET committed_sequence = 1, "
            "committed_head_digest = ?, last_committed_record_id = ? WHERE singleton = 1",
            (row[3], "aspm_core_1_45_binance_spot_1"),
        )
        target.commit()
    with pytest.raises(ValueError, match="CORRUPT_MEMBERSHIP_JOURNAL"):
        SourceProducerMembershipAuthority(JsonlMembershipCarrier(prod_path))


def test_test_store_domain_mutated_to_production_fails_closed(tmp_path):
    path = tmp_path / "test-domain-mutated.sqlite3"
    carrier = TestSQLiteMembershipCarrier(path)
    authority = TestSourceProducerMembershipAuthority(
        carrier, TestCoreClock("2000-01-01T00:00:00Z")
    )
    assert authority.admit_release_grant("core_release_1_45_binance_spot") is not None
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE membership_authority_store_metadata SET authority_domain = ?",
            ("cryptohunter.source_producer_membership.production.v1",),
        )
        connection.commit()
    with pytest.raises(ValueError, match="MEMBERSHIP_AUTHORITY_DOMAIN_MISMATCH"):
        TestSQLiteMembershipCarrier(path)


def test_sqlite_primary_key_rejects_duplicate_sequence(tmp_path):
    authority, _ = make_authority(tmp_path)
    with sqlite3.connect(authority._carrier.path) as connection:
        row = connection.execute(
            "SELECT canonical_record, previous_record_digest, record_digest "
            "FROM membership_authority_records WHERE journal_sequence = 1"
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO membership_authority_records VALUES (1, ?, ?, ?)", row
            )


def test_same_grant_concurrent_threads_is_one_physical_record(tmp_path):
    path = tmp_path / "thread-grant.sqlite3"
    authority = SourceProducerMembershipAuthority(JsonlMembershipCarrier(path))
    barrier = Barrier(2)

    def admit():
        barrier.wait()
        return authority.admit_release_grant("core_release_1_45_binance_spot")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result() for future in (pool.submit(admit), pool.submit(admit))]
    assert all(result is not None for result in results)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM membership_authority_records").fetchone()[0] == 1
    SourceProducerMembershipAuthority(JsonlMembershipCarrier(path))


def test_same_grant_concurrent_processes_is_one_physical_record(tmp_path):
    path = str(tmp_path / "process-grant.sqlite3")
    SourceProducerMembershipAuthority(JsonlMembershipCarrier(path))
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [context.Process(target=_process_admit_same_grant, args=(path, barrier, results)) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(20)
        assert process.exitcode == 0
    assert results.get(timeout=2) and results.get(timeout=2)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM membership_authority_records").fetchone()[0] == 1
    SourceProducerMembershipAuthority(JsonlMembershipCarrier(path))


def test_same_event_concurrent_threads_is_one_physical_event(tmp_path):
    authority, _ = make_authority(tmp_path, "thread-event")
    barrier = Barrier(2)

    def admit():
        barrier.wait()
        return authority.admit_release_event("core_release_1_45_revoke_binance_spot")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result() for future in (pool.submit(admit), pool.submit(admit))]
    assert all(result is not None for result in results)
    with sqlite3.connect(authority._carrier.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM membership_authority_records").fetchone()[0] == 2
    SourceProducerMembershipAuthority(JsonlMembershipCarrier(authority._carrier.path))


def test_revoke_vs_supersede_race_commits_exactly_one_terminal(tmp_path):
    authority, _ = make_authority(tmp_path, "terminal-race")
    authority.admit_release_grant("core_release_1_45_binance_spot_generation_2")
    barrier = Barrier(2)

    def admit(name):
        barrier.wait()
        return authority.admit_release_event(name)

    names = ("core_release_1_45_revoke_binance_spot", "core_release_1_45_supersede_binance_spot")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result() for future in (pool.submit(admit, names[0]), pool.submit(admit, names[1]))]
    assert sum(result is not None for result in results) == 1
    with sqlite3.connect(authority._carrier.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM membership_authority_records").fetchone()[0] == 3
    SourceProducerMembershipAuthority(JsonlMembershipCarrier(authority._carrier.path))


def test_late_backdated_event_is_non_retroactive_and_scheduled_cutoff_works(tmp_path):
    authority, grant = make_authority(tmp_path, "late-event")
    at_0330 = snapshot(grant, "2026-09-15T03:30:00Z")
    assert validate_accepted_source_snapshot_membership_evidence(at_0330, authority)
    _TEST_CLOCK.set_utc("2026-09-15T04:00:00Z")
    event = authority.admit_release_event("core_release_1_45_late_revoke_binance_spot")
    assert event is not None and event.authority_admitted_at_utc == "2026-09-15T04:00:00Z"
    assert validate_accepted_source_snapshot_membership_evidence(at_0330, authority)
    assert authority.resolve_current(identity(grant), 1, "2026-09-15T03:30:00Z") == grant
    assert authority.resolve_current(identity(grant), 1, "2026-09-15T04:00:00Z") is None
    assert not validate_accepted_source_snapshot_membership_evidence(
        snapshot(grant, "2026-09-15T04:00:00Z"), authority
    )


def test_late_grant_cannot_retroactively_authorize_snapshot(tmp_path):
    from bot_core.instruments.source_producer_membership import _G1, _materialize_grant

    authority = SourceProducerMembershipAuthority(JsonlMembershipCarrier(tmp_path / "late-grant.sqlite3"))
    _TEST_CLOCK.set_utc("2026-09-15T02:00:00Z")
    candidate = _materialize_grant(_G1, "2026-09-15T02:00:00Z")
    grant = type(_G1)(**candidate)
    old = snapshot(grant, "2026-09-15T01:00:00Z")
    assert not validate_accepted_source_snapshot_membership_evidence(old, authority)
    admitted = authority.admit_release_grant(
        "core_release_1_45_binance_spot"
    )
    assert admitted == grant
    assert not validate_accepted_source_snapshot_membership_evidence(old, authority)
    assert validate_accepted_source_snapshot_membership_evidence(
        snapshot(grant, "2026-09-15T02:00:00Z"), authority
    )


def test_scheduled_future_grant_requires_both_admission_and_effective_time(tmp_path):
    authority = SourceProducerMembershipAuthority(
        JsonlMembershipCarrier(tmp_path / "scheduled-grant.sqlite3")
    )
    _TEST_CLOCK.set_utc("2025-12-31T23:00:00Z")
    grant = authority.admit_release_grant(
        "core_release_1_45_generic_testnet_spot",
    )
    assert grant is not None
    claimant = identity(grant)
    assert authority.resolve_current(claimant, 1, "2025-12-31T23:30:00Z") is None
    assert authority.resolve_historical(
        grant.accepted_source_producer_membership_id,
        1,
        grant.content_fingerprint,
        claimant,
        "2025-12-31T23:30:00Z",
    ) is None
    assert authority.resolve_current(claimant, 1, "2026-01-01T00:00:00Z") == grant
    restored = SourceProducerMembershipAuthority(JsonlMembershipCarrier(authority._carrier.path))
    assert restored.resolve_current(claimant, 1, "2026-01-01T00:00:00Z") == grant
