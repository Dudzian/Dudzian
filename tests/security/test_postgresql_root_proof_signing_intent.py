"""Autentyczna lineage Stage 9 i prawdziwe rezerwacje PostgreSQL bez podpisywania.

Fixture native-keyring zastępuje jedynie adapter systemu operacyjnego. Każda
operacja weryfikuje InitialBinding, niezmienne podpisy CHA i żywe autorytety
PostgreSQL. Procesy fork dziedziczą capability wydane przez weryfikatory,
następnie ponownie sprawdzają trwałe źródła. DTO callera nie staje się autorytetem.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from bot_core import (
    cha_root_proof_signing_custody as request_custody,
    local_signing_custody as custody,
    postgresql_root_proof_retention as retention,
    postgresql_root_proof_signing_intent as implementation,
    postgresql_root_proof_signing_intent_schema as intent_schema,
    root_proof_issuer_orchestration as orchestration,
)
from bot_core.cha_attempt_store import AttemptState, SQLiteCHAAttemptStore
from bot_core.licensing import (
    cha_root_proof_attempt_reservation as reservation,
    cha_root_proof_signed_attempt as signed,
)
from bot_core.licensing.canonical import parse_canonical
from bot_core.licensing.cha_account_reservation import VerifiedAccountGenesisInitialBinding
from bot_core.postgresql_authenticated_issuer_history import PostgreSQLAuthenticatedIssuerHistory
from bot_core.postgresql_checkpoint_authority import PostgreSQLCheckpointAuthority
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.root_proof_issuer_substrate import (
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
)
from tests.security import test_root_proof_issuer_runtime as preflight_tests
from tests.security._local_lifecycle_reader_test_seam import (
    simulate_qualified_local_downgrade_for_reader_test,
)
from tests.security._local_signing_platform import requires_native_custody_locking
from tests.security.test_postgresql_checkpoint_authority import BASE_DSN
from tests.security.test_root_proof_issuance_reconciliation import _history_boundary
from tests.security.test_root_proof_issuer_orchestration import (
    _durable_snapshot,
    _no_grant,
    _prohibit_mutations,
)

pytestmark = [pytest.mark.external_postgresql, requires_native_custody_locking]
native_keyring = preflight_tests.native_keyring
installed_signed_attempt = preflight_tests.installed_signed_attempt

ROOT_DOMAIN = b"CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_V1\x00"
fork_only = pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="procesy testowe dziedziczą pełną lineage autentycznych capability przez fork",
)


@pytest.fixture(scope="module")
def lineage_monkeypatch():
    """Oddziela stały earlier-lineage harness od mutacji poszczególnych scenariuszy."""
    with pytest.MonkeyPatch.context() as patcher:
        yield patcher


@pytest.fixture(scope="module")
def lineage_path(tmp_path_factory):
    return tmp_path_factory.mktemp("immutable-initial-binding-lineage")


@pytest.fixture(scope="module")
def lineage_challenge(lineage_monkeypatch, lineage_path):
    return preflight_tests.challenge_harness.__wrapped__(lineage_monkeypatch, lineage_path)


@pytest.fixture(scope="module")
def lineage_integration(lineage_challenge, lineage_monkeypatch, lineage_path):
    yield from preflight_tests.integration.__wrapped__(
        lineage_challenge, lineage_monkeypatch, lineage_path
    )


@pytest.fixture(scope="module")
def lineage_package(lineage_integration, lineage_monkeypatch):
    return preflight_tests.package.__wrapped__(lineage_integration, lineage_monkeypatch)


@pytest.fixture(scope="module")
def lineage_lppi(lineage_integration, lineage_package, lineage_monkeypatch, lineage_path):
    return preflight_tests.lppi.__wrapped__(
        lineage_integration, lineage_package, lineage_monkeypatch, lineage_path
    )


@pytest.fixture(scope="module")
def lineage_active(lineage_lppi, lineage_monkeypatch):
    yield from preflight_tests.active.__wrapped__(lineage_lppi, lineage_monkeypatch)


@pytest.fixture(scope="module")
def lineage_committed(lineage_active, lineage_lppi):
    return preflight_tests.committed.__wrapped__(lineage_active, lineage_lppi)


@pytest.fixture(scope="module")
def lineage_cha(lineage_committed, lineage_monkeypatch):
    return preflight_tests.cha.__wrapped__(lineage_committed, lineage_monkeypatch)


@pytest.fixture(scope="module")
def lineage_reserved(lineage_cha, lineage_monkeypatch):
    """Zachowuje tylko niezmienny binding wydany przez cały oryginalny harness."""
    return preflight_tests.reserved.__wrapped__(lineage_cha, lineage_monkeypatch)


@pytest.fixture
def reserved(lineage_reserved):
    """Każdy test nadal tworzy własne live registries, klucze i podpisaną próbę CHA."""
    return lineage_reserved


def _reject_mutation(*args, **kwargs):
    pytest.fail("rezerwacja RESERVED wywołała podpisywanie albo inną mutację issuera/CHA")


def _root_signer(value):
    return custody.LocalRootProofSigningProvider(
        value.directory,
        keyring_index_path=value.directory / "keyring-index.json",
        security=value.security,
    )


def _provider(value, **changes):
    return implementation.PostgreSQLRootProofSigningIntent(
        changes.get("connection", value.connection),
        schema=value.setup.schema,
        scope=changes.get("scope", value.scope),
        verification_authority=changes.get("signer", _root_signer(value)),
        issuer_authority=changes.get("issuer_authority", value.genuine.authority),
    )


def _call(value, operation="reserve", provider=None, **changes):
    return getattr(value.provider if provider is None else provider, operation)(
        binding=changes.get("binding", value.genuine.binding),
        signed_attempt=changes.get("signed_attempt", value.genuine.opaque),
    )


def _rows(value):
    with psycopg.connect(BASE_DSN) as conn:
        return conn.execute(
            sql.SQL("SELECT canonical_envelope,canonical_payload,status FROM {}.intents").format(
                sql.Identifier(value.setup.schema)
            )
        ).fetchall()


def _assert_reserved(value, record):
    auth = value.genuine.current.reservation.authorization
    snapshot = value.signer.credential_snapshot()
    assert type(record) is implementation.RootProofSigningIntent
    assert record.state == "RESERVED"
    assert record.authorizes_operations is False
    assert record.authorization == auth
    assert record.retained_request == value.genuine.retained
    assert record.credential_snapshot == snapshot
    assert record.bytes_to_sign == ROOT_DOMAIN + record.canonical_payload
    assert parse_canonical(record.canonical_payload) == {
        "schema_version": "1",
        "environment": auth.environment,
        "trust_domain": auth.trust_domain,
        "issuer_id": value.scope.issuer_id,
        "issuer_key_id": snapshot.credential_identity.credential_identity,
        "issuer_key_version": snapshot.key_version,
        "root_proof_id": record.root_proof_id,
        "intended_action": "ACCOUNT_GENESIS_BOOTSTRAP",
        "product_scope": auth.product_scope,
        "account_id": auth.account_id,
        "logical_operation_id": auth.logical_operation_id,
        "canonical_genesis_request_fingerprint_sha256": (
            auth.canonical_genesis_request_fingerprint_sha256
        ),
        "bootstrap_entitlement_id": auth.bootstrap_entitlement_id,
        "entitlement_generation": auth.entitlement_generation,
        "authorization_constraints": {"max_logical_operations": 1, "max_account_ids": 1},
        "issuance_attempt_id": value.genuine.retained.attempt_id,
    }
    identifier = uuid.UUID(record.root_proof_id.removeprefix("rpf_"))
    assert identifier.version == 7 and identifier.variant == uuid.RFC_4122
    assert record.root_proof_id == "rpf_" + str(identifier)
    assert _rows(value) == [(record.canonical_envelope, record.canonical_payload, "RESERVED")]
    assert not hasattr(record, "signature")
    assert not hasattr(record, "signature_ed25519_base64url")
    with pytest.raises(FrozenInstanceError):
        record.state = "ISSUED"


@pytest.fixture
def intent_boundary(installed_signed_attempt, native_keyring, tmp_path: Path, monkeypatch):
    """Instaluje prawdziwy PostgreSQL i custody bez utworzenia podpisu Root Proof."""
    genuine = installed_signed_attempt
    auth = genuine.current.reservation.authorization
    token = uuid.uuid4().hex[:12]
    setup = intent_schema.PostgreSQLRootProofSigningIntentProvisioning(
        f"rpi_{token}", f"rpio_{token}", f"rpiw_{token}", auth.trust_domain
    )
    directory = tmp_path / "root-signing-intent-custody"
    directory.mkdir()
    security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, auth.trust_domain)
    administrator = custody.NativeKeyringSigningSecretAdministrator(
        index_path=directory / "keyring-index.json",
        security=security,
        role=ProviderRole.ROOT_PROOF_SIGNING,
    )
    custody.provision_local_signing_authority(
        directory, administrator, security=security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    scope = retention.RootProofRetentionScope(
        SecurityProfile.PRODUCTION_LOCAL,
        auth.environment,
        auth.trust_domain,
        auth.product_scope,
        "IndependentAccountGenesisRootProofIssuer",
    )
    intent_schema.provision_postgresql_root_proof_signing_intent(
        PostgreSQLConnectionConfig(BASE_DSN), setup
    )
    try:
        value = SimpleNamespace(
            genuine=genuine,
            setup=setup,
            directory=directory,
            security=security,
            scope=scope,
            connection=PostgreSQLConnectionConfig(
                make_conninfo(BASE_DSN, user=setup.issuer_writer_role)
            ),
        )
        value.signer = _root_signer(value)
        value.provider = _provider(value, signer=value.signer)
        for target, methods in (
            (custody.LocalRootProofSigningProvider, ("sign_root_proof",)),
            (custody._LocalSigningProviderBase, ("_sign", "_sign_result")),
            (request_custody._RuntimeCustody, ("_sign",)),
            (signed, ("execute_local_issuance",)),
            (type(genuine.authority.entitlement_registry), ("compare_and_swap_bind",)),
            (retention.PostgreSQLRootProofRetention, ("retain_prepared",)),
            (
                PostgreSQLAuthenticatedIssuerHistory,
                ("append", "append_exact_successor", "retain_attested_head"),
            ),
            (PostgreSQLCheckpointAuthority, ("advance", "advance_exact_successor")),
            (
                SQLiteCHAAttemptStore,
                (
                    "reserve_or_resolve_attempt_id",
                    "finalize_attempt",
                    "prepare_signature",
                    "persist_signature",
                    "record_recovery_resolution",
                    "replace_after_authoritative_unbound",
                    "supersede_pre_send",
                    "_write",
                ),
            ),
        ):
            for method in methods:
                monkeypatch.setattr(target, method, _reject_mutation)
        yield value
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(setup.schema))
            )
            for role in (setup.issuer_writer_role, setup.schema_owner_role):
                conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


def _process_reserve(value, mode, ready, start, results):
    """Nowy runtime i połączenia sprawdzają źródła odziedziczonych genuine capability."""
    try:
        custody.LocalRootProofSigningProvider.sign_root_proof = _reject_mutation
        request_custody._RuntimeCustody._sign = _reject_mutation
        original_inputs = implementation.PostgreSQLRootProofSigningIntent._inputs
        original_cut = implementation._persistence_cut

        def verified_inputs(self, binding, signed_attempt):
            value = original_inputs(self, binding, signed_attempt)
            results.put(("phase", os.getpid(), "INPUTS_VERIFIED"))
            return value

        def observed_cut(cut):
            original_cut(cut)
            results.put(("phase", os.getpid(), cut))
            if cut == mode:
                # Ukończona rzeczywista faza musi dotrzeć przed natychmiastową śmiercią.
                results.close()
                results.join_thread()
                os._exit(73)

        implementation.PostgreSQLRootProofSigningIntent._inputs = verified_inputs
        implementation._persistence_cut = observed_cut
        provider = _provider(value)
        if ready is not None:
            ready.put(os.getpid())
        if start is not None and not start.wait(20):
            raise RuntimeError("przekroczony czas bariery procesów")
        if mode == "read":
            implementation._new_root_proof_id = _reject_mutation
            record = _call(value, "read_intent", provider)
        else:
            record = _call(value, provider=provider)
        results.put(
            (
                "ok",
                os.getpid(),
                None
                if record is None
                else (
                    record.root_proof_id,
                    record.canonical_envelope,
                    record.canonical_payload,
                ),
            )
        )
    except BaseException as error:
        results.put(("error", os.getpid(), type(error).__name__))


def _join(process):
    process.join(30)
    if process.is_alive():
        process.terminate()
        process.join(5)
        pytest.fail("proces rezerwacji signing intent nie zakończył pracy")


def _process_outcomes(results, count, *, expected_cut=None):
    """Oczekuje maksymalnie 30 s bez ukończenia rzeczywistej, ograniczonej fazy.

    Zdarzenia powstają wyłącznie po oryginalnym _inputs, rzeczywistym cut lub wyniku.
    Nie ma heartbeatów. Każdy worker może ukończyć najwyżej cztery weryfikacje oraz
    dwa commit cuts; brak dalszego postępu lub dodatkowe zdarzenie jest błędem.
    """
    outcomes = []
    phases = {}
    while len(outcomes) < count:
        item = results.get(timeout=30)
        assert type(item) is tuple and len(item) == 3, item
        kind, pid, detail = item
        if kind == "phase":
            assert detail in {"INPUTS_VERIFIED", "BEFORE_COMMIT", "AFTER_COMMIT"}, item
            observed = phases.setdefault(pid, [])
            assert len(phases) <= count, phases
            if detail == "INPUTS_VERIFIED":
                assert observed.count(detail) < 4, phases
            else:
                assert detail not in observed and "INPUTS_VERIFIED" in observed, phases
                if detail == "AFTER_COMMIT":
                    assert "BEFORE_COMMIT" in observed, phases
            observed.append(detail)
            assert len(observed) <= 6, phases
            if expected_cut is not None and detail == expected_cut:
                return item
        else:
            assert kind in {"ok", "error"} and expected_cut is None, item
            outcomes.append(item)
    return outcomes


def _terminate_remaining(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
        process.join(5)


class TestImmutableSignedSource:
    """Niezmienny genuine signed source jest wspólny; każdy store i root key jest osobny.

    Oryginalny harness tworzy autentyczny binding, publiczne rejestry PostgreSQL,
    custody i podpisaną próbę tylko raz. Testy tej klasy nie modyfikują źródeł CHA
    ani requester/claimant/entitlement. Każde publiczne wywołanie nadal ponownie
    weryfikuje wszystkie żywe źródła; współdzielenie usuwa wyłącznie powtórny setup.
    """

    @pytest.fixture(scope="class")
    def source_monkeypatch(self):
        with pytest.MonkeyPatch.context() as patcher:
            yield patcher

    @pytest.fixture(scope="class")
    def native_keyring(self, source_monkeypatch):
        return preflight_tests.native_keyring.__wrapped__(source_monkeypatch)

    @pytest.fixture(scope="class")
    def source_path(self, tmp_path_factory):
        return tmp_path_factory.mktemp("immutable-signed-cha-source")

    @pytest.fixture(scope="class")
    def installed_signed_attempt(
        self, lineage_reserved, native_keyring, source_path, source_monkeypatch, request
    ):
        yield from preflight_tests.installed_signed_attempt.__wrapped__(
            lineage_reserved, native_keyring, source_path, source_monkeypatch, request
        )

    def test_genuine_intent_reservation_never_signs_or_opens_blocked_orchestration(
        self, intent_boundary, monkeypatch
    ):
        value = intent_boundary
        with _history_boundary(value.genuine, monkeypatch) as authorities:
            before = _durable_snapshot(authorities)
            calls = _prohibit_mutations(authorities, monkeypatch)
            monkeypatch.setattr(custody._LocalSigningProviderBase, "_sign_result", _reject_mutation)
            monkeypatch.setattr(
                retention.PostgreSQLRootProofRetention, "retain_prepared", _reject_mutation
            )
            assert _call(value, "read_intent") is None
            record = _call(value)
            _assert_reserved(value, record)
            assert _call(value) == record
            assert _call(value, "read_intent", _provider(value)) == record
            for operation in ("reserve", "read_intent"):
                with pytest.raises(implementation.RootProofSigningIntentConflict):
                    _call(value, operation, signed_attempt=record)
            with pytest.raises(implementation.RootProofSigningIntentNotIssued):
                _call(value, "require_issued")
            report = orchestration.orchestrate_root_proof_issuance(value.genuine.binding)
            _no_grant(report)
            assert calls == []
            assert _durable_snapshot(authorities) == before
            assert value.genuine.current.state is AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT

    def test_exact_retry_and_new_runtime_never_allocate_another_root_id(
        self, intent_boundary, monkeypatch
    ):
        value = intent_boundary
        record = _call(value)
        monkeypatch.setattr(implementation, "_new_root_proof_id", _reject_mutation)
        assert _call(value, "read_intent", _provider(value)) == record
        assert _call(value, "reserve", _provider(value)) == record
        _assert_reserved(value, record)

    def test_same_tuple_threads_converge_on_one_durable_intent(self, intent_boundary, monkeypatch):
        value = intent_boundary
        barrier = threading.Barrier(2)
        monkeypatch.setattr(
            custody.LocalRootProofSigningProvider, "sign_root_proof", _reject_mutation
        )
        providers = (_provider(value), _provider(value))

        def reserve(provider):
            barrier.wait(timeout=20)
            return _call(value, provider=provider)

        with ThreadPoolExecutor(max_workers=2) as workers:
            records = list(workers.map(reserve, providers))
        assert records[0] == records[1]
        _assert_reserved(value, records[0])

    @fork_only
    def test_independent_processes_reserve_and_restart_with_same_bytes(self, intent_boundary):
        value = intent_boundary
        context = multiprocessing.get_context("fork")
        ready, start, results = context.Queue(), context.Event(), context.Queue()
        processes = [
            context.Process(target=_process_reserve, args=(value, "reserve", ready, start, results))
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        identifiers = [ready.get(timeout=30) for _ in processes]
        assert len(set(identifiers)) == 2
        start.set()
        try:
            outcomes = _process_outcomes(results, len(processes))
            for process in processes:
                _join(process)
                assert process.exitcode == 0
        finally:
            _terminate_remaining(processes)
        assert all(outcome[0] == "ok" for outcome in outcomes), outcomes
        assert outcomes[0][2] == outcomes[1][2]
        recovered = _call(value, "read_intent", _provider(value))
        assert outcomes[0][2] == (
            recovered.root_proof_id,
            recovered.canonical_envelope,
            recovered.canonical_payload,
        )
        _assert_reserved(value, recovered)
        restarted = context.Process(
            target=_process_reserve, args=(value, "read", None, None, results)
        )
        restarted.start()
        try:
            outcome = _process_outcomes(results, 1)[0]
            _join(restarted)
        finally:
            _terminate_remaining([restarted])
        assert restarted.exitcode == 0 and outcome[0] == "ok", outcome
        assert outcome[2] == outcomes[0][2]

    @fork_only
    @pytest.mark.parametrize("cut", ["BEFORE_COMMIT", "AFTER_COMMIT"])
    def test_process_death_at_commit_cut_has_exact_durable_recovery(
        self, intent_boundary, cut, monkeypatch
    ):
        value = intent_boundary
        context = multiprocessing.get_context("fork")
        results = context.Queue()
        process = context.Process(target=_process_reserve, args=(value, cut, None, None, results))
        process.start()
        try:
            assert _process_outcomes(results, 1, expected_cut=cut) == ("phase", process.pid, cut)
            _join(process)
        finally:
            _terminate_remaining([process])
        assert process.exitcode == 73
        recovered = _call(value, "read_intent", _provider(value))
        if cut == "BEFORE_COMMIT":
            assert recovered is None and _rows(value) == []
            recovered = _call(value)
        else:
            assert recovered is not None
            monkeypatch.setattr(implementation, "_new_root_proof_id", _reject_mutation)
            assert _call(value) == recovered
        _assert_reserved(value, recovered)

    @pytest.mark.parametrize("cut", ["BEFORE_COMMIT", "AFTER_COMMIT"])
    def test_commit_exception_is_indeterminate_and_retry_resolves_durable_state(
        self, intent_boundary, monkeypatch, cut
    ):
        value = intent_boundary

        def lose_response(name):
            if name == cut:
                raise OSError("TEST_ONLY commit response unavailable")

        monkeypatch.setattr(implementation, "_persistence_cut", lose_response)
        with pytest.raises(implementation.RootProofSigningIntentIndeterminate):
            _call(value)
        monkeypatch.setattr(implementation, "_persistence_cut", lambda name: None)
        recovered = _call(value, "read_intent", _provider(value))
        if cut == "BEFORE_COMMIT":
            assert recovered is None and _rows(value) == []
            recovered = _call(value)
        else:
            assert recovered is not None
            monkeypatch.setattr(implementation, "_new_root_proof_id", _reject_mutation)
            assert _call(value) == recovered
        _assert_reserved(value, recovered)

    def test_fabricated_inputs_cannot_reserve_or_read_intent(self, intent_boundary):
        value = intent_boundary
        report = preflight_tests.runtime.preflight_root_proof_issuance_attempt(value.genuine.opaque)
        for target, cls in (
            ("binding", VerifiedAccountGenesisInitialBinding),
            ("signed_attempt", signed.VerifiedSignedImmutableRootProofIssuanceAttempt),
        ):
            for untrusted in (
                None,
                value.genuine.retained.canonical_bytes,
                asdict(value.genuine.current.identity),
                object.__new__(cls),
                report,
            ):
                for operation in ("reserve", "read_intent"):
                    with pytest.raises(implementation.RootProofSigningIntentConflict):
                        _call(value, operation, **{target: untrusted})
                assert _rows(value) == []

    @pytest.mark.parametrize(
        "target", [custody.SigningKeyLifecycle.VERIFY_ONLY, custody.SigningKeyLifecycle.REVOKED]
    )
    def test_root_signer_lifecycle_change_cannot_reinterpret_reserved_intent(
        self, intent_boundary, target
    ):
        value = intent_boundary
        _call(value)
        before = _rows(value)
        simulate_qualified_local_downgrade_for_reader_test(
            value.directory,
            security=value.security,
            target=target,
        )
        for operation in ("reserve", "read_intent"):
            with pytest.raises(implementation.RootProofSigningIntentConflict):
                _call(value, operation)
        assert _rows(value) == before

    def test_store_owner_substitution_refuses_without_creating_another_store(
        self, intent_boundary, monkeypatch
    ):
        value = intent_boundary
        original = _call(value)
        substitute = value.genuine.path.with_name("untrusted-attempts.sqlite3")
        monkeypatch.setattr(reservation, "_attempt_store_path", lambda: substitute)
        for operation in ("reserve", "read_intent"):
            with pytest.raises(implementation.RootProofSigningIntentConflict):
                _call(value, operation)
        assert not substitute.exists()
        assert _rows(value) == [
            (original.canonical_envelope, original.canonical_payload, "RESERVED")
        ]

    def test_issuer_identity_substitution_cannot_reinterpret_existing_intent(self, intent_boundary):
        value = intent_boundary
        _call(value)
        before = _rows(value)
        substitute = _provider(
            value, scope=replace(value.scope, issuer_id="other-independent-issuer")
        )
        for operation in ("reserve", "read_intent"):
            with pytest.raises(implementation.RootProofSigningIntentConflict):
                _call(value, operation, substitute)
        assert _rows(value) == before

    def test_equivalent_unpinned_public_authority_does_not_grant_reservation(self, intent_boundary):
        value = intent_boundary
        authority = value.genuine.authority
        substitute = type(authority)(
            authority.entitlement_registry,
            authority.requester_registry,
            authority.claimant_registry,
            value.genuine.subject,
        )
        provider = _provider(value, issuer_authority=substitute)
        for operation in ("reserve", "read_intent"):
            with pytest.raises(implementation.RootProofSigningIntentConflict):
                _call(value, operation, provider)
        assert _rows(value) == []

    def test_active_signing_key_substitution_preserves_original_bytes(self, intent_boundary):
        value = intent_boundary
        original = _call(value)
        directory = value.directory.with_name("other-root-signing-custody")
        directory.mkdir()
        administrator = custody.NativeKeyringSigningSecretAdministrator(
            index_path=directory / "keyring-index.json",
            security=value.security,
            role=ProviderRole.ROOT_PROOF_SIGNING,
        )
        custody.provision_local_signing_authority(
            directory, administrator, security=value.security, role=ProviderRole.ROOT_PROOF_SIGNING
        )
        signer = _root_signer(SimpleNamespace(directory=directory, security=value.security))
        assert signer.credential_snapshot().public_key != original.credential_snapshot.public_key
        substitute = _provider(value, signer=signer)
        for operation in ("reserve", "read_intent"):
            with pytest.raises(implementation.RootProofSigningIntentConflict):
                _call(value, operation, substitute)
        assert _rows(value) == [
            (original.canonical_envelope, original.canonical_payload, "RESERVED")
        ]

    def test_root_id_caller_override_is_rejected_by_closed_public_api(self, intent_boundary):
        value = intent_boundary
        with pytest.raises(TypeError):
            value.provider.reserve(
                binding=value.genuine.binding,
                signed_attempt=value.genuine.opaque,
                root_proof_id="rpf_018f3e70-7b5e-7c21-8b9a-0123456789ab",
            )
        assert _rows(value) == []

    def test_lock_selector_is_only_scheduling_and_never_authorization(
        self, intent_boundary, monkeypatch
    ):
        value = intent_boundary
        monkeypatch.setattr(
            implementation,
            "_signed_attempt_lock_selector",
            lambda candidate: "rpa_018f3e70-7b5e-7c21-8b9a-0123456789ab",
        )
        monkeypatch.setattr(implementation, "_new_root_proof_id", _reject_mutation)
        for operation in ("reserve", "read_intent"):
            with pytest.raises(implementation.RootProofSigningIntentConflict):
                _call(value, operation)
        assert _rows(value) == []

    def test_genuine_cached_locator_is_not_a_signed_attempt_capability(
        self, intent_boundary, monkeypatch
    ):
        value = intent_boundary
        locator = signed._signed_attempt_lock_selector(value.genuine.opaque)
        assert locator == value.genuine.retained.attempt_id
        for untrusted in (
            None,
            locator,
            value.genuine.current,
            object.__new__(signed.VerifiedSignedImmutableRootProofIssuanceAttempt),
        ):
            with pytest.raises(signed.SignedIssuanceAttemptError):
                signed._signed_attempt_lock_selector(untrusted)
        monkeypatch.setattr(implementation, "_new_root_proof_id", _reject_mutation)
        for operation in ("reserve", "read_intent"):
            with pytest.raises(implementation.RootProofSigningIntentConflict):
                _call(value, operation, signed_attempt=locator)
        assert _rows(value) == []

    @pytest.mark.parametrize("column", ["canonical_envelope", "canonical_payload"])
    def test_privileged_persisted_corruption_never_allocates_a_replacement(
        self, intent_boundary, monkeypatch, column
    ):
        value = intent_boundary
        _call(value)
        with psycopg.connect(BASE_DSN) as conn:
            conn.execute(
                sql.SQL("ALTER TABLE {}.intents DISABLE TRIGGER USER").format(
                    sql.Identifier(value.setup.schema)
                )
            )
            conn.execute(
                sql.SQL("UPDATE {}.intents SET {}=set_byte({},0,get_byte({},0)#1)").format(
                    sql.Identifier(value.setup.schema),
                    sql.Identifier(column),
                    sql.Identifier(column),
                    sql.Identifier(column),
                )
            )
            for trigger in ("immutable_row", "immutable_truncate"):
                conn.execute(
                    sql.SQL("ALTER TABLE {}.intents ENABLE ALWAYS TRIGGER {}").format(
                        sql.Identifier(value.setup.schema), sql.Identifier(trigger)
                    )
                )
        damaged = _rows(value)
        monkeypatch.setattr(implementation, "_new_root_proof_id", _reject_mutation)
        for operation in ("reserve", "read_intent"):
            with pytest.raises(implementation.RootProofSigningIntentCorrupt):
                _call(value, operation)
        assert _rows(value) == damaged

    def test_connection_failure_is_unavailable_and_does_not_authorize_absence(
        self, intent_boundary, monkeypatch
    ):
        value = intent_boundary
        record = _call(value)
        original_connect = implementation.PostgreSQLRootProofSigningIntent._connect

        def unavailable(self):
            raise OSError("TEST_ONLY password=do-not-disclose")

        monkeypatch.setattr(
            implementation.PostgreSQLRootProofSigningIntent, "_connect", unavailable
        )
        for operation in ("reserve", "read_intent"):
            with pytest.raises(implementation.RootProofSigningIntentUnavailable) as failure:
                _call(value, operation)
            assert "do-not-disclose" not in str(failure.value)
            assert failure.value.__cause__ is None
        monkeypatch.setattr(
            implementation.PostgreSQLRootProofSigningIntent, "_connect", original_connect
        )
        monkeypatch.setattr(implementation, "_new_root_proof_id", _reject_mutation)
        assert _call(value, provider=_provider(value)) == record


class TestMutableUpstreamAuthorities:
    """Mutacje upstream dostają świeże genuine signed source i rejestry w każdym teście."""

    @pytest.mark.parametrize(
        "mutation",
        [
            "requester_inactive",
            "claimant_rotation",
            "entitlement_revoked",
            "entitlement_superseded",
        ],
    )
    @pytest.mark.parametrize("reserve_first", [False, True])
    def test_live_authority_change_refuses_existing_and_new_intent(
        self, intent_boundary, mutation, reserve_first
    ):
        value = intent_boundary
        if reserve_first:
            _call(value)
        before = _rows(value)
        preflight_tests._change_authority(value.genuine, mutation)
        for operation in ("reserve", "read_intent"):
            # Wcześniejsze porty lineage mogą odmówić przed rozróżnieniem przyczyny.
            with pytest.raises(implementation.RootProofSigningIntentError):
                _call(value, operation)
        assert _rows(value) == before

    @pytest.mark.parametrize("operation", ["reserve", "read_intent"])
    def test_exact_retry_rechecks_live_authority_after_loading_retained_row(
        self, intent_boundary, monkeypatch, operation
    ):
        value = intent_boundary
        _call(value)
        before = _rows(value)
        original_read = implementation.PostgreSQLRootProofSigningIntent._read_row
        changes = []

        def change_after_read(self, connection, request):
            row = original_read(self, connection, request)
            if not changes:
                changes.append(True)
                preflight_tests._change_authority(value.genuine, "requester_inactive")
            return row

        monkeypatch.setattr(
            implementation.PostgreSQLRootProofSigningIntent, "_read_row", change_after_read
        )
        monkeypatch.setattr(implementation, "_new_root_proof_id", _reject_mutation)
        with pytest.raises(implementation.RootProofSigningIntentError):
            _call(value, operation)
        assert changes == [True]
        assert _rows(value) == before
