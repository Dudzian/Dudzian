"""Preparation public readers cannot acquire production or BIND authority."""

from __future__ import annotations

from types import SimpleNamespace
from contextlib import nullcontext

import pytest

import bot_core.postgresql_root_proof_issuance_authority_v2 as public_v2
from bot_core.entitlement_registry_contract import RegistrySubject
from bot_core.licensing import cha_root_proof_attempt_reservation as reservation
from bot_core.licensing.cha_account_reservation import AccountInitialBindingError
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_root_proof_issuance_authority import (
    PostgreSQLRootProofIssuanceAuthority,
    ProductionLocalIssuanceAuthorityError,
)
from bot_core.postgresql_root_proof_preparation_qualification import (
    RootProofPreparationV2QualificationError,
)
from bot_core.postgresql_preaccount_credentials import CredentialResolutionError
from bot_core.root_proof_issuer_substrate import (
    ProviderQualificationPolicy,
    SecurityProfile,
)


def _connection_must_not_open(*args, **kwargs):
    raise AssertionError("unreviewed public authority reached a database connection")


def test_unreviewed_deployment_cannot_construct_public_authority(monkeypatch):
    monkeypatch.setattr(public_v2.psycopg, "connect", _connection_must_not_open)
    connection = PostgreSQLConnectionConfig("host=unavailable password=fixture-secret")
    with pytest.raises(RootProofPreparationV2QualificationError, match="reviewed preparation"):
        public_v2.PostgreSQLRootProofIssuanceAuthorityV2(
            connection,
            connection,
            connection,
            deployment=object(),
            entitlement_subject=RegistrySubject("lookup", "PRODUCTION", "trust-domain"),
        )


@pytest.mark.parametrize(
    "implementation",
    [
        public_v2.PostgreSQLEntitlementRegistryProviderV2,
        public_v2.PostgreSQLRequesterCredentialRegistryProviderV2,
        public_v2.PostgreSQLClaimantIdentityRegistryProviderV2,
    ],
)
def test_unreviewed_deployment_cannot_construct_any_public_reader(monkeypatch, implementation):
    monkeypatch.setattr(public_v2.psycopg, "connect", _connection_must_not_open)
    with pytest.raises(RootProofPreparationV2QualificationError, match="reviewed preparation"):
        implementation(
            PostgreSQLConnectionConfig("host=unavailable"),
            deployment=object(),
            schema="caller_selected",
            environment="PRODUCTION",
            trust_domain="trust-domain",
        )


@pytest.mark.parametrize(
    "implementation",
    [
        public_v2.PostgreSQLRootProofIssuanceAuthorityV2,
        public_v2.PostgreSQLEntitlementRegistryProviderV2,
        public_v2.PostgreSQLRequesterCredentialRegistryProviderV2,
        public_v2.PostgreSQLClaimantIdentityRegistryProviderV2,
    ],
)
def test_caller_subclasses_cannot_enter_the_closed_authority_boundary(implementation):
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("UnreviewedPreparationAuthority", (implementation,), {})


def test_public_entitlement_reader_cannot_bind_or_pass_production_readiness(monkeypatch):
    monkeypatch.setattr(public_v2.psycopg, "connect", _connection_must_not_open)
    reader = object.__new__(public_v2.PostgreSQLEntitlementRegistryProviderV2)
    reader._schema = "reviewed_entitlement"
    reader._trust_domain = "trust-domain"
    assert reader.capabilities.authoritative_reads
    assert reader.capabilities.durable_state
    assert not reader.capabilities.compare_and_swap
    assert ProviderQualificationPolicy().failures_for(
        SecurityProfile.PRODUCTION_LOCAL, reader.identity, reader.capabilities
    )
    with pytest.raises(ProductionLocalIssuanceAuthorityError, match="NO_BIND_AUTHORITY"):
        reader.compare_and_swap_bind(object())


def test_explicit_preparation_authority_keeps_the_legacy_production_allowlist():
    assert reservation._TRUSTED_PROVIDER_TYPES == (PostgreSQLRootProofIssuanceAuthority,)
    assert reservation._TRUSTED_PREPARATION_V2_PROVIDER_TYPES == (
        public_v2.PostgreSQLRootProofIssuanceAuthorityV2,
    )
    with pytest.raises(reservation.RootProofAttemptReservationError, match="PROVENANCE_REQUIRED"):
        reservation._resolve_from_provider(object(), object())


@pytest.mark.parametrize(
    "authority",
    [
        object(),
        {},
        b"caller-supplied-public-authority",
        object.__new__(PostgreSQLRootProofIssuanceAuthority),
    ],
)
def test_explicit_v2_resolver_rejects_non_v2_sources_before_selecting_production(
    monkeypatch, authority
):
    monkeypatch.setattr(reservation, "_issuance_authority_provider", _connection_must_not_open)
    before = len(reservation._AUTHORIZATIONS)
    with pytest.raises(reservation.RootProofAttemptReservationError, match="EXACT_V2_PREPARATION"):
        reservation.resolve_root_proof_issuance_authorization_v2(object(), authority)
    assert len(reservation._AUTHORIZATIONS) == before


def test_explicit_v2_resolver_rejects_raw_binding_without_issuing_authorization(monkeypatch):
    monkeypatch.setattr(reservation, "_issuance_authority_provider", _connection_must_not_open)
    authority = object.__new__(public_v2.PostgreSQLRootProofIssuanceAuthorityV2)
    before = len(reservation._AUTHORIZATIONS)
    with pytest.raises(
        AccountInitialBindingError, match="VERIFIED_ACCOUNT_INITIAL_BINDING_REQUIRED"
    ):
        reservation.resolve_root_proof_issuance_authorization_v2(
            b"retained-binding-bytes", authority
        )
    assert len(reservation._AUTHORIZATIONS) == before


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("_component", "history"),
        ("_schema", "caller_selected_entitlement"),
        ("_environment", "TEST"),
        ("_trust_domain", "other-trust-domain"),
    ],
)
def test_mutable_reader_scope_cannot_redirect_pinned_authority(field, replacement):
    reader = object.__new__(public_v2.PostgreSQLEntitlementRegistryProviderV2)
    reader._schema = "reviewed_entitlement"
    reader._environment = "PRODUCTION"
    reader._trust_domain = "trust-domain"
    reader._deployment = SimpleNamespace(
        provisioning=SimpleNamespace(
            entitlement_schema="reviewed_entitlement", trust_domain="trust-domain"
        )
    )
    setattr(reader, field, replacement)
    with pytest.raises(
        ProductionLocalIssuanceAuthorityError, match="PINNED_PUBLIC_AUTHORITY_SCOPE"
    ):
        reader._qualify_connection(object())


class _ReadConnectionProbe:
    """Observe read ordering only; this probe models no transaction outcomes."""

    def __init__(self):
        self.closed = False
        self.qualifications = 0
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def transaction(self):
        return nullcontext()

    def close(self):
        self.closed = True

    def execute(self, query, params=None):
        statement = query if isinstance(query, str) else query.as_string()
        self.queries.append(statement)
        if not statement.startswith("SET "):
            assert self.qualifications == 1, "data read preceded its live qualification"
        return self

    def fetchall(self):
        return []

    def fetchone(self):
        return None


_ACTUAL_READERS = [
    (public_v2.PostgreSQLEntitlementRegistryProviderV2, "retained_history", "subject"),
    (public_v2.PostgreSQLEntitlementRegistryProviderV2, "authoritative_state", "subject"),
    (public_v2.PostgreSQLEntitlementRegistryProviderV2, "state_at_revision", "subject_revision"),
    (public_v2.PostgreSQLRequesterCredentialRegistryProviderV2, "retained_history", "requester"),
    (public_v2.PostgreSQLRequesterCredentialRegistryProviderV2, "credential_identities", "none"),
    (public_v2.PostgreSQLRequesterCredentialRegistryProviderV2, "public_key", "key"),
    (
        public_v2.PostgreSQLRequesterCredentialRegistryProviderV2,
        "historical_generation",
        "generation",
    ),
    (
        public_v2.PostgreSQLRequesterCredentialRegistryProviderV2,
        "active_requester_credential",
        "requester",
    ),
    (
        public_v2.PostgreSQLRequesterCredentialRegistryProviderV2,
        "historical_requester_credential",
        "key",
    ),
    (public_v2.PostgreSQLClaimantIdentityRegistryProviderV2, "retained_history", "claimant"),
    (public_v2.PostgreSQLClaimantIdentityRegistryProviderV2, "credential_identities", "none"),
    (public_v2.PostgreSQLClaimantIdentityRegistryProviderV2, "public_key", "key"),
    (public_v2.PostgreSQLClaimantIdentityRegistryProviderV2, "historical_generation", "generation"),
    (public_v2.PostgreSQLClaimantIdentityRegistryProviderV2, "resolve_claimant", "claimant"),
    (public_v2.PostgreSQLClaimantIdentityRegistryProviderV2, "historical_claimant", "generation"),
]


def _reader_probe(monkeypatch, implementation, *, refuse=False):
    # Public V2 readers import the isolated read-only qualifier since #3119.
    # Exercise that exact live-call seam; the administrative migration module
    # is deliberately not reachable from a public issuer.
    import bot_core.postgresql_root_proof_preparation_readonly as qualification

    connections = []

    def connect(*args, **kwargs):
        conn = _ReadConnectionProbe()
        connections.append(conn)
        return conn

    def qualify(conn, deployment, *, component):
        assert component == implementation._component
        assert not conn.closed
        conn.qualifications += 1
        if refuse:
            raise RootProofPreparationV2QualificationError("live read qualification refused")

    monkeypatch.setattr(public_v2.psycopg, "connect", connect)
    monkeypatch.setattr(qualification, "qualify_root_proof_preparation_v2", qualify)
    reader = object.__new__(implementation)
    reader._schema = "reviewed_public_authority"
    reader._environment = "PRODUCTION"
    reader._trust_domain = "trust-domain"
    reader._connection = PostgreSQLConnectionConfig("host=read-order-probe")
    reader._deployment = SimpleNamespace(
        provisioning=SimpleNamespace(
            **{implementation._component + "_schema": reader._schema},
            trust_domain=reader._trust_domain,
        )
    )
    return reader, connections


def _reader_arguments(selector):
    subject = RegistrySubject("lookup", "PRODUCTION", "trust-domain")
    return {
        "subject": (subject,),
        "subject_revision": (subject, 1),
        "requester": ("CryptoHunterAccountAuthority",),
        "claimant": ("provisioning-principal",),
        "key": ("credential-key",),
        "generation": ("credential-key", 1),
        "none": (),
    }[selector]


@pytest.mark.parametrize(("implementation", "method", "arguments"), _ACTUAL_READERS)
def test_each_public_reader_qualifies_its_actual_connection_once_before_data(
    monkeypatch, implementation, method, arguments
):
    reader, connections = _reader_probe(monkeypatch, implementation)
    try:
        getattr(reader, method)(*_reader_arguments(arguments))
    except CredentialResolutionError:
        # Empty history deliberately has no ACTIVE or historical winner.
        pass
    assert len(connections) == 1
    assert connections[0].qualifications == 1
    assert connections[0].closed
    assert any(not query.startswith("SET ") for query in connections[0].queries)


@pytest.mark.parametrize(("implementation", "method", "arguments"), _ACTUAL_READERS)
def test_each_public_reader_closes_connection_when_live_qualification_refuses(
    monkeypatch, implementation, method, arguments
):
    reader, connections = _reader_probe(monkeypatch, implementation, refuse=True)
    with pytest.raises(RootProofPreparationV2QualificationError, match="live read qualification"):
        getattr(reader, method)(*_reader_arguments(arguments))
    assert len(connections) == 1
    assert connections[0].qualifications == 1
    assert connections[0].closed
    assert all(query.startswith("SET ") for query in connections[0].queries)


@pytest.mark.parametrize(
    "implementation",
    [
        public_v2.PostgreSQLEntitlementRegistryProviderV2,
        public_v2.PostgreSQLRequesterCredentialRegistryProviderV2,
        public_v2.PostgreSQLClaimantIdentityRegistryProviderV2,
    ],
)
def test_explicit_requalification_always_checks_a_new_connection(monkeypatch, implementation):
    reader, connections = _reader_probe(monkeypatch, implementation)
    reader._qualify()
    reader._qualify()
    assert len(connections) == 2
    assert all(conn.qualifications == 1 and conn.closed for conn in connections)


@pytest.mark.parametrize(
    "implementation",
    [
        public_v2.PostgreSQLEntitlementRegistryProviderV2,
        public_v2.PostgreSQLRequesterCredentialRegistryProviderV2,
        public_v2.PostgreSQLClaimantIdentityRegistryProviderV2,
    ],
)
def test_explicit_requalification_refusal_closes_its_connection(monkeypatch, implementation):
    reader, connections = _reader_probe(monkeypatch, implementation, refuse=True)
    with pytest.raises(RootProofPreparationV2QualificationError, match="live read qualification"):
        reader._qualify()
    assert len(connections) == 1
    assert connections[0].qualifications == 1
    assert connections[0].closed
    assert all(query.startswith("SET ") for query in connections[0].queries)
