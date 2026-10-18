"""Boundary logic only: simulated IPC never proves installed process isolation."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from bot_core import production_root_lifecycle as boundary
from bot_core import production_signer_custody as custody
from bot_core import production_signer_ipc as ipc
from tests.security.test_production_root_lifecycle_identity import installed


def _facts():
    return {
        "semantic_role": "ROOT_PROOF_SIGNING",
        "authority_identity": "root-authority",
        "identity": "root-credential",
        "public_material_identity": "sha256:" + "a" * 64,
        "key_version": 1,
        "lifecycle_generation": 1,
        "state": "ACTIVE",
    }


def _row(facts=None, **changes):
    value = dict(facts or _facts())
    value.update(pending_operation_id=None, is_current=True, new_operations_authorized=False)
    value.update(changes)
    return (value,)


@pytest.mark.parametrize(
    "field,value",
    [
        ("semantic_role", "HISTORY_ATTESTATION_SIGNING"),
        ("authority_identity", "history-authority"),
        ("identity", "replaced-credential"),
        ("public_material_identity", "sha256:" + "b" * 64),
        ("key_version", 2),
        ("key_version", True),
        ("lifecycle_generation", 2),
        ("lifecycle_generation", True),
        ("state", "VERIFY_ONLY"),
        ("pending_operation_id", "uncommitted-close"),
        ("new_operations_authorized", True),
        ("is_current", 1),
    ],
)
def test_independent_lifecycle_mismatch_fails_closed(field, value):
    with pytest.raises(boundary.ProductionRootLifecycleError, match="MISMATCH"):
        boundary._compare_signer_lifecycle_facts(_facts(), _row(**{field: value}))


@pytest.mark.parametrize("row", [None, (), (None,), ({},), [None], ({"state": "ACTIVE"},)])
def test_missing_or_ambiguous_independent_observation_never_qualifies(row):
    with pytest.raises(boundary.ProductionRootLifecycleError, match="OBSERVATION_REQUIRED"):
        boundary._compare_signer_lifecycle_facts(_facts(), row)


@pytest.mark.parametrize("state", ["ACTIVE", "VERIFY_ONLY", "REVOKED", "PENDING_CLOSED"])
def test_observation_keeps_closed_and_retired_states_without_authorization(state):
    facts = dict(_facts(), state=state if state != "PENDING_CLOSED" else "ACTIVE")
    row = _row(
        facts, state=state, pending_operation_id="close" if state == "PENDING_CLOSED" else None
    )
    observed = boundary._compare_signer_lifecycle_facts(facts, row)
    assert observed["state"] == state
    assert observed["new_operations_authorized"] is False


@pytest.fixture
def diagnostic_boundary(installed, monkeypatch):
    installed_boundary = boundary._installed_boundary()
    service = boundary.ProductionRootLifecycleBoundary(installed_boundary, "issuer")
    monkeypatch.setattr(service, "_current", lambda: installed_boundary)
    monkeypatch.setattr(service, "qualify", lambda connection: None)
    facts = dict(
        _facts(),
        trust_domain=installed_boundary.pin.provisioning.trust_domain,
        installation_sha256=installed_boundary.source_sha256,
    )
    # Validation has its own strict untrusted-wire tests. These mocks exercise
    # only orchestration and independent SQL comparison, never physical custody.
    monkeypatch.setattr(custody, "validate_signer_custody_facts", lambda role, value: dict(value))
    observation = SimpleNamespace(
        peer_pid=1234,
        peer_uid=123,
        peer_gid=456,
        peer_namespaces=(),
        response={"facts": facts, "blockers": ["SELF_ASSERTION_IS_NOT_ATTESTATION"]},
    )
    calls = []
    monkeypatch.setattr(
        ipc, "qualify_signer_ipc", lambda policy, operation="DIAGNOSTIC": observation
    )

    def execute(query, arguments):
        calls.append((query, arguments))
        return SimpleNamespace(fetchone=lambda: _row())

    return service, observation, SimpleNamespace(execute=execute), calls


def test_matching_public_facts_stay_blocked(diagnostic_boundary):
    service, observation, connection, calls = diagnostic_boundary
    result = service.qualify_signer_custody(connection, "root_signer")
    assert (
        result["status"]
        == result["private_control_status"]
        == result["production_emission"]
        == "BLOCKED"
    )
    assert result["public_binding_matches"] is True
    assert "INDEPENDENT_PRIVATE_KEY_CONTROL_UNPROVEN" in result["blockers"]
    assert "INDEPENDENT_SIGNER_NAMESPACE_EVIDENCE_UNAVAILABLE" in result["blockers"]
    assert len(calls) == 1
    assert calls[0][1] == ("ROOT_PROOF_SIGNING", "root-credential")


def test_wrong_signer_role_is_rejected_before_ipc(diagnostic_boundary):
    service, _, connection, calls = diagnostic_boundary
    with pytest.raises(boundary.ProductionRootLifecycleError, match="SIGNER_ROLE"):
        service.qualify_signer_custody(connection, "requester")
    assert not calls


@pytest.mark.parametrize("field", ["trust_domain", "installation_sha256"])
def test_signer_cannot_substitute_another_installation(diagnostic_boundary, field):
    service, observation, connection, calls = diagnostic_boundary
    observation.response["facts"][field] = "substitution"
    with pytest.raises(boundary.ProductionRootLifecycleError, match="INSTALLATION_BINDING"):
        service.qualify_signer_custody(connection, "root_signer")
    assert not calls


def test_shared_namespace_fails_closed_when_independently_observable(diagnostic_boundary):
    service, observation, connection, _ = diagnostic_boundary
    observation.peer_namespaces = tuple(
        (name, os.stat(f"/proc/self/ns/{name}").st_ino) for name in ("mnt", "ipc", "net")
    )
    with pytest.raises(boundary.ProductionRootLifecycleError, match="NAMESPACE_NOT_SEPARATED"):
        service.qualify_signer_custody(connection, "root_signer")


def test_lost_ipc_is_not_replayed(diagnostic_boundary, monkeypatch):
    service, _, connection, calls = diagnostic_boundary
    attempts = []

    def disconnected(*args):
        attempts.append(args)
        raise ipc.ProductionSignerIPCError("disconnect")

    monkeypatch.setattr(ipc, "qualify_signer_ipc", disconnected)
    with pytest.raises(boundary.ProductionRootLifecycleError, match="CUSTODY_UNAVAILABLE"):
        service.qualify_signer_custody(connection, "root_signer")
    assert len(attempts) == 1
    assert not calls


def test_postgresql_unavailable_is_not_retried(diagnostic_boundary):
    service, _, _, calls = diagnostic_boundary
    attempts = []

    def unavailable(*args):
        attempts.append(args)
        raise OSError("isolated authority unavailable")

    with pytest.raises(boundary.ProductionRootLifecycleError, match="QUALIFICATION_UNAVAILABLE"):
        service.qualify_signer_custody(SimpleNamespace(execute=unavailable), "root_signer")
    assert len(attempts) == 1
    assert not calls


def test_restart_identity_result_cannot_authorize_bind(diagnostic_boundary):
    service, _, _, _ = diagnostic_boundary
    assert service.qualify_signer_peer("root_signer") == (1234, 123, 456)
    with pytest.raises(boundary.ProductionRootLifecycleError, match="PRODUCTION_BIND_BLOCKED"):
        service.commit()


def test_signer_factory_has_no_endpoint_or_identity_override():
    with pytest.raises(TypeError):
        boundary.open_installed_production_signer_policy("root_signer", path="/tmp/override")


@pytest.mark.parametrize("shared", ["mnt", "ipc", "net"])
def test_signers_cannot_share_an_observed_namespace(shared):
    first = {"mnt": 1, "ipc": 2, "net": 3}
    second = {"mnt": 11, "ipc": 12, "net": 13}
    second[shared] = first[shared]
    with pytest.raises(boundary.ProductionRootLifecycleError, match="NOT_INDEPENDENT"):
        boundary._compare_signer_namespaces(
            [
                {"peer_namespaces": first},
                {"peer_namespaces": second},
            ]
        )


def test_distinct_observed_namespaces_clear_only_that_specific_blocker():
    observations = [
        {
            "peer_namespaces": {"mnt": 1, "ipc": 2, "net": 3},
            "blockers": ["CROSS_SIGNER_NAMESPACE_ISOLATION_UNPROVEN", "PRIVATE_CONTROL_UNPROVEN"],
        },
        {
            "peer_namespaces": {"mnt": 11, "ipc": 12, "net": 13},
            "blockers": ["CROSS_SIGNER_NAMESPACE_ISOLATION_UNPROVEN", "PRIVATE_CONTROL_UNPROVEN"],
        },
    ]
    boundary._compare_signer_namespaces(observations)
    assert all(value["blockers"] == ["PRIVATE_CONTROL_UNPROVEN"] for value in observations)


def test_unreadable_namespace_evidence_keeps_cross_signer_blocker():
    observations = [
        {"peer_namespaces": {}, "blockers": ["CROSS_SIGNER_NAMESPACE_ISOLATION_UNPROVEN"]}
    ] * 2
    boundary._compare_signer_namespaces(observations)
    assert observations[0]["blockers"] == ["CROSS_SIGNER_NAMESPACE_ISOLATION_UNPROVEN"]
