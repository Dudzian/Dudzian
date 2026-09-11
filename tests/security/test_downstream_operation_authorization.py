"""M0.10-R1 cross-milestone conformance for downstream privileged operations."""

from dataclasses import replace
from types import MappingProxyType

import pytest

from bot_core.security.authentication import (
    AuthenticationError,
    AuthorizationRequest,
    DownstreamOperationDefinition,
    _seed_trusted_downstream_operation_definition,
    downstream_mutation_fingerprint,
    downstream_operation_definition_fingerprint,
    downstream_scope_fingerprint,
)
from bot_core.security.authorization import (
    AuthorizationAuthority,
    AuthorizationError,
    OperationEntitlementProjection,
    _seed_trusted_operation_entitlement,
    operation_entitlement_fingerprint,
)
from tests.security.test_authentication import NOW, RAW_PIN, prepared, request

OPERATIONS = (
    "M0.12/ALERT_ACKNOWLEDGE",
    "M0.12/ALERT_SET_SUPPRESSION",
    "M0.12/ALERT_CLEAR_SUPPRESSION",
    "M0.12/ALERT_MANUAL_FACT_RESOLUTION",
)


def definition(operation: str = OPERATIONS[0], **changes: object) -> DownstreamOperationDefinition:
    item = DownstreamOperationDefinition(
        owner_milestone="M0.12",
        operation=operation,
        factor_policy="PIN",
        freshness_seconds=60,
        authorization_scope="alert_lifecycle",
        environments=("PAPER", "TESTNET", "LIVE"),
        target_scope_contract=("alert_id", "expected_alert_revision", "alert_scope"),
        mutation_binding_contract=("intent", "value"),
        dependency_fingerprint_sha256="d" * 64,
        definition_revision=1,
        content_fingerprint_sha256="",
    )
    item = replace(item, **changes)  # type: ignore[arg-type]
    return replace(
        item,
        content_fingerprint_sha256=downstream_operation_definition_fingerprint(item),
    )


def exact_request(
    item: DownstreamOperationDefinition,
    *,
    alert_id: str = "alert-001",
    revision: int = 7,
    environment: str = "PAPER",
    intent: str = "ACKNOWLEDGE",
    causation: str = "cause-1",
    correlation: str = "correlation-1",
) -> AuthorizationRequest:
    provisional = request(
        item.operation,
        environment=environment,
        causation_id=causation,
        correlation_id=correlation,
        scope_fingerprint_sha256="0" * 64,
        mutation_fingerprint_sha256="0" * 64,
    )
    target = {
        "alert_id": alert_id,
        "expected_alert_revision": revision,
        "alert_scope": "account-alerts",
    }
    mutation = {"intent": intent, "value": True}
    scoped = replace(
        provisional,
        scope_fingerprint_sha256=downstream_scope_fingerprint(item, provisional, target),
    )
    return replace(
        scoped,
        mutation_fingerprint_sha256=downstream_mutation_fingerprint(item, scoped, target, mutation),
    )


def entitlement(req: AuthorizationRequest) -> OperationEntitlementProjection:
    item = OperationEntitlementProjection(
        req.account_id,
        req.operator_id,
        req.operation,
        req.environment,
        "alert_lifecycle",
        1,
        1,
        "",
    )
    return replace(item, content_fingerprint_sha256=operation_entitlement_fingerprint(item))


def arranged(tmp_path, operation: str = OPERATIONS[0]):  # type: ignore[no-untyped-def]
    _, authentication, _ = prepared(tmp_path)
    item = definition(operation)
    _seed_trusted_downstream_operation_definition(authentication, item)
    req = exact_request(item, intent=operation.split("ALERT_", 1)[-1])
    proof = authentication.issue_authentication_proof(req, RAW_PIN, NOW)
    authority = AuthorizationAuthority(authentication)
    return authentication, authority, item, req, proof


@pytest.mark.parametrize("operation", OPERATIONS)
def test_each_required_alert_operation_uses_exact_core_authority(tmp_path, operation: str) -> None:  # type: ignore[no-untyped-def]
    _, authority, _, req, proof = arranged(tmp_path, operation)
    with pytest.raises(AuthorizationError, match="AUTHORIZATION_DENIED"):
        authority.authorize(proof, req, NOW)
    _seed_trusted_operation_entitlement(authority, entitlement(req))
    assert authority.authorize(proof, req, NOW) == "AUTHORIZED"


def test_unknown_nominal_and_noncurrent_definitions_fail_closed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _, authentication, _ = prepared(tmp_path)
    item = definition()
    req = exact_request(item)
    with pytest.raises(AuthenticationError, match="OPERATION_UNSUPPORTED"):
        authentication.issue_authentication_proof(req, RAW_PIN, NOW)
    assert downstream_operation_definition_fingerprint(item) == item.content_fingerprint_sha256
    _seed_trusted_downstream_operation_definition(authentication, item, current=False)
    with pytest.raises(AuthenticationError, match="OPERATION_UNSUPPORTED"):
        authentication.issue_authentication_proof(req, RAW_PIN, NOW)


def test_caller_cannot_supply_or_register_definition(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _, authority, item, req, proof = arranged(tmp_path)
    assert not any(
        hasattr(type(authority), name)
        for name in ("register_operation", "accept_operation", "set_operation")
    )
    with pytest.raises(TypeError):
        authority.authorize(proof, req, NOW, item)  # type: ignore[call-arg]


def test_owner_mismatch_is_rejected_without_mutation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _, authentication, _ = prepared(tmp_path)
    bad = definition(owner_milestone="M0.11")
    before = authentication.snapshot
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
        _seed_trusted_downstream_operation_definition(authentication, bad)
    assert authentication.snapshot == before


@pytest.mark.parametrize(
    "changed",
    (
        {"operation": OPERATIONS[1]},
        {"alert_id": "alert-002"},
        {"revision": 8},
        {"environment": "LIVE"},
        {"intent": "SET_SUPPRESSION"},
        {"causation": "cause-2"},
        {"correlation": "correlation-2"},
    ),
)
def test_downstream_proof_replay_across_exact_context_fails(tmp_path, changed) -> None:  # type: ignore[no-untyped-def]
    _, authority, item, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(authority, entitlement(req))
    if "operation" in changed:
        replay = replace(req, operation=changed["operation"])
    else:
        replay = exact_request(item, **changed)
    with pytest.raises(AuthorizationError, match="AUTHORIZATION_DENIED|OPERATION_UNSUPPORTED"):
        authority.authorize(proof, replay, NOW)


def test_definition_drift_fences_previously_issued_proof(tmp_path) -> None:  # type: ignore[no-untyped-def]
    authentication, authority, item, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(authority, entitlement(req))
    successor = definition(definition_revision=2, dependency_fingerprint_sha256="e" * 64)
    _seed_trusted_downstream_operation_definition(authentication, successor)
    with pytest.raises(AuthorizationError, match="PROOF_STALE"):
        authority.authorize(proof, req, NOW)


def test_definition_registries_are_deeply_immutable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    authentication, _, _, _, _ = arranged(tmp_path)
    assert isinstance(
        authentication.snapshot.accepted_downstream_operation_definitions, MappingProxyType
    )
    assert isinstance(
        authentication.snapshot.current_downstream_operation_definitions, MappingProxyType
    )
