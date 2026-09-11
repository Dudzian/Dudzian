"""M0.10-R1 cross-milestone conformance for downstream privileged operations."""

from dataclasses import replace
from types import MappingProxyType

import pytest

from bot_core.security.authentication import (
    AuthenticationError,
    AuthorizationRequest,
    DownstreamOperationDefinition,
    _architecture_downstream_definitions,
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
    item = next(
        item for item in _architecture_downstream_definitions() if item.operation == operation
    )
    if not changes:
        return item
    item = replace(item, **changes)  # type: ignore[arg-type]
    return replace(
        item, content_fingerprint_sha256=downstream_operation_definition_fingerprint(item)
    )


def exact_request(
    item: DownstreamOperationDefinition,
    *,
    alert_id: str = "alert-001",
    revision: int = 7,
    environment: str = "PAPER",
    intent: str | None = None,
    causation: str = "cause-1",
    correlation: str = "correlation-1",
) -> AuthorizationRequest:
    intent = item.declared_intent if intent is None else intent
    provisional = request(
        item.operation,
        environment=environment,
        declared_intent=intent,
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
    assert downstream_operation_definition_fingerprint(item) == item.content_fingerprint_sha256
    with authentication._state.lock:  # noqa: SLF001
        authentication._state.snapshot = replace(  # noqa: SLF001
            authentication.snapshot,
            current_downstream_operation_definitions=MappingProxyType({}),
        )
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
    elif "intent" in changed:
        with pytest.raises(AuthenticationError, match="MALFORMED_UNTRUSTED_CONTEXT"):
            exact_request(item, **changed)
        return
    else:
        replay = exact_request(item, **changed)
    with pytest.raises(AuthorizationError, match="AUTHORIZATION_DENIED|OPERATION_UNSUPPORTED"):
        authority.authorize(proof, replay, NOW)


def test_duplicate_exact_current_definition_is_idempotent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    authentication, _, item, _, _ = arranged(tmp_path)
    before = authentication.snapshot
    _seed_trusted_downstream_operation_definition(authentication, item)
    assert authentication.snapshot == before


def test_arbitrary_dependency_and_undeclared_operation_do_not_establish_provenance(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    _, authentication, _ = prepared(tmp_path)
    item = definition()
    changes = (
        {"dependency_fingerprint_sha256": "a" * 64},
        {"operation": "M0.12/FAKE_OPERATION"},
        {"owner_milestone": "M0.99", "operation": "M0.99/FAKE_OPERATION"},
        {"owner_artifact": "identity_device_authentication_and_secrets.json"},
        {"owner_json_pointer": "/downstream_operation_declarations/1"},
    )
    for change in changes:
        bad = replace(item, **change, content_fingerprint_sha256="")
        bad = replace(
            bad, content_fingerprint_sha256=downstream_operation_definition_fingerprint(bad)
        )
        with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
            _seed_trusted_downstream_operation_definition(authentication, bad)


def test_definition_registries_are_deeply_immutable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    authentication, _, _, _, _ = arranged(tmp_path)
    assert isinstance(
        authentication.snapshot.accepted_downstream_operation_definitions, MappingProxyType
    )
    assert isinstance(
        authentication.snapshot.current_downstream_operation_definitions, MappingProxyType
    )


def test_proof_definition_binding_malformed_state_is_contract_inconsistent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    authentication, authority, item, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(authority, entitlement(req))
    cases = ({}, {proof.proof_fingerprint_sha256: "f" * 64})
    for bindings in cases:
        with authentication._state.lock:  # noqa: SLF001
            authentication._state.snapshot = replace(  # noqa: SLF001
                authentication.snapshot,
                authentication_proof_operation_definitions=MappingProxyType(bindings),
            )
        with pytest.raises(AuthorizationError, match="CONTRACT_INCONSISTENT"):
            authority.authorize(proof, req, NOW)
    other = definition(OPERATIONS[1])
    with authentication._state.lock:  # noqa: SLF001
        authentication._state.snapshot = replace(  # noqa: SLF001
            authentication.snapshot,
            authentication_proof_operation_definitions=MappingProxyType(
                {proof.proof_fingerprint_sha256: other.content_fingerprint_sha256}
            ),
        )
    with pytest.raises(AuthorizationError, match="CONTRACT_INCONSISTENT"):
        authority.authorize(proof, req, NOW)


def test_restart_bootstraps_definitions_and_invalidates_proofs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from bot_core.security.authentication import AuthenticationAuthority
    from tests.security.test_authentication import Comparator

    security, authentication, _ = prepared(tmp_path)
    item = definition()
    req = exact_request(item)
    proof = authentication.issue_authentication_proof(req, RAW_PIN, NOW)
    with security._state.lock:  # noqa: SLF001
        security._state.snapshot = replace(  # noqa: SLF001
            security._state.snapshot,  # noqa: SLF001
            accepted_downstream_operation_definitions=MappingProxyType({}),
            current_downstream_operation_definitions=MappingProxyType({}),
        )
    restarted = AuthenticationAuthority(security, Comparator())
    assert restarted.snapshot.current_downstream_operation_definitions[item.operation]
    assert restarted.snapshot.accepted_downstream_operation_definitions
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_REQUIRED"):
        restarted.resolve_accepted_proof(proof)
    assert restarted.snapshot.authentication_proof_operation_definitions == {}


def test_actual_pointer_drift_fences_unchanged_descriptor(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import json
    import shutil
    import bot_core.security.authentication as module

    root = tmp_path / "architecture"
    root.mkdir()
    source = module._ARCHITECTURE_ROOT / "audit_observability_alerts_and_updater.json"  # noqa: SLF001
    artifact = root / source.name
    shutil.copyfile(source, artifact)
    monkeypatch.setattr(module, "_ARCHITECTURE_ROOT", root)
    authentication, authority, item, req, proof = arranged(tmp_path / "runtime")
    _seed_trusted_operation_entitlement(authority, entitlement(req))
    document = json.loads(artifact.read_text())
    document["downstream_operation_declarations"][0]["freshness_seconds"] = 61
    artifact.write_text(json.dumps(document, indent=2) + "\n")
    with pytest.raises(AuthorizationError, match="CONTRACT_INCONSISTENT"):
        authority.authorize(proof, req, NOW)


def test_monotonic_successor_prevents_proof_resurrection(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import json
    import shutil
    import bot_core.security.authentication as module

    root = tmp_path / "architecture"
    root.mkdir()
    source = module._ARCHITECTURE_ROOT / "audit_observability_alerts_and_updater.json"  # noqa: SLF001
    artifact = root / source.name
    shutil.copyfile(source, artifact)
    monkeypatch.setattr(module, "_ARCHITECTURE_ROOT", root)
    authentication, authority, rev1, req, proof = arranged(tmp_path / "runtime")
    _seed_trusted_operation_entitlement(authority, entitlement(req))
    document = json.loads(artifact.read_text())
    successor = dict(document["downstream_operation_declarations"][0])
    successor["definition_revision"] = 2
    successor["freshness_seconds"] = 61
    document["downstream_operation_declarations"].append(successor)
    document["current_downstream_operation_definition_revisions"][rev1.operation] = 2
    artifact.write_text(json.dumps(document, indent=2) + "\n")
    rev2 = next(
        item
        for item in module._architecture_downstream_definitions()  # noqa: SLF001
        if item.operation == rev1.operation and item.definition_revision == 2
    )
    _seed_trusted_downstream_operation_definition(authentication, rev2)
    with pytest.raises(AuthorizationError, match="PROOF_STALE"):
        authority.authorize(proof, req, NOW)
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
        _seed_trusted_downstream_operation_definition(authentication, rev1)
    with pytest.raises(AuthorizationError, match="PROOF_STALE"):
        authority.authorize(proof, req, NOW)
    _seed_trusted_downstream_operation_definition(authentication, rev2)
    altered = replace(rev2, freshness_seconds=62, content_fingerprint_sha256="")
    altered = replace(
        altered, content_fingerprint_sha256=downstream_operation_definition_fingerprint(altered)
    )
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
        _seed_trusted_downstream_operation_definition(authentication, altered)


def test_downstream_mutation_boundary_compares_actual_values(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _, authority, _, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(authority, entitlement(req))
    target = {
        "alert_id": "alert-001",
        "expected_alert_revision": 7,
        "alert_scope": "account-alerts",
    }
    mutation = {"intent": "ACKNOWLEDGE", "value": True}
    assert (
        authority.validate_downstream_authorized_mutation(proof, req, NOW, target, mutation)
        == "AUTHORIZED_MUTATION"
    )
    with pytest.raises(AuthorizationError, match="AUTHORIZATION_DENIED"):
        authority.validate_downstream_authorized_mutation(
            proof, req, NOW, {**target, "expected_alert_revision": 8}, mutation
        )


@pytest.mark.parametrize(
    ("operation", "wrong_intent"),
    (
        (OPERATIONS[0], "SET_SUPPRESSION"),
        (OPERATIONS[0], "MANUAL_FACT_RESOLUTION"),
        (OPERATIONS[1], "CLEAR_SUPPRESSION"),
        (OPERATIONS[2], "ACKNOWLEDGE"),
        (OPERATIONS[3], "ACKNOWLEDGE"),
    ),
)
def test_canonical_operation_rejects_wrong_intent_before_proof_issuance(
    tmp_path, operation: str, wrong_intent: str
) -> None:  # type: ignore[no-untyped-def]
    _, authentication, _ = prepared(tmp_path)
    item = definition(operation)
    with pytest.raises(AuthenticationError, match="MALFORMED_UNTRUSTED_CONTEXT"):
        exact_request(item, intent=wrong_intent)
    canonical_request = exact_request(item)
    mismatched_request = replace(
        canonical_request,
        declared_intent=wrong_intent,
        mutation_fingerprint_sha256="a" * 64,
    )
    with pytest.raises(AuthenticationError, match="AUTHORIZATION_DENIED"):
        authentication.issue_authentication_proof(mismatched_request, RAW_PIN, NOW)
    assert not authentication.snapshot.accepted_authentication_proofs


@pytest.mark.parametrize("operation", OPERATIONS)
def test_each_canonical_intent_reaches_actual_mutation_authority(tmp_path, operation: str) -> None:  # type: ignore[no-untyped-def]
    _, authority, item, req, proof = arranged(tmp_path, operation)
    _seed_trusted_operation_entitlement(authority, entitlement(req))
    target = {
        "alert_id": "alert-001",
        "expected_alert_revision": 7,
        "alert_scope": "account-alerts",
    }
    mutation = {"intent": item.declared_intent, "value": True}
    assert (
        authority.validate_downstream_authorized_mutation(proof, req, NOW, target, mutation)
        == "AUTHORIZED_MUTATION"
    )


def _write_corrupt_architecture(root, transform) -> None:  # type: ignore[no-untyped-def]
    import json
    import shutil
    import bot_core.security.authentication as module

    root.mkdir()
    source = module._ARCHITECTURE_ROOT / "audit_observability_alerts_and_updater.json"  # noqa: SLF001
    artifact = root / source.name
    shutil.copyfile(source, artifact)
    document = json.loads(artifact.read_text())
    transform(document)
    artifact.write_text(json.dumps(document, indent=2) + "\n")


@pytest.mark.parametrize(
    "transform",
    (
        lambda d: d["current_downstream_operation_definition_revisions"].__setitem__(
            OPERATIONS[0], 2
        ),
        lambda d: (
            d["downstream_operation_declarations"].append(
                {**d["downstream_operation_declarations"][0], "definition_revision": 3}
            ),
            d["current_downstream_operation_definition_revisions"].__setitem__(OPERATIONS[0], 3),
        ),
        lambda d: d["downstream_operation_declarations"].append(
            {**d["downstream_operation_declarations"][0], "freshness_seconds": 61}
        ),
        lambda d: d["current_downstream_operation_definition_revisions"].__setitem__(
            "M0.12/UNDECLARED", 1
        ),
        lambda d: d["current_downstream_operation_definition_revisions"].pop(OPERATIONS[0]),
        lambda d: d["current_downstream_operation_definition_revisions"].__setitem__(
            OPERATIONS[0], True
        ),
        lambda d: d["current_downstream_operation_definition_revisions"].__setitem__(
            OPERATIONS[0], 0
        ),
        lambda d: d["current_downstream_operation_definition_revisions"].__setitem__(
            OPERATIONS[0], -1
        ),
        lambda d: d["current_downstream_operation_definition_revisions"].__setitem__(
            OPERATIONS[0], "1"
        ),
    ),
)
def test_restart_bootstrap_rejects_inexact_current_history(
    tmp_path, monkeypatch, transform
) -> None:  # type: ignore[no-untyped-def]
    import bot_core.security.authentication as module

    root = tmp_path / "architecture"
    _write_corrupt_architecture(root, transform)
    monkeypatch.setattr(module, "_ARCHITECTURE_ROOT", root)
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
        module._architecture_downstream_definitions()  # noqa: SLF001


def test_restart_replays_contiguous_history_to_exact_declared_current(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    import bot_core.security.authentication as module

    def revisions(document) -> None:  # type: ignore[no-untyped-def]
        base = document["downstream_operation_declarations"][0]
        document["downstream_operation_declarations"].extend(
            ({**base, "definition_revision": 2}, {**base, "definition_revision": 3})
        )
        document["current_downstream_operation_definition_revisions"][OPERATIONS[0]] = 3

    root = tmp_path / "architecture"
    _write_corrupt_architecture(root, revisions)
    monkeypatch.setattr(module, "_ARCHITECTURE_ROOT", root)
    _, authentication, _ = prepared(tmp_path / "runtime")
    fingerprint = authentication.snapshot.current_downstream_operation_definitions[OPERATIONS[0]]
    current = authentication.snapshot.accepted_downstream_operation_definitions[fingerprint]
    assert current.definition_revision == 3
    assert sorted(
        item.definition_revision
        for item in authentication.snapshot.accepted_downstream_operation_definitions.values()
        if item.operation == OPERATIONS[0]
    ) == [1, 2, 3]


def test_mutation_validation_uses_one_locked_semantic_core() -> None:
    import inspect
    from bot_core.security.authorization import AuthorizationAuthority

    source = inspect.getsource(AuthorizationAuthority.validate_downstream_authorized_mutation)
    assert "self.authorize(" not in source
    assert source.count("with self._state.lock:") == 1
    assert "self._authorize_against_snapshot_locked(" in source


@pytest.mark.parametrize(
    "authority_change",
    (
        "identity_revoked",
        "device_revoked",
        "security_generation_advanced",
        "session_generation_advanced",
        "entitlement_removed",
        "definition_advanced",
    ),
)
def test_authority_change_before_combined_locked_decision_never_authorizes_mutation(
    tmp_path, monkeypatch, authority_change: str
) -> None:  # type: ignore[no-untyped-def]
    from bot_core.security.authorization import AuthorizationAuthority

    authentication, authority, _, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(authority, entitlement(req))
    original = AuthorizationAuthority._authorize_against_snapshot_locked

    def changed_before_decision(self, candidate, candidate_request, now):  # type: ignore[no-untyped-def]
        snapshot = self._state.snapshot
        updates = {}
        if authority_change in {"identity_revoked", "security_generation_advanced"}:
            accepted = dict(snapshot.accepted_identities)
            fingerprint = snapshot.current_identities[(req.account_id, req.operator_id)]
            current = accepted[fingerprint]
            accepted[fingerprint] = replace(
                current,
                state="REVOKED" if authority_change == "identity_revoked" else current.state,
                security_generation=(
                    current.security_generation + 1
                    if authority_change == "security_generation_advanced"
                    else current.security_generation
                ),
            )
            updates["accepted_identities"] = MappingProxyType(accepted)
        elif authority_change == "device_revoked":
            accepted = dict(snapshot.accepted_devices)
            fingerprint = snapshot.current_devices[(req.account_id, req.device_installation_id)]
            accepted[fingerprint] = replace(accepted[fingerprint], state="REVOKED")
            updates["accepted_devices"] = MappingProxyType(accepted)
        elif authority_change == "session_generation_advanced":
            accepted = dict(snapshot.accepted_sessions)
            key = (req.account_id, req.operator_id, req.device_installation_id)
            fingerprint = snapshot.current_sessions[key]
            current = accepted[fingerprint]
            accepted[fingerprint] = replace(
                current, session_generation=current.session_generation + 1
            )
            updates["accepted_sessions"] = MappingProxyType(accepted)
        elif authority_change == "entitlement_removed":
            updates["current_operation_entitlements"] = MappingProxyType({})
        else:
            accepted = dict(snapshot.accepted_downstream_operation_definitions)
            fingerprint = snapshot.current_downstream_operation_definitions[req.operation]
            current = accepted[fingerprint]
            advanced = replace(
                current,
                definition_revision=current.definition_revision + 1,
                content_fingerprint_sha256="",
            )
            advanced = replace(
                advanced,
                content_fingerprint_sha256=downstream_operation_definition_fingerprint(advanced),
            )
            accepted[advanced.content_fingerprint_sha256] = advanced
            updates["accepted_downstream_operation_definitions"] = MappingProxyType(accepted)
            updates["current_downstream_operation_definitions"] = MappingProxyType(
                {
                    **snapshot.current_downstream_operation_definitions,
                    req.operation: advanced.content_fingerprint_sha256,
                }
            )
        self._state.snapshot = replace(snapshot, **updates)
        return original(self, candidate, candidate_request, now)

    monkeypatch.setattr(
        AuthorizationAuthority, "_authorize_against_snapshot_locked", changed_before_decision
    )
    target = {
        "alert_id": "alert-001",
        "expected_alert_revision": 7,
        "alert_scope": "account-alerts",
    }
    mutation = {"intent": "ACKNOWLEDGE", "value": True}
    with pytest.raises(AuthorizationError):
        authority.validate_downstream_authorized_mutation(proof, req, NOW, target, mutation)


@pytest.mark.parametrize("operation", ("LOCK_SESSION", "ROTATE_SECRET_REFERENCE"))
def test_builtin_declared_intent_is_rejected_before_challenge_or_proof_publication(
    tmp_path, operation: str
) -> None:  # type: ignore[no-untyped-def]
    _, authentication, _ = prepared(tmp_path)
    malicious = request(operation, declared_intent="ATTACKER_CONTROLLED")
    with pytest.raises(AuthenticationError, match="AUTHORIZATION_DENIED"):
        authentication.derive_platform_biometric_challenge(malicious)
    with pytest.raises(AuthenticationError, match="AUTHORIZATION_DENIED"):
        authentication.issue_authentication_proof(malicious, RAW_PIN, NOW)
    assert not authentication.snapshot.accepted_authentication_proofs


def test_builtin_request_intent_mutation_cannot_reuse_genuine_proof(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from tests.security.test_authorization import entitlement as builtin_entitlement

    _, authentication, _ = prepared(tmp_path)
    req = request("LOCK_SESSION")
    proof = authentication.issue_authentication_proof(req, RAW_PIN, NOW)
    authority = AuthorizationAuthority(authentication)
    _seed_trusted_operation_entitlement(authority, builtin_entitlement(req))
    changed = replace(req, declared_intent="ATTACKER_CONTROLLED")
    with pytest.raises(AuthorizationError, match="AUTHORIZATION_DENIED"):
        authority.authorize(proof, changed, NOW)


def test_upstream_handoff_intent_mutation_cannot_reuse_genuine_proof(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from tests.security.test_authorization import (
        _upstream_arranged,
        entitlement as builtin_entitlement,
    )

    _, _, authority, req, proof = _upstream_arranged(tmp_path)
    _seed_trusted_operation_entitlement(authority, builtin_entitlement(req))
    changed = replace(req, declared_intent="ATTACKER_CONTROLLED")
    with pytest.raises(AuthorizationError, match="AUTHORIZATION_DENIED"):
        authority.authorize(proof, changed, NOW)
    assert (
        authority.authorize_upstream_security_request(proof, changed, NOW) == "AUTHORIZATION_DENIED"
    )


@pytest.mark.parametrize(
    "transform",
    (
        lambda d: d["downstream_operation_declarations"][0].pop("declared_intent"),
        lambda d: d["downstream_operation_declarations"][0].pop("factor_policy"),
        lambda d: d.pop("schema_version"),
        lambda d: d["downstream_operation_declarations"][0].__setitem__("environments", None),
        lambda d: d["downstream_operation_declarations"][0].__setitem__(
            "target_scope_contract", None
        ),
        lambda d: d["downstream_operation_declarations"][0].__setitem__(
            "mutation_binding_contract", None
        ),
        lambda d: d["downstream_operation_declarations"][0].__setitem__("freshness_seconds", "60"),
        lambda d: d["downstream_operation_declarations"][0].__setitem__(
            "declared_intent", ["ACKNOWLEDGE"]
        ),
    ),
)
def test_malformed_canonical_declaration_always_fails_controlled(
    tmp_path, monkeypatch, transform
) -> None:  # type: ignore[no-untyped-def]
    import bot_core.security.authentication as module

    root = tmp_path / "architecture"
    _write_corrupt_architecture(root, transform)
    monkeypatch.setattr(module, "_ARCHITECTURE_ROOT", root)
    with pytest.raises(AuthenticationError) as caught:
        module._architecture_downstream_definitions()  # noqa: SLF001
    assert caught.value.reason == "CONTRACT_INCONSISTENT"


def test_restart_atomically_replaces_extra_ephemeral_definition_authority(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from bot_core.security.authentication import (
        AuthenticationAuthority,
        _architecture_downstream_definitions,
    )
    from tests.security.test_authentication import Comparator

    security, _, _ = prepared(tmp_path)
    extra = object()
    with security._state.lock:  # noqa: SLF001
        security._state.snapshot = replace(  # noqa: SLF001
            security._state.snapshot,  # noqa: SLF001
            accepted_downstream_operation_definitions=MappingProxyType({"extra": extra}),
            current_downstream_operation_definitions=MappingProxyType({"M0.99/EXTRA": "extra"}),
        )
    restarted = AuthenticationAuthority(security, Comparator())
    canonical = _architecture_downstream_definitions()
    expected_accepted = {item.content_fingerprint_sha256: item for item in canonical}
    expected_current = {item.operation: item.content_fingerprint_sha256 for item in canonical}
    assert dict(restarted.snapshot.accepted_downstream_operation_definitions) == expected_accepted
    assert dict(restarted.snapshot.current_downstream_operation_definitions) == expected_current


def test_failed_restart_validation_publishes_no_partial_authority(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import bot_core.security.authentication as module
    from tests.security.test_authentication import Comparator

    security, authentication, _ = prepared(tmp_path / "runtime")
    req = exact_request(definition())
    authentication.issue_authentication_proof(req, RAW_PIN, NOW)
    before = security._state.snapshot  # noqa: SLF001
    root = tmp_path / "architecture"
    _write_corrupt_architecture(
        root, lambda d: d["downstream_operation_declarations"][0].pop("declared_intent")
    )
    monkeypatch.setattr(module, "_ARCHITECTURE_ROOT", root)
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
        module.AuthenticationAuthority(security, Comparator())
    assert security._state.snapshot is before  # noqa: SLF001


@pytest.mark.parametrize(
    "transform",
    (
        lambda document: document.pop("m0_element"),
        lambda document: document.__setitem__("m0_element", 12),
        lambda document: document.__setitem__("m0_element", "M0.99"),
    ),
)
def test_bootstrap_rejects_actual_artifact_owner_corruption_before_publication(
    tmp_path, monkeypatch, transform
) -> None:  # type: ignore[no-untyped-def]
    import bot_core.security.authentication as module

    root = tmp_path / "architecture"
    _write_corrupt_architecture(root, transform)
    monkeypatch.setattr(module, "_ARCHITECTURE_ROOT", root)
    with pytest.raises(AuthenticationError) as caught:
        module._architecture_downstream_definitions()  # noqa: SLF001
    assert caught.value.reason == "CONTRACT_INCONSISTENT"


def test_wrong_artifact_owner_constructor_failure_is_identity_atomic(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import bot_core.security.authentication as module
    from tests.security.test_authentication import Comparator

    security, authentication, _ = prepared(tmp_path / "runtime")
    req = exact_request(definition())
    authentication.issue_authentication_proof(req, RAW_PIN, NOW)
    before = security._state.snapshot  # noqa: SLF001
    root = tmp_path / "architecture"
    _write_corrupt_architecture(root, lambda document: document.__setitem__("m0_element", "M0.99"))
    monkeypatch.setattr(module, "_ARCHITECTURE_ROOT", root)
    with pytest.raises(AuthenticationError) as caught:
        module.AuthenticationAuthority(security, Comparator())
    assert caught.value.reason == "CONTRACT_INCONSISTENT"
    assert security._state.snapshot is before  # noqa: SLF001


def test_canonical_artifact_owner_and_declaration_provenance_agree() -> None:
    import bot_core.security.authentication as module

    definitions = module._architecture_downstream_definitions()  # noqa: SLF001
    assert definitions
    assert all(item.owner_milestone == "M0.12" for item in definitions)
    assert all(module._canonical_declaration(item) for item in definitions)  # noqa: SLF001
