"""Executable test-only model for the frozen entitlement registry contract."""

from dataclasses import replace
import inspect

import pytest

from bot_core.entitlement_registry_contract import (
    AdminOutcome,
    AdminResult,
    AuthoritativeEntitlementState,
    AuthoritativelyUnboundQuery,
    BindOutcome,
    BindRequest,
    BindResolutionKind,
    BindResult,
    BoundBinding,
    ContractValidationError,
    EntitlementIdentity,
    EntitlementLifecycle,
    EntitlementProvenance,
    EntitlementProvisioningAdminProvider,
    HistoricalStateResult,
    ProvisionEntitlementRequest,
    RegistryReadOutcome,
    RegistryReadResult,
    RegistrySubject,
    RetainedHistoryResult,
    RevokeEntitlementRequest,
    SupersedeEntitlementRequest,
    UnboundBinding,
    admin_predecessor_for,
    authoritative_identity_key,
    binding_identity_canonical_bytes,
    history_proves_authoritatively_unbound,
    initial_state_for,
    legal_lifecycle_transition,
    predecessor_for,
    resolve_bind_request,
    revoked_state_for,
    supersession_states_for,
    validate_complete_lineage,
    validate_exact_snapshot,
)
from bot_core.root_proof_issuer_substrate import EntitlementRegistryProvider


U = "018f3e70-7b5a-7c21-8b9a-0123456789ab"


def subject(handle: str = "deployment-secret-handle") -> RegistrySubject:
    return RegistrySubject(handle, "TEST", "td_example")


def identity(
    generation: int = 1,
    entitlement_id: str = f"ent_{U}",
    product_scope: str = "CryptoHunter",
) -> EntitlementIdentity:
    return EntitlementIdentity(
        entitlement_id, generation, "TEST", "td_example", product_scope
    )


def provenance(
    principal: str = "prv_authority_1",
    claimant: str = "clm_test_1",
    version: int = 1,
    reference: str = "immutable:provisioning:1",
) -> EntitlementProvenance:
    return EntitlementProvenance(
        principal, claimant, version, "deployment-security-authority", reference, "a" * 64
    )


def binding(
    *,
    attempt: str = f"rpa_{U}",
    proof: str = f"rpf_{U}",
    principal: str = "prv_authority_1",
    claimant: str = "clm_test_1",
    claimant_version: int = 1,
) -> BoundBinding:
    return BoundBinding(
        f"ago_{U}", f"acct_{U}", "b" * 64, 1, attempt,
        "CryptoHunterAccountAuthority", "rpr_test_1", 1,
        principal, claimant, claimant_version, "c" * 64,
        "immutable:req:1", proof, "issuer-signing-key", 1,
    )


class SemanticRegistryModel:
    """Per-subject revision model; independent subjects never advance A's head."""

    def __init__(self) -> None:
        self._current: dict[RegistrySubject, AuthoritativeEntitlementState] = {}
        self._history: dict[RegistrySubject, list[AuthoritativeEntitlementState]] = {}
        self._identity_subject: dict[tuple[str, str, str, str], RegistrySubject] = {}
        self.fail_next_serialization = False

    def _record(self, state: AuthoritativeEntitlementState) -> None:
        validate_exact_snapshot(state)
        lineage = (*self._history.get(state.subject, []), state)
        validate_complete_lineage(state.subject, lineage)
        self._history.setdefault(state.subject, []).append(state)
        self._current[state.subject] = state

    def provision_entitlement(self, request: ProvisionEntitlementRequest) -> AdminResult:
        request = validate_exact_snapshot(request)
        assert isinstance(request, ProvisionEntitlementRequest)
        if request.subject in self._current:
            return AdminResult(AdminOutcome.CONFLICT, None)
        authority_key = authoritative_identity_key(request.identity)
        if authority_key in self._identity_subject:
            return AdminResult(AdminOutcome.CONFLICT, None)
        state = initial_state_for(request)
        self._record(state)
        self._identity_subject[authority_key] = request.subject
        return AdminResult(AdminOutcome.COMMITTED, state)

    def authoritative_state(self, requested: RegistrySubject) -> RegistryReadResult:
        requested = validate_exact_snapshot(requested)
        assert isinstance(requested, RegistrySubject)
        state = self._current.get(requested)
        if state is None:
            return RegistryReadResult(RegistryReadOutcome.NOT_FOUND, None)
        return RegistryReadResult(RegistryReadOutcome.FOUND, state)

    def compare_and_swap_bind(self, request: BindRequest) -> BindResult:
        request = validate_exact_snapshot(request)
        assert isinstance(request, BindRequest)
        if self.fail_next_serialization:
            self.fail_next_serialization = False
            return BindResult(BindOutcome.RETRYABLE_SERIALIZATION_FAILURE, None)
        if request.subject not in self._current:
            return BindResult(BindOutcome.NOT_FOUND, None)
        history = self.retained_history(request.subject)
        resolution = resolve_bind_request(history, request)
        outcomes = {
            BindResolutionKind.EXACT_REPLAY: BindOutcome.EXACT_REPLAY,
            BindResolutionKind.CONFLICT_BOUND_TO_DIFFERENT_TUPLE:
                BindOutcome.CONFLICT_BOUND_TO_DIFFERENT_TUPLE,
            BindResolutionKind.STALE_PREDECESSOR: BindOutcome.STALE_PREDECESSOR,
            BindResolutionKind.NOT_FOUND: BindOutcome.NOT_FOUND,
            BindResolutionKind.INACTIVE_REVOKED: BindOutcome.INACTIVE_REVOKED,
            BindResolutionKind.INACTIVE_SUPERSEDED: BindOutcome.INACTIVE_SUPERSEDED,
        }
        if resolution.kind is not BindResolutionKind.NEW_BIND_ELIGIBLE:
            return BindResult(
                outcomes[resolution.kind], resolution.historical_bound_state
            )
        state = self._current[request.subject]
        committed = replace(
            state,
            binding=request.attempted_binding,
            authoritative_state_revision=state.authoritative_state_revision + 1,
            predecessor_revision=state.authoritative_state_revision,
        )
        self._record(committed)
        return BindResult(BindOutcome.NEW_BIND_COMMITTED, committed)

    def state_at_revision(
        self, requested: RegistrySubject, revision: int
    ) -> HistoricalStateResult:
        current = self._current.get(requested)
        if current is None:
            return HistoricalStateResult(
                requested,
                RegistryReadResult(RegistryReadOutcome.NOT_FOUND, None),
                revision,
                None,
                None,
            )
        current_revision = current.authoritative_state_revision
        state = next(
            (item for item in self._history.get(requested, [])
             if item.authoritative_state_revision == revision),
            None,
        )
        if state is None:
            outcome = (
                RegistryReadOutcome.CORRUPT
                if revision <= current_revision
                else RegistryReadOutcome.NOT_FOUND
            )
            return HistoricalStateResult(
                requested, RegistryReadResult(outcome, None), revision, None, None
            )
        return HistoricalStateResult(
            requested,
            RegistryReadResult(RegistryReadOutcome.FOUND, state),
            revision,
            current_revision,
            1,
        )

    def retained_history(self, requested: RegistrySubject) -> RetainedHistoryResult:
        states = tuple(self._history.get(requested, []))
        if not states:
            return RetainedHistoryResult(
                RegistryReadOutcome.NOT_FOUND, requested, (), None, None
            )
        return RetainedHistoryResult(
            RegistryReadOutcome.FOUND,
            requested,
            states,
            states[-1].authoritative_state_revision,
            states[0].authoritative_state_revision,
        )

    def supersede_entitlement(self, request: SupersedeEntitlementRequest) -> AdminResult:
        request = validate_exact_snapshot(request)
        assert isinstance(request, SupersedeEntitlementRequest)
        old = self._current.get(request.expected.subject)
        if old is None:
            return AdminResult(AdminOutcome.NOT_FOUND, None)
        if admin_predecessor_for(old) != request.expected:
            return AdminResult(AdminOutcome.CONFLICT, None)
        retired, successor = supersession_states_for(old, request)
        self._record(retired)
        self._record(successor)
        return AdminResult(AdminOutcome.COMMITTED, successor)

    def revoke_entitlement(self, request: RevokeEntitlementRequest) -> AdminResult:
        request = validate_exact_snapshot(request)
        assert isinstance(request, RevokeEntitlementRequest)
        current = self._current.get(request.expected.subject)
        if current is None:
            return AdminResult(AdminOutcome.NOT_FOUND, None)
        if admin_predecessor_for(current) != request.expected:
            return AdminResult(AdminOutcome.CONFLICT, None)
        if current.lifecycle is not EntitlementLifecycle.ACTIVE:
            return AdminResult(AdminOutcome.CONFLICT, None)
        revoked = revoked_state_for(current, request)
        self._record(revoked)
        return AdminResult(AdminOutcome.COMMITTED, revoked)


def provisioned(
    requested: RegistrySubject | None = None,
) -> tuple[SemanticRegistryModel, AuthoritativeEntitlementState]:
    model = SemanticRegistryModel()
    result = model.provision_entitlement(
        ProvisionEntitlementRequest(requested or subject(), identity(), provenance())
    )
    assert result.state is not None
    return model, result.state


def test_unbound_first_bind_exact_retry_conflict_and_lost_response() -> None:
    model, initial = provisioned()
    request = BindRequest(subject(), predecessor_for(initial), binding())
    committed = model.compare_and_swap_bind(request)
    assert committed.outcome is BindOutcome.NEW_BIND_COMMITTED
    assert committed.authoritative_state is not None
    assert committed.authoritative_state.identity.entitlement_generation == 1
    assert committed.authoritative_state.authoritative_state_revision == 2
    replay = model.compare_and_swap_bind(request)
    assert replay.outcome is BindOutcome.EXACT_REPLAY
    assert replay.authoritative_state == committed.authoritative_state
    other = replace(binding(), root_proof_id="rpf_018f3e70-7b5a-7c21-8b9a-1123456789ab")
    assert model.compare_and_swap_bind(
        BindRequest(subject(), predecessor_for(initial), other)
    ).outcome is BindOutcome.CONFLICT_BOUND_TO_DIFFERENT_TUPLE


def test_two_subject_interleaving_uses_one_per_subject_revision_meaning() -> None:
    model, initial_a = provisioned()
    committed_a = model.compare_and_swap_bind(
        BindRequest(subject(), predecessor_for(initial_a), binding())
    )
    assert committed_a.authoritative_state is not None
    subject_b = subject("deployment-handle-b")
    provision_b = model.provision_entitlement(
        ProvisionEntitlementRequest(
            subject_b,
            EntitlementIdentity(
                "ent_018f3e70-7b5a-7c21-8b9a-1123456789ab",
                1, "TEST", "td_example", "CryptoHunter",
            ),
            provenance(reference="immutable:provisioning:b"),
        )
    )
    assert provision_b.state is not None
    binding_b = replace(
        binding(),
        logical_operation_id="ago_018f3e70-7b5a-7c21-8b9a-1123456789ab",
        account_id="acct_018f3e70-7b5a-7c21-8b9a-1123456789ab",
        issuance_attempt_id="rpa_018f3e70-7b5a-7c21-8b9a-1123456789ab",
        root_proof_id="rpf_018f3e70-7b5a-7c21-8b9a-1123456789ab",
    )
    model.compare_and_swap_bind(
        BindRequest(subject_b, predecessor_for(provision_b.state), binding_b)
    )
    current_a = model.authoritative_state(subject()).state
    at_a = model.state_at_revision(subject(), 2)
    history_a = model.retained_history(subject())
    assert current_a is not None
    assert current_a.authoritative_state_revision == 2
    assert at_a.current_authoritative_state_revision == 2
    assert history_a.current_authoritative_state_revision == 2
    query = AuthoritativelyUnboundQuery(
        subject(),
        "CryptoHunter",
        initial_a.identity.bootstrap_entitlement_id, 1, f"ago_{U}", f"acct_{U}",
        "b" * 64, f"rpa_{U}",
    )
    assert not history_proves_authoritatively_unbound(
        history_a, query, observed_current_revision=2
    )


def test_missing_is_not_unbound_and_handle_cannot_redirect() -> None:
    model, initial = provisioned()
    assert model.authoritative_state(subject("other")).outcome is RegistryReadOutcome.NOT_FOUND
    assert model.compare_and_swap_bind(
        BindRequest(subject("other"), predecessor_for(initial), binding())
    ).outcome is BindOutcome.NOT_FOUND


def test_authoritative_identity_has_one_subject_lineage_and_one_possible_winner() -> None:
    model, initial = provisioned()
    second_subject = subject("second-handle")
    duplicate = model.provision_entitlement(
        ProvisionEntitlementRequest(second_subject, identity(), provenance())
    )
    assert duplicate.outcome is AdminOutcome.CONFLICT
    assert duplicate.state is None
    assert second_subject not in model._current
    assert second_subject not in model._history
    assert model.compare_and_swap_bind(
        BindRequest(subject(), predecessor_for(initial), binding())
    ).outcome is BindOutcome.NEW_BIND_COMMITTED
    assert model.authoritative_state(second_subject).outcome is RegistryReadOutcome.NOT_FOUND


def test_unbound_evidence_is_self_bound_to_subject_and_trust_domain() -> None:
    model, initial = provisioned()
    model.compare_and_swap_bind(BindRequest(subject(), predecessor_for(initial), binding()))
    query = AuthoritativelyUnboundQuery(
        subject(), "CryptoHunter", initial.identity.bootstrap_entitlement_id, 1,
        f"ago_{U}", f"acct_{U}", "b" * 64, f"rpa_{U}",
    )
    other_subject_model, _ = provisioned(subject("other-handle"))
    assert not history_proves_authoritatively_unbound(
        other_subject_model.retained_history(subject("other-handle")),
        query,
        observed_current_revision=1,
    )
    cross_domain_subject = RegistrySubject("domain-handle", "TEST", "other-domain")
    cross_domain_model = SemanticRegistryModel()
    cross_domain_model.provision_entitlement(
        ProvisionEntitlementRequest(
            cross_domain_subject,
            EntitlementIdentity(f"ent_{U}", 1, "TEST", "other-domain", "CryptoHunter"),
            provenance(),
        )
    )
    assert not history_proves_authoritatively_unbound(
        cross_domain_model.retained_history(cross_domain_subject),
        query,
        observed_current_revision=1,
    )


def test_composite_authority_key_and_cross_product_unbound_attack() -> None:
    model = SemanticRegistryModel()
    subject_a = subject("product-a-handle")
    subject_b = subject("product-b-handle")
    identity_a = identity(product_scope="ProductA")
    identity_b = identity(product_scope="ProductB")
    assert authoritative_identity_key(identity_a) != authoritative_identity_key(identity_b)
    provision_a = model.provision_entitlement(
        ProvisionEntitlementRequest(subject_a, identity_a, provenance(reference="product:a"))
    )
    provision_b = model.provision_entitlement(
        ProvisionEntitlementRequest(subject_b, identity_b, provenance(reference="product:b"))
    )
    assert provision_a.outcome is AdminOutcome.COMMITTED
    assert provision_b.outcome is AdminOutcome.COMMITTED
    assert provision_a.state is not None
    assert model.compare_and_swap_bind(
        BindRequest(subject_a, predecessor_for(provision_a.state), binding())
    ).outcome is BindOutcome.NEW_BIND_COMMITTED
    history_b = model.retained_history(subject_b)
    wrong_product = AuthoritativelyUnboundQuery(
        subject_b, "ProductA", identity_b.bootstrap_entitlement_id, 1,
        f"ago_{U}", f"acct_{U}", "b" * 64, f"rpa_{U}",
    )
    assert not history_proves_authoritatively_unbound(
        history_b, wrong_product, observed_current_revision=1
    )
    correct_product = replace(wrong_product, product_scope="ProductB")
    assert history_proves_authoritatively_unbound(
        history_b, correct_product, observed_current_revision=1
    )
    with pytest.raises(ContractValidationError):
        replace(correct_product, product_scope=" ")


@pytest.mark.parametrize("revision", [1, 2, 999])
def test_missing_subject_history_lookup_has_no_fabricated_revision_metadata(
    revision: int,
) -> None:
    model = SemanticRegistryModel()
    result = model.state_at_revision(subject("missing"), revision)
    assert result.read.outcome is RegistryReadOutcome.NOT_FOUND
    assert result.current_authoritative_state_revision is None
    assert result.retained_from_authoritative_state_revision is None
    retained = model.retained_history(subject("missing"))
    assert retained.outcome is RegistryReadOutcome.NOT_FOUND
    assert retained.current_authoritative_state_revision is None
    assert retained.retained_from_authoritative_state_revision is None


def test_existing_subject_missing_in_range_revision_is_corrupt() -> None:
    model, initial = provisioned()
    model.compare_and_swap_bind(BindRequest(subject(), predecessor_for(initial), binding()))
    del model._history[subject()][0]  # controlled corruption fixture
    result = model.state_at_revision(subject(), 1)
    assert result.read.outcome is RegistryReadOutcome.CORRUPT
    assert result.current_authoritative_state_revision is None
    assert result.retained_from_authoritative_state_revision is None


def test_revoke_active_unbound_is_executable_and_terminal() -> None:
    model, initial = provisioned()
    request = RevokeEntitlementRequest(admin_predecessor_for(initial))
    revoked = model.revoke_entitlement(request)
    assert revoked.outcome is AdminOutcome.COMMITTED
    assert revoked.state is not None
    assert revoked.state.lifecycle is EntitlementLifecycle.REVOKED
    assert type(revoked.state.binding) is UnboundBinding
    assert revoked.state.authoritative_state_revision == 2
    retry = model.revoke_entitlement(
        RevokeEntitlementRequest(admin_predecessor_for(revoked.state))
    )
    assert retry.outcome is AdminOutcome.CONFLICT
    assert model.authoritative_state(subject()).state == revoked.state
    assert model.compare_and_swap_bind(
        BindRequest(subject(), predecessor_for(revoked.state), binding())
    ).outcome is BindOutcome.INACTIVE_REVOKED
    with pytest.raises(ContractValidationError):
        SupersedeEntitlementRequest(
            admin_predecessor_for(revoked.state), identity(2), provenance()
        )


def test_superseded_unbound_cannot_first_bind() -> None:
    model, initial = provisioned()
    successor = model.supersede_entitlement(
        SupersedeEntitlementRequest(
            admin_predecessor_for(initial), identity(2), provenance(reference="generation:2")
        )
    )
    assert successor.state is not None
    old_request = BindRequest(subject(), predecessor_for(initial), binding())
    assert model.compare_and_swap_bind(old_request).outcome is BindOutcome.STALE_PREDECESSOR


@pytest.mark.parametrize(
    "changed",
    [
        {"principal": "prv_authority_2"},
        {"claimant": "clm_test_2"},
        {"claimant_version": 2},
    ],
)
def test_bind_cannot_replace_provisioned_claimant_anchor(changed: dict[str, object]) -> None:
    _, initial = provisioned()
    with pytest.raises(ContractValidationError):
        BindRequest(subject(), predecessor_for(initial), binding(**changed))  # type: ignore[arg-type]


def test_stale_and_serialization_failure_are_distinct() -> None:
    model, initial = provisioned()
    stale = replace(predecessor_for(initial), authoritative_state_revision=99)
    assert model.compare_and_swap_bind(
        BindRequest(subject(), stale, binding())
    ).outcome is BindOutcome.STALE_PREDECESSOR
    model.fail_next_serialization = True
    assert model.compare_and_swap_bind(
        BindRequest(subject(), predecessor_for(initial), binding())
    ).outcome is BindOutcome.RETRYABLE_SERIALIZATION_FAILURE


def test_exact_historical_result_binds_subject_and_requested_revision() -> None:
    model, initial = provisioned()
    with pytest.raises(ContractValidationError):
        HistoricalStateResult(subject(), RegistryReadResult(RegistryReadOutcome.FOUND, initial), 2, 2, 1)
    with pytest.raises(ContractValidationError):
        HistoricalStateResult(subject("other"), RegistryReadResult(RegistryReadOutcome.FOUND, initial), 1, 1, 1)


def test_supersession_is_admin_owned_retains_bound_and_allows_reviewed_anchor_rotation() -> None:
    model, initial = provisioned()
    bound = model.compare_and_swap_bind(BindRequest(subject(), predecessor_for(initial), binding()))
    assert bound.authoritative_state is not None
    request = SupersedeEntitlementRequest(
        admin_predecessor_for(bound.authoritative_state),
        identity(2),
        provenance("prv_authority_2", "clm_test_2", 2, "immutable:provisioning:2"),
    )
    successor = model.supersede_entitlement(request)
    assert successor.state is not None and successor.state.identity.entitlement_generation == 2
    assert type(successor.state.binding) is UnboundBinding
    historical = model.state_at_revision(subject(), 2)
    assert historical.read.state == bound.authoritative_state
    assert not hasattr(EntitlementRegistryProvider, "provision_entitlement")
    assert not hasattr(EntitlementRegistryProvider, "supersede_entitlement")
    assert not hasattr(EntitlementRegistryProvider, "revoke_entitlement")
    assert hasattr(EntitlementProvisioningAdminProvider, "revoke_entitlement")


def test_exact_replay_after_revoke_returns_original_active_bound_revision() -> None:
    model, initial = provisioned()
    bind_request = BindRequest(subject(), predecessor_for(initial), binding())
    committed = model.compare_and_swap_bind(bind_request)
    assert committed.authoritative_state is not None
    revoked = model.revoke_entitlement(
        RevokeEntitlementRequest(admin_predecessor_for(committed.authoritative_state))
    )
    assert revoked.state is not None
    assert revoked.state.authoritative_state_revision == 3
    assert revoked.state.binding == committed.authoritative_state.binding
    replay = model.compare_and_swap_bind(bind_request)
    assert replay.outcome is BindOutcome.EXACT_REPLAY
    assert replay.authoritative_state == committed.authoritative_state
    assert replay.authoritative_state.lifecycle is EntitlementLifecycle.ACTIVE
    assert replay.authoritative_state.authoritative_state_revision == 2
    different = replace(binding(), root_proof_id="rpf_018f3e70-7b5a-7c21-8b9a-1123456789ab")
    assert model.compare_and_swap_bind(
        BindRequest(subject(), predecessor_for(initial), different)
    ).outcome is BindOutcome.CONFLICT_BOUND_TO_DIFFERENT_TUPLE


def test_exact_replay_after_supersession_returns_original_bound_revision() -> None:
    model, initial = provisioned()
    bind_request = BindRequest(subject(), predecessor_for(initial), binding())
    committed = model.compare_and_swap_bind(bind_request)
    assert committed.authoritative_state is not None
    successor = model.supersede_entitlement(
        SupersedeEntitlementRequest(
            admin_predecessor_for(committed.authoritative_state),
            identity(2),
            provenance("prv_authority_2", "clm_test_2", 2, "generation:2"),
        )
    )
    assert successor.state is not None
    assert successor.state.authoritative_state_revision == 4
    replay = model.compare_and_swap_bind(bind_request)
    assert replay.outcome is BindOutcome.EXACT_REPLAY
    assert replay.authoritative_state == committed.authoritative_state
    different = replace(binding(), root_proof_id="rpf_018f3e70-7b5a-7c21-8b9a-1123456789ab")
    assert model.compare_and_swap_bind(
        BindRequest(subject(), predecessor_for(initial), different)
    ).outcome is BindOutcome.CONFLICT_BOUND_TO_DIFFERENT_TUPLE


@pytest.mark.parametrize(
    "changed",
    [
        {"bootstrap_entitlement_id": "ent_018f3e70-7b5a-7c21-8b9a-1123456789ab"},
        {"authoritative_state_revision": 2},
        {"binding_identity_digest_sha256": "f" * 64},
        {"lifecycle": EntitlementLifecycle.REVOKED},
    ],
)
def test_same_winner_with_wrong_historical_predecessor_does_not_replay(
    changed: dict[str, object],
) -> None:
    model, initial = provisioned()
    original = BindRequest(subject(), predecessor_for(initial), binding())
    model.compare_and_swap_bind(original)
    wrong = replace(original.expected, **changed)
    result = model.compare_and_swap_bind(BindRequest(subject(), wrong, binding()))
    assert result.outcome is BindOutcome.STALE_PREDECESSOR


def test_initial_generation_and_cross_product_supersession_are_rejected() -> None:
    with pytest.raises(ContractValidationError):
        ProvisionEntitlementRequest(subject(), identity(2), provenance())
    _, initial = provisioned()
    with pytest.raises(ContractValidationError):
        SupersedeEntitlementRequest(
            admin_predecessor_for(initial), identity(2, product_scope="OtherProduct"), provenance()
        )


def test_history_gap_bound_rollback_and_winner_mutation_are_rejected() -> None:
    _, genesis = provisioned()
    gap = replace(genesis, authoritative_state_revision=3, predecessor_revision=2)
    with pytest.raises(ContractValidationError):
        RetainedHistoryResult(RegistryReadOutcome.FOUND, subject(), (genesis, gap), 3, 1)
    bound = replace(genesis, binding=binding(), authoritative_state_revision=2, predecessor_revision=1)
    rollback = replace(bound, binding=UnboundBinding(), authoritative_state_revision=3, predecessor_revision=2)
    with pytest.raises(ContractValidationError):
        RetainedHistoryResult(RegistryReadOutcome.FOUND, subject(), (genesis, bound, rollback), 3, 1)
    mutation = replace(
        bound,
        binding=replace(binding(), root_proof_id="rpf_018f3e70-7b5a-7c21-8b9a-1123456789ab"),
        authoritative_state_revision=3,
        predecessor_revision=2,
    )
    with pytest.raises(ContractValidationError):
        RetainedHistoryResult(RegistryReadOutcome.FOUND, subject(), (genesis, bound, mutation), 3, 1)
    no_op = replace(genesis, authoritative_state_revision=2, predecessor_revision=1)
    with pytest.raises(ContractValidationError):
        RetainedHistoryResult(RegistryReadOutcome.FOUND, subject(), (genesis, no_op), 2, 1)


def _fabricated(cls: type[object], **values: object) -> object:
    item = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(item, name, value)
    return item


def test_fabricated_exact_nested_objects_fail_closed_without_attribute_error() -> None:
    malformed_identity = _fabricated(
        EntitlementIdentity,
        bootstrap_entitlement_id="bad", entitlement_generation=True,
        environment="TEST", trust_domain="td_example", product_scope="CryptoHunter",
        intended_action="ACCOUNT_GENESIS_BOOTSTRAP",
    )
    malformed_binding = _fabricated(
        BoundBinding,
        **{field.name: getattr(binding(), field.name) for field in __import__("dataclasses").fields(BoundBinding)},
    )
    object.__setattr__(malformed_binding, "claimant_key_version", True)
    malformed_state = _fabricated(
        AuthoritativeEntitlementState,
        subject=subject(), identity=malformed_identity, provenance=provenance(),
        lifecycle=EntitlementLifecycle.ACTIVE, binding=UnboundBinding(),
        authoritative_state_revision=1, predecessor_revision=None,
    )
    malformed_request = _fabricated(
        BindRequest, subject=subject(), expected=object.__new__(type(predecessor_for(provisioned()[1]))),
        attempted_binding=malformed_binding,
    )
    malformed_revoke = _fabricated(
        RevokeEntitlementRequest, expected=object.__new__(type(admin_predecessor_for(provisioned()[1])))
    )
    for value in (
        malformed_identity, malformed_binding, malformed_state, malformed_request,
        malformed_revoke,
    ):
        with pytest.raises(ContractValidationError):
            validate_exact_snapshot(value)
    incomplete = object.__new__(EntitlementIdentity)
    with pytest.raises(ContractValidationError):
        validate_exact_snapshot(incomplete)
    _, genesis = provisioned()
    gap = replace(genesis, authoritative_state_revision=3, predecessor_revision=2)
    fabricated_history = _fabricated(
        RetainedHistoryResult,
        outcome=RegistryReadOutcome.FOUND,
        subject=subject(),
        states=(genesis, gap),
        current_authoritative_state_revision=3,
        retained_from_authoritative_state_revision=1,
    )
    with pytest.raises(ContractValidationError):
        validate_exact_snapshot(fabricated_history)


def test_binding_digest_uses_frozen_canonical_json_and_rejects_fabrication() -> None:
    expected = (
        b'{"account_id":"acct_018f3e70-7b5a-7c21-8b9a-0123456789ab",'
        b'"canonical_genesis_request_fingerprint_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
        b'"claimant_key_id":"clm_test_1","claimant_key_version":1,'
        b'"entitlement_generation":1,"issuance_attempt_id":"rpa_018f3e70-7b5a-7c21-8b9a-0123456789ab",'
        b'"issuer_signing_credential_id":"issuer-signing-key","issuer_signing_key_version":1,'
        b'"kind":"BOUND","logical_operation_id":"ago_018f3e70-7b5a-7c21-8b9a-0123456789ab",'
        b'"provisioning_principal_id":"prv_authority_1","requester_key_id":"rpr_test_1",'
        b'"requester_key_version":1,"requester_principal_id":"CryptoHunterAccountAuthority",'
        b'"root_proof_id":"rpf_018f3e70-7b5a-7c21-8b9a-0123456789ab",'
        b'"signed_request_canonical_bytes_reference":"immutable:req:1",'
        b'"signed_request_payload_digest_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}'
    )
    assert binding_identity_canonical_bytes(binding()) == expected
    with pytest.raises(ContractValidationError):
        binding_identity_canonical_bytes(object.__new__(BoundBinding))  # type: ignore[arg-type]


def test_authoritatively_unbound_requires_valid_current_complete_history() -> None:
    model, initial = provisioned()
    history = model.retained_history(subject())
    query = AuthoritativelyUnboundQuery(
        subject(),
        "CryptoHunter",
        initial.identity.bootstrap_entitlement_id, 1, f"ago_{U}", f"acct_{U}",
        "b" * 64, f"rpa_{U}",
    )
    assert history_proves_authoritatively_unbound(history, query, observed_current_revision=1)
    assert not history_proves_authoritatively_unbound(history, query, observed_current_revision=2)
    missing = RetainedHistoryResult(
        RegistryReadOutcome.NOT_FOUND, subject("missing"), (), None, None
    )
    assert not history_proves_authoritatively_unbound(missing, query, observed_current_revision=1)


def test_port_remains_typed_and_bind_request_has_no_successor_revision() -> None:
    annotations = inspect.get_annotations(EntitlementRegistryProvider.compare_and_swap_bind)
    assert annotations["return"] != bool
    assert inspect.get_annotations(EntitlementRegistryProvider.authoritative_state)["return"] != object
    assert set(inspect.signature(BindRequest).parameters) == {
        "subject", "expected", "attempted_binding"
    }
    with pytest.raises(ContractValidationError):
        BindResult(True, None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "outcome",
    [
        BindOutcome.CONFLICT_BOUND_TO_DIFFERENT_TUPLE,
        BindOutcome.STALE_PREDECESSOR,
        BindOutcome.NOT_FOUND,
        BindOutcome.INACTIVE_REVOKED,
        BindOutcome.INACTIVE_SUPERSEDED,
        BindOutcome.CORRUPT,
        BindOutcome.UNAVAILABLE,
        BindOutcome.RETRYABLE_SERIALIZATION_FAILURE,
    ],
)
def test_bind_result_failure_payload_matrix_rejects_state(outcome: BindOutcome) -> None:
    model, initial = provisioned()
    committed = model.compare_and_swap_bind(
        BindRequest(subject(), predecessor_for(initial), binding())
    )
    assert committed.authoritative_state is not None
    with pytest.raises(ContractValidationError):
        BindResult(outcome, committed.authoritative_state)
    assert BindResult(outcome, None).authoritative_state is None


def test_bind_result_success_payload_matrix_requires_bound_state() -> None:
    model, initial = provisioned()
    committed = model.compare_and_swap_bind(
        BindRequest(subject(), predecessor_for(initial), binding())
    )
    assert committed.authoritative_state is not None
    assert BindResult(
        BindOutcome.NEW_BIND_COMMITTED, committed.authoritative_state
    ).authoritative_state == committed.authoritative_state
    assert BindResult(
        BindOutcome.EXACT_REPLAY, committed.authoritative_state
    ).authoritative_state == committed.authoritative_state
    with pytest.raises(ContractValidationError):
        BindResult(BindOutcome.NEW_BIND_COMMITTED, None)
    with pytest.raises(ContractValidationError):
        BindResult(BindOutcome.EXACT_REPLAY, None)


@pytest.mark.parametrize(
    "outcome",
    [
        AdminOutcome.CONFLICT,
        AdminOutcome.NOT_FOUND,
        AdminOutcome.CORRUPT,
        AdminOutcome.UNAVAILABLE,
        AdminOutcome.RETRYABLE_SERIALIZATION_FAILURE,
    ],
)
def test_admin_result_payload_matrix_is_closed(outcome: AdminOutcome) -> None:
    _, state = provisioned()
    with pytest.raises(ContractValidationError):
        AdminResult(outcome, state)
    assert AdminResult(outcome, None).state is None
    with pytest.raises(ContractValidationError):
        AdminResult(AdminOutcome.COMMITTED, None)
    assert AdminResult(AdminOutcome.COMMITTED, state).state == state


def test_lifecycle_graph_is_closed() -> None:
    assert legal_lifecycle_transition(EntitlementLifecycle.ACTIVE, EntitlementLifecycle.REVOKED)
    assert legal_lifecycle_transition(EntitlementLifecycle.ACTIVE, EntitlementLifecycle.SUPERSEDED)
    assert not legal_lifecycle_transition(EntitlementLifecycle.REVOKED, EntitlementLifecycle.ACTIVE)
