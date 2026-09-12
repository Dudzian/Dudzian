"""Executable M0.12 alert lifecycle authority.

All accepted changes are persisted through an atomic carrier before publication.
Source facts and delivery outcomes are accepted only through injected authority
boundaries; hashes provide integrity and never membership authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from threading import RLock
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, cast

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.security.authentication import AuthenticationProof, AuthorizationRequest
from bot_core.security.authorization import AuthorizationAuthority

OPERATIONS = MappingProxyType(
    {
        "M0.12/ALERT_ACKNOWLEDGE": ("ACKNOWLEDGE", True),
        "M0.12/ALERT_SET_SUPPRESSION": ("SET_SUPPRESSION", True),
        "M0.12/ALERT_CLEAR_SUPPRESSION": ("CLEAR_SUPPRESSION", False),
        "M0.12/ALERT_MANUAL_FACT_RESOLUTION": ("MANUAL_FACT_RESOLUTION", True),
    }
)
MANUALLY_RESOLVABLE_TYPES = frozenset({"OPERATOR_WORKFLOW_REQUIRED"})
SUPPRESSION_DURATION = timedelta(hours=1)
ESCALATION_FAILURES = 1
SOURCE_FRESHNESS = timedelta(minutes=5)
SEVERITY_RANK = {"INFO": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}
ESCALATION_POLICY = MappingProxyType(
    {
        "INFO": (None, ("IN_APP",)),
        "WARNING": (timedelta(seconds=900), ("IN_APP", "LOCAL_OS_NOTIFICATION")),
        "ERROR": (timedelta(seconds=300), ("IN_APP", "LOCAL_OS_NOTIFICATION", "TRAY_PERSISTENT")),
        "CRITICAL": (timedelta(0), ("IN_APP", "TRAY_PERSISTENT", "OPERATOR_ATTENTION_REQUIRED")),
    }
)
CANONICAL_ROUTE_DESTINATIONS = MappingProxyType(
    {
        "IN_APP": "core_host/in_app",
        "LOCAL_OS_NOTIFICATION": "desktop_shell/os_notification",
        "TRAY_PERSISTENT": "tray_agent/persistent",
        "OPERATOR_ATTENTION_REQUIRED": "core_host/operator_attention",
    }
)
SUPPORTED_ENVIRONMENTS = frozenset({"PAPER", "TESTNET", "LIVE"})
SYNTHETIC_TEST_SOURCE_RESOLUTION_POLICIES = MappingProxyType(
    {
        (
            "TEST_MULTI_SOURCE",
            "OBSERVATION",
            "PAPER",
            "account-alerts",
            "HEALTH",
            ("source-a", "source-b"),
        ): (True, "S9D/TEST_MULTI_CURRENT_HEALTH_V1"),
        (
            "OPERATOR_WORKFLOW_REQUIRED",
            "OBSERVATION",
            "PAPER",
            "account-alerts",
            "HEALTH",
            ("source-a", "source-b"),
        ): (True, "S9D/OPERATOR_WORKFLOW_HEALTH_V1"),
    }
)
# Core/M0.12-owned resolution authority. Selectors must match; they cannot grant it.
CANONICAL_SOURCE_RESOLUTION_POLICIES = MappingProxyType(
    {
        **SYNTHETIC_TEST_SOURCE_RESOLUTION_POLICIES,
        (
            "DOMAIN_EXECUTION_FAILURE",
            "DOMAIN_EVENT",
            "PAPER",
            "account-alerts",
            "EXECUTION",
            ("source-a",),
        ): (False, "M0.7/NO_UNAMBIGUOUS_CORRECTIVE_SUCCESSOR"),
    }
)
# Production catalog parity projection. These remain explicitly open until the
# named frozen upstream authorities are integrated into this executable store.
PRODUCTION_SOURCE_RESOLUTION_POLICIES = MappingProxyType(
    {
        "MARKET_DATA_CURRENT_CONDITION": (
            "S9C effective-current OK exact category/key/source/environment/scope and not expired",
            "OPEN_SOURCE_AUTHORITY",
        ),
        "EXECUTION_ROUTE_CONDITION": (
            "S9C effective-current OK exact category/key/source/environment/scope and not expired",
            "OPEN_SOURCE_AUTHORITY",
        ),
        "KILL_SWITCH_ACTIVE": (
            "current accepted M0.9 INACTIVE exact scope/environment and generation >= alert source",
            "OPEN_SOURCE_AUTHORITY",
        ),
        "RECONCILIATION_DIVERGENCE": (
            "accepted M0.8 MATCH exact complete reconciliation key",
            "OPEN_SOURCE_AUTHORITY",
        ),
        "PERSISTENCE_RECOVERY_REQUIRED": (
            "current accepted M0.11 COMPLETED exact device/store identity",
            "OPEN_SOURCE_AUTHORITY",
        ),
        "RISK_DECISION_DENIED": (
            "accepted M0.9 ALLOW exact command/request/execution scope",
            "OPEN_SOURCE_AUTHORITY",
        ),
        "DOMAIN_EXECUTION_FAILURE": (
            "OPEN: frozen M0.7 defines facts and transition graph but no single corrective successor mapping for ORDER_REJECTED, ORDER_EXTERNAL_OUTCOME_UNKNOWN, or IDEMPOTENCY_CONFLICT; self/unrelated event rejected",
            "OPEN_SOURCE_AUTHORITY",
        ),
        "SECURITY_PRIVILEGED_FAILURE": (
            "manual resolution only for exact closed-policy types; M0.10 downstream mutation validation required",
            "OPEN_SOURCE_AUTHORITY",
        ),
    }
)


class AlertStoreError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class SourceSelector:
    alert_type: str
    source_family: str
    environment: str
    alert_scope: str
    fact_type: str
    required_source_ids: tuple[str, ...]
    healthy_resolution_supported: bool
    failing_severity: str = "ERROR"
    resolution_policy_id: str = ""


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    evidence_id: str
    alert_type: str
    source_family: str
    source_id: str
    environment: str
    alert_scope: str
    fact_type: str
    condition_key: str
    source_revision: int
    source_generation: int
    observed_result: str
    observed_at_utc: str
    source_severity: str
    content_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class SourceEvidenceSet:
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ValidatedSourceFact:
    selector: SourceSelector
    evidence: tuple[SourceEvidence, ...]
    result: str
    evidence_reference: str


@dataclass(frozen=True, slots=True)
class HistoricalSourceDecision:
    decision_id: str
    evidence_reference: str
    evidence_ids: tuple[str, ...]
    transaction_time_utc: str
    result: str
    severity: str
    resolution_policy_id: str
    source_fence: tuple[tuple[str, int, int], ...]


class SourceEvidenceAuthority(Protocol):
    def validate_current(
        self, evidence_set: SourceEvidenceSet, now_utc: datetime
    ) -> ValidatedSourceFact: ...
    def validates_reference(self, reference: str) -> bool: ...
    def consume_current(
        self,
        evidence_set: SourceEvidenceSet,
        now_utc: datetime,
        consumer: Callable[[ValidatedSourceFact], Alert],
    ) -> Alert: ...
    def reference_matches(self, reference: str, identity: DedupIdentity) -> bool: ...
    def historical_fact(self, reference: str) -> ValidatedSourceFact: ...
    def authorize_historical_transition(
        self, reference: str, transaction_time_utc: str
    ) -> HistoricalSourceDecision: ...
    def validates_historical_decision(self, decision: HistoricalSourceDecision) -> bool: ...
    def historical_fact_for_decision(
        self, decision: HistoricalSourceDecision
    ) -> ValidatedSourceFact: ...
    def reference_for(self, evidence_set: SourceEvidenceSet) -> str: ...


def _source_fingerprint(value: SourceEvidence) -> str:
    body = asdict(value)
    body.pop("content_fingerprint_sha256")
    return str(canonical_json_sha256(body))


class InMemorySourceEvidenceAuthority:
    """Test/reference projection of membership owned by an upstream source."""

    def __init__(self, selectors: tuple[SourceSelector, ...]) -> None:
        if not isinstance(selectors, tuple):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        normalized: dict[tuple[str, str, str, str, str], SourceSelector] = {}
        for selector in selectors:
            if not isinstance(selector, SourceSelector):
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            fields = (
                selector.alert_type,
                selector.source_family,
                selector.environment,
                selector.alert_scope,
                selector.fact_type,
            )
            if (
                not all(isinstance(value, str) and value for value in fields)
                or selector.environment not in SUPPORTED_ENVIRONMENTS
                or not isinstance(selector.required_source_ids, tuple)
                or not selector.required_source_ids
                or not all(
                    isinstance(value, str) and value for value in selector.required_source_ids
                )
                or tuple(sorted(set(selector.required_source_ids))) != selector.required_source_ids
                or type(selector.healthy_resolution_supported) is not bool
                or not isinstance(selector.resolution_policy_id, str)
                or not isinstance(selector.failing_severity, str)
                or selector.failing_severity not in SEVERITY_RANK
            ):
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            policy = CANONICAL_SOURCE_RESOLUTION_POLICIES.get(
                (*fields, selector.required_source_ids)
            )
            if (
                policy is None
                or selector.healthy_resolution_supported is not policy[0]
                or selector.resolution_policy_id not in {"", policy[1]}
            ):
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            if fields in normalized:
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            normalized[fields] = replace(selector, resolution_policy_id=policy[1])
        self._selectors = normalized
        self._accepted: dict[str, SourceEvidence] = {}
        self._current: dict[tuple[str, str, str, str, str, str, str], str] = {}
        self._validated_references: set[str] = set()
        self._reference_sets: dict[tuple[str, ...], str] = {}
        self._lock = RLock()

    def _validate_current_locked(
        self, evidence_set: SourceEvidenceSet, now_utc: datetime
    ) -> ValidatedSourceFact:
        _stamp(now_utc)
        if not isinstance(evidence_set, SourceEvidenceSet) or not evidence_set.evidence_ids:
            raise AlertStoreError("SOURCE_EVIDENCE_REQUIRED")
        if len(set(evidence_set.evidence_ids)) != len(evidence_set.evidence_ids):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        try:
            items = tuple(self._accepted[x] for x in evidence_set.evidence_ids)
        except KeyError:
            raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED") from None
        first = items[0]
        selector = self._selectors.get(
            (
                first.alert_type,
                first.source_family,
                first.environment,
                first.alert_scope,
                first.fact_type,
            )
        )
        if selector is None:
            if any(
                candidate.alert_type == first.alert_type
                and candidate.environment == first.environment
                and candidate.alert_scope == first.alert_scope
                for candidate in self._selectors.values()
            ):
                raise AlertStoreError("SOURCE_SELECTOR_MISMATCH")
            raise AlertStoreError("SOURCE_SELECTOR_UNKNOWN")
        if any(
            (x.alert_type, x.source_family, x.environment, x.alert_scope, x.fact_type)
            != (
                selector.alert_type,
                selector.source_family,
                selector.environment,
                selector.alert_scope,
                selector.fact_type,
            )
            for x in items
        ):
            raise AlertStoreError("SOURCE_SELECTOR_MISMATCH")
        by_source = {x.source_id: x for x in items}
        if set(by_source) != set(selector.required_source_ids) or len(by_source) != len(items):
            raise AlertStoreError("SOURCE_EVIDENCE_INCOMPLETE")
        if len({x.condition_key for x in items}) != 1:
            raise AlertStoreError("SOURCE_EVIDENCE_CONDITION_MISMATCH")
        for item in items:
            key = (
                item.alert_type,
                item.source_family,
                item.environment,
                item.alert_scope,
                item.fact_type,
                item.condition_key,
                item.source_id,
            )
            if self._current.get(key) != item.evidence_id:
                raise AlertStoreError("SOURCE_EVIDENCE_STALE")
            if _source_fingerprint(item) != item.content_fingerprint_sha256:
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            observed = _parse(item.observed_at_utc)
            if observed > now_utc or now_utc - observed > SOURCE_FRESHNESS:
                raise AlertStoreError("SOURCE_EVIDENCE_STALE")
        results = {x.observed_result for x in items}
        if len(results) != 1 or results.pop() not in {"FAILING", "HEALTHY"}:
            raise AlertStoreError("SOURCE_EVIDENCE_AMBIGUOUS")
        severities = {x.source_severity for x in items}
        if len(severities) != 1 or not severities <= set(SEVERITY_RANK):
            raise AlertStoreError("SOURCE_EVIDENCE_AMBIGUOUS")
        result = items[0].observed_result
        canonical_items = tuple(sorted(items, key=lambda item: item.source_id))
        canonical_ids = tuple(item.evidence_id for item in canonical_items)
        reference = str(canonical_json_sha256({"evidence_ids": canonical_ids}))
        self._validated_references.add(reference)
        self._reference_sets[canonical_ids] = reference
        return ValidatedSourceFact(selector, canonical_items, result, reference)

    def validate_current(
        self, evidence_set: SourceEvidenceSet, now_utc: datetime
    ) -> ValidatedSourceFact:
        with self._lock:
            return self._validate_current_locked(evidence_set, now_utc)

    def consume_current(
        self,
        evidence_set: SourceEvidenceSet,
        now_utc: datetime,
        consumer: Callable[[ValidatedSourceFact], Alert],
    ) -> Alert:
        """Hold source currentness authority through the consumer's durable commit."""
        with self._lock:
            fact = self._validate_current_locked(evidence_set, now_utc)
            return consumer(fact)

    def validates_reference(self, reference: str) -> bool:
        """Prove membership for a bounded accepted evidence-set reference."""
        with self._lock:
            return reference in self._validated_references

    def reference_matches(self, reference: str, identity: DedupIdentity) -> bool:
        with self._lock:
            for ids, candidate in self._reference_sets.items():
                if candidate != reference:
                    continue
                items = tuple(self._accepted[item] for item in ids)
                condition = items[0].condition_key
                return (
                    all(
                        (
                            x.alert_type,
                            x.environment,
                            x.alert_scope,
                            x.source_family,
                            x.fact_type,
                            x.condition_key,
                        )
                        == (
                            identity.alert_type,
                            identity.environment,
                            identity.alert_scope,
                            identity.source_family,
                            identity.fact_type,
                            condition,
                        )
                        for x in items
                    )
                    and identity.condition_key == condition
                )
            return False

    def historical_fact(self, reference: str) -> ValidatedSourceFact:
        with self._lock:
            ids = next(
                (ids for ids, value in self._reference_sets.items() if value == reference), None
            )
            if ids is None:
                raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
            items = tuple(self._accepted[item] for item in ids)
            selector = self._selectors[
                (
                    items[0].alert_type,
                    items[0].source_family,
                    items[0].environment,
                    items[0].alert_scope,
                    items[0].fact_type,
                )
            ]
            return ValidatedSourceFact(selector, items, items[0].observed_result, reference)

    def authorize_historical_transition(
        self, reference: str, transaction_time_utc: str
    ) -> HistoricalSourceDecision:
        """Issue immutable upstream facts for the carrier's atomic source transaction."""
        transaction_time = _parse(transaction_time_utc)
        with self._lock:
            fact = self.historical_fact(reference)
            for item in fact.evidence:
                observed = _parse(item.observed_at_utc)
                if observed > transaction_time or transaction_time - observed > SOURCE_FRESHNESS:
                    raise AlertStoreError("SOURCE_EVIDENCE_STALE")
                key = (
                    item.alert_type, item.source_family, item.environment, item.alert_scope,
                    item.fact_type, item.condition_key, item.source_id,
                )
                if self._current.get(key) != item.evidence_id:
                    raise AlertStoreError("SOURCE_EVIDENCE_STALE")
            decision = HistoricalSourceDecision(
                "", reference, tuple(item.evidence_id for item in fact.evidence),
                transaction_time_utc, fact.result,
                fact.evidence[0].source_severity, fact.selector.resolution_policy_id,
                tuple((x.source_id, x.source_generation, x.source_revision) for x in fact.evidence),
            )
            return replace(decision, decision_id=_historical_source_decision_id(decision))

    def validates_historical_decision(self, decision: HistoricalSourceDecision) -> bool:
        try:
            self.historical_fact_for_decision(decision)
            return True
        except AlertStoreError:
            return False

    def historical_fact_for_decision(
        self, decision: HistoricalSourceDecision
    ) -> ValidatedSourceFact:
        """Resolve durable evidence identity without any live-validation cache."""
        with self._lock:
            if _historical_source_decision_id(decision) != decision.decision_id:
                raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
            if type(decision.evidence_ids) is not tuple or not decision.evidence_ids:
                raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
            try:
                items = tuple(self._accepted[item_id] for item_id in decision.evidence_ids)
            except KeyError:
                raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED") from None
            canonical = tuple(sorted(items, key=lambda item: item.source_id))
            canonical_ids = tuple(item.evidence_id for item in canonical)
            reference = str(canonical_json_sha256({"evidence_ids": canonical_ids}))
            first = canonical[0]
            selector = self._selectors.get(
                (first.alert_type, first.source_family, first.environment,
                 first.alert_scope, first.fact_type)
            )
            transaction_time = _parse(decision.transaction_time_utc)
            if (
                selector is None
                or canonical_ids != decision.evidence_ids
                or reference != decision.evidence_reference
                or set(item.source_id for item in canonical) != set(selector.required_source_ids)
                or len(canonical) != len(selector.required_source_ids)
                or len({item.condition_key for item in canonical}) != 1
                or any(
                    (item.alert_type, item.source_family, item.environment,
                     item.alert_scope, item.fact_type)
                    != (selector.alert_type, selector.source_family, selector.environment,
                        selector.alert_scope, selector.fact_type)
                    for item in canonical
                )
                or any(_source_fingerprint(item) != item.content_fingerprint_sha256 for item in canonical)
                or any(
                    _parse(item.observed_at_utc) > transaction_time
                    or transaction_time - _parse(item.observed_at_utc) > SOURCE_FRESHNESS
                    for item in canonical
                )
                or len({item.observed_result for item in canonical}) != 1
                or len({item.source_severity for item in canonical}) != 1
                or decision.result != first.observed_result
                or decision.severity != first.source_severity
                or decision.resolution_policy_id != selector.resolution_policy_id
                or decision.source_fence
                != tuple((x.source_id, x.source_generation, x.source_revision) for x in canonical)
            ):
                raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
            return ValidatedSourceFact(selector, canonical, decision.result, reference)

    def reference_for(self, evidence_set: SourceEvidenceSet) -> str:
        """Derive a canonical historical reference without asserting freshness/currentness."""
        with self._lock:
            try:
                items = tuple(self._accepted[item] for item in evidence_set.evidence_ids)
            except KeyError:
                raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED") from None
            canonical_ids = tuple(
                item.evidence_id for item in sorted(items, key=lambda item: item.source_id)
            )
            return str(canonical_json_sha256({"evidence_ids": canonical_ids}))


def _seed_trusted_source_evidence(
    authority: InMemorySourceEvidenceAuthority, evidence: SourceEvidence, *, current: bool = True
) -> None:
    """Harness for evidence already accepted by its upstream owner."""
    if _source_fingerprint(evidence) != evidence.content_fingerprint_sha256:
        raise AlertStoreError("CONTRACT_INCONSISTENT")
    if (
        not isinstance(evidence.source_generation, int)
        or isinstance(evidence.source_generation, bool)
        or evidence.source_generation < 1
        or not isinstance(evidence.source_revision, int)
        or isinstance(evidence.source_revision, bool)
        or evidence.source_revision < 1
    ):
        raise AlertStoreError("CONTRACT_INCONSISTENT")
    key = (
        evidence.alert_type,
        evidence.source_family,
        evidence.environment,
        evidence.alert_scope,
        evidence.fact_type,
        evidence.condition_key,
        evidence.source_id,
    )
    with authority._lock:
        accepted = authority._accepted.get(evidence.evidence_id)
        if accepted is not None and accepted != evidence:
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        old_id = authority._current.get(key)
        if old_id:
            old = authority._accepted[old_id]
            if (evidence.source_generation, evidence.source_revision) <= (
                old.source_generation,
                old.source_revision,
            ):
                raise AlertStoreError("SOURCE_EVIDENCE_STALE")
        authority._accepted[evidence.evidence_id] = evidence
        if current:
            authority._current[key] = evidence.evidence_id


@dataclass(frozen=True, slots=True)
class DedupIdentity:
    alert_type: str
    environment: str
    alert_scope: str
    source_family: str
    fact_type: str
    condition_key: str


@dataclass(frozen=True, slots=True)
class SuppressionState:
    suppressed: bool = False
    scope: str | None = None
    effective_at_utc: str | None = None
    expires_at_utc: str | None = None
    reason_code: str | None = None
    revision: int = 0


@dataclass(frozen=True, slots=True)
class DeliveryState:
    state: str = "PENDING"
    channel: str = "IN_APP"
    destination: str | None = None
    attempt: int = 0
    delivery_revision: int = 0
    failure_count: int = 0
    escalation_level: int = 0
    escalation_revision: int = 0
    escalation_routes: tuple[str, ...] = ()
    last_transition_at_utc: str | None = None


@dataclass(frozen=True, slots=True)
class Alert:
    alert_id: str
    alert_revision: int
    alert_scope: str
    alert_type: str
    severity: str
    lifecycle_state: str
    acknowledged: bool
    acknowledged_at_utc: str | None
    suppression: SuppressionState
    fact_state: str
    resolution_mode: str | None
    dedup_identity: DedupIdentity
    delivery: DeliveryState
    created_at_utc: str
    current_at_utc: str
    raised_at_utc: str
    last_seen_at_utc: str
    occurrence_count: int
    source_fence: tuple[tuple[str, int, int], ...]
    source_evidence_reference: str
    content_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class AuditObligation:
    obligation_id: str
    alert_id: str
    pre_revision: int
    post_revision: int
    operation: str
    declared_intent: str
    account_id: str
    operator_id: str
    device_installation_id: str
    causation_id: str
    correlation_id: str
    authorization_proof_fingerprint: str
    historical_authorization_decision_id: str
    timestamp_utc: str
    result: str = "COMMITTED"


@dataclass(frozen=True, slots=True)
class HistoricalAuthorizationDecision:
    decision_id: str
    proof_fingerprint_sha256: str
    account_id: str
    operator_id: str
    device_installation_id: str
    environment: str
    operation: str
    declared_intent: str
    target_scope: tuple[tuple[str, object], ...]
    mutation: tuple[tuple[str, object], ...]
    causation_id: str
    correlation_id: str
    authorized_at_utc: str


class HistoricalAuthorizationDecisionAuthority(Protocol):
    def bind_transaction_carrier(self, carrier: AlertStoreCarrier) -> None: ...
    def resolve(self, decision_id: str) -> HistoricalAuthorizationDecision: ...


class InMemoryHistoricalAuthorizationDecisionAuthority:
    """Durable logical history only; decisions are never execution credentials."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._carrier: AlertStoreCarrier | None = None
        self.fail_next = False

    def bind_transaction_carrier(self, carrier: AlertStoreCarrier) -> None:
        with self._lock:
            if self._carrier is not None and self._carrier is not carrier:
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            self._carrier = carrier

    @property
    def committed_decisions(self) -> tuple[HistoricalAuthorizationDecision, ...]:
        with self._lock:
            if self._carrier is None:
                return ()
            return tuple(self._carrier.load_historical_decisions().values())

    @property
    def _decisions(self) -> Mapping[str, HistoricalAuthorizationDecision]:
        """Compatibility projection; authority remains owned by combined carrier state."""
        if self._carrier is None:
            return MappingProxyType({})
        return self._carrier.load_historical_decisions()

    def resolve(self, decision_id: str) -> HistoricalAuthorizationDecision:
        with self._lock:
            if self._carrier is None:
                raise AlertStoreError("AUDIT_PROVENANCE_INVALID")
            try:
                return self._carrier.load_historical_decisions()[decision_id]
            except KeyError:
                raise AlertStoreError("AUDIT_PROVENANCE_INVALID") from None


@dataclass(frozen=True, slots=True)
class MutationHistoryEntry:
    mutation_id: str
    alert_id: str
    pre_revision: int
    post_revision: int
    mutation_type: str
    before_fingerprint: str
    after_fingerprint: str
    timestamp_utc: str
    source_evidence_reference: str | None = None
    audit_obligation_id: str | None = None
    delivery_attempt_id: str | None = None
    causation_id: str | None = None
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class OperatorReplayEntry:
    replay_id: str
    account_id: str
    operator_id: str
    device_installation_id: str
    environment: str
    operation: str
    declared_intent: str
    alert_id: str
    expected_alert_revision: int
    mutation: tuple[tuple[str, object], ...]
    causation_id: str
    correlation_id: str
    result_revision: int
    audit_obligation_id: str


@dataclass(frozen=True, slots=True)
class DeliveryAttempt:
    alert_id: str
    expected_alert_revision: int
    expected_delivery_revision: int
    attempt_id: str
    channel: str
    destination: str
    outcome: str
    attempted_at_utc: str
    retry_count: int
    escalation_level: int
    escalation_route_intent_id: str | None = None
    content_fingerprint_sha256: str = ""


class DeliveryAdapter(Protocol):
    def attempt_idempotent(self, intent: DeliveryAttempt) -> DeliveryAttempt: ...
    def historical_result(self, intent: DeliveryAttempt) -> DeliveryAttempt | None: ...
    def validates_result(self, result: DeliveryAttempt) -> bool: ...


@dataclass(frozen=True, slots=True)
class EscalationRouteIntent:
    intent_id: str
    route: str
    destination: str
    alert_id: str
    alert_revision: int
    escalation_revision: int
    attempt_id: str


@dataclass(frozen=True, slots=True)
class AlertStoreSnapshot:
    store_revision: int
    accepted_revisions: tuple[Alert, ...]
    current_designations: Mapping[str, int]
    dedup_index: Mapping[DedupIdentity, str]
    mutation_history: tuple[MutationHistoryEntry, ...]
    audit_outbox: tuple[AuditObligation, ...]
    delivery_attempts: tuple[DeliveryAttempt, ...]
    escalation_route_intents: tuple[EscalationRouteIntent, ...]
    operator_replays: tuple[OperatorReplayEntry, ...]


@dataclass(frozen=True, slots=True)
class AtomicAlertAuthorityState:
    alert_store_snapshot: AlertStoreSnapshot
    committed_historical_authorization_decisions: Mapping[str, HistoricalAuthorizationDecision]
    committed_historical_source_decisions: Mapping[str, HistoricalSourceDecision] = field(
        default_factory=lambda: MappingProxyType({})
    )


class AlertStoreCarrier(Protocol):
    def load_atomic_state(self) -> AtomicAlertAuthorityState: ...
    def load(self) -> AlertStoreSnapshot: ...
    def load_historical_decisions(
        self,
    ) -> Mapping[str, HistoricalAuthorizationDecision]: ...
    def commit(self, expected_store_revision: int, snapshot: AlertStoreSnapshot) -> None: ...


class InMemoryAlertStoreCarrier:
    """Atomic deterministic reference carrier with crash injection."""

    def __init__(self) -> None:
        self._lock = RLock()
        self.fail_next: str | None = None
        self.fail_combined_next: str | None = None
        self._state = AtomicAlertAuthorityState(
            _empty_snapshot(), MappingProxyType({}), MappingProxyType({})
        )

    @property
    def _snapshot(self) -> AlertStoreSnapshot:
        """Compatibility projection used by corruption-focused tests only."""
        return self._state.alert_store_snapshot

    @_snapshot.setter
    def _snapshot(self, snapshot: AlertStoreSnapshot) -> None:
        self._state = replace(self._state, alert_store_snapshot=snapshot)

    def load(self) -> AlertStoreSnapshot:
        with self._lock:
            return self._state.alert_store_snapshot

    def load_atomic_state(self) -> AtomicAlertAuthorityState:
        with self._lock:
            return _pin_atomic_state(self._state)

    def load_historical_decisions(self) -> Mapping[str, HistoricalAuthorizationDecision]:
        with self._lock:
            return self._state.committed_historical_authorization_decisions

    def commit(self, expected_store_revision: int, snapshot: AlertStoreSnapshot) -> None:
        with self._lock:
            if self._state.alert_store_snapshot.store_revision != expected_store_revision:
                raise AlertStoreError("STORE_CAS_CONFLICT")
            if self.fail_next in {"BEFORE", "DURING"}:
                self.fail_next = None
                raise AlertStoreError("CARRIER_COMMIT_FAILED")
            if snapshot.store_revision != expected_store_revision + 1:
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            _validate_historical_decision_membership(
                snapshot, self._state.committed_historical_authorization_decisions
            )
            _validate_historical_source_membership(
                snapshot, self._state.committed_historical_source_decisions
            )
            self._state = replace(self._state, alert_store_snapshot=snapshot)


def _empty_snapshot() -> AlertStoreSnapshot:
    return AlertStoreSnapshot(0, (), MappingProxyType({}), MappingProxyType({}), (), (), (), (), ())


def _stamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise AlertStoreError("MALFORMED_UNTRUSTED_CONTEXT")
    return value.isoformat().replace("+00:00", "Z")


def _fence_advances(
    old: tuple[tuple[str, int, int], ...], new: tuple[tuple[str, int, int], ...]
) -> bool:
    if tuple(x[0] for x in old) != tuple(x[0] for x in new):
        return False
    advanced = False
    for (_, old_generation, old_revision), (_, generation, revision) in zip(old, new):
        if generation < old_generation or (
            generation == old_generation and revision < old_revision
        ):
            return False
        if generation > old_generation or revision > old_revision:
            advanced = True
    return advanced


def _parse(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.utcoffset() != timedelta(0):
            raise ValueError
        return result
    except (AttributeError, ValueError):
        raise AlertStoreError("CONTRACT_INCONSISTENT") from None


def _max_utc_stamp(*values: str) -> str:
    return _stamp(max(_parse(value) for value in values))


def _alert_fingerprint(value: Alert) -> str:
    body = asdict(value)
    body.pop("content_fingerprint_sha256")
    return str(canonical_json_sha256(body))


def _seal_alert(value: Alert) -> Alert:
    return replace(value, content_fingerprint_sha256=_alert_fingerprint(value))


def _obligation_id(value: AuditObligation) -> str:
    body = asdict(value)
    body.pop("obligation_id")
    return str(canonical_json_sha256(body))


def _seal_obligation(value: AuditObligation) -> AuditObligation:
    return replace(value, obligation_id=_obligation_id(value))


def _decision_id(value: HistoricalAuthorizationDecision) -> str:
    body = asdict(value)
    body.pop("decision_id")
    return str(canonical_json_sha256(body))


def _historical_source_decision_id(value: HistoricalSourceDecision) -> str:
    body = asdict(value)
    body.pop("decision_id")
    return str(canonical_json_sha256(body))


def _validate_historical_source_decision_shape(value: object) -> None:
    """Reject carrier-controlled nested source identities without coercion or repair."""
    if not isinstance(value, HistoricalSourceDecision):
        raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
    string_fields = (
        value.decision_id,
        value.evidence_reference,
        value.transaction_time_utc,
        value.result,
        value.severity,
        value.resolution_policy_id,
    )
    if any(not isinstance(item, str) or not item for item in string_fields):
        raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
    if (
        type(value.evidence_ids) is not tuple
        or not value.evidence_ids
        or any(not isinstance(item, str) or not item for item in value.evidence_ids)
        or len(set(value.evidence_ids)) != len(value.evidence_ids)
        or type(value.source_fence) is not tuple
        or not value.source_fence
    ):
        raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
    source_ids: list[str] = []
    for fence in value.source_fence:
        if type(fence) is not tuple or len(fence) != 3:
            raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
        source_id, generation, revision = fence
        if (
            not isinstance(source_id, str)
            or not source_id
            or type(generation) is not int
            or generation < 1
            or type(revision) is not int
            or revision < 1
        ):
            raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
        source_ids.append(source_id)
    if tuple(sorted(set(source_ids))) != tuple(source_ids):
        raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")


def _seal_decision(value: HistoricalAuthorizationDecision) -> HistoricalAuthorizationDecision:
    return replace(value, decision_id=_decision_id(value))


def _validate_historical_decision_membership(
    snapshot: AlertStoreSnapshot,
    decisions: Mapping[str, HistoricalAuthorizationDecision],
) -> None:
    """Validate the exact operator-transaction <-> trusted-decision bijection."""
    references = tuple(
        obligation.historical_authorization_decision_id for obligation in snapshot.audit_outbox
    )
    if len(references) != len(set(references)) or set(references) != set(decisions):
        raise AlertStoreError("AUDIT_PROVENANCE_INVALID")
    for key, decision in decisions.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(decision, HistoricalAuthorizationDecision)
            or key != decision.decision_id
            or _decision_id(decision) != decision.decision_id
        ):
            raise AlertStoreError("AUDIT_PROVENANCE_INVALID")


def _validate_historical_source_membership(
    snapshot: AlertStoreSnapshot, decisions: Mapping[str, HistoricalSourceDecision]
) -> None:
    references = tuple(
        entry.source_evidence_reference
        for entry in snapshot.mutation_history
        if entry.source_evidence_reference is not None
    )
    if len(references) != len(set(references)) or set(references) != set(decisions):
        raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
    for reference, decision in decisions.items():
        _validate_historical_source_decision_shape(decision)
        if (
            not isinstance(reference, str)
            or not reference
            or reference != decision.evidence_reference
            or _historical_source_decision_id(decision) != decision.decision_id
        ):
            raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")


def _pin_atomic_state(state: AtomicAlertAuthorityState) -> AtomicAlertAuthorityState:
    """Defensively own immutable mappings without changing semantic content."""
    if not isinstance(state, AtomicAlertAuthorityState) or not isinstance(
        state.alert_store_snapshot, AlertStoreSnapshot
    ):
        raise AlertStoreError("CONTRACT_INCONSISTENT")
    snapshot_value = state.alert_store_snapshot
    tuple_fields = (
        "accepted_revisions",
        "mutation_history",
        "audit_outbox",
        "delivery_attempts",
        "escalation_route_intents",
        "operator_replays",
    )
    if any(type(getattr(snapshot_value, name)) is not tuple for name in tuple_fields):
        raise AlertStoreError("CONTRACT_INCONSISTENT")
    for alert in snapshot_value.accepted_revisions:
        if type(alert.source_fence) is not tuple or type(alert.delivery.escalation_routes) is not tuple:
            raise AlertStoreError("CONTRACT_INCONSISTENT")
    for replay in snapshot_value.operator_replays:
        if type(replay.mutation) is not tuple or any(type(item) is not tuple for item in replay.mutation):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
    for decision in state.committed_historical_authorization_decisions.values():
        if (
            type(decision.target_scope) is not tuple
            or type(decision.mutation) is not tuple
            or any(type(item) is not tuple for item in (*decision.target_scope, *decision.mutation))
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
    for decision in state.committed_historical_source_decisions.values():
        _validate_historical_source_decision_shape(decision)
        if (
            type(decision.evidence_ids) is not tuple
            or type(decision.source_fence) is not tuple
            or any(type(item) is not tuple for item in decision.source_fence)
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
    try:
        snapshot = replace(
            snapshot_value,
            current_designations=MappingProxyType(
                dict(state.alert_store_snapshot.current_designations)
            ),
            dedup_index=MappingProxyType(dict(state.alert_store_snapshot.dedup_index)),
        )
        decisions = MappingProxyType(dict(state.committed_historical_authorization_decisions))
        source_decisions = MappingProxyType(dict(state.committed_historical_source_decisions))
    except (TypeError, ValueError):
        raise AlertStoreError("CONTRACT_INCONSISTENT") from None
    return AtomicAlertAuthorityState(snapshot, decisions, source_decisions)


def _history_id(value: MutationHistoryEntry) -> str:
    body = asdict(value)
    body.pop("mutation_id")
    return str(canonical_json_sha256(body))


def _seal_history(value: MutationHistoryEntry) -> MutationHistoryEntry:
    return replace(value, mutation_id=_history_id(value))


def _replay_id(value: OperatorReplayEntry) -> str:
    body = asdict(value)
    body.pop("replay_id")
    return str(canonical_json_sha256(body))


def _seal_replay(value: OperatorReplayEntry) -> OperatorReplayEntry:
    return replace(value, replay_id=_replay_id(value))


def _delivery_fingerprint(value: DeliveryAttempt) -> str:
    body = asdict(value)
    body.pop("content_fingerprint_sha256")
    return str(canonical_json_sha256(body))


def _delivery_semantic_context(value: DeliveryAttempt) -> tuple[object, ...]:
    """Stable idempotency context; outcome and first-attempt time are result fields."""
    return (
        value.attempt_id,
        value.alert_id,
        value.expected_alert_revision,
        value.expected_delivery_revision,
        value.channel,
        value.destination,
        value.retry_count,
        value.escalation_level,
        value.escalation_route_intent_id,
    )


def _seal_delivery(value: DeliveryAttempt) -> DeliveryAttempt:
    return replace(value, content_fingerprint_sha256=_delivery_fingerprint(value))


def _route_intent_id(value: EscalationRouteIntent) -> str:
    body = asdict(value)
    body.pop("intent_id")
    return str(canonical_json_sha256(body))


def _seal_route_intent(value: EscalationRouteIntent) -> EscalationRouteIntent:
    return replace(value, intent_id=_route_intent_id(value))


class AlertStore:
    def __init__(
        self,
        authorization: AuthorizationAuthority,
        source_authority: SourceEvidenceAuthority,
        carrier: AlertStoreCarrier,
        *,
        delivery_adapter: DeliveryAdapter | None = None,
        route_adapters: Mapping[str, DeliveryAdapter] | None = None,
        historical_authorizations: HistoricalAuthorizationDecisionAuthority | None = None,
        audit_ready: Callable[[], bool] = lambda: True,
        destinations: Mapping[str, str] | None = None,
    ) -> None:
        self._authorization = authorization
        self._sources = source_authority
        self._carrier = carrier
        self._delivery_adapter = delivery_adapter
        self._route_adapters = dict(route_adapters or {})
        self._historical_authorizations = (
            historical_authorizations or InMemoryHistoricalAuthorizationDecisionAuthority()
        )
        self._historical_authorizations.bind_transaction_carrier(carrier)

        def publish_operator_transaction(
            expected_store_revision: int,
            snapshot: AlertStoreSnapshot,
            decision: HistoricalAuthorizationDecision,
        ) -> None:
            """Trusted publisher captured only by this operator mutation runtime."""
            if _decision_id(decision) != decision.decision_id:
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            if not isinstance(carrier, InMemoryAlertStoreCarrier):
                raise AlertStoreError("HISTORICAL_TRANSACTION_COMMIT_FAILED")
            with carrier._lock:
                current = carrier._state
                if current.alert_store_snapshot.store_revision != expected_store_revision:
                    raise AlertStoreError("STORE_CAS_CONFLICT")
                if getattr(self._historical_authorizations, "fail_next", False):
                    self._historical_authorizations.fail_next = False  # type: ignore[attr-defined]
                    raise AlertStoreError("HISTORICAL_TRANSACTION_COMMIT_FAILED")
                if carrier.fail_next in {"BEFORE", "DURING"}:
                    carrier.fail_next = None
                    raise AlertStoreError("CARRIER_COMMIT_FAILED")
                if carrier.fail_combined_next in {
                    "BEFORE_PUBLICATION",
                    "FORMER_POST_CARRIER_PRE_DECISION",
                }:
                    carrier.fail_combined_next = None
                    raise AlertStoreError("COMBINED_COMMIT_FAILED")
                if snapshot.store_revision != expected_store_revision + 1:
                    raise AlertStoreError("CONTRACT_INCONSISTENT")
                decisions = dict(current.committed_historical_authorization_decisions)
                if decision.decision_id in decisions:
                    raise AlertStoreError("CONTRACT_INCONSISTENT")
                decisions[decision.decision_id] = decision
                _validate_historical_decision_membership(snapshot, decisions)
                carrier._state = AtomicAlertAuthorityState(
                    snapshot,
                    MappingProxyType(decisions),
                    current.committed_historical_source_decisions,
                )

        self.__publish_operator_transaction = publish_operator_transaction

        def publish_source_transaction(
            expected_store_revision: int,
            snapshot: AlertStoreSnapshot,
            decision: HistoricalSourceDecision,
        ) -> None:
            if not isinstance(carrier, InMemoryAlertStoreCarrier):
                raise AlertStoreError("SOURCE_TRANSACTION_COMMIT_FAILED")
            with carrier._lock:
                current = carrier._state
                if current.alert_store_snapshot.store_revision != expected_store_revision:
                    raise AlertStoreError("STORE_CAS_CONFLICT")
                if carrier.fail_next in {"BEFORE", "DURING", "SOURCE_BEFORE_PUBLICATION"}:
                    carrier.fail_next = None
                    raise AlertStoreError("CARRIER_COMMIT_FAILED")
                source_decisions = dict(current.committed_historical_source_decisions)
                if decision.evidence_reference in source_decisions:
                    raise AlertStoreError("CONTRACT_INCONSISTENT")
                source_decisions[decision.evidence_reference] = decision
                _validate_historical_source_membership(snapshot, source_decisions)
                carrier._state = AtomicAlertAuthorityState(
                    snapshot,
                    current.committed_historical_authorization_decisions,
                    MappingProxyType(source_decisions),
                )

        self.__publish_source_transaction = publish_source_transaction
        if delivery_adapter is not None:
            for route in {route for _, routes in ESCALATION_POLICY.values() for route in routes}:
                self._route_adapters.setdefault(route, delivery_adapter)
        self._audit_ready = audit_ready
        self._destinations = dict(destinations or CANONICAL_ROUTE_DESTINATIONS)
        if self._destinations != dict(CANONICAL_ROUTE_DESTINATIONS):
            raise AlertStoreError("DELIVERY_DESTINATION_CONTRACT_INCONSISTENT")
        self._lock = RLock()
        loaded_state = _pin_atomic_state(carrier.load_atomic_state())
        self._snapshot = self.validate_snapshot(loaded_state.alert_store_snapshot)
        loaded_decisions = loaded_state.committed_historical_authorization_decisions
        loaded_source_decisions = loaded_state.committed_historical_source_decisions
        _validate_historical_decision_membership(self._snapshot, loaded_decisions)
        _validate_historical_source_membership(self._snapshot, loaded_source_decisions)
        for entry in self._snapshot.mutation_history:
            if not entry.source_evidence_reference:
                continue
            after = next(
                alert
                for alert in self._snapshot.accepted_revisions
                if alert.alert_id == entry.alert_id and alert.alert_revision == entry.post_revision
            )
            source_decision = loaded_source_decisions[entry.source_evidence_reference]
            if (
                source_decision.transaction_time_utc != entry.timestamp_utc
                or not source_authority.validates_historical_decision(source_decision)
            ):
                raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
            historical = source_authority.historical_fact_for_decision(source_decision)
            fence = tuple(
                (item.source_id, item.source_generation, item.source_revision)
                for item in historical.evidence
            )
            historical_seen = _stamp(
                max(_parse(item.observed_at_utc) for item in historical.evidence)
            )
            before = next(
                (
                    alert
                    for alert in self._snapshot.accepted_revisions
                    if alert.alert_id == entry.alert_id
                    and alert.alert_revision == entry.pre_revision
                ),
                None,
            )
            expected_seen = (
                _max_utc_stamp(before.last_seen_at_utc, historical_seen)
                if before
                else historical_seen
            )
            expected_occurrences = (
                before.occurrence_count + 1
                if before and entry.mutation_type == "SOURCE_FAILING_OBSERVATION"
                else (before.occurrence_count if before else 1)
            )
            policy = CANONICAL_SOURCE_RESOLUTION_POLICIES.get(
                (
                    historical.selector.alert_type,
                    historical.selector.source_family,
                    historical.selector.environment,
                    historical.selector.alert_scope,
                    historical.selector.fact_type,
                    historical.selector.required_source_ids,
                )
            )
            source_severity = historical.evidence[0].source_severity
            historical_identity_matches = all(
                (
                    item.alert_type,
                    item.environment,
                    item.alert_scope,
                    item.source_family,
                    item.fact_type,
                    item.condition_key,
                )
                == (
                    after.dedup_identity.alert_type,
                    after.dedup_identity.environment,
                    after.dedup_identity.alert_scope,
                    after.dedup_identity.source_family,
                    after.dedup_identity.fact_type,
                    after.dedup_identity.condition_key,
                )
                for item in historical.evidence
            )
            if (
                not historical_identity_matches
                or after.source_fence != fence
                or after.last_seen_at_utc != expected_seen
                or after.occurrence_count != expected_occurrences
                or historical.result
                != ("HEALTHY" if entry.mutation_type == "SOURCE_HEALTHY_RESOLUTION" else "FAILING")
                or source_severity not in SEVERITY_RANK
                or SEVERITY_RANK[source_severity]
                < SEVERITY_RANK[historical.selector.failing_severity]
                or policy is None
                or historical.selector.resolution_policy_id != policy[1]
                or (
                    entry.mutation_type == "SOURCE_HEALTHY_RESOLUTION"
                    and (
                        not policy[0]
                        or before is None
                        or before.fact_state != "FAILING"
                        or before.lifecycle_state not in {"RAISED", "ACKNOWLEDGED"}
                    )
                )
                or (
                    entry.mutation_type == "SOURCE_FAILING_OBSERVATION"
                    and (
                        after.severity != source_severity
                        or (
                            before is not None
                            and (
                                before.fact_state != "FAILING"
                                or before.lifecycle_state == "RESOLVED"
                                or SEVERITY_RANK[source_severity]
                                < SEVERITY_RANK[before.severity]
                            )
                        )
                    )
                )
            ):
                raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
        for attempt in self._snapshot.delivery_attempts:
            adapter = self._route_adapters.get(attempt.channel)
            if (
                CANONICAL_ROUTE_DESTINATIONS.get(attempt.channel) != attempt.destination
                or adapter is None
                or not adapter.validates_result(attempt)
            ):
                raise AlertStoreError("DELIVERY_PROVENANCE_INVALID")
        if any(
            intent.route not in self._route_adapters
            for intent in self._snapshot.escalation_route_intents
        ):
            raise AlertStoreError("ESCALATION_ROUTE_UNAVAILABLE")
        for obligation in self._snapshot.audit_outbox:
            decision = loaded_decisions[obligation.historical_authorization_decision_id]
            alert = next(
                item
                for item in self._snapshot.accepted_revisions
                if item.alert_id == obligation.alert_id
                and item.alert_revision == obligation.post_revision
            )
            expected_target = (
                ("alert_id", obligation.alert_id),
                ("expected_alert_revision", obligation.pre_revision),
                ("alert_scope", alert.alert_scope),
            )
            expected_mutation = (
                ("intent", obligation.declared_intent),
                ("value", OPERATIONS[obligation.operation][1]),
            )
            if (
                decision.proof_fingerprint_sha256,
                decision.account_id,
                decision.operator_id,
                decision.device_installation_id,
                decision.operation,
                decision.declared_intent,
                decision.environment,
                decision.causation_id,
                decision.correlation_id,
                decision.target_scope,
                decision.mutation,
                decision.authorized_at_utc,
            ) != (
                obligation.authorization_proof_fingerprint,
                obligation.account_id,
                obligation.operator_id,
                obligation.device_installation_id,
                obligation.operation,
                obligation.declared_intent,
                alert.dedup_identity.environment,
                obligation.causation_id,
                obligation.correlation_id,
                expected_target,
                expected_mutation,
                obligation.timestamp_utc,
            ):
                raise AlertStoreError("AUDIT_PROVENANCE_INVALID")

    def current(self, alert_id: str) -> Alert:
        revision = self._snapshot.current_designations.get(alert_id)
        matches = [
            x
            for x in self._snapshot.accepted_revisions
            if x.alert_id == alert_id and x.alert_revision == revision
        ]
        if len(matches) != 1:
            raise AlertStoreError("ALERT_NOT_FOUND")
        return matches[0]

    def observe(
        self, *, alert_id: str, evidence_set: SourceEvidenceSet, now_utc: datetime
    ) -> Alert:
        """Consume current evidence under source -> store -> carrier lock ordering."""
        stamp = _stamp(now_utc)
        reference = self._sources.reference_for(evidence_set)
        with self._lock:
            replay = next(
                (
                    item
                    for item in reversed(self._snapshot.accepted_revisions)
                    if item.source_evidence_reference == reference
                ),
                None,
            )
            if replay is not None:
                return self.current(replay.alert_id)
        return self._sources.consume_current(
            evidence_set, now_utc, lambda fact: self._observe_fact(alert_id, fact, stamp)
        )

    def _observe_fact(self, alert_id: str, fact: ValidatedSourceFact, stamp: str) -> Alert:
        severity = fact.evidence[0].source_severity
        if severity not in SEVERITY_RANK:
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if SEVERITY_RANK[severity] < SEVERITY_RANK[fact.selector.failing_severity]:
            raise AlertStoreError("SOURCE_SEVERITY_POLICY_MISMATCH")
        identity = DedupIdentity(
            fact.selector.alert_type,
            fact.selector.environment,
            fact.selector.alert_scope,
            fact.selector.source_family,
            fact.selector.fact_type,
            fact.evidence[0].condition_key,
        )
        source_fence = tuple(
            sorted((x.source_id, x.source_generation, x.source_revision) for x in fact.evidence)
        )
        last_seen = _stamp(max(_parse(item.observed_at_utc) for item in fact.evidence))
        with self._lock:
            replay = next(
                (
                    item
                    for item in reversed(self._snapshot.accepted_revisions)
                    if item.source_evidence_reference == fact.evidence_reference
                ),
                None,
            )
            if replay is not None:
                return self.current(replay.alert_id)
            if fact.result == "FAILING":
                return self._observe_failing(
                    alert_id,
                    identity,
                    severity,
                    source_fence,
                    fact.evidence_reference,
                    stamp,
                    last_seen,
                )
            resolution_policy = CANONICAL_SOURCE_RESOLUTION_POLICIES.get(
                (
                    fact.selector.alert_type,
                    fact.selector.source_family,
                    fact.selector.environment,
                    fact.selector.alert_scope,
                    fact.selector.fact_type,
                    fact.selector.required_source_ids,
                )
            )
            if (
                resolution_policy is None
                or fact.selector.resolution_policy_id != resolution_policy[1]
                or not resolution_policy[0]
            ):
                raise AlertStoreError("SOURCE_RESOLUTION_AUTHORITY_OPEN")
            existing_id = self._snapshot.dedup_index.get(identity)
            if not existing_id:
                raise AlertStoreError("ALERT_NOT_FOUND")
            old = self.current(existing_id)
            if not _fence_advances(old.source_fence, source_fence):
                raise AlertStoreError("SOURCE_EVIDENCE_STALE")
            successor = _seal_alert(
                replace(
                    old,
                    alert_revision=old.alert_revision + 1,
                    lifecycle_state="RESOLVED",
                    fact_state="HEALTHY",
                    resolution_mode="AUTOMATIC_FACT",
                    current_at_utc=stamp,
                    source_fence=source_fence,
                    source_evidence_reference=fact.evidence_reference,
                    last_seen_at_utc=_max_utc_stamp(old.last_seen_at_utc, last_seen),
                    content_fingerprint_sha256="",
                )
            )
            return self._commit_edge(
                old,
                successor,
                "SOURCE_HEALTHY_RESOLUTION",
                stamp,
                source_reference=fact.evidence_reference,
            )

    def _observe_failing(
        self,
        alert_id: str,
        identity: DedupIdentity,
        severity: str,
        source_fence: tuple[tuple[str, int, int], ...],
        reference: str,
        stamp: str,
        last_seen: str | None = None,
    ) -> Alert:
        existing_id = self._snapshot.dedup_index.get(identity)
        last_seen = last_seen or stamp
        if existing_id:
            old = self.current(existing_id)
            if not _fence_advances(old.source_fence, source_fence):
                raise AlertStoreError("SOURCE_EVIDENCE_STALE")
            if SEVERITY_RANK[severity] < SEVERITY_RANK[old.severity]:
                raise AlertStoreError("SEVERITY_DOWNGRADE_FORBIDDEN")
            successor = _seal_alert(
                replace(
                    old,
                    alert_revision=old.alert_revision + 1,
                    severity=severity,
                    source_fence=source_fence,
                    source_evidence_reference=reference,
                    last_seen_at_utc=_max_utc_stamp(old.last_seen_at_utc, last_seen),
                    occurrence_count=old.occurrence_count + 1,
                    current_at_utc=stamp,
                    content_fingerprint_sha256="",
                )
            )
            return self._commit_edge(
                old, successor, "SOURCE_FAILING_OBSERVATION", stamp, source_reference=reference
            )
        if any(x.alert_id == alert_id for x in self._snapshot.accepted_revisions):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        alert = _seal_alert(
            Alert(
                alert_id,
                1,
                identity.alert_scope,
                identity.alert_type,
                severity,
                "RAISED",
                False,
                None,
                SuppressionState(),
                "FAILING",
                None,
                identity,
                DeliveryState(last_transition_at_utc=stamp),
                stamp,
                stamp,
                stamp,
                last_seen,
                1,
                source_fence,
                reference,
                "",
            )
        )
        return self._commit_initial(alert, stamp, reference)

    def mutate(
        self,
        *,
        alert_id: str,
        proof: AuthenticationProof | None,
        request: AuthorizationRequest,
        expected_alert_revision: int,
        now_utc: datetime,
    ) -> Alert:
        stamp = _stamp(now_utc)
        with self._authorization._state.lock:  # noqa: SLF001 - shared linearization boundary
            with self._lock:
                old = self.current(alert_id)
                spec = OPERATIONS.get(request.operation)
                if spec is None:
                    raise AlertStoreError("OPERATION_UNSUPPORTED")
                intent, value = spec
                mutation_items = (("intent", intent), ("value", value))
                semantic = (
                    request.account_id,
                    request.operator_id,
                    request.device_installation_id,
                    request.environment,
                    request.operation,
                    intent,
                    alert_id,
                    expected_alert_revision,
                    mutation_items,
                    request.causation_id,
                    request.correlation_id,
                )
                replay = next(
                    (
                        item
                        for item in self._snapshot.operator_replays
                        if (
                            item.account_id,
                            item.operator_id,
                            item.device_installation_id,
                            item.environment,
                            item.operation,
                            item.declared_intent,
                            item.alert_id,
                            item.expected_alert_revision,
                            item.mutation,
                            item.causation_id,
                            item.correlation_id,
                        )
                        == semantic
                    ),
                    None,
                )
                if replay is not None:
                    return next(
                        item
                        for item in self._snapshot.accepted_revisions
                        if item.alert_id == replay.alert_id
                        and item.alert_revision == replay.result_revision
                    )
                if any(
                    (
                        item.account_id,
                        item.operator_id,
                        item.device_installation_id,
                        item.causation_id,
                    )
                    == (
                        request.account_id,
                        request.operator_id,
                        request.device_installation_id,
                        request.causation_id,
                    )
                    for item in self._snapshot.operator_replays
                ):
                    raise AlertStoreError("OPERATOR_REPLAY_CONFLICT")
                if old.alert_revision != expected_alert_revision:
                    raise AlertStoreError("STALE_ALERT_REVISION")
                if request.environment != old.dedup_identity.environment:
                    raise AlertStoreError("AUTHORIZATION_ENVIRONMENT_MISMATCH")
                target = {
                    "alert_id": alert_id,
                    "expected_alert_revision": expected_alert_revision,
                    "alert_scope": old.alert_scope,
                }
                mutation = {"intent": intent, "value": value}
                self._authorization.validate_downstream_authorized_mutation(
                    proof, request, now_utc, target, mutation
                )
                if not self._audit_ready():
                    raise AlertStoreError("AUDIT_UNAVAILABLE")
                accepted_proof = cast(AuthenticationProof, proof)
                decision = _seal_decision(
                    HistoricalAuthorizationDecision(
                        "",
                        accepted_proof.proof_fingerprint_sha256,
                        request.account_id,
                        request.operator_id,
                        request.device_installation_id,
                        request.environment,
                        request.operation,
                        intent,
                        tuple(target.items()),
                        tuple(mutation.items()),
                        request.causation_id,
                        request.correlation_id,
                        stamp,
                    )
                )
                successor = self._operator_successor(old, request.operation, stamp)
                obligation = _seal_obligation(
                    AuditObligation(
                        "",
                        alert_id,
                        old.alert_revision,
                        old.alert_revision + 1,
                        request.operation,
                        intent,
                        request.account_id,
                        request.operator_id,
                        request.device_installation_id,
                        request.causation_id,
                        request.correlation_id,
                        accepted_proof.proof_fingerprint_sha256,
                        decision.decision_id,
                        stamp,
                    )
                )
                replay_entry = _seal_replay(
                    OperatorReplayEntry(
                        "",
                        request.account_id,
                        request.operator_id,
                        request.device_installation_id,
                        request.environment,
                        request.operation,
                        intent,
                        alert_id,
                        expected_alert_revision,
                        mutation_items,
                        request.causation_id,
                        request.correlation_id,
                        old.alert_revision + 1,
                        obligation.obligation_id,
                    )
                )
                return self._commit_edge(
                    old,
                    successor,
                    intent,
                    stamp,
                    obligation=obligation,
                    causation=request.causation_id,
                    correlation=request.correlation_id,
                    operator_replay=replay_entry,
                    historical_decision=decision,
                )

    def _operator_successor(self, old: Alert, operation: str, stamp: str) -> Alert:
        base = {
            "alert_revision": old.alert_revision + 1,
            "current_at_utc": stamp,
            "content_fingerprint_sha256": "",
        }
        if operation.endswith("ALERT_ACKNOWLEDGE"):
            if old.acknowledged:
                raise AlertStoreError("ALREADY_APPLIED")
            return _seal_alert(
                replace(
                    old,
                    acknowledged=True,
                    acknowledged_at_utc=stamp,
                    lifecycle_state=(
                        "RESOLVED" if old.lifecycle_state == "RESOLVED" else "ACKNOWLEDGED"
                    ),
                    **base,
                )
            )
        if operation.endswith("ALERT_SET_SUPPRESSION"):
            if old.suppression.suppressed:
                raise AlertStoreError("ALREADY_APPLIED")
            expiry = _stamp(_parse(stamp) + SUPPRESSION_DURATION)
            suppression = SuppressionState(
                True,
                old.alert_scope,
                stamp,
                expiry,
                "OPERATOR_POLICY",
                old.suppression.revision + 1,
            )
            return _seal_alert(replace(old, suppression=suppression, **base))
        if operation.endswith("ALERT_CLEAR_SUPPRESSION"):
            if not old.suppression.suppressed:
                raise AlertStoreError("ALREADY_APPLIED")
            return _seal_alert(
                replace(
                    old, suppression=SuppressionState(revision=old.suppression.revision + 1), **base
                )
            )
        if (
            old.alert_type not in MANUALLY_RESOLVABLE_TYPES
            or old.fact_state != "FAILING"
            or old.lifecycle_state == "RESOLVED"
        ):
            raise AlertStoreError("MANUAL_RESOLUTION_DENIED")
        return _seal_alert(
            replace(
                old,
                lifecycle_state="RESOLVED",
                fact_state="MANUALLY_RESOLVED",
                resolution_mode="MANUAL_CLOSED_POLICY",
                **base,
            )
        )

    def expire_suppression(
        self, alert_id: str, *, expected_alert_revision: int, now_utc: datetime
    ) -> Alert:
        stamp = _stamp(now_utc)
        with self._lock:
            old = self.current(alert_id)
            suppression = old.suppression
            if old.alert_revision != expected_alert_revision:
                raise AlertStoreError("STALE_ALERT_REVISION")
            if not suppression.suppressed or suppression.expires_at_utc is None:
                raise AlertStoreError("SUPPRESSION_NOT_EXPIRABLE")
            last = _parse(old.current_at_utc)
            now = _parse(stamp)
            if now < last:
                raise AlertStoreError("TIME_ROLLBACK")
            if now < _parse(suppression.expires_at_utc):
                raise AlertStoreError("SUPPRESSION_NOT_EXPIRED")
            successor = _seal_alert(
                replace(
                    old,
                    alert_revision=old.alert_revision + 1,
                    suppression=SuppressionState(revision=suppression.revision + 1),
                    current_at_utc=stamp,
                    content_fingerprint_sha256="",
                )
            )
            return self._commit_edge(old, successor, "UNSUPPRESS_EXPIRED", stamp)

    def request_delivery(
        self,
        alert_id: str,
        *,
        expected_alert_revision: int,
        expected_delivery_revision: int,
        attempt_id: str,
        now_utc: datetime,
        route: str | None = None,
        escalation_route_intent_id: str | None = None,
    ) -> Alert:
        stamp = _stamp(now_utc)
        with self._lock:
            old = self.current(alert_id)
            effect_before = next(
                (
                    item
                    for item in self._snapshot.accepted_revisions
                    if item.alert_id == alert_id
                    and item.alert_revision == expected_alert_revision
                ),
                None,
            )
            if effect_before is None:
                raise AlertStoreError("STALE_DELIVERY_REVISION")
            channel = route or effect_before.delivery.channel
            route_intent = None
            if route is not None:
                route_intent = next(
                    (
                        item
                        for item in self._snapshot.escalation_route_intents
                        if item.intent_id == escalation_route_intent_id
                    ),
                    None,
                )
                if (
                    route_intent is None
                    or route_intent.route != route
                    or route_intent.alert_id != alert_id
                    or route_intent.attempt_id != attempt_id
                ):
                    raise AlertStoreError("DELIVERY_ROUTE_INTENT_REQUIRED")
            destination = self._destinations.get(channel)
            prior = next(
                (
                    item
                    for item in self._snapshot.delivery_attempts
                    if item.attempt_id == attempt_id
                ),
                None,
            )
            if prior is not None:
                proposed = replace(
                    prior,
                    alert_id=alert_id,
                    expected_alert_revision=expected_alert_revision,
                    expected_delivery_revision=expected_delivery_revision,
                    channel=channel,
                    destination=destination or "",
                    escalation_route_intent_id=escalation_route_intent_id,
                )
                if _delivery_semantic_context(proposed) != _delivery_semantic_context(prior):
                    raise AlertStoreError("DELIVERY_ATTEMPT_REPLAY_CONFLICT")
                edge = next(
                    (
                        item
                        for item in self._snapshot.mutation_history
                        if item.delivery_attempt_id == attempt_id
                    ),
                    None,
                )
                if edge is None:
                    raise AlertStoreError("CONTRACT_INCONSISTENT")
                matches = [
                    item
                    for item in self._snapshot.accepted_revisions
                    if item.alert_id == alert_id and item.alert_revision == edge.post_revision
                ]
                if len(matches) != 1:
                    raise AlertStoreError("CONTRACT_INCONSISTENT")
                return matches[0]
            if effect_before.delivery.delivery_revision != expected_delivery_revision:
                raise AlertStoreError("STALE_DELIVERY_REVISION")
            recovering = old.alert_revision != expected_alert_revision
            if _parse(stamp) < _parse(old.current_at_utc):
                raise AlertStoreError("TIME_ROLLBACK")
            if route is None and effect_before.delivery.state == "ESCALATION_PENDING":
                raise AlertStoreError("DELIVERY_ROUTE_INTENT_REQUIRED")
            if route_intent is not None and (
                route_intent.alert_revision != effect_before.alert_revision
                or route_intent.escalation_revision != effect_before.delivery.escalation_revision
                or route_intent.route not in effect_before.delivery.escalation_routes
                or route_intent.destination != CANONICAL_ROUTE_DESTINATIONS.get(route_intent.route)
            ):
                raise AlertStoreError("STALE_ESCALATION_ROUTE_INTENT")
            if route is not None and route not in effect_before.delivery.escalation_routes:
                raise AlertStoreError("DELIVERY_ROUTE_NOT_INTENDED")
            mandatory_despite_suppression = effect_before.severity == "CRITICAL" and channel in {
                "IN_APP",
                "TRAY_PERSISTENT",
            }
            if not recovering and (
                effect_before.suppression.suppressed
                and effect_before.suppression.expires_at_utc is not None
                and _parse(stamp) >= _parse(effect_before.suppression.expires_at_utc)
            ):
                self.expire_suppression(
                    alert_id, expected_alert_revision=expected_alert_revision, now_utc=now_utc
                )
                raise AlertStoreError("SUPPRESSION_EXPIRED_RETRY")
            if not recovering and effect_before.suppression.suppressed and not mandatory_despite_suppression:
                raise AlertStoreError("SUPPRESSED_DELIVERY")
            if not destination:
                raise AlertStoreError("DELIVERY_DESTINATION_UNRESOLVED")
            adapter = self._route_adapters.get(channel)
            if adapter is None:
                raise AlertStoreError("DELIVERY_AUTHORITY_UNAVAILABLE")
            intent = DeliveryAttempt(
                alert_id,
                expected_alert_revision,
                expected_delivery_revision,
                attempt_id,
                channel,
                destination,
                "PENDING",
                stamp,
                effect_before.delivery.attempt,
                effect_before.delivery.escalation_level,
                escalation_route_intent_id,
            )
            if recovering:
                result = adapter.historical_result(intent)
                if result is None:
                    if old.lifecycle_state == "RESOLVED" or old.fact_state != "FAILING":
                        raise AlertStoreError("ALERT_RESOLVED")
                    if route_intent is None:
                        raise AlertStoreError("STALE_DELIVERY_REVISION")
                    if (
                        route_intent.escalation_revision
                        != old.delivery.escalation_revision
                        or route_intent.route not in old.delivery.escalation_routes
                    ):
                        raise AlertStoreError("STALE_ESCALATION_ROUTE_INTENT")
                    current_mandatory = old.severity == "CRITICAL" and channel in {
                        "IN_APP",
                        "TRAY_PERSISTENT",
                    }
                    if (
                        old.suppression.suppressed
                        and old.suppression.expires_at_utc is not None
                        and _parse(stamp) >= _parse(old.suppression.expires_at_utc)
                    ):
                        self.expire_suppression(
                            alert_id,
                            expected_alert_revision=old.alert_revision,
                            now_utc=now_utc,
                        )
                        raise AlertStoreError("SUPPRESSION_EXPIRED_RETRY")
                    if old.suppression.suppressed and not current_mandatory:
                        raise AlertStoreError("SUPPRESSED_DELIVERY")
                    result = adapter.attempt_idempotent(intent)
            else:
                if old.lifecycle_state == "RESOLVED" or old.fact_state != "FAILING":
                    raise AlertStoreError("ALERT_RESOLVED")
                result = adapter.attempt_idempotent(intent)
            if (
                _delivery_semantic_context(result) != _delivery_semantic_context(intent)
                or result.outcome not in {"DELIVERED", "FAILED"}
                or result.content_fingerprint_sha256
                or _parse(result.attempted_at_utc) > _parse(stamp)
                or _parse(result.attempted_at_utc) < _parse(effect_before.current_at_utc)
                or not adapter.validates_result(result)
            ):
                raise AlertStoreError("DELIVERY_RESULT_INVALID")
            accepted_result = _seal_delivery(result)
            delivery = replace(
                old.delivery,
                state=accepted_result.outcome,
                channel=channel,
                destination=destination,
                attempt=old.delivery.attempt + 1,
                delivery_revision=old.delivery.delivery_revision + 1,
                failure_count=old.delivery.failure_count + (accepted_result.outcome == "FAILED"),
                last_transition_at_utc=(
                    _max_utc_stamp(
                        old.delivery.last_transition_at_utc,
                        accepted_result.attempted_at_utc,
                    )
                    if old.delivery.last_transition_at_utc
                    else accepted_result.attempted_at_utc
                ),
            )

            successor = _seal_alert(
                replace(
                    old,
                    alert_revision=old.alert_revision + 1,
                    delivery=delivery,
                    current_at_utc=stamp,
                    content_fingerprint_sha256="",
                )
            )
            return self._commit_edge(
                old,
                successor,
                "DELIVERY_ATTEMPT",
                stamp,
                delivery_id=attempt_id,
                delivery_attempt=accepted_result,
            )

    def execute_escalation_route(self, *, intent_id: str, now_utc: datetime) -> Alert:
        """Consume exactly one persisted escalation route intent."""
        intent = next(
            (
                item
                for item in self._snapshot.escalation_route_intents
                if item.intent_id == intent_id
            ),
            None,
        )
        if intent is None:
            raise AlertStoreError("DELIVERY_ROUTE_INTENT_REQUIRED")
        prior = next(
            (
                item
                for item in self._snapshot.delivery_attempts
                if item.escalation_route_intent_id == intent_id
            ),
            None,
        )
        if prior is not None:
            edge = next(
                item
                for item in self._snapshot.mutation_history
                if item.delivery_attempt_id == prior.attempt_id
            )
            return next(
                item
                for item in self._snapshot.accepted_revisions
                if item.alert_id == prior.alert_id
                and item.alert_revision == edge.post_revision
            )
        effect_before = next(
            item
            for item in self._snapshot.accepted_revisions
            if item.alert_id == intent.alert_id
            and item.alert_revision == intent.alert_revision
        )
        return self.request_delivery(
            intent.alert_id,
            expected_alert_revision=effect_before.alert_revision,
            expected_delivery_revision=effect_before.delivery.delivery_revision,
            attempt_id=intent.attempt_id,
            now_utc=now_utc,
            route=intent.route,
            escalation_route_intent_id=intent.intent_id,
        )

    def escalate(
        self,
        alert_id: str,
        *,
        expected_alert_revision: int,
        expected_delivery_revision: int,
        expected_escalation_revision: int,
        now_utc: datetime,
    ) -> Alert:
        stamp = _stamp(now_utc)
        with self._lock:
            old = self.current(alert_id)
            delivery = old.delivery
            if (
                old.alert_revision != expected_alert_revision
                or delivery.delivery_revision != expected_delivery_revision
                or delivery.escalation_revision != expected_escalation_revision
            ):
                raise AlertStoreError("STALE_ESCALATION")
            now = _parse(stamp)
            last = _parse(delivery.last_transition_at_utc or old.created_at_utc)
            if now < last:
                raise AlertStoreError("TIME_ROLLBACK")
            if (
                old.suppression.suppressed
                and old.suppression.expires_at_utc is not None
                and now >= _parse(old.suppression.expires_at_utc)
            ):
                self.expire_suppression(
                    alert_id, expected_alert_revision=expected_alert_revision, now_utc=now_utc
                )
                raise AlertStoreError("SUPPRESSION_EXPIRED_RETRY")
            threshold, routes = ESCALATION_POLICY.get(old.severity, (None, ()))
            if old.suppression.suppressed:
                if old.severity != "CRITICAL":
                    raise AlertStoreError("SUPPRESSED_DELIVERY")
                routes = tuple(route for route in routes if route in {"IN_APP", "TRAY_PERSISTENT"})
            if (
                old.fact_state != "FAILING"
                or old.lifecycle_state == "RESOLVED"
                or delivery.state != "FAILED"
                or delivery.failure_count < ESCALATION_FAILURES
                or threshold is None
                or now - last < threshold
            ):
                raise AlertStoreError("ESCALATION_POLICY_NOT_SATISFIED")
            if len(set(routes)) != len(routes) or any(
                not self._destinations.get(route) or route not in self._route_adapters
                for route in routes
            ):
                raise AlertStoreError("ESCALATION_ROUTE_UNAVAILABLE")
            updated = replace(
                delivery,
                state="ESCALATION_PENDING",
                escalation_level=delivery.escalation_level + 1,
                escalation_revision=delivery.escalation_revision + 1,
                escalation_routes=routes,
                last_transition_at_utc=stamp,
            )
            successor = _seal_alert(
                replace(
                    old,
                    alert_revision=old.alert_revision + 1,
                    delivery=updated,
                    current_at_utc=stamp,
                    content_fingerprint_sha256="",
                )
            )
            route_intents = tuple(
                _seal_route_intent(
                    EscalationRouteIntent(
                        "",
                        route,
                        self._destinations[route],
                        old.alert_id,
                        successor.alert_revision,
                        updated.escalation_revision,
                        f"esc-{old.alert_id}-{updated.escalation_revision}-{route}",
                    )
                )
                for route in routes
            )
            return self._commit_edge(
                old, successor, "ESCALATION", stamp, route_intents=route_intents
            )

    def snapshot(self) -> AlertStoreSnapshot:
        return self._snapshot

    def _commit_initial(self, alert: Alert, stamp: str, reference: str) -> Alert:
        entry = _seal_history(
            MutationHistoryEntry(
                "",
                alert.alert_id,
                0,
                1,
                "SOURCE_FAILING_OBSERVATION",
                "GENESIS",
                alert.content_fingerprint_sha256,
                stamp,
                source_evidence_reference=reference,
            )
        )
        return self._commit_snapshot(alert, entry, None, None)

    def _commit_edge(
        self,
        old: Alert,
        successor: Alert,
        kind: str,
        stamp: str,
        *,
        source_reference: str | None = None,
        obligation: AuditObligation | None = None,
        delivery_id: str | None = None,
        delivery_attempt: DeliveryAttempt | None = None,
        route_intents: tuple[EscalationRouteIntent, ...] = (),
        operator_replay: OperatorReplayEntry | None = None,
        historical_decision: HistoricalAuthorizationDecision | None = None,
        causation: str | None = None,
        correlation: str | None = None,
    ) -> Alert:
        entry = _seal_history(
            MutationHistoryEntry(
                "",
                old.alert_id,
                old.alert_revision,
                successor.alert_revision,
                kind,
                old.content_fingerprint_sha256,
                successor.content_fingerprint_sha256,
                stamp,
                source_reference,
                obligation.obligation_id if obligation else None,
                delivery_id,
                causation,
                correlation,
            )
        )
        return self._commit_snapshot(
            successor,
            entry,
            obligation,
            delivery_attempt,
            route_intents,
            operator_replay,
            historical_decision,
        )

    def _commit_snapshot(
        self,
        successor: Alert,
        entry: MutationHistoryEntry,
        obligation: AuditObligation | None,
        delivery_attempt: DeliveryAttempt | None,
        route_intents: tuple[EscalationRouteIntent, ...] = (),
        operator_replay: OperatorReplayEntry | None = None,
        historical_decision: HistoricalAuthorizationDecision | None = None,
    ) -> Alert:
        before = self._snapshot
        accepted = tuple(
            x
            for x in before.accepted_revisions
            if not (
                x.alert_id == successor.alert_id and x.alert_revision == successor.alert_revision
            )
        ) + (successor,)
        current = dict(before.current_designations)
        current[successor.alert_id] = successor.alert_revision
        dedup = dict(before.dedup_index)
        if successor.fact_state == "FAILING":
            dedup[successor.dedup_identity] = successor.alert_id
        elif dedup.get(successor.dedup_identity) == successor.alert_id:
            dedup.pop(successor.dedup_identity)
        candidate = AlertStoreSnapshot(
            before.store_revision + 1,
            accepted,
            MappingProxyType(current),
            MappingProxyType(dedup),
            before.mutation_history + (entry,),
            before.audit_outbox + ((obligation,) if obligation else ()),
            before.delivery_attempts + ((delivery_attempt,) if delivery_attempt else ()),
            before.escalation_route_intents + route_intents,
            before.operator_replays + ((operator_replay,) if operator_replay else ()),
        )
        self.validate_snapshot(candidate)
        if historical_decision is not None:
            self.__publish_operator_transaction(
                before.store_revision, candidate, historical_decision
            )
        elif entry.source_evidence_reference is not None:
            source_decision = self._sources.authorize_historical_transition(
                entry.source_evidence_reference, entry.timestamp_utc
            )
            self.__publish_source_transaction(
                before.store_revision, candidate, source_decision
            )
        else:
            self._carrier.commit(before.store_revision, candidate)
        self._snapshot = candidate
        return successor

    @classmethod
    def restore(
        cls,
        authorization: AuthorizationAuthority,
        source_authority: SourceEvidenceAuthority,
        carrier: AlertStoreCarrier,
        **kwargs: object,
    ) -> AlertStore:
        return cls(authorization, source_authority, carrier, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def validate_snapshot(cls, snapshot: AlertStoreSnapshot) -> AlertStoreSnapshot:
        if not isinstance(snapshot, AlertStoreSnapshot):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if snapshot.store_revision != len(snapshot.mutation_history):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        histories: dict[str, list[Alert]] = {}
        for alert in snapshot.accepted_revisions:
            histories.setdefault(alert.alert_id, []).append(alert)
        edge_map: dict[tuple[str, int], MutationHistoryEntry] = {}
        for entry in snapshot.mutation_history:
            key = (entry.alert_id, entry.post_revision)
            if key in edge_map or _history_id(entry) != entry.mutation_id:
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            edge_map[key] = entry
        active: dict[DedupIdentity, str] = {}
        for aid, history in histories.items():
            if [x.alert_revision for x in history] != list(
                range(1, len(history) + 1)
            ) or snapshot.current_designations.get(aid) != len(history):
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            if any(_alert_fingerprint(x) != x.content_fingerprint_sha256 for x in history):
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            for index, after in enumerate(history):
                before = history[index - 1] if index else None
                entry = edge_map.get((aid, after.alert_revision))
                if (
                    entry is None
                    or entry.pre_revision != after.alert_revision - 1
                    or entry.after_fingerprint != after.content_fingerprint_sha256
                    or entry.before_fingerprint
                    != (before.content_fingerprint_sha256 if before else "GENESIS")
                ):
                    raise AlertStoreError("CONTRACT_INCONSISTENT")
                cls._validate_transition(before, after, entry)
            last = history[-1]
            if last.fact_state == "FAILING":
                if last.dedup_identity in active:
                    raise AlertStoreError("CONTRACT_INCONSISTENT")
                active[last.dedup_identity] = aid
        revision_keys = {
            (item.alert_id, item.alert_revision) for item in snapshot.accepted_revisions
        }
        if (
            set(snapshot.current_designations) != set(histories)
            or dict(snapshot.dedup_index) != active
            or len(edge_map) != len(snapshot.mutation_history)
            or set(edge_map) != revision_keys
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        obligations: dict[str, AuditObligation] = {}
        edge_obligations: dict[tuple[str, int], AuditObligation] = {}
        for item in snapshot.audit_outbox:
            if (
                item.obligation_id in obligations
                or _obligation_id(item) != item.obligation_id
                or item.result != "COMMITTED"
            ):
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            if not all(
                isinstance(x, str) and x
                for x in (
                    item.account_id,
                    item.operator_id,
                    item.device_installation_id,
                    item.causation_id,
                    item.correlation_id,
                    item.timestamp_utc,
                )
            ):
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            _parse(item.timestamp_utc)
            key = (item.alert_id, item.post_revision)
            if key in edge_obligations:
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            obligations[item.obligation_id] = item
            edge_obligations[key] = item
        operator_kinds = set(x[0] for x in OPERATIONS.values())
        for key, entry in edge_map.items():
            if entry.mutation_type in operator_kinds:
                item = obligations.get(cast(str, entry.audit_obligation_id))
                if item is None or (
                    item.alert_id,
                    item.pre_revision,
                    item.post_revision,
                    item.declared_intent,
                    item.operation,
                    item.causation_id,
                    item.correlation_id,
                    item.timestamp_utc,
                ) != (
                    entry.alert_id,
                    entry.pre_revision,
                    entry.post_revision,
                    entry.mutation_type,
                    next((o for o, s in OPERATIONS.items() if s[0] == entry.mutation_type), ""),
                    entry.causation_id,
                    entry.correlation_id,
                    entry.timestamp_utc,
                ):
                    raise AlertStoreError("CONTRACT_INCONSISTENT")
            elif entry.audit_obligation_id is not None:
                raise AlertStoreError("CONTRACT_INCONSISTENT")
        if set(edge_obligations) != {
            k for k, e in edge_map.items() if e.mutation_type in operator_kinds
        }:
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        replay_edges: set[tuple[str, int]] = set()
        replay_keys: set[tuple[str, str, str, str]] = set()
        for replay in snapshot.operator_replays:
            edge_key = (replay.alert_id, replay.result_revision)
            edge = edge_map.get(edge_key)
            replay_key = (
                replay.account_id,
                replay.operator_id,
                replay.device_installation_id,
                replay.causation_id,
            )
            obligation = obligations.get(replay.audit_obligation_id)
            if (
                _replay_id(replay) != replay.replay_id
                or edge_key in replay_edges
                or replay_key in replay_keys
                or obligation is None
                or edge is None
                or replay.mutation
                != (
                    ("intent", replay.declared_intent),
                    ("value", OPERATIONS.get(replay.operation, (None, None))[1]),
                )
                or (
                    replay.alert_id,
                    replay.result_revision,
                    replay.expected_alert_revision,
                    replay.operation,
                    replay.declared_intent,
                    replay.account_id,
                    replay.operator_id,
                    replay.device_installation_id,
                    replay.environment,
                    replay.causation_id,
                    replay.correlation_id,
                )
                != (
                    obligation.alert_id,
                    obligation.post_revision,
                    obligation.pre_revision,
                    obligation.operation,
                    obligation.declared_intent,
                    obligation.account_id,
                    obligation.operator_id,
                    obligation.device_installation_id,
                    next(
                        item.dedup_identity.environment
                        for item in snapshot.accepted_revisions
                        if item.alert_id == replay.alert_id
                    ),
                    obligation.causation_id,
                    obligation.correlation_id,
                )
                or (
                    edge.alert_id,
                    edge.pre_revision,
                    edge.post_revision,
                    edge.audit_obligation_id,
                )
                != (
                    replay.alert_id,
                    replay.expected_alert_revision,
                    replay.result_revision,
                    replay.audit_obligation_id,
                )
            ):
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            replay_edges.add(edge_key)
            replay_keys.add(replay_key)
        if replay_edges != set(edge_obligations):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        attempts: dict[str, DeliveryAttempt] = {}
        for attempt in snapshot.delivery_attempts:
            if (
                attempt.attempt_id in attempts
                or _delivery_fingerprint(attempt) != attempt.content_fingerprint_sha256
                or attempt.outcome not in {"DELIVERED", "FAILED"}
            ):
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            attempts[attempt.attempt_id] = attempt
        delivery_edges = [
            entry
            for entry in snapshot.mutation_history
            if entry.mutation_type == "DELIVERY_ATTEMPT"
        ]
        if {entry.delivery_attempt_id for entry in delivery_edges} != set(attempts):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        for entry in delivery_edges:
            attempt = attempts[cast(str, entry.delivery_attempt_id)]
            before = next(
                item
                for item in snapshot.accepted_revisions
                if item.alert_id == entry.alert_id and item.alert_revision == entry.pre_revision
            )
            after = next(
                item
                for item in snapshot.accepted_revisions
                if item.alert_id == entry.alert_id and item.alert_revision == entry.post_revision
            )
            effect_before = next(
                (
                    item
                    for item in snapshot.accepted_revisions
                    if item.alert_id == entry.alert_id
                    and item.alert_revision == attempt.expected_alert_revision
                ),
                None,
            )
            if effect_before is None:
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            expected_state = replace(
                before.delivery,
                state=attempt.outcome,
                channel=attempt.channel,
                destination=attempt.destination,
                attempt=before.delivery.attempt + 1,
                delivery_revision=before.delivery.delivery_revision + 1,
                failure_count=before.delivery.failure_count + (attempt.outcome == "FAILED"),
                last_transition_at_utc=(
                    _max_utc_stamp(
                        before.delivery.last_transition_at_utc,
                        attempt.attempted_at_utc,
                    )
                    if before.delivery.last_transition_at_utc
                    else attempt.attempted_at_utc
                ),
            )
            if (
                attempt.alert_id,
                attempt.expected_alert_revision,
                attempt.expected_delivery_revision,
                attempt.retry_count,
                attempt.escalation_level,
            ) != (
                entry.alert_id,
                effect_before.alert_revision,
                effect_before.delivery.delivery_revision,
                effect_before.delivery.attempt,
                effect_before.delivery.escalation_level,
            ) or (
                _parse(attempt.attempted_at_utc) > _parse(entry.timestamp_utc)
                or _parse(attempt.attempted_at_utc) < _parse(effect_before.current_at_utc)
                or after.delivery != expected_state
            ):
                raise AlertStoreError("CONTRACT_INCONSISTENT")
        route_ids: set[str] = set()
        for intent in snapshot.escalation_route_intents:
            if (
                intent.intent_id != _route_intent_id(intent)
                or intent.intent_id in route_ids
                or CANONICAL_ROUTE_DESTINATIONS.get(intent.route) != intent.destination
                or intent.attempt_id
                != (f"esc-{intent.alert_id}-{intent.escalation_revision}-{intent.route}")
            ):
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            route_ids.add(intent.intent_id)
        escalation_edges = [
            entry for entry in snapshot.mutation_history if entry.mutation_type == "ESCALATION"
        ]
        linked_route_ids: set[str] = set()
        for entry in escalation_edges:
            before = next(
                item
                for item in snapshot.accepted_revisions
                if item.alert_id == entry.alert_id and item.alert_revision == entry.pre_revision
            )
            after = next(
                item
                for item in snapshot.accepted_revisions
                if item.alert_id == entry.alert_id and item.alert_revision == entry.post_revision
            )
            expected_routes = ESCALATION_POLICY[before.severity][1]
            if before.suppression.suppressed and before.severity == "CRITICAL":
                expected_routes = tuple(
                    route for route in expected_routes if route in {"IN_APP", "TRAY_PERSISTENT"}
                )
            linked = [
                intent
                for intent in snapshot.escalation_route_intents
                if intent.alert_id == entry.alert_id
                and intent.alert_revision == entry.post_revision
                and intent.escalation_revision == after.delivery.escalation_revision
            ]
            if (
                tuple(intent.route for intent in linked) != expected_routes
                or after.delivery.escalation_routes != expected_routes
            ):
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            linked_route_ids.update(intent.intent_id for intent in linked)
        if linked_route_ids != route_ids:
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        linked_attempt_intents: set[str] = set()
        for attempt in snapshot.delivery_attempts:
            if attempt.escalation_route_intent_id is None:
                continue
            intent = next(
                (
                    item
                    for item in snapshot.escalation_route_intents
                    if item.intent_id == attempt.escalation_route_intent_id
                ),
                None,
            )
            if (
                intent is None
                or attempt.escalation_route_intent_id in linked_attempt_intents
                or (
                    attempt.alert_id,
                    attempt.channel,
                    attempt.destination,
                    attempt.attempt_id,
                    attempt.escalation_level,
                )
                != (
                    intent.alert_id,
                    intent.route,
                    intent.destination,
                    intent.attempt_id,
                    intent.escalation_revision,
                )
            ):
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            linked_attempt_intents.add(attempt.escalation_route_intent_id)
        return snapshot

    @staticmethod
    def _validate_transition(
        before: Alert | None, after: Alert, entry: MutationHistoryEntry
    ) -> None:
        _parse(after.created_at_utc)
        _parse(after.current_at_utc)
        _parse(entry.timestamp_utc)
        if after.current_at_utc != entry.timestamp_utc:
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if before is None:
            if (
                entry.mutation_type != "SOURCE_FAILING_OBSERVATION"
                or after.alert_revision != 1
                or after.lifecycle_state != "RAISED"
                or after.fact_state != "FAILING"
                or after.resolution_mode is not None
                or after.acknowledged
                or after.acknowledged_at_utc is not None
                or after.suppression != SuppressionState()
                or after.delivery != DeliveryState(last_transition_at_utc=after.created_at_utc)
                or not after.source_fence
                or tuple(item[0] for item in after.source_fence)
                != tuple(sorted({item[0] for item in after.source_fence}))
                or any(
                    not isinstance(generation, int)
                    or isinstance(generation, bool)
                    or not isinstance(revision, int)
                    or isinstance(revision, bool)
                    or generation < 1
                    or revision < 1
                    for _, generation, revision in after.source_fence
                )
                or after.created_at_utc != after.current_at_utc
                or after.raised_at_utc != after.created_at_utc
                or after.created_at_utc != entry.timestamp_utc
                or after.occurrence_count != 1
                or not entry.source_evidence_reference
                or entry.source_evidence_reference != after.source_evidence_reference
            ):
                raise AlertStoreError("CONTRACT_INCONSISTENT")
            return
        immutable = (
            "alert_id",
            "alert_scope",
            "alert_type",
            "dedup_identity",
            "created_at_utc",
            "raised_at_utc",
        )
        if any(getattr(before, x) != getattr(after, x) for x in immutable) or _parse(
            after.current_at_utc
        ) < _parse(before.current_at_utc):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if before.acknowledged and not after.acknowledged:
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        allowed = {
            "ACKNOWLEDGE": {
                "alert_revision",
                "current_at_utc",
                "acknowledged",
                "acknowledged_at_utc",
                "lifecycle_state",
                "content_fingerprint_sha256",
            },
            "SET_SUPPRESSION": {
                "alert_revision",
                "current_at_utc",
                "suppression",
                "content_fingerprint_sha256",
            },
            "CLEAR_SUPPRESSION": {
                "alert_revision",
                "current_at_utc",
                "suppression",
                "content_fingerprint_sha256",
            },
            "UNSUPPRESS_EXPIRED": {
                "alert_revision",
                "current_at_utc",
                "suppression",
                "content_fingerprint_sha256",
            },
            "MANUAL_FACT_RESOLUTION": {
                "alert_revision",
                "current_at_utc",
                "lifecycle_state",
                "fact_state",
                "resolution_mode",
                "content_fingerprint_sha256",
            },
            "SOURCE_FAILING_OBSERVATION": {
                "alert_revision",
                "current_at_utc",
                "severity",
                "source_fence",
                "source_evidence_reference",
                "last_seen_at_utc",
                "occurrence_count",
                "content_fingerprint_sha256",
            },
            "SOURCE_HEALTHY_RESOLUTION": {
                "alert_revision",
                "current_at_utc",
                "lifecycle_state",
                "fact_state",
                "resolution_mode",
                "source_fence",
                "source_evidence_reference",
                "last_seen_at_utc",
                "content_fingerprint_sha256",
            },
            "DELIVERY_ATTEMPT": {
                "alert_revision",
                "current_at_utc",
                "delivery",
                "content_fingerprint_sha256",
            },
            "ESCALATION": {
                "alert_revision",
                "current_at_utc",
                "delivery",
                "content_fingerprint_sha256",
            },
        }
        changed = {name for name in asdict(before) if getattr(before, name) != getattr(after, name)}
        if entry.mutation_type not in allowed or not changed <= allowed[entry.mutation_type]:
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if entry.mutation_type.startswith("SOURCE_") and (
            not entry.source_evidence_reference
            or entry.source_evidence_reference != after.source_evidence_reference
            or not _fence_advances(before.source_fence, after.source_fence)
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if entry.mutation_type == "SOURCE_FAILING_OBSERVATION" and (
            before.fact_state != "FAILING"
            or before.lifecycle_state == "RESOLVED"
            or after.fact_state != "FAILING"
            or SEVERITY_RANK.get(after.severity, -1) < SEVERITY_RANK.get(before.severity, 99)
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if entry.mutation_type == "SOURCE_HEALTHY_RESOLUTION" and (
            before.fact_state != "FAILING"
            or before.lifecycle_state not in {"RAISED", "ACKNOWLEDGED"}
            or after.fact_state != "HEALTHY"
            or after.lifecycle_state != "RESOLVED"
            or after.resolution_mode != "AUTOMATIC_FACT"
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if entry.mutation_type == "MANUAL_FACT_RESOLUTION" and (
            after.alert_type not in MANUALLY_RESOLVABLE_TYPES
            or before.fact_state != "FAILING"
            or after.fact_state != "MANUALLY_RESOLVED"
            or after.lifecycle_state != "RESOLVED"
            or after.resolution_mode != "MANUAL_CLOSED_POLICY"
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if entry.mutation_type == "ACKNOWLEDGE" and (
            before.acknowledged
            or not after.acknowledged
            or after.acknowledged_at_utc != entry.timestamp_utc
            or after.lifecycle_state
            != ("RESOLVED" if before.lifecycle_state == "RESOLVED" else "ACKNOWLEDGED")
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if entry.mutation_type == "SET_SUPPRESSION" and (
            before.suppression.suppressed
            or not after.suppression.suppressed
            or after.suppression.revision != before.suppression.revision + 1
            or after.suppression.scope != after.alert_scope
            or after.suppression.effective_at_utc != entry.timestamp_utc
            or after.suppression.expires_at_utc
            != _stamp(_parse(entry.timestamp_utc) + SUPPRESSION_DURATION)
            or after.suppression.reason_code != "OPERATOR_POLICY"
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if entry.mutation_type in {"CLEAR_SUPPRESSION", "UNSUPPRESS_EXPIRED"} and (
            not before.suppression.suppressed
            or after.suppression.suppressed
            or after.suppression.revision != before.suppression.revision + 1
            or after.suppression != SuppressionState(revision=before.suppression.revision + 1)
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if entry.mutation_type == "UNSUPPRESS_EXPIRED" and (
            before.suppression.expires_at_utc is None
            or _parse(entry.timestamp_utc) < _parse(before.suppression.expires_at_utc)
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        s = after.suppression
        if s.suppressed and (
            not s.effective_at_utc
            or not s.expires_at_utc
            or _parse(s.expires_at_utc) <= _parse(s.effective_at_utc)
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if (
            after.suppression.revision < before.suppression.revision
            or after.delivery.attempt < before.delivery.attempt
            or after.delivery.delivery_revision < before.delivery.delivery_revision
            or after.delivery.escalation_level < before.delivery.escalation_level
            or after.delivery.escalation_revision < before.delivery.escalation_revision
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if entry.mutation_type == "DELIVERY_ATTEMPT" and (
            not entry.delivery_attempt_id
            or after.delivery.attempt != before.delivery.attempt + 1
            or after.delivery.delivery_revision != before.delivery.delivery_revision + 1
            or after.delivery.escalation_level != before.delivery.escalation_level
            or after.delivery.escalation_revision != before.delivery.escalation_revision
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if entry.mutation_type == "ESCALATION" and (
            before.fact_state != "FAILING"
            or before.lifecycle_state == "RESOLVED"
            or
            after.delivery.escalation_level != before.delivery.escalation_level + 1
            or after.delivery.escalation_revision != before.delivery.escalation_revision + 1
            or after.delivery.attempt != before.delivery.attempt
            or after.delivery.delivery_revision != before.delivery.delivery_revision
            or before.delivery.state != "FAILED"
            or before.delivery.failure_count < ESCALATION_FAILURES
            or after.delivery.state != "ESCALATION_PENDING"
            or after.delivery.escalation_routes
            != (
                tuple(
                    route
                    for route in ESCALATION_POLICY[before.severity][1]
                    if route in {"IN_APP", "TRAY_PERSISTENT"}
                )
                if before.suppression.suppressed and before.severity == "CRITICAL"
                else ESCALATION_POLICY[before.severity][1]
            )
            or ESCALATION_POLICY[before.severity][0] is None
            or _parse(entry.timestamp_utc)
            - _parse(before.delivery.last_transition_at_utc or before.created_at_utc)
            < cast(timedelta, ESCALATION_POLICY[before.severity][0])
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if (
            entry.mutation_type == "ESCALATION"
            and before.suppression.suppressed
            and before.suppression.expires_at_utc is not None
            and _parse(entry.timestamp_utc) >= _parse(before.suppression.expires_at_utc)
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if (
            entry.mutation_type == "ESCALATION"
            and before.suppression.suppressed
            and before.severity != "CRITICAL"
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if (
            after.delivery.last_transition_at_utc
            and before.delivery.last_transition_at_utc
            and _parse(after.delivery.last_transition_at_utc)
            < _parse(before.delivery.last_transition_at_utc)
        ):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        if after.lifecycle_state == "RESOLVED" and after.fact_state == "FAILING":
            raise AlertStoreError("CONTRACT_INCONSISTENT")
