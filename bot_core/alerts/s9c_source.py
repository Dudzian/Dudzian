"""Production S9C AcceptedObservation projection into AlertStore source facts.

The adapter owns no membership.  Its constant generation is an adapter
representation epoch, not an S9C security generation.  The S9C carrier fence is
held while AlertStore performs its separate atomic source publication; this is
not a physical transaction spanning the two carriers.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Callable

from bot_core.alerts.store import (
    Alert,
    AlertStoreError,
    DedupIdentity,
    HistoricalSourceDecision,
    SourceEvidence,
    SourceEvidenceSet,
    SourceSelector,
    ValidatedSourceFact,
    _historical_source_decision_id,
)
from bot_core.observability.authority import (
    AcceptedObservation,
    ObservationAuthority,
    ObservationSemanticKey,
)
from bot_core.persistence.fingerprints import canonical_json_sha256

S9C_RESOLUTION_POLICY_ID = "S9D/S9C_EFFECTIVE_CURRENT_OK_EXACT_SCOPE_V1"
S9C_PRODUCTION_PROJECTION = MappingProxyType({
    "MARKET_DATA_FRESHNESS": MappingProxyType({
        "alert_type": "MARKET_DATA_CURRENT_CONDITION", "threshold": "WARNING",
        "severity": MappingProxyType({"UNKNOWN": "ERROR", "DEGRADED": "WARNING", "BLOCKED": "ERROR"}),
        "scope_fields": ("market_data_route_id", "instrument_id"),
    }),
    "EXECUTION_PATH_HEALTH": MappingProxyType({
        "alert_type": "EXECUTION_ROUTE_CONDITION", "threshold": "ERROR",
        "severity": MappingProxyType({"UNKNOWN": "ERROR", "DEGRADED": "ERROR", "BLOCKED": "CRITICAL"}),
        "scope_fields": ("exchange_account_id", "instrument_id", "execution_route_id"),
    }),
})


def _s9c_stamp(value: datetime) -> str:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
        or value.microsecond
    ):
        raise AlertStoreError("MALFORMED_UNTRUSTED_CONTEXT")
    return value.isoformat().replace("+00:00", "Z")


class S9CObservationSourceAuthority:
    """Read-only, restart-safe adapter over carrier-owned S9C membership."""

    def __init__(self, authority: ObservationAuthority) -> None:
        if not isinstance(authority, ObservationAuthority):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        self._authority = authority

    @staticmethod
    def _one(evidence_set: SourceEvidenceSet) -> str:
        if not isinstance(evidence_set, SourceEvidenceSet) or not evidence_set.evidence_ids:
            raise AlertStoreError("SOURCE_EVIDENCE_REQUIRED")
        if len(evidence_set.evidence_ids) != 1:
            raise AlertStoreError("SOURCE_EVIDENCE_INCOMPLETE")
        acceptance_id = evidence_set.evidence_ids[0]
        if not isinstance(acceptance_id, str) or not acceptance_id:
            raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
        return acceptance_id

    def _accepted(self, acceptance_id: str) -> AcceptedObservation:
        accepted = self._authority.resolve_historical_acceptance(acceptance_id)
        if accepted is None:
            raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
        if accepted.observation.category not in S9C_PRODUCTION_PROJECTION:
            raise AlertStoreError("SOURCE_SELECTOR_UNKNOWN")
        return accepted

    @staticmethod
    def _fact(accepted: AcceptedObservation) -> ValidatedSourceFact:
        observation = accepted.observation
        path = S9C_PRODUCTION_PROJECTION.get(observation.category)
        if path is None:
            raise AlertStoreError("SOURCE_SELECTOR_UNKNOWN")
        alert_type = str(path["alert_type"])
        threshold = str(path["threshold"])
        semantic_key = ObservationSemanticKey.from_observation(observation)
        semantic = {
            "alert_type": alert_type,
            "category": semantic_key.category,
            "environment": semantic_key.environment,
            "source_component": semantic_key.source_component,
            "scope": dict(semantic_key.scope),
        }
        condition_key = str(canonical_json_sha256(semantic))
        source_id = "s9c-source-" + condition_key
        result = "HEALTHY" if accepted.effective_condition == "OK" else "FAILING"
        severity = threshold if result == "HEALTHY" else path["severity"][accepted.effective_condition]
        selector = SourceSelector(
            alert_type, "OBSERVATION_CONDITION", observation.environment,
            condition_key, observation.category, (source_id,), True, threshold,
            S9C_RESOLUTION_POLICY_ID,
        )
        evidence = SourceEvidence(
            accepted.acceptance_id, alert_type, selector.source_family, source_id,
            observation.environment, condition_key, observation.category, condition_key,
            accepted.transaction_revision, 1, result, observation.observed_at_utc,
            severity, accepted.content_fingerprint,
        )
        return ValidatedSourceFact(selector, (evidence,), result, accepted.acceptance_id)

    def validate_current(self, evidence_set: SourceEvidenceSet, now_utc: datetime) -> ValidatedSourceFact:
        result: ValidatedSourceFact | None = None

        def retain(fact: ValidatedSourceFact) -> Alert:  # protocol return is irrelevant here
            nonlocal result
            result = fact
            return None  # type: ignore[return-value]

        self.consume_current(evidence_set, now_utc, retain)
        assert result is not None
        return result

    def consume_current(
        self, evidence_set: SourceEvidenceSet, now_utc: datetime,
        consumer: Callable[[ValidatedSourceFact], Alert],
    ) -> Alert:
        acceptance_id = self._one(evidence_set)
        accepted = self._accepted(acceptance_id)
        now = _s9c_stamp(now_utc)

        def consume(current: AcceptedObservation) -> Alert:
            if current.acceptance_id != acceptance_id:
                raise AlertStoreError("SOURCE_EVIDENCE_STALE")
            if now < current.accepted_at_utc:
                raise AlertStoreError("TIME_ROLLBACK")
            return consumer(self._fact(current))

        try:
            return self._authority.consume_semantic_effective_current(
                ObservationSemanticKey.from_observation(accepted.observation),
                now_utc=now,
                consumer=consume,
            )
        except ValueError as error:
            if str(error) == "OBSERVATION_NOT_EFFECTIVE_CURRENT":
                raise AlertStoreError("SOURCE_EVIDENCE_STALE") from None
            raise

    def reference_for(self, evidence_set: SourceEvidenceSet) -> str:
        acceptance_id = self._one(evidence_set)
        self._accepted(acceptance_id)
        return acceptance_id

    def validates_reference(self, reference: str) -> bool:
        try:
            self._accepted(reference)
            return True
        except AlertStoreError:
            return False

    def historical_fact(self, reference: str) -> ValidatedSourceFact:
        return self._fact(self._accepted(reference))

    def reference_matches(self, reference: str, identity: DedupIdentity) -> bool:
        try:
            fact = self.historical_fact(reference)
        except AlertStoreError:
            return False
        item = fact.evidence[0]
        return (
            identity.alert_type, identity.environment, identity.alert_scope,
            identity.source_family, identity.fact_type, identity.condition_key,
        ) == (
            item.alert_type, item.environment, item.alert_scope,
            item.source_family, item.fact_type, item.condition_key,
        )

    def _historical(self, reference: str, transaction_time_utc: str) -> ValidatedSourceFact:
        try:
            accepted = self._authority.validate_historical_semantic_effective_acceptance(
                reference, at_utc=transaction_time_utc
            )
        except ValueError as error:
            if str(error) in {"UNKNOWN_ACCEPTANCE", "ACCEPTANCE_NOT_HISTORICALLY_EFFECTIVE"}:
                raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED") from None
            if str(error) == "MALFORMED_TIMESTAMP":
                raise AlertStoreError("CONTRACT_INCONSISTENT") from None
            raise
        return self._fact(accepted)

    def authorize_historical_transition(
        self, reference: str, transaction_time_utc: str
    ) -> HistoricalSourceDecision:
        fact = self._historical(reference, transaction_time_utc)
        item = fact.evidence[0]
        decision = HistoricalSourceDecision(
            "", reference, (reference,), transaction_time_utc, fact.result,
            item.source_severity, fact.selector.resolution_policy_id,
            ((item.source_id, item.source_generation, item.source_revision),),
        )
        return replace(decision, decision_id=_historical_source_decision_id(decision))

    def historical_fact_for_decision(
        self, decision: HistoricalSourceDecision
    ) -> ValidatedSourceFact:
        if (
            not isinstance(decision, HistoricalSourceDecision)
            or _historical_source_decision_id(decision) != decision.decision_id
            or decision.evidence_ids != (decision.evidence_reference,)
        ):
            raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
        fact = self._historical(decision.evidence_reference, decision.transaction_time_utc)
        item = fact.evidence[0]
        if (
            decision.result != fact.result
            or decision.severity != item.source_severity
            or decision.resolution_policy_id != fact.selector.resolution_policy_id
            or decision.source_fence != ((item.source_id, 1, item.source_revision),)
        ):
            raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
        return fact

    def validates_historical_decision(self, decision: HistoricalSourceDecision) -> bool:
        try:
            self.historical_fact_for_decision(decision)
            return True
        except AlertStoreError:
            return False
