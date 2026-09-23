"""Production projection of durable M0.9 kill-switch authority into M0.12."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Callable

from bot_core.alerts.store import (
    M09_PRODUCTION_RESOLUTION_POLICY_ID,
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
from bot_core.m09_kill_switch_authority import (
    AcceptedKillSwitchAuthorityEntry,
    KillSwitchAuthority,
    KillSwitchAuthorityError,
    KillSwitchRecord,
)
from bot_core.persistence.fingerprints import canonical_json_sha256


def _stamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise AlertStoreError("MALFORMED_UNTRUSTED_CONTEXT")
    return value.isoformat().replace("+00:00", "Z")


def _instant(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except (AttributeError, ValueError):
        raise AlertStoreError("CONTRACT_INCONSISTENT") from None
    if result.utcoffset() != timedelta(0):
        raise AlertStoreError("CONTRACT_INCONSISTENT")
    return result


class M09KillSwitchSourceAuthority:
    """Read-only adapter; only accepted M0.9 carrier membership grants authority."""

    def __init__(self, authority: KillSwitchAuthority) -> None:
        if not isinstance(authority, KillSwitchAuthority):
            raise AlertStoreError("CONTRACT_INCONSISTENT")
        self._authority = authority

    @staticmethod
    def _reference(entry: AcceptedKillSwitchAuthorityEntry, record: KillSwitchRecord) -> str:
        return (
            "m09-reference:" + entry.context.membership_id + ":" + record.record_fingerprint_sha256
        )

    @staticmethod
    def _current_record(
        entry: AcceptedKillSwitchAuthorityEntry, scope: tuple[str, str, str]
    ) -> KillSwitchRecord:
        matches = [
            record
            for record in entry.context.history
            if (record.scope_type, record.scope_id, record.environment) == scope
        ]
        if not matches:
            raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
        return max(matches, key=lambda record: record.generation)

    def evidence_set_for_current(
        self, *, scope_type: str, scope_id: str, environment: str
    ) -> SourceEvidenceSet:
        def project(entry: AcceptedKillSwitchAuthorityEntry) -> SourceEvidenceSet:
            record = self._current_record(entry, (scope_type, scope_id, environment))
            return SourceEvidenceSet((self._reference(entry, record),))

        try:
            return self._authority.consume_current(
                scope_type=scope_type,
                scope_id=scope_id,
                environment=environment,
                consumer=project,
            )
        except KillSwitchAuthorityError:
            raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED") from None

    @staticmethod
    def _one(evidence_set: SourceEvidenceSet) -> str:
        if not isinstance(evidence_set, SourceEvidenceSet) or not evidence_set.evidence_ids:
            raise AlertStoreError("SOURCE_EVIDENCE_REQUIRED")
        if len(evidence_set.evidence_ids) != 1:
            raise AlertStoreError("SOURCE_EVIDENCE_INCOMPLETE")
        reference = evidence_set.evidence_ids[0]
        if not isinstance(reference, str) or not reference:
            raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
        return reference

    def _historical(
        self, reference: str
    ) -> tuple[AcceptedKillSwitchAuthorityEntry, KillSwitchRecord]:
        # Resolve from durable authority history, never from a process-local cache.
        if not isinstance(reference, str):
            raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
        prefix, separator, tail = reference.partition(":")
        membership_id, final_separator, fingerprint = tail.rpartition(":")
        if prefix == "m09-reference" and separator and final_separator:
            resolved = self._authority.resolve_historical_record(
                membership_id=membership_id, record_fingerprint_sha256=fingerprint
            )
            if resolved is not None and self._reference(*resolved) == reference:
                return resolved
        raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")

    def _historical_current(
        self, reference: str, transaction_time_utc: str
    ) -> tuple[AcceptedKillSwitchAuthorityEntry, KillSwitchRecord]:
        historical = self._historical(reference)
        entry, record = historical
        try:
            resolved = self._authority.resolve_historical_current_record(
                membership_id=entry.context.membership_id,
                record_fingerprint_sha256=record.record_fingerprint_sha256,
                at_utc=transaction_time_utc,
            )
        except KillSwitchAuthorityError:
            raise AlertStoreError("CONTRACT_INCONSISTENT") from None
        if resolved != historical:
            raise AlertStoreError("SOURCE_EVIDENCE_STALE")
        return resolved

    @staticmethod
    def _fact(
        entry: AcceptedKillSwitchAuthorityEntry, record: KillSwitchRecord
    ) -> ValidatedSourceFact:
        semantic = {
            "alert_type": "KILL_SWITCH_ACTIVE",
            "environment": record.environment,
            "scope_type": record.scope_type,
            "scope_id": record.scope_id,
            "variant": "M09_KILL_SWITCH",
        }
        condition_key = str(canonical_json_sha256(semantic))
        source_id = "m09-source-" + condition_key
        reference = M09KillSwitchSourceAuthority._reference(entry, record)
        result = "FAILING" if record.state == "ACTIVE" else "HEALTHY"
        selector = SourceSelector(
            "KILL_SWITCH_ACTIVE",
            "UPSTREAM_STATE_CONDITION",
            record.environment,
            condition_key,
            "M09_KILL_SWITCH",
            (source_id,),
            True,
            "CRITICAL",
            M09_PRODUCTION_RESOLUTION_POLICY_ID,
        )
        evidence = SourceEvidence(
            reference,
            selector.alert_type,
            selector.source_family,
            source_id,
            record.environment,
            condition_key,
            selector.fact_type,
            condition_key,
            record.source_revision,
            record.generation,
            result,
            record.effective_at_utc,
            "CRITICAL",
            record.record_fingerprint_sha256,
        )
        return ValidatedSourceFact(selector, (evidence,), result, reference)

    def validate_current(
        self, evidence_set: SourceEvidenceSet, now_utc: datetime
    ) -> ValidatedSourceFact:
        result: ValidatedSourceFact | None = None

        def retain(fact: ValidatedSourceFact) -> Alert:
            nonlocal result
            result = fact
            return None  # type: ignore[return-value]

        self.consume_current(evidence_set, now_utc, retain)
        assert result is not None
        return result

    def consume_current(
        self,
        evidence_set: SourceEvidenceSet,
        now_utc: datetime,
        consumer: Callable[[ValidatedSourceFact], Alert],
    ) -> Alert:
        reference = self._one(evidence_set)
        _entry, historical = self._historical(reference)
        now = _stamp(now_utc)

        def consume(current_entry: AcceptedKillSwitchAuthorityEntry) -> Alert:
            current = self._current_record(
                current_entry,
                (
                    historical.scope_type,
                    historical.scope_id,
                    historical.environment,
                ),
            )
            if self._reference(current_entry, current) != reference:
                raise AlertStoreError("SOURCE_EVIDENCE_STALE")
            if _instant(now) < _instant(current.effective_at_utc):
                raise AlertStoreError("TIME_ROLLBACK")
            return consumer(self._fact(current_entry, current))

        try:
            return self._authority.consume_current(
                scope_type=historical.scope_type,
                scope_id=historical.scope_id,
                environment=historical.environment,
                consumer=consume,
            )
        except KillSwitchAuthorityError:
            raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED") from None

    def reference_for(self, evidence_set: SourceEvidenceSet) -> str:
        reference = self._one(evidence_set)
        self._historical(reference)
        return reference

    def validates_reference(self, reference: str) -> bool:
        try:
            self._historical(reference)
            return True
        except AlertStoreError:
            return False

    def historical_fact(self, reference: str) -> ValidatedSourceFact:
        return self._fact(*self._historical(reference))

    def reference_matches(self, reference: str, identity: DedupIdentity) -> bool:
        try:
            item = self.historical_fact(reference).evidence[0]
        except AlertStoreError:
            return False
        return (
            identity.alert_type,
            identity.environment,
            identity.alert_scope,
            identity.source_family,
            identity.fact_type,
            identity.condition_key,
        ) == (
            item.alert_type,
            item.environment,
            item.alert_scope,
            item.source_family,
            item.fact_type,
            item.condition_key,
        )

    def authorize_historical_transition(
        self, reference: str, transaction_time_utc: str
    ) -> HistoricalSourceDecision:
        fact = self._fact(*self._historical_current(reference, transaction_time_utc))
        item = fact.evidence[0]
        decision = HistoricalSourceDecision(
            "",
            reference,
            (reference,),
            transaction_time_utc,
            fact.result,
            item.source_severity,
            fact.selector.resolution_policy_id,
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
        fact = self._fact(
            *self._historical_current(decision.evidence_reference, decision.transaction_time_utc)
        )
        item = fact.evidence[0]
        if (
            decision.result != fact.result
            or decision.severity != item.source_severity
            or decision.resolution_policy_id != fact.selector.resolution_policy_id
            or decision.source_fence
            != ((item.source_id, item.source_generation, item.source_revision),)
        ):
            raise AlertStoreError("SOURCE_EVIDENCE_UNACCEPTED")
        return fact

    def validates_historical_decision(self, decision: HistoricalSourceDecision) -> bool:
        try:
            self.historical_fact_for_decision(decision)
            return True
        except AlertStoreError:
            return False
