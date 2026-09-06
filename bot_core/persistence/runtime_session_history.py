"""Protected publication of immutable RuntimeSession identity history."""

from __future__ import annotations

from dataclasses import dataclass, replace

from bot_core.runtime.runtime_session import RuntimeSession

from .fingerprints import canonical_json_sha256
from .lifecycle_records import persistence_record
from .protected_freshness_handoff import ProtectedFreshnessHandoffCoordinator
from .records import PersistenceRecord
from .state_store import SQLiteStateStore, StateStoreMetadata, StateStoreSnapshot

RUNTIME_SESSION_REPRESENTATION = "RuntimeSession canonical identity/history"


class RuntimeSessionHistoryError(RuntimeError):
    """RuntimeSession history could not be published and verified exactly."""


@dataclass(frozen=True, slots=True)
class RuntimeSessionPublicationResult:
    runtime_session_id: str
    source_generation: int
    final_metadata: StateStoreMetadata


def runtime_session_carrier(session: RuntimeSession) -> PersistenceRecord:
    upstream = {
        "runtime_session_id": session.runtime_session_id,
        "device_installation_id": session.device_installation_id,
    }
    return persistence_record(
        RUNTIME_SESSION_REPRESENTATION,
        session.runtime_session_id,
        {
            "fact_kind": "RuntimeSession",
            "upstream_payload": upstream,
            "upstream_payload_fingerprint_sha256": canonical_json_sha256(upstream),
        },
    )


class RuntimeSessionHistoryPublisher:
    """Own only the canonical carrier and its ordinary M0.11 mutation."""

    def __init__(
        self, store: SQLiteStateStore, protected: ProtectedFreshnessHandoffCoordinator
    ) -> None:
        self._store = store
        self._protected = protected

    def publish_current_session(self, session: RuntimeSession) -> RuntimeSessionPublicationResult:
        carrier = runtime_session_carrier(session)
        observed: StateStoreSnapshot | None = None

        def build(source: StateStoreSnapshot):
            nonlocal observed
            observed = source
            metadata = source.metadata
            if metadata.device_installation_id != session.device_installation_id:
                raise RuntimeSessionHistoryError("RuntimeSession device binding mismatch")
            if any(
                item.record_key == session.runtime_session_id for item in source.immutable_history
            ):
                raise RuntimeSessionHistoryError("RuntimeSession ID collision")
            return (
                replace(
                    metadata,
                    protected_freshness_generation=metadata.protected_freshness_generation + 1,
                ),
                (),
                (carrier,),
            )

        final_metadata = self._protected.advance_protected_mutation(build)
        if observed is None:
            raise RuntimeSessionHistoryError("protected mutation did not observe a source")
        final = self._store.read_verified_snapshot()
        if final is None or final.metadata != final_metadata:
            raise RuntimeSessionHistoryError("post-publication metadata mismatch")
        source = observed
        if final_metadata.protected_freshness_generation != (
            source.metadata.protected_freshness_generation + 1
        ):
            raise RuntimeSessionHistoryError("publication did not advance exactly G+1")
        for field in (
            "account_id",
            "device_installation_id",
            "state_store_identity_fingerprint_sha256",
            "state_store_schema_version",
        ):
            if getattr(final_metadata, field) != getattr(source.metadata, field):
                raise RuntimeSessionHistoryError("StateStore scope changed during publication")
        matches = [
            item
            for item in final.immutable_history
            if item.record_key == session.runtime_session_id
        ]
        if matches != [carrier]:
            raise RuntimeSessionHistoryError("exact RuntimeSession carrier is not durable once")
        old = {
            (item.representation_name, item.record_key): item for item in source.immutable_history
        }
        now = {
            (item.representation_name, item.record_key): item for item in final.immutable_history
        }
        if any(now.get(key) != value for key, value in old.items()):
            raise RuntimeSessionHistoryError("old immutable history changed")
        if final.current_records != source.current_records:
            raise RuntimeSessionHistoryError("RuntimeSession publication changed current records")
        return RuntimeSessionPublicationResult(
            session.runtime_session_id,
            source.metadata.protected_freshness_generation,
            final_metadata,
        )


__all__ = [
    "RUNTIME_SESSION_REPRESENTATION",
    "RuntimeSessionHistoryError",
    "RuntimeSessionHistoryPublisher",
    "RuntimeSessionPublicationResult",
    "runtime_session_carrier",
]
