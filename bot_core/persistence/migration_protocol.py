"""Production M0.11 forward-only migration lifecycle and registry boundary."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from threading import RLock
from typing import Any, Protocol

from .fingerprints import canonical_json_sha256
from .lifecycle_records import (
    LifecycleIntegrityError,
    append_idempotently,
    fingerprint_without,
    persistence_record,
    validate_chain,
)
from .state_store import StateStoreSnapshot
from .migration_execution import MigrationExecutionPlan
from .migration_execution_contract import migration_definition_fingerprint

MIGRATION_STATES = frozenset({"PREPARED", "APPLYING", "DURABLE_MIGRATED", "COMPLETED", "FAILED"})
MIGRATION_TRANSITIONS = {
    "PREPARED": frozenset({"APPLYING", "FAILED"}),
    "APPLYING": frozenset({"DURABLE_MIGRATED", "FAILED"}),
    "DURABLE_MIGRATED": frozenset({"COMPLETED", "FAILED"}),
    "COMPLETED": frozenset(),
    "FAILED": frozenset(),
}
_SHA = __import__("re").compile(r"^[0-9a-f]{64}$")


class MigrationError(LifecycleIntegrityError):
    pass


@dataclass(frozen=True, slots=True)
class MigrationRecord:
    migration_id: str
    source_schema_version: int
    target_schema_version: int
    ordered_path: tuple[str, ...]
    scope: tuple[str, str]
    environment: str
    pre_state_fingerprint_sha256: str
    post_state_fingerprint_sha256: str
    transaction_fingerprint_sha256: str
    protected_freshness_generation: int
    rollback_policy: str

    def __post_init__(self) -> None:
        for name in (
            "source_schema_version",
            "target_schema_version",
            "protected_freshness_generation",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise MigrationError(f"{name} must be a positive non-boolean integer")
        if not isinstance(self.migration_id, str) or not self.migration_id:
            raise MigrationError("migration_id must be non-empty")
        if not isinstance(self.ordered_path, tuple) or not all(
            isinstance(x, str) and x for x in self.ordered_path
        ):
            raise MigrationError("ordered_path must be an exact string tuple")
        if (
            not isinstance(self.scope, tuple)
            or len(self.scope) != 2
            or not all(isinstance(x, str) and x for x in self.scope)
        ):
            raise MigrationError("scope must contain account and device identifiers")
        if self.environment not in {"PAPER", "TESTNET", "LIVE"}:
            raise MigrationError("invalid environment")
        if self.rollback_policy != "FORWARD_ONLY":
            raise MigrationError("rollback policy must be FORWARD_ONLY")
        for name in (
            "pre_state_fingerprint_sha256",
            "post_state_fingerprint_sha256",
            "transaction_fingerprint_sha256",
        ):
            if not isinstance(getattr(self, name), str) or not _SHA.fullmatch(getattr(self, name)):
                raise MigrationError(f"{name} must be lowercase SHA-256")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MigrationRecord":
        if set(value) != {f.name for f in fields(cls)}:
            raise MigrationError("MigrationRecord requires its exact field set")
        data = dict(value)
        if not isinstance(data["ordered_path"], list) or not isinstance(data["scope"], list):
            raise MigrationError("JSON ordered_path and scope must be arrays")
        data["ordered_path"], data["scope"] = tuple(data["ordered_path"]), tuple(data["scope"])
        return cls(**data)

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["ordered_path"] = list(self.ordered_path)
        value["scope"] = list(self.scope)
        return value


@dataclass(frozen=True, slots=True)
class MigrationDefinition:
    """Trusted build-time definition, deliberately free of runtime StateStore facts."""

    migration_id: str
    source_schema_version: int
    target_schema_version: int
    ordered_path: tuple[str, ...]
    rollback_policy: str = "FORWARD_ONLY"

    def __post_init__(self) -> None:
        if not isinstance(self.migration_id, str) or not self.migration_id:
            raise MigrationError("migration definition ID must be non-empty")
        for name in ("source_schema_version", "target_schema_version"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise MigrationError(f"{name} must be a positive non-boolean integer")
        if not isinstance(self.ordered_path, tuple) or not all(
            isinstance(item, str) and item for item in self.ordered_path
        ):
            raise MigrationError("ordered_path must be an exact string tuple")
        if self.rollback_policy != "FORWARD_ONLY":
            raise MigrationError("rollback policy must be FORWARD_ONLY")

    def matches_static_fields(self, record: MigrationRecord) -> bool:
        return (
            record.migration_id == self.migration_id
            and record.source_schema_version == self.source_schema_version
            and record.target_schema_version == self.target_schema_version
            and record.ordered_path == self.ordered_path
            and record.rollback_policy == self.rollback_policy
        )

    def fingerprint(self) -> str:
        return migration_definition_fingerprint(
            migration_id=self.migration_id,
            source_schema_version=self.source_schema_version,
            target_schema_version=self.target_schema_version,
            ordered_path=self.ordered_path,
            rollback_policy=self.rollback_policy,
        )


def migration_transition(**fields_: Any) -> dict[str, Any]:
    value = dict(fields_)
    value["transition_fingerprint_sha256"] = fingerprint_without(
        value, "transition_fingerprint_sha256"
    )
    return value


def migration_current(**fields_: Any) -> dict[str, Any]:
    value = dict(fields_)
    value["designation_fingerprint_sha256"] = fingerprint_without(
        value, "designation_fingerprint_sha256"
    )
    return value


def migration_transition_carrier(value: Mapping[str, Any]):
    return persistence_record(
        "Migration transition/history revisions",
        f"migration-transition:{value['migration_id']}:{value['transition_revision']}",
        value,
    )


def migration_current_carrier(value: Mapping[str, Any]):
    return persistence_record(
        "Migration current state/designation",
        f"migration-current:{value['migration_id']}",
        value,
    )


def validate_migration_lifecycle(
    record: MigrationRecord, history: Sequence[Mapping[str, Any]], current: Mapping[str, Any] | None
) -> None:
    try:
        validate_chain(
            history,
            current,
            identity_field="migration_id",
            allowed=MIGRATION_TRANSITIONS,
            transition_hash_field="transition_fingerprint_sha256",
            current_hash_field="designation_fingerprint_sha256",
        )
        if current is not None and (
            current["authoritative_state_fingerprint_sha256"]
            != history[-1]["state_fingerprint_sha256"]
            or current["protected_freshness_generation"]
            != history[-1]["protected_freshness_generation"]
        ):
            raise MigrationError("current designation is not bound to latest observation")
    except LifecycleIntegrityError as exc:
        raise MigrationError(str(exc)) from exc


def bind_runtime_migration_instance(
    definition: MigrationDefinition,
    record: MigrationRecord,
    snapshot: StateStoreSnapshot,
    *,
    derived_post_state_fingerprint_sha256: str,
) -> None:
    """Bind one runtime instance to one verified source snapshot and derived target."""

    metadata = snapshot.metadata
    if not definition.matches_static_fields(record):
        raise MigrationError("runtime record does not match trusted static definition")
    if (
        record.scope != (metadata.account_id, metadata.device_installation_id)
        or record.environment != metadata.environment
        or record.source_schema_version != metadata.state_store_schema_version
        or record.pre_state_fingerprint_sha256 != metadata.state_fingerprint_sha256
        or record.transaction_fingerprint_sha256 != metadata.transaction_fingerprint_sha256
        or record.protected_freshness_generation != metadata.protected_freshness_generation
        or record.post_state_fingerprint_sha256 != derived_post_state_fingerprint_sha256
    ):
        raise MigrationError("runtime migration instance is not exactly source/target bound")


def migration_target_materialized(
    definition: MigrationDefinition, record: MigrationRecord, snapshot: StateStoreSnapshot
) -> bool:
    """Prove the frozen DURABLE_MIGRATED gate from one verified snapshot.

    The caller must obtain ``snapshot`` through ``read_verified_snapshot``.  This
    predicate mints no authority; it only compares the immutable plan to the
    already verified local observation.
    """

    if not definition.matches_static_fields(record):
        return False
    metadata = snapshot.metadata
    if not snapshot.transaction_descriptors:
        return False
    descriptor = max(snapshot.transaction_descriptors, key=lambda item: item.target_generation)
    return (
        metadata.account_id == record.scope[0]
        and metadata.device_installation_id == record.scope[1]
        and metadata.environment == record.environment
        and metadata.state_store_schema_version == record.target_schema_version
        and descriptor.expected_current_generation == record.protected_freshness_generation
        and descriptor.pre_state_fingerprint_sha256 == record.pre_state_fingerprint_sha256
        and metadata.state_fingerprint_sha256 == record.post_state_fingerprint_sha256
        and descriptor.account_id == record.scope[0]
        and descriptor.device_installation_id == record.scope[1]
        and descriptor.environment == record.environment
        and descriptor.state_store_identity_fingerprint_sha256
        == metadata.state_store_identity_fingerprint_sha256
        and descriptor.target_generation == record.protected_freshness_generation + 1
    )


class MigrationPlanner(Protocol):
    """Trusted deterministic planner without database mutation authority."""

    def __call__(self, snapshot: StateStoreSnapshot) -> MigrationExecutionPlan: ...


class MigrationRegistry:
    """Sealed trusted build-time registry; an empty registry authorizes nothing."""

    def __init__(
        self, entries: Iterable[tuple[MigrationDefinition, MigrationPlanner]] = ()
    ) -> None:
        definitions: dict[str, MigrationDefinition] = {}
        steps: dict[str, MigrationPlanner] = {}
        for definition, step in entries:
            if definition.migration_id in definitions:
                raise MigrationError("duplicate migration definition ID")
            definitions[definition.migration_id] = definition
            steps[definition.migration_id] = step
        self._definitions = definitions
        self._steps = steps

    def definition_for(self, migration_id: str) -> MigrationDefinition:
        try:
            return self._definitions[migration_id]
        except KeyError as exc:
            raise MigrationError("migration definition unavailable") from exc

    def assert_static_match(self, candidate: MigrationRecord) -> MigrationDefinition:
        definition = self.definition_for(candidate.migration_id)
        if not definition.matches_static_fields(candidate):
            raise MigrationError("candidate static fields do not match trusted definition")
        return definition

    def resolve(self, record: MigrationRecord) -> MigrationPlanner:
        self.assert_static_match(record)
        try:
            return self._steps[record.migration_id]
        except KeyError as exc:
            raise MigrationError("exact migration path unavailable") from exc


class MigrationCoordinator:
    """Serializes an ID and derives recovery solely from validated durable carriers."""

    def __init__(self, registry: MigrationRegistry) -> None:
        self._registry, self._lock = registry, RLock()

    def resume(
        self,
        record: MigrationRecord,
        history: Sequence[Mapping[str, Any]],
        current: Mapping[str, Any] | None,
        *,
        verified_schema_version: int,
        apply_once: Any,
        finalize: Any,
    ) -> str:
        with self._lock:
            self._registry.assert_static_match(record)
            validate_migration_lifecycle(record, history, current)
            if current is None:
                raise MigrationError("descriptor alone has no lifecycle authority")
            state = str(current["state"])
            if state in {"COMPLETED", "FAILED"}:
                return state
            if state == "PREPARED" and verified_schema_version != record.source_schema_version:
                raise MigrationError("verified source schema mismatch")
            if state == "APPLYING":
                if verified_schema_version == record.target_schema_version:
                    return "DURABLE_MIGRATED"
                if verified_schema_version != record.source_schema_version:
                    raise MigrationError("partial or unknown schema state")
                apply_once(self._registry.resolve(record))
                return "DURABLE_MIGRATED"
            if state == "DURABLE_MIGRATED":
                if verified_schema_version != record.target_schema_version:
                    raise MigrationError("durable target is not verified")
                finalize()
                return "COMPLETED"
            return "APPLYING"


PRODUCTION_MIGRATION_REGISTRY = MigrationRegistry()
