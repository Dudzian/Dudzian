"""Production M0.11 forward-only migration lifecycle and registry boundary."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from threading import RLock
from typing import Any, Protocol, cast

from .fingerprints import canonical_json_sha256
from .lifecycle_records import (
    LifecycleIntegrityError,
    append_idempotently,
    fingerprint_without,
    persistence_record,
    validate_chain,
)
from .protected_freshness_handoff import (
    ProtectedFreshnessHandoffCoordinator,
    ProtectedFreshnessHandoffError,
)
from .state_store import SQLiteStateStore, StateStoreSnapshot
from .migration_execution import (
    MigrationExecutionAuthority,
    MigrationExecutionError,
    MigrationExecutionPlan,
)
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


def migration_mapping_payload(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MigrationError("migration carrier payload must be an object")
    return value


MIGRATION_FAMILY_REPRESENTATIONS = frozenset(
    {
        "Migration transition/history revisions",
        "Migration current state/designation",
        "Migration execution declaration",
    }
)


def migration_family_ids(snapshot: StateStoreSnapshot) -> tuple[str, ...]:
    """Discover exact Migration families in one pinned verified snapshot."""

    identities: set[str] = set()
    for carrier in (*snapshot.current_records, *snapshot.immutable_history):
        if carrier.representation_name in MIGRATION_FAMILY_REPRESENTATIONS:
            migration_id = migration_mapping_payload(carrier.payload).get("migration_id")
            if not isinstance(migration_id, str) or not migration_id:
                raise MigrationError("migration carrier identity is invalid")
            identities.add(migration_id)
    return tuple(sorted(identities))


@dataclass(frozen=True, slots=True)
class MigrationRecord:
    """Ephemeral, snapshot-bound runtime candidate; never durable authority."""

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
        return cast(
            str,
            migration_definition_fingerprint(
                migration_id=self.migration_id,
                source_schema_version=self.source_schema_version,
                target_schema_version=self.target_schema_version,
                ordered_path=self.ordered_path,
                rollback_policy=self.rollback_policy,
            ),
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
    record: MigrationRecord,
    history: Sequence[Mapping[str, Any]],
    current: Mapping[str, Any] | None,
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


def derive_runtime_migration_record(
    definition: MigrationDefinition,
    snapshot: StateStoreSnapshot,
    *,
    derived_post_state_fingerprint_sha256: str,
) -> MigrationRecord:
    """Reconstruct a runtime binding from sealed authority and a fresh snapshot."""

    metadata = snapshot.metadata
    record = MigrationRecord(
        migration_id=definition.migration_id,
        source_schema_version=definition.source_schema_version,
        target_schema_version=definition.target_schema_version,
        ordered_path=definition.ordered_path,
        scope=(metadata.account_id, metadata.device_installation_id),
        environment=metadata.environment,
        pre_state_fingerprint_sha256=metadata.state_fingerprint_sha256,
        post_state_fingerprint_sha256=derived_post_state_fingerprint_sha256,
        transaction_fingerprint_sha256=metadata.transaction_fingerprint_sha256,
        protected_freshness_generation=metadata.protected_freshness_generation,
        rollback_policy=definition.rollback_policy,
    )
    bind_runtime_migration_instance(
        definition,
        record,
        snapshot,
        derived_post_state_fingerprint_sha256=derived_post_state_fingerprint_sha256,
    )
    return record


def migration_target_materialized(
    definition: MigrationDefinition,
    record: MigrationRecord,
    snapshot: StateStoreSnapshot,
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
    return bool(
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
        self,
        entries: Iterable[
            tuple[MigrationDefinition, MigrationExecutionAuthority, MigrationPlanner]
        ] = (),
    ) -> None:
        definitions: dict[str, MigrationDefinition] = {}
        authorities: dict[str, MigrationExecutionAuthority] = {}
        steps: dict[str, MigrationPlanner] = {}
        for entry in entries:
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise MigrationError(
                    "migration registry entries require definition, authority, planner"
                )
            definition, authority, step = entry
            if (
                not isinstance(definition, MigrationDefinition)
                or not isinstance(authority, MigrationExecutionAuthority)
                or not callable(step)
            ):
                raise MigrationError("migration registry entry has invalid types")
            if definition.migration_id in definitions:
                raise MigrationError("duplicate migration definition ID")
            try:
                authority.assert_definition(definition)
            except MigrationExecutionError as exc:
                raise MigrationError("migration execution authority mismatch") from exc
            definitions[definition.migration_id] = definition
            authorities[definition.migration_id] = authority
            steps[definition.migration_id] = step
        self._definitions = definitions
        self._authorities = authorities
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

    def execution_authority_for(self, migration_id: str) -> MigrationExecutionAuthority:
        try:
            return self._authorities[migration_id]
        except KeyError as exc:
            raise MigrationError("migration execution authority unavailable") from exc

    def resolve(self, record: MigrationRecord) -> MigrationPlanner:
        self.assert_static_match(record)
        try:
            return self._steps[record.migration_id]
        except KeyError as exc:
            raise MigrationError("exact migration path unavailable") from exc

    def planner_for(self, definition: MigrationDefinition) -> MigrationPlanner:
        """Resolve only a definition object minted by this sealed registry."""

        trusted = self.definition_for(definition.migration_id)
        if definition is not trusted:
            raise MigrationError("planner requires the exact trusted definition")
        try:
            return self._steps[definition.migration_id]
        except KeyError as exc:
            raise MigrationError("exact migration path unavailable") from exc

    def authorized_plan_for(
        self, definition: MigrationDefinition, snapshot: StateStoreSnapshot
    ) -> MigrationExecutionPlan:
        authority = self.execution_authority_for(definition.migration_id)
        planner = self.planner_for(definition)
        plan = planner(snapshot)
        if not isinstance(plan, MigrationExecutionPlan):
            raise MigrationError("trusted planner returned an invalid execution plan")
        try:
            authority.assert_definition(definition)
            authority.assert_plan(plan)
        except MigrationExecutionError as exc:
            raise MigrationError("trusted plan does not match execution authority") from exc
        return plan


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
                # This legacy semantic coordinator has neither the verified
                # source snapshot nor the protected execution capability needed
                # to resolve a sealed plan. Effects are owned exclusively by
                # MigrationExecutionCoordinator.authorized_plan_for().
                raise MigrationError("migration effects require the sealed execution coordinator")
            if state == "DURABLE_MIGRATED":
                if verified_schema_version != record.target_schema_version:
                    raise MigrationError("durable target is not verified")
                finalize()
                return "COMPLETED"
            return "APPLYING"


from .state_store_v2_migration import (
    STATE_STORE_V2_MIGRATION_AUTHORITY,
    STATE_STORE_V2_MIGRATION_DEFINITION,
    plan_state_store_v2_migration,
)

PRODUCTION_MIGRATION_REGISTRY = MigrationRegistry(
    (
        (
            STATE_STORE_V2_MIGRATION_DEFINITION,
            STATE_STORE_V2_MIGRATION_AUTHORITY,
            plan_state_store_v2_migration,
        ),
    )
)


@dataclass(frozen=True, slots=True)
class DurableMigrationLifecycle:
    """Validated durable view of one sealed migration definition."""

    definition: MigrationDefinition
    history: tuple[Mapping[str, Any], ...]
    current: Mapping[str, Any] | None


class DurableMigrationLifecycleCoordinator:
    """StateStore-backed semantic CAS entry points for migration lifecycle facts."""

    def __init__(
        self,
        store: SQLiteStateStore,
        registry: MigrationRegistry,
        protected: ProtectedFreshnessHandoffCoordinator,
    ) -> None:
        self._store = store
        self._registry = registry
        self._protected = protected

    def discover(self, migration_id: str) -> DurableMigrationLifecycle:
        snapshot = self._store.read_verified_snapshot()
        if snapshot is None:
            raise MigrationError("migration lifecycle requires initialized StateStore")
        return self._view(snapshot, migration_id)

    def view_verified_snapshot(
        self, snapshot: StateStoreSnapshot, migration_id: str
    ) -> DurableMigrationLifecycle:
        """Validate one lifecycle against the caller's pinned verified snapshot."""

        if not isinstance(snapshot, StateStoreSnapshot):
            raise TypeError("snapshot must be a StateStoreSnapshot")
        return self._view(snapshot, migration_id)

    def _view(self, snapshot: StateStoreSnapshot, migration_id: str) -> DurableMigrationLifecycle:
        definition = self._registry.definition_for(migration_id)
        current_record, history_records = SQLiteStateStore._select_lifecycle(
            snapshot,
            identity=migration_id,
            current_name="Migration current state/designation",
            history_name="Migration transition/history revisions",
            current_key=f"migration-current:{migration_id}",
            history_key_prefix=f"migration-transition:{migration_id}:",
        )
        history = tuple(migration_mapping_payload(record.payload) for record in history_records)
        current = (
            None if current_record is None else migration_mapping_payload(current_record.payload)
        )
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
        except (LifecycleIntegrityError, KeyError, TypeError, ValueError) as exc:
            raise MigrationError(str(exc)) from exc
        return DurableMigrationLifecycle(definition, history, current)

    def prepare(self, migration_id: str) -> DurableMigrationLifecycle:
        return self._advance(migration_id, expected=None, target="PREPARED")

    def begin_applying(self, migration_id: str) -> DurableMigrationLifecycle:
        return self._advance(migration_id, expected="PREPARED", target="APPLYING")

    def fail(self, migration_id: str) -> DurableMigrationLifecycle:
        return self._advance(
            migration_id,
            expected=("PREPARED", "APPLYING", "DURABLE_MIGRATED"),
            target="FAILED",
        )

    def complete(self, migration_id: str) -> DurableMigrationLifecycle:
        return self._advance(migration_id, expected="DURABLE_MIGRATED", target="COMPLETED")

    def record_durable_migrated(
        self, migration_id: str, materialization: MigrationRecord
    ) -> DurableMigrationLifecycle:
        if materialization.migration_id != migration_id:
            raise MigrationError("materialization proof identity mismatch")
        return self._advance(
            migration_id,
            expected="APPLYING",
            target="DURABLE_MIGRATED",
            materialization=materialization,
        )

    def _advance(
        self,
        migration_id: str,
        *,
        expected: str | tuple[str, ...] | None,
        target: str,
        materialization: MigrationRecord | None = None,
    ) -> DurableMigrationLifecycle:
        lifecycle = self.discover(migration_id)
        state = None if lifecycle.current is None else str(lifecycle.current["state"])
        if state == target:
            return self._recover_duplicate(migration_id, target)
        allowed_predecessors = (expected,) if isinstance(expected, str) else expected
        if state is not None or expected is not None:
            if allowed_predecessors is None or state not in allowed_predecessors:
                raise MigrationError(f"expected {expected!r}, found {state!r}")

        def build(source: StateStoreSnapshot):
            authoritative = self._view(source, migration_id)
            source_state = (
                None if authoritative.current is None else str(authoritative.current["state"])
            )
            if source_state == target:
                raise _LifecycleAlreadyApplied
            if source_state is not None or expected is not None:
                if allowed_predecessors is None or source_state not in allowed_predecessors:
                    raise MigrationError(f"expected {expected!r}, found {source_state!r}")
            if materialization is not None and not migration_target_materialized(
                authoritative.definition, materialization, source
            ):
                raise MigrationError("verified migration target is not materially proven")
            required_schema = (
                authoritative.definition.source_schema_version
                if target in {"PREPARED", "APPLYING"}
                else (
                    authoritative.definition.target_schema_version
                    if target == "COMPLETED"
                    else None
                )
            )
            if (
                required_schema is not None
                and source.metadata.state_store_schema_version != required_schema
            ):
                raise MigrationError(
                    f"{target} requires verified StateStore schema {required_schema}"
                )
            observation = source.metadata
            revision = len(authoritative.history) + 1
            transition = migration_transition(
                migration_id=migration_id,
                transition_revision=revision,
                previous_state=source_state,
                state=target,
                transaction_fingerprint_sha256=observation.transaction_fingerprint_sha256,
                state_fingerprint_sha256=observation.state_fingerprint_sha256,
                protected_freshness_generation=observation.protected_freshness_generation,
            )
            current = migration_current(
                migration_id=migration_id,
                current_transition_revision=revision,
                state=target,
                authoritative_state_fingerprint_sha256=observation.state_fingerprint_sha256,
                protected_freshness_generation=observation.protected_freshness_generation,
            )
            return (
                replace(
                    observation,
                    protected_freshness_generation=observation.protected_freshness_generation + 1,
                ),
                (migration_current_carrier(current),),
                (migration_transition_carrier(transition),),
            )

        try:
            self._protected.advance_protected_mutation(build)
        except _LifecycleAlreadyApplied:
            return self._recover_duplicate(migration_id, target)
        except MigrationError:
            raise
        except (ProtectedFreshnessHandoffError, RuntimeError) as exc:
            raise MigrationError("protected migration CAS failed") from exc
        return self.discover(migration_id)

    def _recover_duplicate(self, migration_id: str, target: str) -> DurableMigrationLifecycle:
        snapshot = self._store.read_verified_snapshot()
        if snapshot is None:
            raise MigrationError("migration lifecycle requires initialized StateStore")
        lifecycle = self._view(snapshot, migration_id)
        if lifecycle.current is None or lifecycle.current["state"] != target:
            raise MigrationError("durable migration duplicate is no longer current")
        scope = (
            snapshot.metadata.account_id,
            snapshot.metadata.device_installation_id,
            snapshot.metadata.state_store_identity_fingerprint_sha256,
        )
        try:
            recovered = self._protected.recover_protected_state(scope)
        except ProtectedFreshnessHandoffError as exc:
            raise MigrationError("protected migration duplicate recovery failed") from exc
        if recovered != snapshot.metadata:
            raise MigrationError("protected migration duplicate recovery changed local state")
        current = self.discover(migration_id)
        if current.current is None or current.current["state"] != target:
            raise MigrationError("migration duplicate changed during protected recovery")
        return current


class _LifecycleAlreadyApplied(RuntimeError):
    pass
