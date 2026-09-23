"""Carrier-owned executable accepted-content and M0.9 kill-switch authorities."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import re
from threading import RLock
from typing import Callable, ContextManager, Iterator, Protocol, TypeVar

from bot_core.persistence.fingerprints import canonical_json_sha256

SCOPE_PREFIXES = {
    "PRODUCT_SYSTEM": None,
    "WORKSPACE": "ws",
    "PORTFOLIO": "port",
    "EXCHANGE_ACCOUNT": "xacc",
    "STRATEGY_INSTANCE": "sinst",
    "INSTRUMENT": "instr",
    "EXECUTION_ROUTE": "xroute",
}
ENVIRONMENTS = frozenset({"PAPER", "TESTNET", "LIVE"})
STATES = frozenset({"INACTIVE", "ACTIVE"})
UUID7 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
T = TypeVar("T")


class KillSwitchAuthorityError(ValueError):
    """Fail-closed M0.9 trusted-context failure."""


@dataclass(frozen=True, slots=True)
class CoreAcceptedContentBinding:
    membership_id: str
    content_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class AtomicCoreAcceptedContentState:
    store_revision: int = 0
    accepted: tuple[CoreAcceptedContentBinding, ...] = ()


class CoreAcceptedContentCarrier(Protocol):
    def authority_fence(self) -> ContextManager[None]: ...
    def read(self) -> AtomicCoreAcceptedContentState: ...
    def compare_and_swap(
        self, expected_revision: int, state: AtomicCoreAcceptedContentState
    ) -> None: ...


class InMemoryCoreAcceptedContentCarrier:
    """Reference durable carrier for Core-owned, append-only content membership."""

    def __init__(self, state: AtomicCoreAcceptedContentState | None = None) -> None:
        self._lock = RLock()
        self._state = state or AtomicCoreAcceptedContentState()

    @contextmanager
    def authority_fence(self) -> Iterator[None]:
        with self._lock:
            yield

    def read(self) -> AtomicCoreAcceptedContentState:
        with self._lock:
            return self._state

    def compare_and_swap(
        self, expected_revision: int, state: AtomicCoreAcceptedContentState
    ) -> None:
        with self._lock:
            if self._state.store_revision != expected_revision:
                raise RuntimeError("AUTHORITY_CAS_CONFLICT")
            self._state = state


class _CoreAcceptedContentWriter:
    """Non-exported Core owner capability; DTO construction grants no membership."""

    def __init__(self, authority: CoreAcceptedContentAuthority) -> None:
        self.__authority = authority

    def accept(self, binding: CoreAcceptedContentBinding) -> None:
        self.__authority._accept(binding)  # noqa: SLF001


class CoreAcceptedContentAuthority:
    """Independent carrier-owned registry required by frozen Core bindings."""

    @classmethod
    def compose(
        cls, carrier: CoreAcceptedContentCarrier
    ) -> tuple[CoreAcceptedContentAuthority, _CoreAcceptedContentWriter]:
        authority = cls(carrier)
        return authority, _CoreAcceptedContentWriter(authority)

    def __init__(self, carrier: CoreAcceptedContentCarrier) -> None:
        self._carrier = carrier
        with carrier.authority_fence():
            self._validate_state(carrier.read())

    @staticmethod
    def _valid(binding: object) -> bool:
        return (
            isinstance(binding, CoreAcceptedContentBinding)
            and type(binding.membership_id) is str
            and bool(binding.membership_id)
            and type(binding.content_fingerprint_sha256) is str
            and bool(SHA256.fullmatch(binding.content_fingerprint_sha256))
        )

    @classmethod
    def _validate_state(cls, state: object) -> None:
        if (
            not isinstance(state, AtomicCoreAcceptedContentState)
            or type(state.store_revision) is not int
            or state.store_revision < 0
            or type(state.accepted) is not tuple
            or state.store_revision != len(state.accepted)
            or any(not cls._valid(item) for item in state.accepted)
            or len({item.membership_id for item in state.accepted}) != len(state.accepted)
        ):
            raise KillSwitchAuthorityError("CORRUPT_CORE_ACCEPTED_CONTENT_STATE")

    def _accept(self, binding: CoreAcceptedContentBinding) -> None:
        if not self._valid(binding):
            raise KillSwitchAuthorityError("TRUSTED_CONTEXT_FAILURE")
        with self._carrier.authority_fence():
            state = self._carrier.read()
            self._validate_state(state)
            prior = next(
                (item for item in state.accepted if item.membership_id == binding.membership_id),
                None,
            )
            if prior is not None:
                if prior != binding:
                    raise KillSwitchAuthorityError("TRUSTED_CONTEXT_FAILURE")
                return
            self._carrier.compare_and_swap(
                state.store_revision,
                AtomicCoreAcceptedContentState(
                    state.store_revision + 1, state.accepted + (binding,)
                ),
            )

    def resolve(self, membership_id: str) -> CoreAcceptedContentBinding | None:
        with self._carrier.authority_fence():
            state = self._carrier.read()
            self._validate_state(state)
            return next(
                (item for item in state.accepted if item.membership_id == membership_id), None
            )

    def has_content_fingerprint(self, fingerprint: str) -> bool:
        """Return whether any accepted membership binds this exact content.

        Membership identity is ``membership_id``.  Consequently multiple legal
        memberships may bind identical content and must not make that content
        cease to be accepted.
        """
        with self._carrier.authority_fence():
            state = self._carrier.read()
            self._validate_state(state)
            return any(item.content_fingerprint_sha256 == fingerprint for item in state.accepted)


@dataclass(frozen=True, slots=True)
class KillSwitchRecord:
    scope_type: str
    scope_id: str
    environment: str
    state: str
    source_revision: int
    effective_at_utc: str
    generation: int
    accepted_authority_fingerprint_sha256: str
    record_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class PrevalidatedKillSwitchContext:
    history: tuple[KillSwitchRecord, ...]
    membership_id: str
    context_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class AcceptedKillSwitchAuthorityEntry:
    transaction_revision: int
    accepted_at_utc: str
    context: PrevalidatedKillSwitchContext


@dataclass(frozen=True, slots=True)
class AtomicKillSwitchAuthorityState:
    store_revision: int = 0
    accepted: tuple[AcceptedKillSwitchAuthorityEntry, ...] = ()
    current: tuple[tuple[tuple[str, str, str], str], ...] = ()


class KillSwitchAuthorityCarrier(Protocol):
    def authority_fence(self) -> ContextManager[None]: ...
    def read(self) -> AtomicKillSwitchAuthorityState: ...
    def compare_and_swap(
        self, expected_revision: int, state: AtomicKillSwitchAuthorityState
    ) -> None: ...


class InMemoryKillSwitchAuthorityCarrier:
    def __init__(self, state: AtomicKillSwitchAuthorityState | None = None) -> None:
        self._lock = RLock()
        self._state = state or AtomicKillSwitchAuthorityState()
        self.fail_next = False

    @contextmanager
    def authority_fence(self) -> Iterator[None]:
        with self._lock:
            yield

    def read(self) -> AtomicKillSwitchAuthorityState:
        with self._lock:
            return self._state

    def compare_and_swap(
        self, expected_revision: int, state: AtomicKillSwitchAuthorityState
    ) -> None:
        with self._lock:
            if self.fail_next:
                self.fail_next = False
                raise OSError("INJECTED_CARRIER_FAILURE")
            if self._state.store_revision != expected_revision:
                raise RuntimeError("AUTHORITY_CAS_CONFLICT")
            self._state = state


class _KillSwitchWriter:
    def __init__(self, authority: KillSwitchAuthority) -> None:
        self.__authority = authority

    def accept(
        self, context: PrevalidatedKillSwitchContext, *, now_utc: str
    ) -> AcceptedKillSwitchAuthorityEntry:
        return self.__authority._accept(context, now_utc=now_utc)  # noqa: SLF001


class KillSwitchAuthority:
    @classmethod
    def compose(
        cls, carrier: KillSwitchAuthorityCarrier, *, core_membership: CoreAcceptedContentAuthority
    ) -> tuple[KillSwitchAuthority, _KillSwitchWriter]:
        authority = cls(carrier, core_membership=core_membership)
        return authority, _KillSwitchWriter(authority)

    def __init__(
        self, carrier: KillSwitchAuthorityCarrier, *, core_membership: CoreAcceptedContentAuthority
    ) -> None:
        if not isinstance(core_membership, CoreAcceptedContentAuthority):
            raise KillSwitchAuthorityError("TRUSTED_CONTEXT_FAILURE")
        self._carrier = carrier
        self._core = core_membership
        with carrier.authority_fence():
            self._validate_state(carrier.read())

    @staticmethod
    def record_fingerprint(record: KillSwitchRecord) -> str:
        payload = asdict(record)
        payload.pop("record_fingerprint_sha256")
        return canonical_json_sha256(payload)

    @staticmethod
    def history_fingerprint(history: tuple[KillSwitchRecord, ...]) -> str:
        return canonical_json_sha256([asdict(record) for record in history])

    @staticmethod
    def context_fingerprint(context: PrevalidatedKillSwitchContext) -> str:
        return canonical_json_sha256(
            [[asdict(record) for record in context.history], context.membership_id]
        )

    @staticmethod
    def _scope(record: KillSwitchRecord) -> tuple[str, str, str]:
        return record.scope_type, record.scope_id, record.environment

    @classmethod
    def _validate_record(cls, record: object) -> KillSwitchRecord:
        if not isinstance(record, KillSwitchRecord):
            raise KillSwitchAuthorityError("TRUSTED_CONTEXT_FAILURE")
        prefix = SCOPE_PREFIXES.get(record.scope_type, "UNKNOWN")
        valid_scope = (
            record.scope_id == "product"
            if prefix is None
            else (
                prefix != "UNKNOWN"
                and isinstance(record.scope_id, str)
                and record.scope_id.startswith(prefix + "_")
                and bool(UUID7.fullmatch(record.scope_id[len(prefix) + 1 :]))
            )
        )
        try:
            parsed = datetime.fromisoformat(record.effective_at_utc.removesuffix("Z") + "+00:00")
            valid_time = (
                record.effective_at_utc.endswith("Z")
                and parsed.tzinfo == timezone.utc
                and parsed.isoformat().replace("+00:00", "Z") == record.effective_at_utc
            )
        except (AttributeError, ValueError):
            valid_time = False
        if (
            not valid_scope
            or record.environment not in ENVIRONMENTS
            or record.state not in STATES
            or type(record.source_revision) is not int
            or record.source_revision < 1
            or type(record.generation) is not int
            or record.generation < 1
            or not valid_time
            or type(record.accepted_authority_fingerprint_sha256) is not str
            or not SHA256.fullmatch(record.accepted_authority_fingerprint_sha256)
            or type(record.record_fingerprint_sha256) is not str
            or not SHA256.fullmatch(record.record_fingerprint_sha256)
            or cls.record_fingerprint(record) != record.record_fingerprint_sha256
        ):
            raise KillSwitchAuthorityError("TRUSTED_CONTEXT_FAILURE")
        return record

    def _validate_context(self, context: object) -> PrevalidatedKillSwitchContext:
        if (
            not isinstance(context, PrevalidatedKillSwitchContext)
            or type(context.history) is not tuple
            or not context.history
            or type(context.membership_id) is not str
            or not context.membership_id
            or type(context.context_fingerprint_sha256) is not str
            or not SHA256.fullmatch(context.context_fingerprint_sha256)
        ):
            raise KillSwitchAuthorityError("TRUSTED_CONTEXT_FAILURE")
        generations: dict[tuple[str, str, str], int] = {}
        for raw in context.history:
            record = self._validate_record(raw)
            scope = self._scope(record)
            if record.generation <= generations.get(scope, 0):
                raise KillSwitchAuthorityError("TRUSTED_CONTEXT_FAILURE")
            generations[scope] = record.generation
            # This field is an accepted-membership reference, not merely SHA syntax.
            if not self._core.has_content_fingerprint(record.accepted_authority_fingerprint_sha256):
                raise KillSwitchAuthorityError("TRUSTED_CONTEXT_FAILURE")
        binding = self._core.resolve(context.membership_id)
        if (
            binding is None
            or binding.content_fingerprint_sha256 != self.history_fingerprint(context.history)
            or context.context_fingerprint_sha256 != self.context_fingerprint(context)
        ):
            raise KillSwitchAuthorityError("TRUSTED_CONTEXT_FAILURE")
        return context

    @staticmethod
    def _transaction_time(value: object, *, error: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
            valid = (
                type(value) is str
                and value.endswith("Z")
                and parsed.tzinfo == timezone.utc
                and parsed.isoformat().replace("+00:00", "Z") == value
            )
        except (AttributeError, ValueError):
            valid = False
        if not valid:
            raise KillSwitchAuthorityError(error)
        return parsed

    def _accept(
        self, context: PrevalidatedKillSwitchContext, *, now_utc: str
    ) -> AcceptedKillSwitchAuthorityEntry:
        context = self._validate_context(context)
        with self._carrier.authority_fence():
            before = self._carrier.read()
            self._validate_state(before)
            replay = next(
                (
                    entry
                    for entry in before.accepted
                    if entry.context.membership_id == context.membership_id
                ),
                None,
            )
            if replay is not None:
                if replay.context != context:
                    raise KillSwitchAuthorityError("TRUSTED_CONTEXT_FAILURE")
                return replay
            accepted_at = self._transaction_time(now_utc, error="TRUSTED_CONTEXT_FAILURE")
            known, generations, current = self._reconstructed_history(before)
            genuinely_new = self._reconcile_context(
                context,
                known=known,
                generations=generations,
                error="TRUSTED_CONTEXT_FAILURE",
            )
            if before.accepted:
                previous = self._transaction_time(
                    before.accepted[-1].accepted_at_utc,
                    error="CORRUPT_AUTHORITY_STATE",
                )
                if accepted_at < previous:
                    raise KillSwitchAuthorityError("TRUSTED_CONTEXT_FAILURE")
            latest_by_scope = self._latest_transition_times(before)
            if any(
                scope in latest_by_scope and accepted_at <= latest_by_scope[scope]
                for scope in genuinely_new
            ):
                raise KillSwitchAuthorityError("TRUSTED_CONTEXT_FAILURE")
            revision = before.store_revision + 1
            entry = AcceptedKillSwitchAuthorityEntry(revision, now_utc, context)
            for scope in genuinely_new:
                current[scope] = context.membership_id
            replacement = AtomicKillSwitchAuthorityState(
                revision, before.accepted + (entry,), tuple(current.items())
            )
            self._carrier.compare_and_swap(before.store_revision, replacement)
            return entry

    @classmethod
    def _reconcile_context(
        cls,
        context: PrevalidatedKillSwitchContext,
        *,
        known: dict[tuple[str, str, str, int], KillSwitchRecord],
        generations: dict[tuple[str, str, str], int],
        error: str,
    ) -> set[tuple[str, str, str]]:
        """Separate exact snapshot overlap from genuinely-new transitions."""
        genuinely_new: set[tuple[str, str, str]] = set()
        for record in context.history:
            scope = cls._scope(record)
            key = (*scope, record.generation)
            prior = known.get(key)
            if prior is not None:
                if prior != record:
                    raise KillSwitchAuthorityError(error)
                continue
            if record.generation <= generations.get(scope, 0):
                raise KillSwitchAuthorityError(error)
            known[key] = record
            generations[scope] = record.generation
            genuinely_new.add(scope)
        return genuinely_new

    @classmethod
    def _reconstructed_history(
        cls, state: AtomicKillSwitchAuthorityState
    ) -> tuple[
        dict[tuple[str, str, str, int], KillSwitchRecord],
        dict[tuple[str, str, str], int],
        dict[tuple[str, str, str], str],
    ]:
        known: dict[tuple[str, str, str, int], KillSwitchRecord] = {}
        generations: dict[tuple[str, str, str], int] = {}
        current: dict[tuple[str, str, str], str] = {}
        for entry in state.accepted:
            new_scopes = cls._reconcile_context(
                entry.context,
                known=known,
                generations=generations,
                error="CORRUPT_AUTHORITY_STATE",
            )
            for scope in new_scopes:
                current[scope] = entry.context.membership_id
        return known, generations, current

    @classmethod
    def _latest_transition_times(
        cls, state: AtomicKillSwitchAuthorityState
    ) -> dict[tuple[str, str, str], datetime]:
        known: dict[tuple[str, str, str, int], KillSwitchRecord] = {}
        generations: dict[tuple[str, str, str], int] = {}
        result: dict[tuple[str, str, str], datetime] = {}
        for entry in state.accepted:
            new_scopes = cls._reconcile_context(
                entry.context,
                known=known,
                generations=generations,
                error="CORRUPT_AUTHORITY_STATE",
            )
            accepted_at = cls._transaction_time(
                entry.accepted_at_utc, error="CORRUPT_AUTHORITY_STATE"
            )
            for scope in new_scopes:
                if scope in result and accepted_at <= result[scope]:
                    raise KillSwitchAuthorityError("CORRUPT_AUTHORITY_STATE")
                result[scope] = accepted_at
        return result

    def resolve_historical(self, membership_id: str) -> AcceptedKillSwitchAuthorityEntry | None:
        with self._carrier.authority_fence():
            state = self._validated_read()
            return next(
                (item for item in state.accepted if item.context.membership_id == membership_id),
                None,
            )

    def resolve_historical_record(
        self, *, membership_id: str, record_fingerprint_sha256: str
    ) -> tuple[AcceptedKillSwitchAuthorityEntry, KillSwitchRecord] | None:
        """Resolve an exact accepted record under the durable carrier fence."""
        with self._carrier.authority_fence():
            state = self._validated_read()
            entry = next(
                (item for item in state.accepted if item.context.membership_id == membership_id),
                None,
            )
            if entry is None:
                return None
            matches = tuple(
                record
                for record in entry.context.history
                if record.record_fingerprint_sha256 == record_fingerprint_sha256
            )
            if len(matches) != 1:
                return None
            return entry, matches[0]

    def resolve_historical_current_record(
        self, *, membership_id: str, record_fingerprint_sha256: str, at_utc: str
    ) -> tuple[AcceptedKillSwitchAuthorityEntry, KillSwitchRecord] | None:
        """Resolve exact effective-current accepted membership at transaction time."""
        at = self._transaction_time(at_utc, error="TRUSTED_CONTEXT_FAILURE")
        with self._carrier.authority_fence():
            state = self._validated_read()
            historical = tuple(
                entry
                for entry in state.accepted
                if self._transaction_time(entry.accepted_at_utc, error="CORRUPT_AUTHORITY_STATE")
                <= at
            )
            prefix = AtomicKillSwitchAuthorityState(len(historical), historical, ())
            _known, _generations, current = self._reconstructed_history(prefix)
            entry = next(
                (item for item in historical if item.context.membership_id == membership_id),
                None,
            )
            if entry is None:
                return None
            matches = tuple(
                record
                for record in entry.context.history
                if record.record_fingerprint_sha256 == record_fingerprint_sha256
            )
            if len(matches) != 1:
                return None
            record = matches[0]
            if (
                current.get(self._scope(record)) != membership_id
                or self._transaction_time(record.effective_at_utc, error="CORRUPT_AUTHORITY_STATE")
                > at
            ):
                return None
            # A full-history context can contain an older record for a scope it
            # advances.  The reference must identify its exact latest record.
            current_record = max(
                (
                    item
                    for item in entry.context.history
                    if self._scope(item) == self._scope(record)
                ),
                key=lambda item: item.generation,
            )
            return (entry, record) if current_record == record else None

    def resolve_current(
        self, *, scope_type: str, scope_id: str, environment: str
    ) -> AcceptedKillSwitchAuthorityEntry | None:
        with self._carrier.authority_fence():
            return self._resolve_current(
                self._validated_read(), (scope_type, scope_id, environment)
            )

    def consume_current(
        self,
        *,
        scope_type: str,
        scope_id: str,
        environment: str,
        consumer: Callable[[AcceptedKillSwitchAuthorityEntry], T],
    ) -> T:
        with self._carrier.authority_fence():
            current = self._resolve_current(
                self._validated_read(), (scope_type, scope_id, environment)
            )
            if current is None:
                raise KillSwitchAuthorityError("TRUSTED_CONTEXT_FAILURE")
            return consumer(current)

    @staticmethod
    def _resolve_current(
        state: AtomicKillSwitchAuthorityState, scope: tuple[str, str, str]
    ) -> AcceptedKillSwitchAuthorityEntry | None:
        membership_id = dict(state.current).get(scope)
        return next(
            (item for item in state.accepted if item.context.membership_id == membership_id), None
        )

    def _validated_read(self) -> AtomicKillSwitchAuthorityState:
        state = self._carrier.read()
        self._validate_state(state)
        return state

    def _validate_state(self, state: object) -> None:
        try:
            if (
                not isinstance(state, AtomicKillSwitchAuthorityState)
                or type(state.store_revision) is not int
                or state.store_revision < 0
                or type(state.accepted) is not tuple
                or type(state.current) is not tuple
                or state.store_revision != len(state.accepted)
            ):
                raise KillSwitchAuthorityError("CORRUPT_AUTHORITY_STATE")
            memberships: set[str] = set()
            previous_time: datetime | None = None
            for revision, entry in enumerate(state.accepted, 1):
                if (
                    not isinstance(entry, AcceptedKillSwitchAuthorityEntry)
                    or entry.transaction_revision != revision
                ):
                    raise KillSwitchAuthorityError("CORRUPT_AUTHORITY_STATE")
                context = self._validate_context(entry.context)
                accepted_at = self._transaction_time(
                    entry.accepted_at_utc, error="CORRUPT_AUTHORITY_STATE"
                )
                if previous_time is not None and accepted_at < previous_time:
                    raise KillSwitchAuthorityError("CORRUPT_AUTHORITY_STATE")
                previous_time = accepted_at
                if context.membership_id in memberships:
                    raise KillSwitchAuthorityError("CORRUPT_AUTHORITY_STATE")
                memberships.add(context.membership_id)
            _known, _generations, expected = self._reconstructed_history(state)
            self._latest_transition_times(state)
            actual = dict(state.current)
            if len(actual) != len(state.current) or actual != expected:
                raise KillSwitchAuthorityError("CORRUPT_AUTHORITY_STATE")
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            if (
                isinstance(error, KillSwitchAuthorityError)
                and str(error) == "CORRUPT_AUTHORITY_STATE"
            ):
                raise
            raise KillSwitchAuthorityError("CORRUPT_AUTHORITY_STATE") from error
