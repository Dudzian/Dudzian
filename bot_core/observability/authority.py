"""Carrier-owned executable S9C observation membership and currentness authority."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
import math
import re
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, ContextManager, Iterator, Mapping, Protocol, TypeVar

SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
CODE = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")
UUID7 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ACCEPTANCE_ID = re.compile(r"s9c_[0-9a-f]{64}\Z")
RFC3339_UTC_SECONDS = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z\Z"
)
SECRET = re.compile(
    r"(?i)(password|api[_ -]?secret|api[_ -]?key|private[_ -]?key|bearer|token|pin|biometric)"
)
ENVELOPE = (
    "observation_id",
    "category",
    "source_component",
    "source_instance_id",
    "environment",
    "scope",
    "source_event_at_utc",
    "observed_at_utc",
    "ingested_at_utc",
    "expires_at_utc",
    "freshness_policy_id",
    "source_sequence",
    "condition",
    "reason_code",
    "value",
    "source_quality",
    "correlation_reference",
)
SCOPES = {
    "MARKET_DATA_FRESHNESS": (("market_data_route_id", "mdr"), ("instrument_id", "instr")),
    "EXECUTION_PATH_HEALTH": (
        ("exchange_account_id", "xacc"),
        ("instrument_id", "instr"),
        ("execution_route_id", "xroute"),
    ),
}
VALUES = {
    "MARKET_DATA_FRESHNESS": ("last_data_at_utc", "sequence_state"),
    "EXECUTION_PATH_HEALTH": ("path_state",),
}
CANONICAL_CORRELATIONS = {
    "RuntimeSession": "run",
    "ExchangeAccount": "xacc",
    "Instrument": "instr",
    "MarketDataRoute": "mdr",
    "ExecutionRoute": "xroute",
}
CONDITIONS = ("UNKNOWN", "OK", "DEGRADED", "BLOCKED")
GAP = {"OK": "DEGRADED", "DEGRADED": "DEGRADED", "UNKNOWN": "UNKNOWN", "BLOCKED": "BLOCKED"}
T = TypeVar("T")


def _utc(value: Any) -> datetime:
    if type(value) is not str or not RFC3339_UTC_SECONDS.fullmatch(value):
        raise ValueError("MALFORMED_TIMESTAMP")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("MALFORMED_TIMESTAMP") from error
    if result.microsecond:
        raise ValueError("MALFORMED_TIMESTAMP")
    return result


def _safe_scalar(value: Any) -> None:
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("INVALID_VALUE")
        return
    if not isinstance(value, str) or len(value) > 256 or any(ord(char) < 32 for char in value):
        raise ValueError("UNSAFE_VALUE")
    if SECRET.search(value):
        raise ValueError("SECRET_CONTENT")


def _canonical_id(value: Any, prefix: str, error: str) -> None:
    if (
        not isinstance(value, str)
        or not value.startswith(prefix + "_")
        or not UUID7.fullmatch(value[len(prefix) + 1 :])
    ):
        raise ValueError(error)


def _validate_source_sequence(sequence: Any) -> None:
    if sequence is None:
        return
    if type(sequence) is not int or sequence < 0:
        raise ValueError("INVALID_SOURCE_SEQUENCE")
    try:
        _json(sequence)
    except (OverflowError, ValueError) as error:
        raise ValueError("INVALID_SOURCE_SEQUENCE") from error


def _pairs(value: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(value.items()))


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True)
class FreshnessPolicy:
    policy_id: str
    version: int
    category: str
    source_class: str
    validity_horizon_seconds: int
    allowed_observed_future_skew_seconds: int
    allowed_source_event_future_skew_seconds: int
    allowed_ingest_before_observed_skew_seconds: int


@dataclass(frozen=True)
class FrozenEnvironmentRegistryBinding:
    """Pinned projection of the frozen M0.4 environment registry."""

    canonical_environments: frozenset[str] = frozenset({"PAPER", "TESTNET", "LIVE"})

    def __post_init__(self) -> None:
        if type(
            self.canonical_environments
        ) is not frozenset or self.canonical_environments != frozenset(
            {"PAPER", "TESTNET", "LIVE"}
        ):
            raise ValueError("INVALID_M04_ENVIRONMENT_BINDING")


@dataclass(frozen=True)
class ObservationKey:
    category: str
    source_component: str
    source_instance_id: str
    environment: str
    scope: tuple[tuple[str, Any], ...]

    @classmethod
    def exact(
        cls,
        *,
        category: str,
        source_component: str,
        source_instance_id: str,
        environment: str,
        scope: Mapping[str, Any],
    ) -> "ObservationKey":
        if type(scope) is not dict:
            raise ValueError("WRONG_SCOPE")
        return cls(category, source_component, source_instance_id, environment, _pairs(scope))


@dataclass(frozen=True)
class ObservationSemanticKey:
    """Condition identity shared across source runtime-session restarts."""

    category: str
    source_component: str
    environment: str
    scope: tuple[tuple[str, Any], ...]

    @classmethod
    def from_observation(cls, observation: "CanonicalObservation") -> "ObservationSemanticKey":
        return cls(
            observation.category,
            observation.source_component,
            observation.environment,
            observation.scope,
        )


@dataclass(frozen=True)
class CanonicalObservation:
    observation_id: str
    category: str
    source_component: str
    source_instance_id: str
    environment: str
    scope: tuple[tuple[str, Any], ...]
    source_event_at_utc: str | None
    observed_at_utc: str
    ingested_at_utc: str
    expires_at_utc: str
    freshness_policy_id: str
    source_sequence: int | None
    condition: str
    reason_code: str
    value: tuple[tuple[str, Any], ...]
    source_quality: str
    correlation_reference: tuple[tuple[str, Any], ...] | None

    def mapping(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in ENVELOPE}
        result["scope"] = dict(self.scope)
        result["value"] = dict(self.value)
        result["correlation_reference"] = (
            None if self.correlation_reference is None else dict(self.correlation_reference)
        )
        return result


@dataclass(frozen=True)
class AcceptedObservation:
    acceptance_id: str
    transaction_revision: int
    accepted_at_utc: str
    observation: CanonicalObservation
    key: ObservationKey
    effective_condition: str
    effective_reason_codes: tuple[str, ...]
    sequence_gap: bool
    content_fingerprint: str
    freshness_policy: FreshnessPolicy
    freshness_policy_fingerprint_sha256: str


@dataclass(frozen=True)
class EffectiveCurrentObservation:
    status: str
    accepted: AcceptedObservation | None
    projected_condition: str
    projected_reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class AtomicObservationAuthorityState:
    store_revision: int = 0
    accepted: tuple[AcceptedObservation, ...] = ()
    current: tuple[tuple[ObservationKey, str], ...] = ()
    replay: tuple[tuple[tuple[str, str, int], str], ...] = ()
    last_sequence: tuple[tuple[tuple[str, str], int], ...] = ()


class ObservationAuthorityCarrier(Protocol):
    """Carrier contract.

    ``read`` and ``compare_and_swap`` are required to be reentrant-safe while the
    caller holds ``authority_fence``; every authority sharing a carrier must use
    that same serialization domain.
    """

    def authority_fence(self) -> ContextManager[None]: ...
    def read(self) -> AtomicObservationAuthorityState: ...
    def compare_and_swap(
        self, expected_revision: int, state: AtomicObservationAuthorityState
    ) -> None: ...


class InMemoryObservationAuthorityCarrier:
    """Atomic reference carrier whose lock is the carrier-wide authority fence."""

    def __init__(self, state: AtomicObservationAuthorityState | None = None) -> None:
        self._lock = RLock()
        self._state = state or AtomicObservationAuthorityState()
        self.fail_next = False

    @contextmanager
    def authority_fence(self) -> Iterator[None]:
        with self._lock:
            yield

    def read(self) -> AtomicObservationAuthorityState:
        with self._lock:
            return self._state

    def compare_and_swap(
        self, expected_revision: int, state: AtomicObservationAuthorityState
    ) -> None:
        with self._lock:
            if self.fail_next:
                self.fail_next = False
                raise OSError("INJECTED_CARRIER_FAILURE")
            if self._state.store_revision != expected_revision:
                raise RuntimeError("AUTHORITY_CAS_CONFLICT")
            self._state = state


class _ObservationPublisher:
    """Non-exported writer capability minted only by the owner composition boundary."""

    def __init__(self, authority: "ObservationAuthority") -> None:
        self.__authority = authority

    def publish(self, observation: Mapping[str, Any], *, now_utc: str) -> AcceptedObservation:
        return self.__authority._publish(observation, now_utc)  # noqa: SLF001


class ObservationAuthority:
    EXECUTABLE_CATEGORIES = frozenset(SCOPES)

    @classmethod
    def compose(
        cls,
        carrier: ObservationAuthorityCarrier,
        *,
        policies: tuple[FreshnessPolicy, ...],
        environment_binding: FrozenEnvironmentRegistryBinding,
        enabled_environments: frozenset[str] | None = None,
    ) -> tuple["ObservationAuthority", _ObservationPublisher]:
        authority = cls(
            carrier,
            policies=policies,
            environment_binding=environment_binding,
            enabled_environments=enabled_environments,
        )
        return authority, _ObservationPublisher(authority)

    def __init__(
        self,
        carrier: ObservationAuthorityCarrier,
        *,
        policies: tuple[FreshnessPolicy, ...],
        environment_binding: FrozenEnvironmentRegistryBinding,
        enabled_environments: frozenset[str] | None = None,
    ) -> None:
        self._carrier = carrier
        if not isinstance(environment_binding, FrozenEnvironmentRegistryBinding):
            raise ValueError("INVALID_M04_ENVIRONMENT_BINDING")
        self._canonical_environments = frozenset(environment_binding.canonical_environments)
        self._enabled_environments = (
            self._canonical_environments
            if enabled_environments is None
            else frozenset(enabled_environments)
        )
        if not self._enabled_environments <= self._canonical_environments:
            raise ValueError("ILLEGAL_ENVIRONMENT")
        self._policies = MappingProxyType(self._validate_policies(policies))
        with carrier.authority_fence():
            self._pin(carrier.read())

    @staticmethod
    def _validate_freshness_policy(policy: FreshnessPolicy) -> None:
        if not isinstance(policy, FreshnessPolicy):
            raise ValueError("INVALID_FRESHNESS_POLICY")
        integer_fields = (
            policy.version,
            policy.validity_horizon_seconds,
            policy.allowed_observed_future_skew_seconds,
            policy.allowed_source_event_future_skew_seconds,
            policy.allowed_ingest_before_observed_skew_seconds,
        )
        if (
            not isinstance(policy.policy_id, str)
            or not CODE.fullmatch(policy.policy_id)
            or any(type(value) is not int for value in integer_fields)
            or policy.version <= 0
            or policy.validity_horizon_seconds <= 0
            or any(value < 0 for value in integer_fields[2:])
            or policy.category not in SCOPES
            or policy.source_class != "core_host"
        ):
            raise ValueError("INVALID_FRESHNESS_POLICY")
        try:
            for duration in integer_fields[1:]:
                timedelta(seconds=duration)
            _json(asdict(policy))
        except (OverflowError, ValueError) as error:
            raise ValueError("INVALID_FRESHNESS_POLICY") from error

    @classmethod
    def _validate_policies(
        cls, policies: tuple[FreshnessPolicy, ...]
    ) -> dict[str, FreshnessPolicy]:
        if type(policies) is not tuple:
            raise ValueError("INVALID_FRESHNESS_POLICY")
        result: dict[str, FreshnessPolicy] = {}
        for policy in policies:
            cls._validate_freshness_policy(policy)
            if policy.policy_id in result:
                raise ValueError("INVALID_FRESHNESS_POLICY")
            result[policy.policy_id] = policy
        return result

    def _canonical_payload_before_replay(self, raw: Mapping[str, Any]) -> CanonicalObservation:
        if type(raw) is not dict or set(raw) != set(ENVELOPE):
            raise ValueError("CLOSED_SCHEMA")
        category = raw["category"]
        if type(category) is not str or category not in SCOPES:
            raise ValueError("UNKNOWN_CATEGORY")
        if raw["source_component"] != "core_host":
            raise ValueError("WRONG_SOURCE")
        _canonical_id(raw["source_instance_id"], "run", "WRONG_SOURCE_INSTANCE")
        if (
            type(raw["environment"]) is not str
            or raw["environment"] not in self._canonical_environments
        ):
            raise ValueError("ILLEGAL_ENVIRONMENT")
        scope = raw["scope"]
        if type(scope) is not dict or set(scope) != {field for field, _ in SCOPES[category]}:
            raise ValueError("WRONG_SCOPE")
        for field, prefix in SCOPES[category]:
            _canonical_id(scope[field], prefix, "WRONG_SCOPE")
            _safe_scalar(scope[field])
        value = raw["value"]
        if type(value) is not dict or set(value) != set(VALUES[category]):
            raise ValueError("INVALID_VALUE")
        for scalar in value.values():
            _safe_scalar(scalar)
        try:
            serialized_value = _json(value)
        except (OverflowError, ValueError) as error:
            raise ValueError("INVALID_VALUE") from error
        if len(serialized_value) > 2048:
            raise ValueError("INVALID_VALUE")
        if not isinstance(raw["observation_id"], str) or not SAFE.fullmatch(raw["observation_id"]):
            raise ValueError("INVALID_OBSERVATION_ID")
        _safe_scalar(raw["observation_id"])
        if raw["condition"] not in CONDITIONS:
            raise ValueError("UNKNOWN_CONDITION")
        if not isinstance(raw["reason_code"], str) or not CODE.fullmatch(raw["reason_code"]):
            raise ValueError("INVALID_REASON_CODE")
        _safe_scalar(raw["reason_code"])
        if raw["source_quality"] not in ("DIRECT", "DERIVED", "CACHED"):
            raise ValueError("UNKNOWN_SOURCE_QUALITY")
        correlation = raw["correlation_reference"]
        if correlation is not None:
            if type(correlation) is not dict or correlation.get("kind") not in (
                "LOCAL",
                "CANONICAL",
            ):
                raise ValueError("INVALID_CORRELATION")
            expected = (
                {"kind", "value"} if correlation["kind"] == "LOCAL" else {"kind", "entity", "value"}
            )
            if set(correlation) != expected:
                raise ValueError("INVALID_CORRELATION")
            if correlation["kind"] == "LOCAL":
                if not isinstance(correlation["value"], str) or not SAFE.fullmatch(
                    correlation["value"]
                ):
                    raise ValueError("INVALID_CORRELATION")
                _safe_scalar(correlation["value"])
            else:
                entity = correlation["entity"]
                if type(entity) is not str:
                    raise ValueError("INVALID_CORRELATION")
                prefix = CANONICAL_CORRELATIONS.get(entity)
                if prefix is None:
                    raise ValueError("INVALID_CORRELATION")
                _canonical_id(correlation["value"], prefix, "INVALID_CORRELATION")
        sequence = raw["source_sequence"]
        _validate_source_sequence(sequence)
        if type(raw["freshness_policy_id"]) is not str:
            raise ValueError("UNKNOWN_FRESHNESS_POLICY")
        return CanonicalObservation(
            *(
                raw[name]
                if name not in ("scope", "value", "correlation_reference")
                else (_pairs(raw[name]) if raw[name] is not None else None)
                for name in ENVELOPE
            )
        )

    def _validate_unseen_temporal_and_freshness(
        self,
        observation: CanonicalObservation,
        now: str,
        *,
        historical_policy: FreshnessPolicy | None = None,
        enforce_enabled: bool = True,
    ) -> None:
        if enforce_enabled and observation.environment not in self._enabled_environments:
            raise ValueError("ILLEGAL_ENVIRONMENT")
        observed, ingested, expires, logical_now = map(
            _utc,
            (
                observation.observed_at_utc,
                observation.ingested_at_utc,
                observation.expires_at_utc,
                now,
            ),
        )
        source_event = (
            _utc(observation.source_event_at_utc)
            if observation.source_event_at_utc is not None
            else None
        )
        policy = historical_policy or self._policies.get(observation.freshness_policy_id)
        if (
            policy is None
            or policy.category != observation.category
            or policy.source_class != "core_host"
        ):
            raise ValueError("UNKNOWN_FRESHNESS_POLICY")
        if expires <= observed or expires - observed != timedelta(
            seconds=policy.validity_horizon_seconds
        ):
            raise ValueError("INVALID_EXPIRY")
        if observed - logical_now > timedelta(seconds=policy.allowed_observed_future_skew_seconds):
            raise ValueError("FUTURE_TIMESTAMP")
        if source_event and source_event - logical_now > timedelta(
            seconds=policy.allowed_source_event_future_skew_seconds
        ):
            raise ValueError("FUTURE_SOURCE_EVENT")
        if observed - ingested > timedelta(
            seconds=policy.allowed_ingest_before_observed_skew_seconds
        ):
            raise ValueError("INGEST_SKEW")

    def _canonical(
        self,
        raw: Mapping[str, Any],
        now: str,
        *,
        historical_policy: FreshnessPolicy | None = None,
        enforce_enabled: bool = True,
    ) -> CanonicalObservation:
        observation = self._canonical_payload_before_replay(raw)
        self._validate_unseen_temporal_and_freshness(
            observation,
            now,
            historical_policy=historical_policy,
            enforce_enabled=enforce_enabled,
        )
        return observation

    @staticmethod
    def _fingerprint(observation: CanonicalObservation) -> str:
        return sha256(_json(observation.mapping()).encode()).hexdigest()

    @staticmethod
    def _policy_fingerprint(policy: FreshnessPolicy) -> str:
        return sha256(_json(asdict(policy)).encode()).hexdigest()

    @staticmethod
    def _acceptance_id(
        revision: int, accepted_at_utc: str, fingerprint: str, policy_fingerprint: str
    ) -> str:
        return (
            "s9c_"
            + sha256(
                _json((revision, accepted_at_utc, fingerprint, policy_fingerprint)).encode()
            ).hexdigest()
        )

    @staticmethod
    def _key(observation: CanonicalObservation) -> ObservationKey:
        return ObservationKey(
            observation.category,
            observation.source_component,
            observation.source_instance_id,
            observation.environment,
            observation.scope,
        )

    @staticmethod
    def _validate_transition(
        observation: CanonicalObservation,
        fingerprint: str,
        history: Mapping[str, AcceptedObservation],
        current: Mapping[ObservationKey, str],
        replay: Mapping[tuple[str, str, int], str],
        last: Mapping[tuple[str, str], int],
    ) -> tuple[AcceptedObservation | None, bool]:
        """Apply the canonical live ordering rules to one candidate transition."""
        source = (observation.source_component, observation.source_instance_id)
        if observation.source_sequence is not None:
            replay_id = replay.get((*source, observation.source_sequence))
            if replay_id is not None:
                prior_replay = history[replay_id]
                if prior_replay.content_fingerprint != fingerprint:
                    raise ValueError("DUPLICATE_SEQUENCE_CONFLICT")
                return prior_replay, False
        key = ObservationAuthority._key(observation)
        prior = history.get(current.get(key, ""))
        if prior and _utc(observation.observed_at_utc) < _utc(prior.observation.observed_at_utc):
            raise ValueError("CLOCK_REGRESSION")
        if prior and observation.source_sequence is None:
            if observation.observed_at_utc == prior.observation.observed_at_utc:
                raise ValueError("SEQUENCE_REGRESSION")
            if (
                observation.source_event_at_utc
                and prior.observation.source_event_at_utc
                and _utc(observation.source_event_at_utc)
                < _utc(prior.observation.source_event_at_utc)
            ):
                raise ValueError("SOURCE_EVENT_REGRESSION")
        gap = False
        if observation.source_sequence is not None:
            previous = last.get(source)
            if previous is not None and observation.source_sequence < previous:
                raise ValueError("SEQUENCE_REGRESSION")
            gap = previous is not None and observation.source_sequence > previous + 1
        return None, gap

    def _publish(self, raw: Mapping[str, Any], now: str) -> AcceptedObservation:
        with self._carrier.authority_fence():
            state = self._load_locked()
            payload = self._canonical_payload_before_replay(raw)
            # Replay is already-authorized historical membership.  Resolve it before
            # today's policy registry can reinterpret the original transaction.
            if payload.source_sequence is not None:
                replay_key = (
                    payload.source_component,
                    payload.source_instance_id,
                    payload.source_sequence,
                )
                replay_id = dict(state.replay).get(replay_key)
                if replay_id is not None:
                    prior = next(
                        record for record in state.accepted if record.acceptance_id == replay_id
                    )
                    if self._fingerprint(payload) != prior.content_fingerprint:
                        raise ValueError("DUPLICATE_SEQUENCE_CONFLICT")
                    return prior
            self._validate_unseen_temporal_and_freshness(payload, now)
            if state.accepted and _utc(now) < _utc(state.accepted[-1].accepted_at_utc):
                raise ValueError("TRANSACTION_TIME_ROLLBACK")
            semantic_key = ObservationSemanticKey.from_observation(payload)
            semantic_prior = next(
                (
                    item
                    for item in reversed(state.accepted)
                    if ObservationSemanticKey.from_observation(item.observation) == semantic_key
                ),
                None,
            )
            if semantic_prior is not None:
                prior_time = _utc(semantic_prior.accepted_at_utc)
                transaction_time = _utc(now)
                if transaction_time < prior_time:
                    raise ValueError("TRANSACTION_TIME_ROLLBACK")
                if transaction_time == prior_time:
                    raise ValueError("SEMANTIC_TRANSACTION_TIME_COLLISION")
            observation = payload
            fingerprint = self._fingerprint(observation)
            policy = self._policies[observation.freshness_policy_id]
            policy_fingerprint = self._policy_fingerprint(policy)
            history = {record.acceptance_id: record for record in state.accepted}
            replay = dict(state.replay)
            last = dict(state.last_sequence)
            current = dict(state.current)
            source = (observation.source_component, observation.source_instance_id)
            key = self._key(observation)
            replayed, gap = self._validate_transition(
                observation, fingerprint, history, current, replay, last
            )
            if replayed is not None:
                return replayed
            condition = GAP[observation.condition] if gap else observation.condition
            reasons = (
                (observation.reason_code, "SEQUENCE_GAP") if gap else (observation.reason_code,)
            )
            revision = state.store_revision + 1
            record = AcceptedObservation(
                self._acceptance_id(revision, now, fingerprint, policy_fingerprint),
                revision,
                now,
                observation,
                key,
                condition,
                reasons,
                gap,
                fingerprint,
                policy,
                policy_fingerprint,
            )
            if observation.source_sequence is not None:
                replay[(*source, observation.source_sequence)] = record.acceptance_id
                last[source] = observation.source_sequence
            current[key] = record.acceptance_id
            replacement = AtomicObservationAuthorityState(
                revision,
                state.accepted + (record,),
                tuple(current.items()),
                tuple(replay.items()),
                tuple(last.items()),
            )
            self._carrier.compare_and_swap(state.store_revision, replacement)
            self._state = replacement
            return record

    def resolve_historical_acceptance(self, acceptance_id: str) -> AcceptedObservation | None:
        with self._carrier.authority_fence():
            state = self._load_locked()
            return next(
                (item for item in state.accepted if item.acceptance_id == acceptance_id), None
            )

    def validate_historical_effective_acceptance(
        self, acceptance_id: str, *, at_utc: str
    ) -> AcceptedObservation:
        """Prove accepted membership was effective-current at an historical instant.

        This deliberately uses the accepted record's pinned policy and expiry, not
        today's policy registry or enabled-environment projection.  Revision breaks
        ties when multiple acceptances have the same whole-second acceptance time.
        """
        instant = _utc(at_utc)
        with self._carrier.authority_fence():
            state = self._load_locked()
            requested = next(
                (item for item in state.accepted if item.acceptance_id == acceptance_id),
                None,
            )
            if requested is None:
                raise ValueError("UNKNOWN_ACCEPTANCE")
            eligible = tuple(
                item
                for item in state.accepted
                if item.key == requested.key and _utc(item.accepted_at_utc) <= instant
            )
            if (
                not eligible
                or max(eligible, key=lambda item: item.transaction_revision) != requested
                or instant >= _utc(requested.observation.expires_at_utc)
            ):
                raise ValueError("ACCEPTANCE_NOT_HISTORICALLY_EFFECTIVE")
            return requested

    def validate_historical_semantic_effective_acceptance(
        self, acceptance_id: str, *, at_utc: str
    ) -> AcceptedObservation:
        """Prove S9D semantic currentness across runtime-session identities."""
        instant = _utc(at_utc)
        with self._carrier.authority_fence():
            state = self._load_locked()
            requested = next(
                (item for item in state.accepted if item.acceptance_id == acceptance_id),
                None,
            )
            if requested is None:
                raise ValueError("UNKNOWN_ACCEPTANCE")
            semantic_key = ObservationSemanticKey.from_observation(requested.observation)
            eligible = tuple(
                item
                for item in state.accepted
                if ObservationSemanticKey.from_observation(item.observation) == semantic_key
                and _utc(item.accepted_at_utc) <= instant
            )
            if (
                not eligible
                or max(eligible, key=lambda item: item.transaction_revision) != requested
                or instant >= _utc(requested.observation.expires_at_utc)
            ):
                raise ValueError("ACCEPTANCE_NOT_HISTORICALLY_EFFECTIVE")
            return requested

    def resolve_current(
        self, selector: ObservationKey, *, now_utc: str
    ) -> EffectiveCurrentObservation:
        with self._carrier.authority_fence():
            return self._resolve_current(self._load_locked(), selector, now_utc)

    @staticmethod
    def _resolve_current(
        state: AtomicObservationAuthorityState, selector: ObservationKey, now: str
    ) -> EffectiveCurrentObservation:
        acceptance_id = dict(state.current).get(selector)
        record = next(
            (item for item in state.accepted if item.acceptance_id == acceptance_id), None
        )
        if record is None:
            return EffectiveCurrentObservation("ABSENT", None, "UNKNOWN", ("MISSING_OBSERVATION",))
        if _utc(now) >= _utc(record.observation.expires_at_utc):
            return EffectiveCurrentObservation("STALE", record, "UNKNOWN", ("OBSERVATION_EXPIRED",))
        return EffectiveCurrentObservation(
            "FRESH", record, record.effective_condition, record.effective_reason_codes
        )

    def consume_effective_current(
        self,
        selector: ObservationKey,
        *,
        now_utc: str,
        consumer: Callable[[AcceptedObservation], T],
    ) -> T:
        with self._carrier.authority_fence():
            result = self._resolve_current(self._load_locked(), selector, now_utc)
            if result.status != "FRESH" or result.accepted is None:
                raise ValueError("OBSERVATION_NOT_EFFECTIVE_CURRENT")
            return consumer(result.accepted)

    def consume_semantic_effective_current(
        self,
        selector: ObservationSemanticKey,
        *,
        now_utc: str,
        consumer: Callable[[AcceptedObservation], T],
    ) -> T:
        """Hold the carrier fence through consumption of the latest semantic record.

        Exact-key currentness remains independently available.  Here the newest
        transaction across runtime sessions wins, and an expired winner prevents
        fallback to an older runtime's record.
        """
        now = _utc(now_utc)
        if not isinstance(selector, ObservationSemanticKey):
            raise ValueError("INVALID_SEMANTIC_KEY")
        with self._carrier.authority_fence():
            state = self._load_locked()
            matches = tuple(
                item
                for item in state.accepted
                if ObservationSemanticKey.from_observation(item.observation) == selector
            )
            if not matches:
                raise ValueError("OBSERVATION_NOT_EFFECTIVE_CURRENT")
            winner = max(matches, key=lambda item: item.transaction_revision)
            if now >= _utc(winner.observation.expires_at_utc):
                raise ValueError("OBSERVATION_NOT_EFFECTIVE_CURRENT")
            return consumer(winner)

    def _load_locked(self) -> AtomicObservationAuthorityState:
        return self._pin(self._carrier.read())

    def _pin(self, candidate: AtomicObservationAuthorityState) -> AtomicObservationAuthorityState:
        try:
            self._validate_state(candidate)
            state = AtomicObservationAuthorityState(
                candidate.store_revision,
                tuple(candidate.accepted),
                tuple(candidate.current),
                tuple(candidate.replay),
                tuple(candidate.last_sequence),
            )
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("CORRUPT_AUTHORITY_STATE") from error
        self._state = state
        return state

    def _validate_state(self, state: AtomicObservationAuthorityState) -> None:
        if (
            not isinstance(state, AtomicObservationAuthorityState)
            or type(state.store_revision) is not int
            or state.store_revision < 0
            or any(
                type(value) is not tuple
                for value in (state.accepted, state.current, state.replay, state.last_sequence)
            )
            or state.store_revision != len(state.accepted)
        ):
            raise ValueError("CORRUPT_AUTHORITY_STATE")
        by_id: dict[str, AcceptedObservation] = {}
        expected_current: dict[ObservationKey, str] = {}
        expected_replay: dict[tuple[str, str, int], str] = {}
        expected_last: dict[tuple[str, str], int] = {}
        previous_accepted_at: datetime | None = None
        semantic_accepted_at: dict[ObservationSemanticKey, datetime] = {}
        for revision, record in enumerate(state.accepted, 1):
            if (
                not isinstance(record, AcceptedObservation)
                or type(record.effective_reason_codes) is not tuple
                or not isinstance(record.observation, CanonicalObservation)
                or type(record.observation.scope) is not tuple
                or type(record.observation.value) is not tuple
                or (
                    record.observation.correlation_reference is not None
                    and type(record.observation.correlation_reference) is not tuple
                )
            ):
                raise ValueError("CORRUPT_AUTHORITY_STATE")
            if (
                type(record.transaction_revision) is not int
                or record.transaction_revision < 1
                or type(record.sequence_gap) is not bool
                or type(record.acceptance_id) is not str
                or not ACCEPTANCE_ID.fullmatch(record.acceptance_id)
                or type(record.accepted_at_utc) is not str
                or type(record.effective_condition) is not str
                or any(type(reason) is not str for reason in record.effective_reason_codes)
                or type(record.content_fingerprint) is not str
                or not SHA256.fullmatch(record.content_fingerprint)
                or type(record.freshness_policy_fingerprint_sha256) is not str
                or not SHA256.fullmatch(record.freshness_policy_fingerprint_sha256)
            ):
                raise ValueError("CORRUPT_AUTHORITY_STATE")
            self._validate_freshness_policy(record.freshness_policy)
            if record.freshness_policy.policy_id != record.observation.freshness_policy_id:
                raise ValueError("CORRUPT_AUTHORITY_STATE")
            policy_fingerprint = self._policy_fingerprint(record.freshness_policy)
            accepted_at = _utc(record.accepted_at_utc)
            if previous_accepted_at is not None and accepted_at < previous_accepted_at:
                raise ValueError("CORRUPT_AUTHORITY_STATE")
            previous_accepted_at = accepted_at
            semantic_key = ObservationSemanticKey.from_observation(record.observation)
            if (
                semantic_key in semantic_accepted_at
                and accepted_at <= semantic_accepted_at[semantic_key]
            ):
                raise ValueError("CORRUPT_AUTHORITY_STATE")
            semantic_accepted_at[semantic_key] = accepted_at
            canonical = self._canonical(
                record.observation.mapping(),
                record.accepted_at_utc,
                historical_policy=record.freshness_policy,
                enforce_enabled=False,
            )
            fingerprint = self._fingerprint(canonical)
            expected_id = self._acceptance_id(
                revision, record.accepted_at_utc, fingerprint, policy_fingerprint
            )
            key = self._key(canonical)
            if (
                record.transaction_revision != revision
                or record.observation != canonical
                or record.content_fingerprint != fingerprint
                or record.freshness_policy_fingerprint_sha256 != policy_fingerprint
                or record.acceptance_id != expected_id
                or record.acceptance_id in by_id
                or record.key != key
            ):
                raise ValueError("CORRUPT_AUTHORITY_STATE")
            replayed, gap = self._validate_transition(
                canonical,
                fingerprint,
                by_id,
                expected_current,
                expected_replay,
                expected_last,
            )
            if replayed is not None:
                raise ValueError("CORRUPT_AUTHORITY_STATE")
            expected_condition = GAP[canonical.condition] if gap else canonical.condition
            expected_reasons = (
                (canonical.reason_code, "SEQUENCE_GAP") if gap else (canonical.reason_code,)
            )
            if (record.sequence_gap, record.effective_condition, record.effective_reason_codes) != (
                gap,
                expected_condition,
                expected_reasons,
            ):
                raise ValueError("CORRUPT_AUTHORITY_STATE")
            by_id[expected_id] = record
            expected_current[key] = expected_id
            if canonical.source_sequence is not None:
                source = (canonical.source_component, canonical.source_instance_id)
                expected_replay[(*source, canonical.source_sequence)] = expected_id
                expected_last[source] = canonical.source_sequence
        for entry in state.current:
            if (
                type(entry) is not tuple
                or len(entry) != 2
                or not isinstance(entry[0], ObservationKey)
                or any(
                    type(value) is not str
                    for value in (
                        entry[0].category,
                        entry[0].source_component,
                        entry[0].source_instance_id,
                        entry[0].environment,
                    )
                )
                or type(entry[0].scope) is not tuple
                or any(
                    type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str
                    for pair in entry[0].scope
                )
                or type(entry[1]) is not str
                or not ACCEPTANCE_ID.fullmatch(entry[1])
            ):
                raise ValueError("CORRUPT_AUTHORITY_STATE")
        for entry in state.replay:
            if (
                type(entry) is not tuple
                or len(entry) != 2
                or type(entry[0]) is not tuple
                or len(entry[0]) != 3
                or type(entry[0][0]) is not str
                or type(entry[0][1]) is not str
                or type(entry[0][2]) is not int
                or entry[0][2] < 0
                or type(entry[1]) is not str
                or not ACCEPTANCE_ID.fullmatch(entry[1])
            ):
                raise ValueError("CORRUPT_AUTHORITY_STATE")
        for entry in state.last_sequence:
            if (
                type(entry) is not tuple
                or len(entry) != 2
                or type(entry[0]) is not tuple
                or len(entry[0]) != 2
                or any(type(value) is not str for value in entry[0])
                or type(entry[1]) is not int
                or entry[1] < 0
            ):
                raise ValueError("CORRUPT_AUTHORITY_STATE")
        actual_current = dict(state.current)
        actual_replay = dict(state.replay)
        actual_last = dict(state.last_sequence)
        if (
            len(actual_current) != len(state.current)
            or len(actual_replay) != len(state.replay)
            or len(actual_last) != len(state.last_sequence)
            or actual_current != expected_current
            or actual_replay != expected_replay
            or actual_last != expected_last
        ):
            raise ValueError("CORRUPT_AUTHORITY_STATE")
