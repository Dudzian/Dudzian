"""Carrier-owned membership for exact M0.8 observed balance facts.

An :class:`ObservedBalanceFact` is only frozen data.  Publication through the
owner capability returned by ``compose`` is what creates accepted membership.
This module deliberately does not derive reconciliation outcomes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import re
from threading import RLock
from types import MappingProxyType
from typing import Callable, ContextManager, Iterator, Mapping, Protocol, TypeVar

from bot_core.persistence.fingerprints import canonical_json_sha256

ENVIRONMENTS = frozenset({"PAPER", "TESTNET", "LIVE"})
MAPPING_STATUSES = frozenset({"EXACT", "EXPLICIT_ALIAS", "UNKNOWN", "AMBIGUOUS"})
OBSERVATION_SEMANTICS = frozenset({"BALANCE", "UNSUPPORTED"})
DECIMAL = re.compile(r"(0|[1-9][0-9]*)(\.[0-9]*[1-9])?\Z")
ID = re.compile(
    r"[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d*[1-9])?Z\Z")
FACT_FIELDS = frozenset(
    {
        "workspace_id",
        "portfolio_id",
        "environment",
        "exchange_account_id",
        "asset_reference",
        "observed_quantity",
        "as_of_utc",
        "source_id",
        "source_fingerprint_sha256",
    }
)
T = TypeVar("T")


class ObservedBalanceAuthorityError(ValueError):
    """Fail-closed observed-balance authority error."""


def _identity(value: object, prefix: str) -> bool:
    return type(value) is str and value.startswith(prefix + "_") and ID.fullmatch(value) is not None


def _timestamp(value: object) -> None:
    if type(value) is not str or TIME.fullmatch(value) is None:
        raise ObservedBalanceAuthorityError("TRUSTED_CONTEXT_FAILURE")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ObservedBalanceAuthorityError("TRUSTED_CONTEXT_FAILURE") from exc
    if parsed.tzinfo != timezone.utc:
        raise ObservedBalanceAuthorityError("TRUSTED_CONTEXT_FAILURE")


@dataclass(frozen=True, slots=True)
class RawAssetReference:
    venue_asset_code: str
    canonical_display_code: str
    asset_namespace: str
    mapping_status: str

    @classmethod
    def from_mapping(cls, value: object) -> RawAssetReference:
        fields = {"venue_asset_code", "canonical_display_code", "asset_namespace", "mapping_status"}
        if type(value) is not dict or set(value) != fields:
            raise ObservedBalanceAuthorityError("TRUSTED_CONTEXT_FAILURE")
        result = cls(**value)  # type: ignore[arg-type]
        if (
            any(type(item) is not str or not item for item in asdict(result).values())
            or result.mapping_status not in MAPPING_STATUSES
        ):
            raise ObservedBalanceAuthorityError("TRUSTED_CONTEXT_FAILURE")
        return result

    @property
    def mapping_authoritative(self) -> bool:
        return self.mapping_status in {"EXACT", "EXPLICIT_ALIAS"}


@dataclass(frozen=True, slots=True)
class ObservedBalanceFact:
    workspace_id: str
    portfolio_id: str
    environment: str
    exchange_account_id: str
    asset_reference: RawAssetReference
    observed_quantity: str
    as_of_utc: str
    source_id: str
    source_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class AcceptedObservedBalanceFact:
    source_id: str
    source_fingerprint_sha256: str
    workspace_id: str
    portfolio_id: str
    environment: str
    exchange_account_id: str
    asset_reference: RawAssetReference
    observed_quantity: str
    as_of_utc: str
    semantics: str
    membership_fingerprint_sha256: str
    canonical_observed_fact_json: str

    @property
    def mapping_authoritative(self) -> bool:
        return self.asset_reference.mapping_authoritative

    def content(self) -> Mapping[str, object]:
        return MappingProxyType(json.loads(self.canonical_observed_fact_json))


@dataclass(frozen=True, slots=True)
class AtomicObservedBalanceState:
    store_revision: int = 0
    accepted: tuple[AcceptedObservedBalanceFact, ...] = ()


class ObservedBalanceCarrier(Protocol):
    def authority_fence(self) -> ContextManager[None]: ...
    def read(self) -> AtomicObservedBalanceState: ...
    def compare_and_swap(
        self, expected_revision: int, state: AtomicObservedBalanceState
    ) -> None: ...


class InMemoryObservedBalanceCarrier:
    def __init__(self, state: AtomicObservedBalanceState | None = None) -> None:
        self._lock = RLock()
        self._state = state or AtomicObservedBalanceState()
        self.fail_next = False

    @contextmanager
    def authority_fence(self) -> Iterator[None]:
        with self._lock:
            yield

    def read(self) -> AtomicObservedBalanceState:
        with self._lock:
            return self._state

    def compare_and_swap(self, expected_revision: int, state: AtomicObservedBalanceState) -> None:
        with self._lock:
            if self.fail_next:
                self.fail_next = False
                raise OSError("INJECTED_CARRIER_FAILURE")
            if self._state.store_revision != expected_revision:
                raise RuntimeError("AUTHORITY_CAS_CONFLICT")
            self._state = state


class _ObservedBalancePublisher:
    def __init__(self, authority: CoreAcceptedObservedBalanceFactProjection) -> None:
        self.__authority = authority

    def publish(
        self, payload: Mapping[str, object], *, semantics: str
    ) -> AcceptedObservedBalanceFact:
        return self.__authority._publish(payload, semantics=semantics)  # noqa: SLF001


class CoreAcceptedObservedBalanceFactProjection:
    """Read-side view of immutable carrier-owned observation membership."""

    @classmethod
    def compose(
        cls, carrier: ObservedBalanceCarrier
    ) -> tuple[CoreAcceptedObservedBalanceFactProjection, _ObservedBalancePublisher]:
        authority = cls(carrier)
        return authority, _ObservedBalancePublisher(authority)

    def __init__(self, carrier: ObservedBalanceCarrier) -> None:
        self._carrier = carrier
        with carrier.authority_fence():
            self._validate_state(carrier.read())

    @staticmethod
    def _membership_fingerprint(source_id: str, source_fingerprint: str, semantics: str) -> str:
        """Bind accepted semantics separately from the frozen observed-fact hash."""
        return canonical_json_sha256(
            {
                "source_id": source_id,
                "source_fingerprint_sha256": source_fingerprint,
                "semantics": semantics,
            }
        )

    @staticmethod
    def _validate_payload(payload: object) -> tuple[dict[str, object], RawAssetReference, str]:
        if type(payload) is not dict or set(payload) != FACT_FIELDS:
            raise ObservedBalanceAuthorityError("TRUSTED_CONTEXT_FAILURE")
        item = dict(payload)
        # M0.8 types source_id only as str and does not declare an M0.2 entity
        # kind or prefix for it.  It is an opaque accepted-source identity.
        if (
            not _identity(item["workspace_id"], "ws")
            or not _identity(item["portfolio_id"], "port")
            or not _identity(item["exchange_account_id"], "xacc")
            or type(item["source_id"]) is not str
            or item["environment"] not in ENVIRONMENTS
        ):
            raise ObservedBalanceAuthorityError("TRUSTED_CONTEXT_FAILURE")
        asset = RawAssetReference.from_mapping(item["asset_reference"])
        if (
            type(item["observed_quantity"]) is not str
            or DECIMAL.fullmatch(item["observed_quantity"]) is None
        ):
            raise ObservedBalanceAuthorityError("TRUSTED_CONTEXT_FAILURE")
        _timestamp(item["as_of_utc"])
        projected = {
            key: value for key, value in item.items() if key != "source_fingerprint_sha256"
        }
        computed = canonical_json_sha256(projected)
        if (
            type(item["source_fingerprint_sha256"]) is not str
            or SHA256.fullmatch(item["source_fingerprint_sha256"]) is None
            or item["source_fingerprint_sha256"] != computed
        ):
            raise ObservedBalanceAuthorityError("TRUSTED_CONTEXT_FAILURE")
        return item, asset, computed

    @classmethod
    def _record(cls, payload: object, semantics: object) -> AcceptedObservedBalanceFact:
        if type(semantics) is not str or semantics not in OBSERVATION_SEMANTICS:
            raise ObservedBalanceAuthorityError("TRUSTED_CONTEXT_FAILURE")
        item, asset, fingerprint = cls._validate_payload(payload)
        canonical = json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        membership_fingerprint = cls._membership_fingerprint(
            item["source_id"],
            fingerprint,
            semantics,  # type: ignore[arg-type]
        )
        return AcceptedObservedBalanceFact(
            item["source_id"],
            fingerprint,
            item["workspace_id"],
            item["portfolio_id"],
            item["environment"],
            item["exchange_account_id"],
            asset,
            item["observed_quantity"],
            item["as_of_utc"],
            semantics,
            membership_fingerprint,
            canonical,
        )  # type: ignore[arg-type]

    def _validate_state(self, state: object) -> None:
        if (
            not isinstance(state, AtomicObservedBalanceState)
            or type(state.store_revision) is not int
            or state.store_revision != len(state.accepted)
            or type(state.accepted) is not tuple
        ):
            raise ObservedBalanceAuthorityError("CORRUPT_OBSERVED_BALANCE_AUTHORITY")
        seen: set[str] = set()
        for position, record in enumerate(state.accepted, 1):
            if (
                not isinstance(record, AcceptedObservedBalanceFact)
                or record.source_id in seen
                or position > state.store_revision
            ):
                raise ObservedBalanceAuthorityError("CORRUPT_OBSERVED_BALANCE_AUTHORITY")
            seen.add(record.source_id)
            try:
                restored = self._record(
                    json.loads(record.canonical_observed_fact_json), record.semantics
                )
            except (json.JSONDecodeError, ObservedBalanceAuthorityError) as exc:
                raise ObservedBalanceAuthorityError("CORRUPT_OBSERVED_BALANCE_AUTHORITY") from exc
            if (
                restored != record
                or record.membership_fingerprint_sha256
                != self._membership_fingerprint(
                    record.source_id,
                    record.source_fingerprint_sha256,
                    record.semantics,
                )
            ):
                raise ObservedBalanceAuthorityError("CORRUPT_OBSERVED_BALANCE_AUTHORITY")

    def _publish(
        self, payload: Mapping[str, object], *, semantics: str
    ) -> AcceptedObservedBalanceFact:
        record = self._record(payload, semantics)
        with self._carrier.authority_fence():
            state = self._carrier.read()
            self._validate_state(state)
            prior = next(
                (item for item in state.accepted if item.source_id == record.source_id), None
            )
            if prior is not None:
                if prior != record:
                    raise ObservedBalanceAuthorityError("OBSERVED_BALANCE_IDENTITY_CONFLICT")
                return prior
            replacement = AtomicObservedBalanceState(
                state.store_revision + 1, state.accepted + (record,)
            )
            self._validate_state(replacement)
            self._carrier.compare_and_swap(state.store_revision, replacement)
            return record

    def resolve(
        self, source_id: str, *, expected_fingerprint: str | None = None
    ) -> AcceptedObservedBalanceFact | None:
        with self._carrier.authority_fence():
            state = self._carrier.read()
            self._validate_state(state)
            record = next((item for item in state.accepted if item.source_id == source_id), None)
            if (
                record is not None
                and expected_fingerprint is not None
                and record.source_fingerprint_sha256 != expected_fingerprint
            ):
                raise ObservedBalanceAuthorityError("TRUSTED_CONTEXT_FAILURE")
            return record

    def consume_accepted(
        self,
        source_id: str,
        expected_fingerprint: str,
        consumer: Callable[[AcceptedObservedBalanceFact], T],
    ) -> T:
        """Consume exact membership while retaining the reentrant carrier fence."""
        with self._carrier.authority_fence():
            state = self._carrier.read()
            self._validate_state(state)
            record = next((item for item in state.accepted if item.source_id == source_id), None)
            if record is None or record.source_fingerprint_sha256 != expected_fingerprint:
                raise ObservedBalanceAuthorityError("TRUSTED_CONTEXT_FAILURE")
            return consumer(record)
