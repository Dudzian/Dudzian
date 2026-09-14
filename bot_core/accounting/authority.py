"""Carrier-owned M0.8 accounting journal and typed economic-fact authority.

Generic accepted-content membership is deliberately only the bottom link of the
trust chain.  It never authorizes an accounting source by itself.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import re
from threading import RLock
from types import MappingProxyType
from typing import ContextManager, Iterator, Mapping, Protocol

from bot_core.m09_kill_switch_authority import CoreAcceptedContentAuthority
from bot_core.persistence.fingerprints import canonical_json_sha256

ACCOUNTING_RULE_VERSION = "ACCOUNTING_SPOT_FIFO_V1"
ENVIRONMENTS = frozenset({"PAPER", "TESTNET", "LIVE"})
OWNED_ROLES = frozenset({"OWNED_AVAILABLE", "OWNED_RESERVED"})
SUPPORTED_ECONOMIC_SOURCES = frozenset({"deposit", "withdrawal"})
BLOCKED_SOURCE_TYPES = frozenset(
    {"fill", "internal_transfer", "capital_reservation", "capital_release", "reconciliation_correction"}
)
DECIMAL = re.compile(r"(0|[1-9][0-9]*)(\.[0-9]*[1-9])?\Z")
ID = re.compile(r"[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d*[1-9])?Z\Z")
ECONOMIC_FIELDS = frozenset(
    {"audit_event_id", "source_type", "workspace_id", "portfolio_id", "environment",
     "effective_at_utc", "provenance", "source_fingerprint_sha256", "exchange_account_id",
     "asset_reference", "quantity", "capital_flow_kind", "basis_valuation_unit",
     "unit_cost_basis"}
)


class AccountingAuthorityError(ValueError):
    """Fail-closed accounting authority error using frozen literals."""


def _id(value: object, prefix: str) -> bool:
    return type(value) is str and value.startswith(prefix + "_") and ID.fullmatch(value) is not None


def _quantity(value: object) -> Fraction:
    if type(value) is not str or DECIMAL.fullmatch(value) is None or value == "0":
        raise AccountingAuthorityError("MALFORMED_ACCOUNTING_FACT")
    return Fraction(value)


def _time(value: object) -> datetime:
    if type(value) is not str or TIME.fullmatch(value) is None:
        raise AccountingAuthorityError("MALFORMED_ACCOUNTING_FACT")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AccountingAuthorityError("MALFORMED_ACCOUNTING_FACT") from exc
    if parsed.tzinfo != timezone.utc:
        raise AccountingAuthorityError("MALFORMED_ACCOUNTING_FACT")
    return parsed


@dataclass(frozen=True, slots=True)
class AssetReference:
    venue_asset_code: str
    canonical_display_code: str
    asset_namespace: str
    mapping_status: str

    @classmethod
    def from_mapping(cls, value: object) -> AssetReference:
        if type(value) is not dict or set(value) != {
            "venue_asset_code", "canonical_display_code", "asset_namespace", "mapping_status"
        }:
            raise AccountingAuthorityError("TRUSTED_CONTEXT_FAILURE")
        result = cls(**value)  # type: ignore[arg-type]
        result.validate()
        return result

    def validate(self) -> None:
        if any(type(item) is not str or not item for item in asdict(self).values()) or self.mapping_status not in {"EXACT", "EXPLICIT_ALIAS"}:
            raise AccountingAuthorityError("TRUSTED_CONTEXT_FAILURE")


@dataclass(frozen=True, slots=True)
class AcceptedAccountingFact:
    audit_event_id: str
    source_type: str
    source_fingerprint_sha256: str
    workspace_id: str
    portfolio_id: str
    environment: str
    canonical_payload_json: str

    def payload(self) -> Mapping[str, object]:
        return MappingProxyType(json.loads(self.canonical_payload_json))


@dataclass(frozen=True, slots=True)
class AtomicAccountingFactState:
    store_revision: int = 0
    accepted: tuple[AcceptedAccountingFact, ...] = ()


class AccountingFactCarrier(Protocol):
    def authority_fence(self) -> ContextManager[None]: ...
    def read(self) -> AtomicAccountingFactState: ...
    def compare_and_swap(self, expected_revision: int, state: AtomicAccountingFactState) -> None: ...


class InMemoryAccountingFactCarrier:
    def __init__(self, state: AtomicAccountingFactState | None = None) -> None:
        self._lock = RLock()
        self._state = state or AtomicAccountingFactState()

    @contextmanager
    def authority_fence(self) -> Iterator[None]:
        with self._lock:
            yield

    def read(self) -> AtomicAccountingFactState:
        with self._lock:
            return self._state

    def compare_and_swap(self, expected_revision: int, state: AtomicAccountingFactState) -> None:
        with self._lock:
            if self._state.store_revision != expected_revision:
                raise RuntimeError("AUTHORITY_CAS_CONFLICT")
            self._state = state


class _AccountingFactWriter:
    def __init__(self, authority: CoreAcceptedAccountingFactProjection) -> None:
        self.__authority = authority

    def accept(self, payload: Mapping[str, object]) -> AcceptedAccountingFact:
        return self.__authority._accept(payload)  # noqa: SLF001


class CoreAcceptedAccountingFactProjection:
    """Typed, carrier-owned accepted exact-economic AuditEvent projection."""

    @classmethod
    def compose(cls, carrier: AccountingFactCarrier, *, content_membership: CoreAcceptedContentAuthority) -> tuple[CoreAcceptedAccountingFactProjection, _AccountingFactWriter]:
        authority = cls(carrier, content_membership=content_membership)
        return authority, _AccountingFactWriter(authority)

    def __init__(self, carrier: AccountingFactCarrier, *, content_membership: CoreAcceptedContentAuthority) -> None:
        if not isinstance(content_membership, CoreAcceptedContentAuthority):
            raise AccountingAuthorityError("TRUSTED_CONTEXT_FAILURE")
        self._carrier = carrier
        self._content = content_membership
        with carrier.authority_fence():
            self._validate_state(carrier.read(), require_content=True)

    @staticmethod
    def _validate_payload(payload: object) -> tuple[dict[str, object], str]:
        if type(payload) is not dict or set(payload) != ECONOMIC_FIELDS:
            raise AccountingAuthorityError("TRUSTED_CONTEXT_FAILURE")
        item = dict(payload)
        source_type = item["source_type"]
        if source_type in BLOCKED_SOURCE_TYPES:
            raise AccountingAuthorityError("UNSUPPORTED_ACCOUNTING_SEMANTICS")
        if source_type not in SUPPORTED_ECONOMIC_SOURCES:
            raise AccountingAuthorityError("UNSUPPORTED_ACCOUNTING_SEMANTICS")
        if not _id(item["audit_event_id"], "evt") or not _id(item["workspace_id"], "ws") or not _id(item["portfolio_id"], "port") or not _id(item["exchange_account_id"], "xacc") or item["environment"] not in ENVIRONMENTS:
            raise AccountingAuthorityError("TRUSTED_CONTEXT_FAILURE")
        if type(item["provenance"]) is not str or not item["provenance"]:
            raise AccountingAuthorityError("TRUSTED_CONTEXT_FAILURE")
        _time(item["effective_at_utc"])
        AssetReference.from_mapping(item["asset_reference"])
        AssetReference.from_mapping(item["basis_valuation_unit"])
        _quantity(item["quantity"])
        _quantity(item["unit_cost_basis"])
        expected_kind = "EXTERNAL_CONTRIBUTION" if source_type == "deposit" else "EXTERNAL_WITHDRAWAL"
        if item["capital_flow_kind"] != expected_kind:
            raise AccountingAuthorityError("TRUSTED_CONTEXT_FAILURE")
        projected = {key: value for key, value in item.items() if key != "source_fingerprint_sha256"}
        computed = canonical_json_sha256(projected)
        if item["source_fingerprint_sha256"] != computed:
            raise AccountingAuthorityError("TRUSTED_CONTEXT_FAILURE")
        return item, computed

    def _validate_state(self, state: object, *, require_content: bool) -> None:
        if not isinstance(state, AtomicAccountingFactState) or type(state.store_revision) is not int or state.store_revision != len(state.accepted) or type(state.accepted) is not tuple:
            raise AccountingAuthorityError("CORRUPT_ACCOUNTING_SOURCE_AUTHORITY")
        seen: set[str] = set()
        for record in state.accepted:
            if not isinstance(record, AcceptedAccountingFact) or record.audit_event_id in seen:
                raise AccountingAuthorityError("CORRUPT_ACCOUNTING_SOURCE_AUTHORITY")
            seen.add(record.audit_event_id)
            try:
                payload, computed = self._validate_payload(json.loads(record.canonical_payload_json))
            except (json.JSONDecodeError, AccountingAuthorityError) as exc:
                raise AccountingAuthorityError("CORRUPT_ACCOUNTING_SOURCE_AUTHORITY") from exc
            if (record.audit_event_id, record.source_type, record.source_fingerprint_sha256, record.workspace_id, record.portfolio_id, record.environment) != (payload["audit_event_id"], payload["source_type"], computed, payload["workspace_id"], payload["portfolio_id"], payload["environment"]):
                raise AccountingAuthorityError("CORRUPT_ACCOUNTING_SOURCE_AUTHORITY")
            binding = self._content.resolve(record.audit_event_id)
            if require_content and (binding is None or binding.content_fingerprint_sha256 != computed):
                raise AccountingAuthorityError("CORRUPT_ACCOUNTING_SOURCE_AUTHORITY")

    def _accept(self, payload: Mapping[str, object]) -> AcceptedAccountingFact:
        item, computed = self._validate_payload(payload)
        identity = item["audit_event_id"]
        binding = self._content.resolve(identity)  # type: ignore[arg-type]
        if binding is None or binding.content_fingerprint_sha256 != computed:
            raise AccountingAuthorityError("TRUSTED_CONTEXT_FAILURE")
        canonical_payload = json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        record = AcceptedAccountingFact(identity, item["source_type"], computed, item["workspace_id"], item["portfolio_id"], item["environment"], canonical_payload)  # type: ignore[arg-type]
        with self._carrier.authority_fence():
            state = self._carrier.read()
            self._validate_state(state, require_content=True)
            prior = next((entry for entry in state.accepted if entry.audit_event_id == identity), None)
            if prior is not None:
                if prior != record:
                    raise AccountingAuthorityError("ACCOUNTING_IDENTITY_CONFLICT")
                return prior
            self._carrier.compare_and_swap(state.store_revision, AtomicAccountingFactState(state.store_revision + 1, state.accepted + (record,)))
            return record

    def resolve(self, identity: str) -> AcceptedAccountingFact | None:
        with self._carrier.authority_fence():
            state = self._carrier.read()
            self._validate_state(state, require_content=True)
            return next((entry for entry in state.accepted if entry.audit_event_id == identity), None)


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    ledger_entry_id: str
    workspace_id: str
    portfolio_id: str
    environment: str
    exchange_account_id: str | None
    strategy_instance_id: str | None
    asset_reference: Mapping[str, str]
    account_role: str
    direction: str
    quantity: str
    source_type: str
    accounting_source_identity: str
    accounting_source_fingerprint_sha256: str
    accounting_rule_version: str
    posting_index: int
    posting_role: str
    batch_fingerprint_sha256: str
    effective_at_utc: str
    append_sequence: int
    order_id: str | None
    fill_id: str | None
    audit_event_id: str | None
    correction_reason: str | None


@dataclass(frozen=True, slots=True)
class AcceptedAccountingBatch:
    source_type: str
    accounting_source_identity: str
    accounting_source_fingerprint_sha256: str
    accounting_rule_version: str
    batch_fingerprint_sha256: str
    first_append_sequence: int
    entry_count: int


@dataclass(frozen=True, slots=True)
class AtomicAccountingState:
    store_revision: int = 0
    accepted_batches: tuple[AcceptedAccountingBatch, ...] = ()
    journal: tuple[LedgerEntry, ...] = ()


class AccountingCarrier(Protocol):
    def authority_fence(self) -> ContextManager[None]: ...
    def read(self) -> AtomicAccountingState: ...
    def compare_and_swap(self, expected_revision: int, state: AtomicAccountingState) -> None: ...


class InMemoryAccountingCarrier:
    def __init__(self, state: AtomicAccountingState | None = None) -> None:
        self._lock = RLock()
        self._state = state or AtomicAccountingState()
        self.fail_next = False

    @contextmanager
    def authority_fence(self) -> Iterator[None]:
        with self._lock:
            yield

    def read(self) -> AtomicAccountingState:
        with self._lock:
            return self._state

    def compare_and_swap(self, expected_revision: int, state: AtomicAccountingState) -> None:
        with self._lock:
            if self.fail_next:
                self.fail_next = False
                raise OSError("INJECTED_CARRIER_FAILURE")
            if self._state.store_revision != expected_revision:
                raise RuntimeError("AUTHORITY_CAS_CONFLICT")
            self._state = state


@dataclass(frozen=True, slots=True)
class InternalQuantityProjection:
    status: str
    quantity: Fraction | None


class _AccountingWriter:
    def __init__(self, authority: AccountingAuthority) -> None:
        self.__authority = authority

    def accept_source(self, accounting_source_identity: str) -> AcceptedAccountingBatch:
        return self.__authority._accept_source(accounting_source_identity)  # noqa: SLF001


class AccountingAuthority:
    @classmethod
    def compose(cls, carrier: AccountingCarrier, *, source_authority: CoreAcceptedAccountingFactProjection) -> tuple[AccountingAuthority, _AccountingWriter]:
        authority = cls(carrier, source_authority=source_authority)
        return authority, _AccountingWriter(authority)

    def __init__(self, carrier: AccountingCarrier, *, source_authority: CoreAcceptedAccountingFactProjection) -> None:
        if not isinstance(source_authority, CoreAcceptedAccountingFactProjection):
            raise AccountingAuthorityError("TRUSTED_CONTEXT_FAILURE")
        self._carrier = carrier
        self._sources = source_authority
        with carrier.authority_fence():
            self._validate_state(carrier.read(), validate_upstream=True)

    @staticmethod
    def _posting_projections(source: AcceptedAccountingFact) -> list[dict[str, object]]:
        payload = source.payload()
        common = {"workspace_id": payload["workspace_id"], "portfolio_id": payload["portfolio_id"], "environment": payload["environment"], "exchange_account_id": payload["exchange_account_id"], "strategy_instance_id": None, "asset_reference": payload["asset_reference"], "quantity": payload["quantity"]}
        if source.source_type == "deposit":
            roles = (("OWNED_AVAILABLE", "DEBIT", "ASSET_RECEIVED"), ("EXTERNAL_CAPITAL", "CREDIT", "CAPITAL_CLASSIFIED"))
        elif source.source_type == "withdrawal":
            roles = (("EXTERNAL_CAPITAL", "DEBIT", "CAPITAL_CLASSIFIED"), ("OWNED_AVAILABLE", "CREDIT", "ASSET_PAID"))
        else:
            raise AccountingAuthorityError("UNSUPPORTED_ACCOUNTING_SEMANTICS")
        return [{**common, "account_role": role, "direction": direction, "posting_role": posting_role, "posting_index": index} for index, (role, direction, posting_role) in enumerate(roles)]

    @staticmethod
    def _ledger_id(batch_fingerprint: str, append_sequence: int) -> str:
        raw = list(hashlib.sha256(f"{batch_fingerprint}:{append_sequence}".encode()).hexdigest()[:32])
        raw[12], raw[16] = "7", "8"
        value = "".join(raw)
        return f"led_{value[:8]}-{value[8:12]}-{value[12:16]}-{value[16:20]}-{value[20:]}"

    @classmethod
    def _entries(cls, source: AcceptedAccountingFact, start: int) -> tuple[LedgerEntry, ...]:
        postings = cls._posting_projections(source)
        fingerprint = canonical_json_sha256(postings)
        payload = source.payload()
        return tuple(LedgerEntry(cls._ledger_id(fingerprint, start + index), posting["workspace_id"], posting["portfolio_id"], posting["environment"], posting["exchange_account_id"], None, dict(posting["asset_reference"]), posting["account_role"], posting["direction"], posting["quantity"], source.source_type, source.audit_event_id, source.source_fingerprint_sha256, ACCOUNTING_RULE_VERSION, index, posting["posting_role"], fingerprint, payload["effective_at_utc"], start + index, None, None, source.audit_event_id, None) for index, posting in enumerate(postings))  # type: ignore[arg-type]

    def _validate_state(self, state: object, *, validate_upstream: bool) -> None:
        if not isinstance(state, AtomicAccountingState) or type(state.store_revision) is not int or state.store_revision != len(state.accepted_batches) or type(state.accepted_batches) is not tuple or type(state.journal) is not tuple:
            raise AccountingAuthorityError("CONTRACT_INCONSISTENT")
        cursor, seen_sources, seen_ids = 1, set(), set()
        balances: dict[tuple[object, ...], Fraction] = {}
        for accepted in state.accepted_batches:
            if not isinstance(accepted, AcceptedAccountingBatch) or accepted.accounting_source_identity in seen_sources or accepted.first_append_sequence != cursor or accepted.entry_count <= 0:
                raise AccountingAuthorityError("CONTRACT_INCONSISTENT")
            seen_sources.add(accepted.accounting_source_identity)
            source = self._sources.resolve(accepted.accounting_source_identity) if validate_upstream else None
            if source is None:
                raise AccountingAuthorityError("CONTRACT_INCONSISTENT")
            expected = self._entries(source, cursor)
            entries = state.journal[cursor - 1:cursor - 1 + accepted.entry_count]
            if not entries or entries != expected or accepted.entry_count != len(expected) or accepted.batch_fingerprint_sha256 != expected[0].batch_fingerprint_sha256 or accepted.source_type != source.source_type or accepted.accounting_source_fingerprint_sha256 != source.source_fingerprint_sha256 or accepted.accounting_rule_version != ACCOUNTING_RULE_VERSION:
                raise AccountingAuthorityError("CONTRACT_INCONSISTENT")
            for entry in entries:
                if entry.ledger_entry_id in seen_ids:
                    raise AccountingAuthorityError("CONTRACT_INCONSISTENT")
                seen_ids.add(entry.ledger_entry_id)
                if entry.account_role in OWNED_ROLES:
                    key = (entry.workspace_id, entry.portfolio_id, entry.environment, entry.exchange_account_id, tuple(entry.asset_reference.items()), entry.account_role)
                    amount = _quantity(entry.quantity)
                    balances[key] = balances.get(key, Fraction()) + (amount if entry.direction == "DEBIT" else -amount)
            if any(value < 0 for value in balances.values()):
                raise AccountingAuthorityError("INVALID_ACCOUNTING_TRANSITION")
            cursor += len(entries)
        if cursor - 1 != len(state.journal):
            raise AccountingAuthorityError("CONTRACT_INCONSISTENT")

    def _accept_source(self, identity: str) -> AcceptedAccountingBatch:
        source = self._sources.resolve(identity)
        if source is None:
            raise AccountingAuthorityError("TRUSTED_CONTEXT_FAILURE")
        with self._carrier.authority_fence():
            state = self._carrier.read()
            self._validate_state(state, validate_upstream=True)
            prior = next((item for item in state.accepted_batches if item.accounting_source_identity == identity), None)
            if prior is not None:
                if prior.accounting_source_fingerprint_sha256 != source.source_fingerprint_sha256:
                    raise AccountingAuthorityError("ACCOUNTING_IDENTITY_CONFLICT")
                return prior
            entries = self._entries(source, len(state.journal) + 1)
            accepted = AcceptedAccountingBatch(source.source_type, identity, source.source_fingerprint_sha256, ACCOUNTING_RULE_VERSION, entries[0].batch_fingerprint_sha256, len(state.journal) + 1, len(entries))
            replacement = AtomicAccountingState(state.store_revision + 1, state.accepted_batches + (accepted,), state.journal + entries)
            self._validate_state(replacement, validate_upstream=True)
            self._carrier.compare_and_swap(state.store_revision, replacement)
            return accepted

    def resolve_internal_quantity(self, *, workspace_id: str, portfolio_id: str, environment: str, exchange_account_id: str, asset_reference: AssetReference) -> InternalQuantityProjection:
        """Resolve the current owned projection for an exact accounting bucket.

        ``ObservedBalanceFact.as_of_utc`` belongs to the future reconciliation
        key.  Frozen M0.8 does not define it as a historical journal cutoff, so
        this internal authority intentionally accepts no observation time.
        """
        if not _id(workspace_id, "ws") or not _id(portfolio_id, "port") or environment not in ENVIRONMENTS or not _id(exchange_account_id, "xacc") or not isinstance(asset_reference, AssetReference):
            raise AccountingAuthorityError("TRUSTED_CONTEXT_FAILURE")
        asset_reference.validate()
        with self._carrier.authority_fence():
            state = self._carrier.read()
            self._validate_state(state, validate_upstream=True)
            selected = tuple(entry for entry in state.journal if entry.workspace_id == workspace_id and entry.portfolio_id == portfolio_id and entry.environment == environment and entry.exchange_account_id == exchange_account_id and dict(entry.asset_reference) == asdict(asset_reference) and entry.account_role in OWNED_ROLES)
            if not selected:
                return InternalQuantityProjection("MISSING_INTERNAL_HISTORY", None)
            quantity = sum(((_quantity(entry.quantity) if entry.direction == "DEBIT" else -_quantity(entry.quantity)) for entry in selected), Fraction())
            return InternalQuantityProjection("INTERNAL_HISTORY_PRESENT", quantity)

    def journal(self) -> tuple[LedgerEntry, ...]:
        with self._carrier.authority_fence():
            state = self._carrier.read()
            self._validate_state(state, validate_upstream=True)
            return state.journal
