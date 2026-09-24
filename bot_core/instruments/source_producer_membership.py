"""Core-owned, append-only source-producer membership authority.

Fingerprints are integrity evidence, never admission proof.  Only immutable declarations
compiled into a trusted CryptoHunter release may be copied to the journal.  Core process
code is trusted; plugin/config/manifest data is not.  A plugin capable of arbitrary code
execution inside the Core process is outside this declaration boundary/threat model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping
import sqlite3

from bot_core.instruments.core_time import PRODUCTION_CORE_CLOCK

_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MARKETS = frozenset({"SPOT", "MARGIN", "PERPETUAL", "DELIVERY_FUTURES", "OPTIONS"})


def _time(value: object) -> datetime | None:
    if type(value) is not str or _UTC.fullmatch(value) is None:
        return None
    try:
        result = datetime.strptime(
            value, "%Y-%m-%dT%H:%M:%SZ" if "." not in value else "%Y-%m-%dT%H:%M:%S.%fZ"
        )
    except ValueError:
        return None
    return result.replace(tzinfo=timezone.utc)


def _safe(value: object) -> bool:
    if value is None or type(value) in {str, bool, int}:
        return True
    if type(value) is dict:
        return all(type(key) is str and _safe(item) for key, item in value.items())
    return False


def _fingerprint(domain: str, value: Mapping[str, Any], excluded: frozenset[str]) -> str | None:
    try:
        payload = {key: value[key] for key in sorted(set(value) - excluded)}
        if not _safe(payload):
            return None
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        return hashlib.sha256((domain + "\n" + encoded).encode()).hexdigest()
    except (KeyError, TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class AcceptedSourceProducerMembershipGrant:
    record_type: str
    accepted_source_producer_membership_id: str
    source_exchange_id: str
    market_type: str
    source_adapter_family_id: str
    source_adapter_implementation_id: str
    source_adapter_release_id: str
    source_adapter_version: str
    producer_generation: int
    effective_at_utc: str
    authority_admitted_at_utc: str
    previous_membership_id: str | None
    core_admission_evidence: str
    content_fingerprint: str

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AcceptedSourceProducerMembershipEvent:
    record_type: str
    event_id: str
    membership_id: str
    event_type: str
    effective_at_utc: str
    authority_admitted_at_utc: str
    successor_membership_id: str | None
    previous_event_id: str | None
    core_admission_evidence: str
    content_fingerprint: str

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


_GRANT_FIELDS = frozenset(field.name for field in fields(AcceptedSourceProducerMembershipGrant))
_EVENT_FIELDS = frozenset(field.name for field in fields(AcceptedSourceProducerMembershipEvent))


def membership_fingerprint(record: Mapping[str, Any] | object) -> str | None:
    """Return grant integrity hash. CONTENT_FINGERPRINT_IS_NOT_ADMISSION_PROOF."""
    if not isinstance(record, Mapping):
        return None
    return _fingerprint(
        "cryptohunter.m0.12.source-producer-membership-grant.v2",
        record,
        frozenset({"accepted_source_producer_membership_id", "content_fingerprint"}),
    )


def membership_event_fingerprint(record: Mapping[str, Any] | object) -> str | None:
    if not isinstance(record, Mapping):
        return None
    return _fingerprint(
        "cryptohunter.m0.12.source-producer-membership-event.v1",
        record,
        frozenset({"event_id", "content_fingerprint"}),
    )


def validate_source_producer_membership(record: object) -> bool:
    """Structural/integrity validation only; this never proves Core admission."""
    try:
        generation = record.get("producer_generation") if isinstance(record, Mapping) else None
        return bool(
            isinstance(record, Mapping)
            and set(record) == _GRANT_FIELDS
            and _safe(dict(record))
            and record.get("record_type") == "GRANT"
            and type(record.get("accepted_source_producer_membership_id")) is str
            and record["accepted_source_producer_membership_id"].startswith("aspm_")
            and all(
                type(record.get(name)) is str and bool(record[name])
                for name in (
                    "source_exchange_id",
                    "source_adapter_family_id",
                    "source_adapter_implementation_id",
                    "source_adapter_release_id",
                    "source_adapter_version",
                    "core_admission_evidence",
                )
            )
            and record.get("market_type") in _MARKETS
            and type(generation) is int
            and generation > 0
            and _time(record.get("effective_at_utc")) is not None
            and _time(record.get("authority_admitted_at_utc")) is not None
            and (
                record.get("previous_membership_id") is None
                or (
                    type(record["previous_membership_id"]) is str
                    and record["previous_membership_id"].startswith("aspm_")
                )
            )
            and record.get("previous_membership_id")
            != record.get("accepted_source_producer_membership_id")
            and type(record.get("content_fingerprint")) is str
            and _SHA256.fullmatch(record["content_fingerprint"]) is not None
            and record["content_fingerprint"] == membership_fingerprint(record)
        )
    except (KeyError, TypeError, ValueError):
        return False


def validate_source_producer_membership_event(record: object) -> bool:
    try:
        return bool(
            isinstance(record, Mapping)
            and set(record) == _EVENT_FIELDS
            and _safe(dict(record))
            and record.get("record_type") == "EVENT"
            and type(record.get("event_id")) is str
            and record["event_id"].startswith("aspme_")
            and type(record.get("membership_id")) is str
            and record["membership_id"].startswith("aspm_")
            and record.get("event_type") in {"REVOKE", "SUPERSEDE"}
            and _time(record.get("effective_at_utc")) is not None
            and _time(record.get("authority_admitted_at_utc")) is not None
            and (
                (record["event_type"] == "REVOKE" and record.get("successor_membership_id") is None)
                or (
                    record["event_type"] == "SUPERSEDE"
                    and type(record.get("successor_membership_id")) is str
                    and record["successor_membership_id"].startswith("aspm_")
                )
            )
            and (
                record.get("previous_event_id") is None
                or (
                    type(record["previous_event_id"]) is str
                    and record["previous_event_id"].startswith("aspme_")
                )
            )
            and type(record.get("core_admission_evidence")) is str
            and bool(record["core_admission_evidence"])
            and type(record.get("content_fingerprint")) is str
            and _SHA256.fullmatch(record["content_fingerprint"]) is not None
            and record["content_fingerprint"] == membership_event_fingerprint(record)
        )
    except (KeyError, TypeError, ValueError):
        return False


class _SQLiteMembershipCarrierBase:
    """Cross-platform transactional journal with an exact durable monotonic head.

    Records and the singleton head are committed in one SQLite ``BEGIN IMMEDIATE``
    transaction. A successful COMMIT is the acknowledgement boundary. Replay requires
    sequences exactly ``1..committed_sequence`` and a digest chain ending at the durable
    head. This detects missing acknowledged rows in the current database, including a
    syntactically clean tail deletion. Rollback of the entire database together with its
    head needs an external TPM/remote monotonic anchor and is explicitly out of scope.
    """

    _ZERO = "0" * 64
    _AUTHORITY_DOMAIN = ""
    _SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                metadata_existed = "membership_authority_store_metadata" in tables
                legacy_nonempty = False
                if not metadata_existed:
                    if "membership_authority_records" in tables:
                        legacy_nonempty = (
                            connection.execute(
                                "SELECT EXISTS(SELECT 1 FROM membership_authority_records)"
                            ).fetchone()[0]
                            == 1
                        )
                    if "membership_authority_head" in tables:
                        head = connection.execute(
                            "SELECT committed_sequence FROM membership_authority_head WHERE singleton = 1"
                        ).fetchone()
                        legacy_nonempty = legacy_nonempty or (head is not None and head[0] != 0)
                if legacy_nonempty:
                    raise ValueError("MEMBERSHIP_AUTHORITY_DOMAIN_MISSING")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS membership_authority_store_metadata ("
                    "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
                    "authority_domain TEXT NOT NULL, schema_version INTEGER NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS membership_authority_records ("
                    "journal_sequence INTEGER PRIMARY KEY, canonical_record TEXT NOT NULL, "
                    "previous_record_digest TEXT NOT NULL, record_digest TEXT NOT NULL UNIQUE)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS membership_authority_head ("
                    "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
                    "committed_sequence INTEGER NOT NULL, committed_head_digest TEXT NOT NULL, "
                    "last_committed_record_id TEXT)"
                )
                if not metadata_existed:
                    connection.execute(
                        "INSERT INTO membership_authority_store_metadata VALUES (1, ?, ?)",
                        (self._AUTHORITY_DOMAIN, self._SCHEMA_VERSION),
                    )
                self._validate_domain(connection)
                connection.execute(
                    "INSERT OR IGNORE INTO membership_authority_head VALUES (1, 0, ?, NULL)",
                    (self._ZERO,),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    @classmethod
    def _validate_domain(cls, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT singleton, authority_domain, schema_version "
            "FROM membership_authority_store_metadata"
        ).fetchall()
        if rows != [(1, cls._AUTHORITY_DOMAIN, cls._SCHEMA_VERSION)]:
            raise ValueError("MEMBERSHIP_AUTHORITY_DOMAIN_MISMATCH")

    @staticmethod
    def _record_id(record: Mapping[str, Any]) -> str:
        value = (
            record.get("accepted_source_producer_membership_id")
            if record.get("record_type") == "GRANT"
            else record.get("event_id")
        )
        if type(value) is not str or not value:
            raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
        return value

    @classmethod
    def _digest(cls, sequence: int, canonical: str, previous: str) -> str:
        material = json.dumps(
            {
                "journal_sequence": sequence,
                "canonical_record": canonical,
                "previous_record_digest": previous,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        separator = f"cryptohunter.m0.12.membership-journal-row.v2\n{cls._AUTHORITY_DOMAIN}\n"
        return hashlib.sha256((separator + material).encode()).hexdigest()

    @classmethod
    def _validated_rows(cls, connection: sqlite3.Connection) -> tuple[dict[str, Any], ...]:
        cls._validate_domain(connection)
        head = connection.execute(
            "SELECT committed_sequence, committed_head_digest, last_committed_record_id "
            "FROM membership_authority_head WHERE singleton = 1"
        ).fetchone()
        if head is None:
            raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
        sequence, head_digest, last_id = head
        rows = connection.execute(
            "SELECT journal_sequence, canonical_record, previous_record_digest, record_digest "
            "FROM membership_authority_records ORDER BY journal_sequence"
        ).fetchall()
        if type(sequence) is not int or sequence < 0 or len(rows) != sequence:
            raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
        previous = cls._ZERO
        records = []
        for expected, row in enumerate(rows, 1):
            row_sequence, canonical, row_previous, digest = row
            if (
                row_sequence != expected
                or row_previous != previous
                or digest != cls._digest(expected, canonical, previous)
            ):
                raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
            try:
                record = json.loads(canonical)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL") from exc
            if type(record) is not dict:
                raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
            records.append(record)
            previous = digest
        expected_last = None if not records else cls._record_id(records[-1])
        if head_digest != previous or last_id != expected_last:
            raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
        return tuple(records)

    def read(self) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            return self._validated_rows(connection)

    def _transact_append(self, record_factory: Any, semantic_validator: Any) -> Any:
        """Serialize semantic admission and durable append in one writer transaction."""
        with self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                records = self._validated_rows(connection)
                decision, result = semantic_validator(records, None)
                if decision == "EXISTING":
                    connection.commit()
                    return result
                if decision != "READY":
                    connection.rollback()
                    return None
                record = record_factory()
                decision, result = semantic_validator(records, record)
                if decision != "APPEND":
                    connection.rollback()
                    return None
                canonical = json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                record_id = self._record_id(record)
                head = connection.execute(
                    "SELECT committed_sequence, committed_head_digest FROM membership_authority_head WHERE singleton = 1"
                ).fetchone()
                sequence, previous = head
                next_sequence = sequence + 1
                digest = self._digest(next_sequence, canonical, previous)
                connection.execute(
                    "INSERT INTO membership_authority_records VALUES (?, ?, ?, ?)",
                    (next_sequence, canonical, previous, digest),
                )
                connection.execute(
                    "UPDATE membership_authority_head SET committed_sequence = ?, committed_head_digest = ?, last_committed_record_id = ? WHERE singleton = 1",
                    (next_sequence, digest, record_id),
                )
                durable_records = self._validated_rows(connection)
                if durable_records != records + (record,):
                    raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
                # Full authority semantics, not only carrier hashes, must hold pre-COMMIT.
                post_decision, _ = semantic_validator(durable_records, None)
                if post_decision not in {"VALID", "EXISTING"}:
                    raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise


class SQLiteMembershipCarrier(_SQLiteMembershipCarrierBase):
    """Production-only durable carrier with immutable production provenance."""

    _AUTHORITY_DOMAIN = "cryptohunter.source_producer_membership.production.v1"

    @classmethod
    def open_existing(cls, path: str | Path) -> "SQLiteMembershipCarrier":
        """Open an existing carrier without provisioning or repairing its schema."""
        durable_path = Path(path).resolve()
        if not durable_path.is_file():
            raise ValueError("MEMBERSHIP_AUTHORITY_STORAGE_MISSING")
        carrier = cls.__new__(cls)
        carrier.path = durable_path
        with carrier._connect() as connection:
            carrier._validate_domain(connection)
            carrier._validated_rows(connection)
        return carrier


# Compatibility name for the proposed 1.45 API; storage is SQLite, not JSONL.
JsonlMembershipCarrier = SQLiteMembershipCarrier


def _key(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        record[name]
        for name in (
            "source_exchange_id",
            "market_type",
            "source_adapter_family_id",
            "source_adapter_implementation_id",
            "source_adapter_release_id",
            "source_adapter_version",
        )
    )


def _matches_release_declaration(record: Mapping[str, Any], declaration: object) -> bool:
    """Authenticate release-owned policy fields, never the Core runtime timestamp."""
    declared = declaration.to_mapping()
    runtime_fields = {"authority_admitted_at_utc", "content_fingerprint"}
    return {key: value for key, value in record.items() if key not in runtime_fields} == {
        key: value for key, value in declared.items() if key not in runtime_fields
    }


@dataclass(frozen=True, slots=True)
class _Replay:
    grants: Mapping[str, AcceptedSourceProducerMembershipGrant]
    events: Mapping[str, AcceptedSourceProducerMembershipEvent]
    terminal_by_membership: Mapping[str, AcceptedSourceProducerMembershipEvent]


class _SourceProducerMembershipAuthorityBase:
    """Shared implementation; concrete types encode trusted-time provenance."""

    __slots__ = ("_carrier",)

    def __init__(self, carrier: _SQLiteMembershipCarrierBase) -> None:
        self._carrier = carrier
        self._replay(carrier.read())

    def _now_utc(self) -> str:
        raise NotImplementedError

    def __deepcopy__(self, memo: dict[int, Any]) -> SourceProducerMembershipAuthority:
        """Authority is an opaque process dependency, not caller-copyable state."""
        memo[id(self)] = self
        return self

    @staticmethod
    def _replay(records: object) -> _Replay:
        if type(records) is not tuple:
            raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
        grants: dict[str, AcceptedSourceProducerMembershipGrant] = {}
        events: dict[str, AcceptedSourceProducerMembershipEvent] = {}
        terminal: dict[str, AcceptedSourceProducerMembershipEvent] = {}
        generations: set[tuple[tuple[str, ...], int]] = set()
        last_authority_admitted_at: datetime | None = None
        for raw in records:
            if validate_source_producer_membership(raw):
                expected = _RELEASE_GRANT_BY_ID.get(raw["accepted_source_producer_membership_id"])
                if expected is None or not _matches_release_declaration(raw, expected):
                    raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
                grant = AcceptedSourceProducerMembershipGrant(**raw)
                admitted_at = _time(grant.authority_admitted_at_utc)
                if (
                    last_authority_admitted_at is not None
                    and admitted_at < last_authority_admitted_at
                ):
                    raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
                identifier = grant.accepted_source_producer_membership_id
                if identifier in grants or (_key(raw), grant.producer_generation) in generations:
                    raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
                if grant.producer_generation == 1:
                    if grant.previous_membership_id is not None:
                        raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
                else:
                    previous = grants.get(grant.previous_membership_id or "")
                    if (
                        previous is None
                        or _key(previous.to_mapping()) != _key(raw)
                        or previous.producer_generation + 1 != grant.producer_generation
                        or _time(previous.effective_at_utc) >= _time(grant.effective_at_utc)
                    ):
                        raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
                grants[identifier] = grant
                generations.add((_key(raw), grant.producer_generation))
                last_authority_admitted_at = admitted_at
            elif validate_source_producer_membership_event(raw):
                expected_event = _RELEASE_EVENT_BY_ID.get(raw["event_id"])
                if expected_event is None or not _matches_release_declaration(raw, expected_event):
                    raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
                event = AcceptedSourceProducerMembershipEvent(**raw)
                grant = grants.get(event.membership_id)
                admitted_at = _time(event.authority_admitted_at_utc)
                if (
                    event.event_id in events
                    or grant is None
                    or event.membership_id in terminal
                    or admitted_at < _time(grant.authority_admitted_at_utc)
                    or (
                        last_authority_admitted_at is not None
                        and admitted_at < last_authority_admitted_at
                    )
                ):
                    raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
                if event.previous_event_id is not None and event.previous_event_id not in events:
                    raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
                if event.event_type == "SUPERSEDE":
                    successor = grants.get(event.successor_membership_id or "")
                    if (
                        successor is None
                        or successor.previous_membership_id
                        != grant.accepted_source_producer_membership_id
                        or _key(successor.to_mapping()) != _key(grant.to_mapping())
                        or successor.producer_generation != grant.producer_generation + 1
                        or _time(successor.effective_at_utc) != _time(event.effective_at_utc)
                        or admitted_at < _time(successor.authority_admitted_at_utc)
                    ):
                        raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
                events[event.event_id] = event
                terminal[event.membership_id] = event
                last_authority_admitted_at = admitted_at
            else:
                raise ValueError("CORRUPT_MEMBERSHIP_JOURNAL")
        return _Replay(
            MappingProxyType(grants), MappingProxyType(events), MappingProxyType(terminal)
        )

    def _state(self) -> _Replay:
        return self._replay(self._carrier.read())

    def admit_release_grant(
        self, declaration_id: object
    ) -> AcceptedSourceProducerMembershipGrant | None:
        declaration = (
            _RELEASE_GRANTS_BY_NAME.get(declaration_id) if type(declaration_id) is str else None
        )
        if declaration is None:
            return None
        identifier = declaration.accepted_source_producer_membership_id

        def materialize() -> dict[str, Any]:
            return _materialize_grant(declaration, self._now_utc())

        def validate(
            records: tuple[dict[str, Any], ...], proposed: dict[str, Any] | None
        ) -> tuple[str, Any]:
            state = self._replay(records)
            existing = state.grants.get(identifier)
            if existing is not None:
                return "EXISTING", existing
            if proposed is None:
                return "READY", state
            candidate = proposed
            result = AcceptedSourceProducerMembershipGrant(**candidate)
            self._replay(records + (candidate,))
            return "APPEND", result

        return self._carrier._transact_append(materialize, validate)

    # Backward-compatible name accepts only an immutable release declaration name.
    admit_release_declaration = admit_release_grant

    def admit_release_event(
        self, declaration_id: object
    ) -> AcceptedSourceProducerMembershipEvent | None:
        declaration = (
            _RELEASE_EVENTS_BY_NAME.get(declaration_id) if type(declaration_id) is str else None
        )
        if declaration is None:
            return None
        identifier = declaration.event_id

        def materialize() -> dict[str, Any]:
            return _materialize_event(declaration, self._now_utc())

        def validate(
            records: tuple[dict[str, Any], ...], proposed: dict[str, Any] | None
        ) -> tuple[str, Any]:
            state = self._replay(records)
            existing = state.events.get(identifier)
            if existing is not None:
                return "EXISTING", existing
            if proposed is None:
                return "READY", state
            candidate = proposed
            # Any different terminal already committed for this membership loses the race.
            if candidate["membership_id"] in state.terminal_by_membership:
                return "DENY", None
            result = AcceptedSourceProducerMembershipEvent(**candidate)
            self._replay(records + (candidate,))
            return "APPEND", result

        return self._carrier._transact_append(materialize, validate)

    def resolve_current(
        self, producer_identity: object, producer_generation: object, at_utc: object
    ) -> AcceptedSourceProducerMembershipGrant | None:
        try:
            when = _time(at_utc)
            if (
                when is None
                or type(producer_generation) is not int
                or producer_generation < 1
                or not isinstance(producer_identity, Mapping)
            ):
                return None
            state = self._state()
            candidates = []
            for grant in state.grants.values():
                terminal = state.terminal_by_membership.get(
                    grant.accepted_source_producer_membership_id
                )
                cutoff = (
                    None
                    if terminal is None
                    else max(
                        _time(terminal.effective_at_utc),
                        _time(terminal.authority_admitted_at_utc),
                    )
                )
                active = terminal is None or when < cutoff
                if (
                    active
                    and _time(grant.authority_admitted_at_utc) <= when
                    and _time(grant.effective_at_utc) <= when
                    and _exact(grant, producer_identity)
                ):
                    candidates.append(grant)
            return (
                candidates[0]
                if len(candidates) == 1 and candidates[0].producer_generation == producer_generation
                else None
            )
        except (KeyError, TypeError, ValueError):
            return None

    def resolve_historical(
        self,
        membership_id: object,
        producer_generation: object,
        fingerprint: object,
        producer_identity: object,
        accepted_at_utc: object,
    ) -> AcceptedSourceProducerMembershipGrant | None:
        try:
            when = _time(accepted_at_utc)
            if (
                when is None
                or type(membership_id) is not str
                or not membership_id.startswith("aspm_")
                or type(producer_generation) is not int
                or producer_generation < 1
                or type(fingerprint) is not str
            ):
                return None
            state = self._state()
            grant = state.grants.get(membership_id)
            terminal = state.terminal_by_membership.get(membership_id)
            cutoff = (
                None
                if terminal is None
                else max(
                    _time(terminal.effective_at_utc),
                    _time(terminal.authority_admitted_at_utc),
                )
            )
            if (
                grant is None
                or grant.producer_generation != producer_generation
                or grant.content_fingerprint != fingerprint
                or not _exact(grant, producer_identity)
                or _time(grant.authority_admitted_at_utc) > when
                or _time(grant.effective_at_utc) > when
                or (cutoff is not None and cutoff <= when)
            ):
                return None
            return grant
        except (KeyError, TypeError, ValueError):
            return None


class SourceProducerMembershipAuthority(_SourceProducerMembershipAuthorityBase):
    """Production authority permanently bound to the sealed Core UTC authority."""

    __slots__ = ()

    def __init__(self, carrier: SQLiteMembershipCarrier) -> None:
        if type(carrier) is not SQLiteMembershipCarrier:
            raise TypeError("exact production SQLiteMembershipCarrier required")
        super().__init__(carrier)

    def _now_utc(self) -> str:
        return PRODUCTION_CORE_CLOCK.now_utc()


def _exact(grant: AcceptedSourceProducerMembershipGrant, identity: Mapping[str, Any]) -> bool:
    names = (
        "source_exchange_id",
        "market_type",
        "source_adapter_family_id",
        "source_adapter_implementation_id",
        "source_adapter_release_id",
        "source_adapter_version",
    )
    return set(identity) == set(names) and all(
        type(identity.get(name)) is str and identity[name] == getattr(grant, name) for name in names
    )


def _materialize_grant(
    grant: AcceptedSourceProducerMembershipGrant, admitted_at: str | None = None
) -> dict[str, Any]:
    value = grant.to_mapping()
    if admitted_at is not None:
        value["authority_admitted_at_utc"] = admitted_at
    value["content_fingerprint"] = membership_fingerprint(value)
    return value


def _materialize_event(
    event: AcceptedSourceProducerMembershipEvent, admitted_at: str | None = None
) -> dict[str, Any]:
    value = event.to_mapping()
    if admitted_at is not None:
        value["authority_admitted_at_utc"] = admitted_at
    value["content_fingerprint"] = membership_event_fingerprint(value)
    return value


def _grant(
    identifier: str, generation: int, effective: str, previous: str | None
) -> AcceptedSourceProducerMembershipGrant:
    return AcceptedSourceProducerMembershipGrant(
        "GRANT",
        identifier,
        "binance",
        "SPOT",
        "binance_public_catalog",
        "impl_ccxt_binance",
        "release_2026_09_15",
        "4.5.1",
        generation,
        effective,
        "",
        previous,
        "CryptoHunter/CoreRelease/1.45.0/source-producer-declarations",
        "",
    )


_G1 = _grant("aspm_core_1_45_binance_spot_1", 1, "2026-09-14T00:00:00Z", None)
_G2 = _grant(
    "aspm_core_1_45_binance_spot_2",
    2,
    "2026-09-15T02:00:00Z",
    _G1.accepted_source_producer_membership_id,
)
_GTEST = AcceptedSourceProducerMembershipGrant(
    "GRANT",
    "aspm_core_1_45_generic_testnet_spot_1",
    "generic_testnet_venue",
    "SPOT",
    "generic_testnet_adapter_family",
    "impl_ccxt_binance",
    "release_2026_09_15",
    "4.5.1",
    1,
    "2026-01-01T00:00:00Z",
    "",
    None,
    "CryptoHunter/CoreRelease/1.45.0/source-producer-declarations",
    "",
)
_REVOKE = AcceptedSourceProducerMembershipEvent(
    "EVENT",
    "aspme_core_1_45_revoke_binance_spot_1",
    _G1.accepted_source_producer_membership_id,
    "REVOKE",
    "2026-09-15T02:00:00Z",
    "",
    None,
    None,
    "CryptoHunter/CoreRelease/1.45.0/source-producer-events",
    "",
)
_LATE_REVOKE = AcceptedSourceProducerMembershipEvent(
    "EVENT",
    "aspme_core_1_45_late_revoke_binance_spot_1",
    _G1.accepted_source_producer_membership_id,
    "REVOKE",
    "2026-09-15T02:00:00Z",
    "",
    None,
    None,
    "CryptoHunter/CoreRelease/1.45.0/source-producer-events",
    "",
)
_SUPERSEDE = AcceptedSourceProducerMembershipEvent(
    "EVENT",
    "aspme_core_1_45_supersede_binance_spot_1",
    _G1.accepted_source_producer_membership_id,
    "SUPERSEDE",
    "2026-09-15T02:00:00Z",
    "",
    _G2.accepted_source_producer_membership_id,
    None,
    "CryptoHunter/CoreRelease/1.45.0/source-producer-events",
    "",
)
_RELEASE_GRANTS_BY_NAME = MappingProxyType(
    {
        "core_release_1_45_binance_spot": _G1,
        "core_release_1_45_binance_spot_generation_2": _G2,
        "core_release_1_45_generic_testnet_spot": _GTEST,
    }
)
_RELEASE_EVENTS_BY_NAME = MappingProxyType(
    {
        "core_release_1_45_revoke_binance_spot": _REVOKE,
        "core_release_1_45_late_revoke_binance_spot": _LATE_REVOKE,
        "core_release_1_45_supersede_binance_spot": _SUPERSEDE,
    }
)
_RELEASE_GRANT_BY_ID = MappingProxyType(
    {item.accepted_source_producer_membership_id: item for item in _RELEASE_GRANTS_BY_NAME.values()}
)
_RELEASE_EVENT_BY_ID = MappingProxyType(
    {item.event_id: item for item in _RELEASE_EVENTS_BY_NAME.values()}
)

__all__ = [
    "AcceptedSourceProducerMembershipGrant",
    "AcceptedSourceProducerMembershipEvent",
    "SQLiteMembershipCarrier",
    "JsonlMembershipCarrier",
    "SourceProducerMembershipAuthority",
    "membership_fingerprint",
    "membership_event_fingerprint",
    "validate_source_producer_membership",
    "validate_source_producer_membership_event",
]
