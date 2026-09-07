"""Lokalny, trwały kernel metadanych StateStore zgodny z M0.11."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from threading import RLock
from typing import Any, Iterator

from bot_core.persistence.fingerprints import (
    canonical_json,
    history_tail_fingerprint_sha256,
    state_fingerprint_sha256,
    transaction_fingerprint_sha256,
)
from bot_core.persistence.transaction_descriptor import (
    StateStoreTransactionDescriptor,
    TransactionDescriptorError,
)
from bot_core.persistence.records import (
    PersistenceRecord,
    PersistenceRecordError,
    validate_persistence_record,
    validate_persistence_record_for_schema,
    record_durability_class,
    validate_record_bucket_for_schema,
)
from bot_core.persistence.record_registry import (
    PERSISTENCE_RECORD_REGISTRY,
    STATE_STORE_SCOPE_BINDINGS,
)

_CANONICAL_ID_RE = re.compile(
    r"^[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENVIRONMENTS = frozenset({"PAPER", "TESTNET", "LIVE"})
_IDENTITY_FIELDS = (
    "account_id",
    "device_installation_id",
    "state_store_identity_fingerprint_sha256",
)
_CURRENT_BUCKET = "DURABLE AUTHORITATIVE CURRENT STATE"
_HISTORY_BUCKET = "DURABLE IMMUTABLE / APPEND-ONLY HISTORY"


class StateStoreError(RuntimeError):
    """Błąd zamkniętej walidacji lub ogrodzenia trwałego StateStore."""


def _validate_schema_version_transition(
    previous: StateStoreTransactionDescriptor,
    current: StateStoreTransactionDescriptor,
) -> None:
    """Bind the sole legal descriptor schema edge to its sealed production authority."""

    if current.state_store_schema_version == previous.state_store_schema_version:
        return
    try:
        from .migration_execution import MigrationExecutionDeclaration
        from .migration_execution_contract import thaw_json
        from .state_store_v2_migration import STATE_STORE_V2_MIGRATION_AUTHORITY

        if (previous.state_store_schema_version, current.state_store_schema_version) != (1, 2):
            raise ValueError("unsupported schema edge")
        if current.current_record_mutations:
            raise ValueError("migration descriptor has current mutations")
        if len(current.immutable_history_appends) != 1:
            raise ValueError("migration descriptor append shape mismatch")
        carrier = current.immutable_history_appends[0]
        if carrier.representation_name != "Migration execution declaration":
            raise ValueError("migration declaration is unavailable")
        declaration = MigrationExecutionDeclaration.from_mapping(thaw_json(carrier.payload))
        if declaration.carrier() != carrier:
            raise ValueError("migration declaration carrier mismatch")
        STATE_STORE_V2_MIGRATION_AUTHORITY.assert_declaration(declaration)
        if current.immutable_history_appends != (declaration.carrier(),):
            raise ValueError("migration descriptor append shape mismatch")
        if (
            declaration.source_schema_version != previous.state_store_schema_version
            or declaration.target_schema_version != current.state_store_schema_version
            or declaration.account_id != current.account_id
            or declaration.account_id != previous.account_id
            or declaration.device_installation_id != current.device_installation_id
            or declaration.device_installation_id != previous.device_installation_id
            or declaration.environment != current.environment
            or declaration.environment != previous.environment
            or declaration.state_store_identity_fingerprint_sha256
            != current.state_store_identity_fingerprint_sha256
            or declaration.state_store_identity_fingerprint_sha256
            != previous.state_store_identity_fingerprint_sha256
            or declaration.expected_current_generation != previous.target_generation
            or declaration.expected_current_generation != current.expected_current_generation
            or declaration.target_generation != current.target_generation
            or declaration.pre_state_fingerprint_sha256 != previous.post_state_fingerprint_sha256
            or declaration.pre_state_fingerprint_sha256 != current.pre_state_fingerprint_sha256
            or declaration.pre_history_tail_fingerprint_sha256
            != previous.post_history_tail_fingerprint_sha256
            or declaration.pre_history_tail_fingerprint_sha256
            != current.pre_history_tail_fingerprint_sha256
        ):
            raise ValueError("migration declaration runtime binding mismatch")
    except (TypeError, ValueError) as exc:
        raise StateStoreError("unauthorized descriptor schema-version edge") from exc


def _validate_canonical_id(value: object, *, prefix: str, field_name: str) -> None:
    if not isinstance(value, str) or not _CANONICAL_ID_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a canonical M0.2 identifier")
    if not value.startswith(f"{prefix}_"):
        raise ValueError(f"{field_name} must use the canonical {prefix!r} prefix")


def _validate_positive_integer(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be an integer greater than or equal to 1")


def _validate_sha256(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 fingerprint")


@dataclass(frozen=True, slots=True)
class StateStoreMetadata:
    """Dokładna produkcyjna reprezentacja ``M0.11/StateStoreMetadata``."""

    account_id: str
    device_installation_id: str
    state_store_schema_version: int
    state_store_identity_fingerprint_sha256: str
    environment: str
    protected_freshness_generation: int
    state_fingerprint_sha256: str
    transaction_fingerprint_sha256: str
    history_tail_fingerprint_sha256: str

    def __post_init__(self) -> None:
        _validate_canonical_id(self.account_id, prefix="acct", field_name="account_id")
        _validate_canonical_id(
            self.device_installation_id,
            prefix="dev",
            field_name="device_installation_id",
        )
        _validate_positive_integer(
            self.state_store_schema_version,
            field_name="state_store_schema_version",
        )
        _validate_positive_integer(
            self.protected_freshness_generation,
            field_name="protected_freshness_generation",
        )
        if not isinstance(self.environment, str) or self.environment not in _ENVIRONMENTS:
            raise ValueError("environment must be PAPER, TESTNET, or LIVE")
        for field_name in (
            "state_store_identity_fingerprint_sha256",
            "state_fingerprint_sha256",
            "transaction_fingerprint_sha256",
            "history_tail_fingerprint_sha256",
        ):
            _validate_sha256(getattr(self, field_name), field_name=field_name)

    def to_mapping(self) -> dict[str, str | int]:
        """Zwróć deterministyczne mapowanie zawierające wyłącznie pola kontraktu."""

        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> StateStoreMetadata:
        """Zbuduj metadata bez koercji, defaultów ani tolerowania obcych pól."""

        if not isinstance(value, Mapping):
            raise ValueError("StateStoreMetadata input must be a mapping")
        expected = tuple(field.name for field in fields(cls))
        if set(value.keys()) != set(expected):
            raise ValueError("StateStoreMetadata mapping must contain the exact field set")
        return cls(**{name: value[name] for name in expected})


def _validate_record_store_scope(
    record: PersistenceRecord,
    metadata: StateStoreMetadata,
) -> None:
    """Sprawdź wyłącznie lokalną zgodność scope carriera ze StateStore."""

    if record.representation_name == "CryptoHunterAccount current record":
        payload = record.payload
        if not isinstance(payload, Mapping) or payload.get("entity_id") != metadata.account_id:
            raise StateStoreError("CryptoHunterAccount record is outside StateStore scope")
        return
    if record.representation_name == "RuntimeSession canonical identity/history":
        payload = record.payload
        upstream = payload.get("upstream_payload") if isinstance(payload, Mapping) else None
        if (
            not isinstance(upstream, Mapping)
            or upstream.get("device_installation_id") != metadata.device_installation_id
        ):
            raise StateStoreError("RuntimeSession record is outside StateStore scope")
        return

    def at_path(value: object, path: tuple[str | int, ...]) -> object:
        current = value
        for part in path:
            if isinstance(part, int):
                if not isinstance(current, (list, tuple)) or part >= len(current):
                    raise StateStoreError("record scope index is absent")
                current = current[part]
            else:
                if not isinstance(current, Mapping) or part not in current:
                    raise StateStoreError("record scope path is absent")
                current = current[part]
        return current

    binding = STATE_STORE_SCOPE_BINDINGS[record.representation_name]
    if any(
        at_path(record.payload, path) != metadata.account_id for path in binding["account_paths"]
    ):
        raise StateStoreError("record account scope mismatch")
    if any(
        at_path(record.payload, path) != metadata.device_installation_id
        for path in binding["device_paths"]
    ):
        raise StateStoreError("record device scope mismatch")


def _record_bucket(record: PersistenceRecord) -> str:
    try:
        return record_durability_class(record)
    except PersistenceRecordError as exc:
        raise StateStoreError("record representation has no durable bucket") from exc


@dataclass(frozen=True, slots=True)
class StateStoreSnapshot:
    """One generation-pinned logical SQLite snapshot; it carries no authority."""

    metadata: StateStoreMetadata
    current_records: tuple[PersistenceRecord, ...]
    immutable_history: tuple[PersistenceRecord, ...]
    transaction_descriptors: tuple[StateStoreTransactionDescriptor, ...]


class SQLiteStateStore:
    """SQLite-backed, fail-closed durable StateStore kernel."""

    _handles_lock = RLock()
    _open_handles: dict[Path, int] = {}

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 30_000) -> None:
        self._path = Path(path).resolve()
        self._closed = False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Opening and registering use the same path gate as atomic replacement.
        # Therefore no unregistered connection can retain the old inode while a
        # successful replacement publishes a new main file at this pathname.
        with self._handles_lock:
            self._connection = sqlite3.connect(
                self._path, isolation_level=None, timeout=busy_timeout_ms / 1000
            )
            try:
                self._connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
                mode = self._connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                self._connection.execute("PRAGMA synchronous = FULL")
                self._connection.execute("PRAGMA foreign_keys = ON")
                if str(mode).lower() != "wal":
                    raise StateStoreError("SQLite did not enable WAL journal mode")
                self._create_schema()
                self._open_handles[self._path] = self._open_handles.get(self._path, 0) + 1
            except BaseException:
                self._connection.close()
                self._closed = True
                raise

    def _create_schema(self) -> None:
        self._connection.execute("""CREATE TABLE IF NOT EXISTS state_store_metadata (
            singleton_key INTEGER PRIMARY KEY CHECK (singleton_key = 1), account_id TEXT NOT NULL,
            device_installation_id TEXT NOT NULL, state_store_schema_version INTEGER NOT NULL,
            state_store_identity_fingerprint_sha256 TEXT NOT NULL, environment TEXT NOT NULL,
            protected_freshness_generation INTEGER NOT NULL, state_fingerprint_sha256 TEXT NOT NULL,
            transaction_fingerprint_sha256 TEXT NOT NULL, history_tail_fingerprint_sha256 TEXT NOT NULL)""")
        for table in ("state_store_current_records", "state_store_immutable_history"):
            self._connection.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
                record_key TEXT PRIMARY KEY, representation_name TEXT NOT NULL, record_json TEXT NOT NULL)""")
        self._connection.execute("""CREATE TABLE IF NOT EXISTS state_store_transaction_descriptors (
            state_store_identity_fingerprint_sha256 TEXT NOT NULL,
            target_generation INTEGER NOT NULL, descriptor_json TEXT NOT NULL,
            PRIMARY KEY (state_store_identity_fingerprint_sha256, target_generation))""")

    def close(self) -> None:
        if self._closed:
            return
        self._connection.close()
        self._closed = True
        with self._handles_lock:
            remaining = self._open_handles.get(self._path, 1) - 1
            if remaining:
                self._open_handles[self._path] = remaining
            else:
                self._open_handles.pop(self._path, None)

    @property
    def path(self) -> Path:
        """Return the filesystem identity of this store (not an authority reference)."""

        return self._path

    def sqlite_schema_fingerprint(self) -> str:
        """Fingerprint the current physical schema through this store's connection."""

        from .migration_execution import sqlite_schema_fingerprint

        return sqlite_schema_fingerprint(self._connection)

    def capture_verified_physical_snapshot(self, destination: str | Path) -> StateStoreSnapshot:
        """Capture semantic state and its Online Backup image at one read snapshot."""
        target = Path(destination).resolve()
        if target == self._path:
            raise StateStoreError("physical backup destination must differ from live store")
        target.parent.mkdir(parents=True, exist_ok=True)
        destination_connection: sqlite3.Connection | None = None
        try:
            # The first read pins the WAL snapshot; backup() copies that same
            # snapshot while independent writers remain free to commit.
            self._connection.execute("BEGIN")
            snapshot = self._snapshot_inside_transaction()
            if snapshot is None:
                raise StateStoreError("physical backup requires initialized StateStore")
            self.verify_snapshot(snapshot)
            destination_connection = sqlite3.connect(target, isolation_level=None)
            self._connection.backup(destination_connection)
            destination_connection.close()
            destination_connection = None
            self._connection.execute("COMMIT")
            return snapshot
        except BaseException:
            if destination_connection is not None:
                destination_connection.close()
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    @classmethod
    def read_isolated_verified_snapshot(cls, path: str | Path) -> StateStoreSnapshot | None:
        """Read an authenticated candidate without registering or mutating it."""
        snapshot, _ = cls.read_isolated_verified_snapshot_and_schema(path)
        return snapshot

    @classmethod
    def read_isolated_verified_snapshot_and_schema(
        cls, path: str | Path
    ) -> tuple[StateStoreSnapshot | None, str]:
        """Read semantic and physical state through one immutable connection."""
        candidate = Path(path).resolve()
        connection = sqlite3.connect(f"file:{candidate}?mode=ro&immutable=1", uri=True)
        reader = object.__new__(cls)
        reader._path, reader._closed, reader._connection = candidate, False, connection
        try:
            return reader.read_verified_snapshot(), reader.sqlite_schema_fingerprint()
        finally:
            connection.close()
            reader._closed = True

    @classmethod
    @contextmanager
    def installation_gate(cls, live_path: str | Path) -> Iterator[None]:
        """Require path quiescence, then serialize classify/checkpoint/install."""

        target = Path(live_path).resolve()
        with cls._handles_lock:
            if cls._open_handles.get(target, 0):
                raise StateStoreError("live StateStore has pre-existing process-local handles")
            yield

    def write_restored_snapshot(self, snapshot: StateStoreSnapshot) -> None:
        """Write an exact verified snapshot into a new, empty isolated store.

        This deliberately cannot repair or merge an initialized store.  Restore
        orchestration verifies the resulting database through the ordinary read
        path before it can be installed as live.
        """

        if not isinstance(snapshot, StateStoreSnapshot):
            raise StateStoreError("restore input must be a StateStoreSnapshot")
        self.verify_snapshot(snapshot)
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            if self._snapshot_inside_transaction() is not None:
                raise StateStoreError("isolated restore target is not empty")
            self._write_metadata(snapshot.metadata)
            for table, records in (
                ("state_store_current_records", snapshot.current_records),
                ("state_store_immutable_history", snapshot.immutable_history),
            ):
                self._connection.executemany(
                    f"INSERT INTO {table} (record_key, representation_name, record_json) VALUES (?, ?, ?)",
                    [
                        (record.record_key, record.representation_name, self._encode_record(record))
                        for record in records
                    ],
                )
            self._connection.executemany(
                "INSERT INTO state_store_transaction_descriptors (state_store_identity_fingerprint_sha256, target_generation, descriptor_json) VALUES (?, ?, ?)",
                [
                    (
                        descriptor.state_store_identity_fingerprint_sha256,
                        descriptor.target_generation,
                        canonical_json(descriptor.to_mapping()),
                    )
                    for descriptor in snapshot.transaction_descriptors
                ],
            )
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def prepare_for_atomic_install(self) -> None:
        """Checkpoint and close every handle owned by this SQLite store."""

        try:
            row = self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if row is None or row[0] != 0:
                raise StateStoreError("SQLite WAL checkpoint did not complete")
        finally:
            self.close()

    @staticmethod
    def atomic_replace(isolated_path: str | Path, live_path: str | Path) -> None:
        """Atomically replace one closed same-filesystem live database file."""

        source, target = Path(isolated_path).resolve(), Path(live_path).resolve()
        if source.parent.resolve() != target.parent.resolve():
            raise StateStoreError("atomic restore requires a same-directory isolated store")
        with SQLiteStateStore._handles_lock:
            if SQLiteStateStore._open_handles.get(target, 0):
                raise StateStoreError("live StateStore still has an open process-local handle")
            try:
                # For readable stores the caller has already checkpointed and
                # closed SQLite. For corrupt/missing stores these are untrusted
                # orphan artifacts. Removing them under the path gate before
                # publishing the new main eliminates new-main/old-WAL recovery.
                Path(f"{target}-wal").unlink(missing_ok=True)
                Path(f"{target}-shm").unlink(missing_ok=True)
                os.replace(source, target)
            except OSError as exc:
                raise StateStoreError("atomic StateStore replacement failed") from exc

    def __enter__(self) -> SQLiteStateStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def read_metadata(self) -> StateStoreMetadata | None:
        row = self._connection.execute("""SELECT account_id, device_installation_id,
            state_store_schema_version, state_store_identity_fingerprint_sha256, environment,
            protected_freshness_generation, state_fingerprint_sha256,
            transaction_fingerprint_sha256, history_tail_fingerprint_sha256
            FROM state_store_metadata WHERE singleton_key=1""").fetchone()
        if row is None:
            return None
        names = tuple(field.name for field in fields(StateStoreMetadata))
        try:
            return StateStoreMetadata.from_mapping(dict(zip(names, row, strict=True)))
        except (TypeError, ValueError) as exc:
            raise StateStoreError("persisted StateStoreMetadata is malformed") from exc

    def _read_records(
        self, table: str, bucket: str, metadata: StateStoreMetadata | None
    ) -> tuple[PersistenceRecord, ...]:
        if bucket in PERSISTENCE_RECORD_REGISTRY:
            bucket = str(PERSISTENCE_RECORD_REGISTRY[bucket]["durability_class"])
        try:
            rows = self._connection.execute(
                f"SELECT record_key, representation_name, record_json FROM {table}"
            ).fetchall()
            if rows and metadata is None:
                raise StateStoreError("persisted records exist without StateStoreMetadata")
            result = []
            for sql_key, sql_name, encoded in rows:
                record = PersistenceRecord.from_mapping(json.loads(encoded))
                if metadata is None:
                    raise StateStoreError("persisted record has no StateStore scope")
                validate_persistence_record_for_schema(
                    record,
                    state_store_schema_version=metadata.state_store_schema_version,
                )
                if sql_key != record.record_key or sql_name != record.representation_name:
                    raise StateStoreError("persisted record SQL carrier mismatch")
                if _record_bucket(record) != bucket:
                    raise StateStoreError("persisted record is in the wrong storage bucket")
                _validate_record_store_scope(record, metadata)
                result.append(record)
            return tuple(sorted(result, key=lambda r: (r.representation_name, r.record_key)))
        except StateStoreError:
            raise
        except (
            json.JSONDecodeError,
            PersistenceRecordError,
            TypeError,
            ValueError,
            sqlite3.Error,
        ) as exc:
            raise StateStoreError("persisted PersistenceRecord is malformed") from exc

    def _read_descriptors(self) -> tuple[StateStoreTransactionDescriptor, ...]:
        try:
            rows = self._connection.execute(
                "SELECT state_store_identity_fingerprint_sha256, target_generation, descriptor_json FROM state_store_transaction_descriptors"
            ).fetchall()
            result = []
            for sql_identity, sql_generation, encoded in rows:
                descriptor = StateStoreTransactionDescriptor.from_mapping(json.loads(encoded))
                if (
                    sql_identity != descriptor.state_store_identity_fingerprint_sha256
                    or sql_generation != descriptor.target_generation
                ):
                    raise StateStoreError("persisted descriptor SQL carrier mismatch")
                result.append(descriptor)
            return tuple(result)
        except StateStoreError:
            raise
        except (
            json.JSONDecodeError,
            TransactionDescriptorError,
            TypeError,
            ValueError,
            sqlite3.Error,
        ) as exc:
            raise StateStoreError("persisted transaction descriptor is malformed") from exc

    def read_current_records(self) -> tuple[PersistenceRecord, ...]:
        return self._read_records(
            "state_store_current_records", _CURRENT_BUCKET, self.read_metadata()
        )

    def read_immutable_history(self) -> tuple[PersistenceRecord, ...]:
        return self._read_records(
            "state_store_immutable_history", _HISTORY_BUCKET, self.read_metadata()
        )

    def read_transaction_descriptors(self) -> tuple[StateStoreTransactionDescriptor, ...]:
        return self._read_descriptors()

    def _snapshot_inside_transaction(self) -> StateStoreSnapshot | None:
        metadata = self.read_metadata()
        current = self._read_records("state_store_current_records", _CURRENT_BUCKET, metadata)
        history = self._read_records("state_store_immutable_history", _HISTORY_BUCKET, metadata)
        descriptors = self._read_descriptors()
        if metadata is None:
            if current or history or descriptors:
                raise StateStoreError("durable content exists without StateStoreMetadata")
            return None
        return StateStoreSnapshot(metadata, current, history, descriptors)

    def read_snapshot(self) -> StateStoreSnapshot | None:
        """Read all carriers from one SQLite read transaction."""
        try:
            self._connection.execute("BEGIN")
            snapshot = self._snapshot_inside_transaction()
            self._connection.execute("COMMIT")
            return snapshot
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    @staticmethod
    def verify_snapshot(snapshot: StateStoreSnapshot) -> None:
        metadata = snapshot.metadata
        if metadata.state_store_schema_version not in {1, 2}:
            raise StateStoreError("unsupported StateStore schema version")
        for records, bucket in (
            (snapshot.current_records, _CURRENT_BUCKET),
            (snapshot.immutable_history, _HISTORY_BUCKET),
        ):
            for record in records:
                try:
                    validate_record_bucket_for_schema(
                        record,
                        bucket,
                        state_store_schema_version=metadata.state_store_schema_version,
                    )
                except (PersistenceRecordError, TypeError, ValueError) as exc:
                    raise StateStoreError("snapshot record is in the wrong durable bucket") from exc
        for record in (*snapshot.current_records, *snapshot.immutable_history):
            try:
                validate_persistence_record_for_schema(
                    record,
                    state_store_schema_version=metadata.state_store_schema_version,
                )
            except (PersistenceRecordError, TypeError, ValueError) as exc:
                raise StateStoreError("snapshot contains invalid PersistenceRecord") from exc
            _validate_record_store_scope(record, metadata)
        history_hash = history_tail_fingerprint_sha256(snapshot.immutable_history)
        if history_hash != metadata.history_tail_fingerprint_sha256:
            raise StateStoreError("history-tail fingerprint mismatch")
        state_hash = state_fingerprint_sha256(
            account_id=metadata.account_id,
            device_installation_id=metadata.device_installation_id,
            state_store_schema_version=metadata.state_store_schema_version,
            state_store_identity_fingerprint_sha256=metadata.state_store_identity_fingerprint_sha256,
            environment=metadata.environment,
            protected_freshness_generation=metadata.protected_freshness_generation,
            current_records=snapshot.current_records,
            history_tail_fingerprint_sha256=history_hash,
        )
        if state_hash != metadata.state_fingerprint_sha256:
            raise StateStoreError("state fingerprint mismatch")
        descriptors = snapshot.transaction_descriptors
        if len(descriptors) != metadata.protected_freshness_generation:
            raise StateStoreError("descriptor chain cardinality mismatch")
        ordered = sorted(descriptors, key=lambda d: d.target_generation)
        if [d.target_generation for d in ordered] != list(
            range(1, metadata.protected_freshness_generation + 1)
        ):
            raise StateStoreError("descriptor chain must contain exactly generations 1..G")
        for descriptor in ordered:
            if descriptor.account_id != metadata.account_id:
                raise StateStoreError("descriptor account scope mismatch")
            if descriptor.device_installation_id != metadata.device_installation_id:
                raise StateStoreError("descriptor device scope mismatch")
            if (
                descriptor.state_store_identity_fingerprint_sha256
                != metadata.state_store_identity_fingerprint_sha256
            ):
                raise StateStoreError("descriptor StateStore identity mismatch")
            if not descriptor.has_valid_transaction_fingerprint():
                raise StateStoreError("descriptor transaction fingerprint mismatch")
        genesis = ordered[0]
        if (
            genesis.expected_current_generation is not None
            or genesis.pre_state_fingerprint_sha256 is not None
            or genesis.pre_history_tail_fingerprint_sha256 is not None
        ):
            raise StateStoreError("invalid genesis descriptor")
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if (
                current.expected_current_generation != previous.target_generation
                or current.target_generation != previous.target_generation + 1
            ):
                raise StateStoreError("descriptor generation edge mismatch")
            if (
                current.pre_state_fingerprint_sha256 != previous.post_state_fingerprint_sha256
                or current.pre_history_tail_fingerprint_sha256
                != previous.post_history_tail_fingerprint_sha256
            ):
                raise StateStoreError("descriptor fingerprint edge mismatch")
            _validate_schema_version_transition(previous, current)
        current = ordered[-1]
        bindings = {
            "account_id": metadata.account_id,
            "device_installation_id": metadata.device_installation_id,
            "state_store_identity_fingerprint_sha256": metadata.state_store_identity_fingerprint_sha256,
            "state_store_schema_version": metadata.state_store_schema_version,
            "environment": metadata.environment,
            "target_generation": metadata.protected_freshness_generation,
            "post_state_fingerprint_sha256": metadata.state_fingerprint_sha256,
            "post_history_tail_fingerprint_sha256": metadata.history_tail_fingerprint_sha256,
            "transaction_fingerprint_sha256": metadata.transaction_fingerprint_sha256,
        }
        if any(getattr(current, name) != value for name, value in bindings.items()):
            raise StateStoreError("current descriptor does not bind current metadata")

    def read_verified_snapshot(self) -> StateStoreSnapshot | None:
        snapshot = self.read_snapshot()
        if snapshot is not None:
            self.verify_snapshot(snapshot)
        return snapshot

    @staticmethod
    def _select_lifecycle(
        snapshot: StateStoreSnapshot,
        *,
        identity: str,
        current_name: str,
        history_name: str,
        current_key: str,
        history_key_prefix: str,
    ) -> tuple[PersistenceRecord | None, tuple[PersistenceRecord, ...]]:
        current = tuple(
            record
            for record in snapshot.current_records
            if record.representation_name == current_name and record.record_key == current_key
        )
        history = tuple(
            sorted(
                (
                    record
                    for record in snapshot.immutable_history
                    if record.representation_name == history_name
                    and record.record_key.startswith(history_key_prefix)
                ),
                key=lambda record: int(record.record_key.rsplit(":", 1)[1]),
            )
        )
        if len(current) > 1:
            raise StateStoreError(f"duplicate current lifecycle carrier for {identity}")
        return (current[0] if current else None), history

    def read_lifecycle_records(
        self,
        *,
        identity: str,
        current_name: str,
        history_name: str,
        current_key: str,
        history_key_prefix: str,
    ) -> tuple[PersistenceRecord | None, tuple[PersistenceRecord, ...]]:
        """Read one lifecycle only from a fresh, fully verified SQLite snapshot."""

        snapshot = self.read_verified_snapshot()
        if snapshot is None:
            raise StateStoreError("lifecycle read requires initialized StateStore")
        return self._select_lifecycle(
            snapshot,
            identity=identity,
            current_name=current_name,
            history_name=history_name,
            current_key=current_key,
            history_key_prefix=history_key_prefix,
        )

    def commit_prepared_metadata(
        self, metadata: StateStoreMetadata, *, expected_current_generation: int | None
    ) -> None:
        self.commit_prepared_state(
            metadata,
            current_records=(),
            immutable_history=(),
            expected_current_generation=expected_current_generation,
        )

    def derive_prepared_metadata(
        self,
        metadata: StateStoreMetadata,
        *,
        current_records: Iterable[PersistenceRecord] = (),
        immutable_history: Iterable[PersistenceRecord] = (),
        expected_current_generation: int | None,
    ) -> StateStoreMetadata:
        """Derive caller-visible hashes without weakening commit-time verification."""

        current_delta = self._snapshot_records(
            current_records, expected_bucket=_CURRENT_BUCKET, metadata=metadata
        )
        history_delta = self._snapshot_records(
            immutable_history, expected_bucket=_HISTORY_BUCKET, metadata=metadata
        )
        before = self.read_verified_snapshot()
        old_current = () if before is None else before.current_records
        old_history = () if before is None else before.immutable_history
        current_map = {record.record_key: record for record in old_current}
        current_map.update({record.record_key: record for record in current_delta})
        history_map = {record.record_key: record for record in old_history}
        if any(record.record_key in history_map for record in history_delta):
            raise StateStoreError("immutable history record_key already exists")
        history_map.update({record.record_key: record for record in history_delta})
        post_current = tuple(current_map.values())
        post_history_records = tuple(history_map.values())
        post_history = history_tail_fingerprint_sha256(post_history_records)
        post_state = state_fingerprint_sha256(
            account_id=metadata.account_id,
            device_installation_id=metadata.device_installation_id,
            state_store_schema_version=metadata.state_store_schema_version,
            state_store_identity_fingerprint_sha256=metadata.state_store_identity_fingerprint_sha256,
            environment=metadata.environment,
            protected_freshness_generation=metadata.protected_freshness_generation,
            current_records=post_current,
            history_tail_fingerprint_sha256=post_history,
        )
        projection = {
            "account_id": metadata.account_id,
            "device_installation_id": metadata.device_installation_id,
            "state_store_identity_fingerprint_sha256": metadata.state_store_identity_fingerprint_sha256,
            "state_store_schema_version": metadata.state_store_schema_version,
            "environment": metadata.environment,
            "expected_current_generation": expected_current_generation,
            "target_generation": metadata.protected_freshness_generation,
            "pre_state_fingerprint_sha256": None
            if before is None
            else before.metadata.state_fingerprint_sha256,
            "pre_history_tail_fingerprint_sha256": None
            if before is None
            else before.metadata.history_tail_fingerprint_sha256,
            "post_state_fingerprint_sha256": post_state,
            "post_history_tail_fingerprint_sha256": post_history,
            "current_record_mutations": [record.to_mapping() for record in current_delta],
            "immutable_history_appends": [record.to_mapping() for record in history_delta],
        }
        return StateStoreMetadata.from_mapping(
            {
                **metadata.to_mapping(),
                "history_tail_fingerprint_sha256": post_history,
                "state_fingerprint_sha256": post_state,
                "transaction_fingerprint_sha256": transaction_fingerprint_sha256(projection),
            }
        )

    @staticmethod
    def _snapshot_records(
        records: Iterable[PersistenceRecord],
        *,
        expected_bucket: str,
        metadata: StateStoreMetadata,
    ) -> tuple[PersistenceRecord, ...]:
        try:
            snapshot = tuple(records)
        except (TypeError, RuntimeError) as exc:
            raise StateStoreError("record collection must be a finite iterable") from exc
        keys = set()
        for record in snapshot:
            try:
                validate_persistence_record_for_schema(
                    record,
                    state_store_schema_version=metadata.state_store_schema_version,
                )
            except (PersistenceRecordError, TypeError, ValueError) as exc:
                raise StateStoreError("input PersistenceRecord failed Stage 1") from exc
            if _record_bucket(record) != expected_bucket:
                raise StateStoreError("input PersistenceRecord is in the wrong storage bucket")
            if record.record_key in keys:
                raise StateStoreError("duplicate input record_key")
            keys.add(record.record_key)
            _validate_record_store_scope(record, metadata)
        return tuple(sorted(snapshot, key=lambda r: (r.representation_name, r.record_key)))

    @staticmethod
    def _encode_record(record: PersistenceRecord) -> str:
        return canonical_json(record.to_mapping())

    def _write_metadata(self, metadata: StateStoreMetadata) -> None:
        values = metadata.to_mapping()
        columns = tuple(values)
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(f"{name}=excluded.{name}" for name in columns)
        self._connection.execute(
            f"INSERT INTO state_store_metadata (singleton_key,{','.join(columns)}) VALUES (1,{placeholders}) ON CONFLICT(singleton_key) DO UPDATE SET {updates}",
            tuple(values[name] for name in columns),
        )

    def commit_prepared_state(
        self,
        metadata: StateStoreMetadata,
        *,
        current_records: Iterable[PersistenceRecord],
        immutable_history: Iterable[PersistenceRecord],
        expected_current_generation: int | None,
    ) -> None:
        if not isinstance(metadata, StateStoreMetadata):
            raise StateStoreError("metadata must be validated StateStoreMetadata")
        if expected_current_generation is not None:
            try:
                _validate_positive_integer(
                    expected_current_generation, field_name="expected_current_generation"
                )
            except ValueError as exc:
                raise StateStoreError(str(exc)) from exc
        current_delta = self._snapshot_records(
            current_records, expected_bucket=_CURRENT_BUCKET, metadata=metadata
        )
        history_delta = self._snapshot_records(
            immutable_history, expected_bucket=_HISTORY_BUCKET, metadata=metadata
        )
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            before = self._snapshot_inside_transaction()
            if before is None:
                if (
                    expected_current_generation is not None
                    or metadata.protected_freshness_generation != 1
                ):
                    raise StateStoreError(
                        "fresh store requires generation 1 and an empty-store fence"
                    )
                pre_state = pre_history = None
                old_current: tuple[PersistenceRecord, ...] = ()
                old_history: tuple[PersistenceRecord, ...] = ()
            else:
                self.verify_snapshot(before)
                old = before.metadata
                if expected_current_generation != old.protected_freshness_generation:
                    raise StateStoreError("stale expected generation")
                if (
                    metadata.protected_freshness_generation
                    != old.protected_freshness_generation + 1
                ):
                    raise StateStoreError("prepared metadata must advance exactly to G+1")
                for name in _IDENTITY_FIELDS:
                    if getattr(metadata, name) != getattr(old, name):
                        raise StateStoreError(f"immutable store identity changed: {name}")
                if metadata.state_store_schema_version != old.state_store_schema_version:
                    raise StateStoreError("schema-version drift requires a migration engine")
                pre_state, pre_history = (
                    old.state_fingerprint_sha256,
                    old.history_tail_fingerprint_sha256,
                )
                old_current, old_history = before.current_records, before.immutable_history
            current_map = {r.record_key: r for r in old_current}
            current_map.update({r.record_key: r for r in current_delta})
            history_map = {r.record_key: r for r in old_history}
            if any(r.record_key in history_map for r in history_delta):
                raise StateStoreError("immutable history record_key already exists")
            history_map.update({r.record_key: r for r in history_delta})
            expected_current = tuple(
                sorted(current_map.values(), key=lambda r: (r.representation_name, r.record_key))
            )
            expected_history = tuple(
                sorted(history_map.values(), key=lambda r: (r.representation_name, r.record_key))
            )
            post_history = history_tail_fingerprint_sha256(expected_history)
            post_state = state_fingerprint_sha256(
                account_id=metadata.account_id,
                device_installation_id=metadata.device_installation_id,
                state_store_schema_version=metadata.state_store_schema_version,
                state_store_identity_fingerprint_sha256=metadata.state_store_identity_fingerprint_sha256,
                environment=metadata.environment,
                protected_freshness_generation=metadata.protected_freshness_generation,
                current_records=expected_current,
                history_tail_fingerprint_sha256=post_history,
            )
            projection = dict(
                account_id=metadata.account_id,
                device_installation_id=metadata.device_installation_id,
                state_store_identity_fingerprint_sha256=metadata.state_store_identity_fingerprint_sha256,
                state_store_schema_version=metadata.state_store_schema_version,
                environment=metadata.environment,
                expected_current_generation=expected_current_generation,
                target_generation=metadata.protected_freshness_generation,
                pre_state_fingerprint_sha256=pre_state,
                pre_history_tail_fingerprint_sha256=pre_history,
                post_state_fingerprint_sha256=post_state,
                post_history_tail_fingerprint_sha256=post_history,
                current_record_mutations=[r.to_mapping() for r in current_delta],
                immutable_history_appends=[r.to_mapping() for r in history_delta],
            )
            transaction_hash = transaction_fingerprint_sha256(projection)
            if (
                metadata.history_tail_fingerprint_sha256,
                metadata.state_fingerprint_sha256,
                metadata.transaction_fingerprint_sha256,
            ) != (post_history, post_state, transaction_hash):
                raise StateStoreError(
                    "caller metadata fingerprints do not match derived transition"
                )
            descriptor = StateStoreTransactionDescriptor.from_mapping(
                {**projection, "transaction_fingerprint_sha256": transaction_hash}
            )
            for record in current_delta:
                self._connection.execute(
                    "INSERT INTO state_store_current_records(record_key,representation_name,record_json) VALUES(?,?,?) ON CONFLICT(record_key) DO UPDATE SET record_json=excluded.record_json",
                    (record.record_key, record.representation_name, self._encode_record(record)),
                )
            for record in history_delta:
                self._connection.execute(
                    "INSERT INTO state_store_immutable_history VALUES(?,?,?)",
                    (record.record_key, record.representation_name, self._encode_record(record)),
                )
            self._connection.execute(
                "INSERT INTO state_store_transaction_descriptors VALUES(?,?,?)",
                (
                    descriptor.state_store_identity_fingerprint_sha256,
                    descriptor.target_generation,
                    canonical_json(descriptor.to_mapping()),
                ),
            )
            self._write_metadata(metadata)
            after = self._snapshot_inside_transaction()
            expected_descriptors = tuple(
                sorted(
                    (*(() if before is None else before.transaction_descriptors), descriptor),
                    key=lambda item: item.target_generation,
                )
            )
            actual_descriptors = (
                ()
                if after is None
                else tuple(
                    sorted(
                        after.transaction_descriptors,
                        key=lambda item: item.target_generation,
                    )
                )
            )
            if (
                after is None
                or after.metadata != metadata
                or after.current_records != expected_current
                or after.immutable_history != expected_history
                or actual_descriptors != expected_descriptors
            ):
                raise StateStoreError("pre-COMMIT durable POST verification failed")
            self.verify_snapshot(after)
            self._connection.execute("COMMIT")
        except BaseException as exc:
            if self._connection.in_transaction:
                try:
                    self._connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            if isinstance(exc, sqlite3.Error):
                raise StateStoreError("atomic SQLite state commit failed") from exc
            raise

    def _commit_migration_execution(
        self,
        metadata: StateStoreMetadata,
        declaration: Any,
        *,
        expected_current_generation: int,
    ) -> None:
        """Atomically execute one intrinsically validated migration declaration.

        This is deliberately separate from ``commit_prepared_state``: only the
        migration declaration's exact ordered SQL may accompany schema-version
        drift.
        """

        from .migration_execution import (
            MigrationExecutionDeclaration,
            sqlite_schema_fingerprint,
        )

        if not isinstance(metadata, StateStoreMetadata) or not isinstance(
            declaration, MigrationExecutionDeclaration
        ):
            raise StateStoreError("validated migration metadata and declaration required")
        carrier = declaration.carrier()
        history_delta = self._snapshot_records(
            (carrier,), expected_bucket=_HISTORY_BUCKET, metadata=metadata
        )
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            before = self._snapshot_inside_transaction()
            if before is None:
                raise StateStoreError("migration execution requires initialized StateStore")
            self.verify_snapshot(before)
            old = before.metadata
            if expected_current_generation != old.protected_freshness_generation:
                raise StateStoreError("stale expected generation")
            if (
                declaration.expected_current_generation != expected_current_generation
                or declaration.target_generation != expected_current_generation + 1
                or metadata.protected_freshness_generation != declaration.target_generation
            ):
                raise StateStoreError("migration declaration generation fence mismatch")
            if (
                old.state_store_schema_version != declaration.source_schema_version
                or metadata.state_store_schema_version != declaration.target_schema_version
            ):
                raise StateStoreError("migration schema-version fence mismatch")
            bindings = {
                "account_id": declaration.account_id,
                "device_installation_id": declaration.device_installation_id,
                "environment": declaration.environment,
                "state_store_identity_fingerprint_sha256": declaration.state_store_identity_fingerprint_sha256,
            }
            if any(
                getattr(old, name) != value or getattr(metadata, name) != value
                for name, value in bindings.items()
            ):
                raise StateStoreError("migration immutable source binding mismatch")
            if (
                declaration.pre_state_fingerprint_sha256 != old.state_fingerprint_sha256
                or declaration.pre_history_tail_fingerprint_sha256
                != old.history_tail_fingerprint_sha256
            ):
                raise StateStoreError("migration exact source fingerprint fence mismatch")
            if (
                sqlite_schema_fingerprint(self._connection)
                != declaration.pre_sqlite_schema_fingerprint_sha256
            ):
                raise StateStoreError("migration pre-schema fingerprint mismatch")
            if sqlite3.sqlite_version_info < (3, 38, 0):
                raise StateStoreError("migration requires SQLite 3.38.0 or newer")
            try:
                self._connection.execute("SELECT json_extract('{}', '$'), json_set('{}', '$.x', 1)")
            except sqlite3.Error as exc:
                raise StateStoreError("migration requires built-in SQLite JSON functions") from exc
            for operation in declaration.operations:
                self._connection.execute(operation.statement, operation.parameters)
            if (
                sqlite_schema_fingerprint(self._connection)
                != declaration.target_sqlite_schema_fingerprint_sha256
            ):
                raise StateStoreError("migration target-schema fingerprint mismatch")

            post_current = self._read_records(
                "state_store_current_records", _CURRENT_BUCKET, metadata
            )
            post_operation_history = self._read_records(
                "state_store_immutable_history", _HISTORY_BUCKET, metadata
            )
            history_map = {record.record_key: record for record in post_operation_history}
            if carrier.record_key in history_map:
                raise StateStoreError("migration declaration already exists")
            history_map[carrier.record_key] = carrier
            expected_history = tuple(
                sorted(
                    history_map.values(),
                    key=lambda item: (item.representation_name, item.record_key),
                )
            )
            post_history = history_tail_fingerprint_sha256(expected_history)
            post_state = state_fingerprint_sha256(
                account_id=metadata.account_id,
                device_installation_id=metadata.device_installation_id,
                state_store_schema_version=metadata.state_store_schema_version,
                state_store_identity_fingerprint_sha256=metadata.state_store_identity_fingerprint_sha256,
                environment=metadata.environment,
                protected_freshness_generation=metadata.protected_freshness_generation,
                current_records=post_current,
                history_tail_fingerprint_sha256=post_history,
            )
            projection = dict(
                account_id=metadata.account_id,
                device_installation_id=metadata.device_installation_id,
                state_store_identity_fingerprint_sha256=metadata.state_store_identity_fingerprint_sha256,
                state_store_schema_version=metadata.state_store_schema_version,
                environment=metadata.environment,
                expected_current_generation=expected_current_generation,
                target_generation=metadata.protected_freshness_generation,
                pre_state_fingerprint_sha256=old.state_fingerprint_sha256,
                pre_history_tail_fingerprint_sha256=old.history_tail_fingerprint_sha256,
                post_state_fingerprint_sha256=post_state,
                post_history_tail_fingerprint_sha256=post_history,
                current_record_mutations=[],
                immutable_history_appends=[item.to_mapping() for item in history_delta],
            )
            transaction_hash = transaction_fingerprint_sha256(projection)
            if (
                metadata.history_tail_fingerprint_sha256,
                metadata.state_fingerprint_sha256,
                metadata.transaction_fingerprint_sha256,
            ) != (post_history, post_state, transaction_hash):
                raise StateStoreError(
                    "migration candidate fingerprints do not match actual post-operation rows"
                )

            descriptor = StateStoreTransactionDescriptor.from_mapping(
                {**projection, "transaction_fingerprint_sha256": transaction_hash}
            )
            self._connection.execute(
                "INSERT INTO state_store_immutable_history VALUES(?,?,?)",
                (carrier.record_key, carrier.representation_name, self._encode_record(carrier)),
            )
            self._connection.execute(
                "INSERT INTO state_store_transaction_descriptors VALUES(?,?,?)",
                (
                    descriptor.state_store_identity_fingerprint_sha256,
                    descriptor.target_generation,
                    canonical_json(descriptor.to_mapping()),
                ),
            )
            self._write_metadata(metadata)
            after = self._snapshot_inside_transaction()
            if after != StateStoreSnapshot(
                metadata,
                post_current,
                expected_history,
                tuple(
                    sorted(
                        (*before.transaction_descriptors, descriptor),
                        key=lambda item: item.target_generation,
                    )
                ),
            ):
                raise StateStoreError("migration post-write snapshot mismatch")
            self.verify_snapshot(after)
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def _derive_migration_execution_metadata(
        self, target_seed: StateStoreMetadata, declaration: Any
    ) -> StateStoreMetadata:
        """Observe the exact rolled-back post-operation carriers for protected PREPARE."""

        from .migration_execution import MigrationExecutionDeclaration, sqlite_schema_fingerprint

        if not isinstance(declaration, MigrationExecutionDeclaration):
            raise StateStoreError("validated migration declaration required")
        carrier = declaration.carrier()
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            before = self._snapshot_inside_transaction()
            if before is None:
                raise StateStoreError("migration derivation requires initialized StateStore")
            self.verify_snapshot(before)
            if sqlite3.sqlite_version_info < (3, 38, 0):
                raise StateStoreError("migration requires SQLite 3.38.0 or newer")
            try:
                self._connection.execute("SELECT json_extract('{}', '$'), json_set('{}', '$.x', 1)")
            except sqlite3.Error as exc:
                raise StateStoreError("migration requires built-in SQLite JSON functions") from exc
            if (
                sqlite_schema_fingerprint(self._connection)
                != declaration.pre_sqlite_schema_fingerprint_sha256
            ):
                raise StateStoreError("migration pre-schema fingerprint mismatch")
            for operation in declaration.operations:
                self._connection.execute(operation.statement, operation.parameters)
            if (
                sqlite_schema_fingerprint(self._connection)
                != declaration.target_sqlite_schema_fingerprint_sha256
            ):
                raise StateStoreError("migration target-schema fingerprint mismatch")
            current = self._read_records(
                "state_store_current_records", _CURRENT_BUCKET, target_seed
            )
            history = list(
                self._read_records("state_store_immutable_history", _HISTORY_BUCKET, target_seed)
            )
            if any(item.record_key == carrier.record_key for item in history):
                raise StateStoreError("migration declaration already exists")
            history.append(carrier)
            history_tail = history_tail_fingerprint_sha256(history)
            state = state_fingerprint_sha256(
                account_id=target_seed.account_id,
                device_installation_id=target_seed.device_installation_id,
                state_store_schema_version=target_seed.state_store_schema_version,
                state_store_identity_fingerprint_sha256=target_seed.state_store_identity_fingerprint_sha256,
                environment=target_seed.environment,
                protected_freshness_generation=target_seed.protected_freshness_generation,
                current_records=current,
                history_tail_fingerprint_sha256=history_tail,
            )
            projection = dict(
                account_id=target_seed.account_id,
                device_installation_id=target_seed.device_installation_id,
                state_store_identity_fingerprint_sha256=target_seed.state_store_identity_fingerprint_sha256,
                state_store_schema_version=target_seed.state_store_schema_version,
                environment=target_seed.environment,
                expected_current_generation=declaration.expected_current_generation,
                target_generation=target_seed.protected_freshness_generation,
                pre_state_fingerprint_sha256=before.metadata.state_fingerprint_sha256,
                pre_history_tail_fingerprint_sha256=before.metadata.history_tail_fingerprint_sha256,
                post_state_fingerprint_sha256=state,
                post_history_tail_fingerprint_sha256=history_tail,
                current_record_mutations=[],
                immutable_history_appends=[carrier.to_mapping()],
            )
            return StateStoreMetadata.from_mapping(
                {
                    **target_seed.to_mapping(),
                    "history_tail_fingerprint_sha256": history_tail,
                    "state_fingerprint_sha256": state,
                    "transaction_fingerprint_sha256": transaction_fingerprint_sha256(projection),
                }
            )
        finally:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")


__all__ = [
    "SQLiteStateStore",
    "StateStoreError",
    "StateStoreMetadata",
    "StateStoreSnapshot",
]
