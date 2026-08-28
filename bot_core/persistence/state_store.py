"""Lokalny, trwały kernel metadanych StateStore zgodny z M0.11."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from bot_core.persistence.records import (
    PersistenceRecord,
    PersistenceRecordError,
    validate_persistence_record,
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
_CURRENT_REPRESENTATION = "CryptoHunterAccount current record"
_HISTORY_REPRESENTATION = "RuntimeSession canonical identity/history"


class StateStoreError(RuntimeError):
    """Błąd zamkniętej walidacji lub ogrodzenia trwałego StateStore."""


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

    if record.representation_name == _CURRENT_REPRESENTATION:
        payload = record.payload
        if not isinstance(payload, Mapping) or payload.get("entity_id") != metadata.account_id:
            raise StateStoreError("CryptoHunterAccount record is outside StateStore scope")
        return
    if record.representation_name == _HISTORY_REPRESENTATION:
        payload = record.payload
        upstream = payload.get("upstream_payload") if isinstance(payload, Mapping) else None
        if (
            not isinstance(upstream, Mapping)
            or upstream.get("device_installation_id") != metadata.device_installation_id
        ):
            raise StateStoreError("RuntimeSession record is outside StateStore scope")
        return
    raise StateStoreError("record representation has no supported StateStore scope binding")


class SQLiteStateStore:
    """SQLite-backed kernel przygotowanych metadata i rekordów Stage 1."""

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 30_000) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self._path,
            isolation_level=None,
            timeout=busy_timeout_ms / 1000,
        )
        try:
            self._connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
            journal_mode = self._connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.execute("PRAGMA foreign_keys = ON")
            if str(journal_mode).lower() != "wal":
                raise StateStoreError("SQLite did not enable WAL journal mode")
            self._create_schema()
        except BaseException:
            self._connection.close()
            raise

    def _create_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS state_store_metadata (
                singleton_key INTEGER PRIMARY KEY CHECK (singleton_key = 1),
                account_id TEXT NOT NULL,
                device_installation_id TEXT NOT NULL,
                state_store_schema_version INTEGER NOT NULL,
                state_store_identity_fingerprint_sha256 TEXT NOT NULL,
                environment TEXT NOT NULL,
                protected_freshness_generation INTEGER NOT NULL,
                state_fingerprint_sha256 TEXT NOT NULL,
                transaction_fingerprint_sha256 TEXT NOT NULL,
                history_tail_fingerprint_sha256 TEXT NOT NULL
            )
            """
        )
        for table_name in (
            "state_store_current_records",
            "state_store_immutable_history",
        ):
            self._connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    record_key TEXT PRIMARY KEY,
                    representation_name TEXT NOT NULL,
                    record_json TEXT NOT NULL
                )
                """
            )

    def close(self) -> None:
        """Zamknij lokalne połączenie SQLite."""

        self._connection.close()

    def __enter__(self) -> SQLiteStateStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def read_metadata(self) -> StateStoreMetadata | None:
        """Odczytaj i ponownie zwaliduj cały trwały rekord metadata."""

        row = self._connection.execute(
            """
            SELECT account_id, device_installation_id, state_store_schema_version,
                   state_store_identity_fingerprint_sha256, environment,
                   protected_freshness_generation, state_fingerprint_sha256,
                   transaction_fingerprint_sha256, history_tail_fingerprint_sha256
            FROM state_store_metadata WHERE singleton_key = 1
            """
        ).fetchone()
        if row is None:
            return None
        names = tuple(field.name for field in fields(StateStoreMetadata))
        try:
            return StateStoreMetadata.from_mapping(dict(zip(names, row, strict=True)))
        except (TypeError, ValueError) as exc:
            raise StateStoreError("persisted StateStoreMetadata is malformed") from exc

    def _read_records(
        self,
        table_name: str,
        expected_representation: str,
        metadata: StateStoreMetadata | None,
    ) -> tuple[PersistenceRecord, ...]:
        try:
            rows = self._connection.execute(
                f"""
                SELECT record_key, representation_name, record_json
                FROM {table_name} ORDER BY record_key
                """
            ).fetchall()
            if rows and metadata is None:
                raise StateStoreError("persisted records exist without StateStoreMetadata")
            records: list[PersistenceRecord] = []
            for sql_key, sql_representation, record_json in rows:
                mapping = json.loads(record_json)
                record = PersistenceRecord.from_mapping(mapping)
                validate_persistence_record(record)
                if sql_key != record.record_key:
                    raise StateStoreError("persisted record_key does not match its carrier")
                if sql_representation != record.representation_name:
                    raise StateStoreError(
                        "persisted representation_name does not match its carrier"
                    )
                if record.representation_name != expected_representation:
                    raise StateStoreError("persisted record is in the wrong storage bucket")
                if metadata is None:
                    raise StateStoreError("persisted record has no StateStore scope")
                _validate_record_store_scope(record, metadata)
                records.append(record)
            return tuple(records)
        except StateStoreError:
            raise
        except (json.JSONDecodeError, PersistenceRecordError, TypeError, ValueError) as exc:
            raise StateStoreError("persisted PersistenceRecord is malformed") from exc
        except sqlite3.Error as exc:
            raise StateStoreError("SQLite record read failed") from exc

    def read_current_records(self) -> tuple[PersistenceRecord, ...]:
        """Odczytaj deterministycznie pełny, zwalidowany current-state bucket."""

        return self._read_records(
            "state_store_current_records",
            _CURRENT_REPRESENTATION,
            self.read_metadata(),
        )

    def read_immutable_history(self) -> tuple[PersistenceRecord, ...]:
        """Odczytaj deterministycznie pełny, zwalidowany append-only bucket."""

        return self._read_records(
            "state_store_immutable_history",
            _HISTORY_REPRESENTATION,
            self.read_metadata(),
        )

    def commit_prepared_metadata(
        self,
        metadata: StateStoreMetadata,
        *,
        expected_current_generation: int | None,
    ) -> None:
        """Atomowo utrwal przygotowane metadata po sprawdzeniu ogrodzeń S1."""

        self.commit_prepared_state(
            metadata,
            current_records=(),
            immutable_history=(),
            expected_current_generation=expected_current_generation,
        )

    @staticmethod
    def _snapshot_records(
        records: Iterable[PersistenceRecord],
        *,
        expected_representation: str,
        metadata: StateStoreMetadata,
    ) -> tuple[PersistenceRecord, ...]:
        try:
            snapshot = tuple(records)
        except (TypeError, RuntimeError) as exc:
            raise StateStoreError("record collection must be a finite iterable") from exc
        keys: set[str] = set()
        for record in snapshot:
            try:
                validate_persistence_record(record)
            except (PersistenceRecordError, TypeError, ValueError) as exc:
                raise StateStoreError("input PersistenceRecord failed Stage 1") from exc
            if record.representation_name != expected_representation:
                raise StateStoreError("input PersistenceRecord is in the wrong storage bucket")
            if record.record_key in keys:
                raise StateStoreError("duplicate input record_key")
            keys.add(record.record_key)
            _validate_record_store_scope(record, metadata)
        return snapshot

    @staticmethod
    def _encode_record(record: PersistenceRecord) -> str:
        try:
            return json.dumps(
                record.to_mapping(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise StateStoreError("PersistenceRecord storage encoding failed") from exc

    def _validate_metadata_fences(
        self,
        metadata: StateStoreMetadata,
        expected_current_generation: int | None,
    ) -> StateStoreMetadata | None:
        current = self.read_metadata()
        if current is None:
            if expected_current_generation is not None:
                raise StateStoreError("fresh store requires an explicit empty-store fence")
            return None
        if expected_current_generation != current.protected_freshness_generation:
            raise StateStoreError("stale expected generation")
        if metadata.protected_freshness_generation != expected_current_generation + 1:
            raise StateStoreError("prepared metadata must advance exactly to G+1")
        for field_name in _IDENTITY_FIELDS:
            if getattr(metadata, field_name) != getattr(current, field_name):
                raise StateStoreError(f"immutable store identity changed: {field_name}")
        if metadata.state_store_schema_version != current.state_store_schema_version:
            raise StateStoreError("schema-version drift requires a migration engine")
        return current

    def _write_metadata(self, metadata: StateStoreMetadata) -> None:
        values = metadata.to_mapping()
        columns = tuple(values)
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(f"{name} = excluded.{name}" for name in columns)
        self._connection.execute(
            f"""
            INSERT INTO state_store_metadata (singleton_key, {", ".join(columns)})
            VALUES (1, {placeholders})
            ON CONFLICT(singleton_key) DO UPDATE SET {updates}
            """,
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
        """Atomowo utrwal caller-prepared metadata, current state i historię."""

        if not isinstance(metadata, StateStoreMetadata):
            raise StateStoreError("metadata must be validated StateStoreMetadata")
        if expected_current_generation is not None:
            try:
                _validate_positive_integer(
                    expected_current_generation,
                    field_name="expected_current_generation",
                )
            except ValueError as exc:
                raise StateStoreError(str(exc)) from exc

        current_snapshot = self._snapshot_records(
            current_records,
            expected_representation=_CURRENT_REPRESENTATION,
            metadata=metadata,
        )
        history_snapshot = self._snapshot_records(
            immutable_history,
            expected_representation=_HISTORY_REPRESENTATION,
            metadata=metadata,
        )

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            current_metadata = self._validate_metadata_fences(metadata, expected_current_generation)
            self._read_records(
                "state_store_current_records",
                _CURRENT_REPRESENTATION,
                current_metadata,
            )
            self._read_records(
                "state_store_immutable_history",
                _HISTORY_REPRESENTATION,
                current_metadata,
            )

            for record in current_snapshot:
                existing = self._connection.execute(
                    """
                    SELECT representation_name FROM state_store_current_records
                    WHERE record_key = ?
                    """,
                    (record.record_key,),
                ).fetchone()
                if existing is not None and existing[0] != record.representation_name:
                    raise StateStoreError("current record_key representation collision")
                encoded = self._encode_record(record)
                if existing is None:
                    self._connection.execute(
                        """
                        INSERT INTO state_store_current_records
                            (record_key, representation_name, record_json)
                        VALUES (?, ?, ?)
                        """,
                        (record.record_key, record.representation_name, encoded),
                    )
                else:
                    self._connection.execute(
                        """
                        UPDATE state_store_current_records SET record_json = ?
                        WHERE record_key = ?
                        """,
                        (encoded, record.record_key),
                    )

            for record in history_snapshot:
                if (
                    self._connection.execute(
                        "SELECT 1 FROM state_store_immutable_history WHERE record_key = ?",
                        (record.record_key,),
                    ).fetchone()
                    is not None
                ):
                    raise StateStoreError("immutable history record_key already exists")
                self._connection.execute(
                    """
                    INSERT INTO state_store_immutable_history
                        (record_key, representation_name, record_json)
                    VALUES (?, ?, ?)
                    """,
                    (
                        record.record_key,
                        record.representation_name,
                        self._encode_record(record),
                    ),
                )

            self._write_metadata(metadata)
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


__all__ = ["SQLiteStateStore", "StateStoreError", "StateStoreMetadata"]
