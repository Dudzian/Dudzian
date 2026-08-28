"""Lokalny, trwały kernel metadanych StateStore zgodny z M0.11."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

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


class SQLiteStateStore:
    """SQLite-backed kernel przechowujący wyłącznie przygotowane metadata."""

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

    def commit_prepared_metadata(
        self,
        metadata: StateStoreMetadata,
        *,
        expected_current_generation: int | None,
    ) -> None:
        """Atomowo utrwal przygotowane metadata po sprawdzeniu ogrodzeń S1."""

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

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = self.read_metadata()
            if current is None:
                if expected_current_generation is not None:
                    raise StateStoreError("fresh store requires an explicit empty-store fence")
            else:
                if expected_current_generation != current.protected_freshness_generation:
                    raise StateStoreError("stale expected generation")
                if metadata.protected_freshness_generation != expected_current_generation + 1:
                    raise StateStoreError("prepared metadata must advance exactly to G+1")
                for field_name in _IDENTITY_FIELDS:
                    if getattr(metadata, field_name) != getattr(current, field_name):
                        raise StateStoreError(f"immutable store identity changed: {field_name}")
                if metadata.state_store_schema_version != current.state_store_schema_version:
                    raise StateStoreError("schema-version drift requires a migration engine")

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
            self._connection.execute("COMMIT")
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise


__all__ = ["SQLiteStateStore", "StateStoreError", "StateStoreMetadata"]
