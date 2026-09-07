"""Immutable declaration of one exact durable StateStore transition."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from .fingerprints import canonical_records, transaction_fingerprint_sha256
from .records import (
    PersistenceRecord,
    PersistenceRecordError,
    validate_record_bucket_for_schema,
)

_CURRENT_BUCKET = "DURABLE AUTHORITATIVE CURRENT STATE"
_HISTORY_BUCKET = "DURABLE IMMUTABLE / APPEND-ONLY HISTORY"

_ID = re.compile(
    r"^(?P<prefix>[a-z][a-z0-9]*)_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SHA = re.compile(r"^[0-9a-f]{64}$")


class TransactionDescriptorError(ValueError):
    """An intrinsic descriptor contract violation."""


def _positive(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TransactionDescriptorError(f"{name} must be a positive non-boolean integer")


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise TransactionDescriptorError(f"{name} must be lowercase SHA-256 hexadecimal")


@dataclass(frozen=True, slots=True)
class StateStoreTransactionDescriptor:
    account_id: str
    device_installation_id: str
    state_store_identity_fingerprint_sha256: str
    state_store_schema_version: int
    environment: str
    expected_current_generation: int | None
    target_generation: int
    pre_state_fingerprint_sha256: str | None
    pre_history_tail_fingerprint_sha256: str | None
    post_state_fingerprint_sha256: str
    post_history_tail_fingerprint_sha256: str
    current_record_mutations: tuple[PersistenceRecord, ...]
    immutable_history_appends: tuple[PersistenceRecord, ...]
    transaction_fingerprint_sha256: str

    def __post_init__(self) -> None:
        for name, prefix in (("account_id", "acct"), ("device_installation_id", "dev")):
            value = getattr(self, name)
            match = _ID.fullmatch(value) if isinstance(value, str) else None
            if match is None or match.group("prefix") != prefix:
                raise TransactionDescriptorError(f"{name} must be a canonical {prefix} UUIDv7 ID")
        _sha(
            self.state_store_identity_fingerprint_sha256, "state_store_identity_fingerprint_sha256"
        )
        _positive(self.state_store_schema_version, "state_store_schema_version")
        if self.state_store_schema_version not in {1, 2}:
            raise TransactionDescriptorError("unsupported StateStore schema version")
        _positive(self.target_generation, "target_generation")
        if self.environment not in {"PAPER", "TESTNET", "LIVE"}:
            raise TransactionDescriptorError("environment must be PAPER, TESTNET, or LIVE")
        if self.target_generation == 1:
            if any(
                value is not None
                for value in (
                    self.expected_current_generation,
                    self.pre_state_fingerprint_sha256,
                    self.pre_history_tail_fingerprint_sha256,
                )
            ):
                raise TransactionDescriptorError("genesis pre-state fields must be null")
        else:
            _positive(self.expected_current_generation, "expected_current_generation")
            _sha(self.pre_state_fingerprint_sha256, "pre_state_fingerprint_sha256")
            _sha(self.pre_history_tail_fingerprint_sha256, "pre_history_tail_fingerprint_sha256")
        for name in (
            "post_state_fingerprint_sha256",
            "post_history_tail_fingerprint_sha256",
            "transaction_fingerprint_sha256",
        ):
            _sha(getattr(self, name), name)
        for name, bucket in (
            ("current_record_mutations", _CURRENT_BUCKET),
            ("immutable_history_appends", _HISTORY_BUCKET),
        ):
            records = getattr(self, name)
            if not isinstance(records, tuple):
                raise TransactionDescriptorError(f"{name} must be an immutable tuple")
            try:
                for record in records:
                    validate_record_bucket_for_schema(
                        record,
                        bucket,
                        state_store_schema_version=self.state_store_schema_version,
                    )
            except (PersistenceRecordError, TypeError, ValueError) as exc:
                raise TransactionDescriptorError(f"{name} contains an invalid record") from exc
            if list(records) != sorted(
                records, key=lambda item: (item.representation_name, item.record_key)
            ):
                raise TransactionDescriptorError(f"{name} must already be canonically sorted")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> StateStoreTransactionDescriptor:
        expected = tuple(field.name for field in fields(cls))
        if not isinstance(value, Mapping) or set(value) != set(expected):
            raise TransactionDescriptorError("descriptor requires its exact 14-field set")
        parsed = dict(value)
        for name in ("current_record_mutations", "immutable_history_appends"):
            raw = parsed[name]
            if not isinstance(raw, list):
                raise TransactionDescriptorError(f"{name} must be a JSON array")
            try:
                parsed[name] = tuple(PersistenceRecord.from_mapping(item) for item in raw)
            except (PersistenceRecordError, TypeError, ValueError) as exc:
                raise TransactionDescriptorError(f"{name} contains malformed records") from exc
        return cls(**parsed)

    def to_mapping(self) -> dict[str, object]:
        result = self.to_fingerprint_mapping()
        result["transaction_fingerprint_sha256"] = self.transaction_fingerprint_sha256
        return result

    def to_fingerprint_mapping(self) -> dict[str, object]:
        return {
            "account_id": self.account_id,
            "device_installation_id": self.device_installation_id,
            "state_store_identity_fingerprint_sha256": self.state_store_identity_fingerprint_sha256,
            "state_store_schema_version": self.state_store_schema_version,
            "environment": self.environment,
            "expected_current_generation": self.expected_current_generation,
            "target_generation": self.target_generation,
            "pre_state_fingerprint_sha256": self.pre_state_fingerprint_sha256,
            "pre_history_tail_fingerprint_sha256": self.pre_history_tail_fingerprint_sha256,
            "post_state_fingerprint_sha256": self.post_state_fingerprint_sha256,
            "post_history_tail_fingerprint_sha256": self.post_history_tail_fingerprint_sha256,
            "current_record_mutations": canonical_records(self.current_record_mutations),
            "immutable_history_appends": canonical_records(self.immutable_history_appends),
        }

    def has_valid_transaction_fingerprint(self) -> bool:
        return self.transaction_fingerprint_sha256 == transaction_fingerprint_sha256(self)
