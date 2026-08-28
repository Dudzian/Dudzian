"""Storage-neutral canonical JSON fingerprints for the persistence kernel."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .records import PersistenceRecord


def canonical_json(value: object) -> str:
    """Encode a JSON-domain value using the single production canonical kernel."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_json_sha256(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_records(records: Sequence[PersistenceRecord]) -> list[dict[str, object]]:
    return [
        record.to_mapping()
        for record in sorted(records, key=lambda item: (item.representation_name, item.record_key))
    ]


def history_tail_fingerprint_sha256(records: Sequence[PersistenceRecord]) -> str:
    return canonical_json_sha256(canonical_records(records))


def state_fingerprint_sha256(
    *,
    account_id: str,
    device_installation_id: str,
    state_store_schema_version: int,
    state_store_identity_fingerprint_sha256: str,
    environment: str,
    protected_freshness_generation: int,
    current_records: Sequence[PersistenceRecord],
    history_tail_fingerprint_sha256: str,
) -> str:
    return canonical_json_sha256(
        {
            "account_id": account_id,
            "device_installation_id": device_installation_id,
            "state_store_schema_version": state_store_schema_version,
            "state_store_identity_fingerprint_sha256": state_store_identity_fingerprint_sha256,
            "environment": environment,
            "protected_freshness_generation": protected_freshness_generation,
            "canonical_durable_current_records": canonical_records(current_records),
            "history_tail_fingerprint_sha256": history_tail_fingerprint_sha256,
        }
    )


def transaction_fingerprint_sha256(value: Mapping[str, Any] | object) -> str:
    mapping = value.to_fingerprint_mapping() if hasattr(value, "to_fingerprint_mapping") else value
    return canonical_json_sha256(mapping)
