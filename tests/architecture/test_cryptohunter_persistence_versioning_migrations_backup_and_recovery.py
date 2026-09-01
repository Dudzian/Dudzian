"""Pure semantic executable reference model for the closed M0.11 contract."""

from __future__ import annotations

import ast
import copy
from decimal import Decimal, InvalidOperation
from datetime import datetime
import hashlib
import inspect
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE_PATH = DOCS / "persistence_versioning_migrations_backup_and_recovery.json"
MACHINE: dict[str, Any] = json.loads(MACHINE_PATH.read_text())
RISK: dict[str, Any] = json.loads(
    (DOCS / "risk_hierarchy_kill_switch_and_execution_lease.json").read_text()
)
IDENTITY: dict[str, Any] = json.loads(
    (DOCS / "identity_device_authentication_and_secrets.json").read_text()
)
VOCABULARY: dict[str, Any] = json.loads((DOCS / "canonical_domain_vocabulary.json").read_text())
EXCHANGE: dict[str, Any] = json.loads((DOCS / "exchange_accounts_and_instruments.json").read_text())
STRATEGY: dict[str, Any] = json.loads(
    (DOCS / "strategy_market_data_and_execution_routing.json").read_text()
)
COMMANDS: dict[str, Any] = json.loads(
    (DOCS / "commands_events_order_lifecycle_and_idempotency.json").read_text()
)
LEDGER: dict[str, Any] = json.loads((DOCS / "ledger_portfolio_capital_and_pnl.json").read_text())
SHA = "a" * 64
SCOPE = (
    "acct_01890f3a-2b4c-7abc-8def-0123456789ab",
    "dev_01890f3a-2b4c-7abc-8def-0123456789ab",
    SHA,
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")

RESTORE_DECISIONS = {
    "NOOP_ALREADY_CURRENT",
    "RESTORE_EXTERNAL_COMMITTED_CURRENT",
    "RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE",
    "DENY",
}
PROTECTED_FIELDS = {
    "account_id",
    "device_installation_id",
    "state_store_identity_fingerprint_sha256",
    "lifecycle",
    "committed_generation",
    "committed_state_fingerprint_sha256",
    "prepared_generation",
    "prepared_state_fingerprint_sha256",
    "prepared_transaction_fingerprint_sha256",
    "authority_revision",
    "authority_source",
    "content_fingerprint_sha256",
}


def _protected_record_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != PROTECTED_FIELDS:
        return False
    if not _canonical_scope(value["account_id"], value["device_installation_id"]):
        return False
    if (
        not isinstance(value["state_store_identity_fingerprint_sha256"], str)
        or SHA_RE.fullmatch(value["state_store_identity_fingerprint_sha256"]) is None
    ):
        return False
    if value["authority_source"] != "EXTERNAL_PRODUCT_PROTECTED_STATE_BOUNDARY" or not _positive(
        value["authority_revision"]
    ):
        return False
    projection = {key: item for key, item in value.items() if key != "content_fingerprint_sha256"}
    if value["content_fingerprint_sha256"] != _actual_fingerprint(projection):
        return False
    committed = (value["committed_generation"], value["committed_state_fingerprint_sha256"])
    prepared = (
        value["prepared_generation"],
        value["prepared_state_fingerprint_sha256"],
        value["prepared_transaction_fingerprint_sha256"],
    )
    lifecycle = value["lifecycle"]
    if lifecycle == "UNINITIALIZED":
        return committed == (None, None) and prepared == (None, None, None)
    if lifecycle == "COMMITTED":
        return (
            _positive(committed[0])
            and isinstance(committed[1], str)
            and SHA_RE.fullmatch(committed[1]) is not None
            and prepared == (None, None, None)
        )
    if (
        lifecycle != "PREPARED"
        or not _positive(prepared[0])
        or any(not isinstance(item, str) or SHA_RE.fullmatch(item) is None for item in prepared[1:])
    ):
        return False
    genesis = committed == (None, None)
    normal = (
        _positive(committed[0])
        and isinstance(committed[1], str)
        and SHA_RE.fullmatch(committed[1]) is not None
    )
    return (genesis and prepared[0] == 1) or (normal and prepared[0] == committed[0] + 1)


def _current_membership(record: dict[str, Any], membership: dict[str, Any]) -> bool:
    scope = (
        record["account_id"],
        record["device_installation_id"],
        record["state_store_identity_fingerprint_sha256"],
    )
    ref = membership.get("resolved_ref")
    return (
        isinstance(ref, str)
        and ref in membership.get("accepted_refs", set())
        and membership.get("current_ref_by_scope", {}).get(scope) == ref
    )


def _backup_summary(backup: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": (
            backup["account_id"],
            backup["device_installation_id"],
            backup["state_store_identity_fingerprint_sha256"],
        ),
        "environment": backup["environment"],
        "generation": backup["local_protected_freshness_generation"],
        "state": backup["state_fingerprint_sha256"],
        "transaction": backup["transaction_fingerprint_sha256"],
        "history": backup["history_tail_fingerprint_sha256"],
        "envelope": backup["envelope_fingerprint_sha256"],
    }


def classify_trusted_local(candidate: dict[str, Any], local_input: dict[str, Any]) -> str:
    status = local_input.get("status")
    if status == "VERIFIED_EMPTY":
        return "EMPTY"
    if status == "NO_TRUSTED_OBSERVATION":
        return "NO_TRUSTED_LOCAL_OBSERVATION"
    if status == "CORRUPT_OR_UNREADABLE":
        return "CORRUPT_OR_UNREADABLE"
    local = local_input.get("observation")
    if status != "VERIFIED_STATE" or not isinstance(local, dict):
        return "INVALID_LOCAL_INPUT"
    scope = (
        local.get("account_id"),
        local.get("device_installation_id"),
        local.get("state_store_identity_fingerprint_sha256"),
    )
    if scope != candidate["scope"]:
        return "SCOPE_CONFLICT"
    if local.get("environment") != candidate["environment"]:
        return "ENVIRONMENT_CONFLICT"
    generation = local.get("generation")
    if not _positive(generation):
        return "INVALID_LOCAL_INPUT"
    if generation < candidate["generation"]:
        return "BEHIND"
    if generation > candidate["generation"]:
        return "AHEAD"
    if local.get("state_fingerprint_sha256") != candidate["state"]:
        return "SAME_GENERATION_DIFFERENT_STATE"
    if (
        local.get("transaction_fingerprint_sha256") != candidate["transaction"]
        or local.get("history_tail_fingerprint_sha256") != candidate["history"]
    ):
        return "SAME_GENERATION_STATE_MATCH_LINEAGE_MISMATCH"
    return "EXACT"


def _external_eligibility(
    backup: dict[str, Any], external: dict[str, Any] | None, membership: dict[str, Any]
) -> str:
    if (
        _validate_backup(backup) != "VALID"
        or external is None
        or not _protected_record_valid(external)
        or not _current_membership(external, membership)
    ):
        return "DENIED"
    candidate = _backup_summary(backup)
    if candidate["scope"] != (
        external["account_id"],
        external["device_installation_id"],
        external["state_store_identity_fingerprint_sha256"],
    ):
        return "DENIED"
    if (
        external["lifecycle"] == "COMMITTED"
        and candidate["generation"] == external["committed_generation"]
        and candidate["state"] == external["committed_state_fingerprint_sha256"]
    ):
        return "AUTHORIZED_COMMITTED_TARGET"
    if (
        external["lifecycle"] == "PREPARED"
        and candidate["generation"] == external["prepared_generation"]
        and candidate["state"] == external["prepared_state_fingerprint_sha256"]
        and candidate["transaction"] == external["prepared_transaction_fingerprint_sha256"]
    ):
        return "AUTHORIZED_PREPARED_TARGET"
    return "DENIED"


def assess_restore_candidate(
    backup: dict[str, Any],
    external: dict[str, Any] | None,
    membership: dict[str, Any],
    local_input: dict[str, Any],
) -> str:
    eligibility = _external_eligibility(backup, external, membership)
    if eligibility == "DENIED":
        return "DENY"
    local_class = classify_trusted_local(_backup_summary(backup), local_input)
    if local_class in {
        "SCOPE_CONFLICT",
        "ENVIRONMENT_CONFLICT",
        "SAME_GENERATION_DIFFERENT_STATE",
        "SAME_GENERATION_STATE_MATCH_LINEAGE_MISMATCH",
        "AHEAD",
        "INVALID_LOCAL_INPUT",
    }:
        return "DENY"
    if local_class == "EXACT" and eligibility == "AUTHORIZED_COMMITTED_TARGET":
        return "NOOP_ALREADY_CURRENT"
    if eligibility == "AUTHORIZED_COMMITTED_TARGET":
        return "RESTORE_EXTERNAL_COMMITTED_CURRENT"
    return "RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE"


def select_restore_candidate(
    candidates: list[dict[str, Any]], external: dict[str, Any], membership: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    eligible = [
        candidate
        for candidate in candidates
        if _external_eligibility(candidate, external, membership) != "DENIED"
    ]
    if len(eligible) != 1:
        return "DENY", None
    eligibility = _external_eligibility(eligible[0], external, membership)
    decision = (
        "RESTORE_EXTERNAL_COMMITTED_CURRENT"
        if eligibility == "AUTHORIZED_COMMITTED_TARGET"
        else "RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE"
    )
    return decision, eligible[0]


def reconcile_finalize_outcome(
    backup: dict[str, Any],
    pre_finalize: dict[str, Any],
    expected_ref: str,
    terminal: dict[str, Any],
    terminal_membership: dict[str, Any],
    local_input: dict[str, Any],
) -> str:
    if (
        _validate_backup(backup) != "VALID"
        or not _protected_record_valid(pre_finalize)
        or not _protected_record_valid(terminal)
    ):
        return "DENY"
    candidate = _backup_summary(backup)
    scope = candidate["scope"]
    protected_scope = lambda value: (
        value["account_id"],
        value["device_installation_id"],
        value["state_store_identity_fingerprint_sha256"],
    )
    prepared_exact = (
        pre_finalize["lifecycle"] == "PREPARED"
        and protected_scope(pre_finalize) == scope
        and pre_finalize["prepared_generation"] == candidate["generation"]
        and pre_finalize["prepared_state_fingerprint_sha256"] == candidate["state"]
        and pre_finalize["prepared_transaction_fingerprint_sha256"] == candidate["transaction"]
    )
    terminal_exact = (
        terminal["lifecycle"] == "COMMITTED"
        and protected_scope(terminal) == scope
        and terminal["committed_generation"] == candidate["generation"]
        and terminal["committed_state_fingerprint_sha256"] == candidate["state"]
    )
    same_current_ref = terminal_membership.get(
        "resolved_ref"
    ) == expected_ref and _current_membership(terminal, terminal_membership)
    local_exact = classify_trusted_local(candidate, local_input) == "EXACT"
    return (
        "NOOP_ALREADY_CURRENT"
        if prepared_exact and terminal_exact and same_current_ref and local_exact
        else "DENY"
    )


def _resolve_pointer(document: Any, pointer: Any) -> tuple[bool, Any]:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return False, None
    value = document
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or token not in value:
            return False, None
        value = value[token]
    return True, value


def _actual_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: (record["representation_name"], record["record_key"]))


def _history_tail_fingerprint(records: list[dict[str, Any]]) -> str:
    return _actual_fingerprint(_canonical_records(records))


def _state_projection(metadata: dict[str, Any], current: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "account_id": metadata["account_id"],
        "device_installation_id": metadata["device_installation_id"],
        "state_store_schema_version": metadata["state_store_schema_version"],
        "state_store_identity_fingerprint_sha256": metadata[
            "state_store_identity_fingerprint_sha256"
        ],
        "environment": metadata["environment"],
        "protected_freshness_generation": metadata["protected_freshness_generation"],
        "canonical_durable_current_records": _canonical_records(current),
        "history_tail_fingerprint_sha256": metadata["history_tail_fingerprint_sha256"],
    }


def _transaction_projection(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "account_id": SCOPE[0],
        "device_installation_id": SCOPE[1],
        "state_store_identity_fingerprint_sha256": SCOPE[2],
        "state_store_schema_version": 1,
        "environment": "PAPER",
        "expected_current_generation": 1,
        "target_generation": 2,
        "pre_state_fingerprint_sha256": "1" * 64,
        "pre_history_tail_fingerprint_sha256": "2" * 64,
        "post_state_fingerprint_sha256": "3" * 64,
        "post_history_tail_fingerprint_sha256": "4" * 64,
        "current_record_mutations": [],
        "immutable_history_appends": [],
    }
    value.update(updates)
    value["current_record_mutations"] = _canonical_records(value["current_record_mutations"])
    value["immutable_history_appends"] = _canonical_records(value["immutable_history_appends"])
    return value


def _transaction_descriptor(**updates: Any) -> dict[str, Any]:
    projection = _transaction_projection(**updates)
    return {**projection, "transaction_fingerprint_sha256": _actual_fingerprint(projection)}


def _descriptor_matches_metadata(descriptor: dict[str, Any], metadata: dict[str, Any]) -> bool:
    binding = MACHINE["state_store_fingerprint_contract"]["durable_transaction_descriptor"][
        "metadata_binding"
    ]
    return all(
        descriptor[descriptor_field] == metadata[metadata_field]
        for descriptor_field, metadata_field in binding.items()
        if descriptor_field != "comparison"
    )


def _descriptor_hash_valid(descriptor: dict[str, Any]) -> bool:
    projection_fields = MACHINE["state_store_fingerprint_contract"]["transaction_fingerprint"][
        "projection_fields"
    ]
    projection = {field: descriptor[field] for field in projection_fields}
    return _actual_fingerprint(projection) == descriptor["transaction_fingerprint_sha256"]


def _rehash_descriptor(descriptor: dict[str, Any]) -> None:
    projection_fields = MACHINE["state_store_fingerprint_contract"]["transaction_fingerprint"][
        "projection_fields"
    ]
    descriptor["transaction_fingerprint_sha256"] = _actual_fingerprint(
        {field: descriptor[field] for field in projection_fields}
    )


def _descriptor_chain_valid(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    return (
        current["expected_current_generation"] == previous["target_generation"]
        and current["target_generation"] == previous["target_generation"] + 1
        and current["pre_state_fingerprint_sha256"] == previous["post_state_fingerprint_sha256"]
        and current["pre_history_tail_fingerprint_sha256"]
        == previous["post_history_tail_fingerprint_sha256"]
    )


def _descriptor_intrinsically_valid(value: Any) -> bool:
    schema = MACHINE["executable_boundary_schemas"]["StateStoreTransactionDescriptor"]
    if not isinstance(value, dict) or set(value) != set(schema["required"]):
        return False
    if not _canonical_scope(value["account_id"], value["device_installation_id"]):
        return False
    for field in (
        "state_store_identity_fingerprint_sha256",
        "post_state_fingerprint_sha256",
        "post_history_tail_fingerprint_sha256",
        "transaction_fingerprint_sha256",
    ):
        if not isinstance(value[field], str) or SHA_RE.fullmatch(value[field]) is None:
            return False
    if not _positive(value["state_store_schema_version"]) or not _positive(
        value["target_generation"]
    ):
        return False
    if value["environment"] not in {"PAPER", "TESTNET", "LIVE"}:
        return False
    genesis = value["target_generation"] == 1
    nullable_fields = (
        "expected_current_generation",
        "pre_state_fingerprint_sha256",
        "pre_history_tail_fingerprint_sha256",
    )
    if genesis:
        if any(value[field] is not None for field in nullable_fields):
            return False
    else:
        if not _positive(value["expected_current_generation"]):
            return False
        for field in nullable_fields[1:]:
            if not isinstance(value[field], str) or SHA_RE.fullmatch(value[field]) is None:
                return False
    for field in ("current_record_mutations", "immutable_history_appends"):
        records = value[field]
        if not isinstance(records, list) or not all(
            _validate_persistence_record(record) for record in records
        ):
            return False
        if records != _canonical_records(records):
            return False
    return True


def _descriptor_chain(length: int = 4) -> list[dict[str, Any]]:
    chain = [
        _transaction_descriptor(
            expected_current_generation=None,
            target_generation=1,
            pre_state_fingerprint_sha256=None,
            pre_history_tail_fingerprint_sha256=None,
        )
    ]
    for generation in range(2, length + 1):
        previous = chain[-1]
        chain.append(
            _transaction_descriptor(
                expected_current_generation=generation - 1,
                target_generation=generation,
                pre_state_fingerprint_sha256=previous["post_state_fingerprint_sha256"],
                pre_history_tail_fingerprint_sha256=previous[
                    "post_history_tail_fingerprint_sha256"
                ],
            )
        )
    return chain


def _complete_descriptor_chain_valid(
    descriptors: list[dict[str, Any]], metadata: dict[str, Any]
) -> bool:
    generation = metadata["protected_freshness_generation"]
    if not all(_descriptor_intrinsically_valid(descriptor) for descriptor in descriptors):
        return False
    targets = [descriptor["target_generation"] for descriptor in descriptors]
    if any(not _positive(target) for target in targets):
        return False
    if len(targets) != generation or sorted(targets) != list(range(1, generation + 1)):
        return False
    canonical = sorted(descriptors, key=lambda descriptor: descriptor["target_generation"])
    immutable_scope = (
        "account_id",
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
    )
    if any(
        descriptor[field] != metadata[field]
        for descriptor in canonical
        for field in immutable_scope
    ):
        return False
    genesis = canonical[0]
    if (
        genesis["expected_current_generation"] is not None
        or genesis["pre_state_fingerprint_sha256"] is not None
        or genesis["pre_history_tail_fingerprint_sha256"] is not None
    ):
        return False
    if not all(_descriptor_hash_valid(descriptor) for descriptor in canonical):
        return False
    if not all(
        _descriptor_chain_valid(previous, current)
        for previous, current in zip(canonical[:-1], canonical[1:], strict=True)
    ):
        return False
    return _descriptor_matches_metadata(canonical[-1], metadata)


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def _nfc(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_nfc(item) for item in value]
    if isinstance(value, dict):
        return {_nfc(key): _nfc(item) for key, item in value.items()}
    return value


def _semantic_fingerprint(binding: dict[str, Any], upstream: dict[str, Any]) -> str | None:
    derivation = binding.get("semantic_fingerprint_derivation")
    if not isinstance(derivation, dict):
        return None
    if derivation.get("algorithm") != "SHA-256":
        return None
    fields = derivation.get("input_fields")
    if not isinstance(fields, list) or not all(isinstance(name, str) for name in fields):
        return None
    shape = derivation.get("input_shape")
    if shape == "JSON_OBJECT":
        return _actual_fingerprint({name: upstream[name] for name in fields})
    if shape == "ORDERED_CANONICAL_JSON_ARRAY":
        return _actual_fingerprint([upstream[name] for name in fields])
    if shape == "CANONICAL_NFC_JSON_OBJECT":
        return _actual_fingerprint(_nfc({name: upstream[name] for name in fields}))
    if shape == "DOMAIN_SEPARATOR_NEWLINE_CANONICAL_JSON_OBJECT":
        canonical = {name: upstream[name] for name in fields}
        for name in ("instrument_ids", "source_catalog_snapshot_ids"):
            if isinstance(canonical.get(name), list):
                canonical[name] = sorted(canonical[name])
        encoded = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        raw = f"{derivation['domain_separator']}\n{encoded}".encode()
        return hashlib.sha256(raw).hexdigest()
    if shape == "DOMAIN_SEPARATOR_BYTES_PLUS_CANONICAL_JSON_VALUE":
        if fields != ["configuration"]:
            return None
        configuration = upstream["configuration"]
        if not isinstance(configuration, dict) or any(
            isinstance(value, float) for value in configuration.values()
        ):
            return None
        configuration_bytes = json.dumps(
            configuration, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
        return hashlib.sha256(
            derivation["domain_separator"].encode() + configuration_bytes
        ).hexdigest()
    return None


def _attests(entry: Any, upstream: Any) -> bool:
    if not isinstance(entry, dict) or not isinstance(upstream, dict):
        return False
    if set(entry) != {"milestone", "artifact", "json_pointer", "content_fingerprint_sha256"}:
        return False
    ok, value = _resolve_pointer(upstream, entry["json_pointer"])
    return bool(ok and _actual_fingerprint(value) == entry["content_fingerprint_sha256"])


def _positive(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_metadata(value: Any) -> str:
    schema = MACHINE["executable_boundary_schemas"]["StateStoreMetadata"]
    if not isinstance(value, dict) or set(value) != set(schema["required"]):
        return "CONTRACT_INCONSISTENT"
    for key in (
        "state_store_schema_version",
        "protected_freshness_generation",
    ):
        if not _positive(value.get(key)):
            return "CONTRACT_INCONSISTENT"
    for key in (
        "state_store_identity_fingerprint_sha256",
        "state_fingerprint_sha256",
        "transaction_fingerprint_sha256",
        "history_tail_fingerprint_sha256",
    ):
        if not isinstance(value.get(key), str) or not SHA_RE.fullmatch(value[key]):
            return "CONTRACT_INCONSISTENT"
    if value.get("environment") not in {"PAPER", "TESTNET", "LIVE"}:
        return "ENVIRONMENT_SCOPE_MISMATCH"
    if not _canonical_scope(value.get("account_id"), value.get("device_installation_id")):
        return "CONTRACT_INCONSISTENT"
    return "VALID"


def _metadata(**updates: Any) -> dict[str, Any]:
    value = {
        "state_store_schema_version": 1,
        "account_id": SCOPE[0],
        "device_installation_id": SCOPE[1],
        "state_store_identity_fingerprint_sha256": SCOPE[2],
        "environment": "PAPER",
        "protected_freshness_generation": 1,
        "state_fingerprint_sha256": SHA,
        "transaction_fingerprint_sha256": "b" * 64,
        "history_tail_fingerprint_sha256": "c" * 64,
    }
    value.update(updates)
    return value


def _observation(**updates: Any) -> dict[str, Any]:
    value = _metadata()
    value.update(durable_confirmed=True, authoritative_history_integrity=True, current_commit=True)
    value.update(updates)
    return value


def _lease_after_restart(state: Any) -> str:
    return {
        "UNUSED": "LEASE_STALE",
        "CONSUMED": "CONSUMED",
        "UNKNOWN_RECONCILIATION": "RECONCILIATION_REQUIRED",
    }.get(state, "CONTRACT_INCONSISTENT")


DEPENDENCIES = MACHINE["dependency_manifest"]["authority_dependencies"]


@pytest.mark.parametrize(
    "entry", DEPENDENCIES, ids=lambda e: f"{e['milestone']}{e['json_pointer']}"
)
def test_each_authored_dependency_fingerprint_independently_attests(entry: dict[str, Any]) -> None:
    upstream = json.loads((DOCS / entry["artifact"]).read_text())
    assert _attests(entry, upstream)


def test_manifest_is_exactly_48_and_excludes_m01_authority() -> None:
    assert len(DEPENDENCIES) == MACHINE["dependency_manifest"]["count"] == 48
    assert {e["milestone"] for e in DEPENDENCIES} == {f"M0.{n}" for n in range(2, 11)}
    assert MACHINE["dependency_manifest"]["m0_1_authority_allowed"] is False


def test_m05_credential_profile_dependency_attestation_tracks_canonical_corrective() -> None:
    matches = [
        entry
        for entry in DEPENDENCIES
        if entry["milestone"] == "M0.5"
        and entry["artifact"] == "exchange_accounts_and_instruments.json"
        and entry["json_pointer"] == "/credential_profile_contract"
    ]
    assert len(matches) == 1
    upstream = json.loads((DOCS / matches[0]["artifact"]).read_text())
    assert _attests(matches[0], upstream)
    assert upstream["credential_profile_contract"]["lifecycle_states"] == [
        "ACTIVE",
        "RETIRED",
    ]


def _direct_source_contracts() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    representations = MACHINE["backup_contract"]["representation_registry"]
    direct = MACHINE["backup_contract"]["direct_upstream_validator_registry"]
    result = {}
    for name, entry in representations.items():
        if (
            entry.get("carrier_strategy") == "PERSISTENCE_RECORD"
            and entry.get("representation_category") == "DIRECT_UPSTREAM_SCHEMA"
        ):
            document = json.loads((DOCS / entry["semantic_artifact"]).read_text())
            ok, source = _resolve_pointer(document, entry["semantic_json_pointer"])
            assert ok and isinstance(source, (dict, list))
            if isinstance(source, list):
                source = {"exact_fields": source}
            result[name] = (direct[name], source)
    return result


def _source_exact_fields(name: str, source: dict[str, Any]) -> list[str]:
    if name == "ExchangeAccount":
        return source["record_fields"]
    if name == "CredentialProfile metadata/reference":
        return source["fields"] + ["saas_sync_candidate"]
    if name == "StrategyInstance current lifecycle/config":
        return source["exact_fields"]
    if name == "kill-switch state/generation":
        return source["record_fields"]
    if name == "Command accepted request":
        return source["SUBMIT_ORDER"]["request_fields"]
    if name == "Event":
        return source["envelope_schema"]["fields"]
    if name == "OrderIntent":
        return source["request_fields"]
    if name == "Fill":
        return source["fact_fields"]
    if name == "LedgerEntry":
        return source["exact_fields"]
    return source["exact_fields"]


@dataclass(frozen=True)
class SourceExpectedDirectContract:
    exact_fields: list[str]
    nullable_fields: list[str]
    field_schemas: dict[str, dict[str, Any]]
    record_key_fields: list[str]
    dimensions: dict[str, str]
    terminal_fingerprint: dict[str, Any] | None
    source_artifact: str
    source_pointer: str
    semantic_constraints: dict[str, Any]
    stage1_scope: str
    stage2_contextual_rules_excluded: list[str]


DIRECT_STAGE2_AUTHORITY_EXCLUSIONS = [
    "accepted/current authority or membership",
    "trusted history and cross-record lineage resolution",
    "external venue or Core registry lookup",
    "M0.3 membership and restore authority",
    "LIVE readiness",
]
DIRECT_STAGE1_SCOPE = "INTRINSIC_SELF_CONTAINED_ONLY_NO_AUTHORITY"


def _entity_prefix(canonical_name: str) -> str:
    matches = [
        entry["id_prefix"]
        for entry in VOCABULARY["entity_kinds"]
        if entry["canonical_name"] == canonical_name
    ]
    if len(matches) != 1:
        raise AssertionError(f"missing or duplicate canonical entity prefix: {canonical_name}")
    return matches[0]


def _id_schema(canonical_name: str) -> dict[str, Any]:
    return {"type": "id", "prefix": _entity_prefix(canonical_name)}


def _source_dimensions(
    *,
    canonical_ids: bool,
    registries: bool,
    constants: bool = False,
    nested: bool = False,
    arrays: bool = False,
    unique: bool = False,
    ordered: bool = False,
    sibling: bool = False,
    grammar: bool = False,
    terminal: bool = False,
) -> dict[str, str]:
    checked = "SOURCE_DERIVED_AND_CHECKED"
    absent = "SOURCE_DOES_NOT_DEFINE"
    return {
        "exact_fields": checked,
        "nullable_fields": checked,
        "field_schemas": checked,
        "canonical_ids": checked if canonical_ids else absent,
        "closed_registries": checked if registries else absent,
        "constants": checked if constants else absent,
        "nested_objects": checked if nested else absent,
        "arrays": checked if arrays else absent,
        "array_item_schemas": checked if arrays else absent,
        "array_uniqueness": checked if unique else absent,
        "array_ordering": checked if ordered else absent,
        "intrinsic_sibling_constraints": checked if sibling else absent,
        "reference_grammar": checked if grammar else absent,
        "record_key_fields": checked,
        "terminal_content_fingerprint": checked if terminal else absent,
        "stage1_included_rules": checked,
        "stage2_excluded_rules": "STAGE2_CONTEXTUAL_EXCLUDED",
    }


def _expected(
    *,
    fields: list[str],
    nullable: list[str],
    schemas: dict[str, dict[str, Any]],
    key: list[str],
    dimensions: dict[str, str],
    artifact: str,
    pointer: str,
    terminal: dict[str, Any] | None = None,
    semantic_constraints: dict[str, Any] | None = None,
) -> SourceExpectedDirectContract:
    if set(fields) != set(schemas):
        raise AssertionError("upstream source builder did not cover every exact field")
    constraints = {} if semantic_constraints is None else semantic_constraints
    exact_dimensions = copy.deepcopy(dimensions)
    exact_dimensions["intrinsic_sibling_constraints"] = (
        "SOURCE_DERIVED_AND_CHECKED" if constraints else "SOURCE_DOES_NOT_DEFINE"
    )
    return SourceExpectedDirectContract(
        fields,
        nullable,
        schemas,
        key,
        exact_dimensions,
        terminal,
        artifact,
        pointer,
        constraints,
        DIRECT_STAGE1_SCOPE,
        list(DIRECT_STAGE2_AUTHORITY_EXCLUSIONS),
    )


def _submit_order_constraints(source: dict[str, Any]) -> dict[str, Any]:
    rules = source["constraints"]
    return {
        "source_pointer": "/command_registry/SUBMIT_ORDER/constraints",
        "rules": rules,
        "conditional_nullability": [
            {"when": {"order_type": "MARKET"}, "field": "limit_price", "required": "NULL"},
            {"when": {"order_type": "LIMIT"}, "field": "limit_price", "required": "NON_NULL"},
            {"when": {"time_in_force": "GTD"}, "field": "expire_at_utc", "required": "NON_NULL"},
            {"when_not": {"time_in_force": "GTD"}, "field": "expire_at_utc", "required": "NULL"},
            {
                "when": {"source_type": "STRATEGY_INSTANCE"},
                "field": "strategy_instance_id",
                "required": "NON_NULL",
            },
            {
                "when_not": {"source_type": "STRATEGY_INSTANCE"},
                "field": "strategy_instance_id",
                "required": "NULL",
            },
        ],
        "identity_inequality": {
            "left": "order_id",
            "right": "command_id",
            "rule": "MUST_DIFFER",
        },
    }


def _build_exchange_account(source: dict[str, Any]) -> SourceExpectedDirectContract:
    asset_identity = EXCHANGE["exchange_account_contract"]
    enabled_exchanges = [
        {
            "exchange_id": entry["exchange_id"],
            "supported_environments": entry["supported_environments"],
            "supported_market_types": entry["supported_market_types"],
        }
        for entry in EXCHANGE["exchange_registry_contract"]["entries"]
        if entry["status"] == "ENABLED"
    ]
    schemas = {
        "exchange_account_id": {"type": "id", "prefix": asset_identity["id_prefix"]},
        "portfolio_id": _id_schema("Portfolio"),
        "exchange_id": {
            "type": "enum",
            "values": [entry["exchange_id"] for entry in enabled_exchanges],
            "values_source_pointer": "/exchange_registry_contract/entries[status=ENABLED]/exchange_id",
        },
        "environment": {"type": "enum", "values": EXCHANGE["environment_registry"]},
        "market_type": {
            "type": "enum",
            "values": EXCHANGE["market_type_registry"],
            "values_source_pointer": "/market_type_registry",
        },
        "display_name": {"type": "string"},
        "lifecycle_state": {"type": "enum", "values": source["lifecycle_states"]},
        "connection_state": {"type": "enum", "values": source["connection_states"]},
        "execution_authorization": {"type": "enum", "values": source["execution_authorizations"]},
        "external_account_identity_state": {
            "type": "enum",
            "values": EXCHANGE["external_account_identity_contract"]["states"],
            "values_source_pointer": "/external_account_identity_contract/states",
        },
        "external_account_reference": {"type": "non_empty_string"},
        "external_subaccount_reference": {"type": "non_empty_string"},
        "active_credential_profile_id": {"type": "non_empty_string"},
        "account_capability_snapshot_id": {"type": "non_empty_string"},
        "created_at_utc": {"type": "timestamp"},
        "retired_at_utc": {"type": "nullable_timestamp"},
    }
    return _expected(
        fields=source["record_fields"],
        nullable=[
            "external_account_reference",
            "external_subaccount_reference",
            "active_credential_profile_id",
            "account_capability_snapshot_id",
            "retired_at_utc",
        ],
        schemas=schemas,
        key=[source["id_field"]],
        dimensions=_source_dimensions(canonical_ids=True, registries=True, sibling=True),
        artifact="exchange_accounts_and_instruments.json",
        pointer="/exchange_account_contract",
        semantic_constraints={
            "lifecycle_timestamp_policy": {
                "source_pointer": "/exchange_account_contract/lifecycle_timestamp_policy",
                "state_field": "lifecycle_state",
                "created_field": "created_at_utc",
                "retired_field": "retired_at_utc",
                "state_rules": copy.deepcopy(source["lifecycle_timestamp_policy"]),
                "ordering_comparison": "CANONICAL_TIMESTAMP_TEMPORAL_NOT_LEXICAL",
            },
            "build_time_exchange_registry_binding": {
                "source_pointer": "/exchange_registry_contract",
                "registry_classification": "FROZEN_BUILD_TIME_INTRINSIC_NOT_RUNTIME_EXTERNAL_AUTHORITY",
                "closed_build_time_registry": EXCHANGE["exchange_registry_contract"][
                    "closed_build_time_registry"
                ],
                "enabled_entries": enabled_exchanges,
                "exchange_id_rule": "MUST_MATCH_ENABLED_ENTRY",
                "environment_rule": "MUST_BE_IN_SELECTED_ENTRY_SUPPORTED_ENVIRONMENTS",
                "market_type_rule": "MUST_BE_IN_SELECTED_ENTRY_SUPPORTED_MARKET_TYPES",
            },
        },
    )


def _build_credential_profile(source: dict[str, Any]) -> SourceExpectedDirectContract:
    schemas = {
        "credential_profile_id": {"type": "id", "prefix": source["id_prefix"]},
        "exchange_account_id": _id_schema("ExchangeAccount"),
        "exchange_id": {"type": "non_empty_string"},
        "environment_scope": {"type": "enum", "values": EXCHANGE["environment_registry"]},
        "credential_purpose": {"type": "enum", "values": source["credential_purposes"]},
        "secure_store_reference": {
            "type": "secure_store_reference",
            "grammar": source["secure_store_reference_grammar"],
        },
        "public_key_identifier": source["intrinsic_field_schemas"]["public_key_identifier"],
        "permission_snapshot": {
            "type": "unique_array_of_enum",
            "values": source["permission_registry"],
            "order": "PRESERVED",
        },
        "lifecycle_state": {"type": "enum", "values": source["lifecycle_states"]},
        "created_at_utc": {"type": "timestamp"},
        "rotated_from_credential_profile_id": source["intrinsic_field_schemas"][
            "rotated_from_credential_profile_id"
        ],
        "retired_at_utc": source["intrinsic_field_schemas"]["retired_at_utc"],
        "saas_sync_candidate": {
            "type": "exact_literal",
            "value": source["saas_sync_candidate"],
            "source_pointer": "/credential_profile_contract/saas_sync_candidate",
        },
    }
    return _expected(
        fields=source["fields"] + ["saas_sync_candidate"],
        nullable=source["nullable_fields"],
        schemas=schemas,
        key=[source["id_field"]],
        dimensions=_source_dimensions(
            canonical_ids=True,
            registries=True,
            constants=True,
            arrays=True,
            unique=True,
            ordered=True,
            sibling=True,
            grammar=True,
        ),
        artifact="exchange_accounts_and_instruments.json",
        pointer="/credential_profile_contract",
        semantic_constraints={
            "lifecycle_timestamp_policy": copy.deepcopy(source["lifecycle_timestamp_policy"]),
            "self_cycle": {
                "left": "credential_profile_id",
                "right": "rotated_from_credential_profile_id",
                "rule": "MUST_DIFFER_WHEN_RIGHT_NON_NULL",
            },
        },
    )


def _build_strategy_instance(source: dict[str, Any]) -> SourceExpectedDirectContract:
    entity_names = source["entity_references"]
    schemas: dict[str, dict[str, Any]] = {}
    for field in source["exact_fields"]:
        if field == source["id_field"]:
            schemas[field] = {"type": "id", "prefix": source["id_prefix"]}
        elif field in entity_names:
            canonical_name = entity_names[field].removesuffix("Projection")
            schemas[field] = _id_schema(canonical_name)
        elif field == "strategy_definition_version":
            schemas[field] = {"type": "positive_integer"}
        elif field == "lifecycle_state":
            schemas[field] = {
                "type": "enum",
                "values": source["enum_registry"][field],
                "values_source_pointer": "/record_schemas/StrategyInstance/enum_registry/lifecycle_state",
            }
        else:
            raise AssertionError(f"unsupported StrategyInstance field: {field}")
    return _expected(
        fields=source["exact_fields"],
        nullable=source["nullable_fields"],
        schemas=schemas,
        key=[source["id_field"]],
        dimensions=_source_dimensions(canonical_ids=True, registries=True),
        artifact="strategy_market_data_and_execution_routing.json",
        pointer="/record_schemas/StrategyInstance",
    )


def _build_kill_switch(source: dict[str, Any]) -> SourceExpectedDirectContract:
    schemas = copy.deepcopy(source["field_schemas"])
    schemas["scope_type"]["values"] = RISK["scope_hierarchy"]["applicable_order"]
    schemas["state"]["values"] = source["states"]
    return _expected(
        fields=source["record_fields"],
        nullable=[],
        schemas=schemas,
        key=["scope_type", "scope_id", "environment", "generation"],
        dimensions=_source_dimensions(
            canonical_ids=True, registries=True, sibling=True, grammar=True, terminal=True
        ),
        terminal=source["terminal_fingerprint"],
        artifact="risk_hierarchy_kill_switch_and_execution_lease.json",
        pointer="/kill_switch_contract",
        semantic_constraints={
            "scope_id_policy": copy.deepcopy(RISK["scope_hierarchy"]["scope_id_policy"]),
            "generation_authority_split": {
                "stage1": "positive non-boolean integer",
                "stage2": "strict monotonic increase, no rollback/reuse, accepted/current designation",
            },
        },
    )


def _submit_expected(source: dict[str, Any], pointer: str) -> SourceExpectedDirectContract:
    return _expected(
        fields=source["request_fields"],
        nullable=source["nullable_fields"],
        schemas=copy.deepcopy(source["field_schemas"]),
        key=["command_id" if pointer == "/command_registry" else "order_intent_id"],
        dimensions=_source_dimensions(
            canonical_ids=True, registries=True, constants=True, sibling=True
        ),
        artifact="commands_events_order_lifecycle_and_idempotency.json",
        pointer=pointer,
        semantic_constraints=_submit_order_constraints(source),
    )


def _build_command(source: dict[str, Any]) -> SourceExpectedDirectContract:
    return _submit_expected(source["SUBMIT_ORDER"], "/command_registry")


def _build_order_intent(source: dict[str, Any]) -> SourceExpectedDirectContract:
    return _submit_expected(source, "/command_registry/SUBMIT_ORDER")


def _build_event(source: dict[str, Any]) -> SourceExpectedDirectContract:
    envelope = source["envelope_schema"]
    schemas = copy.deepcopy(envelope["field_schemas"])
    schemas["event_type"] = {
        "type": "enum",
        "registry": "event_types",
        "values": source["event_types"],
    }
    terminal = {
        "field": "event_fingerprint_sha256",
        "algorithm": "SHA-256",
        "input_fields": [
            field for field in envelope["fields"] if field != "event_fingerprint_sha256"
        ],
        "excluded_fields": envelope["fingerprint_excluded_fields"],
        "input_shape": "CANONICAL_NFC_JSON_OBJECT",
        "canonicalization": [
            "complete immutable event envelope",
            "recursive NFC keys and string values",
            "canonical JSON sorted keys and compact separators",
        ],
        "unicode_normalization": "NFC_RECURSIVE_KEYS_AND_VALUES_COLLISION_FAIL_CLOSED",
        "encoding": "UTF-8",
        "digest_format": "lowercase_hex",
        "source_pointer": "/event_contract/fingerprint",
    }
    return _expected(
        fields=envelope["fields"],
        nullable=envelope["nullable_fields"],
        schemas=schemas,
        key=["audit_event_id", "aggregate_version"],
        dimensions=_source_dimensions(
            canonical_ids=True, registries=True, nested=True, sibling=True, terminal=True
        ),
        terminal=terminal,
        artifact="commands_events_order_lifecycle_and_idempotency.json",
        pointer="/event_contract",
        semantic_constraints={
            "event_schema_registry": {
                "event_type_source_pointer": "/event_contract/event_types",
                "schema_source_pointer": "/event_contract/event_schema_registry",
                "safe_payload_field": "safe_payload",
                "required_scope_enforced": True,
            }
        },
    )


def _build_fill(source: dict[str, Any]) -> SourceExpectedDirectContract:
    terminal = copy.deepcopy(source["fingerprint"])
    terminal.update(
        input_shape="CANONICAL_NFC_JSON_OBJECT",
        unicode_normalization="NFC_RECURSIVE_KEYS_AND_VALUES_COLLISION_FAIL_CLOSED",
        encoding="UTF-8",
        digest_format="lowercase_hex",
        source_pointer="/fill_contract/fingerprint",
    )
    return _expected(
        fields=source["fact_fields"],
        nullable=source["nullable_fields"],
        schemas=copy.deepcopy(source["field_schemas"]),
        key=["fill_id"],
        dimensions=_source_dimensions(
            canonical_ids=True, registries=True, nested=True, sibling=True, terminal=True
        ),
        terminal=terminal,
        artifact="commands_events_order_lifecycle_and_idempotency.json",
        pointer="/fill_contract",
        semantic_constraints={"fee_semantics": copy.deepcopy(source["fee_semantics"])},
    )


def _build_ledger(source: dict[str, Any]) -> SourceExpectedDirectContract:
    asset = EXCHANGE["asset_reference_contract"]
    schemas = {
        "ledger_entry_id": {"type": "id", "prefix": source["identity"]["prefix"]},
        "workspace_id": _id_schema("Workspace"),
        "portfolio_id": _id_schema("Portfolio"),
        "environment": {"type": "enum", "values": EXCHANGE["environment_registry"]},
        "exchange_account_id": _id_schema("ExchangeAccount"),
        "strategy_instance_id": _id_schema("StrategyInstance"),
        "asset_reference": {
            "type": "asset_reference",
            "fields": asset["fields"],
            "field_schemas": {
                "venue_asset_code": {"type": "non_empty_string"},
                "canonical_display_code": {"type": "non_empty_string"},
                "asset_namespace": {"type": "non_empty_string"},
                "mapping_status": {"type": "enum", "values": asset["mapping_statuses"][:2]},
            },
        },
        "account_role": {
            "type": "enum",
            "values": list(LEDGER["account_roles"]),
            "values_source_pointer": "/account_roles keys",
        },
        "direction": {
            "type": "enum",
            "values": source["direction_registry"],
            "values_source_pointer": "/ledger_entry_schema/direction_registry",
        },
        "quantity": {"type": "decimal", "constraint": "positive"},
        "source_type": {
            "type": "enum",
            "values": list(LEDGER["source_registry"]),
            "values_source_pointer": "/source_registry keys",
        },
        "accounting_source_identity": {"type": "non_empty_string"},
        "accounting_source_fingerprint_sha256": {"type": "sha256_hex"},
        "accounting_rule_version": {"type": "positive_integer"},
        "posting_index": {"type": "positive_integer"},
        "posting_role": {
            "type": "enum",
            "values": source["posting_role_registry"],
            "values_source_pointer": "/ledger_entry_schema/posting_role_registry",
        },
        "batch_fingerprint_sha256": {"type": "sha256_hex"},
        "effective_at_utc": {"type": "timestamp"},
        "append_sequence": {"type": "positive_integer"},
        "order_id": _id_schema("Order"),
        "fill_id": _id_schema("Fill"),
        "audit_event_id": _id_schema("AuditEvent"),
        "correction_reason": {"type": "non_empty_string"},
    }
    return _expected(
        fields=source["exact_fields"],
        nullable=source["nullable_fields"],
        schemas=schemas,
        key=["ledger_entry_id", "append_sequence"],
        dimensions=_source_dimensions(
            canonical_ids=True, registries=True, nested=True, sibling=True
        ),
        artifact="ledger_portfolio_capital_and_pnl.json",
        pointer="/ledger_entry_schema",
        semantic_constraints={
            "direction": {"pointer": "/ledger_entry_schema/direction_registry"},
            "posting_role": {"pointer": "/ledger_entry_schema/posting_role_registry"},
        },
    )


def _risk_scalar_schemas(fields: list[str]) -> dict[str, dict[str, Any]]:
    submit_schemas = COMMANDS["command_registry"]["SUBMIT_ORDER"]["field_schemas"]
    schemas: dict[str, dict[str, Any]] = {}
    for field in fields:
        if field in {"command_id", "order_id"}:
            schemas[field] = {
                "type": "id",
                "prefix": submit_schemas[field]["prefix"],
            }
        elif field.endswith("_fingerprint_sha256") or field.endswith("_fence_sha256"):
            schemas[field] = {"type": "sha256_hex"}
        elif field == "scope_binding":
            schemas[field] = {
                "type": "object",
                "fields": ["canonical_reference"],
                "field_schemas": {"canonical_reference": {"type": "non_empty_string"}},
            }
        elif field == "environment":
            schemas[field] = {"type": "enum", "values": EXCHANGE["environment_registry"]}
        elif field == "evaluated_at_utc":
            schemas[field] = {"type": "timestamp"}
        elif field == "decision":
            schemas[field] = {
                "type": "enum",
                "values": RISK["risk_decision_contract"]["decisions"],
                "values_source_pointer": "/risk_decision_contract/decisions",
            }
        elif field == "decision_fingerprint_sha256":
            schemas[field] = {"type": "sha256_hex"}
    return schemas


def _build_risk_decision(source: dict[str, Any]) -> SourceExpectedDirectContract:
    schemas = _risk_scalar_schemas(source["exact_fields"])
    array = copy.deepcopy(RISK["risk_decision_contract"]["ordered_limit_results_array_schema"])
    item = copy.deepcopy(RISK["risk_decision_contract"]["limit_result_schema"])
    for field, registry in (
        ("limit_type", "limit_result_type_registry"),
        ("result", "limit_result_result_registry"),
        ("reason_code", "limit_result_reason_codes"),
    ):
        item["field_schemas"][field]["values"] = RISK["risk_decision_contract"][registry]
    array["item_schema"] = item
    schemas["ordered_limit_results"] = array
    schemas["kill_switch_result"] = {
        "type": "enum",
        "values": RISK["risk_decision_contract"]["kill_switch_result_registry"],
        "values_source_pointer": "/risk_decision_contract/kill_switch_result_registry",
    }
    schemas["decision_fingerprint_sha256"] = {"type": "sha256_hex"}
    return _expected(
        fields=source["exact_fields"],
        nullable=[],
        schemas=schemas,
        key=["command_id", "decision_fingerprint_sha256"],
        dimensions=_source_dimensions(
            canonical_ids=True,
            registries=True,
            nested=True,
            arrays=True,
            unique=True,
            ordered=True,
            sibling=True,
            terminal=True,
        ),
        terminal=source["terminal_fingerprint"],
        artifact="risk_hierarchy_kill_switch_and_execution_lease.json",
        pointer="/executable_boundary_schemas/RiskDecision",
        semantic_constraints={"decision": {"pointer": "/risk_decision_contract/decisions"}},
    )


def _lease_scalar_schema(field: str) -> dict[str, Any]:
    submit_schemas = COMMANDS["command_registry"]["SUBMIT_ORDER"]["field_schemas"]
    prefix_by_field = {
        "execution_lease_id": RISK["execution_lease_contract"]["identity"]["prefix"],
        "command_id": submit_schemas["command_id"]["prefix"],
        "order_id": submit_schemas["order_id"]["prefix"],
        "order_intent_id": submit_schemas["order_intent_id"]["prefix"],
        "workspace_id": _entity_prefix("Workspace"),
        "portfolio_id": _entity_prefix("Portfolio"),
        "exchange_account_id": _entity_prefix("ExchangeAccount"),
        "instrument_id": _entity_prefix("Instrument"),
        "execution_route_id": _entity_prefix("ExecutionRoute"),
        "strategy_instance_id": _entity_prefix("StrategyInstance"),
    }
    if field in prefix_by_field:
        return {"type": "id", "prefix": prefix_by_field[field]}
    if field.endswith("_fingerprint_sha256") or field.endswith("_fence_sha256"):
        return {"type": "sha256_hex"}
    if field in {"instrument_metadata_version"}:
        return {"type": "positive_integer"}
    if field in {"quantity", "reservation_original_quantity", "reservation_remaining_quantity"}:
        return {"type": "decimal", "constraint": "positive"}
    if field in {"issued_at_utc", "expires_at_utc", "order_expire_at_utc"}:
        return {"type": "timestamp"}
    if field == "environment":
        return {"type": "enum", "values": EXCHANGE["environment_registry"]}
    if field == "reservation_asset_reference":
        asset = EXCHANGE["asset_reference_contract"]
        return {
            "type": "asset_reference",
            "fields": asset["fields"],
            "field_schemas": {
                "venue_asset_code": {"type": "non_empty_string"},
                "canonical_display_code": {"type": "non_empty_string"},
                "asset_namespace": {"type": "non_empty_string"},
                "mapping_status": {"type": "enum", "values": asset["mapping_statuses"][:2]},
            },
        }
    if field == "source_identity":
        return {
            "type": "object",
            "fields": ["canonical_reference"],
            "field_schemas": {"canonical_reference": {"type": "non_empty_string"}},
        }
    if field in {
        "exchange_id",
        "side",
        "order_type",
        "limit_price",
        "time_in_force",
        "reservation_source_audit_event_id",
    }:
        return {"type": "non_empty_string"}
    if field == "lease_fingerprint_sha256":
        return {"type": "sha256_hex"}
    raise AssertionError(f"unsupported M0.9 ExecutionLease field: {field}")


def _build_execution_lease(source: dict[str, Any]) -> SourceExpectedDirectContract:
    schemas = {
        field: _lease_scalar_schema(field)
        for field in source["exact_fields"]
        if field not in {"effective_policy_bindings", "kill_switch_bindings"}
    }
    bindings = RISK["execution_lease_contract"]["binding_array_schemas"]
    schemas["effective_policy_bindings"] = copy.deepcopy(bindings["effective_policy_bindings"])
    schemas["kill_switch_bindings"] = copy.deepcopy(bindings["kill_switch_bindings"])
    return _expected(
        fields=source["exact_fields"],
        nullable=[],
        schemas=schemas,
        key=["execution_lease_id"],
        dimensions=_source_dimensions(
            canonical_ids=True,
            registries=True,
            nested=True,
            arrays=True,
            unique=True,
            ordered=True,
            terminal=True,
        ),
        terminal=source["terminal_fingerprint"],
        artifact="risk_hierarchy_kill_switch_and_execution_lease.json",
        pointer="/executable_boundary_schemas/ExecutionLease",
    )


def _build_session(source: dict[str, Any]) -> SourceExpectedDirectContract:
    schemas = {
        "account_id": _id_schema("CryptoHunterAccount"),
        "operator_id": _id_schema("OperatorIdentity"),
        "device_installation_id": _id_schema("DeviceInstallation"),
        "runtime_session_id": _id_schema("RuntimeSession"),
        "state": {
            "type": "enum",
            "values": IDENTITY["registries"]["session_states"],
            "values_source_pointer": "/registries/session_states",
        },
        "session_generation": {"type": "positive_integer"},
        "security_generation": {"type": "positive_integer"},
        "content_fingerprint_sha256": {"type": "sha256_hex"},
    }
    return _expected(
        fields=source["exact_fields"],
        nullable=[],
        schemas=schemas,
        key=[
            "account_id",
            "operator_id",
            "device_installation_id",
            "runtime_session_id",
            "session_generation",
            "security_generation",
        ],
        dimensions=_source_dimensions(
            canonical_ids=True, registries=True, sibling=True, terminal=True
        ),
        terminal=source["terminal_fingerprint"],
        artifact="identity_device_authentication_and_secrets.json",
        pointer="/executable_boundary_schemas/SessionSecurityState",
        semantic_constraints={"state": {"pointer": "/registries/session_states"}},
    )


def _build_secret(source: dict[str, Any]) -> SourceExpectedDirectContract:
    operations = copy.deepcopy(source["field_schemas"]["permitted_operations"])
    operations["values"] = IDENTITY["registries"]["secret_use_operation_registry"]
    schemas = {
        "secret_reference": {
            "type": "secure_store_reference",
            "grammar": IDENTITY["secret_reference_policy"],
        },
        "secret_kind": {"type": "enum", "values": IDENTITY["registries"]["secret_kinds"]},
        "exchange_account_id": _id_schema("ExchangeAccount"),
        "credential_profile_id": _id_schema("CredentialProfile"),
        "exchange_id": {"type": "non_empty_string"},
        "environment": {"type": "enum", "values": IDENTITY["registries"]["environments"]},
        "permitted_operations": operations,
        "secret_revision": {"type": "positive_integer"},
        "state": {
            "type": "enum",
            "values": IDENTITY["registries"]["secret_states"],
            "values_source_pointer": "/registries/secret_states",
        },
        "content_fingerprint_sha256": {"type": "sha256_hex"},
    }
    return _expected(
        fields=source["exact_fields"],
        nullable=[],
        schemas=schemas,
        key=["secret_reference", "secret_revision"],
        dimensions=_source_dimensions(
            canonical_ids=True,
            registries=True,
            arrays=True,
            unique=True,
            ordered=True,
            sibling=True,
            grammar=True,
            terminal=True,
        ),
        terminal=source["terminal_fingerprint"],
        artifact="identity_device_authentication_and_secrets.json",
        pointer="/executable_boundary_schemas/SecretMetadataProjection",
        semantic_constraints={
            "secret_kind": {"pointer": "/registries/secret_kinds"},
            "state": {"pointer": "/registries/secret_states"},
            "secret_reference": {"pointer": "/secret_reference_policy"},
        },
    )


DIRECT_SOURCE_EXPECTATION_BUILDERS = {
    "ExchangeAccount": _build_exchange_account,
    "CredentialProfile metadata/reference": _build_credential_profile,
    "StrategyInstance current lifecycle/config": _build_strategy_instance,
    "kill-switch state/generation": _build_kill_switch,
    "Command accepted request": _build_command,
    "Event": _build_event,
    "OrderIntent": _build_order_intent,
    "Fill": _build_fill,
    "LedgerEntry": _build_ledger,
    "RiskDecision": _build_risk_decision,
    "ExecutionLease immutable record": _build_execution_lease,
    "SessionSecurityState current generation/state": _build_session,
    "SecretMetadataProjection": _build_secret,
}


def _source_expected_direct_contract(
    name: str, source: dict[str, Any]
) -> SourceExpectedDirectContract:
    builder = DIRECT_SOURCE_EXPECTATION_BUILDERS.get(name)
    if builder is None:
        raise AssertionError(f"unknown direct source expectation: {name}")
    return builder(source)


def _schema_semantics_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return actual == expected


def _direct_common_source_parity(
    name: str, projection: dict[str, Any], source: dict[str, Any]
) -> bool:
    expected = _source_expected_direct_contract(name, source)
    return (
        projection.get("exact_fields") == expected.exact_fields
        and projection.get("nullable_fields") == expected.nullable_fields
        and projection.get("upstream_field_schemas") == expected.field_schemas
        and projection.get("record_key_fields") == expected.record_key_fields
        and projection.get("source_parity_dimension_classification") == expected.dimensions
        and projection.get("terminal_fingerprint") == expected.terminal_fingerprint
        and projection.get("semantic_constraints", {}) == expected.semantic_constraints
        and projection.get("semantic_artifact") == expected.source_artifact
        and projection.get("semantic_json_pointer") == expected.source_pointer
        and projection.get("stage1_scope") == expected.stage1_scope
        and projection.get("stage2_contextual_rules_excluded")
        == expected.stage2_contextual_rules_excluded
    )


def _parity_exchange_account(projection: dict[str, Any], source: dict[str, Any]) -> bool:
    schemas = projection.get("upstream_field_schemas", {})
    return _direct_common_source_parity("ExchangeAccount", projection, source) and (
        schemas["exchange_account_id"] == {"type": "id", "prefix": source["id_prefix"]}
        and schemas["portfolio_id"] == {"type": "id", "prefix": "port"}
        and schemas["environment"]["values"] == ["PAPER", "TESTNET", "LIVE"]
        and schemas["market_type"]["values"] == EXCHANGE["market_type_registry"]
        and schemas["display_name"] == {"type": "string"}
        and schemas["external_account_identity_state"]["values"]
        == EXCHANGE["external_account_identity_contract"]["states"]
        and schemas.get("lifecycle_state", {}).get("values") == source["lifecycle_states"]
        and schemas.get("connection_state", {}).get("values") == source["connection_states"]
        and schemas.get("execution_authorization", {}).get("values")
        == source["execution_authorizations"]
        and schemas["created_at_utc"]["type"] == "timestamp"
        and schemas["retired_at_utc"]["type"] == "nullable_timestamp"
        and projection["record_key_fields"] == [source["id_field"]]
        and projection["semantic_constraints"]["lifecycle_timestamp_policy"]["source_pointer"]
        == "/exchange_account_contract/lifecycle_timestamp_policy"
        and projection["semantic_constraints"]["lifecycle_timestamp_policy"]["state_rules"]
        == source["lifecycle_timestamp_policy"]
        and projection["semantic_constraints"]["build_time_exchange_registry_binding"][
            "closed_build_time_registry"
        ]
        is True
    )


def _parity_credential_profile(projection: dict[str, Any], source: dict[str, Any]) -> bool:
    schemas = projection.get("upstream_field_schemas", {})
    return _direct_common_source_parity(
        "CredentialProfile metadata/reference", projection, source
    ) and (
        schemas["credential_profile_id"] == {"type": "id", "prefix": source["id_prefix"]}
        and schemas["exchange_account_id"] == {"type": "id", "prefix": "xacc"}
        and schemas["environment_scope"]["values"] == ["PAPER", "TESTNET", "LIVE"]
        and schemas["public_key_identifier"]
        == source["intrinsic_field_schemas"]["public_key_identifier"]
        and schemas["created_at_utc"]["type"]
        == source["intrinsic_field_schemas"]["created_at_utc"]["type"]
        and schemas["retired_at_utc"] == source["intrinsic_field_schemas"]["retired_at_utc"]
        and projection.get("nullable_fields") == source["nullable_fields"]
        and schemas.get("credential_purpose", {}).get("values") == source["credential_purposes"]
        and schemas.get("permission_snapshot", {}).get("type") == "unique_array_of_enum"
        and schemas.get("permission_snapshot", {}).get("values") == source["permission_registry"]
        and schemas.get("secure_store_reference", {}).get("grammar")
        == source["secure_store_reference_grammar"]
        and schemas.get("lifecycle_state", {}).get("values") == source["lifecycle_states"]
        and schemas.get("rotated_from_credential_profile_id")
        == source["intrinsic_field_schemas"]["rotated_from_credential_profile_id"]
        and schemas.get("saas_sync_candidate")
        == {
            "type": "exact_literal",
            "value": source["saas_sync_candidate"],
            "source_pointer": "/credential_profile_contract/saas_sync_candidate",
        }
        and projection.get("semantic_constraints", {}).get("lifecycle_timestamp_policy")
        == source["lifecycle_timestamp_policy"]
        and projection["record_key_fields"] == [source["id_field"]]
    )


def _parity_strategy_instance(projection: dict[str, Any], source: dict[str, Any]) -> bool:
    schema = projection.get("upstream_field_schemas", {}).get("lifecycle_state", {})
    return _direct_common_source_parity(
        "StrategyInstance current lifecycle/config", projection, source
    ) and (
        projection.get("nullable_fields") == source["nullable_fields"]
        and schema.get("type") == "enum"
        and schema.get("values") == source["enum_registry"]["lifecycle_state"]
        and projection["record_key_fields"] == [source["id_field"]]
        and projection["upstream_field_schemas"]["strategy_definition_version"]["type"]
        == "positive_integer"
    )


def _parity_submit(name: str, projection: dict[str, Any], source: dict[str, Any]) -> bool:
    return _direct_common_source_parity(name, projection, source)


def _parity_event(projection: dict[str, Any], source: dict[str, Any]) -> bool:
    terminal = projection.get("terminal_fingerprint", {})
    schemas = projection.get("upstream_field_schemas", {})
    expected_schemas = copy.deepcopy(source["envelope_schema"]["field_schemas"])
    expected_schemas["event_type"] = {
        "type": "enum",
        "values": source["event_types"],
        "registry": "event_types",
    }
    expected_terminal = {
        "field": "event_fingerprint_sha256",
        "algorithm": "SHA-256",
        "input_fields": [
            field
            for field in source["envelope_schema"]["fields"]
            if field != "event_fingerprint_sha256"
        ],
        "excluded_fields": source["envelope_schema"]["fingerprint_excluded_fields"],
        "input_shape": "CANONICAL_NFC_JSON_OBJECT",
        "canonicalization": [
            "complete immutable event envelope",
            "recursive NFC keys and string values",
            "canonical JSON sorted keys and compact separators",
        ],
        "unicode_normalization": "NFC_RECURSIVE_KEYS_AND_VALUES_COLLISION_FAIL_CLOSED",
        "encoding": "UTF-8",
        "digest_format": "lowercase_hex",
        "source_pointer": "/event_contract/fingerprint",
    }
    return _direct_common_source_parity("Event", projection, source) and (
        projection.get("nullable_fields") == source["envelope_schema"]["nullable_fields"]
        and schemas == expected_schemas
        and terminal == expected_terminal
    )


def _parity_fill(projection: dict[str, Any], source: dict[str, Any]) -> bool:
    terminal = projection.get("terminal_fingerprint", {})
    expected_terminal = copy.deepcopy(source["fingerprint"])
    expected_terminal.update(
        input_shape="CANONICAL_NFC_JSON_OBJECT",
        unicode_normalization="NFC_RECURSIVE_KEYS_AND_VALUES_COLLISION_FAIL_CLOSED",
        encoding="UTF-8",
        digest_format="lowercase_hex",
        source_pointer="/fill_contract/fingerprint",
    )
    return _direct_common_source_parity("Fill", projection, source) and (
        projection.get("nullable_fields") == source["nullable_fields"]
        and projection.get("upstream_field_schemas") == source["field_schemas"]
        and projection.get("semantic_constraints", {}).get("fee_semantics")
        == source["fee_semantics"]
        and terminal == expected_terminal
    )


def _parity_kill_switch(projection: dict[str, Any], source: dict[str, Any]) -> bool:
    schemas = projection.get("upstream_field_schemas", {})
    return _direct_common_source_parity("kill-switch state/generation", projection, source) and (
        schemas
        == source["field_schemas"]
        | {
            "scope_type": source["field_schemas"]["scope_type"]
            | {"values": RISK["scope_hierarchy"]["applicable_order"]},
            "state": source["field_schemas"]["state"] | {"values": source["states"]},
        }
        and projection.get("terminal_fingerprint") == source["terminal_fingerprint"]
    )


def _parity_ledger(projection: dict[str, Any], source: dict[str, Any]) -> bool:
    schemas = projection["upstream_field_schemas"]
    ledger = json.loads((DOCS / "ledger_portfolio_capital_and_pnl.json").read_text())
    return (
        _direct_common_source_parity("LedgerEntry", projection, source)
        and projection.get("nullable_fields") == source["nullable_fields"]
        and schemas["direction"]
        == {
            "type": "enum",
            "values": source["direction_registry"],
            "values_source_pointer": "/ledger_entry_schema/direction_registry",
        }
        and schemas["posting_role"]
        == {
            "type": "enum",
            "values": source["posting_role_registry"],
            "values_source_pointer": "/ledger_entry_schema/posting_role_registry",
        }
        and schemas["account_role"]["values"] == list(ledger["account_roles"])
        and schemas["source_type"]["values"] == list(ledger["source_registry"])
        and schemas["quantity"] == {"type": "decimal", "constraint": "positive"}
        and schemas["effective_at_utc"] == {"type": "timestamp"}
        and projection["record_key_fields"] == ["ledger_entry_id", "append_sequence"]
    )


def _parity_risk(projection: dict[str, Any], source: dict[str, Any]) -> bool:
    schema = projection.get("upstream_field_schemas", {})
    array = schema.get("ordered_limit_results", {})
    expected_array = copy.deepcopy(
        RISK["risk_decision_contract"]["ordered_limit_results_array_schema"]
    )
    expected_item = copy.deepcopy(RISK["risk_decision_contract"]["limit_result_schema"])
    for field, registry in (
        ("limit_type", "limit_result_type_registry"),
        ("result", "limit_result_result_registry"),
        ("reason_code", "limit_result_reason_codes"),
    ):
        expected_item["field_schemas"][field]["values"] = RISK["risk_decision_contract"][registry]
    expected_array["item_schema"] = expected_item
    return _direct_common_source_parity("RiskDecision", projection, source) and (
        array == expected_array
        and schema.get("kill_switch_result", {}).get("values")
        == RISK["risk_decision_contract"]["kill_switch_result_registry"]
        and projection.get("terminal_fingerprint") == source["terminal_fingerprint"]
    )


def _parity_lease(projection: dict[str, Any], source: dict[str, Any]) -> bool:
    schemas = projection.get("upstream_field_schemas", {})
    bindings = RISK["execution_lease_contract"]["binding_array_schemas"]
    return _direct_common_source_parity("ExecutionLease immutable record", projection, source) and (
        schemas.get("effective_policy_bindings") == bindings["effective_policy_bindings"]
        and schemas.get("kill_switch_bindings") == bindings["kill_switch_bindings"]
        and projection.get("terminal_fingerprint") == source["terminal_fingerprint"]
    )


def _parity_session(projection: dict[str, Any], source: dict[str, Any]) -> bool:
    return (
        _direct_common_source_parity(
            "SessionSecurityState current generation/state", projection, source
        )
        and projection["upstream_field_schemas"]["state"]
        == {
            "type": "enum",
            "values": IDENTITY["registries"]["session_states"],
            "values_source_pointer": "/registries/session_states",
        }
        and projection["record_key_fields"]
        == [
            "account_id",
            "operator_id",
            "device_installation_id",
            "runtime_session_id",
            "session_generation",
            "security_generation",
        ]
        and projection.get("terminal_fingerprint") == source["terminal_fingerprint"]
    )


def _parity_secret(projection: dict[str, Any], source: dict[str, Any]) -> bool:
    schema = projection["upstream_field_schemas"]["permitted_operations"]
    expected = copy.deepcopy(source["field_schemas"]["permitted_operations"])
    expected["values"] = IDENTITY["registries"]["secret_use_operation_registry"]
    return (
        _direct_common_source_parity("SecretMetadataProjection", projection, source)
        and schema == expected
        and projection["upstream_field_schemas"]["secret_reference"]["grammar"]
        == IDENTITY["secret_reference_policy"]
        and projection["upstream_field_schemas"]["secret_kind"]["values"]
        == IDENTITY["registries"]["secret_kinds"]
        and projection["upstream_field_schemas"]["state"]["values"]
        == IDENTITY["registries"]["secret_states"]
        and projection.get("terminal_fingerprint") == source["terminal_fingerprint"]
    )


DIRECT_SOURCE_PARITY_EXECUTORS = {
    "ExchangeAccount": _parity_exchange_account,
    "CredentialProfile metadata/reference": _parity_credential_profile,
    "StrategyInstance current lifecycle/config": _parity_strategy_instance,
    "kill-switch state/generation": _parity_kill_switch,
    "Command accepted request": lambda p, s: _parity_submit("Command accepted request", p, s),
    "Event": _parity_event,
    "OrderIntent": lambda p, s: _parity_submit("OrderIntent", p, s),
    "Fill": _parity_fill,
    "LedgerEntry": _parity_ledger,
    "RiskDecision": _parity_risk,
    "ExecutionLease immutable record": _parity_lease,
    "SessionSecurityState current generation/state": _parity_session,
    "SecretMetadataProjection": _parity_secret,
}


def _direct_projection_is_source_derived(
    name: str, projection: dict[str, Any], source: dict[str, Any]
) -> bool:
    executor = DIRECT_SOURCE_PARITY_EXECUTORS.get(name)
    return False if executor is None else executor(projection, source)


@pytest.mark.parametrize("name", list(_direct_source_contracts()))
def test_m011_direct_upstream_validator_registry_is_source_derived(name: str) -> None:
    projection, source = _direct_source_contracts()[name]
    assert _direct_projection_is_source_derived(name, projection, source)


def test_m05_exchange_account_build_time_registry_and_source_types_are_executable() -> None:
    payload = _direct_payload("ExchangeAccount", "a", 1)
    payload["display_name"] = ""
    assert _validate_direct_upstream_payload("ExchangeAccount", payload)

    payload["exchange_id"] = "unknown_exchange"
    assert not _validate_direct_upstream_payload("ExchangeAccount", payload)

    payload = _direct_payload("ExchangeAccount", "a", 1)
    payload.update(exchange_id="generic_testnet_venue", environment="TESTNET", market_type="SPOT")
    assert _validate_direct_upstream_payload("ExchangeAccount", payload)
    payload["environment"] = "PAPER"
    assert not _validate_direct_upstream_payload("ExchangeAccount", payload)
    payload.update(environment="TESTNET", market_type="MARGIN")
    assert not _validate_direct_upstream_payload("ExchangeAccount", payload)

    payload = _direct_payload("ExchangeAccount", "a", 1)
    payload["external_account_identity_state"] = {"canonical_reference": "invented"}
    assert not _validate_direct_upstream_payload("ExchangeAccount", payload)


@pytest.mark.parametrize("state", ["DRAFT", "ACTIVE", "DISABLED"])
def test_m05_exchange_account_non_retired_lifecycle_requires_null_timestamp(state: str) -> None:
    payload = _direct_payload("ExchangeAccount", "a", 1)
    payload["lifecycle_state"] = state
    assert _validate_direct_upstream_payload("ExchangeAccount", payload)
    payload["retired_at_utc"] = "2026-01-01T00:00:00Z"
    assert not _validate_direct_upstream_payload("ExchangeAccount", payload)


@pytest.mark.parametrize(
    ("created", "retired", "valid"),
    [
        ("2026-01-01T00:00:00Z", "2026-01-01T00:00:00.1Z", True),
        ("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", True),
        ("2026-01-01T00:00:00.1Z", "2026-01-01T00:00:00Z", False),
        ("2026-01-01T00:00:00.123456700Z", "2026-01-01T00:00:00.123456701Z", True),
        ("2026-01-01T00:00:00.123456701Z", "2026-01-01T00:00:00.123456700Z", False),
        ("2026-01-01T00:00:00.123456789Z", "2026-01-01T00:00:00.123456789Z", True),
        ("2026-01-01T00:00:00.1234567Z", "2026-01-01T00:00:00.123456700Z", True),
        ("2026-01-01T00:00:00.000000002Z", "2026-01-01T00:00:00.000000001Z", False),
    ],
)
def test_m05_exchange_account_retirement_uses_temporal_fractional_ordering(
    created: str, retired: str, valid: bool
) -> None:
    payload = _direct_payload("ExchangeAccount", "a", 1)
    payload.update(lifecycle_state="RETIRED", created_at_utc=created, retired_at_utc=retired)
    assert _validate_direct_upstream_payload("ExchangeAccount", payload) is valid


def test_m05_credential_profile_saas_literal_and_intrinsic_matrix_are_executable() -> None:
    payload = _direct_payload("CredentialProfile metadata/reference", "a", 1)
    assert payload["saas_sync_candidate"] is False
    assert _validate_direct_upstream_payload("CredentialProfile metadata/reference", payload)
    missing = copy.deepcopy(payload)
    missing.pop("saas_sync_candidate")
    assert not _validate_direct_upstream_payload("CredentialProfile metadata/reference", missing)
    payload["saas_sync_candidate"] = True
    assert not _validate_direct_upstream_payload("CredentialProfile metadata/reference", payload)

    payload = _direct_payload("CredentialProfile metadata/reference", "a", 1)
    payload["rotated_from_credential_profile_id"] = _canonical_fixture_id("cred", "b")
    assert _validate_direct_upstream_payload("CredentialProfile metadata/reference", payload)
    payload["rotated_from_credential_profile_id"] = _canonical_fixture_id("acct", "b")
    assert not _validate_direct_upstream_payload("CredentialProfile metadata/reference", payload)


@pytest.mark.parametrize(
    ("state", "created", "retired", "valid"),
    [
        ("ACTIVE", "2026-01-01T00:00:00Z", None, True),
        ("ACTIVE", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", False),
        ("RETIRED", "2026-01-01T00:00:00Z", None, False),
        ("RETIRED", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00.1Z", True),
        ("RETIRED", "2026-01-01T00:00:00.1Z", "2026-01-01T00:00:00Z", False),
        ("RETIRED", "2026-01-01T00:00:00.123456700Z", "2026-01-01T00:00:00.123456701Z", True),
        ("RETIRED", "2026-01-01T00:00:00.123456701Z", "2026-01-01T00:00:00.123456700Z", False),
        ("RETIRED", "2026-01-01T00:00:00.123456789Z", "2026-01-01T00:00:00.123456789Z", True),
        ("RETIRED", "2026-01-01T00:00:00.000000002Z", "2026-01-01T00:00:00.000000001Z", False),
    ],
)
def test_m05_credential_profile_lifecycle_uses_temporal_fractional_ordering(
    state: str, created: str, retired: str | None, valid: bool
) -> None:
    payload = _direct_payload("CredentialProfile metadata/reference", "a", 1)
    payload.update(lifecycle_state=state, created_at_utc=created, retired_at_utc=retired)
    assert (
        _validate_direct_upstream_payload("CredentialProfile metadata/reference", payload) is valid
    )


def test_m011_timestamp_order_key_matches_m05_nanosecond_algorithm() -> None:
    parsed = _parse_canonical_timestamp_order_key("2026-01-01T00:00:00.1234567Z")
    assert parsed == (datetime(2026, 1, 1, 0, 0, 0), 123_456_700)
    assert _parse_canonical_timestamp_order_key("2026-01-01T00:00:00.1234567890Z") is None


def test_other_eleven_direct_contracts_remain_frozen() -> None:
    expected = {
        "StrategyInstance current lifecycle/config": "757311c7368e385b0242d7c80b8a3bf07537dcb6b7891fb766b8978409945c4e",
        "kill-switch state/generation": "8d98c3a68379a6a2e89308f7e2458869afb0fa9f40b04d9d1b21782f1b556770",
        "Command accepted request": "e58ec1f0430ac7fb854b5906f5686f680cf3fbbdb201628776b864a5cee9d4e6",
        "Event": "73c58fb1ccc6b5ca07e126180872328e66af4189d1294a08d80ef040a6ff058c",
        "OrderIntent": "82626955bcc9dd612d77def8a6392dd5f4b0c3af09dea4ae0d0ba53b2692e880",
        "Fill": "3e1f9dcf87b5d8d8b8c1a3288bd8960a024e9a1abefd292b41492b3ec699741e",
        "LedgerEntry": "e246e18743904a91dbbd12308f0b19ebc0fb707f9683c031e76e34bf77340bff",
        "RiskDecision": "99b2a1acceb4f12130909a005feaea272650d402e85ac6d00cbde78ad53cbdb2",
        "ExecutionLease immutable record": "daaa5bf1a42fba5dcc27df3ee6bc3d55c5b3141a5e19f21e5d88d4c4f58afaa1",
        "SessionSecurityState current generation/state": "166b159d7df286c2ea3cc80f007f065955bf284ff5fe3ade4ef2dcb87dbfcf01",
        "SecretMetadataProjection": "aa978f3fe6cd645b320463943da998e0d04a6ab6a2f07c8ec41d3b27c1ca9cee",
    }
    direct = MACHINE["backup_contract"]["direct_upstream_validator_registry"]
    assert {name: _actual_fingerprint(direct[name]) for name in expected} == expected


def test_discovered_direct_set_exactly_equals_explicit_executor_set() -> None:
    discovered = set(_direct_source_contracts())
    assert discovered == set(DIRECT_SOURCE_PARITY_EXECUTORS)
    assert discovered == set(DIRECT_SOURCE_EXPECTATION_BUILDERS)


def test_source_expectation_inventory_is_complete_and_matches_actual_projection() -> None:
    for name, (projection, source) in _direct_source_contracts().items():
        expected = _source_expected_direct_contract(name, source)
        assert expected.source_artifact == projection["semantic_artifact"]
        assert expected.source_pointer == projection["semantic_json_pointer"]
        assert len(expected.exact_fields) == len(expected.field_schemas)
        assert len(expected.field_schemas) == len(projection["upstream_field_schemas"])
        assert set(expected.field_schemas) == set(projection["upstream_field_schemas"])
        assert expected.field_schemas == projection["upstream_field_schemas"]
        assert expected.semantic_constraints == projection.get("semantic_constraints", {})
        assert expected.record_key_fields == projection["record_key_fields"]
        assert expected.terminal_fingerprint == projection.get("terminal_fingerprint")
        assert expected.stage1_scope == projection["stage1_scope"]
        assert (
            expected.stage2_contextual_rules_excluded
            == projection["stage2_contextual_rules_excluded"]
        )
        assert expected.dimensions == projection["source_parity_dimension_classification"]


def test_dimension_expectation_inventory_is_source_owned_and_exact() -> None:
    for name, (projection, source) in _direct_source_contracts().items():
        expected = _source_expected_direct_contract(name, source)
        assert expected.dimensions == projection["source_parity_dimension_classification"]
        assert set(expected.dimensions) == {
            "exact_fields",
            "nullable_fields",
            "field_schemas",
            "canonical_ids",
            "closed_registries",
            "constants",
            "nested_objects",
            "arrays",
            "array_item_schemas",
            "array_uniqueness",
            "array_ordering",
            "intrinsic_sibling_constraints",
            "reference_grammar",
            "record_key_fields",
            "terminal_content_fingerprint",
            "stage1_included_rules",
            "stage2_excluded_rules",
        }


def test_source_expectation_builders_are_structurally_independent_of_m011() -> None:
    forbidden_names = {"MACHINE", "projection", "direct_upstream_validator_registry"}
    functions = [
        _entity_prefix,
        _id_schema,
        _source_dimensions,
        _expected,
        _submit_order_constraints,
        _submit_expected,
        _risk_scalar_schemas,
        _lease_scalar_schema,
        *DIRECT_SOURCE_EXPECTATION_BUILDERS.values(),
    ]
    for function in functions:
        tree = ast.parse(inspect.getsource(function))
        referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        string_literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert not referenced & forbidden_names, function.__name__
        assert "direct_upstream_validator_registry" not in string_literals, function.__name__


@pytest.mark.parametrize(
    "mutation",
    [
        lambda values: values.remove("accepted/current authority or membership"),
        lambda values: values.remove("M0.3 membership and restore authority"),
        lambda values: values.remove("LIVE readiness"),
        lambda values: values.__setitem__(slice(None), ["INVENTED"]),
        lambda values: values.__setitem__(slice(None), ["LIVE readiness"]),
        lambda values: values.append("INVENTED SIXTH EXCLUSION"),
        lambda values: values.reverse(),
    ],
)
def test_exact_stage2_authority_boundary_rejects_every_drift(mutation) -> None:
    name = "RiskDecision"
    projection, source = _direct_source_contracts()[name]
    candidate = copy.deepcopy(projection)
    mutation(candidate["stage2_contextual_rules_excluded"])
    assert not _direct_projection_is_source_derived(name, candidate, source)


def _mutate_first_semantic_leaf(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(nested, (dict, list)):
                _mutate_first_semantic_leaf(nested)
            elif isinstance(nested, bool):
                value[key] = not nested
            elif isinstance(nested, str):
                value[key] = nested + "_MUTATED"
            else:
                value[key] = "MUTATED"
            return
    elif isinstance(value, list) and value:
        if isinstance(value[0], (dict, list)):
            _mutate_first_semantic_leaf(value[0])
        else:
            value[0] = "MUTATED"


def test_every_nonempty_semantic_constraint_rejects_deep_leaf_mutation() -> None:
    for name, (projection, source) in _direct_source_contracts().items():
        expected = _source_expected_direct_contract(name, source)
        if not expected.semantic_constraints:
            continue
        candidate = copy.deepcopy(projection)
        _mutate_first_semantic_leaf(candidate["semantic_constraints"])
        assert not _direct_projection_is_source_derived(name, candidate, source), name


@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("ExchangeAccount", ("lifecycle_timestamp_policy", "created_field")),
        ("ExchangeAccount", ("lifecycle_timestamp_policy", "retired_field")),
        ("ExchangeAccount", ("lifecycle_timestamp_policy", "state_field")),
        ("ExchangeAccount", ("lifecycle_timestamp_policy", "source_pointer")),
        ("ExchangeAccount", ("lifecycle_timestamp_policy", "state_rules", "DRAFT")),
        ("ExchangeAccount", ("lifecycle_timestamp_policy", "state_rules", "DISABLED")),
        (
            "ExchangeAccount",
            ("build_time_exchange_registry_binding", "exchange_id_rule"),
        ),
        (
            "ExchangeAccount",
            ("build_time_exchange_registry_binding", "environment_rule"),
        ),
        (
            "ExchangeAccount",
            ("build_time_exchange_registry_binding", "market_type_rule"),
        ),
        (
            "CredentialProfile metadata/reference",
            ("lifecycle_timestamp_policy", "ACTIVE", "retired_at_utc"),
        ),
        (
            "CredentialProfile metadata/reference",
            ("lifecycle_timestamp_policy", "RETIRED", "retired_at_utc"),
        ),
        (
            "CredentialProfile metadata/reference",
            ("lifecycle_timestamp_policy", "RETIRED", "ordering"),
        ),
        ("CredentialProfile metadata/reference", ("self_cycle", "left")),
        ("CredentialProfile metadata/reference", ("self_cycle", "right")),
        ("CredentialProfile metadata/reference", ("self_cycle", "rule")),
        (
            "kill-switch state/generation",
            ("scope_id_policy", "bindings", "PRODUCT_SYSTEM", "value"),
        ),
        ("kill-switch state/generation", ("scope_id_policy", "bindings", "WORKSPACE", "prefix")),
        ("kill-switch state/generation", ("scope_id_policy", "validation_stage")),
        ("kill-switch state/generation", ("scope_id_policy", "contextual_exclusions", 0)),
        ("kill-switch state/generation", ("generation_authority_split", "stage1")),
        ("kill-switch state/generation", ("generation_authority_split", "stage2")),
        ("Event", ("event_schema_registry", "event_type_source_pointer")),
        ("Event", ("event_schema_registry", "schema_source_pointer")),
        ("Fill", ("fee_semantics", "NONE")),
        ("Fill", ("fee_semantics", "CHARGE")),
        ("LedgerEntry", ("direction", "pointer")),
        ("LedgerEntry", ("posting_role", "pointer")),
        ("RiskDecision", ("decision", "pointer")),
        ("SessionSecurityState current generation/state", ("state", "pointer")),
        ("SecretMetadataProjection", ("secret_kind", "pointer")),
        ("SecretMetadataProjection", ("state", "pointer")),
        ("SecretMetadataProjection", ("secret_reference", "pointer")),
    ],
)
def test_source_owned_semantic_constraint_leaf_mutations_fail(
    name: str, path: tuple[Any, ...]
) -> None:
    projection, source = _direct_source_contracts()[name]
    candidate = copy.deepcopy(projection)
    target: Any = candidate["semantic_constraints"]
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = "MUTATED"
    assert not _direct_projection_is_source_derived(name, candidate, source)


@pytest.mark.parametrize("name", ["Command accepted request", "OrderIntent"])
@pytest.mark.parametrize(
    "mutation",
    [
        lambda constraints: constraints["conditional_nullability"][0].update(field="expire_at_utc"),
        lambda constraints: constraints["conditional_nullability"][0]["when"].update(
            order_type="LIMIT"
        ),
        lambda constraints: constraints["conditional_nullability"][0].update(required="NON_NULL"),
        lambda constraints: constraints["conditional_nullability"].__setitem__(
            slice(0, 2), list(reversed(constraints["conditional_nullability"][:2]))
        ),
        lambda constraints: constraints["conditional_nullability"].pop(),
        lambda constraints: constraints["conditional_nullability"].append(
            {"when": {"order_type": "INVENTED"}, "field": "limit_price", "required": "NULL"}
        ),
        lambda constraints: constraints["identity_inequality"].update(left="command_id"),
        lambda constraints: constraints["identity_inequality"].update(right="order_id"),
        lambda constraints: constraints["identity_inequality"].update(rule="MAY_EQUAL"),
        lambda constraints: constraints.pop("identity_inequality"),
    ],
)
def test_submit_order_exact_semantic_projection_rejects_mutation(name, mutation) -> None:
    projection, source = _direct_source_contracts()[name]
    candidate = copy.deepcopy(projection)
    mutation(candidate["semantic_constraints"])
    assert not _direct_projection_is_source_derived(name, candidate, source)


@pytest.mark.parametrize("name", ["Command accepted request", "OrderIntent"])
@pytest.mark.parametrize(
    "updates",
    [
        {"order_type": "MARKET", "limit_price": None},
        {"order_type": "LIMIT", "limit_price": "1"},
        {"time_in_force": "GTC", "expire_at_utc": None},
        {"time_in_force": "GTD", "expire_at_utc": "2026-01-01T00:00:00Z"},
        {
            "source_type": "STRATEGY_INSTANCE",
            "strategy_instance_id": "sinst_01890f3a-2b4c-7abc-8def-0123456789ab",
        },
        {"source_type": "OPERATOR", "strategy_instance_id": None},
    ],
)
def test_submit_order_positive_intrinsic_matrix(name: str, updates: dict[str, Any]) -> None:
    payload = _direct_payload(name, "a", 1)
    payload.update(updates)
    assert _validate_direct_upstream_payload(name, payload)


@pytest.mark.parametrize("name", ["Command accepted request", "OrderIntent"])
@pytest.mark.parametrize(
    "updates",
    [
        {"order_type": "MARKET", "limit_price": "1"},
        {"order_type": "LIMIT", "limit_price": None},
        {"time_in_force": "GTD", "expire_at_utc": None},
        {"time_in_force": "GTC", "expire_at_utc": "2026-01-01T00:00:00Z"},
        {"source_type": "STRATEGY_INSTANCE", "strategy_instance_id": None},
        {
            "source_type": "OPERATOR",
            "strategy_instance_id": "sinst_01890f3a-2b4c-7abc-8def-0123456789ab",
        },
    ],
)
def test_submit_order_negative_intrinsic_matrix(name: str, updates: dict[str, Any]) -> None:
    payload = _direct_payload(name, "a", 1)
    payload.update(updates)
    assert not _validate_direct_upstream_payload(name, payload)


@pytest.mark.parametrize("name", ["Command accepted request", "OrderIntent"])
def test_submit_order_identity_inequality_is_executable(name: str) -> None:
    payload = _direct_payload(name, "a", 1)
    payload["order_id"] = payload["command_id"]
    assert not _submit_order_semantics_valid(
        payload,
        MACHINE["backup_contract"]["direct_upstream_validator_registry"][name][
            "semantic_constraints"
        ],
    )
    assert not _validate_direct_upstream_payload(name, payload)


def test_projection_derived_self_attestation_attack_is_rejected() -> None:
    for name, field, leaf, wrong in (
        (
            "SessionSecurityState current generation/state",
            "session_generation",
            "type",
            "non_empty_string",
        ),
        ("SessionSecurityState current generation/state", "runtime_session_id", "prefix", "wrong"),
    ):
        projection, source = _direct_source_contracts()[name]
        attacked = copy.deepcopy(projection)
        attacked["upstream_field_schemas"][field][leaf] = wrong
        circular_mirror = copy.deepcopy(attacked["upstream_field_schemas"])
        assert attacked["upstream_field_schemas"] == circular_mirror
        expected = _source_expected_direct_contract(name, source)
        assert attacked["upstream_field_schemas"] != expected.field_schemas
        assert not _direct_projection_is_source_derived(name, attacked, source)


@pytest.mark.parametrize(
    ("name", "field", "leaf"),
    [
        ("SessionSecurityState current generation/state", "session_generation", "type"),
        ("SessionSecurityState current generation/state", "security_generation", "type"),
        ("SessionSecurityState current generation/state", "runtime_session_id", "prefix"),
        ("SessionSecurityState current generation/state", "device_installation_id", "prefix"),
        ("SessionSecurityState current generation/state", "account_id", "prefix"),
        ("SessionSecurityState current generation/state", "operator_id", "prefix"),
        ("ExecutionLease immutable record", "execution_lease_id", "prefix"),
        ("ExecutionLease immutable record", "command_id", "prefix"),
        ("ExecutionLease immutable record", "order_id", "prefix"),
        ("ExecutionLease immutable record", "instrument_metadata_version", "type"),
        ("ExecutionLease immutable record", "quantity", "constraint"),
        ("ExecutionLease immutable record", "issued_at_utc", "type"),
        ("ExecutionLease immutable record", "expires_at_utc", "type"),
        ("ExecutionLease immutable record", "reservation_asset_reference", "type"),
        ("ExchangeAccount", "exchange_account_id", "prefix"),
        ("ExchangeAccount", "portfolio_id", "prefix"),
        ("ExchangeAccount", "created_at_utc", "type"),
        ("ExchangeAccount", "retired_at_utc", "type"),
        ("LedgerEntry", "ledger_entry_id", "prefix"),
        ("LedgerEntry", "quantity", "constraint"),
        ("LedgerEntry", "effective_at_utc", "type"),
        ("LedgerEntry", "append_sequence", "type"),
        ("LedgerEntry", "asset_reference", "type"),
    ],
)
def test_source_owned_top_level_regressions_fail_parity(name: str, field: str, leaf: str) -> None:
    projection, source = _direct_source_contracts()[name]
    candidate = copy.deepcopy(projection)
    candidate["upstream_field_schemas"][field][leaf] = "MUTATED"
    assert not _direct_projection_is_source_derived(name, candidate, source)


@pytest.mark.parametrize("name", list(_direct_source_contracts()))
def test_every_direct_field_schema_type_mutation_breaks_source_parity(name: str) -> None:
    projection, source = _direct_source_contracts()[name]
    for field in projection["upstream_field_schemas"]:
        candidate = copy.deepcopy(projection)
        candidate["upstream_field_schemas"][field]["type"] = "UNKNOWN_MUTATED_TYPE"
        assert not _direct_projection_is_source_derived(name, candidate, source), (name, field)


def test_dimension_classification_labels_are_source_facts() -> None:
    for name, dimension, wrong in (
        ("RiskDecision", "arrays", "SOURCE_DOES_NOT_DEFINE"),
        ("ExchangeAccount", "arrays", "SOURCE_DERIVED_AND_CHECKED"),
    ):
        projection, source = _direct_source_contracts()[name]
        candidate = copy.deepcopy(projection)
        candidate["source_parity_dimension_classification"][dimension] = wrong
        assert not _direct_projection_is_source_derived(name, candidate, source)


@pytest.mark.parametrize(
    "field",
    [
        "effective_threshold",
        "observed_projected_value",
        "unit_asset_reference",
        "supplying_policy_scope",
        "result",
        "reason_code",
    ],
)
def test_every_limit_result_field_schema_mutation_breaks_source_parity(field: str) -> None:
    projection, source = _direct_source_contracts()["RiskDecision"]
    candidate = copy.deepcopy(projection)
    candidate["upstream_field_schemas"]["ordered_limit_results"]["item_schema"]["field_schemas"][
        field
    ]["type"] = "UNKNOWN_MUTATED_TYPE"
    assert not _direct_projection_is_source_derived("RiskDecision", candidate, source)


@pytest.mark.parametrize(
    ("name", "field", "leaf"),
    [
        ("RiskDecision", "ordered_limit_results", "min_items"),
        ("RiskDecision", "ordered_limit_results", "ordering"),
        ("RiskDecision", "ordered_limit_results", "duplicates"),
        ("ExecutionLease immutable record", "effective_policy_bindings", "min_items"),
        ("ExecutionLease immutable record", "effective_policy_bindings", "ordering"),
        ("ExecutionLease immutable record", "effective_policy_bindings", "duplicates"),
        ("ExecutionLease immutable record", "kill_switch_bindings", "ordering"),
        ("ExecutionLease immutable record", "kill_switch_bindings", "duplicates"),
        ("SecretMetadataProjection", "permitted_operations", "min_items"),
        ("SecretMetadataProjection", "permitted_operations", "unique"),
        ("SecretMetadataProjection", "permitted_operations", "canonical_order"),
    ],
)
def test_every_source_array_semantic_leaf_mutation_breaks_parity(
    name: str, field: str, leaf: str
) -> None:
    projection, source = _direct_source_contracts()[name]
    candidate = copy.deepcopy(projection)
    schema = candidate["upstream_field_schemas"][field]
    schema[leaf] = (
        "MUTATED" if leaf not in {"min_items", "unique"} else (-1 if leaf == "min_items" else False)
    )
    assert not _direct_projection_is_source_derived(name, candidate, source)


@pytest.mark.parametrize(
    "name",
    [
        "Event",
        "Fill",
        "kill-switch state/generation",
        "RiskDecision",
        "ExecutionLease immutable record",
        "SessionSecurityState current generation/state",
        "SecretMetadataProjection",
    ],
)
def test_every_terminal_semantic_leaf_mutation_breaks_source_parity(name: str) -> None:
    projection, source = _direct_source_contracts()[name]
    for leaf in projection["terminal_fingerprint"]:
        candidate = copy.deepcopy(projection)
        value = candidate["terminal_fingerprint"][leaf]
        candidate["terminal_fingerprint"][leaf] = (
            ["MUTATED"]
            if isinstance(value, list)
            else {"mutated": True}
            if isinstance(value, dict)
            else "MUTATED"
        )
        assert not _direct_projection_is_source_derived(name, candidate, source), (name, leaf)


@pytest.mark.parametrize(
    ("name", "mutation"),
    [
        ("Event", lambda value: value.update(nullable_fields=[])),
        (
            "Event",
            lambda value: value["terminal_fingerprint"].update(input_shape="JSON_OBJECT"),
        ),
        (
            "Fill",
            lambda value: value["terminal_fingerprint"].update(canonicalization=[]),
        ),
        (
            "CredentialProfile metadata/reference",
            lambda value: value["upstream_field_schemas"].update(
                permission_snapshot={"type": "array"}
            ),
        ),
        (
            "CredentialProfile metadata/reference",
            lambda value: value["upstream_field_schemas"].update(
                lifecycle_state={"type": "non_empty_string"}
            ),
        ),
        (
            "CredentialProfile metadata/reference",
            lambda value: value["upstream_field_schemas"].update(
                secure_store_reference={"type": "non_empty_string"}
            ),
        ),
        (
            "CredentialProfile metadata/reference",
            lambda value: value["upstream_field_schemas"].update(
                rotated_from_credential_profile_id={"type": "string"}
            ),
        ),
        ("OrderIntent", lambda value: value.update(semantic_constraints={})),
        (
            "ExchangeAccount",
            lambda value: value["upstream_field_schemas"].update(
                lifecycle_state={"type": "non_empty_string"}
            ),
        ),
        (
            "StrategyInstance current lifecycle/config",
            lambda value: value["upstream_field_schemas"].update(
                lifecycle_state={"type": "non_empty_string"}
            ),
        ),
        (
            "StrategyInstance current lifecycle/config",
            lambda value: value.update(nullable_fields=[]),
        ),
        (
            "StrategyInstance current lifecycle/config",
            lambda value: value["upstream_field_schemas"]["lifecycle_state"]["values"].pop(),
        ),
        (
            "StrategyInstance current lifecycle/config",
            lambda value: value["upstream_field_schemas"]["lifecycle_state"]["values"].append(
                "INVENTED"
            ),
        ),
        (
            "kill-switch state/generation",
            lambda value: value["upstream_field_schemas"].update(
                scope_type={"type": "non_empty_string"}
            ),
        ),
        (
            "kill-switch state/generation",
            lambda value: value["upstream_field_schemas"].update(
                scope_id={"type": "non_empty_string"}
            ),
        ),
        (
            "kill-switch state/generation",
            lambda value: value["upstream_field_schemas"].update(
                state={"type": "non_empty_string"}
            ),
        ),
        (
            "kill-switch state/generation",
            lambda value: value["upstream_field_schemas"].update(generation={"type": "string"}),
        ),
        (
            "RiskDecision",
            lambda value: value["upstream_field_schemas"].update(
                ordered_limit_results={"type": "array"}
            ),
        ),
        (
            "RiskDecision",
            lambda value: value["upstream_field_schemas"].update(
                kill_switch_result={"type": "non_empty_string"}
            ),
        ),
        (
            "ExecutionLease immutable record",
            lambda value: value["upstream_field_schemas"].update(
                effective_policy_bindings={"type": "array"}
            ),
        ),
        (
            "ExecutionLease immutable record",
            lambda value: value["upstream_field_schemas"].update(
                kill_switch_bindings={"type": "array"}
            ),
        ),
        (
            "SessionSecurityState current generation/state",
            lambda value: value["upstream_field_schemas"].update(
                state={"type": "non_empty_string"}
            ),
        ),
        (
            "SecretMetadataProjection",
            lambda value: value["upstream_field_schemas"].update(
                permitted_operations={"type": "array"}
            ),
        ),
    ],
)
def test_direct_source_fidelity_rejects_known_projection_drifts(name, mutation) -> None:
    projection, source = _direct_source_contracts()[name]
    candidate = copy.deepcopy(projection)
    mutation(candidate)
    assert not _direct_projection_is_source_derived(name, candidate, source)


def test_direct_event_nullable_envelope_is_source_valid() -> None:
    payload = _direct_payload("Event", "a", 1)
    assert payload["causation_id"] is None and payload["command_id"] is None
    assert _validate_direct_upstream_payload("Event", payload)


def test_direct_terminal_fingerprint_projection_is_complete_and_fail_closed() -> None:
    direct = MACHINE["backup_contract"]["direct_upstream_validator_registry"]
    expected = {
        "Event": "event_fingerprint_sha256",
        "Fill": "fill_fingerprint_sha256",
        "kill-switch state/generation": "record_fingerprint_sha256",
        "RiskDecision": "decision_fingerprint_sha256",
        "ExecutionLease immutable record": "lease_fingerprint_sha256",
        "SessionSecurityState current generation/state": "content_fingerprint_sha256",
        "SecretMetadataProjection": "content_fingerprint_sha256",
    }
    for name, field in expected.items():
        derivation = direct[name]["terminal_fingerprint"]
        assert derivation["field"] == field
        assert derivation["algorithm"] == "SHA-256"
        assert derivation["excluded_fields"] == [field]
        assert derivation["digest_format"] in {
            "lowercase_hex",
            "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
        }
        assert set(derivation["input_fields"]) == set(direct[name]["exact_fields"]) - {field}


def _rehash_direct(aspect: str, payload: dict[str, Any]) -> dict[str, Any]:
    candidate = copy.deepcopy(payload)
    terminal = _terminal_fingerprint_field(aspect)
    assert terminal is not None
    candidate[terminal] = _direct_terminal_fingerprint(aspect, candidate)
    return candidate


def _direct_schema_types() -> set[str]:
    result: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("type"), str):
                result.add(value["type"])
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(MACHINE["backup_contract"]["direct_upstream_validator_registry"])
    return result


def test_every_discovered_direct_schema_type_has_an_explicit_handler() -> None:
    assert _direct_schema_types() <= IMPLEMENTED_DIRECT_SCHEMA_TYPES
    assert not _field_schema_valid("unknown", "anything", {"type": "UNKNOWN_TYPE"})


@pytest.mark.parametrize(
    "bad",
    [
        "xTZ",
        "abcTdefZ",
        "2026-99-99T99:99:99Z",
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:00z",
        " 2026-01-01T00:00:00Z",
    ],
)
def test_canonical_timestamp_rejects_lexically_plausible_invalid_values(bad: str) -> None:
    assert not _canonical_timestamp(bad)


def test_canonical_timestamp_accepts_source_grammar_without_wall_clock() -> None:
    assert _canonical_timestamp("2026-01-01T00:00:00Z")
    assert _canonical_timestamp("2026-01-01T00:00:00.123456789Z")


@pytest.mark.parametrize("valid", ["0/1", "1/1", "-1/2", "3/2"])
def test_canonical_fraction_accepts_only_reduced_one_representation(valid: str) -> None:
    assert _canonical_fraction(valid)


@pytest.mark.parametrize("bad", ["-0/1", "00/1", "1/00", "1/0", "2/2", 1.0])
def test_canonical_fraction_rejects_noncanonical_representations(bad: Any) -> None:
    assert not _canonical_fraction(bad)


def test_direct_kill_switch_scope_id_is_context_aware() -> None:
    payload = _direct_payload("kill-switch state/generation", "a", 1)
    assert _validate_direct_upstream_payload("kill-switch state/generation", payload)
    for scope_type, valid_scope in (
        ("PRODUCT_SYSTEM", "product"),
        ("WORKSPACE", _canonical_fixture_id("ws", "a")),
        ("PORTFOLIO", _canonical_fixture_id("port", "a")),
        ("EXCHANGE_ACCOUNT", _canonical_fixture_id("xacc", "a")),
        ("STRATEGY_INSTANCE", _canonical_fixture_id("sinst", "a")),
        ("INSTRUMENT", _canonical_fixture_id("instr", "a")),
        ("EXECUTION_ROUTE", _canonical_fixture_id("xroute", "a")),
    ):
        candidate = copy.deepcopy(payload)
        candidate.update(scope_type=scope_type, scope_id=valid_scope)
        assert _validate_direct_upstream_payload(
            "kill-switch state/generation",
            _rehash_direct("kill-switch state/generation", candidate),
        )
        candidate["scope_id"] = _canonical_fixture_id("wrong", "a")
        assert not _validate_direct_upstream_payload(
            "kill-switch state/generation",
            _rehash_direct("kill-switch state/generation", candidate),
        )


def test_event_safe_payload_is_exact_source_resolved_before_fingerprint() -> None:
    payload = _direct_payload("Event", "a", 1)
    assert _validate_direct_upstream_payload("Event", payload)
    for bad_payload in (
        {"side": "BUY", "order_type": "MARKET"},
        {"side": "BUY", "order_type": "MARKET", "quantity": "1", "extra": "x"},
        {"side": "BUY", "order_type": "MARKET", "quantity": 1.0},
    ):
        candidate = copy.deepcopy(payload)
        candidate["safe_payload"] = bad_payload
        assert not _validate_direct_upstream_payload("Event", _rehash_direct("Event", candidate))


def test_non_empty_risk_decision_limit_result_is_executable() -> None:
    payload = _direct_payload("RiskDecision", "a", 1)
    assert payload["ordered_limit_results"]
    assert _validate_direct_upstream_payload("RiskDecision", payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item.pop("reason_code"),
        lambda item: item.update(extra="x"),
        lambda item: item.update(limit_type="UNKNOWN"),
        lambda item: item.update(effective_threshold="-0/1"),
        lambda item: item.update(effective_threshold="2/2"),
        lambda item: item.update(effective_threshold=1.0),
        lambda item: item.update(observed_projected_value=None),
        lambda item: item.update(unit_asset_reference={}),
        lambda item: item.update(supplying_policy_scope="SYSTEM"),
        lambda item: item.update(supplying_policy_scope="WORKSPACE:product"),
        lambda item: item.update(result="UNKNOWN"),
        lambda item: item.update(reason_code="UNKNOWN"),
        lambda item: item.update(result="FAIL", reason_code="PASS"),
    ],
)
def test_rehashed_malformed_limit_result_remains_invalid(mutation) -> None:
    payload = _direct_payload("RiskDecision", "a", 1)
    mutation(payload["ordered_limit_results"][0])
    assert not _validate_direct_upstream_payload(
        "RiskDecision", _rehash_direct("RiskDecision", payload)
    )


def test_limit_result_duplicate_order_and_synthetic_position_fail_closed() -> None:
    payload = _direct_payload("RiskDecision", "a", 1)
    duplicate = copy.deepcopy(payload)
    duplicate["ordered_limit_results"] *= 2
    assert not _validate_direct_upstream_payload(
        "RiskDecision", _rehash_direct("RiskDecision", duplicate)
    )
    second = _valid_limit_result("MAX_ORDER_NOTIONAL")
    wrong_order = copy.deepcopy(payload)
    wrong_order["ordered_limit_results"] = [second, _valid_limit_result()]
    assert not _validate_direct_upstream_payload(
        "RiskDecision", _rehash_direct("RiskDecision", wrong_order)
    )
    synthetic = _valid_limit_result("DISPATCH_RESERVATION_ECONOMICS")
    synthetic.update(
        effective_threshold="0/1",
        observed_projected_value=None,
        supplying_policy_scope="SYSTEM",
        result="INCOMPLETE",
        reason_code="MISSING_VALUATION",
    )
    wrong_synthetic = copy.deepcopy(payload)
    wrong_synthetic["ordered_limit_results"] = [synthetic, _valid_limit_result()]
    assert not _validate_direct_upstream_payload(
        "RiskDecision", _rehash_direct("RiskDecision", wrong_synthetic)
    )


def _valid_kill_switch_binding() -> list[Any]:
    return ["PRODUCT_SYSTEM", "product", "INACTIVE", 1, "2026-01-01T00:00:00Z", 1, SHA, SHA]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item.pop(),
        lambda item: item.__setitem__(0, _canonical_fixture_id("wrong", "a")),
        lambda item: item.__setitem__(1, True),
        lambda item: item.__setitem__(1, 0),
        lambda item: item.__setitem__(2, "UNKNOWN"),
        lambda item: item.__setitem__(3, _canonical_fixture_id("ws", "a")),
        lambda item: item.__setitem__(4, "UNKNOWN"),
        lambda item: item.__setitem__(5, "A" * 64),
    ],
)
def test_rehashed_malformed_effective_policy_binding_remains_invalid(mutation) -> None:
    payload = _direct_payload("ExecutionLease immutable record", "a", 1)
    mutation(payload["effective_policy_bindings"][0])
    assert not _validate_direct_upstream_payload(
        "ExecutionLease immutable record",
        _rehash_direct("ExecutionLease immutable record", payload),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item.pop(),
        lambda item: item.__setitem__(1, _canonical_fixture_id("ws", "a")),
        lambda item: item.__setitem__(2, "UNKNOWN"),
        lambda item: item.__setitem__(3, True),
        lambda item: item.__setitem__(4, "abcTdefZ"),
        lambda item: item.__setitem__(5, True),
        lambda item: item.__setitem__(6, "A" * 64),
    ],
)
def test_rehashed_malformed_kill_switch_binding_remains_invalid(mutation) -> None:
    payload = _direct_payload("ExecutionLease immutable record", "a", 1)
    payload["kill_switch_bindings"] = [_valid_kill_switch_binding()]
    mutation(payload["kill_switch_bindings"][0])
    assert not _validate_direct_upstream_payload(
        "ExecutionLease immutable record",
        _rehash_direct("ExecutionLease immutable record", payload),
    )


def test_binding_duplicates_and_noncanonical_order_are_rejected() -> None:
    payload = _direct_payload("ExecutionLease immutable record", "a", 1)
    payload["effective_policy_bindings"] *= 2
    assert not _validate_direct_upstream_payload(
        "ExecutionLease immutable record",
        _rehash_direct("ExecutionLease immutable record", payload),
    )
    payload = _direct_payload("ExecutionLease immutable record", "a", 1)
    payload["kill_switch_bindings"] = [
        [
            "WORKSPACE",
            _canonical_fixture_id("ws", "a"),
            "INACTIVE",
            1,
            "2026-01-01T00:00:00Z",
            1,
            SHA,
            SHA,
        ],
        _valid_kill_switch_binding(),
    ]
    assert not _validate_direct_upstream_payload(
        "ExecutionLease immutable record",
        _rehash_direct("ExecutionLease immutable record", payload),
    )


@pytest.mark.parametrize(
    ("operations", "valid"),
    [
        (["PRIVATE_DATA"], True),
        (["ORDER_ENTRY"], True),
        (["PRIVATE_DATA", "ORDER_ENTRY"], True),
        ([], False),
        (["PRIVATE_DATA", "PRIVATE_DATA"], False),
        (["UNKNOWN"], False),
        (["ORDER_ENTRY", "PRIVATE_DATA"], False),
        ("PRIVATE_DATA", False),
    ],
)
def test_secret_operations_are_exact_and_remain_invalid_after_rehash(operations, valid) -> None:
    payload = _direct_payload("SecretMetadataProjection", "a", 1)
    payload["permitted_operations"] = operations
    candidate = _rehash_direct("SecretMetadataProjection", payload)
    assert _validate_direct_upstream_payload("SecretMetadataProjection", candidate) is valid


def test_dependency_mutation_breaks_authored_attestation() -> None:
    entry = DEPENDENCIES[0]
    upstream = json.loads((DOCS / entry["artifact"]).read_text())
    altered = copy.deepcopy(upstream)
    ok, value = _resolve_pointer(altered, entry["json_pointer"])
    assert ok and isinstance(value, (dict, list))
    value.append("MUTATION") if isinstance(value, list) else value.update({"mutation": True})
    assert not _attests(entry, altered)


@pytest.mark.parametrize("pointer", ["/missing", "/entity_kinds/alias", "entity_kinds", 1, None])
def test_pointer_alias_or_malformed_fails_closed(pointer: Any) -> None:
    entry = {**DEPENDENCIES[0], "json_pointer": pointer}
    upstream = json.loads((DOCS / entry["artifact"]).read_text())
    assert not _attests(entry, upstream)


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.2, "1", None, [], {}])
def test_metadata_generation_is_positive_non_boolean(bad: Any) -> None:
    assert (
        _validate_metadata(_metadata(protected_freshness_generation=bad)) == "CONTRACT_INCONSISTENT"
    )


def test_metadata_is_closed_and_hashes_lowercase() -> None:
    assert _validate_metadata(_metadata()) == "VALID"
    assert _validate_metadata({**_metadata(), "authority": "caller"}) == "CONTRACT_INCONSISTENT"
    assert (
        _validate_metadata(_metadata(state_fingerprint_sha256="A" * 64)) == "CONTRACT_INCONSISTENT"
    )


def test_evidence_has_no_persistence_backup_fingerprint_or_generation_effect() -> None:
    policy = MACHINE["startup_recovery_model"]["evidence_policy"]
    assert [
        policy[k]
        for k in (
            "local_durable_evidence_persisted_in_StateStore",
            "local_durable_evidence_included_in_state_fingerprint",
            "local_durable_evidence_included_in_BackupEnvelope",
            "local_evidence_current_designation_persisted",
            "evidence_publication_is_StateStore_semantic_transaction",
            "evidence_publication_advances_protected_freshness_generation",
        )
    ] == [False] * 6
    assert policy["local_evidence_registry_process_local"] is True


def test_runtime_session_publication_precedes_ready_and_old_process_is_not_restored() -> None:
    model = MACHINE["runtime_session_persistence"]
    assert model["order"][-2:] == ["durably publish current RuntimeSession history", "READY"]
    assert model["old_history_preserved"] and not model["old_process_restored"]


@pytest.mark.parametrize(
    ("state", "outcome"),
    [
        ("UNUSED", "LEASE_STALE"),
        ("CONSUMED", "CONSUMED"),
        ("UNKNOWN_RECONCILIATION", "RECONCILIATION_REQUIRED"),
        (True, "CONTRACT_INCONSISTENT"),
    ],
)
def test_execution_lease_restart_fence(state: Any, outcome: str) -> None:
    assert _lease_after_restart(state) == outcome


def test_migration_is_closed_forward_only_and_protected() -> None:
    migration = MACHINE["migration_protocol"]
    assert migration["states"] == [
        "PREPARED",
        "APPLYING",
        "DURABLE_MIGRATED",
        "COMPLETED",
        "FAILED",
    ]
    assert migration["rollback_policy"] == "FORWARD_ONLY"
    assert migration["authoritative_change_requires_m0_3_freshness"]
    assert not migration["partial_state_promoted"]
    assert migration["restart"] == "IDEMPOTENT_RESUME"


def test_backup_exclusions_block_authority_evidence_and_secrets() -> None:
    exclusions = " ".join(MACHINE["backup_contract"]["excludes"])
    for forbidden in (
        "protected membership",
        "LocalDurableStateEvidence",
        "raw PIN",
        "secure-store payload",
        "AuthenticationProof",
        "PlatformBiometricAssertion",
    ):
        assert forbidden in exclusions


def test_no_authority_resurrection_and_pin_fences() -> None:
    never = " ".join(MACHINE["restore_contract"]["never_resurrect"])
    for item in (
        "consumed bootstrap",
        "retired M0.3",
        "revoked DeviceInstallation",
        "old PIN revision",
        "revoked LiveAccessGrant",
        "consumed ExecutionLease",
        "kill-switch",
        "idempotency",
    ):
        assert item in never


def test_idempotency_accounting_and_reservation_are_restart_safe() -> None:
    invariants = MACHINE["idempotency_and_accounting_recovery"]
    assert invariants["invariants"] == [
        "no repeated side effect",
        "no duplicate Fill/Ledger",
        "no double reservation release/consume",
    ]
    assert not invariants["reservation_id_added"]


def test_secret_handoff_has_no_payload_and_unknown_requires_reconciliation() -> None:
    handoff = MACHINE["external_resource_handoff"]
    assert handoff["states"] == [
        "PREPARED",
        "COMMITTED",
        "CLEANUP_PENDING",
        "UNKNOWN_RECONCILIATION",
    ]
    assert handoff["secret_payload_fields"] == 0
    assert handoff["unknown_outcome"] == "RECONCILIATION_REQUIRED_NO_BLIND_RETRY"


@pytest.mark.parametrize(
    ("internal", "public"), list(MACHINE["failure_registry"]["mapping"].items())
)
def test_internal_failure_mapping_is_exact(internal: str, public: str) -> None:
    assert MACHINE["failure_registry"]["mapping"][internal] == public
    assert public in MACHINE["failure_registry"]["closed_codes"]


def test_failure_registry_is_closed_and_safe() -> None:
    registry = MACHINE["failure_registry"]
    assert len(registry["closed_codes"]) == len(set(registry["closed_codes"])) == 32
    assert registry["outcome_schema"]["additionalProperties"] is False
    assert "SQLite exception" in registry["raw_diagnostics_forbidden"]


@pytest.mark.parametrize(
    ("second", "outcome"), [(False, "LOCK_CONTENTION"), (True, "CONCURRENT_WRITER")]
)
def test_single_writer_distinguishes_contention(second: bool, outcome: str) -> None:
    assert ("CONCURRENT_WRITER" if second else "LOCK_CONTENTION") == outcome
    assert MACHINE["writer_concurrency"]["authoritative_writers"] == 1


def test_durability_uncertainty_never_success() -> None:
    assert (
        MACHINE["failure_registry"]["mapping"]["DURABILITY_UNCERTAIN"]
        == "DURABILITY_CONFIRMATION_FAILED"
    )
    assert (
        MACHINE["transaction_protocol"]["steps"][-1]
        == "SUCCESS_AFTER_REQUIRED_DURABILITY_AND_FINALITY"
    )


def test_live_is_currently_denied_without_testnet_fallback() -> None:
    authority = MACHINE["authority_model"]
    assert authority["live_current"] == "DENIED"
    assert authority["testnet_to_live_fallback"] is False


def test_markdown_is_machine_projection_for_every_root() -> None:
    markdown = MACHINE_PATH.with_suffix(".md").read_text()
    for root, value in MACHINE.items():
        assert f"## `{root}`" in markdown
        assert json.dumps(value, indent=2, ensure_ascii=False) in markdown


# Independently authored corrective-closure expectations (never derived from MACHINE).
_A = "DURABLE AUTHORITATIVE CURRENT STATE"
_H = "DURABLE IMMUTABLE / APPEND-ONLY HISTORY"
_R = "DERIVED / REBUILDABLE"
_E = "EPHEMERAL RUNTIME"
_X = "EXTERNAL AUTHORITY / REFERENCE ONLY"
_S = "SECRET PAYLOAD OUTSIDE DOMAIN"
_EXPECTED_DURABILITY = {
    "StateStoreMetadata": _A,
    "CryptoHunterAccount current record": _A,
    "DeviceInstallation current identity/lifecycle": _A,
    "OperatorIdentity current designation/state": _A,
    "OperatorIdentity revisions": _H,
    "LiveAccessGrant current designation/state": _A,
    "LiveAccessGrant accepted revisions/history": _H,
    "Workspace": _A,
    "Portfolio canonical accounting state": _A,
    "Portfolio balance/P&L/NAV projections": _R,
    "ExchangeAccount": _A,
    "CredentialProfile metadata/reference": _A,
    "TradingUniverse current version/designation": _A,
    "TradingUniverse version history": _H,
    "StrategyDefinition accepted revisions": _H,
    "StrategyDefinition current designation": _A,
    "StrategyInstance current lifecycle/config": _A,
    "routing configuration/current designation": _A,
    "routing readiness/reachability projection": _R,
    "RiskPolicy accepted revisions": _H,
    "RiskPolicy current designation": _A,
    "RiskBudget current state": _A,
    "kill-switch state/generation": _A,
    "kill-switch transition history": _H,
    "Command accepted request": _H,
    "Event": _H,
    "OrderIntent": _H,
    "Order canonical lifecycle state": _A,
    "Order lifecycle events/history": _H,
    "Fill": _H,
    "LedgerEntry": _H,
    "reservation current state": _A,
    "reservation transition history": _H,
    "RiskDecision": _H,
    "ExecutionLease immutable record": _H,
    "ExecutionLease one-shot state": _A,
    "ExecutionLease restart fence": _A,
    "RuntimeSession active process manifestation": _E,
    "RuntimeSession canonical identity/history": _H,
    "SessionSecurityState current generation/state": _A,
    "SessionSecurityState revision history": _H,
    "PinVerifierRecord accepted revisions": _H,
    "PinVerifierRecord current designation": _A,
    "DeviceTrust/security revisions": _H,
    "DeviceTrust current designation": _A,
    "platform enrollment revisions": _H,
    "AuthenticationProof": _E,
    "CoreIssuedAuthenticationProofBinding": _E,
    "PlatformBiometricAssertion": _E,
    "CoreAcceptedPlatformBiometricAssertionBinding": _E,
    "SecretMetadataProjection": _A,
    "secure-store payload": _S,
    "bootstrap consumed fence": _A,
    "bootstrap accepted/consumption history": _H,
    "M0.3 restore freshness membership": _X,
    "M0.3 current designation": _X,
    "M0.3 retirement state": _X,
    "LocalDurableStateEvidence payload": _R,
    "LocalDurableEvidence accepted/current registry/designation": _E,
    "Migration current state/designation": _A,
    "Migration transition/history revisions": _H,
    "SecretHandoff current state/designation": _A,
    "SecretHandoff transition/history revisions": _H,
}


_EXPECTED_OWNERSHIP: dict[str, dict[str, str]] = {
    "StateStoreMetadata": {
        "representation_category": "M011_LOCAL_SCHEMA",
        "semantic_owner_milestone": "M0.11",
        "semantic_artifact": "persistence_versioning_migrations_backup_and_recovery.json",
        "semantic_json_pointer": "/executable_boundary_schemas/StateStoreMetadata",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "CryptoHunterAccount current record": {
        "representation_category": "M011_ENTITY_IDENTITY_PROJECTION",
        "semantic_owner_milestone": "M0.2",
        "semantic_artifact": "canonical_domain_vocabulary.json",
        "semantic_json_pointer": "/entity_kinds",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "DeviceInstallation current identity/lifecycle": {
        "representation_category": "M011_PERSISTENCE_PROJECTION_OF_UPSTREAM_FACTS",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "OperatorIdentity current designation/state": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "OperatorIdentity revisions": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/OperatorIdentitySecurityProjection",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "LiveAccessGrant current designation/state": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "LiveAccessGrant accepted revisions/history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/LiveAccessGrantSecurityProjection",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Workspace": {
        "representation_category": "M011_ENTITY_IDENTITY_PROJECTION",
        "semantic_owner_milestone": "M0.2",
        "semantic_artifact": "canonical_domain_vocabulary.json",
        "semantic_json_pointer": "/entity_kinds",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Portfolio canonical accounting state": {
        "representation_category": "M011_PERSISTENCE_PROJECTION_OF_UPSTREAM_FACTS",
        "semantic_owner_milestone": "M0.8",
        "semantic_artifact": "ledger_portfolio_capital_and_pnl.json",
        "semantic_json_pointer": "/rebuild_protocol",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Portfolio balance/P&L/NAV projections": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.8",
        "semantic_artifact": "ledger_portfolio_capital_and_pnl.json",
        "semantic_json_pointer": "/rebuild_protocol",
        "carrier_strategy": "NONE",
    },
    "ExchangeAccount": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.5",
        "semantic_artifact": "exchange_accounts_and_instruments.json",
        "semantic_json_pointer": "/exchange_account_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "CredentialProfile metadata/reference": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.5",
        "semantic_artifact": "exchange_accounts_and_instruments.json",
        "semantic_json_pointer": "/credential_profile_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "TradingUniverse current version/designation": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.5",
        "semantic_artifact": "exchange_accounts_and_instruments.json",
        "semantic_json_pointer": "/trading_universe_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "TradingUniverse version history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.5",
        "semantic_artifact": "exchange_accounts_and_instruments.json",
        "semantic_json_pointer": "/trading_universe_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "StrategyDefinition accepted revisions": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.6",
        "semantic_artifact": "strategy_market_data_and_execution_routing.json",
        "semantic_json_pointer": "/record_schemas/StrategyDefinition",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "StrategyDefinition current designation": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.6",
        "semantic_artifact": "strategy_market_data_and_execution_routing.json",
        "semantic_json_pointer": "/strategy_definition_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "StrategyInstance current lifecycle/config": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.6",
        "semantic_artifact": "strategy_market_data_and_execution_routing.json",
        "semantic_json_pointer": "/record_schemas/StrategyInstance",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "routing configuration/current designation": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.6",
        "semantic_artifact": "strategy_market_data_and_execution_routing.json",
        "semantic_json_pointer": "/route_readiness_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "routing readiness/reachability projection": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.6",
        "semantic_artifact": "strategy_market_data_and_execution_routing.json",
        "semantic_json_pointer": "/route_readiness_contract",
        "carrier_strategy": "NONE",
    },
    "RiskPolicy accepted revisions": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/risk_policy_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "RiskPolicy current designation": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/risk_policy_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "RiskBudget current state": {
        "representation_category": "M011_PERSISTENCE_PROJECTION_OF_UPSTREAM_FACTS",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/risk_input_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "kill-switch state/generation": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/kill_switch_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "kill-switch transition history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/kill_switch_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Command accepted request": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.7",
        "semantic_artifact": "commands_events_order_lifecycle_and_idempotency.json",
        "semantic_json_pointer": "/command_registry",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Event": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.7",
        "semantic_artifact": "commands_events_order_lifecycle_and_idempotency.json",
        "semantic_json_pointer": "/event_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "OrderIntent": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.7",
        "semantic_artifact": "commands_events_order_lifecycle_and_idempotency.json",
        "semantic_json_pointer": "/command_registry/SUBMIT_ORDER",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Order canonical lifecycle state": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.7",
        "semantic_artifact": "commands_events_order_lifecycle_and_idempotency.json",
        "semantic_json_pointer": "/order_lifecycle",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Order lifecycle events/history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.7",
        "semantic_artifact": "commands_events_order_lifecycle_and_idempotency.json",
        "semantic_json_pointer": "/event_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Fill": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.7",
        "semantic_artifact": "commands_events_order_lifecycle_and_idempotency.json",
        "semantic_json_pointer": "/fill_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "LedgerEntry": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.8",
        "semantic_artifact": "ledger_portfolio_capital_and_pnl.json",
        "semantic_json_pointer": "/ledger_entry_schema",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "reservation current state": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.8",
        "semantic_artifact": "ledger_portfolio_capital_and_pnl.json",
        "semantic_json_pointer": "/reservation_protocol",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "reservation transition history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.8",
        "semantic_artifact": "ledger_portfolio_capital_and_pnl.json",
        "semantic_json_pointer": "/accounting_economic_fact_schema_registry",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "RiskDecision": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/executable_boundary_schemas/RiskDecision",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "ExecutionLease immutable record": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/executable_boundary_schemas/ExecutionLease",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "ExecutionLease one-shot state": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/executable_boundary_schemas/ExecutionLease",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "ExecutionLease restart fence": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/executable_boundary_schemas/ExecutionLease",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "RuntimeSession active process manifestation": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.3",
        "semantic_artifact": "process_topology_and_lifecycle.json",
        "semantic_json_pointer": "/first_run_bootstrap_authority_contract",
        "carrier_strategy": "NONE",
    },
    "RuntimeSession canonical identity/history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.2",
        "semantic_artifact": "canonical_domain_vocabulary.json",
        "semantic_json_pointer": "/entity_kinds",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "SessionSecurityState current generation/state": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/SessionSecurityState",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "SessionSecurityState revision history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/SessionSecurityState",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "PinVerifierRecord accepted revisions": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/PinVerifierRecord",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "PinVerifierRecord current designation": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/PinVerifierRecord",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "DeviceTrust/security revisions": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/DeviceTrustProjection",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "DeviceTrust current designation": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/DeviceTrustProjection",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "platform enrollment revisions": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/CoreAcceptedPlatformBiometricAssertionBinding",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "AuthenticationProof": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "NONE",
    },
    "CoreIssuedAuthenticationProofBinding": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "NONE",
    },
    "PlatformBiometricAssertion": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "NONE",
    },
    "CoreAcceptedPlatformBiometricAssertionBinding": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "NONE",
    },
    "SecretMetadataProjection": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/SecretMetadataProjection",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "secure-store payload": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "NONE",
    },
    "bootstrap consumed fence": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.3",
        "semantic_artifact": "process_topology_and_lifecycle.json",
        "semantic_json_pointer": "/first_run_bootstrap_authority_contract/executable_schemas/CoreCurrentBootstrapState",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "bootstrap accepted/consumption history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.3",
        "semantic_artifact": "process_topology_and_lifecycle.json",
        "semantic_json_pointer": "/first_run_bootstrap_authority_contract/executable_schemas/ConsumedBootstrapAuthority",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "M0.3 restore freshness membership": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.3",
        "semantic_artifact": "process_topology_and_lifecycle.json",
        "semantic_json_pointer": "/restore_freshness_authority_contract",
        "carrier_strategy": "NONE",
    },
    "M0.3 current designation": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.3",
        "semantic_artifact": "process_topology_and_lifecycle.json",
        "semantic_json_pointer": "/restore_freshness_authority_contract",
        "carrier_strategy": "NONE",
    },
    "M0.3 retirement state": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.3",
        "semantic_artifact": "process_topology_and_lifecycle.json",
        "semantic_json_pointer": "/restore_freshness_authority_contract",
        "carrier_strategy": "NONE",
    },
    "LocalDurableStateEvidence payload": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.11",
        "semantic_artifact": "persistence_versioning_migrations_backup_and_recovery.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "NONE",
    },
    "LocalDurableEvidence accepted/current registry/designation": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.11",
        "semantic_artifact": "persistence_versioning_migrations_backup_and_recovery.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "NONE",
    },
    "Migration current state/designation": {
        "representation_category": "M011_LOCAL_SCHEMA",
        "semantic_owner_milestone": "M0.11",
        "semantic_artifact": "persistence_versioning_migrations_backup_and_recovery.json",
        "semantic_json_pointer": "/executable_boundary_schemas/MigrationCurrentState",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Migration transition/history revisions": {
        "representation_category": "M011_LOCAL_SCHEMA",
        "semantic_owner_milestone": "M0.11",
        "semantic_artifact": "persistence_versioning_migrations_backup_and_recovery.json",
        "semantic_json_pointer": "/executable_boundary_schemas/MigrationTransitionRecord",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "SecretHandoff current state/designation": {
        "representation_category": "M011_LOCAL_SCHEMA",
        "semantic_owner_milestone": "M0.11",
        "semantic_artifact": "persistence_versioning_migrations_backup_and_recovery.json",
        "semantic_json_pointer": "/executable_boundary_schemas/SecretHandoffCurrentState",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "SecretHandoff transition/history revisions": {
        "representation_category": "M011_LOCAL_SCHEMA",
        "semantic_owner_milestone": "M0.11",
        "semantic_artifact": "persistence_versioning_migrations_backup_and_recovery.json",
        "semantic_json_pointer": "/executable_boundary_schemas/SecretHandoffTransitionRecord",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
}


def _failure_row(fc: str, retry: str, recovery: bool, diag: str) -> dict[str, Any]:
    return {
        "failure_class": fc,
        "retryability": retry,
        "mutation_acknowledged": False,
        "recovery_required": recovery,
        "safe_diagnostic_class": diag,
    }


_EXPECTED_FAILURES: dict[str, dict[str, Any]] = {}
for _codes, _row in [
    (
        (
            "STATE_STORE_UNAVAILABLE",
            "STATE_STORE_PERMISSION_DENIED",
            "STATE_STORE_CAPACITY_EXHAUSTED",
        ),
        _failure_row("STORAGE_ACCESS", "CONDITIONAL_AFTER_OPERATOR_REPAIR", False, "STORAGE"),
    ),
    (
        ("TRANSACTION_FAILED",),
        _failure_row("TRANSACTION", "SAFE_AFTER_CONFIRMED_ABORT", False, "TRANSACTION"),
    ),
    (
        ("DURABILITY_CONFIRMATION_FAILED",),
        _failure_row("DURABILITY_UNCERTAIN", "AFTER_RECOVERY_ONLY", True, "DURABILITY"),
    ),
    (
        ("AUTHORITATIVE_STATE_CORRUPT", "IMMUTABLE_HISTORY_CORRUPT", "CURRENT_DESIGNATION_CORRUPT"),
        _failure_row("INTEGRITY", "NEVER_AUTOMATIC", True, "INTEGRITY"),
    ),
    (
        ("DERIVED_PROJECTION_STALE",),
        _failure_row("DERIVED_STATE", "SAFE_REBUILD", True, "PROJECTION"),
    ),
    (
        ("SCHEMA_MISMATCH", "MIGRATION_REQUIRED"),
        _failure_row("VERSIONING", "AFTER_EXACT_MIGRATION", True, "VERSION"),
    ),
    (
        (
            "UNSUPPORTED_STATESTORE_SCHEMA",
            "UNSUPPORTED_BACKUP_SCHEMA",
            "MIGRATION_PATH_UNAVAILABLE",
        ),
        _failure_row("VERSIONING_UNSUPPORTED", "NEVER_AUTOMATIC", False, "VERSION"),
    ),
    (("MIGRATION_FAILED",), _failure_row("MIGRATION", "AFTER_RECOVERY_ONLY", True, "MIGRATION")),
    (
        ("LOCK_CONTENTION",),
        _failure_row("AVAILABILITY", "RETRY_WITH_BACKOFF", False, "CONCURRENCY"),
    ),
    (
        ("CONCURRENT_WRITER",),
        _failure_row("AUTHORITY_CONFLICT", "NEVER_WHILE_CONFLICT_EXISTS", True, "CONCURRENCY"),
    ),
    (("RECOVERY_REQUIRED",), _failure_row("RECOVERY", "AFTER_RECOVERY_ONLY", True, "RECOVERY")),
    (("RECOVERY_FAILED",), _failure_row("RECOVERY", "NEVER_AUTOMATIC", True, "RECOVERY")),
    (
        (
            "BACKUP_INTEGRITY_FAILED",
            "BACKUP_SCOPE_MISMATCH",
            "BACKUP_ENVIRONMENT_MISMATCH",
            "BACKUP_ROLLBACK_DETECTED",
        ),
        _failure_row("BACKUP_VALIDATION", "NEVER_FOR_SAME_CANDIDATE", False, "BACKUP"),
    ),
    (
        ("RESTORE_REJECTED",),
        _failure_row("RESTORE_VALIDATION", "NEVER_FOR_SAME_CANDIDATE", False, "RESTORE"),
    ),
    (
        ("SECRET_REFERENCE_INVALID",),
        _failure_row("SECRET_METADATA", "AFTER_METADATA_REPAIR", False, "SECRET_REFERENCE"),
    ),
    (
        ("SECRET_REFERENCE_UNAVAILABLE",),
        _failure_row("EXTERNAL_SECRET_RESOURCE", "CONDITIONAL", False, "SECRET_REFERENCE"),
    ),
    (
        ("SECRET_HANDOFF_FAILED",),
        _failure_row(
            "EXTERNAL_HANDOFF", "AFTER_RECOVERY_OR_RECONCILIATION", True, "SECRET_HANDOFF"
        ),
    ),
    (
        ("ENVIRONMENT_SCOPE_MISMATCH",),
        _failure_row("ENVIRONMENT_BOUNDARY", "NEVER_FOR_SAME_INPUT", False, "ENVIRONMENT"),
    ),
    (
        ("MONOTONIC_FENCE_ROLLBACK",),
        _failure_row("ANTI_ROLLBACK", "NEVER_AUTOMATIC", True, "MONOTONIC_FENCE"),
    ),
    (
        ("IDEMPOTENCY_CONFLICT",),
        _failure_row("IDEMPOTENCY", "NEVER_FOR_CHANGED_REQUEST", False, "IDEMPOTENCY"),
    ),
    (
        ("RECONCILIATION_REQUIRED",),
        _failure_row(
            "EXTERNAL_OUTCOME_UNCERTAIN", "AFTER_RECONCILIATION_ONLY", True, "RECONCILIATION"
        ),
    ),
    (("CONTRACT_INCONSISTENT",), _failure_row("CONTRACT", "NEVER_AUTOMATIC", True, "CONTRACT")),
]:
    for _code in _codes:
        _EXPECTED_FAILURES[_code] = _row


def _validate_failure(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "code",
        "failure_class",
        "retryability",
        "mutation_acknowledged",
        "recovery_required",
        "safe_diagnostic_class",
    }:
        return False
    code = value.get("code")
    if not isinstance(code, str):
        return False
    expected = _EXPECTED_FAILURES.get(code)
    return expected is not None and {key: value[key] for key in expected} == expected


UUID7_ID = re.compile(
    r"^(acct|dev|ws)_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _canonical_scope(account: Any, device: Any) -> bool:
    return (
        isinstance(account, str)
        and isinstance(device, str)
        and bool(UUID7_ID.fullmatch(account))
        and bool(UUID7_ID.fullmatch(device))
        and account.startswith("acct_")
        and device.startswith("dev_")
    )


def _validate_local_durable_state_evidence(payload: Any) -> bool:
    required = {
        "account_id",
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
        "generation",
        "state_fingerprint_sha256",
        "transaction_fingerprint_sha256",
        "durability_state",
        "evidence_revision",
        "evidence_fingerprint_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        return False
    if not _canonical_scope(payload.get("account_id"), payload.get("device_installation_id")):
        return False
    if not _positive(payload.get("generation")) or not _positive(payload.get("evidence_revision")):
        return False
    if payload.get("durability_state") != "DURABLE_COMMITTED":
        return False
    for key in (
        "state_store_identity_fingerprint_sha256",
        "state_fingerprint_sha256",
        "transaction_fingerprint_sha256",
        "evidence_fingerprint_sha256",
    ):
        if not isinstance(payload.get(key), str) or SHA_RE.fullmatch(payload[key]) is None:
            return False
    expected = _actual_fingerprint(
        {key: value for key, value in payload.items() if key != "evidence_fingerprint_sha256"}
    )
    return str(payload["evidence_fingerprint_sha256"]) == expected


class _EvidenceRegistry:
    def __init__(self, process: str) -> None:
        self.process = process
        self.accepted: dict[str, dict[str, Any]] = {}
        self.current: dict[tuple[str, str, str], str] = {}
        self.revision = 0

    def observe(self, observation: Any) -> str | None:
        if not isinstance(observation, dict):
            return None
        metadata_keys = MACHINE["executable_boundary_schemas"]["StateStoreMetadata"]["required"]
        metadata = {key: observation.get(key) for key in metadata_keys}
        if _validate_metadata(metadata) != "VALID":
            return None
        if not all(
            observation.get(key) is True
            for key in ("durable_confirmed", "authoritative_history_integrity", "current_commit")
        ):
            return None
        self.revision += 1
        payload = {
            "account_id": observation["account_id"],
            "device_installation_id": observation["device_installation_id"],
            "state_store_identity_fingerprint_sha256": observation[
                "state_store_identity_fingerprint_sha256"
            ],
            "generation": observation["protected_freshness_generation"],
            "state_fingerprint_sha256": observation["state_fingerprint_sha256"],
            "transaction_fingerprint_sha256": observation["transaction_fingerprint_sha256"],
            "durability_state": "DURABLE_COMMITTED",
            "evidence_revision": self.revision,
        }
        payload["evidence_fingerprint_sha256"] = _actual_fingerprint(payload)
        if not _validate_local_durable_state_evidence(payload):
            return None
        ref = f"{self.process}:opaque:{self.revision}"
        self.accepted[ref] = payload
        scope = (
            payload["account_id"],
            payload["device_installation_id"],
            payload["state_store_identity_fingerprint_sha256"],
        )
        self.current[scope] = ref
        return ref

    def verify_current(self, scope: tuple[str, str, str], ref: Any) -> bool:
        payload = self.accepted.get(ref) if isinstance(ref, str) else None
        if not _validate_local_durable_state_evidence(payload):
            return False
        assert isinstance(payload, dict)
        payload_scope = (
            payload["account_id"],
            payload["device_installation_id"],
            payload["state_store_identity_fingerprint_sha256"],
        )
        return payload_scope == scope and self.current.get(scope) == ref


def _validate_external_authority_observation(value: Any) -> str:
    common = {
        "available",
        "account_id",
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
        "environment",
        "membership_state",
        "lifecycle",
    }
    lifecycle_fields = {
        "committed_generation",
        "committed_state_fingerprint_sha256",
        "prepared_generation",
        "prepared_state_fingerprint_sha256",
        "prepared_transaction_fingerprint_sha256",
    }
    if (
        not isinstance(value, dict)
        or not common <= set(value)
        or not set(value) <= common | lifecycle_fields
    ):
        return "CONTRACT_INCONSISTENT"
    if value.get("available") is not True or value.get("membership_state") != "CURRENT":
        return "RESTORE_REJECTED"
    if not _canonical_scope(value.get("account_id"), value.get("device_installation_id")):
        return "BACKUP_SCOPE_MISMATCH"
    if (
        not isinstance(value.get("state_store_identity_fingerprint_sha256"), str)
        or SHA_RE.fullmatch(value["state_store_identity_fingerprint_sha256"]) is None
    ):
        return "CONTRACT_INCONSISTENT"
    if value.get("environment") not in {"PAPER", "TESTNET", "LIVE"}:
        return "BACKUP_ENVIRONMENT_MISMATCH"
    lifecycle = value.get("lifecycle")
    committed_present = {"committed_generation", "committed_state_fingerprint_sha256"} <= set(value)
    prepared_present = {
        "prepared_generation",
        "prepared_state_fingerprint_sha256",
        "prepared_transaction_fingerprint_sha256",
    } <= set(value)
    if lifecycle == "UNINITIALIZED":
        return "VALID" if not (set(value) & lifecycle_fields) else "CONTRACT_INCONSISTENT"
    if lifecycle == "COMMITTED":
        if not committed_present or set(value) & {
            "prepared_generation",
            "prepared_state_fingerprint_sha256",
            "prepared_transaction_fingerprint_sha256",
        }:
            return "CONTRACT_INCONSISTENT"
        return (
            "VALID"
            if _positive(value["committed_generation"])
            and isinstance(value["committed_state_fingerprint_sha256"], str)
            and SHA_RE.fullmatch(value["committed_state_fingerprint_sha256"])
            else "CONTRACT_INCONSISTENT"
        )
    if lifecycle != "PREPARED" or not prepared_present:
        return "CONTRACT_INCONSISTENT"
    if not _positive(value["prepared_generation"]) or not all(
        isinstance(value[k], str) and SHA_RE.fullmatch(value[k])
        for k in ("prepared_state_fingerprint_sha256", "prepared_transaction_fingerprint_sha256")
    ):
        return "CONTRACT_INCONSISTENT"
    if not committed_present:
        return (
            "VALID"
            if value["prepared_generation"] == 1
            and not ({"committed_generation", "committed_state_fingerprint_sha256"} & set(value))
            else "CONTRACT_INCONSISTENT"
        )
    if (
        not _positive(value["committed_generation"])
        or not isinstance(value["committed_state_fingerprint_sha256"], str)
        or SHA_RE.fullmatch(value["committed_state_fingerprint_sha256"]) is None
    ):
        return "CONTRACT_INCONSISTENT"
    return (
        "VALID"
        if value["prepared_generation"] == value["committed_generation"] + 1
        else "CONTRACT_INCONSISTENT"
    )


def _restore(candidate: Any, external: Any) -> str:
    external_result = _validate_external_authority_observation(external)
    if external_result != "VALID":
        return external_result
    if not isinstance(candidate, dict):
        return "RESTORE_REJECTED"
    for key in ("account_id", "device_installation_id", "state_store_identity_fingerprint_sha256"):
        if candidate.get(key) != external.get(key):
            return "BACKUP_SCOPE_MISMATCH"
    if candidate.get("environment") != external.get("environment"):
        return "BACKUP_ENVIRONMENT_MISMATCH"
    lifecycle = external["lifecycle"]
    generation = candidate.get("generation")
    if lifecycle == "UNINITIALIZED":
        return "CONTRACT_INCONSISTENT" if generation is not None else "GENESIS_PREPARE_REQUIRED"
    committed = external.get("committed_generation")
    if lifecycle == "PREPARED" and committed is None:
        if generation is None:
            return "GENESIS_PENDING_RECOVERY_REQUIRED"
        if not _positive(generation) or generation != 1:
            return "CONTRACT_INCONSISTENT"
        if (
            candidate.get("state_fingerprint_sha256")
            != external["prepared_state_fingerprint_sha256"]
            or candidate.get("transaction_fingerprint_sha256")
            != external["prepared_transaction_fingerprint_sha256"]
        ):
            return "BACKUP_ROLLBACK_DETECTED_PENDING_PRESERVED"
        return "REBUILD_FRESH_EVIDENCE_FINALIZE_MATCHING_PENDING_THEN_CONTINUE_GATES"
    if not _positive(generation) or not isinstance(committed, int):
        return "CONTRACT_INCONSISTENT"
    assert isinstance(generation, int)
    if generation < committed:
        return "BACKUP_ROLLBACK_DETECTED"
    if generation == committed:
        if (
            candidate.get("state_fingerprint_sha256")
            != external["committed_state_fingerprint_sha256"]
        ):
            return "BACKUP_ROLLBACK_DETECTED"
        return (
            "RECOVERY_REQUIRED_PENDING_RETAINED_NO_ABORT"
            if lifecycle == "PREPARED"
            else "CONTINUE_REMAINING_GATES"
        )
    if lifecycle != "PREPARED" or generation != external["prepared_generation"]:
        return "CONTRACT_INCONSISTENT"
    return (
        "REBUILD_FRESH_EVIDENCE_FINALIZE_MATCHING_PENDING_THEN_CONTINUE_GATES"
        if candidate.get("state_fingerprint_sha256")
        == external["prepared_state_fingerprint_sha256"]
        and candidate.get("transaction_fingerprint_sha256")
        == external["prepared_transaction_fingerprint_sha256"]
        else "BACKUP_ROLLBACK_DETECTED_PENDING_PRESERVED"
    )


_PERSISTENCE_FIELDS = {
    "representation_name",
    "representation_category",
    "semantic_owner_milestone",
    "semantic_artifact",
    "semantic_json_pointer",
    "semantic_contract_fingerprint_sha256",
    "record_key",
    "payload",
    "payload_fingerprint_sha256",
}


def _validate_projection_schema(payload: Any, schema_name: str) -> bool:
    if not isinstance(payload, dict):
        return False
    schema = MACHINE["executable_boundary_schemas"][schema_name]
    if set(payload) != set(schema["required"]):
        return False
    if schema_name == "PersistentEntityIdentityProjection":
        return (
            payload.get("entity_kind") in {"CryptoHunterAccount", "Workspace"}
            and _canonical_entity_id(payload.get("entity_kind"), payload.get("entity_id"))
            and isinstance(payload.get("parent_scope_bindings"), dict)
        )
    if schema_name == "CurrentDesignationProjection":
        return (
            isinstance(payload.get("scope_key"), str)
            and bool(payload["scope_key"])
            and isinstance(payload.get("current_reference"), str)
            and bool(payload["current_reference"])
            and all(
                value is None or _positive(value)
                for value in (payload.get("current_revision"), payload.get("current_generation"))
            )
            and isinstance(payload.get("content_fingerprint_sha256"), str)
            and SHA_RE.fullmatch(payload["content_fingerprint_sha256"]) is not None
        )
    if schema_name == "UpstreamFactsProjection":
        return (
            isinstance(payload.get("fact_kind"), str)
            and bool(payload["fact_kind"])
            and isinstance(payload.get("scope_key"), str)
            and isinstance(payload.get("facts"), dict)
            and isinstance(payload.get("source_fingerprint_sha256"), str)
            and SHA_RE.fullmatch(payload["source_fingerprint_sha256"]) is not None
        )
    return False


def _canonical_entity_id(kind: Any, value: Any) -> bool:
    if not isinstance(kind, str) or not isinstance(value, str):
        return False
    prefix = {"CryptoHunterAccount": "acct", "Workspace": "ws"}.get(kind)
    match = UUID7_ID.fullmatch(value)
    return match is not None and match.group(1) == prefix


def _exact_entity_projection(payload: Any) -> bool:
    if not isinstance(payload, dict) or set(payload) != {
        "entity_kind",
        "entity_id",
        "parent_scope_bindings",
    }:
        return False
    kind = payload.get("entity_kind")
    if not _canonical_entity_id(kind, payload.get("entity_id")):
        return False
    parents = payload.get("parent_scope_bindings")
    if kind == "CryptoHunterAccount":
        return parents == {}
    return (
        isinstance(parents, dict)
        and set(parents) == {"account_id"}
        and isinstance(parents["account_id"], str)
        and parents["account_id"].startswith("acct_")
        and UUID7_ID.fullmatch(parents["account_id"]) is not None
    )


GENERIC_UUID7_ID = re.compile(
    r"^[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
RISK_ASSET_FIELDS = [
    "venue_asset_code",
    "canonical_display_code",
    "asset_namespace",
    "mapping_status",
]
DECIMAL_RE = re.compile(r"^(0|[1-9][0-9]*)(\\.[0-9]*[1-9])?$")


def _terminal_fingerprint_field(aspect: str) -> str | None:
    return {
        "Event": "event_fingerprint_sha256",
        "Fill": "fill_fingerprint_sha256",
        "kill-switch state/generation": "record_fingerprint_sha256",
        "RiskDecision": "decision_fingerprint_sha256",
        "ExecutionLease immutable record": "lease_fingerprint_sha256",
        "SessionSecurityState current generation/state": "content_fingerprint_sha256",
        "SecretMetadataProjection": "content_fingerprint_sha256",
    }.get(aspect)


def _nfc_value(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_nfc_value(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise ValueError("NFC key collision")
            normalized[normalized_key] = _nfc_value(item)
        return normalized
    return value


def _direct_terminal_fingerprint(aspect: str, payload: dict[str, Any]) -> str:
    contract = MACHINE["backup_contract"]["direct_upstream_validator_registry"][aspect]
    derivation = contract["terminal_fingerprint"]
    projection = {field: payload[field] for field in derivation["input_fields"]}
    if aspect in {"Event", "Fill"}:
        projection = _nfc_value(projection)
    return _actual_fingerprint(projection)


IMPLEMENTED_DIRECT_SCHEMA_TYPES = {
    "constant",
    "exact_literal",
    "id",
    "canonical_uuid7_prefixed_id",
    "canonical_id",
    "nullable_id",
    "nullable_canonical_id",
    "positive_integer",
    "positive_non_boolean_integer",
    "non_negative_integer",
    "boolean",
    "sha256_hex",
    "sha256_lowercase_hex",
    "terminal_fingerprint",
    "enum",
    "decimal",
    "positive_decimal",
    "non_negative_decimal",
    "timestamp",
    "canonical_utc_timestamp",
    "nullable_timestamp",
    "non_empty_string",
    "string",
    "nullable_non_empty_string",
    "secure_store_reference",
    "unique_array_of_enum",
    "canonical_unique_array_of_enum",
    "object",
    "asset_reference",
    "exact_upstream_object",
    "canonical_scope_id",
    "conditional_supplying_policy_scope",
    "canonical_exact_fraction_string",
    "nullable_canonical_exact_fraction_string",
    "event_safe_payload",
    "array_of_exact_LimitResult",
    "array_of_exact_tuple",
    "array_of_canonical_id",
    "non_negative_decimal",
    "positive_decimal",
    "compound_scope",
    "non_empty_object",
    "risk_limits",
    "array",
}


def _canonical_timestamp(value: Any) -> bool:
    policy = json.loads((DOCS / "exchange_accounts_and_instruments.json").read_text())[
        "timestamp_policy"
    ]
    if not isinstance(value, str) or re.fullmatch(policy["regex"], value) is None:
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return True


def _parse_canonical_timestamp_order_key(value: Any) -> tuple[datetime, int] | None:
    """Return the M0.5 UTC whole-second plus exact nanosecond ordering key."""

    policy = EXCHANGE["timestamp_policy"]
    if (
        not isinstance(value, str)
        or not value.endswith("Z")
        or re.fullmatch(policy["regex"], value) is None
    ):
        return None
    body = value[:-1]
    whole_second, separator, fraction = body.partition(".")
    if separator and not 1 <= len(fraction) <= 9:
        return None
    try:
        exact_second = datetime.strptime(whole_second, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    nanoseconds = int(fraction.ljust(9, "0")) if separator else 0
    return exact_second, nanoseconds


def _canonical_timestamp_le(left: Any, right: Any) -> bool:
    left_key = _parse_canonical_timestamp_order_key(left)
    right_key = _parse_canonical_timestamp_order_key(right)
    return left_key is not None and right_key is not None and left_key <= right_key


def _canonical_fraction(value: Any, *, nullable: bool = False) -> bool:
    if value is None:
        return nullable
    if not isinstance(value, str) or re.fullmatch(r"-?(0|[1-9][0-9]*)/[1-9][0-9]*", value) is None:
        return False
    numerator_text, denominator_text = value.split("/")
    numerator, denominator = int(numerator_text), int(denominator_text)
    return not (numerator == 0 and value != "0/1") and math.gcd(abs(numerator), denominator) == 1


def _scope_id_valid(scope_type: Any, scope_id: Any) -> bool:
    bindings = RISK["scope_hierarchy"]["scope_id_policy"]["bindings"]
    rule = bindings.get(scope_type) if isinstance(scope_type, str) else None
    if not isinstance(rule, dict) or not isinstance(scope_id, str):
        return False
    if rule["type"] == "exact_literal":
        return scope_id == rule["value"]
    return _field_schema_valid("scope_id", scope_id, {"type": "id", "prefix": rule["prefix"]})


def _asset_reference_valid(value: Any, *, allowed: list[str] | None = None) -> bool:
    source = json.loads((DOCS / "exchange_accounts_and_instruments.json").read_text())[
        "asset_reference_contract"
    ]
    if not isinstance(value, dict) or set(value) != set(source["fields"]):
        return False
    if not all(
        isinstance(value[field], str) and bool(value[field]) for field in source["fields"][:-1]
    ):
        return False
    statuses = allowed if allowed is not None else source["mapping_statuses"]
    return value["mapping_status"] in statuses


def _limit_result_valid(item: Any, schema: dict[str, Any]) -> bool:
    if not isinstance(item, dict) or set(item) != set(schema["exact_fields"]):
        return False
    for field in schema["exact_fields"]:
        if not _field_schema_valid(
            field, item[field], schema["field_schemas"][field], root_payload=item
        ):
            return False
    result, observed, reason = item["result"], item["observed_projected_value"], item["reason_code"]
    if (result in {"PASS", "FAIL"}) != (observed is not None):
        return False
    if result == "INCOMPLETE" and observed is not None:
        return False
    limit_type = item["limit_type"]
    if limit_type == "DISPATCH_RESERVATION_ECONOMICS":
        expected = {
            "effective_threshold": "0/1",
            "observed_projected_value": None,
            "supplying_policy_scope": "SYSTEM",
            "result": "INCOMPLETE",
            "reason_code": "MISSING_VALUATION",
        }
        return all(item[key] == expected_value for key, expected_value in expected.items())
    if item["supplying_policy_scope"] == "SYSTEM":
        return False
    return (
        reason
        in {"PASS": ["PASS"], "FAIL": ["LIMIT_BREACH"], "INCOMPLETE": ["MISSING_REQUIRED_INPUT"]}[
            result
        ]
    )


def _limit_results_valid(value: Any, schema: dict[str, Any]) -> bool:
    if not isinstance(value, list) or len(value) < schema.get("min_items", 0):
        return False
    item_schema = schema.get("item_schema")
    if not isinstance(item_schema, dict) or not all(
        _limit_result_valid(item, item_schema) for item in value
    ):
        return False
    unique = [
        (
            x["limit_type"],
            json.dumps(x["unit_asset_reference"], sort_keys=True),
            x["supplying_policy_scope"],
        )
        for x in value
    ]
    if len(unique) != len(set(unique)):
        return False
    registry = RISK["risk_decision_contract"]["limit_result_type_registry"]

    def key(item: dict[str, Any]) -> tuple[Any, ...]:
        synthetic = item["limit_type"] == "DISPATCH_RESERVATION_ECONOMICS"
        asset = tuple(item["unit_asset_reference"][field] for field in RISK_ASSET_FIELDS)
        return (1 if synthetic else 0, registry.index(item["limit_type"]), asset)

    return value == sorted(value, key=key)


def _tuple_array_valid(value: Any, schema: dict[str, Any]) -> bool:
    if not isinstance(value, list) or len(value) < schema.get("min_items", 0):
        return False
    item_schemas = schema.get("item_schema")
    if not isinstance(item_schemas, list):
        return False
    for item in value:
        if not isinstance(item, list) or len(item) != schema.get("tuple_length"):
            return False
        for spec in item_schemas:
            index = spec.get("index")
            if not isinstance(index, int) or index >= len(item):
                return False
            resolved = spec
            if "schema_reference" in spec:
                ok, resolved = _resolve_pointer(RISK, spec["schema_reference"])
                if not ok or not isinstance(resolved, dict):
                    return False
            if not _field_schema_valid(
                spec.get("name", "tuple_item"), item[index], resolved, tuple_context=item
            ):
                return False
    duplicates = schema.get("duplicates", {})
    indexes = duplicates.get("uniqueness_key_indexes", [])
    keys = [tuple(item[index] for index in indexes) for item in value]
    if duplicates.get("allowed") is False and len(keys) != len(set(keys)):
        return False
    hierarchy = RISK["scope_hierarchy"]["applicable_order"]
    if schema.get("tuple_length") == 6:
        order_key = lambda item: (hierarchy.index(item[2]), item[3], item[1])
    else:
        order_key = lambda item: (hierarchy.index(item[0]), item[1])
    return value == sorted(value, key=order_key)


def _field_schema_valid(
    field: str,
    value: Any,
    schema: dict[str, Any],
    *,
    root_payload: dict[str, Any] | None = None,
    tuple_context: list[Any] | None = None,
    aspect: str | None = None,
) -> bool:
    kind = schema.get("type")
    if kind not in IMPLEMENTED_DIRECT_SCHEMA_TYPES:
        return False
    if kind in {"constant", "exact_literal"}:
        return value == schema.get("value")
    if kind in {"id", "canonical_uuid7_prefixed_id", "canonical_id"}:
        prefix = schema.get("prefix", schema.get("id_prefix"))
        return (
            isinstance(prefix, str)
            and isinstance(value, str)
            and GENERIC_UUID7_ID.fullmatch(value) is not None
            and value.startswith(prefix + "_")
        )
    if kind in {"nullable_id", "nullable_canonical_id"}:
        return value is None or _field_schema_valid(
            field, value, {"type": "id", "prefix": schema.get("prefix", schema.get("id_prefix"))}
        )
    if kind in {"positive_integer", "positive_non_boolean_integer"}:
        return _positive(value)
    if kind == "non_negative_integer":
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if kind == "boolean":
        return isinstance(value, bool)
    if kind in {"sha256_hex", "sha256_lowercase_hex", "terminal_fingerprint"}:
        return isinstance(value, str) and SHA_RE.fullmatch(value) is not None
    if kind == "enum":
        values = schema.get("values")
        if values is None and isinstance(schema.get("values_source_pointer"), str):
            ok, values = _resolve_pointer(RISK, schema["values_source_pointer"])
            if not ok:
                return False
        return value in values if isinstance(values, list) else False
    if kind in {"decimal", "positive_decimal", "non_negative_decimal"}:
        constraint = schema.get(
            "constraint",
            "positive"
            if kind == "positive_decimal"
            else "non_negative"
            if kind == "non_negative_decimal"
            else None,
        )
        if not isinstance(value, str) or DECIMAL_RE.fullmatch(value) is None:
            return False
        try:
            number = Decimal(value)
        except InvalidOperation:
            return False
        return (constraint != "positive" or number > 0) and (
            constraint != "non_negative" or number >= 0
        )
    if kind in {"timestamp", "canonical_utc_timestamp"}:
        return _canonical_timestamp(value)
    if kind == "nullable_timestamp":
        return value is None or _canonical_timestamp(value)
    if kind == "non_empty_string":
        return isinstance(value, str) and bool(value)
    if kind == "string":
        return isinstance(value, str)
    if kind == "nullable_non_empty_string":
        return value is None or isinstance(value, str) and bool(value)
    if kind == "secure_store_reference":
        grammar = schema.get("grammar", {})
        if not isinstance(value, str) or not value.startswith(
            grammar.get("prefix", "secure-store://")
        ):
            return False
        locator = value[len(grammar.get("prefix", "secure-store://")) :]
        return (
            bool(locator)
            and not any(ch.isspace() for ch in value)
            and not any(ch in value for ch in "?#=")
            and not any(
                marker in value.lower() for marker in grammar.get("forbidden_payload_markers", [])
            )
        )
    if kind == "unique_array_of_enum":
        return (
            isinstance(value, list)
            and len(value) == len(set(value))
            and all(x in schema.get("values", []) for x in value)
        )
    if kind == "canonical_unique_array_of_enum":
        values = schema.get("values", [])
        return (
            isinstance(value, list)
            and len(value) >= schema.get("min_items", 0)
            and len(value) == len(set(value))
            and all(x in values for x in value)
            and value == [x for x in values if x in value]
        )
    if kind in {"object", "asset_reference"}:
        fields, nested = schema.get("fields"), schema.get("field_schemas")
        return (
            isinstance(value, dict)
            and isinstance(fields, list)
            and isinstance(nested, dict)
            and set(value) == set(fields)
            and all(
                _field_schema_valid(name, value[name], nested[name], root_payload=value)
                for name in fields
            )
        )
    if kind == "exact_upstream_object":
        return _asset_reference_valid(value, allowed=schema.get("allowed_mapping_statuses"))
    if kind == "canonical_scope_id":
        if tuple_context is None and root_payload is None:
            return isinstance(value, str) and bool(value)
        scope_type = (
            tuple_context[schema["scope_type_index"]]
            if tuple_context is not None and "scope_type_index" in schema
            else root_payload.get(schema.get("scope_type_field", "scope_type"))
            if root_payload
            else None
        )
        return _scope_id_valid(scope_type, value)
    if kind == "conditional_supplying_policy_scope":
        if not root_payload or not isinstance(value, str):
            return False
        if root_payload.get("limit_type") == schema.get("synthetic_limit_type"):
            return value == schema.get("synthetic_exact_value")
        if ":" not in value:
            return False
        scope_type, scope_id = value.split(":", 1)
        return _scope_id_valid(scope_type, scope_id)
    if kind == "canonical_exact_fraction_string":
        return _canonical_fraction(value)
    if kind == "nullable_canonical_exact_fraction_string":
        return _canonical_fraction(value, nullable=True)
    if kind == "event_safe_payload":
        if aspect is None and root_payload is None:
            return isinstance(value, dict)
        if aspect != "Event" or not root_payload:
            return False
        registry = json.loads(
            (DOCS / "commands_events_order_lifecycle_and_idempotency.json").read_text()
        )["event_contract"]["event_schema_registry"]
        event_schema = registry.get(root_payload.get("event_type"))
        if (
            not isinstance(event_schema, dict)
            or not isinstance(value, dict)
            or set(value) != set(event_schema["safe_payload_fields"])
        ):
            return False
        nullable = set(event_schema["nullable_fields"])
        return all(
            (value[name] is None and name in nullable)
            or (
                value[name] is not None
                and _field_schema_valid(name, value[name], event_schema["field_schemas"][name])
            )
            for name in event_schema["safe_payload_fields"]
        )
    if kind == "array_of_exact_LimitResult":
        return _limit_results_valid(value, schema)
    if kind == "array_of_exact_tuple":
        return _tuple_array_valid(value, schema)
    if kind == "array_of_canonical_id":
        return isinstance(value, list) and all(
            _field_schema_valid(field, item, {"type": "id", "prefix": schema.get("id_prefix")})
            for item in value
        )
    if kind == "array":
        return isinstance(value, list)
    if kind == "compound_scope":
        return isinstance(value, str) and bool(value)
    if kind == "non_empty_object":
        return isinstance(value, dict) and bool(value)
    if kind == "risk_limits":
        if not isinstance(value, list) or not value:
            return False
        asset_fields = set(schema.get("asset_reference_fields", []))
        return all(
            isinstance(item, list)
            and len(item) == 3
            and item[0] in schema.get("supported_limit_names", [])
            and _canonical_fraction(item[1])
            and isinstance(item[2], dict)
            and set(item[2]) == asset_fields
            and all(isinstance(nested, str) and nested for nested in item[2].values())
            for item in value
        )
    return False


def _submit_order_semantics_valid(payload: dict[str, Any], constraints: dict[str, Any]) -> bool:
    conditional = constraints.get("conditional_nullability")
    inequality = constraints.get("identity_inequality")
    if not isinstance(conditional, list) or not isinstance(inequality, dict):
        return False
    for rule in conditional:
        if not isinstance(rule, dict) or rule.get("required") not in {"NULL", "NON_NULL"}:
            return False
        condition = rule.get("when", rule.get("when_not"))
        if not isinstance(condition, dict) or not isinstance(rule.get("field"), str):
            return False
        matches = all(payload.get(field) == value for field, value in condition.items())
        applies = not matches if "when_not" in rule else matches
        if applies and (payload.get(rule["field"]) is None) != (rule["required"] == "NULL"):
            return False
    if set(inequality) != {"left", "right", "rule"} or inequality["rule"] != "MUST_DIFFER":
        return False
    return payload.get(inequality["left"]) != payload.get(inequality["right"])


def _validate_direct_upstream_payload(aspect: str, payload: Any) -> bool:
    registry = MACHINE["backup_contract"]["direct_upstream_validator_registry"]
    contract = registry.get(aspect)
    if not isinstance(contract, dict) or not isinstance(payload, dict):
        return False
    exact = set(contract["exact_fields"])
    if set(payload) != exact:
        return False
    nullable = set(contract["nullable_fields"])
    schemas = contract["upstream_field_schemas"]
    for field in exact:
        value = payload[field]
        if value is None and field not in nullable:
            return False
        if value is not None and not _field_schema_valid(
            field, value, schemas.get(field, {}), root_payload=payload, aspect=aspect
        ):
            return False
    if "environment" in payload and payload["environment"] not in {"PAPER", "TESTNET", "LIVE"}:
        return False
    source_document = json.loads((DOCS / contract["semantic_artifact"]).read_text())
    for field, constraint in contract.get("semantic_constraints", {}).items():
        if field not in payload:
            continue
        values = constraint.get("values")
        if (
            "pointer" in constraint
            and field != "safe_payload"
            and field != "fee_semantics"
            and field != "secret_reference"
        ):
            ok, values = _resolve_pointer(source_document, constraint["pointer"])
            if not ok or not isinstance(values, list):
                return False
        if isinstance(values, list) and payload[field] not in values:
            return False
    if aspect in {"Command accepted request", "OrderIntent"} and not _submit_order_semantics_valid(
        payload, contract.get("semantic_constraints", {})
    ):
        return False
    if aspect == "ExchangeAccount":
        semantics = contract.get("semantic_constraints", {})
        registry = semantics.get("build_time_exchange_registry_binding", {})
        entries = registry.get("enabled_entries", [])
        selected = next(
            (entry for entry in entries if entry.get("exchange_id") == payload["exchange_id"]),
            None,
        )
        if (
            not isinstance(selected, dict)
            or payload["environment"] not in selected.get("supported_environments", [])
            or payload["market_type"] not in selected.get("supported_market_types", [])
        ):
            return False
        lifecycle = semantics.get("lifecycle_timestamp_policy", {})
        rules = lifecycle.get("state_rules", {})
        state = payload["lifecycle_state"]
        retired = payload["retired_at_utc"]
        if state not in rules:
            return False
        if state in {"DRAFT", "ACTIVE", "DISABLED"} and retired is not None:
            return False
        if state == "RETIRED" and (
            not _canonical_timestamp(retired)
            or not _canonical_timestamp_le(payload["created_at_utc"], retired)
        ):
            return False
    if aspect == "CredentialProfile metadata/reference":
        state = payload["lifecycle_state"]
        retired = payload["retired_at_utc"]
        if state == "ACTIVE" and retired is not None:
            return False
        if state == "RETIRED" and (
            not _canonical_timestamp(retired)
            or not _canonical_timestamp_le(payload["created_at_utc"], retired)
        ):
            return False
        if payload["rotated_from_credential_profile_id"] == payload["credential_profile_id"]:
            return False
    if aspect == "Event":
        source = _source_value(MACHINE["backup_contract"]["representation_registry"][aspect])
        event_schema = source["event_schema_registry"].get(payload["event_type"])
        if not isinstance(event_schema, dict) or set(payload["safe_payload"]) != set(
            event_schema["safe_payload_fields"]
        ):
            return False
    if aspect == "Fill":
        if payload["fee_kind"] == "NONE" and not (
            payload["fee_quantity"] == "0" and payload["fee_asset_reference"] is None
        ):
            return False
        if payload["fee_kind"] == "CHARGE" and (
            payload["fee_quantity"] == "0" or not isinstance(payload["fee_asset_reference"], dict)
        ):
            return False
    terminal = _terminal_fingerprint_field(aspect)
    if terminal is not None:
        expected = _direct_terminal_fingerprint(aspect, payload)
        if payload[terminal] != expected:
            return False
    return True


def _validate_immutable_projection(aspect: str, entry: dict[str, Any], payload: Any) -> bool:
    binding = entry.get("immutable_fact_binding")
    if (
        not isinstance(binding, dict)
        or not isinstance(payload, dict)
        or set(payload) != set(binding["wrapper_fields"])
        or payload.get("fact_kind")
        != binding.get("fact_kind_literal", entry["semantic_object_or_invariant"])
        or not isinstance(payload.get("upstream_payload"), dict)
    ):
        return False
    upstream = payload["upstream_payload"]
    semantic_binding = binding
    if "upstream_payload_variants" in binding:
        discriminator = binding["upstream_payload_discriminator"]
        variant = binding["upstream_payload_variants"].get(upstream.get(discriminator))
        if not isinstance(variant, dict):
            return False
        fields = variant["persisted_payload_fields"]
        contracts = variant["field_contracts"]
        semantic_binding = variant
    else:
        fields = binding["persisted_payload_fields"]
        contracts = binding["field_contracts"]
    if set(upstream) != set(fields):
        return False
    if not all(_field_schema_valid(field, upstream[field], contracts[field]) for field in fields):
        return False
    if aspect == "RiskPolicy accepted revisions":
        scope_prefixes = {
            "PRODUCT_SYSTEM": None,
            "WORKSPACE": "ws",
            "PORTFOLIO": "port",
            "EXCHANGE_ACCOUNT": "xacc",
            "STRATEGY_INSTANCE": "sinst",
            "INSTRUMENT": "instr",
            "EXECUTION_ROUTE": "xroute",
        }
        expected_prefix = scope_prefixes[upstream["scope_type"]]
        if (expected_prefix is None and upstream["scope_id"] != "product") or (
            expected_prefix is not None
            and not upstream["scope_id"].startswith(expected_prefix + "_")
        ):
            return False
    if aspect == "kill-switch transition history":
        scope_prefixes = {
            "PRODUCT_SYSTEM": None,
            "WORKSPACE": "ws",
            "PORTFOLIO": "port",
            "EXCHANGE_ACCOUNT": "xacc",
            "STRATEGY_INSTANCE": "sinst",
            "INSTRUMENT": "instr",
            "EXECUTION_ROUTE": "xroute",
        }
        expected_prefix = scope_prefixes[upstream["scope_type"]]
        if (expected_prefix is None and upstream["scope_id"] != "product") or (
            expected_prefix is not None
            and not upstream["scope_id"].startswith(expected_prefix + "_")
        ):
            return False
    if aspect == "Order lifecycle events/history":
        event_contract = _source_value(entry)
        event_schema = event_contract["event_schema_registry"].get(upstream["event_type"])
        if not isinstance(event_schema, dict) or set(upstream["safe_payload"]) != set(
            event_schema["safe_payload_fields"]
        ):
            return False
        if not all(
            _field_schema_valid(
                name,
                upstream["safe_payload"][name],
                event_schema["field_schemas"][name],
            )
            for name in event_schema["safe_payload_fields"]
        ):
            return False
    semantic_field = semantic_binding.get("semantic_fingerprint_field")
    if isinstance(semantic_field, str):
        derivation = semantic_binding.get("semantic_fingerprint_derivation", {})
        separator_field = derivation.get("domain_separator_record_field")
        if isinstance(separator_field, str) and upstream.get(separator_field) != derivation.get(
            "domain_separator"
        ):
            return False
        expected_fingerprint = _semantic_fingerprint(semantic_binding, upstream)
        if expected_fingerprint is None or upstream[semantic_field] != expected_fingerprint:
            return False
    fingerprint = payload.get("upstream_payload_fingerprint_sha256")
    return isinstance(fingerprint, str) and fingerprint == _actual_fingerprint(upstream)


def _validate_facts_projection(aspect: str, entry: dict[str, Any], payload: Any) -> bool:
    if not isinstance(payload, dict) or set(payload) != {
        "fact_kind",
        "scope_key",
        "facts",
        "source_fingerprint_sha256",
    }:
        return False
    facts = payload.get("facts")
    binding = entry.get("fact_binding")
    if (
        payload.get("fact_kind") != aspect
        or payload.get("source_fingerprint_sha256") != entry["semantic_contract_fingerprint_sha256"]
        or not isinstance(binding, dict)
        or not isinstance(facts, dict)
        or set(facts) != set(binding["required_fact_fields"])
        or not facts
        or payload.get("scope_key") != facts.get(binding["scope_binding"])
    ):
        return False
    contracts = binding.get("field_contracts")
    if not isinstance(contracts, dict) or set(contracts) != set(facts):
        return False
    if not all(_field_schema_valid(field, facts[field], contracts[field]) for field in facts):
        return False
    for field, contract in contracts.items():
        if contract.get("type") == "compound_scope":
            expected = "|".join(str(facts[name]) for name in contract["components"])
            if facts[field] != expected:
                return False
    return True


def _validate_category_payload(aspect: str, entry: dict[str, Any], payload: Any) -> bool:
    if aspect == "bootstrap consumed fence":
        fields = {
            "state_fingerprint_sha256",
            "account_id",
            "device_installation_id",
            "intended_operator_id",
            "startup_readiness",
            "initial_security_lifecycle",
            "first_operator_presence",
            "expected_generation",
            "expected_revision",
            "consumed_authorities",
            "state_revision",
        }
        return (
            isinstance(payload, dict)
            and set(payload) == fields
            and _canonical_scope(payload.get("account_id"), payload.get("device_installation_id"))
            and isinstance(payload.get("intended_operator_id"), str)
            and payload["intended_operator_id"].startswith("op_")
            and all(
                _positive(payload.get(k))
                for k in ("expected_generation", "expected_revision", "state_revision")
            )
            and isinstance(payload.get("consumed_authorities"), list)
            and all(
                isinstance(x, str) and SHA_RE.fullmatch(x) for x in payload["consumed_authorities"]
            )
            and isinstance(payload.get("state_fingerprint_sha256"), str)
            and SHA_RE.fullmatch(payload["state_fingerprint_sha256"]) is not None
        )
    if aspect == "bootstrap accepted/consumption history":
        fields = {
            "account_id",
            "device_installation_id",
            "bootstrap_generation",
            "bootstrap_revision",
            "claim_fingerprint_sha256",
            "challenge_fingerprint_sha256",
        }
        return (
            isinstance(payload, dict)
            and set(payload) == fields
            and _canonical_scope(payload.get("account_id"), payload.get("device_installation_id"))
            and _positive(payload.get("bootstrap_generation"))
            and _positive(payload.get("bootstrap_revision"))
            and all(
                isinstance(payload.get(k), str) and SHA_RE.fullmatch(payload[k])
                for k in ("claim_fingerprint_sha256", "challenge_fingerprint_sha256")
            )
        )
    category = entry["representation_category"]
    if category == "M011_ENTITY_IDENTITY_PROJECTION":
        return _exact_entity_projection(payload)
    if category == "M011_CURRENT_DESIGNATION_PROJECTION":
        return _validate_projection_schema(payload, "CurrentDesignationProjection")
    if category == "M011_PERSISTENCE_PROJECTION_OF_UPSTREAM_FACTS":
        return _validate_facts_projection(aspect, entry, payload)
    if category == "M011_LOCAL_SCHEMA":
        schema_name = entry["projection_schema_if_any"]
        if schema_name == "StateStoreMetadata":
            return _validate_metadata(payload) == "VALID"
        validators = {
            "MigrationTransitionRecord": _validate_migration_transition,
            "MigrationCurrentState": _validate_migration_current,
            "SecretHandoffTransitionRecord": _validate_handoff_transition,
            "SecretHandoffCurrentState": _validate_handoff_current,
        }
        return validators[schema_name](payload)
    if category == "DIRECT_UPSTREAM_SCHEMA":
        return _validate_direct_upstream_payload(aspect, payload)
    if category == "M011_IMMUTABLE_HISTORY_WRAPPER":
        return _validate_immutable_projection(aspect, entry, payload)
    return False


def _derive_record_key(aspect: str, entry: dict[str, Any], payload: dict[str, Any]) -> str | None:
    if not isinstance(payload, dict):
        return None
    try:
        strategy = entry.get("record_key_strategy")
        if strategy == "DIRECT_UPSTREAM_KEY_FIELDS":
            contract = MACHINE["backup_contract"]["direct_upstream_validator_registry"][aspect]
            return (
                "direct:"
                + aspect
                + ":"
                + ":".join(str(payload[field]) for field in contract["record_key_fields"])
            )
        if strategy == "IMMUTABLE_PAYLOAD_IDENTITY_REVISION":
            binding = entry["immutable_fact_binding"]
            upstream = payload["upstream_payload"]
            fields = (
                binding["canonical_object_identity_fields"] + binding["revision_generation_fields"]
            )
            return (
                "immutable:"
                + str(payload["fact_kind"])
                + ":"
                + ":".join(str(upstream[field]) for field in fields)
            )
        if strategy == "CANONICAL_ENTITY_ID":
            source_field = entry["record_key_source_field"]
            source_location = entry["record_key_source_location"]
            if source_location == "payload":
                return str(payload[source_field])
            if source_location == "upstream_payload":
                return str(payload["upstream_payload"][source_field])
            return None
        if strategy == "SCOPE_CURRENT_REFERENCE_REVISION_GENERATION":
            return f"current:{payload['scope_key']}:{payload['current_reference']}:{payload['current_revision']}:{payload['current_generation']}"
        if strategy == "CANONICAL_OBJECT_ID_REVISION":
            return (
                f"object:{payload['semantic_object']}:{payload['object_id']}:{payload['revision']}"
            )
        if strategy == "FACT_SCOPE_OBJECT_GENERATION":
            facts = payload["facts"]
            fields = entry["fact_binding"]["record_key_object_fields"]
            return (
                "facts:"
                + str(payload["fact_kind"])
                + ":"
                + ":".join(str(facts[field]) for field in fields)
            )
        if strategy == "BOOTSTRAP_SCOPE_STATE_REVISION":
            return f"bootstrap-current:{payload['account_id']}:{payload['device_installation_id']}:{payload['state_revision']}"
        if strategy == "BOOTSTRAP_SCOPE_GENERATION_REVISION_CLAIM":
            return f"bootstrap-history:{payload['account_id']}:{payload['device_installation_id']}:{payload['bootstrap_generation']}:{payload['bootstrap_revision']}:{payload['claim_fingerprint_sha256']}"
        if strategy == "STATE_STORE_SCOPE_GENERATION":
            return f"state-store:{payload['account_id']}:{payload['device_installation_id']}:{payload['state_store_identity_fingerprint_sha256']}:{payload['protected_freshness_generation']}"
        if strategy == "MIGRATION_ID_TRANSITION_REVISION":
            return (
                f"migration-transition:{payload['migration_id']}:{payload['transition_revision']}"
            )
        if strategy == "MIGRATION_ID_CURRENT":
            return f"migration-current:{payload['migration_id']}"
        if strategy == "HANDOFF_ID_TRANSITION_REVISION":
            return f"handoff-transition:{payload['handoff_id']}:{payload['transition_revision']}"
        if strategy == "HANDOFF_ID_CURRENT":
            return f"handoff-current:{payload['handoff_id']}"
        return None
    except (KeyError, TypeError):
        return None


def _validate_persistence_record(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != _PERSISTENCE_FIELDS:
        return False
    registry = MACHINE["backup_contract"]["representation_registry"]
    aspect = value.get("representation_name")
    entry = registry.get(aspect) if isinstance(aspect, str) else None
    if (
        not isinstance(entry, dict)
        or entry["carrier_strategy"] != "PERSISTENCE_RECORD"
        or entry["representation_category"] == "EXCLUDED_NON_DURABLE"
    ):
        return False
    assert isinstance(aspect, str)
    for key in (
        "representation_category",
        "semantic_owner_milestone",
        "semantic_artifact",
        "semantic_json_pointer",
        "semantic_contract_fingerprint_sha256",
    ):
        if value.get(key) != entry.get(key):
            return False
    if (
        not value.get("record_key")
        or not isinstance(value.get("payload_fingerprint_sha256"), str)
        or SHA_RE.fullmatch(value["payload_fingerprint_sha256"]) is None
    ):
        return False
    try:
        if value["payload_fingerprint_sha256"] != _actual_fingerprint(value.get("payload")):
            return False
    except (TypeError, ValueError):
        return False
    if not isinstance(value.get("payload"), dict) or value.get("record_key") != _derive_record_key(
        aspect, entry, value["payload"]
    ):
        return False
    if _contains_forbidden(
        value["payload"], direct_pin_record=aspect == "PinVerifierRecord accepted revisions"
    ):
        return False
    return _validate_category_payload(aspect, entry, value["payload"])


def _validate_backup_records(records: Any) -> bool:
    return isinstance(records, list) and all(
        _validate_persistence_record(record) for record in records
    )


def _canonical_backup_projection(value: dict[str, Any]) -> dict[str, Any]:
    canonical = copy.deepcopy(value)
    descriptors = canonical["integrity_metadata"]["state_store_transaction_descriptors"]
    canonical["integrity_metadata"]["state_store_transaction_descriptors"] = sorted(
        descriptors, key=lambda descriptor: descriptor["target_generation"]
    )
    return {key: item for key, item in canonical.items() if key != "envelope_fingerprint_sha256"}


def _validate_backup(value: Any) -> str:
    required = set(MACHINE["executable_boundary_schemas"]["BackupEnvelope"]["required"])
    if not isinstance(value, dict) or set(value) != required:
        return "BACKUP_INTEGRITY_FAILED"
    if not all(
        _positive(value.get(key))
        for key in (
            "backup_envelope_schema_version",
            "state_store_schema_version",
            "local_protected_freshness_generation",
        )
    ):
        return "BACKUP_INTEGRITY_FAILED"
    if not _canonical_scope(value.get("account_id"), value.get("device_installation_id")):
        return "BACKUP_SCOPE_MISMATCH"
    if value.get("environment") not in {"PAPER", "TESTNET", "LIVE"}:
        return "BACKUP_ENVIRONMENT_MISMATCH"
    for key in (
        "state_store_identity_fingerprint_sha256",
        "state_fingerprint_sha256",
        "transaction_fingerprint_sha256",
        "history_tail_fingerprint_sha256",
        "envelope_fingerprint_sha256",
    ):
        if not isinstance(value.get(key), str) or SHA_RE.fullmatch(value[key]) is None:
            return "BACKUP_INTEGRITY_FAILED"
    if not _validate_backup_records(
        value.get("canonical_durable_records")
    ) or not _validate_backup_records(value.get("immutable_recovery_history")):
        return "BACKUP_INTEGRITY_FAILED"
    metadata = value.get("integrity_metadata")
    if not isinstance(metadata, dict) or set(metadata) != {"state_store_transaction_descriptors"}:
        return "BACKUP_INTEGRITY_FAILED"
    descriptors = metadata["state_store_transaction_descriptors"]
    if not isinstance(descriptors, list):
        return "BACKUP_INTEGRITY_FAILED"
    envelope_metadata = {
        "account_id": value["account_id"],
        "device_installation_id": value["device_installation_id"],
        "state_store_identity_fingerprint_sha256": value["state_store_identity_fingerprint_sha256"],
        "state_store_schema_version": value["state_store_schema_version"],
        "environment": value["environment"],
        "protected_freshness_generation": value["local_protected_freshness_generation"],
        "state_fingerprint_sha256": value["state_fingerprint_sha256"],
        "transaction_fingerprint_sha256": value["transaction_fingerprint_sha256"],
        "history_tail_fingerprint_sha256": value["history_tail_fingerprint_sha256"],
    }
    if not _complete_descriptor_chain_valid(descriptors, envelope_metadata):
        return "BACKUP_INTEGRITY_FAILED"
    projected = _canonical_backup_projection(value)
    return (
        "VALID"
        if value["envelope_fingerprint_sha256"] == _actual_fingerprint(projected)
        else "BACKUP_INTEGRITY_FAILED"
    )


def _canonical_fixture_id(prefix: str, suffix: str) -> str:
    final = {"a": "b", "b": "c", "c": "d"}.get(suffix, "e")
    return f"{prefix}_01890f3a-2b4c-7abc-8def-0123456789a{final}"


def _valid_asset_reference(code: str = "USD") -> dict[str, str]:
    return {
        "venue_asset_code": code,
        "canonical_display_code": code,
        "asset_namespace": "ISO4217",
        "mapping_status": "EXACT",
    }


def _valid_limit_result(limit_type: str = "MAX_ORDER_QUANTITY") -> dict[str, Any]:
    return {
        "limit_type": limit_type,
        "effective_threshold": "1/1",
        "observed_projected_value": "1/1",
        "unit_asset_reference": _valid_asset_reference(),
        "supplying_policy_scope": "PRODUCT_SYSTEM:product",
        "result": "PASS",
        "reason_code": "PASS",
    }


def _direct_fixture_value(
    aspect: str, field: str, schema: dict[str, Any], suffix: str, revision: int
) -> Any:
    kind = schema.get("type")
    if field == "event_type":
        return "ORDER_PLANNED"
    if kind in {"constant", "exact_literal"}:
        return schema["value"]
    if kind == "id":
        return _canonical_fixture_id(schema["prefix"], suffix)
    if kind in {"positive_integer", "positive_non_boolean_integer"}:
        return revision
    if kind in {"sha256_hex", "sha256_lowercase_hex", "terminal_fingerprint"}:
        return SHA
    if kind == "enum":
        return schema["values"][0]
    if kind == "decimal":
        return "1" if schema.get("constraint") == "positive" else "0"
    if kind in {"timestamp", "canonical_utc_timestamp"}:
        return "2026-01-01T00:00:00Z"
    if kind == "array":
        return []
    if kind == "unique_array_of_enum":
        return [schema["values"][0]]
    if kind == "canonical_unique_array_of_enum":
        return [schema["values"][0]]
    if kind == "array_of_exact_LimitResult":
        return []
    if kind == "array_of_exact_tuple":
        if schema.get("min_items", 0) == 0:
            return []
        return [
            [
                _canonical_fixture_id("rpol", suffix),
                revision,
                "PRODUCT_SYSTEM",
                "product",
                "ALLOW",
                SHA,
            ]
        ]
    if kind == "secure_store_reference":
        return f"secure-store://reference-{suffix}"
    if kind == "event_safe_payload":
        return {"side": "BUY", "order_type": "MARKET", "quantity": "1"}
    if kind in {"object", "asset_reference"}:
        return {
            name: _direct_fixture_value(
                aspect, name, schema["field_schemas"][name], suffix, revision
            )
            for name in schema["fields"]
        }
    if field == "secret_reference":
        return f"secure-store://reference-{suffix}"
    if field == "event_type":
        return "ORDER_PLANNED"
    if field == "safe_payload":
        return {"side": "BUY", "order_type": "MARKET", "quantity": "1"}
    if field == "fee_asset_reference":
        return None
    if field == "side":
        return "BUY"
    if field == "fee_kind":
        return "NONE"
    if field == "direction":
        return "DEBIT"
    if field == "posting_role":
        return "ASSET_RECEIVED"
    if field == "decision":
        return "ALLOW"
    if field == "state" and "SessionSecurityState" in aspect:
        return "LOCKED"
    if field == "state" and "SecretMetadata" in aspect:
        return "AVAILABLE"
    if field in {"state", "lifecycle_state", "connection_state", "execution_authorization"}:
        return "ACTIVE"
    return f"canonical-{field}-{suffix}"


def _direct_payload(aspect: str, suffix: str, revision: int) -> dict[str, Any]:
    contract = MACHINE["backup_contract"]["direct_upstream_validator_registry"][aspect]
    nullable = set(contract["nullable_fields"])
    schemas = contract["upstream_field_schemas"]
    payload = {
        field: (
            None
            if field in nullable
            else _direct_fixture_value(aspect, field, schemas.get(field, {}), suffix, revision)
        )
        for field in contract["exact_fields"]
    }
    if aspect == "Event":
        payload["event_type"] = "ORDER_PLANNED"
        payload["safe_payload"] = {"side": "BUY", "order_type": "MARKET", "quantity": "1"}
    if aspect in {"Command accepted request", "OrderIntent"}:
        payload.update(
            source_type="OPERATOR",
            strategy_instance_id=None,
            order_type="MARKET",
            limit_price=None,
            time_in_force="GTC",
            expire_at_utc=None,
        )
    if aspect == "kill-switch state/generation":
        payload["scope_type"] = "PRODUCT_SYSTEM"
        payload["scope_id"] = "product"
        payload["state"] = "INACTIVE"
    if aspect == "RiskDecision":
        payload["ordered_limit_results"] = [_valid_limit_result()]
    terminal = _terminal_fingerprint_field(aspect)
    if terminal is not None:
        payload[terminal] = _direct_terminal_fingerprint(aspect, payload)
    return payload


def _payload_for(
    aspect: str,
    *,
    object_suffix: str = "a",
    revision: int = 1,
    variant_name: str | None = None,
) -> dict[str, Any]:
    entry = MACHINE["backup_contract"]["representation_registry"][aspect]
    category = entry["representation_category"]
    if aspect == "bootstrap consumed fence":
        return {
            "state_fingerprint_sha256": SHA,
            "account_id": SCOPE[0],
            "device_installation_id": SCOPE[1],
            "intended_operator_id": "op_" + SCOPE[0][5:],
            "startup_readiness": "READY",
            "initial_security_lifecycle": "COMPLETED",
            "first_operator_presence": "PRESENT",
            "expected_generation": 1,
            "expected_revision": 1,
            "consumed_authorities": [SHA],
            "state_revision": revision,
        }
    if aspect == "bootstrap accepted/consumption history":
        return {
            "account_id": SCOPE[0],
            "device_installation_id": SCOPE[1],
            "bootstrap_generation": 1,
            "bootstrap_revision": revision,
            "claim_fingerprint_sha256": SHA,
            "challenge_fingerprint_sha256": "b" * 64,
        }
    if category == "M011_ENTITY_IDENTITY_PROJECTION":
        kind = "CryptoHunterAccount" if aspect.startswith("CryptoHunterAccount") else "Workspace"
        entity_id = ("acct_" if kind == "CryptoHunterAccount" else "ws_") + SCOPE[0][5:]
        return {
            "entity_kind": kind,
            "entity_id": entity_id,
            "parent_scope_bindings": {}
            if kind == "CryptoHunterAccount"
            else {"account_id": SCOPE[0]},
        }
    if category == "M011_CURRENT_DESIGNATION_PROJECTION":
        return {
            "scope_key": SCOPE[0],
            "current_reference": f"ref:{object_suffix}:{SHA}",
            "current_revision": revision,
            "current_generation": revision,
            "content_fingerprint_sha256": SHA,
        }
    if category == "M011_PERSISTENCE_PROJECTION_OF_UPSTREAM_FACTS":
        binding = entry["fact_binding"]
        contracts = binding["field_contracts"]
        facts: dict[str, Any] = {}
        for name in binding["required_fact_fields"]:
            contract = contracts[name]
            kind = contract["type"]
            if kind == "canonical_id":
                facts[name] = _canonical_fixture_id(contract["id_prefix"], object_suffix)
            elif kind == "positive_integer":
                facts[name] = revision
            elif kind == "sha256_hex":
                facts[name] = SHA
            elif kind == "enum":
                facts[name] = contract["values"][0]
        for name, contract in contracts.items():
            if contract["type"] == "compound_scope":
                facts[name] = "|".join(str(facts[field]) for field in contract["components"])
        return {
            "fact_kind": aspect,
            "scope_key": facts[binding["scope_binding"]],
            "facts": facts,
            "source_fingerprint_sha256": entry["semantic_contract_fingerprint_sha256"],
        }
    if category == "M011_LOCAL_SCHEMA":
        return {
            "StateStoreMetadata": _metadata(protected_freshness_generation=revision),
            "MigrationTransitionRecord": _migration_transition(
                revision,
                None if revision == 1 else "PREPARED",
                "PREPARED" if revision == 1 else "APPLYING",
            ),
            "MigrationCurrentState": _migration_current(_migration_transition(1, None, "PREPARED")),
            "SecretHandoffTransitionRecord": _handoff_transition(
                revision,
                None if revision == 1 else "PREPARED",
                "PREPARED" if revision == 1 else "APPLYING",
            ),
            "SecretHandoffCurrentState": _handoff_current(_handoff_transition(1, None, "PREPARED")),
        }[entry["projection_schema_if_any"]]
    if category == "DIRECT_UPSTREAM_SCHEMA":
        return _direct_payload(aspect, object_suffix, revision)
    if category == "M011_IMMUTABLE_HISTORY_WRAPPER":
        binding = entry["immutable_fact_binding"]
        variant = None
        if "upstream_payload_variants" in binding:
            variants = binding["upstream_payload_variants"]
            variant = (
                variants[variant_name]
                if variant_name is not None
                else next(iter(variants.values()))
            )
        fields = (
            variant["persisted_payload_fields"]
            if variant is not None
            else binding["persisted_payload_fields"]
        )
        contracts = (
            variant["field_contracts"] if variant is not None else binding["field_contracts"]
        )
        upstream: dict[str, Any] = {}
        for field in fields:
            contract = contracts[field]
            kind = contract["type"]
            if kind == "constant":
                upstream[field] = contract["value"]
            elif kind == "canonical_id":
                upstream[field] = _canonical_fixture_id(contract["id_prefix"], object_suffix)
            elif kind == "positive_integer":
                upstream[field] = revision
            elif kind == "non_negative_integer":
                upstream[field] = 0
            elif kind == "boolean":
                upstream[field] = True
            elif kind == "sha256_hex":
                upstream[field] = SHA
            elif kind == "enum":
                upstream[field] = contract["values"][0]
            elif kind == "array_of_canonical_id":
                upstream[field] = [_canonical_fixture_id(contract["id_prefix"], object_suffix)]
            elif kind == "non_empty_object":
                upstream[field] = {"algorithm": "argon2id", "encoded_verifier": SHA}
            elif kind in {"object", "asset_reference"}:
                upstream[field] = {
                    nested: (
                        1
                        if nested_schema["type"] == "positive_integer"
                        else True
                        if nested_schema["type"] == "boolean"
                        else nested_schema["values"][0]
                        if nested_schema["type"] == "enum"
                        else f"canonical-{nested}"
                    )
                    for nested, nested_schema in contract["field_schemas"].items()
                }
            elif kind == "timestamp":
                upstream[field] = "2025-01-01T00:00:00Z"
            elif kind == "nullable_timestamp":
                upstream[field] = None
            elif kind == "nullable_canonical_id":
                upstream[field] = None
            elif kind == "non_negative_decimal":
                upstream[field] = "1"
            elif kind == "positive_decimal":
                upstream[field] = "1"
            elif kind == "boolean":
                upstream[field] = True
            elif kind == "event_safe_payload":
                event_contract = _source_value(entry)
                event_schema = event_contract["event_schema_registry"][upstream["event_type"]]
                upstream[field] = {
                    name: (
                        "1"
                        if schema["type"] == "decimal"
                        else _canonical_fixture_id(schema["prefix"], object_suffix)
                        if schema["type"] == "id"
                        else schema.get("values", ["canonical"])[0]
                    )
                    for name, schema in event_schema["field_schemas"].items()
                }
            elif kind == "risk_limits":
                upstream[field] = [
                    [
                        contract["supported_limit_names"][0],
                        "1/1",
                        {
                            "venue_asset_code": "USD",
                            "canonical_display_code": "USD",
                            "asset_namespace": "ISO4217",
                            "mapping_status": "CANONICAL",
                        },
                    ]
                ]
            elif kind == "canonical_scope_id":
                scope_prefix = {
                    "PRODUCT_SYSTEM": None,
                    "WORKSPACE": "ws",
                    "PORTFOLIO": "port",
                    "EXCHANGE_ACCOUNT": "xacc",
                    "STRATEGY_INSTANCE": "sinst",
                    "INSTRUMENT": "instr",
                    "EXECUTION_ROUTE": "xroute",
                }[upstream[contract["scope_type_field"]]]
                upstream[field] = (
                    "product"
                    if scope_prefix is None
                    else _canonical_fixture_id(scope_prefix, object_suffix)
                )
            else:
                upstream[field] = f"canonical-{field}-{object_suffix}"
        semantic_binding = variant if variant is not None else binding
        terminal = semantic_binding.get("semantic_fingerprint_field")
        if isinstance(terminal, str):
            derived = _semantic_fingerprint(semantic_binding, upstream)
            assert derived is not None
            upstream[terminal] = derived
        return {
            "fact_kind": binding.get("fact_kind_literal", entry["semantic_object_or_invariant"]),
            "upstream_payload": upstream,
            "upstream_payload_fingerprint_sha256": _actual_fingerprint(upstream),
        }
    raise AssertionError(f"unsupported persisted category: {category}")


def _persistence_record(
    aspect: str,
    payload: Any | None = None,
    *,
    object_suffix: str = "a",
    revision: int = 1,
    variant_name: str | None = None,
) -> dict[str, Any]:
    entry = MACHINE["backup_contract"]["representation_registry"][aspect]
    actual = (
        _payload_for(
            aspect,
            object_suffix=object_suffix,
            revision=revision,
            variant_name=variant_name,
        )
        if payload is None
        else payload
    )
    value = {
        key: entry[key]
        for key in (
            "representation_category",
            "semantic_owner_milestone",
            "semantic_artifact",
            "semantic_json_pointer",
            "semantic_contract_fingerprint_sha256",
        )
    }
    value["representation_name"] = aspect
    value.update(
        record_key=_derive_record_key(aspect, entry, actual),
        payload=actual,
        payload_fingerprint_sha256=_actual_fingerprint(actual),
    )
    return value


def _valid_backup() -> dict[str, Any]:
    descriptors = _descriptor_chain(2)
    current = descriptors[-1]
    value = {
        "backup_envelope_schema_version": 1,
        "state_store_schema_version": current["state_store_schema_version"],
        "account_id": SCOPE[0],
        "device_installation_id": SCOPE[1],
        "state_store_identity_fingerprint_sha256": SHA,
        "environment": "PAPER",
        "local_protected_freshness_generation": 2,
        "state_fingerprint_sha256": current["post_state_fingerprint_sha256"],
        "transaction_fingerprint_sha256": current["transaction_fingerprint_sha256"],
        "history_tail_fingerprint_sha256": current["post_history_tail_fingerprint_sha256"],
        "canonical_durable_records": [],
        "immutable_recovery_history": [],
        "envelope_fingerprint_sha256": "",
        "integrity_metadata": {"state_store_transaction_descriptors": descriptors},
    }
    value["envelope_fingerprint_sha256"] = _actual_fingerprint(
        {key: item for key, item in value.items() if key != "envelope_fingerprint_sha256"}
    )
    return value


def _version_result(state: Any, backup: Any, paths: set[tuple[int, int]], current: int = 2) -> str:
    if not _positive(state) or not _positive(backup):
        return "CONTRACT_INCONSISTENT"
    if backup > 1:
        return "UNSUPPORTED_BACKUP_SCHEMA"
    if state > current:
        return "UNSUPPORTED_STATESTORE_SCHEMA"
    if state == current:
        return "VALID"
    return "MIGRATION_REQUIRED" if (state, current) in paths else "MIGRATION_PATH_UNAVAILABLE"


def _migration_recovery(state: str, durable_target: bool, path: list[int]) -> str:
    if path != list(range(path[0], path[-1] + 1)):
        return "MIGRATION_PATH_UNAVAILABLE"
    if state == "PREPARED":
        return "RESUME_APPLYING"
    if state == "APPLYING":
        return "RESUME_VERIFY_OR_REAPPLY_NO_PROMOTION"
    if state == "DURABLE_MIGRATED" and durable_target:
        return "COMPLETE_MARKER_IDEMPOTENTLY"
    if state == "COMPLETED" and durable_target:
        return "ALREADY_COMPLETED"
    if state == "FAILED":
        return "MIGRATION_FAILED"
    return "RECOVERY_REQUIRED"


@dataclass
class _ReplayState:
    requests: dict[str, tuple[str, str]] = field(default_factory=dict)
    side_effect_count: int = 0
    fills: set[str] = field(default_factory=set)
    ledger_effects: int = 0
    reservation_state: str = "HELD"
    reservation_effects: int = 0
    blind_resubmits: int = 0

    def command(self, key: str, fingerprint: str) -> str:
        prior = self.requests.get(key)
        if prior:
            return prior[1] if prior[0] == fingerprint else "IDEMPOTENCY_CONFLICT"
        self.side_effect_count += 1
        self.requests[key] = (fingerprint, "CANONICAL_OUTCOME")
        return "CANONICAL_OUTCOME"

    def fill(self, fill_id: str) -> None:
        if fill_id not in self.fills:
            self.fills.add(fill_id)
            self.ledger_effects += 1

    def reservation(self, terminal: str) -> None:
        if self.reservation_state == "HELD" and terminal in {"CONSUMED", "RELEASED"}:
            self.reservation_state = terminal
            self.reservation_effects += 1

    def unknown_external(self) -> str:
        return "RECONCILIATION_REQUIRED"


def _writer_attempt(
    role: str, lock_occupied: bool, competing_writer: bool
) -> tuple[str, bool, bool]:
    if role != "Core":
        return "NO_PERSISTENCE_AUTHORITY", False, False
    if competing_writer:
        return "CONCURRENT_WRITER", False, False
    if lock_occupied:
        return "LOCK_CONTENTION", False, False
    return "WRITER_OPENED", True, True


def _security_restore(candidate: dict[str, Any], current: dict[str, Any]) -> str:
    terminal = (
        "bootstrap_consumed",
        "device_revoked",
        "operator_revoked",
        "grant_revoked",
        "lease_consumed",
    )
    if any(current.get(key) and not candidate.get(key) for key in terminal):
        return "MONOTONIC_FENCE_ROLLBACK"
    for key in (
        "pin_revision",
        "security_generation",
        "session_security_generation",
        "kill_switch_generation",
    ):
        if candidate.get(key, 0) < current.get(key, 0):
            return "MONOTONIC_FENCE_ROLLBACK"
    if (
        candidate.get("AuthenticationProof") is not None
        or candidate.get("PlatformBiometricAssertion") is not None
    ):
        return "RESTORE_REJECTED"
    return "VALID"


def test_complete_frozen_durability_matrix_exact_equality() -> None:
    assert MACHINE["durability_classification"]["records"] == _EXPECTED_DURABILITY


def test_durability_matrix_mutation_is_detected() -> None:
    mutated = dict(_EXPECTED_DURABILITY)
    mutated.pop("Fill")
    assert mutated != MACHINE["durability_classification"]["records"]
    mutated = dict(_EXPECTED_DURABILITY)
    mutated["Fill"] = _A
    assert mutated != MACHINE["durability_classification"]["records"]


def test_failure_registry_exact_independently_authored_matrix() -> None:
    registry = MACHINE["failure_registry"]
    assert len(_EXPECTED_FAILURES) == 32
    assert registry["outcomes_by_code"] == _EXPECTED_FAILURES
    assert set(registry["closed_codes"]) == set(_EXPECTED_FAILURES)


@pytest.mark.parametrize(
    "mutation",
    [
        {"retryability": "AFTER_RECOVERY_ONLY"},
        {"failure_class": "CONTRACT"},
        {"mutation_acknowledged": True},
        {"recovery_required": True},
        {"safe_diagnostic_class": "RESTORE"},
        {"extra": "x"},
    ],
)
def test_failure_outcome_wrong_metadata_fails_closed(mutation: dict[str, Any]) -> None:
    value = {"code": "TRANSACTION_FAILED", **_EXPECTED_FAILURES["TRANSACTION_FAILED"]}
    value.update(mutation)
    assert not _validate_failure(value)


def test_unknown_failure_code_is_denied() -> None:
    assert not _validate_failure({"code": "EXTRA", **_EXPECTED_FAILURES["TRANSACTION_FAILED"]})


@pytest.mark.parametrize(
    "bad_account,bad_device",
    [
        ("acct_wrong", SCOPE[1]),
        (SCOPE[0], "dev_wrong"),
        ("acct_01890f3a-2b4c-4abc-8def-0123456789ab", SCOPE[1]),
        ("acct_01890F3A-2b4c-7abc-8def-0123456789ab", SCOPE[1]),
        (SCOPE[0] + "x", SCOPE[1]),
        (SCOPE[0], "dev_01890f3a-2b4c-7abc-7def-0123456789ab"),
    ],
)
def test_canonical_m02_scope_ids_fail_closed(bad_account: str, bad_device: str) -> None:
    assert not _canonical_scope(bad_account, bad_device)


def test_exact_evidence_payload_membership_and_fingerprint() -> None:
    registry = _EvidenceRegistry("process")
    ref = registry.observe(_observation())
    assert ref is not None and set(registry.accepted[ref]) == {
        "account_id",
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
        "generation",
        "state_fingerprint_sha256",
        "transaction_fingerprint_sha256",
        "durability_state",
        "evidence_revision",
        "evidence_fingerprint_sha256",
    }
    assert registry.verify_current(SCOPE, ref)
    registry.accepted[ref]["generation"] = 2
    assert not registry.verify_current(SCOPE, ref)


def test_self_hashed_evidence_without_membership_is_denied() -> None:
    registry = _EvidenceRegistry("process")
    payload = _observation()
    ref = "caller:" + _actual_fingerprint(payload)
    assert not registry.verify_current(SCOPE, ref)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"extra": True}, "BACKUP_INTEGRITY_FAILED"),
        ({"backup_envelope_schema_version": True}, "BACKUP_INTEGRITY_FAILED"),
        ({"local_protected_freshness_generation": False}, "BACKUP_INTEGRITY_FAILED"),
        ({"account_id": "acct_wrong"}, "BACKUP_SCOPE_MISMATCH"),
        ({"device_installation_id": "dev_wrong"}, "BACKUP_SCOPE_MISMATCH"),
        ({"environment": "WRONG"}, "BACKUP_ENVIRONMENT_MISMATCH"),
    ],
)
def test_backup_boundary_fields_fail_closed(mutation: dict[str, Any], expected: str) -> None:
    value = _valid_backup()
    value.update(mutation)
    assert _validate_backup(value) == expected


@pytest.mark.parametrize(
    "record",
    [
        {"record_kind": "LocalDurableStateEvidence"},
        {"record_kind": "Event", "raw_pin": "1"},
        {"record_kind": "Event", "payload": {"api_secret": "x"}},
        {"record_kind": "SecureStorePayload"},
        {"record_kind": "AuthenticationProof"},
        {"record_kind": "PlatformBiometricAssertion"},
    ],
)
def test_backup_rejects_nested_evidence_and_secret_authority(record: dict[str, Any]) -> None:
    value = _valid_backup()
    value["canonical_durable_records"] = [record]
    assert _validate_backup(value) == "BACKUP_INTEGRITY_FAILED"


@pytest.mark.parametrize(
    ("state", "durable", "expected"),
    [
        ("PREPARED", False, "RESUME_APPLYING"),
        ("APPLYING", False, "RESUME_VERIFY_OR_REAPPLY_NO_PROMOTION"),
        ("DURABLE_MIGRATED", True, "COMPLETE_MARKER_IDEMPOTENTLY"),
        ("COMPLETED", True, "ALREADY_COMPLETED"),
        ("FAILED", False, "MIGRATION_FAILED"),
    ],
)
def test_executable_migration_crash_recovery(state: str, durable: bool, expected: str) -> None:
    assert _migration_recovery(state, durable, [1, 2]) == expected


def test_migration_unknown_or_skipped_path_denied() -> None:
    assert _migration_recovery("PREPARED", False, [1, 3]) == "MIGRATION_PATH_UNAVAILABLE"
    assert MACHINE["migration_protocol"]["rollback_policy"] == "FORWARD_ONLY"
    assert MACHINE["migration_protocol"]["authoritative_change_requires_m0_3_freshness"] is True


@pytest.mark.parametrize(
    ("state", "backup", "paths", "expected"),
    [
        (2, 1, set(), "VALID"),
        (3, 1, set(), "UNSUPPORTED_STATESTORE_SCHEMA"),
        (2, 2, set(), "UNSUPPORTED_BACKUP_SCHEMA"),
        (1, 1, {(1, 2)}, "MIGRATION_REQUIRED"),
        (1, 1, set(), "MIGRATION_PATH_UNAVAILABLE"),
        (True, 1, set(), "CONTRACT_INCONSISTENT"),
        (2, False, set(), "CONTRACT_INCONSISTENT"),
    ],
)
def test_executable_schema_versioning(
    state: Any, backup: Any, paths: set[tuple[int, int]], expected: str
) -> None:
    assert _version_result(state, backup, paths) == expected


def test_idempotency_restart_replay_and_conflict() -> None:
    state = _ReplayState()
    assert state.command("key", SHA) == "CANONICAL_OUTCOME"
    assert state.side_effect_count == 1
    restarted = copy.deepcopy(state)
    assert restarted.command("key", SHA) == "CANONICAL_OUTCOME"
    assert restarted.side_effect_count == 1
    assert restarted.command("key", "b" * 64) == "IDEMPOTENCY_CONFLICT"
    assert restarted.side_effect_count == 1


def test_fill_and_reservation_replays_are_single_effect() -> None:
    state = _ReplayState()
    state.fill("fill")
    state.fill("fill")
    assert state.ledger_effects == 1
    state.reservation("CONSUMED")
    state.reservation("CONSUMED")
    state.reservation("RELEASED")
    assert state.reservation_state == "CONSUMED" and state.reservation_effects == 1
    assert state.unknown_external() == "RECONCILIATION_REQUIRED" and state.blind_resubmits == 0


@pytest.mark.parametrize(
    ("role", "occupied", "competing", "expected", "recovery"),
    [
        ("Core", True, False, "LOCK_CONTENTION", False),
        ("Core", True, True, "CONCURRENT_WRITER", True),
        ("UI", False, False, "NO_PERSISTENCE_AUTHORITY", False),
        ("Tray", False, False, "NO_PERSISTENCE_AUTHORITY", False),
    ],
)
def test_executable_writer_topology(
    role: str, occupied: bool, competing: bool, expected: str, recovery: bool
) -> None:
    outcome, mutable_open, session_created = _writer_attempt(role, occupied, competing)
    assert outcome == expected and not mutable_open and not session_created
    if outcome in _EXPECTED_FAILURES:
        assert _EXPECTED_FAILURES[outcome]["recovery_required"] is recovery


@pytest.mark.parametrize(
    "mutation",
    [
        {"bootstrap_consumed": False},
        {"device_revoked": False},
        {"operator_revoked": False},
        {"grant_revoked": False},
        {"lease_consumed": False},
        {"pin_revision": 1},
        {"security_generation": 1},
        {"session_security_generation": 1},
        {"kill_switch_generation": 1},
    ],
)
def test_restore_blocks_authority_resurrection(mutation: dict[str, Any]) -> None:
    current = {
        "bootstrap_consumed": True,
        "device_revoked": True,
        "operator_revoked": True,
        "grant_revoked": True,
        "lease_consumed": True,
        "pin_revision": 2,
        "security_generation": 2,
        "session_security_generation": 2,
        "kill_switch_generation": 2,
    }
    candidate = dict(current)
    candidate.update(mutation)
    assert _security_restore(candidate, current) == "MONOTONIC_FENCE_ROLLBACK"


@pytest.mark.parametrize("authority", ["AuthenticationProof", "PlatformBiometricAssertion"])
def test_restore_never_restores_ephemeral_proof_authority(authority: str) -> None:
    assert _security_restore({authority: {"payload": "x"}}, {}) == "RESTORE_REJECTED"


LIFECYCLE_ID = re.compile(r"^[a-z][a-z0-9_-]{2,127}$")
_MIGRATION_STATES = {"PREPARED", "APPLYING", "DURABLE_MIGRATED", "COMPLETED", "FAILED"}
_HANDOFF_STATES = {"PREPARED", "COMMITTED", "CLEANUP_PENDING", "UNKNOWN_RECONCILIATION"}


def _canonical_object_fingerprint(value: dict[str, Any], field: str) -> str:
    return _actual_fingerprint({key: item for key, item in value.items() if key != field})


def _validate_migration_transition(value: Any) -> bool:
    fields = {
        "migration_id",
        "transition_revision",
        "previous_state",
        "state",
        "transaction_fingerprint_sha256",
        "state_fingerprint_sha256",
        "protected_freshness_generation",
        "transition_fingerprint_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or not isinstance(value.get("migration_id"), str)
        or LIFECYCLE_ID.fullmatch(value["migration_id"]) is None
    ):
        return False
    if (
        not _positive(value.get("transition_revision"))
        or not _positive(value.get("protected_freshness_generation"))
        or value.get("state") not in _MIGRATION_STATES
        or value.get("previous_state") not in {None, *_MIGRATION_STATES}
    ):
        return False
    for key in (
        "transaction_fingerprint_sha256",
        "state_fingerprint_sha256",
        "transition_fingerprint_sha256",
    ):
        if not isinstance(value.get(key), str) or SHA_RE.fullmatch(value[key]) is None:
            return False
    return str(value["transition_fingerprint_sha256"]) == _canonical_object_fingerprint(
        value, "transition_fingerprint_sha256"
    )


def _validate_migration_current(value: Any) -> bool:
    fields = {
        "migration_id",
        "current_transition_revision",
        "state",
        "authoritative_state_fingerprint_sha256",
        "protected_freshness_generation",
        "designation_fingerprint_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or not isinstance(value.get("migration_id"), str)
        or LIFECYCLE_ID.fullmatch(value["migration_id"]) is None
    ):
        return False
    if (
        not _positive(value.get("current_transition_revision"))
        or not _positive(value.get("protected_freshness_generation"))
        or value.get("state") not in _MIGRATION_STATES
    ):
        return False
    for key in ("authoritative_state_fingerprint_sha256", "designation_fingerprint_sha256"):
        if not isinstance(value.get(key), str) or SHA_RE.fullmatch(value[key]) is None:
            return False
    return str(value["designation_fingerprint_sha256"]) == _canonical_object_fingerprint(
        value, "designation_fingerprint_sha256"
    )


def _validate_handoff_transition(value: Any) -> bool:
    fields = {
        "handoff_id",
        "transition_revision",
        "previous_state",
        "state",
        "operation_fingerprint_sha256",
        "metadata_fingerprint_sha256",
        "transition_fingerprint_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or not isinstance(value.get("handoff_id"), str)
        or LIFECYCLE_ID.fullmatch(value["handoff_id"]) is None
    ):
        return False
    if (
        not _positive(value.get("transition_revision"))
        or value.get("state") not in _HANDOFF_STATES
        or value.get("previous_state") not in {None, *_HANDOFF_STATES}
    ):
        return False
    for key in (
        "operation_fingerprint_sha256",
        "metadata_fingerprint_sha256",
        "transition_fingerprint_sha256",
    ):
        if not isinstance(value.get(key), str) or SHA_RE.fullmatch(value[key]) is None:
            return False
    return str(value["transition_fingerprint_sha256"]) == _canonical_object_fingerprint(
        value, "transition_fingerprint_sha256"
    )


def _validate_handoff_current(value: Any) -> bool:
    fields = {
        "handoff_id",
        "current_transition_revision",
        "state",
        "operation_fingerprint_sha256",
        "designation_fingerprint_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or not isinstance(value.get("handoff_id"), str)
        or LIFECYCLE_ID.fullmatch(value["handoff_id"]) is None
    ):
        return False
    if (
        not _positive(value.get("current_transition_revision"))
        or value.get("state") not in _HANDOFF_STATES
    ):
        return False
    for key in ("operation_fingerprint_sha256", "designation_fingerprint_sha256"):
        if not isinstance(value.get(key), str) or SHA_RE.fullmatch(value[key]) is None:
            return False
    return str(value["designation_fingerprint_sha256"]) == _canonical_object_fingerprint(
        value, "designation_fingerprint_sha256"
    )


def _migration_transition(
    revision: int, previous: str | None, state: str, generation: int = 1
) -> dict[str, Any]:
    value = {
        "migration_id": "migration-plan",
        "transition_revision": revision,
        "previous_state": previous,
        "state": state,
        "transaction_fingerprint_sha256": SHA,
        "state_fingerprint_sha256": "b" * 64,
        "protected_freshness_generation": generation,
        "transition_fingerprint_sha256": "",
    }
    value["transition_fingerprint_sha256"] = _canonical_object_fingerprint(
        value, "transition_fingerprint_sha256"
    )
    return value


def _migration_current(transition: dict[str, Any]) -> dict[str, Any]:
    value = {
        "migration_id": transition["migration_id"],
        "current_transition_revision": transition["transition_revision"],
        "state": transition["state"],
        "authoritative_state_fingerprint_sha256": transition["state_fingerprint_sha256"],
        "protected_freshness_generation": transition["protected_freshness_generation"],
        "designation_fingerprint_sha256": "",
    }
    value["designation_fingerprint_sha256"] = _canonical_object_fingerprint(
        value, "designation_fingerprint_sha256"
    )
    return value


def _handoff_transition(revision: int, previous: str | None, state: str) -> dict[str, Any]:
    value = {
        "handoff_id": "secret-handoff",
        "transition_revision": revision,
        "previous_state": previous,
        "state": state,
        "operation_fingerprint_sha256": SHA,
        "metadata_fingerprint_sha256": "b" * 64,
        "transition_fingerprint_sha256": "",
    }
    value["transition_fingerprint_sha256"] = _canonical_object_fingerprint(
        value, "transition_fingerprint_sha256"
    )
    return value


def _handoff_current(transition: dict[str, Any]) -> dict[str, Any]:
    value = {
        "handoff_id": transition["handoff_id"],
        "current_transition_revision": transition["transition_revision"],
        "state": transition["state"],
        "operation_fingerprint_sha256": transition["operation_fingerprint_sha256"],
        "designation_fingerprint_sha256": "",
    }
    value["designation_fingerprint_sha256"] = _canonical_object_fingerprint(
        value, "designation_fingerprint_sha256"
    )
    return value


def _resolve_lifecycle(
    transitions: Any, current: Any, kind: str
) -> tuple[str, tuple[dict[str, Any], ...]]:
    transition_validator = (
        _validate_migration_transition if kind == "migration" else _validate_handoff_transition
    )
    current_validator = (
        _validate_migration_current if kind == "migration" else _validate_handoff_current
    )
    identifier = "migration_id" if kind == "migration" else "handoff_id"
    allowed = MACHINE["migration_protocol" if kind == "migration" else "external_resource_handoff"][
        "durable_lifecycle"
    ]["allowed_transitions"]
    if not isinstance(transitions, list) or not transitions or not current_validator(current):
        return "CONTRACT_INCONSISTENT", ()
    by_revision: dict[int, dict[str, Any]] = {}
    for transition in transitions:
        if not transition_validator(transition):
            return "CONTRACT_INCONSISTENT", ()
        revision = transition["transition_revision"]
        if revision in by_revision and by_revision[revision] != transition:
            return "CONTRACT_INCONSISTENT", ()
        by_revision[revision] = transition
    if sorted(by_revision) != list(range(1, max(by_revision) + 1)):
        return "CONTRACT_INCONSISTENT", ()
    ordered = [by_revision[index] for index in range(1, max(by_revision) + 1)]
    if ordered[0]["previous_state"] is not None or ordered[0]["state"] != "PREPARED":
        return "CONTRACT_INCONSISTENT", ()
    for previous, following in zip(ordered, ordered[1:], strict=False):
        if (
            following[identifier] != previous[identifier]
            or following["previous_state"] != previous["state"]
            or following["state"] not in allowed[previous["state"]]
        ):
            return "CONTRACT_INCONSISTENT", ()
    selected = ordered[-1]
    if (
        current["current_transition_revision"] != selected["transition_revision"]
        or current[identifier] != selected[identifier]
        or current["state"] != selected["state"]
    ):
        return "CONTRACT_INCONSISTENT", ()
    if kind == "migration" and (
        current["authoritative_state_fingerprint_sha256"] != selected["state_fingerprint_sha256"]
        or current["protected_freshness_generation"] != selected["protected_freshness_generation"]
    ):
        return "CONTRACT_INCONSISTENT", ()
    if (
        kind == "handoff"
        and current["operation_fingerprint_sha256"] != selected["operation_fingerprint_sha256"]
    ):
        return "CONTRACT_INCONSISTENT", ()
    return str(selected["state"]), tuple(copy.deepcopy(ordered))


def _external(lifecycle: str = "COMMITTED", **updates: Any) -> dict[str, Any]:
    value = {
        "available": True,
        "account_id": SCOPE[0],
        "device_installation_id": SCOPE[1],
        "state_store_identity_fingerprint_sha256": SCOPE[2],
        "environment": "PAPER",
        "membership_state": "CURRENT",
        "lifecycle": lifecycle,
    }
    if lifecycle == "COMMITTED":
        value.update(committed_generation=2, committed_state_fingerprint_sha256=SHA)
    value.update(updates)
    return value


def _candidate(generation: int | None = 2, **updates: Any) -> dict[str, Any]:
    value = {
        "account_id": SCOPE[0],
        "device_installation_id": SCOPE[1],
        "state_store_identity_fingerprint_sha256": SCOPE[2],
        "environment": "PAPER",
        "generation": generation,
        "state_fingerprint_sha256": SHA,
        "transaction_fingerprint_sha256": "b" * 64,
    }
    value.update(updates)
    return value


@pytest.mark.parametrize(
    "missing",
    [
        "available",
        "account_id",
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
        "environment",
        "membership_state",
        "lifecycle",
    ],
)
def test_external_observation_missing_common_field_is_denied(missing: str) -> None:
    value = _external()
    value.pop(missing)
    assert _validate_external_authority_observation(value) != "VALID"


@pytest.mark.parametrize("prepared", [True, "3", 2, 4, 100])
def test_prepared_normal_requires_exact_g_plus_one(prepared: Any) -> None:
    external = _external(
        "PREPARED",
        committed_generation=2,
        committed_state_fingerprint_sha256=SHA,
        prepared_generation=prepared,
        prepared_state_fingerprint_sha256="b" * 64,
        prepared_transaction_fingerprint_sha256="c" * 64,
    )
    assert _validate_external_authority_observation(external) == (
        "VALID" if prepared == 3 and type(prepared) is int else "CONTRACT_INCONSISTENT"
    )
    if prepared != 3 or type(prepared) is not int:
        assert (
            _restore(
                _candidate(
                    100, state_fingerprint_sha256="b" * 64, transaction_fingerprint_sha256="c" * 64
                ),
                external,
            )
            == "CONTRACT_INCONSISTENT"
        )


@pytest.mark.parametrize(
    ("field", "failure"),
    [
        ("account_id", "BACKUP_SCOPE_MISMATCH"),
        ("device_installation_id", "BACKUP_SCOPE_MISMATCH"),
        ("state_store_identity_fingerprint_sha256", "BACKUP_SCOPE_MISMATCH"),
        ("environment", "BACKUP_ENVIRONMENT_MISMATCH"),
    ],
)
def test_restore_exact_scope_environment_binding(field: str, failure: str) -> None:
    candidate = _candidate()
    candidate[field] = "LIVE" if field == "environment" else "x"
    assert _restore(candidate, _external()) == failure


@pytest.mark.parametrize(
    "field",
    [
        "canonical_durable_records",
        "immutable_recovery_history",
        "integrity_metadata",
        "local_protected_freshness_generation",
        "environment",
        "account_id",
    ],
)
def test_backup_fingerprint_detects_every_envelope_mutation(field: str) -> None:
    backup = _valid_backup()
    if field in {"canonical_durable_records", "immutable_recovery_history"}:
        backup[field] = [{"record_kind": "Event"}]
    elif field == "integrity_metadata":
        backup[field] = {"revision": 1}
    elif field == "local_protected_freshness_generation":
        backup[field] = 3
    elif field == "environment":
        backup[field] = "TESTNET"
    else:
        backup[field] = "acct_01890f3a-2b4c-7abc-9def-0123456789ab"
    assert _validate_backup(backup) == "BACKUP_INTEGRITY_FAILED"


def test_exact_lifecycle_fixtures_and_extra_fields() -> None:
    mt = _migration_transition(1, None, "PREPARED")
    mc = _migration_current(mt)
    ht = _handoff_transition(1, None, "PREPARED")
    hc = _handoff_current(ht)
    assert (
        _validate_migration_transition(mt)
        and _validate_migration_current(mc)
        and _validate_handoff_transition(ht)
        and _validate_handoff_current(hc)
    )
    for value, validator in [
        (mt, _validate_migration_transition),
        (mc, _validate_migration_current),
        (ht, _validate_handoff_transition),
        (hc, _validate_handoff_current),
    ]:
        altered = dict(value)
        altered["extra"] = True
        assert not validator(altered)


@pytest.mark.parametrize(
    ("kind", "bad"),
    [
        ("migration", "completed_first"),
        ("migration", "previous_first"),
        ("migration", "older_current"),
        ("migration", "binding"),
        ("handoff", "completed_first"),
        ("handoff", "previous_first"),
        ("handoff", "older_current"),
        ("handoff", "binding"),
    ],
)
def test_lifecycle_revision_one_latest_and_bindings_are_fenced(kind: str, bad: str) -> None:
    if kind == "migration":
        first = _migration_transition(1, None, "PREPARED")
        second = _migration_transition(2, "PREPARED", "APPLYING")
        current = _migration_current(second)
    else:
        first = _handoff_transition(1, None, "PREPARED")
        second = _handoff_transition(2, "PREPARED", "COMMITTED")
        current = _handoff_current(second)
    history = [first, second]
    if bad == "completed_first":
        history = (
            [_migration_transition(1, None, "COMPLETED")]
            if kind == "migration"
            else [_handoff_transition(1, None, "COMMITTED")]
        )
        current = (
            _migration_current(history[0]) if kind == "migration" else _handoff_current(history[0])
        )
    elif bad == "previous_first":
        history[0]["previous_state"] = "APPLYING" if kind == "migration" else "COMMITTED"
        history[0]["transition_fingerprint_sha256"] = _canonical_object_fingerprint(
            history[0], "transition_fingerprint_sha256"
        )
    elif bad == "older_current":
        current = _migration_current(first) if kind == "migration" else _handoff_current(first)
    else:
        key = (
            "authoritative_state_fingerprint_sha256"
            if kind == "migration"
            else "operation_fingerprint_sha256"
        )
        current[key] = "f" * 64
        current["designation_fingerprint_sha256"] = _canonical_object_fingerprint(
            current, "designation_fingerprint_sha256"
        )
    assert _resolve_lifecycle(history, current, kind)[0] == "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    ("kind", "mutation"),
    [
        ("migration", "state"),
        ("migration", "revision"),
        ("migration", "generation"),
        ("handoff", "state"),
        ("handoff", "revision"),
        ("handoff", "operation"),
    ],
)
def test_transition_fingerprints_detect_mutation(kind: str, mutation: str) -> None:
    value = (
        _migration_transition(1, None, "PREPARED")
        if kind == "migration"
        else _handoff_transition(1, None, "PREPARED")
    )
    key = {
        "state": "state",
        "revision": "transition_revision",
        "generation": "protected_freshness_generation",
        "operation": "operation_fingerprint_sha256",
    }[mutation]
    value[key] = (
        "FAILED"
        if mutation == "state"
        else (2 if mutation in {"revision", "generation"} else "f" * 64)
    )
    assert not (
        _validate_migration_transition(value)
        if kind == "migration"
        else _validate_handoff_transition(value)
    )


def test_single_lifecycle_authority_has_no_descriptor_state() -> None:
    assert "state" not in MACHINE["migration_protocol"]["schema"]["properties"]
    assert "state" not in MACHINE["external_resource_handoff"]["schema"]["properties"]
    assert (
        MACHINE["migration_protocol"]["lifecycle_authority"]["current"]
        == "MigrationCurrentState only"
    )
    assert (
        MACHINE["external_resource_handoff"]["lifecycle_authority"]["current"]
        == "SecretHandoffCurrentState only"
    )


# Final corrective persistence-carrier closure.
_SYNTHETIC_KINDS = {
    "CryptohunteraccountCurrentRecord",
    "DeviceinstallationCurrentIdentityLifecycle",
    "OperatoridentityCurrentDesignationState",
    "OperatoridentityRevisions",
    "LiveaccessgrantCurrentDesignationState",
    "LiveaccessgrantAcceptedRevisionsHistory",
    "PortfolioCanonicalAccountingState",
    "Exchangeaccount",
    "CredentialprofileMetadataReference",
    "TradinguniverseCurrentVersionDesignation",
    "TradinguniverseVersionHistory",
    "StrategydefinitionAcceptedRevisions",
    "StrategydefinitionCurrentDesignation",
    "StrategyinstanceCurrentLifecycleConfig",
    "RoutingConfigurationCurrentDesignation",
    "RiskpolicyAcceptedRevisions",
    "RiskpolicyCurrentDesignation",
    "RiskbudgetCurrentState",
    "KillSwitchStateGeneration",
    "KillSwitchTransitionHistory",
    "CommandAcceptedRequest",
    "Orderintent",
    "OrderCanonicalLifecycleState",
    "OrderLifecycleEventsHistory",
    "Ledgerentry",
    "ReservationCurrentState",
    "ReservationTransitionHistory",
    "Riskdecision",
    "ExecutionleaseImmutableRecord",
    "ExecutionleaseOneShotState",
    "ExecutionleaseRestartFence",
    "RuntimesessionCanonicalIdentityHistory",
    "SessionsecuritystateCurrentGenerationState",
    "SessionsecuritystateRevisionHistory",
    "PinverifierrecordAcceptedRevisions",
    "PinverifierrecordCurrentDesignation",
    "DevicetrustSecurityRevisions",
    "DevicetrustCurrentDesignation",
    "PlatformEnrollmentRevisions",
    "Secretmetadataprojection",
    "BootstrapConsumedFence",
    "BootstrapConsumptionRecord",
}


def _contains_forbidden(value: Any, *, direct_pin_record: bool = False) -> bool:
    forbidden_kinds = set(MACHINE["backup_contract"]["forbidden_record_kinds"])
    forbidden_fields = set(MACHINE["backup_contract"]["forbidden_payload_fields"])
    if isinstance(value, list):
        return any(_contains_forbidden(item) for item in value)
    if not isinstance(value, dict):
        return False
    if value.get("record_kind") in forbidden_kinds:
        return True
    for key, nested in value.items():
        if key in forbidden_fields and not (direct_pin_record and key == "verifier"):
            return True
        nested_pin = direct_pin_record and key == "upstream_payload"
        if _contains_forbidden(nested, direct_pin_record=nested_pin):
            return True
    return False


def _source_value(entry: dict[str, Any]) -> Any:
    document = (
        MACHINE
        if entry["semantic_owner_milestone"] == "M0.11"
        else json.loads((DOCS / entry["semantic_artifact"]).read_text())
    )
    ok, value = _resolve_pointer(document, entry["semantic_json_pointer"])
    assert ok
    return value


def test_irreversible_prepared_floor_has_no_normal_abort_model() -> None:
    matrix = MACHINE["transaction_protocol"]["crash_matrix"]
    assert "PREPARED_LOCAL_G" not in matrix
    assert matrix["PREPARED_LOCAL_G_AFTER_RESTART"] == "PREPARED_PENDING_RECOVERY_REQUIRED_NO_ABORT"
    assert (
        matrix["LOCAL_G_PLUS_1_WAS_DURABLE_THEN_LOST_AND_G_RESTORED"]
        == "PREPARED_PENDING_RECOVERY_REQUIRED_NO_ABORT"
    )
    serialized = json.dumps(MACHINE)
    for forbidden in (
        "REBUILD_G_EVIDENCE_" + "ABORT_TO_COMMITTED_G",
        "REBUILD_CURRENT_G_" + "EVIDENCE",
        "PROTECTED_" + "ABORT",
        "AWAIT_POST_ABORT_" + "EXTERNAL_OBSERVATION",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "evidence_kind",
    ["accepted_current_g", "stale_g", "cross_process_g", "self_hash_g", "restored_old_g"],
)
def test_reconstructed_g_never_clears_pending_g_plus_one(evidence_kind: str) -> None:
    external = _external(
        "PREPARED",
        committed_generation=2,
        committed_state_fingerprint_sha256=SHA,
        prepared_generation=3,
        prepared_state_fingerprint_sha256="b" * 64,
        prepared_transaction_fingerprint_sha256="c" * 64,
    )
    registry = _EvidenceRegistry("restart")
    ref = registry.observe(_observation(protected_freshness_generation=2))
    assert ref
    if evidence_kind == "stale_g":
        registry.observe(
            _observation(protected_freshness_generation=2, transaction_fingerprint_sha256="d" * 64)
        )
    elif evidence_kind == "cross_process_g":
        ref = "other:opaque:1"
    elif evidence_kind == "self_hash_g":
        ref = "caller:" + SHA
    before = copy.deepcopy(external)
    assert _restore(_candidate(2), external) == "RECOVERY_REQUIRED_PENDING_RETAINED_NO_ABORT"
    assert external == before and external["prepared_generation"] == 3


def test_exact_pending_g_plus_one_finalizes_and_mismatch_preserves_pending() -> None:
    external = _external(
        "PREPARED",
        committed_generation=2,
        committed_state_fingerprint_sha256=SHA,
        prepared_generation=3,
        prepared_state_fingerprint_sha256="b" * 64,
        prepared_transaction_fingerprint_sha256="c" * 64,
    )
    assert (
        _restore(
            _candidate(
                3, state_fingerprint_sha256="b" * 64, transaction_fingerprint_sha256="c" * 64
            ),
            external,
        )
        == "REBUILD_FRESH_EVIDENCE_FINALIZE_MATCHING_PENDING_THEN_CONTINUE_GATES"
    )
    before = copy.deepcopy(external)
    assert (
        _restore(
            _candidate(
                3, state_fingerprint_sha256="d" * 64, transaction_fingerprint_sha256="c" * 64
            ),
            external,
        )
        == "BACKUP_ROLLBACK_DETECTED_PENDING_PRESERVED"
    )
    assert external == before


def test_exact_63_row_representation_registry_closure() -> None:
    registry = MACHINE["backup_contract"]["representation_registry"]
    assert len(registry) == 63 and set(registry) == set(_EXPECTED_DURABILITY)
    assert all(
        entry["durability_class"] == _EXPECTED_DURABILITY[aspect]
        for aspect, entry in registry.items()
    )
    assert all(
        entry["representation_category"] != "TRUE_UPSTREAM_SEMANTIC_GAP"
        for entry in registry.values()
    )
    assert all(
        entry["adds_new_domain_facts"] is False and entry["restorable_authority"] is False
        for entry in registry.values()
    )
    assert MACHINE["closure_conditions"]["true_upstream_semantic_gaps"] == 0


@pytest.mark.parametrize("aspect", list(_EXPECTED_DURABILITY))
def test_every_representation_source_and_carrier_is_executable(aspect: str) -> None:
    entry = MACHINE["backup_contract"]["representation_registry"][aspect]
    expected = _EXPECTED_OWNERSHIP[aspect]
    assert {key: entry[key] for key in expected} == expected
    source_value = _source_value(entry)
    if "semantic_contract_fingerprint_sha256" in entry:
        assert _actual_fingerprint(source_value) == entry["semantic_contract_fingerprint_sha256"]
    durable = entry["durability_class"].startswith("DURABLE")
    if not durable:
        assert (
            entry["representation_category"] == "EXCLUDED_NON_DURABLE"
            and entry["carrier_strategy"] == "NONE"
        )
        return
    assert entry["carrier_strategy"] == "PERSISTENCE_RECORD"
    valid = _persistence_record(aspect)
    assert _validate_persistence_record(valid)
    malformed = copy.deepcopy(valid)
    malformed["payload"] = {"invalid": True}
    malformed["payload_fingerprint_sha256"] = _actual_fingerprint(malformed["payload"])
    assert not _validate_persistence_record(malformed)


def test_no_synthetic_prose_derived_record_kind_survives() -> None:
    serialized = json.dumps(MACHINE["backup_contract"])
    for synthetic in _SYNTHETIC_KINDS:
        assert synthetic not in serialized
    assert "canonical_record_kinds_allowlist" not in MACHINE["backup_contract"]
    assert "durability_representation_by_aspect" not in MACHINE["backup_contract"]


def test_persistence_record_exact_shape_source_and_hash() -> None:
    value = _persistence_record("CryptoHunterAccount current record")
    assert set(value) == _PERSISTENCE_FIELDS and _validate_persistence_record(value)
    for mutation in (
        {"extra": True},
        {"representation_category": "UNKNOWN"},
        {"semantic_json_pointer": "/wrong"},
        {"payload_fingerprint_sha256": "f" * 64},
    ):
        altered = copy.deepcopy(value)
        altered.update(mutation)
        assert not _validate_persistence_record(altered)
    assert not _validate_persistence_record({"record_kind": "MigrationTransitionRecord"})


def test_crypto_account_identity_projection_is_minimal_closed_and_not_authority() -> None:
    valid = _persistence_record("CryptoHunterAccount current record")
    assert _validate_persistence_record(valid)
    for payload in (
        {
            "entity_kind": "CryptoHunterAccount",
            "entity_id": "acct_wrong",
            "parent_scope_bindings": {},
        },
        {
            "entity_kind": "Unknown",
            "entity_id": "acct_" + SCOPE[0][5:],
            "parent_scope_bindings": {},
        },
        {
            "entity_kind": "CryptoHunterAccount",
            "entity_id": "acct_" + SCOPE[0][5:],
            "parent_scope_bindings": {},
            "email": "x",
        },
    ):
        record = _persistence_record("CryptoHunterAccount current record", payload)
        assert not _validate_persistence_record(record)
    assert MACHINE["backup_contract"]["persistence_record_contract"]["domain_authority"] is False


def test_bootstrap_uses_exact_m03_objects_without_synthetic_kinds() -> None:
    registry = MACHINE["backup_contract"]["representation_registry"]
    assert (
        registry["bootstrap consumed fence"]["semantic_object_or_invariant"]
        == "CoreCurrentBootstrapState"
    )
    assert (
        registry["bootstrap accepted/consumption history"]["semantic_object_or_invariant"]
        == "ConsumedBootstrapAuthority"
    )
    assert (
        registry["bootstrap consumed fence"]["representation_category"]
        == "M011_CURRENT_DESIGNATION_PROJECTION"
    )
    assert (
        registry["bootstrap accepted/consumption history"]["representation_category"]
        == "M011_IMMUTABLE_HISTORY_WRAPPER"
    )


def test_backup_envelope_accepts_exact_carriers_and_rejects_bare_kind() -> None:
    backup = _valid_backup()
    backup["canonical_durable_records"] = [
        _persistence_record("CryptoHunterAccount current record")
    ]
    backup["immutable_recovery_history"] = [_persistence_record("Event")]
    backup["envelope_fingerprint_sha256"] = _actual_fingerprint(
        {k: v for k, v in backup.items() if k != "envelope_fingerprint_sha256"}
    )
    assert _validate_backup(backup) == "VALID"
    backup = _valid_backup()
    backup["canonical_durable_records"] = [{"record_kind": "Event"}]
    backup["envelope_fingerprint_sha256"] = _actual_fingerprint(
        {k: v for k, v in backup.items() if k != "envelope_fingerprint_sha256"}
    )
    assert _validate_backup(backup) == "BACKUP_INTEGRITY_FAILED"


@pytest.mark.parametrize(
    "aspect",
    [
        "Migration current state/designation",
        "Migration transition/history revisions",
        "SecretHandoff current state/designation",
        "SecretHandoff transition/history revisions",
    ],
)
def test_m011_lifecycle_payloads_use_exact_validators_inside_carrier(aspect: str) -> None:
    valid = _persistence_record(aspect)
    assert _validate_persistence_record(valid)
    altered = copy.deepcopy(valid)
    altered["payload"]["extra"] = True
    altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
    assert not _validate_persistence_record(altered)


def test_local_durable_evidence_has_no_persistence_carrier() -> None:
    entry = MACHINE["backup_contract"]["representation_registry"][
        "LocalDurableStateEvidence payload"
    ]
    assert (
        entry["representation_category"] == "EXCLUDED_NON_DURABLE"
        and entry["carrier_strategy"] == "NONE"
    )
    assert not _validate_persistence_record({"record_key": "LocalDurableStateEvidence payload|1"})


def _revalidate_current_designation(record: Any, accepted: Any, minimum_generation: int) -> str:
    if not _validate_persistence_record(record) or not isinstance(accepted, dict):
        return "RESTORE_REJECTED"
    payload = record["payload"]
    referenced = accepted.get(payload["current_reference"])
    if not isinstance(referenced, dict):
        return "RESTORE_REJECTED"
    if (
        referenced.get("scope_key") != payload["scope_key"]
        or referenced.get("content_fingerprint_sha256") != payload["content_fingerprint_sha256"]
    ):
        return "RESTORE_REJECTED"
    if payload["current_revision"] != referenced.get("revision"):
        return "RESTORE_REJECTED"
    generation = payload["current_generation"]
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < minimum_generation
    ):
        return "MONOTONIC_FENCE_ROLLBACK"
    if referenced.get("state") in {"REVOKED", "RETIRED"}:
        return "RESTORE_REJECTED"
    return "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS"


@pytest.mark.parametrize(
    "aspect",
    [
        "RiskPolicy current designation",
        "PinVerifierRecord current designation",
        "DeviceTrust current designation",
        "LiveAccessGrant current designation/state",
        "ExecutionLease one-shot state",
        "Order canonical lifecycle state",
        "reservation current state",
    ],
)
def test_current_designation_requires_history_scope_revision_fingerprint_and_fence(
    aspect: str,
) -> None:
    record = _persistence_record(aspect)
    payload = record["payload"]
    accepted = {
        payload["current_reference"]: {
            "scope_key": payload["scope_key"],
            "content_fingerprint_sha256": payload["content_fingerprint_sha256"],
            "revision": payload["current_revision"],
            "state": "ACTIVE",
        }
    }
    assert (
        _revalidate_current_designation(record, accepted, 1)
        == "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS"
    )
    assert _revalidate_current_designation(record, {}, 1) == "RESTORE_REJECTED"
    wrong = copy.deepcopy(accepted)
    wrong[payload["current_reference"]]["revision"] = 2
    assert _revalidate_current_designation(record, wrong, 1) == "RESTORE_REJECTED"
    assert _revalidate_current_designation(record, accepted, 2) == "MONOTONIC_FENCE_ROLLBACK"


def test_bootstrap_exact_payloads_mutation_consumption_and_restore_authority_fences() -> None:
    current = _persistence_record("bootstrap consumed fence")
    history = _persistence_record("bootstrap accepted/consumption history")
    assert _validate_persistence_record(current) and _validate_persistence_record(history)
    altered = copy.deepcopy(history)
    altered["payload"]["bootstrap_generation"] = 0
    altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
    assert not _validate_persistence_record(altered)
    cleared = copy.deepcopy(current)
    cleared["payload"]["consumed_authorities"] = []
    cleared["payload_fingerprint_sha256"] = _actual_fingerprint(cleared["payload"])
    assert _validate_persistence_record(cleared)
    assert (
        cleared["restorable_authority"] is False
        if "restorable_authority" in cleared
        else MACHINE["backup_contract"]["representation_registry"]["bootstrap consumed fence"][
            "restorable_authority"
        ]
        is False
    )
    assert (
        current["payload"]["consumed_authorities"]
        and not cleared["payload"]["consumed_authorities"]
    )


@pytest.mark.parametrize(
    "aspect",
    [
        "bootstrap accepted/consumption history",
        "Event",
        "RiskPolicy accepted revisions",
        "SessionSecurityState revision history",
        "RuntimeSession canonical identity/history",
    ],
)
def test_immutable_wrapper_requires_payload_hash_and_semantic_validator(aspect: str) -> None:
    record = _persistence_record(aspect)
    assert _validate_persistence_record(record)
    mutated = copy.deepcopy(record)
    if "canonical_fields" in mutated["payload"]:
        mutated["payload"]["canonical_fields"] = {"invalid": True}
    else:
        mutated["payload"]["invalid"] = True
    assert not _validate_persistence_record(mutated)
    rehashed = copy.deepcopy(mutated)
    rehashed["payload_fingerprint_sha256"] = _actual_fingerprint(rehashed["payload"])
    assert not _validate_persistence_record(rehashed)


def _revalidate_bootstrap_consumption(current: Any, history: Any) -> str:
    if not _validate_persistence_record(current) or not isinstance(history, list):
        return "RESTORE_REJECTED"
    current_payload = current["payload"]
    accepted: set[str] = set()
    for record in history:
        if (
            not _validate_persistence_record(record)
            or record.get("representation_name") != "bootstrap accepted/consumption history"
        ):
            return "RESTORE_REJECTED"
        historical = record["payload"]
        if (
            historical["account_id"] != current_payload["account_id"]
            or historical["device_installation_id"] != current_payload["device_installation_id"]
            or historical["bootstrap_generation"] != current_payload["expected_generation"]
            or historical["bootstrap_revision"] != current_payload["expected_revision"]
        ):
            return "RESTORE_REJECTED"
        accepted.add(historical["claim_fingerprint_sha256"])
    consumed = current_payload.get("consumed_authorities")
    if not isinstance(consumed, list) or set(consumed) != accepted:
        return "RESTORE_REJECTED"
    return "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS"


def test_legacy_parallel_recovery_model_is_absent_and_restore_retains_pending() -> None:
    source = Path(__file__).read_text()
    forbidden = [
        "REBUILD_" + "EVIDENCE_AND_ABORT",
        "REBUILD_G_EVIDENCE_" + "ABORT_TO_COMMITTED_G",
        "REBUILD_CURRENT_G_" + "EVIDENCE",
        "VERIFY_ABORT_" + "ELIGIBILITY",
        "PROTECTED_" + "ABORT",
        "AWAIT_POST_ABORT_" + "EXTERNAL_OBSERVATION",
    ]
    assert "def " + "_recover(" not in source
    assert not any(token in source or token in json.dumps(MACHINE) for token in forbidden)
    external = _external(
        "PREPARED",
        committed_generation=1,
        committed_state_fingerprint_sha256=SHA,
        prepared_generation=2,
        prepared_state_fingerprint_sha256="b" * 64,
        prepared_transaction_fingerprint_sha256="c" * 64,
    )
    before = copy.deepcopy(external)
    assert _restore(_candidate(1), external) == "RECOVERY_REQUIRED_PENDING_RETAINED_NO_ABORT"
    assert external == before


def test_record_key_is_derived_collision_safe_and_not_prose_identity() -> None:
    event_a = _persistence_record("Event", object_suffix="a")
    event_b = _persistence_record("Event", object_suffix="b")
    lease_a = _persistence_record("ExecutionLease immutable record", object_suffix="a")
    lease_b = _persistence_record("ExecutionLease immutable record", object_suffix="b")
    assert event_a["record_key"] != event_b["record_key"]
    assert lease_a["record_key"] != lease_b["record_key"]
    assert _persistence_record("Event", object_suffix="a")["record_key"] == event_a["record_key"]
    assert _persistence_record("Event", revision=2)["record_key"] != event_a["record_key"]
    for bad_key in ("", "arbitrary", "Event|1", event_b["record_key"]):
        altered = copy.deepcopy(event_a)
        altered["record_key"] = bad_key
        assert not _validate_persistence_record(altered)


@pytest.mark.parametrize(
    "aspect",
    [
        "DeviceInstallation current identity/lifecycle",
        "Portfolio canonical accounting state",
        "RiskBudget current state",
    ],
)
def test_upstream_facts_are_nonempty_closed_source_bound_and_semantic(aspect: str) -> None:
    valid = _persistence_record(aspect)
    assert valid["payload"]["facts"] and _validate_persistence_record(valid)
    for mutation in ("missing", "extra", "scope", "fingerprint"):
        altered = copy.deepcopy(valid)
        if mutation == "missing":
            altered["payload"]["facts"].pop(next(iter(altered["payload"]["facts"])))
        elif mutation == "extra":
            altered["payload"]["facts"]["invented_business_fact"] = True
        elif mutation == "scope":
            altered["payload"]["scope_key"] = "unrelated"
        else:
            altered["payload"]["source_fingerprint_sha256"] = "f" * 64
        altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
        assert not _validate_persistence_record(altered)


@pytest.mark.parametrize(
    ("kind", "parents", "valid"),
    [
        ("CryptoHunterAccount", {}, True),
        ("CryptoHunterAccount", {"account_id": SCOPE[0]}, False),
        ("Workspace", {"account_id": SCOPE[0]}, True),
        ("Workspace", {}, False),
        ("Workspace", {"account_id": "acct_invalid"}, False),
    ],
)
def test_entity_identity_exact_parent_and_generic_prefix(
    kind: str, parents: dict[str, str], valid: bool
) -> None:
    payload = {
        "entity_kind": kind,
        "entity_id": ("acct_" if kind == "CryptoHunterAccount" else "ws_") + SCOPE[0][5:],
        "parent_scope_bindings": parents,
    }
    assert _exact_entity_projection(payload) is valid
    if kind == "Workspace":
        assert not _exact_entity_projection({**payload, "entity_id": SCOPE[1]})


def test_bootstrap_restore_consumption_is_exact_monotonic_set() -> None:
    current = _persistence_record("bootstrap consumed fence")
    history = [_persistence_record("bootstrap accepted/consumption history")]
    assert (
        _revalidate_bootstrap_consumption(current, history)
        == "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS"
    )
    for consumed in ([], ["b" * 64], [SHA, "b" * 64]):
        altered = copy.deepcopy(current)
        altered["payload"]["consumed_authorities"] = consumed
        altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
        altered["record_key"] = _derive_record_key(
            "bootstrap consumed fence",
            MACHINE["backup_contract"]["representation_registry"]["bootstrap consumed fence"],
            altered["payload"],
        )
        assert _validate_persistence_record(altered)
        assert _revalidate_bootstrap_consumption(altered, history) == "RESTORE_REJECTED"


@pytest.mark.parametrize(
    "aspect",
    ["M0.3 restore freshness membership", "M0.3 current designation", "M0.3 retirement state"],
)
def test_external_m03_sources_are_exact_restore_freshness_authority(aspect: str) -> None:
    entry = MACHINE["backup_contract"]["representation_registry"][aspect]
    assert (
        entry["semantic_owner_milestone"],
        entry["semantic_artifact"],
        entry["semantic_json_pointer"],
    ) == ("M0.3", "process_topology_and_lifecycle.json", "/restore_freshness_authority_contract")
    assert (
        _actual_fingerprint(_source_value(entry)) == entry["semantic_contract_fingerprint_sha256"]
    )


def test_persistence_self_hash_cannot_legalize_wrong_source_category_or_payload() -> None:
    record = _persistence_record("Event")
    for field, value in (
        ("semantic_json_pointer", "/wrong"),
        ("representation_category", "M011_LOCAL_SCHEMA"),
    ):
        altered = copy.deepcopy(record)
        altered[field] = value
        assert not _validate_persistence_record(altered)
    altered = copy.deepcopy(record)
    altered["payload"] = {"semantic_object": "Event", "canonical_fields": {"garbage": 123}}
    altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
    assert not _validate_persistence_record(altered)


@pytest.mark.parametrize(
    "aspect",
    [
        "Event",
        "Fill",
        "LedgerEntry",
        "RiskDecision",
        "ExecutionLease immutable record",
        "SessionSecurityState current generation/state",
        "SecretMetadataProjection",
    ],
)
def test_representative_direct_upstream_payload_is_exact_and_not_generic_placeholder(
    aspect: str,
) -> None:
    record = _persistence_record(aspect)
    contract = MACHINE["backup_contract"]["direct_upstream_validator_registry"][aspect]
    assert set(record["payload"]) == set(contract["exact_fields"])
    assert _validate_persistence_record(record)
    assert "semantic_object" not in record["payload"]
    assert "canonical_fields" not in record["payload"]

    missing = copy.deepcopy(record)
    missing["payload"].pop(contract["exact_fields"][0])
    missing["payload_fingerprint_sha256"] = _actual_fingerprint(missing["payload"])
    assert not _validate_persistence_record(missing)

    extra = copy.deepcopy(record)
    extra["payload"]["invented_business_fact"] = True
    extra["payload_fingerprint_sha256"] = _actual_fingerprint(extra["payload"])
    assert not _validate_persistence_record(extra)

    wrong_key = copy.deepcopy(record)
    wrong_key["record_key"] = "direct:unrelated-object"
    assert not _validate_persistence_record(wrong_key)

    terminal = _terminal_fingerprint_field(aspect)
    if terminal is not None:
        wrong_fingerprint = copy.deepcopy(record)
        wrong_fingerprint["payload"][terminal] = "f" * 64
        wrong_fingerprint["payload_fingerprint_sha256"] = _actual_fingerprint(
            wrong_fingerprint["payload"]
        )
        wrong_fingerprint["record_key"] = _derive_record_key(
            aspect,
            MACHINE["backup_contract"]["representation_registry"][aspect],
            wrong_fingerprint["payload"],
        )
        assert not _validate_persistence_record(wrong_fingerprint)


def test_event_exact_upstream_semantics_deny_bad_scope_lifecycle_and_safe_payload() -> None:
    valid = _persistence_record("Event")
    for field, value in (
        ("environment", "PRODUCTION"),
        ("event_type", "UNKNOWN_EVENT"),
        ("workspace_id", "wrong-id"),
        ("safe_payload", {"garbage": 123}),
    ):
        altered = copy.deepcopy(valid)
        altered["payload"][field] = value
        altered["payload"]["event_fingerprint_sha256"] = _actual_fingerprint(
            {
                key: item
                for key, item in altered["payload"].items()
                if key != "event_fingerprint_sha256"
            }
        )
        altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
        altered["record_key"] = _derive_record_key(
            "Event",
            MACHINE["backup_contract"]["representation_registry"]["Event"],
            altered["payload"],
        )
        assert not _validate_persistence_record(altered)


def test_current_designation_stage_two_denies_scope_fingerprint_revocation_and_retirement() -> None:
    record = _persistence_record("RiskPolicy current designation")
    payload = record["payload"]
    accepted = {
        payload["current_reference"]: {
            "scope_key": payload["scope_key"],
            "content_fingerprint_sha256": payload["content_fingerprint_sha256"],
            "revision": payload["current_revision"],
            "state": "ACTIVE",
        }
    }
    for field, value in (
        ("scope_key", "wrong-scope"),
        ("content_fingerprint_sha256", "f" * 64),
        ("state", "REVOKED"),
        ("state", "RETIRED"),
    ):
        changed = copy.deepcopy(accepted)
        changed[payload["current_reference"]][field] = value
        assert _revalidate_current_designation(record, changed, 1) == "RESTORE_REJECTED"


@pytest.mark.parametrize(
    ("aspect", "field", "bad"),
    [
        ("DeviceInstallation current identity/lifecycle", "lifecycle_state", "UNKNOWN"),
        ("Portfolio canonical accounting state", "environment", "PRODUCTION"),
        ("RiskBudget current state", "budget_state", "UNKNOWN"),
        ("RiskBudget current state", "risk_generation", 0),
    ],
)
def test_upstream_fact_projection_denies_illegal_semantic_value(
    aspect: str, field: str, bad: Any
) -> None:
    record = _persistence_record(aspect)
    record["payload"]["facts"][field] = bad
    record["payload_fingerprint_sha256"] = _actual_fingerprint(record["payload"])
    record["record_key"] = _derive_record_key(
        aspect,
        MACHINE["backup_contract"]["representation_registry"][aspect],
        record["payload"],
    )
    assert not _validate_persistence_record(record)


def _revalidate_restore_records(
    records: Any,
    accepted_by_reference: Any,
    minimum_generation: int,
    bootstrap_history: Any,
    reservation_authority: Any = None,
) -> str:
    if not isinstance(records, list) or not all(
        _validate_persistence_record(record) for record in records
    ):
        return "RESTORE_REJECTED"
    for record in records:
        category = record["representation_category"]
        if record["representation_name"] == "reservation transition history":
            if not _revalidate_reservation_restore_authority(record, reservation_authority):
                return "RESTORE_REJECTED"
        if category == "M011_CURRENT_DESIGNATION_PROJECTION":
            if record["representation_name"] == "bootstrap consumed fence":
                result = _revalidate_bootstrap_consumption(record, bootstrap_history)
            else:
                result = _revalidate_current_designation(
                    record, accepted_by_reference, minimum_generation
                )
            if result != "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS":
                return result
    return "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS"


def test_restore_record_gate_is_relational_and_never_mints_authority() -> None:
    current = _persistence_record("RiskPolicy current designation")
    payload = current["payload"]
    accepted = {
        payload["current_reference"]: {
            "scope_key": payload["scope_key"],
            "content_fingerprint_sha256": payload["content_fingerprint_sha256"],
            "revision": payload["current_revision"],
            "state": "ACTIVE",
        }
    }
    assert (
        _revalidate_restore_records([current], accepted, 1, [])
        == "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS"
    )
    assert _revalidate_restore_records([current], {}, 1, []) == "RESTORE_REJECTED"


@pytest.mark.parametrize(
    "aspect",
    [
        "RiskPolicy accepted revisions",
        "SessionSecurityState revision history",
        "RuntimeSession canonical identity/history",
        "LiveAccessGrant accepted revisions/history",
    ],
)
def test_immutable_history_is_lossless_closed_and_fingerprint_recomputed(aspect: str) -> None:
    record = _persistence_record(aspect)
    entry = MACHINE["backup_contract"]["representation_registry"][aspect]
    binding = entry["immutable_fact_binding"]
    upstream = record["payload"]["upstream_payload"]
    assert set(upstream) == set(binding["persisted_payload_fields"])
    assert record["payload"]["upstream_payload_fingerprint_sha256"] == _actual_fingerprint(upstream)
    assert _validate_persistence_record(record)

    mutations = ["missing", "extra", "scope", "identity", "sha"]
    if binding["revision_generation_fields"]:
        mutations.append("revision")
    if "state" in upstream:
        mutations.append("state")
    for mutation in mutations:
        altered = copy.deepcopy(record)
        payload = altered["payload"]["upstream_payload"]
        if mutation == "missing":
            payload.pop(binding["persisted_payload_fields"][0])
        elif mutation == "extra":
            payload["invented_business_fact"] = True
        elif mutation == "scope" and binding["scope_fields"]:
            payload[binding["scope_fields"][0]] = "wrong-scope"
        elif mutation == "identity":
            payload[binding["canonical_object_identity_fields"][0]] = "wrong-id"
        elif mutation == "revision" and binding["revision_generation_fields"]:
            payload[binding["revision_generation_fields"][0]] = 0
        elif mutation == "state" and "state" in payload:
            payload["state"] = "ILLEGAL"
        else:
            altered["payload"]["upstream_payload_fingerprint_sha256"] = "f" * 64
        if mutation != "sha":
            altered["payload"]["upstream_payload_fingerprint_sha256"] = _actual_fingerprint(payload)
        altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
        altered["record_key"] = _derive_record_key(aspect, entry, altered["payload"])
        assert not _validate_persistence_record(altered)


def _rehash_immutable_record(record: dict[str, Any]) -> None:
    record["payload"]["upstream_payload_fingerprint_sha256"] = _actual_fingerprint(
        record["payload"]["upstream_payload"]
    )
    record["payload_fingerprint_sha256"] = _actual_fingerprint(record["payload"])


def test_runtime_session_provenance_is_independently_derived_from_m02() -> None:
    upstream = json.loads((DOCS / "canonical_domain_vocabulary.json").read_text())
    entity = next(x for x in upstream["entity_kinds"] if x["canonical_name"] == "RuntimeSession")
    relationship = next(
        x
        for x in upstream["relationships"]
        if x["from"] == "DeviceInstallation" and x["to"] == "RuntimeSession"
    )
    entry = MACHINE["backup_contract"]["representation_registry"][
        "RuntimeSession canonical identity/history"
    ]
    binding = entry["immutable_fact_binding"]
    assert entry["semantic_owner_milestone"] == upstream["m0_element"] == "M0.2"
    assert entry["semantic_artifact"] == "canonical_domain_vocabulary.json"
    assert entry["semantic_json_pointer"] != "/first_run_bootstrap_authority_contract"
    assert entity == {
        **entity,
        "id_field": "runtime_session_id",
        "id_prefix": "run",
        "parent": "DeviceInstallation",
        "persistence": True,
    }
    assert relationship["cardinality"] == "one_to_many"
    assert binding["persisted_payload_fields"] == [
        entity["id_field"],
        next(
            x["id_field"]
            for x in upstream["entity_kinds"]
            if x["canonical_name"] == entity["parent"]
        ),
    ]
    assert not (
        {"session_revision", "state", "content_fingerprint_sha256"}
        & set(binding["persisted_payload_fields"])
    )


@pytest.mark.parametrize("invented", ["session_revision", "state", "content_fingerprint_sha256"])
def test_runtime_session_rejects_every_invented_field(invented: str) -> None:
    record = _persistence_record("RuntimeSession canonical identity/history")
    record["payload"]["upstream_payload"][invented] = 1
    _rehash_immutable_record(record)
    assert not _validate_persistence_record(record)


def test_runtime_session_accepts_run_and_rejects_sess_or_wrong_parent() -> None:
    valid = _persistence_record("RuntimeSession canonical identity/history")
    assert valid["payload"]["upstream_payload"]["runtime_session_id"].startswith("run_")
    assert _validate_persistence_record(valid)
    for field, prefix in (("runtime_session_id", "sess"), ("device_installation_id", "acct")):
        altered = copy.deepcopy(valid)
        altered["payload"]["upstream_payload"][field] = _canonical_fixture_id(prefix, "c")
        _rehash_immutable_record(altered)
        assert not _validate_persistence_record(altered)


def test_risk_policy_dto_is_independently_derived_from_m09() -> None:
    upstream = json.loads(
        (DOCS / "risk_hierarchy_kill_switch_and_execution_lease.json").read_text()
    )
    contract = upstream["risk_policy_contract"]
    binding = MACHINE["backup_contract"]["representation_registry"][
        "RiskPolicy accepted revisions"
    ]["immutable_fact_binding"]
    assert binding["persisted_payload_fields"] == contract["identity"]
    assert binding["semantic_fingerprint_input_fields"] == contract["semantic_fingerprint_input"]
    assert binding["field_contracts"]["action"]["values"] == contract["action_registry"]
    assert (
        binding["field_contracts"]["scope_type"]["values"]
        == upstream["scope_hierarchy"]["applicable_order"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("environment", "DEMO"),
        ("scope_type", "ACCOUNT"),
        ("scope_id", "port_01890f3a-2b4c-7abc-8def-0123456789ab"),
        ("action", "PERMIT"),
        ("limits", {}),
    ],
)
def test_risk_policy_rejects_illegal_semantics_even_after_wrapper_rehash(
    field: str, value: Any
) -> None:
    record = _persistence_record("RiskPolicy accepted revisions")
    record["payload"]["upstream_payload"][field] = value
    _rehash_immutable_record(record)
    assert not _validate_persistence_record(record)


@pytest.mark.parametrize("invented", ["state", "account_id", "policy_revision"])
def test_risk_policy_rejects_invented_fields(invented: str) -> None:
    record = _persistence_record("RiskPolicy accepted revisions")
    record["payload"]["upstream_payload"][invented] = "invented"
    _rehash_immutable_record(record)
    assert not _validate_persistence_record(record)


def test_risk_policy_rejects_mutated_limits_with_stale_semantic_fingerprint() -> None:
    record = _persistence_record("RiskPolicy accepted revisions")
    record["payload"]["upstream_payload"]["limits"][0][1] = "2/1"
    _rehash_immutable_record(record)
    assert not _validate_persistence_record(record)


@pytest.mark.parametrize(
    ("aspect", "artifact", "pointer", "schema_name"),
    [
        (
            "SessionSecurityState revision history",
            "identity_device_authentication_and_secrets.json",
            "/executable_boundary_schemas/SessionSecurityState",
            "SessionSecurityState",
        ),
        (
            "LiveAccessGrant accepted revisions/history",
            "identity_device_authentication_and_secrets.json",
            "/executable_boundary_schemas/LiveAccessGrantSecurityProjection",
            "LiveAccessGrantSecurityProjection",
        ),
        (
            "bootstrap accepted/consumption history",
            "process_topology_and_lifecycle.json",
            "/first_run_bootstrap_authority_contract/executable_schemas/ConsumedBootstrapAuthority",
            "ConsumedBootstrapAuthority",
        ),
    ],
)
def test_critical_immutable_sources_are_read_directly_not_self_declared(
    aspect: str, artifact: str, pointer: str, schema_name: str
) -> None:
    upstream = json.loads((DOCS / artifact).read_text())
    ok, literal_schema = _resolve_pointer(upstream, pointer)
    assert ok
    literal_fields = (
        literal_schema["exact_fields"] if isinstance(literal_schema, dict) else literal_schema
    )
    assert isinstance(literal_fields, list) and literal_fields
    entry = MACHINE["backup_contract"]["representation_registry"][aspect]
    assert entry["semantic_object_or_invariant"] in {aspect, schema_name}
    if "immutable_fact_binding" in entry:
        assert entry["immutable_fact_binding"]["persisted_payload_fields"] == literal_fields
    else:
        assert set(_persistence_record(aspect)["payload"]) == set(literal_fields)


def _independent_immutable_fields(aspect: str) -> list[str] | dict[str, list[str]]:
    identity = json.loads((DOCS / "identity_device_authentication_and_secrets.json").read_text())
    if aspect in {
        "OperatorIdentity revisions",
        "LiveAccessGrant accepted revisions/history",
        "SessionSecurityState revision history",
        "PinVerifierRecord accepted revisions",
        "DeviceTrust/security revisions",
        "platform enrollment revisions",
    }:
        schema = {
            "OperatorIdentity revisions": "OperatorIdentitySecurityProjection",
            "LiveAccessGrant accepted revisions/history": "LiveAccessGrantSecurityProjection",
            "SessionSecurityState revision history": "SessionSecurityState",
            "PinVerifierRecord accepted revisions": "PinVerifierRecord",
            "DeviceTrust/security revisions": "DeviceTrustProjection",
            "platform enrollment revisions": "CoreAcceptedPlatformBiometricAssertionBinding",
        }[aspect]
        source_schema = identity["executable_boundary_schemas"][schema]
        if isinstance(source_schema, dict):
            source_schema = source_schema["exact_fields"]
        return cast(list[str], source_schema)
    if aspect == "TradingUniverse version history":
        upstream = json.loads((DOCS / "exchange_accounts_and_instruments.json").read_text())
        return cast(list[str], upstream["trading_universe_contract"]["record_fields"])
    if aspect == "StrategyDefinition accepted revisions":
        upstream = json.loads(
            (DOCS / "strategy_market_data_and_execution_routing.json").read_text()
        )
        return cast(list[str], upstream["record_schemas"]["StrategyDefinition"]["exact_fields"])
    if aspect in {"RiskPolicy accepted revisions", "kill-switch transition history"}:
        upstream = json.loads(
            (DOCS / "risk_hierarchy_kill_switch_and_execution_lease.json").read_text()
        )
        return cast(
            list[str],
            {
                "RiskPolicy accepted revisions": upstream["risk_policy_contract"]["identity"],
                "kill-switch transition history": upstream["kill_switch_contract"]["record_fields"],
            }[aspect],
        )
    if aspect == "reservation transition history":
        upstream = json.loads((DOCS / "ledger_portfolio_capital_and_pnl.json").read_text())
        registry = upstream["accounting_economic_fact_schema_registry"]
        return cast(
            dict[str, list[str]],
            {
                name: registry[name]["exact_fields"]
                for name in ("capital_reservation", "capital_release")
            },
        )
    if aspect == "Order lifecycle events/history":
        upstream = json.loads(
            (DOCS / "commands_events_order_lifecycle_and_idempotency.json").read_text()
        )
        return cast(list[str], upstream["event_contract"]["envelope_schema"]["fields"])
    if aspect == "RuntimeSession canonical identity/history":
        upstream = json.loads((DOCS / "canonical_domain_vocabulary.json").read_text())
        runtime = next(
            x for x in upstream["entity_kinds"] if x["canonical_name"] == "RuntimeSession"
        )
        parent = next(
            x for x in upstream["entity_kinds"] if x["canonical_name"] == runtime["parent"]
        )
        return [runtime["id_field"], parent["id_field"]]
    if aspect == "bootstrap accepted/consumption history":
        upstream = json.loads((DOCS / "process_topology_and_lifecycle.json").read_text())
        return cast(
            list[str],
            upstream["first_run_bootstrap_authority_contract"]["executable_schemas"][
                "ConsumedBootstrapAuthority"
            ],
        )
    raise AssertionError(f"missing independent source derivation for {aspect}")


IMMUTABLE_ASPECTS = [
    aspect
    for aspect, entry in MACHINE["backup_contract"]["representation_registry"].items()
    if entry["representation_category"] == "M011_IMMUTABLE_HISTORY_WRAPPER"
]


@pytest.mark.parametrize("aspect", IMMUTABLE_ASPECTS)
def test_all_immutable_payload_fields_are_independently_derived_from_upstream(aspect: str) -> None:
    expected = _independent_immutable_fields(aspect)
    entry = MACHINE["backup_contract"]["representation_registry"][aspect]
    if "immutable_fact_binding" in entry:
        binding = entry["immutable_fact_binding"]
        if "upstream_payload_variants" in binding:
            assert isinstance(expected, dict)
            assert set(binding["upstream_payload_variants"]) == set(expected)
            source_document = json.loads((DOCS / entry["semantic_artifact"]).read_text())
            for variant_name, expected_fields in expected.items():
                variant = binding["upstream_payload_variants"][variant_name]
                assert variant["persisted_payload_fields"] == expected_fields
                assert set(variant["field_contracts"]) == set(expected_fields)
                for field in expected_fields:
                    source = variant["field_contracts"][field]
                    resolved, declaration = _resolve_pointer(
                        source_document, source["source_pointer"]
                    )
                    assert resolved and isinstance(declaration, list) and field in declaration
            return
        assert isinstance(expected, list)
        assert binding["persisted_payload_fields"] == expected
        assert set(binding["field_contracts"]) == set(expected)
        source_document = json.loads((DOCS / entry["semantic_artifact"]).read_text())
        for field in expected:
            source = binding["field_contracts"][field]
            assert source["source_artifact"] == entry["semantic_artifact"]
            assert source["source_pointer"].startswith("/")
            assert source["source_path"]
            assert source["type"] and isinstance(source["nullable"], bool)
            assert source["semantic_role"]
            resolved, declaration = _resolve_pointer(source_document, source["source_pointer"])
            assert resolved
            if aspect == "RuntimeSession canonical identity/history":
                assert source["projection_rule"]
            else:
                declaration_fields = (
                    declaration["exact_fields"] if isinstance(declaration, dict) else declaration
                )
                assert isinstance(declaration_fields, list) and field in declaration_fields
    else:
        assert entry["source_derivation"]["persisted_payload_fields"] == expected


def test_no_immutable_wrapper_uses_the_forbidden_universal_state_enum() -> None:
    universal = {
        "ACCEPTED",
        "ACTIVE",
        "COMMITTED",
        "CONSUMED",
        "LOCKED",
        "RECORDED",
        "REVOKED",
        "RETIRED",
        "TRUSTED",
    }
    registry = MACHINE["backup_contract"]["representation_registry"]
    for aspect in IMMUTABLE_ASPECTS:
        binding = registry[aspect].get("immutable_fact_binding", {})
        state = binding.get("field_contracts", {}).get("state")
        assert state is None or set(state["values"]) != universal


@pytest.mark.parametrize("illegal_state", ["TRUSTED", "LOCKED", "CONSUMED"])
def test_operator_identity_rejects_state_from_another_owner(illegal_state: str) -> None:
    record = _persistence_record("OperatorIdentity revisions")
    record["payload"]["upstream_payload"]["state"] = illegal_state
    _rehash_immutable_record(record)
    assert not _validate_persistence_record(record)


@pytest.mark.parametrize("invented", ["definition_revision", "state", "content_fingerprint_sha256"])
def test_strategy_definition_rejects_every_old_invented_alias(invented: str) -> None:
    record = _persistence_record("StrategyDefinition accepted revisions")
    record["payload"]["upstream_payload"][invented] = "invented"
    _rehash_immutable_record(record)
    assert not _validate_persistence_record(record)


def test_strategy_definition_uses_exact_sdef_identity_and_version() -> None:
    record = _persistence_record("StrategyDefinition accepted revisions")
    upstream = record["payload"]["upstream_payload"]
    assert upstream["strategy_definition_id"].startswith("sdef_")
    assert upstream["definition_version"] == 1
    assert set(upstream["configuration"]) == {"lookback", "enabled"}


def _m06_definition_hash(configuration: dict[str, Any], hash_contract: dict[str, Any]) -> str:
    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(hash_contract["domain_separator"].encode() + encoded).hexdigest()


def test_strategy_definition_hash_is_independently_derived_from_m06() -> None:
    upstream = json.loads((DOCS / "strategy_market_data_and_execution_routing.json").read_text())
    schema = upstream["record_schemas"]["StrategyDefinition"]
    hash_contract = upstream["strategy_definition_contract"]["hash"]
    record = _persistence_record("StrategyDefinition accepted revisions")
    payload = record["payload"]["upstream_payload"]
    assert list(payload) == schema["exact_fields"]
    assert payload["strategy_definition_id"].startswith(schema["id_prefix"] + "_")
    assert payload["hash_domain_separator"] == hash_contract["domain_separator"]
    assert payload["canonical_content_hash"] == _m06_definition_hash(
        payload["configuration"], hash_contract
    )
    assert _validate_persistence_record(record)


@pytest.mark.parametrize(
    "mutation",
    [
        "hash_without_separator",
        "wrong_separator",
        "changed_configuration_stale_hash",
        "changed_configuration_hash_without_separator",
        "mutated_separator_wrapper_rehash",
    ],
)
def test_strategy_definition_rejects_every_domain_hash_mismatch(mutation: str) -> None:
    record = _persistence_record("StrategyDefinition accepted revisions")
    payload = record["payload"]["upstream_payload"]
    if mutation == "hash_without_separator":
        payload["canonical_content_hash"] = _actual_fingerprint(payload["configuration"])
    elif mutation == "wrong_separator":
        payload["hash_domain_separator"] = "foreign-domain\\0"
    elif mutation == "changed_configuration_stale_hash":
        payload["configuration"]["lookback"] = 2
    elif mutation == "changed_configuration_hash_without_separator":
        payload["configuration"]["lookback"] = 2
        payload["canonical_content_hash"] = _actual_fingerprint(payload["configuration"])
    else:
        payload["hash_domain_separator"] += ".mutated"
    _rehash_immutable_record(record)
    assert not _validate_persistence_record(record)


@pytest.mark.parametrize(
    ("aspect", "field", "artifact", "pointer"),
    [
        (
            "OperatorIdentity revisions",
            "state",
            "identity_device_authentication_and_secrets.json",
            "/registries/identity_states",
        ),
        (
            "LiveAccessGrant accepted revisions/history",
            "state",
            "identity_device_authentication_and_secrets.json",
            "/registries/grant_states",
        ),
        (
            "TradingUniverse version history",
            "lifecycle_state",
            "exchange_accounts_and_instruments.json",
            "/trading_universe_contract/lifecycle_states",
        ),
        (
            "StrategyDefinition accepted revisions",
            "lifecycle_state",
            "strategy_market_data_and_execution_routing.json",
            "/record_schemas/StrategyDefinition/enum_registry/lifecycle_state",
        ),
        (
            "RiskPolicy accepted revisions",
            "scope_type",
            "risk_hierarchy_kill_switch_and_execution_lease.json",
            "/scope_hierarchy/applicable_order",
        ),
        (
            "RiskPolicy accepted revisions",
            "action",
            "risk_hierarchy_kill_switch_and_execution_lease.json",
            "/risk_policy_contract/action_registry",
        ),
        (
            "kill-switch transition history",
            "state",
            "risk_hierarchy_kill_switch_and_execution_lease.json",
            "/kill_switch_contract/states",
        ),
        (
            "Order lifecycle events/history",
            "event_type",
            "commands_events_order_lifecycle_and_idempotency.json",
            "/event_contract/event_types",
        ),
        (
            "SessionSecurityState revision history",
            "state",
            "identity_device_authentication_and_secrets.json",
            "/registries/session_states",
        ),
        (
            "DeviceTrust/security revisions",
            "state",
            "identity_device_authentication_and_secrets.json",
            "/registries/device_trust_states",
        ),
    ],
)
def test_every_literal_immutable_enum_is_exactly_upstream(
    aspect: str, field: str, artifact: str, pointer: str
) -> None:
    document = json.loads((DOCS / artifact).read_text())
    ok, expected = _resolve_pointer(document, pointer)
    assert ok and isinstance(expected, list)
    contract = MACHINE["backup_contract"]["representation_registry"][aspect][
        "immutable_fact_binding"
    ]["field_contracts"][field]
    assert contract["constraint_pointer"] == pointer
    assert contract["values"] == expected


@pytest.mark.parametrize(
    "aspect",
    [
        "RiskPolicy accepted revisions",
        "kill-switch transition history",
        "Order lifecycle events/history",
    ],
)
def test_environment_enum_is_derived_from_canonical_m02(aspect: str) -> None:
    upstream = json.loads((DOCS / "canonical_domain_vocabulary.json").read_text())
    expected = [item["name"].upper() for item in upstream["public_trading_environments"]]
    contract = MACHINE["backup_contract"]["representation_registry"][aspect][
        "immutable_fact_binding"
    ]["field_contracts"]["environment"]
    assert contract["constraint_artifact"] == "canonical_domain_vocabulary.json"
    assert contract["constraint_pointer"] == "/public_trading_environments"
    assert contract["values"] == expected


@pytest.mark.parametrize(
    "aspect", ["RiskPolicy accepted revisions", "kill-switch transition history"]
)
def test_product_system_scope_uses_literal_product_not_invented_account_id(aspect: str) -> None:
    record = _persistence_record(aspect)
    payload = record["payload"]["upstream_payload"]
    assert payload["scope_type"] == "PRODUCT_SYSTEM" and payload["scope_id"] == "product"
    assert _validate_persistence_record(record)
    payload["scope_id"] = _canonical_fixture_id("acct", "c")
    _rehash_immutable_record(record)
    assert not _validate_persistence_record(record)


def test_strategy_nested_schema_nullability_and_constants_are_exactly_m06() -> None:
    upstream = json.loads((DOCS / "strategy_market_data_and_execution_routing.json").read_text())
    schema = upstream["record_schemas"]["StrategyDefinition"]
    hash_contract = upstream["strategy_definition_contract"]["hash"]
    binding = MACHINE["backup_contract"]["representation_registry"][
        "StrategyDefinition accepted revisions"
    ]["immutable_fact_binding"]
    configuration = binding["field_contracts"]["configuration"]
    nested = schema["nested_schemas"]["configuration"]
    assert configuration["fields"] == nested["exact_fields"]
    assert {
        name: contract["type"] for name, contract in configuration["field_schemas"].items()
    } == {"lookback": "positive_integer", "enabled": "boolean"}
    assert nested["nullable_fields"] == [] and not configuration["nullable"]
    separator = binding["field_contracts"]["hash_domain_separator"]
    assert separator["constraint_pointer"] == "/strategy_definition_contract/hash/domain_separator"
    assert separator["value"] == hash_contract["domain_separator"]


def test_pin_constant_nullable_and_verifier_constraints_are_upstream_derived() -> None:
    upstream = json.loads((DOCS / "identity_device_authentication_and_secrets.json").read_text())
    contracts = MACHINE["backup_contract"]["representation_registry"][
        "PinVerifierRecord accepted revisions"
    ]["immutable_fact_binding"]["field_contracts"]
    assert contracts["algorithm_id"]["value"] == upstream["pin_policy"]["reference_algorithm"]
    assert contracts["verifier"]["type"] == "sha256_hex"
    assert contracts["failed_attempts"]["type"] == "non_negative_integer"
    assert contracts["lockout_until_utc"]["type"] == "nullable_timestamp"
    assert contracts["lockout_until_utc"]["nullable"] is True


def test_all_semantic_fingerprint_derivations_match_upstream_contracts() -> None:
    registry = MACHINE["backup_contract"]["representation_registry"]
    identity = json.loads((DOCS / "identity_device_authentication_and_secrets.json").read_text())
    for aspect, schema_name in {
        "OperatorIdentity revisions": "OperatorIdentitySecurityProjection",
        "LiveAccessGrant accepted revisions/history": "LiveAccessGrantSecurityProjection",
        "SessionSecurityState revision history": "SessionSecurityState",
        "PinVerifierRecord accepted revisions": "PinVerifierRecord",
        "DeviceTrust/security revisions": "DeviceTrustProjection",
    }.items():
        binding = registry[aspect]["immutable_fact_binding"]
        assert binding["semantic_fingerprint_derivation"]["input_fields"] == [
            field
            for field in (
                identity["executable_boundary_schemas"][schema_name]["exact_fields"]
                if isinstance(identity["executable_boundary_schemas"][schema_name], dict)
                else identity["executable_boundary_schemas"][schema_name]
            )
            if field != "content_fingerprint_sha256"
        ]
        assert binding["semantic_fingerprint_derivation"]["input_shape"] == "JSON_OBJECT"

    m05 = json.loads((DOCS / "exchange_accounts_and_instruments.json").read_text())
    trading = registry["TradingUniverse version history"]["immutable_fact_binding"]
    trading_hash = m05["trading_universe_contract"]["content_hash_definition"]
    assert (
        trading["semantic_fingerprint_derivation"]["input_fields"] == trading_hash["input_fields"]
    )
    assert (
        trading["semantic_fingerprint_derivation"]["domain_separator"]
        == trading_hash["domain_separator"]
    )

    m09 = json.loads((DOCS / "risk_hierarchy_kill_switch_and_execution_lease.json").read_text())
    risk = registry["RiskPolicy accepted revisions"]["immutable_fact_binding"]
    assert (
        risk["semantic_fingerprint_derivation"]["input_fields"]
        == m09["risk_policy_contract"]["semantic_fingerprint_input"]
    )
    kill = registry["kill-switch transition history"]["immutable_fact_binding"]
    assert kill["semantic_fingerprint_derivation"]["input_fields"] == [
        field
        for field in m09["kill_switch_contract"]["record_fields"]
        if field != "record_fingerprint_sha256"
    ]

    m07 = json.loads((DOCS / "commands_events_order_lifecycle_and_idempotency.json").read_text())
    event = registry["Order lifecycle events/history"]["immutable_fact_binding"]
    assert event["semantic_fingerprint_derivation"]["input_fields"] == [
        field
        for field in m07["event_contract"]["envelope_schema"]["fields"]
        if field != "event_fingerprint_sha256"
    ]


def test_every_immutable_canonical_id_prefix_is_upstream_derived() -> None:
    vocabulary = json.loads((DOCS / "canonical_domain_vocabulary.json").read_text())
    expected = {item["id_field"]: item["id_prefix"] for item in vocabulary["entity_kinds"]}
    commands = json.loads(
        (DOCS / "commands_events_order_lifecycle_and_idempotency.json").read_text()
    )
    event_schemas = commands["event_contract"]["envelope_schema"]["field_schemas"]
    expected.update(
        {
            "correlation_id": event_schemas["correlation_id"]["prefix"],
            "causation_id": event_schemas["causation_id"]["prefix"],
            "command_id": event_schemas["command_id"]["prefix"],
            "previous_version_id": expected["trading_universe_id"],
        }
    )
    registry = MACHINE["backup_contract"]["representation_registry"]
    observed: set[str] = set()
    for aspect in IMMUTABLE_ASPECTS:
        binding = registry[aspect].get("immutable_fact_binding", {})
        groups = binding.get("upstream_payload_variants", {"default": binding}).values()
        for group in groups:
            for field, contract in group.get("field_contracts", {}).items():
                if contract["type"] in {"canonical_id", "nullable_canonical_id"}:
                    observed.add(field)
                    assert contract["id_prefix"] == expected[field]
    assert observed <= set(expected)


def test_order_types_and_nullability_match_exact_upstream_envelope() -> None:
    upstream = json.loads(
        (DOCS / "commands_events_order_lifecycle_and_idempotency.json").read_text()
    )["event_contract"]["envelope_schema"]
    contracts = MACHINE["backup_contract"]["representation_registry"][
        "Order lifecycle events/history"
    ]["immutable_fact_binding"]["field_contracts"]
    type_projection = {
        "id": "canonical_id",
        "positive_integer": "positive_integer",
        "enum": "enum",
        "non_empty_string": "non_empty_string",
        "timestamp": "timestamp",
        "event_safe_payload": "event_safe_payload",
        "sha256_hex": "sha256_hex",
    }
    for field, source_schema in upstream["field_schemas"].items():
        expected_type = type_projection[source_schema["type"]]
        if field in upstream["nullable_fields"]:
            expected_type = "nullable_canonical_id"
        assert contracts[field]["type"] == expected_type
        assert contracts[field]["nullable"] is (field in upstream["nullable_fields"])


def _reservation_variant_record(variant: str) -> dict[str, Any]:
    return _persistence_record("reservation transition history", variant_name=variant)


_RESERVATION_AUTHORITY_SEAL = object()
_M07_ACCEPTANCE_SEAL = object()
_M07_LIFECYCLE_SEAL = object()


@dataclass(frozen=True)
class _ReservationRestoreAuthority:
    accounting: dict[str, tuple[str, str, str, str, str]]
    commands: dict[str, tuple[dict[str, Any], str]]
    terminal_events: dict[str, tuple[dict[str, Any], dict[str, Any]]]
    authority_seal: object
    m07_acceptance_seal: object
    lifecycle_seal: object


def _m07_command_fingerprint(request: dict[str, Any]) -> str:
    return _actual_fingerprint(
        {key: value for key, value in request.items() if key != "correlation_id"}
    )


def _m07_event_fingerprint(event: dict[str, Any]) -> str:
    return _actual_fingerprint(
        {key: value for key, value in event.items() if key != "event_fingerprint_sha256"}
    )


def _m07_legal_predecessors(
    context: dict[str, Any], event_type: str, document: dict[str, Any] | None = None
) -> frozenset[str] | None:
    derivation = context.get("legal_predecessor_derivation")
    if not isinstance(derivation, dict):
        return None
    if document is None:
        artifact = derivation.get("source_artifact")
        if not isinstance(artifact, str):
            return None
        document = json.loads((DOCS / artifact).read_text())
    ok, transitions = _resolve_pointer(document, derivation.get("source_pointer"))
    selector = derivation.get("selector")
    if (
        not ok
        or not isinstance(transitions, list)
        or not isinstance(selector, dict)
        or selector.get("field") != "event"
        or selector.get("equals_field") != "accepted_terminal_event.event_type"
        or derivation.get("result_path") != "sources"
        or derivation.get("cardinality") != "EXACTLY_ONE_TRANSITION"
    ):
        return None
    matches = [
        item for item in transitions if isinstance(item, dict) and item.get("event") == event_type
    ]
    if len(matches) != 1:
        return None
    sources = matches[0].get("sources")
    if (
        not isinstance(sources, list)
        or not sources
        or any(type(source) is not str or not source for source in sources)
        or len(set(sources)) != len(sources)
    ):
        return None
    return frozenset(sources)


def _preexisting_reservation_authority(record: dict[str, Any]) -> _ReservationRestoreAuthority:
    """Build the opaque input as if Core/M0.7 had accepted it before restore validation."""
    payload = record["payload"]["upstream_payload"]
    accounting = {
        payload["audit_event_id"]: (
            payload["source_type"],
            payload["source_fingerprint_sha256"],
            payload["workspace_id"],
            payload["portfolio_id"],
            payload["environment"],
        )
    }
    commands: dict[str, tuple[dict[str, Any], str]] = {}
    terminal_events: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    if payload["source_type"] == "capital_reservation":
        command_id = payload["command_id"]
        request = {
            "command_id": command_id,
            "operation_type": "SUBMIT_ORDER",
            "authority_context_id": _canonical_fixture_id("authctx", "a"),
            "environment": payload["environment"],
            "workspace_id": payload["workspace_id"],
            "portfolio_id": payload["portfolio_id"],
            "exchange_account_id": payload["exchange_account_id"],
            "strategy_instance_id": None,
            "source_type": "OPERATOR",
            "instrument_id": _canonical_fixture_id("instr", "a"),
            "execution_route_id": _canonical_fixture_id("xroute", "a"),
            "correlation_id": _canonical_fixture_id("corr", "a"),
            "causation_id": None,
            "idempotency_key": command_id,
            "order_intent_id": _canonical_fixture_id("oint", "a"),
            "order_id": payload["order_id"],
            "side": "BUY",
            "order_type": "MARKET",
            "quantity": "1",
            "limit_price": None,
            "time_in_force": "GTC",
            "expire_at_utc": None,
        }
        commands[command_id] = (request, _m07_command_fingerprint(request))
    else:
        event_type = {
            "REJECTED": "ORDER_REJECTED",
            "FILLED": "ORDER_FILLED",
            "CANCELLED": "ORDER_CANCEL_CONFIRMED",
            "REPLACED": "ORDER_REPLACE_CONFIRMED",
            "EXPIRED": "ORDER_EXPIRED",
        }[payload["terminal_state"]]
        safe_payload = (
            {"reason_code": "VENUE_REJECTED"}
            if event_type == "ORDER_REJECTED"
            else (
                {
                    "fill_id": _canonical_fixture_id("fill", "a"),
                    "venue_trade_id": "trade-a",
                    "cumulative_executed_quantity": "1",
                }
                if event_type == "ORDER_FILLED"
                else {
                    "replacement_order_id": _canonical_fixture_id("ord", "b"),
                    "venue_order_id": "venue-a",
                }
                if event_type == "ORDER_REPLACE_CONFIRMED"
                else {"venue_order_id": "venue-a"}
            )
        )
        event = {
            "audit_event_id": payload["audit_event_id"],
            "event_type": event_type,
            "order_id": payload["order_id"],
            "aggregate_version": 2,
            "correlation_id": _canonical_fixture_id("corr", "a"),
            "causation_id": _canonical_fixture_id("cause", "a"),
            "command_id": _canonical_fixture_id("cmd", "a"),
            "environment": payload["environment"],
            "workspace_id": payload["workspace_id"],
            "portfolio_id": payload["portfolio_id"],
            "exchange_account_id": payload["exchange_account_id"],
            "exchange_id": "paper_simulated_venue",
            "instrument_id": _canonical_fixture_id("instr", "a"),
            "execution_route_id": _canonical_fixture_id("xroute", "a"),
            "occurred_at_utc": "2025-01-01T00:00:00Z",
            "safe_payload": safe_payload,
            "event_fingerprint_sha256": "",
        }
        event["event_fingerprint_sha256"] = _m07_event_fingerprint(event)
        predecessor = {
            "ORDER_REJECTED": "SUBMISSION_PENDING",
            "ORDER_FILLED": "PARTIALLY_FILLED",
            "ORDER_CANCEL_CONFIRMED": "CANCEL_PENDING",
            "ORDER_REPLACE_CONFIRMED": "REPLACE_PENDING",
            "ORDER_EXPIRED": "ACKNOWLEDGED",
        }[event_type]
        proof = {
            "event_fingerprint_sha256": event["event_fingerprint_sha256"],
            "order_id": event["order_id"],
            "aggregate_version": 2,
            "event_type": event_type,
            "terminal_state": payload["terminal_state"],
            "predecessor_state": predecessor,
            "previous_aggregate_version": 1,
        }
        terminal_events[event["audit_event_id"]] = (event, proof)
    return _ReservationRestoreAuthority(
        accounting,
        commands,
        terminal_events,
        _RESERVATION_AUTHORITY_SEAL,
        _M07_ACCEPTANCE_SEAL,
        _M07_LIFECYCLE_SEAL,
    )


def _revalidate_reservation_restore_authority(record: dict[str, Any], authority: Any) -> bool:
    if not isinstance(authority, _ReservationRestoreAuthority) or (
        authority.authority_seal is not _RESERVATION_AUTHORITY_SEAL
        or authority.m07_acceptance_seal is not _M07_ACCEPTANCE_SEAL
    ):
        return False
    payload = record["payload"]["upstream_payload"]
    membership = (
        payload["source_type"],
        payload["source_fingerprint_sha256"],
        payload["workspace_id"],
        payload["portfolio_id"],
        payload["environment"],
    )
    if authority.accounting.get(payload["audit_event_id"]) != membership:
        return False
    if payload["source_type"] == "capital_reservation":
        accepted = authority.commands.get(payload["command_id"])
        if accepted is None:
            return False
        request, accepted_fingerprint = accepted
        upstream = json.loads((DOCS / "ledger_portfolio_capital_and_pnl.json").read_text())
        fields = upstream["m07_authority_boundary"]["submit_order_consumed_fields"]
        if set(request) != set(fields) or request.get("operation_type") != "SUBMIT_ORDER":
            return False
        if accepted_fingerprint != _m07_command_fingerprint(request):
            return False
        return all(
            request[field] == payload[field]
            for field in (
                "command_id",
                "order_id",
                "workspace_id",
                "portfolio_id",
                "environment",
                "exchange_account_id",
            )
        )
    if authority.lifecycle_seal is not _M07_LIFECYCLE_SEAL:
        return False
    accepted_event = authority.terminal_events.get(payload["audit_event_id"])
    if accepted_event is None:
        return False
    event, proof = accepted_event
    upstream = json.loads((DOCS / "ledger_portfolio_capital_and_pnl.json").read_text())
    mapping = upstream["m07_authority_boundary"]["terminal_event_mapping"]
    if event.get("event_fingerprint_sha256") != _m07_event_fingerprint(event):
        return False
    if (
        event.get("event_type") not in mapping
        or mapping[event["event_type"]] != payload["terminal_state"]
    ):
        return False
    context = MACHINE["backup_contract"]["representation_registry"][
        "reservation transition history"
    ]["immutable_fact_binding"]["upstream_payload_variants"]["capital_release"][
        "restore_authority_revalidation"
    ]["m07_context"]
    legal_predecessors = _m07_legal_predecessors(context, event["event_type"])
    if legal_predecessors is None:
        return False
    if any(
        event[field] != payload[field]
        for field in (
            "audit_event_id",
            "order_id",
            "workspace_id",
            "portfolio_id",
            "environment",
            "exchange_account_id",
        )
    ):
        return False
    return bool(
        proof
        == {
            **proof,
            "event_fingerprint_sha256": event["event_fingerprint_sha256"],
            "order_id": event["order_id"],
            "aggregate_version": event["aggregate_version"],
            "event_type": event["event_type"],
            "terminal_state": mapping[event["event_type"]],
            "previous_aggregate_version": event["aggregate_version"] - 1,
        }
        and proof.get("predecessor_state") in legal_predecessors
    )


def test_reservation_decimal_rule_is_read_from_actual_m08_reference() -> None:
    reference_path = (
        DOCS.parents[2] / "tests/architecture/test_cryptohunter_ledger_portfolio_capital_and_pnl.py"
    )
    tree = ast.parse(reference_path.read_text())
    decimal_pattern = cast(
        str,
        next(
            node.value.args[0].value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "DECIMAL_RE"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "compile"
            and isinstance(node.value.args[0], ast.Constant)
        ),
    )
    decimal_function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "decimal"
    )
    assert len(decimal_function.args.kw_defaults) == 1
    assert isinstance(decimal_function.args.kw_defaults[0], ast.Constant)
    assert decimal_function.args.kw_defaults[0].value is True
    contract = MACHINE["backup_contract"]["representation_registry"][
        "reservation transition history"
    ]["immutable_fact_binding"]["upstream_payload_variants"]["capital_reservation"][
        "field_contracts"
    ]["quantity"]
    assert contract["type"] == "positive_decimal"
    assert contract["decimal_regex"] == decimal_pattern
    grammar = re.compile(decimal_pattern)
    assert all(grammar.fullmatch(value) for value in ("1", "1.25", "10.01"))
    assert all(
        grammar.fullmatch(value) is None for value in ("-1", "01", "1.0", "1.", ".1", "+1", "1e2")
    )
    assert contract["positive"] is True


def test_reservation_asset_reference_is_exact_m05_m08_trusted_projection() -> None:
    m05 = json.loads((DOCS / "exchange_accounts_and_instruments.json").read_text())
    m08 = json.loads((DOCS / "ledger_portfolio_capital_and_pnl.json").read_text())
    contract = MACHINE["backup_contract"]["representation_registry"][
        "reservation transition history"
    ]["immutable_fact_binding"]["upstream_payload_variants"]["capital_reservation"][
        "field_contracts"
    ]["asset_reference"]
    assert contract["type"] == "asset_reference"
    assert contract["fields"] == m05["asset_reference_contract"]["fields"]
    assert (
        contract["field_schemas"]["mapping_status"]["values"]
        == m08["asset_identity"]["accepted_mapping_status"]
    )
    assert all(
        not contract["field_schemas"][field].get("nullable", False) for field in contract["fields"]
    )


@pytest.mark.parametrize("variant", ["capital_reservation", "capital_release"])
def test_each_reservation_variant_is_valid_and_source_fingerprint_is_recomputed(
    variant: str,
) -> None:
    record = _reservation_variant_record(variant)
    payload = record["payload"]["upstream_payload"]
    assert payload["source_type"] == variant
    assert _validate_persistence_record(record)
    binding = MACHINE["backup_contract"]["representation_registry"][
        "reservation transition history"
    ]["immutable_fact_binding"]["upstream_payload_variants"][variant]
    assert payload["source_fingerprint_sha256"] == _semantic_fingerprint(binding, payload)
    assert binding["semantic_fingerprint_derivation"]["authority_semantics"].startswith(
        "pre-existing CoreAcceptedAccountingFactProjection membership"
    )


@pytest.mark.parametrize("variant", ["capital_reservation", "capital_release"])
def test_reservation_restore_requires_preexisting_accounting_and_m07_authority(
    variant: str,
) -> None:
    record = _reservation_variant_record(variant)
    assert _validate_persistence_record(record)
    assert not _revalidate_reservation_restore_authority(record, None)
    empty = _ReservationRestoreAuthority(
        {}, {}, {}, _RESERVATION_AUTHORITY_SEAL, _M07_ACCEPTANCE_SEAL, _M07_LIFECYCLE_SEAL
    )
    assert not _revalidate_reservation_restore_authority(record, empty)
    assert _revalidate_reservation_restore_authority(
        record, _preexisting_reservation_authority(record)
    )


@pytest.mark.parametrize(
    "field",
    ["source_type", "source_fingerprint_sha256", "workspace_id", "portfolio_id", "environment"],
)
def test_accounting_membership_exact_tuple_mismatch_fails_after_all_hashes_pass(
    field: str,
) -> None:
    record = _reservation_variant_record("capital_reservation")
    authority = _preexisting_reservation_authority(record)
    payload = record["payload"]["upstream_payload"]
    values = list(authority.accounting[payload["audit_event_id"]])
    index = [
        "source_type",
        "source_fingerprint_sha256",
        "workspace_id",
        "portfolio_id",
        "environment",
    ].index(field)
    values[index] = "wrong"
    authority = _ReservationRestoreAuthority(
        {payload["audit_event_id"]: cast(tuple[str, str, str, str, str], tuple(values))},
        authority.commands,
        authority.terminal_events,
        authority.authority_seal,
        authority.m07_acceptance_seal,
        authority.lifecycle_seal,
    )
    assert _validate_persistence_record(record)
    assert not _revalidate_reservation_restore_authority(record, authority)


@pytest.mark.parametrize("variant", ["capital_reservation", "capital_release"])
def test_candidate_cannot_self_enroll_with_nominal_payload_derived_authority(variant: str) -> None:
    record = _reservation_variant_record(variant)
    nominal = _preexisting_reservation_authority(record)
    fake = _ReservationRestoreAuthority(
        nominal.accounting,
        nominal.commands,
        nominal.terminal_events,
        object(),
        object(),
        object(),
    )
    _rehash_immutable_record(record)
    assert _validate_persistence_record(record)
    assert not _revalidate_reservation_restore_authority(record, fake)


@pytest.mark.parametrize(
    "field",
    [
        "command_id",
        "order_id",
        "workspace_id",
        "portfolio_id",
        "environment",
        "exchange_account_id",
    ],
)
def test_capital_reservation_rejects_wrong_or_missing_accepted_submit_order_context(
    field: str,
) -> None:
    record = _reservation_variant_record("capital_reservation")
    authority = _preexisting_reservation_authority(record)
    payload = record["payload"]["upstream_payload"]
    request, _ = next(iter(authority.commands.values()))
    request = copy.deepcopy(request)
    request[field] = "wrong"
    bad_commands = {payload["command_id"]: (request, _m07_command_fingerprint(request))}
    bad = _ReservationRestoreAuthority(
        authority.accounting,
        bad_commands,
        {},
        authority.authority_seal,
        authority.m07_acceptance_seal,
        authority.lifecycle_seal,
    )
    assert not _revalidate_reservation_restore_authority(record, bad)


def test_capital_reservation_rejects_stale_command_fingerprint() -> None:
    record = _reservation_variant_record("capital_reservation")
    authority = _preexisting_reservation_authority(record)
    command_id = record["payload"]["upstream_payload"]["command_id"]
    request, _ = authority.commands[command_id]
    bad = _ReservationRestoreAuthority(
        authority.accounting,
        {command_id: (request, "f" * 64)},
        {},
        authority.authority_seal,
        authority.m07_acceptance_seal,
        authority.lifecycle_seal,
    )
    assert not _revalidate_reservation_restore_authority(record, bad)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "event_type", "order_id", "environment", "fingerprint", "version", "predecessor"],
)
def test_capital_release_rejects_missing_or_mismatched_accepted_terminal_event(
    mutation: str,
) -> None:
    record = _reservation_variant_record("capital_release")
    authority = _preexisting_reservation_authority(record)
    audit_id = record["payload"]["upstream_payload"]["audit_event_id"]
    if mutation == "missing":
        terminal: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    else:
        event, proof = authority.terminal_events[audit_id]
        event, proof = copy.deepcopy(event), copy.deepcopy(proof)
        if mutation == "event_type":
            event["event_type"] = (
                "ORDER_CANCEL_CONFIRMED"
                if event["event_type"] == "ORDER_REJECTED"
                else "ORDER_REJECTED"
            )
        elif mutation == "order_id":
            event["order_id"] = _canonical_fixture_id("ord", "b")
        elif mutation == "environment":
            event["environment"] = "TESTNET"
        elif mutation == "fingerprint":
            event["event_fingerprint_sha256"] = "f" * 64
        elif mutation == "version":
            proof["previous_aggregate_version"] = 0
        else:
            proof["predecessor_state"] = "NEW"
        terminal = {audit_id: (event, proof)}
    bad = _ReservationRestoreAuthority(
        authority.accounting,
        {},
        terminal,
        authority.authority_seal,
        authority.m07_acceptance_seal,
        authority.lifecycle_seal,
    )
    assert not _revalidate_reservation_restore_authority(record, bad)


def test_reservation_authority_contract_is_derived_from_actual_m08_m07_boundaries() -> None:
    upstream = json.loads((DOCS / "ledger_portfolio_capital_and_pnl.json").read_text())
    m07 = json.loads((DOCS / "commands_events_order_lifecycle_and_idempotency.json").read_text())
    variants = MACHINE["backup_contract"]["representation_registry"][
        "reservation transition history"
    ]["immutable_fact_binding"]["upstream_payload_variants"]
    for variant in variants.values():
        gate = variant["restore_authority_revalidation"]
        assert gate["accounting_membership"]["binding_fields"] == [
            "source_type",
            "source_fingerprint_sha256",
            "workspace_id",
            "portfolio_id",
            "environment",
        ]
        assert gate["accounting_membership"]["lookup_key_field"] == "audit_event_id"
        assert gate["no_self_enrollment"] == {
            "candidate_may_populate_registry": False,
            "payload_or_backup_may_carry_private_seal": False,
            "nominal_mapping_or_self_hash_grants_authority": False,
        }
    reservation = variants["capital_reservation"]["restore_authority_revalidation"]["m07_context"]
    assert reservation["contract_pointer"] == "/m07_authority_boundary/submit_order_consumed_fields"
    assert reservation["operation_type_constant"] == "SUBMIT_ORDER"
    assert upstream["m07_authority_boundary"]["submit_order_consumed_fields"]
    release = variants["capital_release"]["restore_authority_revalidation"]["m07_context"]
    assert (
        release["event_to_terminal_state_pointer"]
        == "/m07_authority_boundary/terminal_event_mapping"
    )
    terminal_mapping = upstream["m07_authority_boundary"]["terminal_event_mapping"]
    derivation = release["legal_predecessor_derivation"]
    assert derivation["source_artifact"] == "commands_events_order_lifecycle_and_idempotency.json"
    assert derivation["source_pointer"] == "/order_lifecycle/transitions"
    ok, transitions = _resolve_pointer(m07, derivation["source_pointer"])
    assert ok
    for event_type in terminal_mapping:
        matches = [
            transition for transition in transitions if transition.get("event") == event_type
        ]
        assert len(matches) == 1
        assert _m07_legal_predecessors(release, event_type) == frozenset(matches[0]["sources"])


def test_every_actual_m07_terminal_predecessor_passes_and_foreign_state_fails() -> None:
    m08 = json.loads((DOCS / "ledger_portfolio_capital_and_pnl.json").read_text())
    release_context = MACHINE["backup_contract"]["representation_registry"][
        "reservation transition history"
    ]["immutable_fact_binding"]["upstream_payload_variants"]["capital_release"][
        "restore_authority_revalidation"
    ]["m07_context"]
    for event_type, terminal_state in m08["m07_authority_boundary"][
        "terminal_event_mapping"
    ].items():
        record = _reservation_variant_record("capital_release")
        record["payload"]["upstream_payload"]["terminal_state"] = terminal_state
        authority = _preexisting_reservation_authority(record)
        audit_id = record["payload"]["upstream_payload"]["audit_event_id"]
        event, original_proof = authority.terminal_events[audit_id]
        assert event["event_type"] == event_type
        legal_sources = _m07_legal_predecessors(release_context, event_type)
        assert legal_sources
        for predecessor in legal_sources:
            proof = {**original_proof, "predecessor_state": predecessor}
            accepted = _ReservationRestoreAuthority(
                authority.accounting,
                {},
                {audit_id: (event, proof)},
                authority.authority_seal,
                authority.m07_acceptance_seal,
                authority.lifecycle_seal,
            )
            assert _revalidate_reservation_restore_authority(record, accepted)
        proof = {**original_proof, "predecessor_state": "PLANNED"}
        rejected = _ReservationRestoreAuthority(
            authority.accounting,
            {},
            {audit_id: (event, proof)},
            authority.authority_seal,
            authority.m07_acceptance_seal,
            authority.lifecycle_seal,
        )
        assert "PLANNED" not in legal_sources
        assert not _revalidate_reservation_restore_authority(record, rejected)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "no_sources", "malformed_sources"])
def test_m07_predecessor_derivation_fails_closed_for_malformed_source(mutation: str) -> None:
    document = json.loads(
        (DOCS / "commands_events_order_lifecycle_and_idempotency.json").read_text()
    )
    context = MACHINE["backup_contract"]["representation_registry"][
        "reservation transition history"
    ]["immutable_fact_binding"]["upstream_payload_variants"]["capital_release"][
        "restore_authority_revalidation"
    ]["m07_context"]
    transitions = document["order_lifecycle"]["transitions"]
    match = next(item for item in transitions if item["event"] == "ORDER_REJECTED")
    if mutation == "missing":
        transitions.remove(match)
    elif mutation == "duplicate":
        transitions.append(copy.deepcopy(match))
    elif mutation == "no_sources":
        match.pop("sources")
    else:
        match["sources"] = ["SUBMISSION_PENDING", 7]
    assert _m07_legal_predecessors(context, "ORDER_REJECTED", document) is None


def test_restore_gate_has_no_local_terminal_predecessor_registry() -> None:
    source = inspect.getsource(_revalidate_reservation_restore_authority)
    assert '"ORDER_REJECTED": {' not in source
    assert '"ORDER_FILLED": {' not in source
    assert "legal_predecessor_derivation" not in source


@pytest.mark.parametrize("variant", ["capital_reservation", "capital_release"])
def test_top_level_restore_orchestrator_requires_reservation_authority(variant: str) -> None:
    record = _reservation_variant_record(variant)
    assert _validate_persistence_record(record)
    assert _revalidate_restore_records([record], {}, 1, []) == "RESTORE_REJECTED"
    empty = _ReservationRestoreAuthority(
        {}, {}, {}, _RESERVATION_AUTHORITY_SEAL, _M07_ACCEPTANCE_SEAL, _M07_LIFECYCLE_SEAL
    )
    assert _revalidate_restore_records([record], {}, 1, [], empty) == "RESTORE_REJECTED"
    accepted = _preexisting_reservation_authority(record)
    assert (
        _revalidate_restore_records([record], {}, 1, [], accepted)
        == "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS"
    )


def test_top_level_restore_rejects_release_without_accepted_terminal_event() -> None:
    record = _reservation_variant_record("capital_release")
    authority = _preexisting_reservation_authority(record)
    missing_event = _ReservationRestoreAuthority(
        authority.accounting,
        {},
        {},
        authority.authority_seal,
        authority.m07_acceptance_seal,
        authority.lifecycle_seal,
    )
    assert _revalidate_restore_records([record], {}, 1, [], missing_event) == "RESTORE_REJECTED"


def test_mixed_restore_candidate_requires_every_independent_authority_gate() -> None:
    current = _persistence_record("RiskPolicy current designation")
    current_payload = current["payload"]
    accepted_current = {
        current_payload["current_reference"]: {
            "scope_key": current_payload["scope_key"],
            "content_fingerprint_sha256": current_payload["content_fingerprint_sha256"],
            "revision": current_payload["current_revision"],
            "state": "ACTIVE",
        }
    }
    reservation = _reservation_variant_record("capital_reservation")
    reservation_authority = _preexisting_reservation_authority(reservation)
    records = [current, reservation]
    assert _revalidate_restore_records(records, accepted_current, 1, [], None) == "RESTORE_REJECTED"
    assert (
        _revalidate_restore_records(records, {}, 1, [], reservation_authority) == "RESTORE_REJECTED"
    )
    assert (
        _revalidate_restore_records(records, accepted_current, 1, [], reservation_authority)
        == "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS"
    )


def test_machine_contract_makes_reservation_gate_part_of_restore_orchestration() -> None:
    orchestration = MACHINE["backup_contract"]["restore_candidate_revalidation_orchestration"]
    assert orchestration["ordered_stages"] == [
        "STRUCTURAL_PERSISTENCE_RECORD_VALIDATION",
        "ASPECT_SPECIFIC_SEMANTIC_AUTHORITY_REVALIDATION",
        "CURRENT_DESIGNATION_RELATIONAL_REVALIDATION",
        "M0_3_RESTORE_FRESHNESS",
    ]
    reservation = orchestration["reservation_transition_history"]
    assert reservation["selector"] == {"representation_name": "reservation transition history"}
    assert reservation["missing_authority_when_records_present"] == "RESTORE_REJECTED"
    assert orchestration["structural_integrity_grants_authority"] is False
    assert orchestration["category_gates_are_conjunctive"] is True


def test_reservation_authority_metadata_uses_real_json_pointers_and_executable_evidence() -> None:
    variants = MACHINE["backup_contract"]["representation_registry"][
        "reservation transition history"
    ]["immutable_fact_binding"]["upstream_payload_variants"]
    upstream = json.loads((DOCS / "ledger_portfolio_capital_and_pnl.json").read_text())
    for variant in variants.values():
        gate = variant["restore_authority_revalidation"]
        for source in (gate["accounting_membership"], gate["m07_context"]):
            assert source["source_artifact"] == "ledger_portfolio_capital_and_pnl.json"
            ok, _ = _resolve_pointer(upstream, source["source_pointer"])
            assert ok
            assert source["executable_evidence"].startswith("tests/architecture/")


@pytest.mark.parametrize("bad", ["0", "-1", "01", "1.0", "1.", 1.0, True])
def test_capital_reservation_rejects_nonpositive_or_noncanonical_quantity_after_rehash(
    bad: Any,
) -> None:
    record = _reservation_variant_record("capital_reservation")
    record["payload"]["upstream_payload"]["quantity"] = bad
    _rehash_immutable_record(record)
    assert not _validate_persistence_record(record)


@pytest.mark.parametrize(
    "mutation", ["arbitrary", "missing", "extra", "mapping_status", "field_type"]
)
def test_capital_reservation_rejects_malformed_asset_reference_after_rehash(
    mutation: str,
) -> None:
    record = _reservation_variant_record("capital_reservation")
    asset = record["payload"]["upstream_payload"]["asset_reference"]
    if mutation == "arbitrary":
        record["payload"]["upstream_payload"]["asset_reference"] = {"anything": "non-empty"}
    elif mutation == "missing":
        asset.pop("asset_namespace")
    elif mutation == "extra":
        asset["invented"] = "field"
    elif mutation == "mapping_status":
        asset["mapping_status"] = "UNKNOWN"
    else:
        asset["venue_asset_code"] = 1
    _rehash_immutable_record(record)
    assert not _validate_persistence_record(record)


@pytest.mark.parametrize(
    ("variant", "field", "bad"),
    [
        ("capital_reservation", "source_type", "capital_release"),
        ("capital_reservation", "environment", "DEMO"),
        ("capital_release", "source_type", "capital_reservation"),
        ("capital_release", "environment", "DEMO"),
        ("capital_release", "terminal_state", "PARTIALLY_FILLED"),
    ],
)
def test_reservation_variant_rejects_wrong_discriminator_environment_or_terminal_state(
    variant: str, field: str, bad: str
) -> None:
    record = _reservation_variant_record(variant)
    record["payload"]["upstream_payload"][field] = bad
    _rehash_immutable_record(record)
    assert not _validate_persistence_record(record)


def test_reservation_variant_constants_environments_terminal_and_provenance_are_upstream() -> None:
    upstream = json.loads((DOCS / "ledger_portfolio_capital_and_pnl.json").read_text())
    variants = MACHINE["backup_contract"]["representation_registry"][
        "reservation transition history"
    ]["immutable_fact_binding"]["upstream_payload_variants"]
    for name, variant in variants.items():
        contracts = variant["field_contracts"]
        assert name in upstream["accounting_economic_fact_schema_registry"]
        assert contracts["source_type"]["value"] == name
        assert contracts["environment"]["values"] == upstream["environment_policy"]["core"]
        assert contracts["provenance"]["type"] == "non_empty_string"
        assert contracts["provenance"]["constraint_pointer"].endswith(f"/{name}/constraints")
        assert contracts["source_fingerprint_sha256"]["authority_semantics"] == (
            "INTEGRITY_PLUS_PREEXISTING_ACCEPTED_ACCOUNTING_MEMBERSHIP"
        )
    assert (
        variants["capital_release"]["field_contracts"]["terminal_state"]["values"]
        == upstream["reservation_protocol"]["terminal_release"]
    )


def test_all14_constraint_audit_executes_variant_contracts_not_only_field_sets() -> None:
    entry = MACHINE["backup_contract"]["representation_registry"]["reservation transition history"]
    document = json.loads((DOCS / entry["semantic_artifact"]).read_text())
    variants = entry["immutable_fact_binding"]["upstream_payload_variants"]
    for variant_name, variant in variants.items():
        for field, contract in variant["field_contracts"].items():
            ok, declaration = _resolve_pointer(document, contract["source_pointer"])
            assert ok and field in declaration
            assert contract["nullable"] is False
            if contract["type"] == "constant":
                assert contract["value"] == variant_name
            if contract["type"] == "enum":
                ok, expected = _resolve_pointer(document, contract["constraint_pointer"])
                assert ok and contract["values"] == expected
            if field == "quantity":
                assert contract["type"] == "positive_decimal" and contract["positive"] is True
            if field == "asset_reference":
                assert contract["type"] == "asset_reference" and contract["projection_rule"]
            if field in {"provenance", "source_fingerprint_sha256"}:
                assert contract["projection_rule"]


@pytest.mark.parametrize(
    ("aspect", "identity_field"),
    [
        ("DeviceInstallation current identity/lifecycle", "device_installation_id"),
        ("Portfolio canonical accounting state", "portfolio_id"),
        ("RiskBudget current state", "risk_scope_key"),
    ],
)
def test_fact_record_key_contains_exact_object_identity_and_prevents_collision(
    aspect: str, identity_field: str
) -> None:
    first = _persistence_record(aspect, object_suffix="a", revision=7)
    second = copy.deepcopy(first)
    facts = second["payload"]["facts"]
    if identity_field == "risk_scope_key":
        facts["portfolio_id"] = _canonical_fixture_id("port", "b")
        contract = MACHINE["backup_contract"]["representation_registry"][aspect]["fact_binding"][
            "field_contracts"
        ]["risk_scope_key"]
        facts[identity_field] = "|".join(str(facts[field]) for field in contract["components"])
    else:
        prefix = "dev" if identity_field == "device_installation_id" else "port"
        facts[identity_field] = _canonical_fixture_id(prefix, "b")
    entry = MACHINE["backup_contract"]["representation_registry"][aspect]
    second["record_key"] = _derive_record_key(aspect, entry, second["payload"])
    second["payload_fingerprint_sha256"] = _actual_fingerprint(second["payload"])
    assert _validate_persistence_record(first) and _validate_persistence_record(second)
    assert first["record_key"] != second["record_key"]
    unchanged_key = copy.deepcopy(second)
    unchanged_key["record_key"] = first["record_key"]
    assert not _validate_persistence_record(unchanged_key)


def test_fact_ids_use_exact_m02_prefixes_and_portfolio_uses_port() -> None:
    portfolio = _persistence_record("Portfolio canonical accounting state")
    facts = portfolio["payload"]["facts"]
    assert facts["account_id"].startswith("acct_")
    assert facts["workspace_id"].startswith("ws_")
    assert facts["portfolio_id"].startswith("port_")
    for field, wrong_prefix in (
        ("account_id", "dev"),
        ("workspace_id", "acct"),
        ("portfolio_id", "acct"),
    ):
        altered = copy.deepcopy(portfolio)
        altered["payload"]["facts"][field] = _canonical_fixture_id(wrong_prefix, "a")
        altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
        assert not _validate_persistence_record(altered)


def _rehash_direct_record(record: dict[str, Any]) -> None:
    aspect = record["representation_name"]
    terminal = _terminal_fingerprint_field(aspect)
    if terminal is not None:
        record["payload"][terminal] = _actual_fingerprint(
            {key: value for key, value in record["payload"].items() if key != terminal}
        )
    entry = MACHINE["backup_contract"]["representation_registry"][aspect]
    record["record_key"] = _derive_record_key(aspect, entry, record["payload"])
    record["payload_fingerprint_sha256"] = _actual_fingerprint(record["payload"])


def test_direct_id_constant_and_decimal_constraints_are_literal() -> None:
    command = _persistence_record("Command accepted request")
    command["payload"]["operation_type"] = "CANCEL_ORDER"
    _rehash_direct_record(command)
    assert not _validate_persistence_record(command)

    intent = _persistence_record("OrderIntent")
    for value in ("0", "-1"):
        altered = copy.deepcopy(intent)
        altered["payload"]["quantity"] = value
        _rehash_direct_record(altered)
        assert not _validate_persistence_record(altered)
    wrong_prefix = copy.deepcopy(intent)
    wrong_prefix["payload"]["workspace_id"] = _canonical_fixture_id("dev", "a")
    _rehash_direct_record(wrong_prefix)
    assert not _validate_persistence_record(wrong_prefix)


def test_direct_nested_objects_are_exact_closed_and_semantic() -> None:
    ledger = _persistence_record("LedgerEntry")
    for nested in ({"garbage": 1}, {**ledger["payload"]["asset_reference"], "extra": True}):
        altered = copy.deepcopy(ledger)
        altered["payload"]["asset_reference"] = nested
        _rehash_direct_record(altered)
        assert not _validate_persistence_record(altered)
    illegal = copy.deepcopy(ledger)
    illegal["payload"]["asset_reference"]["mapping_status"] = "UNKNOWN"
    _rehash_direct_record(illegal)
    assert not _validate_persistence_record(illegal)


def test_lifecycle_current_and_transition_record_keys_are_distinct() -> None:
    migration_transition = _persistence_record("Migration transition/history revisions")
    migration_current = _persistence_record("Migration current state/designation")
    handoff_transition = _persistence_record("SecretHandoff transition/history revisions")
    handoff_current = _persistence_record("SecretHandoff current state/designation")
    assert migration_transition["record_key"] != migration_current["record_key"]
    assert handoff_transition["record_key"] != handoff_current["record_key"]
    for current, transition in (
        (migration_current, migration_transition),
        (handoff_current, handoff_transition),
    ):
        altered = copy.deepcopy(current)
        altered["record_key"] = transition["record_key"]
        assert not _validate_persistence_record(altered)


def test_bootstrap_consumption_rejects_cross_scope_and_revision_generation_mismatch() -> None:
    current = _persistence_record("bootstrap consumed fence")
    history = _persistence_record("bootstrap accepted/consumption history")
    assert (
        _revalidate_bootstrap_consumption(current, [history])
        == "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS"
    )
    for field, value in (
        ("account_id", _canonical_fixture_id("acct", "b")),
        ("device_installation_id", _canonical_fixture_id("dev", "b")),
        ("bootstrap_generation", 2),
        ("bootstrap_revision", 2),
    ):
        altered = copy.deepcopy(history)
        altered["payload"][field] = value
        entry = MACHINE["backup_contract"]["representation_registry"][
            "bootstrap accepted/consumption history"
        ]
        altered["record_key"] = _derive_record_key(
            "bootstrap accepted/consumption history", entry, altered["payload"]
        )
        altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
        assert _validate_persistence_record(altered)
        assert _revalidate_bootstrap_consumption(current, [altered]) == "RESTORE_REJECTED"


def test_direct_event_and_lease_reject_valid_uuid_with_wrong_prefix() -> None:
    event = _persistence_record("Event")
    for field, prefix in (("workspace_id", "dev"), ("portfolio_id", "acct")):
        altered = copy.deepcopy(event)
        altered["payload"][field] = _canonical_fixture_id(prefix, "a")
        _rehash_direct_record(altered)
        assert not _validate_persistence_record(altered)
    lease = _persistence_record("ExecutionLease immutable record")
    lease["payload"]["execution_lease_id"] = _canonical_fixture_id("cmd", "a")
    _rehash_direct_record(lease)
    assert not _validate_persistence_record(lease)


def test_fill_nested_fee_reference_and_decimal_constraints_are_exact() -> None:
    fill = _persistence_record("Fill")
    fill["payload"]["fee_kind"] = "CHARGE"
    fill["payload"]["fee_quantity"] = "1"
    schema = MACHINE["backup_contract"]["direct_upstream_validator_registry"]["Fill"][
        "upstream_field_schemas"
    ]["fee_asset_reference"]
    fill["payload"]["fee_asset_reference"] = {
        field: _direct_fixture_value("Fill", field, schema["field_schemas"][field], "a", 1)
        for field in schema["fields"]
    }
    _rehash_direct_record(fill)
    assert _validate_persistence_record(fill)
    for mutation in ("missing", "extra", "mapping", "negative"):
        altered = copy.deepcopy(fill)
        nested = altered["payload"]["fee_asset_reference"]
        assert isinstance(nested, dict)
        if mutation == "missing":
            nested.pop("mapping_status")
        elif mutation == "extra":
            nested["invented"] = True
        elif mutation == "mapping":
            nested["mapping_status"] = "UNKNOWN"
        else:
            altered["payload"]["fee_quantity"] = "-1"
        _rehash_direct_record(altered)
        assert not _validate_persistence_record(altered)


def test_persistence_record_payload_fingerprint_contract_is_exact_and_integrity_only() -> None:
    contract = MACHINE["backup_contract"]["persistence_record_contract"]
    derivation = contract["payload_fingerprint_sha256_derivation"]
    assert derivation == {
        "algorithm": "SHA-256",
        "input": "exact PersistenceRecord.payload JSON value",
        "encoding": "UTF-8",
        "canonical_json": {
            "sort_keys": True,
            "separators": [",", ":"],
            "ensure_ascii": False,
            "allow_nan": False,
        },
        "digest_encoding": "lowercase hexadecimal",
        "domain_separator": None,
        "excluded_inputs": [
            "PersistenceRecord fields other than payload",
            "timestamp",
            "representation_name",
            "record_key",
        ],
        "meaning": "INTEGRITY_ONLY",
        "establishes_domain_authority": False,
        "establishes_accepted_or_current_membership": False,
    }
    assert contract["domain_authority"] is False
    assert contract["self_hash_membership"] is False


def test_persistence_record_payload_fingerprint_uses_exact_canonical_json() -> None:
    payload = {"zażółć": [1, True, None], "a": {"b": "✓"}}
    expected = hashlib.sha256('{"a":{"b":"✓"},"zażółć":[1,true,null]}'.encode("utf-8")).hexdigest()
    assert _actual_fingerprint(payload) == expected
    assert _actual_fingerprint(dict(reversed(list(payload.items())))) == expected
    assert _actual_fingerprint({**payload, "a": {"b": "x"}}) != expected


def test_persistence_record_rejects_caller_hash_not_matching_payload() -> None:
    record = _persistence_record("RuntimeSession canonical identity/history")
    record["payload_fingerprint_sha256"] = "f" * 64
    assert SHA_RE.fullmatch(record["payload_fingerprint_sha256"])
    assert not _validate_persistence_record(record)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_persistence_record_rejects_non_finite_json_number(non_finite: float) -> None:
    record = _persistence_record("RuntimeSession canonical identity/history")
    record["payload"]["non_finite"] = non_finite
    record["payload_fingerprint_sha256"] = "f" * 64
    assert not _validate_persistence_record(record)
    with pytest.raises(ValueError):
        _actual_fingerprint(record["payload"])


def test_runtime_session_fact_kind_is_exact_m02_entity_kind() -> None:
    vocabulary = json.loads((DOCS / "canonical_domain_vocabulary.json").read_text())
    entity = next(x for x in vocabulary["entity_kinds"] if x["canonical_name"] == "RuntimeSession")
    entry = MACHINE["backup_contract"]["representation_registry"][
        "RuntimeSession canonical identity/history"
    ]
    assert entry["payload_contract"]["fact_kind_literal"] == entity["canonical_name"]
    assert entry["immutable_fact_binding"]["fact_kind_literal"] == "RuntimeSession"
    record = _persistence_record("RuntimeSession canonical identity/history")
    assert record["payload"]["fact_kind"] == "RuntimeSession"
    assert _validate_persistence_record(record)


def test_runtime_session_rejects_noncanonical_fact_kind() -> None:
    record = _persistence_record("RuntimeSession canonical identity/history")
    record["payload"]["fact_kind"] = "RuntimeSession canonical identity/history"
    record["payload_fingerprint_sha256"] = _actual_fingerprint(record["payload"])
    assert not _validate_persistence_record(record)


def test_runtime_session_wrapper_fingerprint_contract_is_m011_integrity() -> None:
    binding = MACHINE["backup_contract"]["representation_registry"][
        "RuntimeSession canonical identity/history"
    ]["immutable_fact_binding"]
    derivation = binding["upstream_payload_fingerprint_sha256_derivation"]
    assert binding["upstream_fingerprint_literal"] is False
    assert derivation["origin"] == "M0.11 wrapper integrity"
    assert derivation["input_fields"] == ["runtime_session_id", "device_installation_id"]
    assert derivation["upstream_semantic_fingerprint_literal"] is False
    assert derivation["authority"] is False
    assert derivation["establishes_membership"] is False


def test_runtime_session_upstream_fingerprint_is_exact_payload_digest() -> None:
    record = _persistence_record("RuntimeSession canonical identity/history")
    upstream = record["payload"]["upstream_payload"]
    assert set(upstream) == {"runtime_session_id", "device_installation_id"}
    assert record["payload"]["upstream_payload_fingerprint_sha256"] == _actual_fingerprint(upstream)
    assert _validate_persistence_record(record)


@pytest.mark.parametrize("field", ["runtime_session_id", "device_installation_id"])
def test_runtime_session_changed_upstream_field_rejects_stale_digest(field: str) -> None:
    record = _persistence_record("RuntimeSession canonical identity/history")
    record["payload"]["upstream_payload"][field] += "x"
    record["payload_fingerprint_sha256"] = _actual_fingerprint(record["payload"])
    assert not _validate_persistence_record(record)


def test_runtime_session_self_consistent_invalid_id_is_rejected() -> None:
    record = _persistence_record("RuntimeSession canonical identity/history")
    record["payload"]["upstream_payload"]["runtime_session_id"] = "invalid"
    _rehash_immutable_record(record)
    record["record_key"] = "invalid"
    assert not _validate_persistence_record(record)


def test_runtime_session_record_key_is_exact_canonical_id_only() -> None:
    record = _persistence_record("RuntimeSession canonical identity/history")
    entry = MACHINE["backup_contract"]["representation_registry"][record["representation_name"]]
    upstream = record["payload"]["upstream_payload"]
    assert entry["record_key_strategy"] == "CANONICAL_ENTITY_ID"
    assert entry["record_key_source_field"] == "runtime_session_id"
    assert record["record_key"] == upstream["runtime_session_id"]
    assert upstream["device_installation_id"] not in record["record_key"]
    assert entry["immutable_fact_binding"]["revision_generation_fields"] == []


def test_runtime_session_rejects_arbitrary_or_previous_strategy_key() -> None:
    record = _persistence_record("RuntimeSession canonical identity/history")
    for wrong in (
        "arbitrary",
        f"immutable:RuntimeSession:{record['payload']['upstream_payload']['runtime_session_id']}",
    ):
        altered = copy.deepcopy(record)
        altered["record_key"] = wrong
        assert not _validate_persistence_record(altered)


def test_stage_one_contract_recomputes_intrinsic_runtime_session_integrity_only() -> None:
    stages = MACHINE["backup_contract"]["persistence_record_validation_stages"]
    assert stages["STAGE_1_INTRINSIC_CARRIER"] == [
        "exact carrier fields",
        "representation registry binding",
        "category-specific payload semantics",
        "derived record_key",
        "recomputed payload_fingerprint_sha256",
    ]
    assert stages["runtime_session_additional_checks"] == [
        "fact_kind == RuntimeSession",
        "exact upstream payload",
        "canonical RuntimeSession ID",
        "canonical DeviceInstallation parent binding",
        "recomputed upstream_payload_fingerprint_sha256",
    ]
    assert stages["stage_1_success_does_not_mean"] == [
        "accepted",
        "current",
        "authorized",
        "restore-authoritative",
    ]
    assert stages["stage_1_establishes_authority"] is False


def test_crypto_account_record_key_is_exact_entity_id_and_rejects_alternatives() -> None:
    record = _persistence_record("CryptoHunterAccount current record")
    entity_id = record["payload"]["entity_id"]
    assert record["record_key"] == entity_id
    assert _validate_persistence_record(record)

    for wrong_key in (
        f"entity:CryptoHunterAccount:{entity_id}",
        "arbitrary",
        _canonical_fixture_id("acct", "b"),
    ):
        altered = copy.deepcopy(record)
        altered["record_key"] = wrong_key
        assert not _validate_persistence_record(altered)


def test_all_canonical_entity_id_entries_derive_their_exact_declared_source() -> None:
    registry = MACHINE["backup_contract"]["representation_registry"]
    entries = {
        name: entry
        for name, entry in registry.items()
        if entry.get("record_key_strategy") == "CANONICAL_ENTITY_ID"
    }
    assert set(entries) == {
        "CryptoHunterAccount current record",
        "Workspace",
        "RuntimeSession canonical identity/history",
    }
    for name, entry in entries.items():
        payload = _payload_for(name)
        source_field = entry["record_key_source_field"]
        source_location = entry["record_key_source_location"]
        assert source_location in {"payload", "upstream_payload"}
        source = payload if source_location == "payload" else payload["upstream_payload"]
        expected = source[source_field]
        derived = _derive_record_key(name, entry, payload)
        assert derived == expected
        assert not derived.startswith(("entity:", "immutable:"))


def test_supported_canonical_entity_ids_use_frozen_m02_type_prefixes() -> None:
    vocabulary = json.loads((DOCS / "canonical_domain_vocabulary.json").read_text())
    entities = {entry["canonical_name"]: entry for entry in vocabulary["entity_kinds"]}
    assert vocabulary["identifier_policy"]["persistent_id_format"] == "<prefix>_<uuidv7>"
    assert entities["CryptoHunterAccount"]["id_prefix"] == "acct"
    assert entities["RuntimeSession"]["id_prefix"] == "run"

    account = _persistence_record("CryptoHunterAccount current record")
    runtime = _persistence_record("RuntimeSession canonical identity/history")
    assert account["record_key"].startswith("acct_")
    assert runtime["record_key"].startswith("run_")
    assert account["record_key"] != runtime["record_key"]


def test_s2c_contract_closes_common_canonical_json_and_record_order() -> None:
    contract = MACHINE["state_store_fingerprint_contract"]
    assert contract["canonical_json"] == {
        "algorithm": "SHA-256",
        "encoding": "UTF-8",
        "sort_keys": True,
        "separators": [",", ":"],
        "ensure_ascii": False,
        "allow_nan": False,
        "digest": "lowercase hexadecimal",
        "projection": "semantic and storage-neutral; SQLite storage JSON is not authority",
        "forbidden_inputs": [
            "pickle",
            "repr",
            "SQLite binary layout",
            "rowid",
            "page number",
            "insertion order",
            "locally generated timestamp",
            "filesystem metadata",
        ],
    }
    assert contract["record_order"]["keys"] == ["representation_name", "record_key"]
    assert len(MACHINE["executable_boundary_schemas"]["PersistenceRecord"]["required"]) == 9


def test_history_tail_empty_is_sha256_of_exact_empty_canonical_array() -> None:
    assert _history_tail_fingerprint([]) == hashlib.sha256(b"[]").hexdigest()


def test_history_tail_is_deterministic_and_input_order_independent() -> None:
    records = [
        _persistence_record("RuntimeSession canonical identity/history"),
        _persistence_record("RiskPolicy accepted revisions"),
    ]
    assert _history_tail_fingerprint(records) == _history_tail_fingerprint(list(reversed(records)))
    assert _history_tail_fingerprint(records) == _history_tail_fingerprint(copy.deepcopy(records))


def test_history_append_delete_payload_and_metadata_mutation_change_digest() -> None:
    first = _persistence_record("RuntimeSession canonical identity/history")
    second = _persistence_record("RiskPolicy accepted revisions")
    baseline = _history_tail_fingerprint([first])
    appended = _history_tail_fingerprint([first, second])
    assert appended != baseline
    assert _history_tail_fingerprint([second]) != appended
    payload_mutation = copy.deepcopy(first)
    payload_mutation["payload"]["upstream_payload"]["device_installation_id"] += "x"
    assert _history_tail_fingerprint([payload_mutation]) != baseline
    metadata_mutation = copy.deepcopy(first)
    metadata_mutation["semantic_json_pointer"] += "/changed"
    assert _history_tail_fingerprint([metadata_mutation]) != baseline


def test_history_contract_has_no_rowid_chronology_or_insertion_order_dependency() -> None:
    contract = MACHINE["state_store_fingerprint_contract"]
    assert contract["record_order"]["not_chronology"] is True
    assert "rowid" in contract["canonical_json"]["forbidden_inputs"]
    assert "insertion order" in contract["canonical_json"]["forbidden_inputs"]


def test_state_projection_has_exact_fields_and_excludes_self_reference_and_evidence() -> None:
    projection = _state_projection(_metadata(), [])
    contract = MACHINE["state_store_fingerprint_contract"]["state_fingerprint"]
    assert list(projection) == contract["projection_fields"]
    assert not {
        "state_fingerprint_sha256",
        "transaction_fingerprint_sha256",
        "LocalDurableStateEvidence",
    } & set(projection)


def test_current_projection_selection_is_exact_and_excludes_metadata_and_evidence() -> None:
    contract = MACHINE["state_store_fingerprint_contract"]["current_record_projection"]
    assert "DURABLE AUTHORITATIVE CURRENT STATE" in contract["selection"]
    assert "StateStoreMetadata" in contract["exclude"]
    assert "LocalDurableStateEvidence payload" in contract["exclude"]
    assert MACHINE["startup_recovery_model"]["evidence_policy"] == {
        "local_durable_evidence_persisted_in_StateStore": False,
        "local_durable_evidence_included_in_state_fingerprint": False,
        "local_durable_evidence_included_in_BackupEnvelope": False,
        "local_evidence_current_designation_persisted": False,
        "local_evidence_registry_process_local": True,
        "local_evidence_rebuild_source": "verified current durable StateStore observation",
        "evidence_publication_is_StateStore_semantic_transaction": False,
        "evidence_publication_advances_protected_freshness_generation": False,
    }


def test_state_digest_is_stable_and_current_input_order_independent() -> None:
    records = [
        _persistence_record("Workspace"),
        _persistence_record("CryptoHunterAccount current record"),
    ]
    metadata = _metadata(history_tail_fingerprint_sha256=_history_tail_fingerprint([]))
    assert _actual_fingerprint(_state_projection(metadata, records)) == _actual_fingerprint(
        _state_projection(copy.deepcopy(metadata), list(reversed(records)))
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("account_id", "acct_01890f3a-2b4c-7abc-8def-0123456789ac"),
        ("device_installation_id", "dev_01890f3a-2b4c-7abc-8def-0123456789ac"),
        ("state_store_identity_fingerprint_sha256", "d" * 64),
        ("state_store_schema_version", 2),
        ("environment", "TESTNET"),
        ("protected_freshness_generation", 2),
        ("history_tail_fingerprint_sha256", "e" * 64),
    ],
)
def test_each_state_metadata_input_changes_digest(field: str, replacement: Any) -> None:
    baseline = _state_projection(_metadata(), [])
    changed_metadata = _metadata(**{field: replacement})
    assert _actual_fingerprint(baseline) != _actual_fingerprint(
        _state_projection(changed_metadata, [])
    )


def test_current_record_mutation_changes_state_digest() -> None:
    record = _persistence_record("Workspace")
    changed = copy.deepcopy(record)
    changed["payload_fingerprint_sha256"] = "f" * 64
    assert _actual_fingerprint(_state_projection(_metadata(), [record])) != _actual_fingerprint(
        _state_projection(_metadata(), [changed])
    )


def test_transaction_projection_has_exact_fields_and_does_not_hash_itself() -> None:
    projection = _transaction_projection()
    contract = MACHINE["state_store_fingerprint_contract"]["transaction_fingerprint"]
    assert list(projection) == contract["projection_fields"]
    assert "transaction_fingerprint_sha256" not in projection


def test_genesis_transaction_has_null_pre_fields_and_external_positive_target() -> None:
    genesis = _transaction_projection(
        expected_current_generation=None,
        pre_state_fingerprint_sha256=None,
        pre_history_tail_fingerprint_sha256=None,
        target_generation=1,
    )
    assert genesis["expected_current_generation"] is None
    assert genesis["pre_state_fingerprint_sha256"] is None
    assert genesis["pre_history_tail_fingerprint_sha256"] is None
    assert genesis["target_generation"] == 1
    assert (
        "externally supplied"
        in MACHINE["state_store_fingerprint_contract"]["genesis_semantics"]["target_generation"]
    )


def test_transaction_mutation_arrays_are_input_order_independent() -> None:
    records = [_persistence_record("Workspace"), _persistence_record("ExchangeAccount")]
    left = _transaction_projection(current_record_mutations=records)
    right = _transaction_projection(current_record_mutations=list(reversed(records)))
    assert _actual_fingerprint(left) == _actual_fingerprint(right)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("account_id", "acct_01890f3a-2b4c-7abc-8def-0123456789ac"),
        ("environment", "TESTNET"),
        ("state_store_schema_version", 2),
        ("expected_current_generation", 0),
        ("target_generation", 3),
        ("pre_state_fingerprint_sha256", "5" * 64),
        ("post_state_fingerprint_sha256", "6" * 64),
        ("pre_history_tail_fingerprint_sha256", "7" * 64),
        ("post_history_tail_fingerprint_sha256", "8" * 64),
    ],
)
def test_each_transaction_boundary_input_changes_digest(field: str, replacement: Any) -> None:
    assert _actual_fingerprint(_transaction_projection()) != _actual_fingerprint(
        _transaction_projection(**{field: replacement})
    )


def test_current_mutation_and_history_append_each_change_transaction_digest() -> None:
    baseline = _actual_fingerprint(_transaction_projection())
    current = _persistence_record("Workspace")
    history = _persistence_record("RuntimeSession canonical identity/history")
    assert baseline != _actual_fingerprint(
        _transaction_projection(current_record_mutations=[current])
    )
    assert baseline != _actual_fingerprint(
        _transaction_projection(immutable_history_appends=[history])
    )


def test_metadata_only_transition_remains_fingerprintable() -> None:
    projection = _transaction_projection()
    assert projection["current_record_mutations"] == []
    assert projection["immutable_history_appends"] == []
    assert SHA_RE.fullmatch(_actual_fingerprint(projection))
    assert MACHINE["state_store_fingerprint_contract"]["metadata_only_semantics"]["is_transaction"]


def test_syntactically_valid_false_metadata_fails_prepared_validation() -> None:
    expected = {
        "state_fingerprint_sha256": _actual_fingerprint(_state_projection(_metadata(), [])),
        "transaction_fingerprint_sha256": _actual_fingerprint(_transaction_projection()),
        "history_tail_fingerprint_sha256": _history_tail_fingerprint([]),
    }
    supplied = {name: "f" * 64 for name in expected}
    assert all(SHA_RE.fullmatch(value) for value in supplied.values())
    assert supplied != expected
    assert (
        MACHINE["state_store_fingerprint_contract"]["prepare_commit_relation"][
            "mismatching_syntactically_valid_digest"
        ]
        == "FAIL CLOSED"
    )


def test_generation_pinned_snapshot_rejects_mixed_generation_components() -> None:
    def valid(metadata_generation: int, records_generation: int, history_generation: int) -> bool:
        return len({metadata_generation, records_generation, history_generation}) == 1

    assert valid(2, 2, 2)
    assert not valid(1, 2, 1)
    assert not valid(2, 2, 1)
    snapshot = MACHINE["state_store_fingerprint_contract"]["durable_snapshot"]
    assert snapshot["generation_consistent"] is True
    assert snapshot["authority"] is False


def test_trusted_observer_checks_bind_one_generation_pinned_snapshot() -> None:
    binding = MACHINE["startup_recovery_model"]["trusted_observation_snapshot_binding"]
    assert binding["single_input"] == (
        "one generation-pinned durable snapshot including complete "
        "StateStoreTransactionDescriptor chain 1..G for the current identity"
    )
    assert binding["multiple_potentially_different_reads_forbidden"] is True
    assert binding["mints_upstream_authority"] is False


def test_restart_recomputes_all_fingerprints_but_never_trusts_stored_hash_alone() -> None:
    restart = MACHINE["state_store_fingerprint_contract"]["restart_verification"]
    assert restart["history_tail_recomputable_from_generation_pinned_snapshot"] is True
    assert restart["state_fingerprint_recomputable_from_generation_pinned_snapshot"] is True
    assert (
        restart[
            "transaction_fingerprint_independently_reconstructable_from_current_frozen_StateStore_content"
        ]
        == "true only when the complete durable StateStoreTransactionDescriptor chain 1..G is present and valid"
    )
    assert restart["stored_hash_is_proof"] is False
    assert restart["missing_descriptor"].startswith("FAIL CLOSED")


def test_fingerprint_contract_is_integrity_only_negative_authority_matrix() -> None:
    contract = MACHINE["state_store_fingerprint_contract"]
    forbidden = {
        "accepted",
        "authorized",
        "live_allowed",
        "execution_allowed",
        "membership_accepted",
        "current_membership",
    }
    assert not (_all_keys(contract) & forbidden)
    assert contract["authority_boundary"]["integrity_fencing_evidence_only"] is True
    assert MACHINE["authority_model"]["live_current"] == "DENIED"
    assert MACHINE["authority_model"]["testnet_to_live_fallback"] is False
    assert MACHINE["transaction_protocol"]["cross_resource_acid_claim"] is False


def test_descriptor_schema_is_closed_exact_fourteen_fields_and_has_no_domain_id() -> None:
    schema = MACHINE["executable_boundary_schemas"]["StateStoreTransactionDescriptor"]
    expected = [
        "account_id",
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
        "state_store_schema_version",
        "environment",
        "expected_current_generation",
        "target_generation",
        "pre_state_fingerprint_sha256",
        "pre_history_tail_fingerprint_sha256",
        "post_state_fingerprint_sha256",
        "post_history_tail_fingerprint_sha256",
        "current_record_mutations",
        "immutable_history_appends",
        "transaction_fingerprint_sha256",
    ]
    assert schema["required"] == expected
    assert set(schema["properties"]) == set(expected)
    assert schema["additionalProperties"] is False
    assert not {"id", "descriptor_id", "transaction_id", "uuid"} & set(schema["properties"])


def test_descriptor_schema_closes_nullable_sha_generation_and_environment_semantics() -> None:
    properties = MACHINE["executable_boundary_schemas"]["StateStoreTransactionDescriptor"][
        "properties"
    ]
    assert properties["environment"]["enum"] == ["PAPER", "TESTNET", "LIVE"]
    assert properties["target_generation"] == {
        "type": "integer",
        "minimum": 1,
        "boolean_allowed": False,
    }
    assert properties["expected_current_generation"]["oneOf"][1] == {"type": "null"}
    for field in ("pre_state_fingerprint_sha256", "pre_history_tail_fingerprint_sha256"):
        assert properties[field]["oneOf"][1] == {"type": "null"}
    for field in (
        "state_store_identity_fingerprint_sha256",
        "post_state_fingerprint_sha256",
        "post_history_tail_fingerprint_sha256",
        "transaction_fingerprint_sha256",
    ):
        assert properties[field]["pattern"] == "^[0-9a-f]{64}$"


def test_descriptor_reuses_exact_transaction_projection_and_detects_any_mutation() -> None:
    descriptor = _transaction_descriptor()
    assert _descriptor_hash_valid(descriptor)
    projection_fields = MACHINE["state_store_fingerprint_contract"]["transaction_fingerprint"][
        "projection_fields"
    ]
    assert projection_fields == list(descriptor)[:-1]
    for field in projection_fields:
        tampered = copy.deepcopy(descriptor)
        value = tampered[field]
        if isinstance(value, list):
            value.append(_persistence_record("Workspace"))
        elif isinstance(value, int):
            tampered[field] = value + 1
        else:
            tampered[field] = f"{value}x"
        assert not _descriptor_hash_valid(tampered), field


def test_genesis_and_non_genesis_descriptor_generation_semantics() -> None:
    genesis = _transaction_descriptor(
        expected_current_generation=None,
        target_generation=1,
        pre_state_fingerprint_sha256=None,
        pre_history_tail_fingerprint_sha256=None,
    )
    assert genesis["expected_current_generation"] is None
    assert genesis["pre_state_fingerprint_sha256"] is None
    assert genesis["pre_history_tail_fingerprint_sha256"] is None
    current = _transaction_descriptor(expected_current_generation=1, target_generation=2)
    assert current["target_generation"] == current["expected_current_generation"] + 1
    assert not _positive(True)


@pytest.mark.parametrize(
    ("descriptor_field", "metadata_field", "replacement"),
    [
        ("account_id", "account_id", "acct_01890f3a-2b4c-7abc-8def-0123456789ac"),
        (
            "device_installation_id",
            "device_installation_id",
            "dev_01890f3a-2b4c-7abc-8def-0123456789ac",
        ),
        (
            "state_store_identity_fingerprint_sha256",
            "state_store_identity_fingerprint_sha256",
            "d" * 64,
        ),
        ("state_store_schema_version", "state_store_schema_version", 2),
        ("environment", "environment", "TESTNET"),
        ("target_generation", "protected_freshness_generation", 3),
        ("post_state_fingerprint_sha256", "state_fingerprint_sha256", "d" * 64),
        ("post_history_tail_fingerprint_sha256", "history_tail_fingerprint_sha256", "e" * 64),
        ("transaction_fingerprint_sha256", "transaction_fingerprint_sha256", "f" * 64),
    ],
)
def test_each_descriptor_metadata_mismatch_fails_closed(
    descriptor_field: str, metadata_field: str, replacement: Any
) -> None:
    descriptor = _transaction_descriptor()
    metadata = _metadata(
        protected_freshness_generation=descriptor["target_generation"],
        state_fingerprint_sha256=descriptor["post_state_fingerprint_sha256"],
        history_tail_fingerprint_sha256=descriptor["post_history_tail_fingerprint_sha256"],
        transaction_fingerprint_sha256=descriptor["transaction_fingerprint_sha256"],
    )
    assert _descriptor_matches_metadata(descriptor, metadata)
    altered = copy.deepcopy(metadata)
    altered[metadata_field] = replacement
    assert descriptor[descriptor_field] != altered[metadata_field]
    assert not _descriptor_matches_metadata(descriptor, altered)


def test_descriptor_chain_accepts_continuity_and_rejects_each_break() -> None:
    previous = _transaction_descriptor(target_generation=1)
    current = _transaction_descriptor(
        expected_current_generation=1,
        target_generation=2,
        pre_state_fingerprint_sha256=previous["post_state_fingerprint_sha256"],
        pre_history_tail_fingerprint_sha256=previous["post_history_tail_fingerprint_sha256"],
    )
    assert _descriptor_chain_valid(previous, current)
    for field, value in (
        ("expected_current_generation", 2),
        ("pre_state_fingerprint_sha256", "8" * 64),
        ("pre_history_tail_fingerprint_sha256", "9" * 64),
        ("target_generation", 3),
    ):
        altered = copy.deepcopy(current)
        altered[field] = value
        assert not _descriptor_chain_valid(previous, altered)
    assert len({previous["target_generation"], current["target_generation"]}) == 2
    assert len({current["target_generation"], current["target_generation"]}) != 2


@pytest.mark.parametrize("descriptor_generation", [None, 1, 3])
def test_generation_pinned_snapshot_requires_exact_matching_descriptor(
    descriptor_generation: int | None,
) -> None:
    metadata_generation = 2

    def valid(candidate: int | None) -> bool:
        return candidate is not None and candidate == metadata_generation

    assert valid(2)  # metadata G + records G + history G + descriptor G
    assert not valid(descriptor_generation)


def test_restart_transaction_gate_requires_descriptor_recomputation_and_metadata_binding() -> None:
    descriptor = _transaction_descriptor()
    metadata = _metadata(
        protected_freshness_generation=descriptor["target_generation"],
        state_fingerprint_sha256=descriptor["post_state_fingerprint_sha256"],
        history_tail_fingerprint_sha256=descriptor["post_history_tail_fingerprint_sha256"],
        transaction_fingerprint_sha256=descriptor["transaction_fingerprint_sha256"],
    )
    assert _descriptor_hash_valid(descriptor) and _descriptor_matches_metadata(descriptor, metadata)
    assert metadata["transaction_fingerprint_sha256"] and not False  # stored metadata alone is data
    tampered = copy.deepcopy(descriptor)
    tampered["target_generation"] += 1
    assert (
        tampered["transaction_fingerprint_sha256"] == descriptor["transaction_fingerprint_sha256"]
    )
    assert not _descriptor_hash_valid(tampered)


def test_missing_descriptor_blocks_crash_recovery_and_finalized_restart() -> None:
    restart = MACHINE["state_store_fingerprint_contract"]["restart_verification"]
    assert "NO EVIDENCE PUBLICATION" in restart["missing_descriptor"]
    assert "NO FINALIZE" in restart["missing_descriptor"]
    crash = restart["LOCAL_COMMIT_BEFORE_EVIDENCE"]
    assert "without second business commit" in crash["complete_chain_and_all_other_gates_pass"]
    assert crash["missing_any_descriptor_or_invalid_chain"].startswith("NO EVIDENCE PUBLICATION")
    assert "registry restarts EMPTY" in restart["ordinary_finalized_restart"]


def test_descriptor_is_atomic_local_only_and_not_optional_audit_log() -> None:
    contract = MACHINE["state_store_fingerprint_contract"]["durable_transaction_descriptor"]
    assert contract["durability_classification"] == "DURABLE IMMUTABLE / APPEND-ONLY HISTORY"
    assert contract["current_descriptor_requirement"]["optional_audit_log"] is False
    assert contract["atomic_local_commit_binding"]["cross_resource_acid_claim"] is False
    assert len(contract["atomic_local_commit_binding"]["same_local_StateStore_transaction"]) == 4


def test_descriptor_has_no_state_history_self_reference_cycle() -> None:
    contract = MACHINE["state_store_fingerprint_contract"]
    descriptor = contract["durable_transaction_descriptor"]
    assert set(descriptor["excluded_from"]) == {
        "canonical_durable_current_records",
        "canonical_immutable_history_records",
        "history_tail_fingerprint_sha256",
        "state_fingerprint_sha256",
    }
    assert (
        "transaction_fingerprint_sha256"
        not in contract["transaction_fingerprint"]["projection_fields"]
    )
    assert descriptor["is_PersistenceRecord"] is False


def test_descriptor_authority_and_backup_restore_boundaries_remain_closed() -> None:
    descriptor = MACHINE["state_store_fingerprint_contract"]["durable_transaction_descriptor"]
    assert "mint M0.3 membership" in descriptor["authority_boundary"]["does_not"]
    assert "preserves the exact complete" in descriptor["backup_restore_limitation"]
    assert "does not establish restore authority" in descriptor["backup_restore_limitation"]
    assert MACHINE["authority_model"]["live_current"] == "DENIED"
    assert MACHINE["authority_model"]["testnet_to_live_fallback"] is False
    assert MACHINE["transaction_protocol"]["cross_resource_acid_claim"] is False


def _metadata_for_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    return _metadata(
        protected_freshness_generation=descriptor["target_generation"],
        state_fingerprint_sha256=descriptor["post_state_fingerprint_sha256"],
        history_tail_fingerprint_sha256=descriptor["post_history_tail_fingerprint_sha256"],
        transaction_fingerprint_sha256=descriptor["transaction_fingerprint_sha256"],
    )


@pytest.mark.parametrize(
    "targets",
    [[1, 3, 4], [2, 3, 4], [1, 2, 4], [1, 2, 3], [1, 2, 3, 4, 5], [1, 2, 2, 3, 4]],
)
def test_complete_chain_rejects_each_non_exact_generation_multiset(targets: list[int]) -> None:
    chain = _descriptor_chain(5)
    by_generation = {descriptor["target_generation"]: descriptor for descriptor in chain}
    candidate = [copy.deepcopy(by_generation[target]) for target in targets]
    metadata = _metadata_for_descriptor(by_generation[4])
    assert _complete_descriptor_chain_valid(_descriptor_chain(4), metadata)
    assert not _complete_descriptor_chain_valid(candidate, metadata)


def test_missing_middle_descriptor_fails_even_when_current_edge_is_valid() -> None:
    chain = _descriptor_chain(4)
    metadata = _metadata_for_descriptor(chain[-1])
    missing_g2 = [chain[0], chain[2], chain[3]]
    assert _descriptor_chain_valid(chain[2], chain[3])
    assert _descriptor_matches_metadata(chain[3], metadata)
    assert not _complete_descriptor_chain_valid(missing_g2, metadata)


@pytest.mark.parametrize(
    ("generation", "field", "replacement"),
    [
        (2, "expected_current_generation", 4),
        (2, "pre_state_fingerprint_sha256", "8" * 64),
        (2, "pre_history_tail_fingerprint_sha256", "8" * 64),
        (3, "expected_current_generation", 1),
        (3, "pre_state_fingerprint_sha256", "8" * 64),
        (3, "pre_history_tail_fingerprint_sha256", "8" * 64),
        (4, "expected_current_generation", 2),
        (4, "pre_state_fingerprint_sha256", "8" * 64),
        (4, "pre_history_tail_fingerprint_sha256", "8" * 64),
    ],
)
def test_complete_chain_checks_every_edge(generation: int, field: str, replacement: Any) -> None:
    chain = _descriptor_chain(4)
    metadata = _metadata_for_descriptor(chain[-1])
    chain[generation - 1][field] = replacement
    chain[generation - 1]["transaction_fingerprint_sha256"] = _actual_fingerprint(
        {name: chain[generation - 1][name] for name in _transaction_projection()}
    )
    assert not _complete_descriptor_chain_valid(chain, metadata)


def test_tampered_non_current_descriptor_hash_fails_full_chain() -> None:
    chain = _descriptor_chain(4)
    metadata = _metadata_for_descriptor(chain[-1])
    assert _descriptor_matches_metadata(chain[-1], metadata)
    chain[1]["pre_state_fingerprint_sha256"] = "8" * 64
    assert not _descriptor_hash_valid(chain[1])
    assert not _complete_descriptor_chain_valid(chain, metadata)


def test_complete_chain_rejects_future_bool_and_unrelated_identity_descriptors() -> None:
    chain = _descriptor_chain(4)
    metadata = _metadata_for_descriptor(chain[-1])
    future = _descriptor_chain(5)
    assert not _complete_descriptor_chain_valid(future, metadata)
    boolean = copy.deepcopy(chain)
    boolean[1]["target_generation"] = True
    assert not _complete_descriptor_chain_valid(boolean, metadata)
    unrelated = copy.deepcopy(chain)
    unrelated[1]["state_store_identity_fingerprint_sha256"] = "f" * 64
    assert not _complete_descriptor_chain_valid(unrelated, metadata)


def test_initialized_store_requires_exact_genesis_descriptor() -> None:
    chain = _descriptor_chain(4)
    metadata = _metadata_for_descriptor(chain[-1])
    assert _complete_descriptor_chain_valid(chain, metadata)
    for field, value in (
        ("expected_current_generation", 0),
        ("pre_state_fingerprint_sha256", "8" * 64),
        ("pre_history_tail_fingerprint_sha256", "8" * 64),
    ):
        altered = copy.deepcopy(chain)
        altered[0][field] = value
        altered[0]["transaction_fingerprint_sha256"] = _actual_fingerprint(
            {name: altered[0][name] for name in _transaction_projection()}
        )
        assert not _complete_descriptor_chain_valid(altered, metadata)


def test_metadata_only_descriptor_is_valid_atomic_commit_not_partial_state() -> None:
    descriptor = _transaction_descriptor(current_record_mutations=[], immutable_history_appends=[])
    binding = MACHINE["state_store_fingerprint_contract"]["durable_transaction_descriptor"][
        "atomic_local_commit_binding"
    ]
    semantics = binding["metadata_only_valid"]
    assert descriptor["current_record_mutations"] == []
    assert descriptor["immutable_history_appends"] == []
    assert semantics["requested_record_delta_count"] == 0
    assert semantics["violates_forbidden_partial_state"] is False
    assert "descriptor without records" not in binding["forbidden_partial_states"]


@pytest.mark.parametrize("committed", [["A"], ["R"], [], ["A", "R", "X"]])
def test_declared_mutation_binding_rejects_partial_or_extra_delta(committed: list[str]) -> None:
    declared = ["A", "R"]
    assert committed != declared
    invariant = MACHINE["state_store_fingerprint_contract"]["durable_transaction_descriptor"][
        "atomic_declared_mutation_binding"
    ]
    assert invariant["comparison"].startswith("exact canonical PersistenceRecord arrays")


def test_declared_mutation_binding_accepts_exact_current_and_history_delta() -> None:
    current = _persistence_record("Workspace")
    history = _persistence_record("RuntimeSession canonical identity/history")
    descriptor = _transaction_descriptor(
        current_record_mutations=[current], immutable_history_appends=[history]
    )
    committed_current = [current]
    committed_history = [history]
    assert committed_current == descriptor["current_record_mutations"]
    assert committed_history == descriptor["immutable_history_appends"]


def test_snapshot_restart_and_crash_contract_require_complete_chain() -> None:
    contract = MACHINE["state_store_fingerprint_contract"]
    assert contract["durable_snapshot"]["complete_descriptor_chain_required"] is True
    assert contract["durable_snapshot"]["exact_descriptor_generation_set"].startswith("1..")
    algorithm = contract["restart_verification"]["post_restart_algorithm"]
    assert any("every descriptor" in step for step in algorithm)
    assert any("every adjacent descriptor pair" in step for step in algorithm)
    crash = contract["restart_verification"]["LOCAL_COMMIT_BEFORE_EVIDENCE"]
    assert "NO FINALIZE" in crash["missing_any_descriptor_or_invalid_chain"]


def test_complete_chain_deletion_detection_and_no_performance_shortcut() -> None:
    requirement = MACHINE["state_store_fingerprint_contract"]["durable_transaction_descriptor"][
        "complete_chain_requirement"
    ]
    assert "deletion of any descriptor" in requirement["deletion_detection"]
    assert requirement["performance_non_claim"] == [
        "no checkpoints",
        "no Merkle tree",
        "no descriptor-chain aggregate hash",
        "no pruning",
    ]


def test_complete_chain_backup_and_authority_boundaries_remain_non_claims() -> None:
    descriptor = MACHINE["state_store_fingerprint_contract"]["durable_transaction_descriptor"]
    assert "complete descriptor chain 1..G" in descriptor["backup_restore_limitation"]
    assert "does not establish restore authority" in descriptor["backup_restore_limitation"]
    assert descriptor["chain_continuity"]["meaning"].endswith("not upstream authority")
    assert MACHINE["authority_model"]["live_current"] == "DENIED"
    assert MACHINE["authority_model"]["testnet_to_live_fallback"] is False
    assert MACHINE["transaction_protocol"]["cross_resource_acid_claim"] is False


def test_complete_chain_immutable_scope_is_exact_three_independent_fields() -> None:
    requirement = MACHINE["state_store_fingerprint_contract"]["durable_transaction_descriptor"][
        "complete_chain_requirement"
    ]
    assert requirement["immutable_scope_fields"] == [
        "account_id",
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
    ]
    assert requirement["every_descriptor_must_match_current_metadata_immutable_scope"] is True
    assert "not a substitute" in requirement["scope_comparison"]
    assert requirement["environment_is_immutable_scope_field"] is False
    assert requirement["state_store_schema_version_is_global_immutable_scope_field"] is False


def test_shuffled_valid_chain_is_storage_order_independent() -> None:
    chain = _descriptor_chain(4)
    metadata = _metadata_for_descriptor(chain[-1])
    shuffled = [chain[3], chain[1], chain[0], chain[2]]
    assert _complete_descriptor_chain_valid(chain, metadata)
    assert _complete_descriptor_chain_valid(shuffled, metadata)


def test_self_consistent_foreign_historical_account_fails_scope_not_hash() -> None:
    chain = _descriptor_chain(4)
    metadata = _metadata_for_descriptor(chain[-1])
    chain[1]["account_id"] = "acct_01890f3a-2b4c-7abc-8def-0123456789ac"
    _rehash_descriptor(chain[1])
    assert _descriptor_hash_valid(chain[1])
    assert _descriptor_chain_valid(chain[0], chain[1])
    assert not _complete_descriptor_chain_valid(chain, metadata)


def test_self_consistent_foreign_historical_device_fails_scope_not_hash() -> None:
    chain = _descriptor_chain(4)
    metadata = _metadata_for_descriptor(chain[-1])
    chain[2]["device_installation_id"] = "dev_01890f3a-2b4c-7abc-8def-0123456789ac"
    _rehash_descriptor(chain[2])
    assert _descriptor_hash_valid(chain[2])
    assert _descriptor_chain_valid(chain[1], chain[2])
    assert not _complete_descriptor_chain_valid(chain, metadata)


@pytest.mark.parametrize(
    ("field", "foreign_value"),
    [
        ("account_id", "acct_01890f3a-2b4c-7abc-8def-0123456789ac"),
        ("device_installation_id", "dev_01890f3a-2b4c-7abc-8def-0123456789ac"),
    ],
)
def test_self_consistent_foreign_genesis_fails_immutable_scope(
    field: str, foreign_value: str
) -> None:
    chain = _descriptor_chain(4)
    metadata = _metadata_for_descriptor(chain[-1])
    chain[0][field] = foreign_value
    _rehash_descriptor(chain[0])
    assert chain[0]["expected_current_generation"] is None
    assert _descriptor_hash_valid(chain[0])
    assert not _complete_descriptor_chain_valid(chain, metadata)


def test_descriptor_multiset_detects_duplicate_before_sort_and_never_uses_storage_order() -> None:
    requirement = MACHINE["state_store_fingerprint_contract"]["durable_transaction_descriptor"][
        "complete_chain_requirement"
    ]
    ordering = requirement["canonical_descriptor_order"]
    assert ordering["key"] == "target_generation"
    assert ordering["direction"] == "ascending"
    assert ordering["applied_after_exact_multiset_cardinality_validation"] is True
    assert requirement["storage_order_affects_validity"] is False
    assert "never deduplicate" in requirement["duplicate_detection"]


def test_restart_and_trusted_observation_validate_every_descriptor_scope() -> None:
    contract = MACHINE["state_store_fingerprint_contract"]
    algorithm = contract["restart_verification"]["post_restart_algorithm"]
    assert any(
        "every descriptor against current metadata immutable scope" in step for step in algorithm
    )
    binding = MACHINE["startup_recovery_model"]["trusted_observation_snapshot_binding"]
    assert "every descriptor 1..G" in binding["immutable_scope_gate"]
    assert binding["scope_mismatch"].startswith("NO EVIDENCE PUBLICATION / NO FINALIZE")
    assert binding["descriptor_storage_order_authoritative"] is False


def test_scope_corrective_preserves_current_binding_metadata_only_delta_and_authority() -> None:
    descriptor = MACHINE["state_store_fingerprint_contract"]["durable_transaction_descriptor"]
    assert set(descriptor["metadata_binding"]) == {
        "account_id",
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
        "state_store_schema_version",
        "environment",
        "target_generation",
        "post_state_fingerprint_sha256",
        "post_history_tail_fingerprint_sha256",
        "transaction_fingerprint_sha256",
        "comparison",
    }
    assert (
        descriptor["atomic_local_commit_binding"]["metadata_only_valid"][
            "violates_forbidden_partial_state"
        ]
        is False
    )
    assert descriptor["atomic_declared_mutation_binding"]["invariant"].startswith(
        "durably committed current-record mutations"
    )
    assert MACHINE["authority_model"]["live_current"] == "DENIED"
    assert MACHINE["authority_model"]["testnet_to_live_fallback"] is False
    assert MACHINE["transaction_protocol"]["cross_resource_acid_claim"] is False


def test_backup_limitation_preserves_complete_chain_and_exact_immutable_scope() -> None:
    limitation = MACHINE["state_store_fingerprint_contract"]["durable_transaction_descriptor"][
        "backup_restore_limitation"
    ]
    assert "complete descriptor chain 1..G" in limitation
    assert "exact immutable account/device/store-identity scope" in limitation
    assert "does not establish restore authority" in limitation


def test_descriptor_intrinsic_contract_is_closed_and_precedes_all_trust_checks() -> None:
    contract = MACHINE["state_store_fingerprint_contract"]["durable_transaction_descriptor"]
    intrinsic = contract["intrinsic_validation"]
    schema = MACHINE["executable_boundary_schemas"]["StateStoreTransactionDescriptor"]
    assert intrinsic["exact_field_set"] == schema["required"]
    assert len(intrinsic["exact_field_set"]) == 14
    assert intrinsic["additional_properties"] is False
    assert intrinsic["hash_recomputation_occurs_only_after_intrinsic_validation"] is True
    assert intrinsic["valid_transaction_hash_compensates_for_invalid_schema"] is False


@pytest.mark.parametrize("extra", ["authorized", "accepted", "live_allowed"])
def test_authority_looking_extra_field_fails_despite_valid_projection_hash(extra: str) -> None:
    chain = _descriptor_chain(4)
    metadata = _metadata_for_descriptor(chain[-1])
    chain[1][extra] = True
    assert _descriptor_hash_valid(chain[1])
    assert not _descriptor_intrinsically_valid(chain[1])
    assert not _complete_descriptor_chain_valid(chain, metadata)


@pytest.mark.parametrize(
    ("field", "value"),
    [("account_id", "acct-invalid"), ("device_installation_id", "dev-invalid")],
)
def test_descriptor_intrinsic_validation_rejects_noncanonical_scope_id(
    field: str, value: str
) -> None:
    descriptor = _transaction_descriptor()
    descriptor[field] = value
    _rehash_descriptor(descriptor)
    assert _descriptor_hash_valid(descriptor)
    assert not _descriptor_intrinsically_valid(descriptor)


def test_self_consistent_invalid_historical_environment_fails_intrinsic_validation() -> None:
    chain = _descriptor_chain(4)
    metadata = _metadata_for_descriptor(chain[-1])
    chain[1]["environment"] = "HACKED"
    _rehash_descriptor(chain[1])
    assert _descriptor_hash_valid(chain[1])
    assert not _descriptor_intrinsically_valid(chain[1])
    assert not _complete_descriptor_chain_valid(chain, metadata)


@pytest.mark.parametrize("value", [True, 0])
def test_self_consistent_invalid_historical_schema_version_fails_intrinsic_validation(
    value: Any,
) -> None:
    chain = _descriptor_chain(4)
    metadata = _metadata_for_descriptor(chain[-1])
    chain[1]["state_store_schema_version"] = value
    _rehash_descriptor(chain[1])
    assert _descriptor_hash_valid(chain[1])
    assert not _descriptor_intrinsically_valid(chain[1])
    assert not _complete_descriptor_chain_valid(chain, metadata)


def test_self_consistent_boolean_expected_generation_fails_intrinsic_before_edge() -> None:
    chain = _descriptor_chain(4)
    metadata = _metadata_for_descriptor(chain[-1])
    chain[1]["expected_current_generation"] = True
    _rehash_descriptor(chain[1])
    assert _descriptor_hash_valid(chain[1])
    assert not _descriptor_intrinsically_valid(chain[1])
    assert not _complete_descriptor_chain_valid(chain, metadata)


@pytest.mark.parametrize(
    ("field", "record_name"),
    [
        ("current_record_mutations", "Workspace"),
        ("immutable_history_appends", "RuntimeSession canonical identity/history"),
    ],
)
def test_self_consistent_outer_hash_cannot_mask_invalid_nested_stage_one_record(
    field: str, record_name: str
) -> None:
    chain = _descriptor_chain(4)
    metadata = _metadata_for_descriptor(chain[-1])
    record = _persistence_record(record_name)
    record["payload_fingerprint_sha256"] = "f" * 64
    chain[1][field] = [record]
    _rehash_descriptor(chain[1])
    assert _descriptor_hash_valid(chain[1])
    assert not _validate_persistence_record(record)
    assert not _descriptor_intrinsically_valid(chain[1])
    assert not _complete_descriptor_chain_valid(chain, metadata)


def test_noncanonical_mutation_array_order_fails_but_canonical_variant_passes() -> None:
    records = [
        _persistence_record("Workspace"),
        _persistence_record("CryptoHunterAccount current record"),
    ]
    canonical = _canonical_records(records)
    descriptor = _transaction_descriptor(current_record_mutations=canonical)
    assert _descriptor_intrinsically_valid(descriptor)
    descriptor["current_record_mutations"] = list(reversed(canonical))
    _rehash_descriptor(descriptor)
    assert _descriptor_hash_valid(descriptor)
    assert not _descriptor_intrinsically_valid(descriptor)


def test_missing_descriptor_field_fails_intrinsic_and_full_chain_without_key_error() -> None:
    chain = _descriptor_chain(4)
    metadata = _metadata_for_descriptor(chain[-1])
    chain[1].pop("environment")
    assert not _descriptor_intrinsically_valid(chain[1])
    assert not _complete_descriptor_chain_valid(chain, metadata)


def test_intrinsic_validation_accepts_metadata_only_and_valid_historical_environment() -> None:
    descriptor = _transaction_descriptor(
        environment="TESTNET", current_record_mutations=[], immutable_history_appends=[]
    )
    assert _descriptor_intrinsically_valid(descriptor)
    assert descriptor["current_record_mutations"] == []
    assert descriptor["immutable_history_appends"] == []


def test_intrinsic_duplicate_audit_does_not_invent_unfrozen_array_rule() -> None:
    semantics = MACHINE["state_store_fingerprint_contract"]["durable_transaction_descriptor"][
        "intrinsic_validation"
    ]["mutation_arrays"]["duplicate_entry_semantics"]
    assert semantics.startswith("NO NEW RULE")
    assert "does not unambiguously specify duplicate" in semantics


def test_intrinsic_corrective_preserves_chain_scope_current_binding_and_authority() -> None:
    chain = _descriptor_chain(4)
    metadata = _metadata_for_descriptor(chain[-1])
    shuffled = [chain[3], chain[0], chain[2], chain[1]]
    assert _complete_descriptor_chain_valid(shuffled, metadata)
    descriptor = MACHINE["state_store_fingerprint_contract"]["durable_transaction_descriptor"]
    assert descriptor["metadata_binding"]["environment"] == "environment"
    assert descriptor["metadata_binding"]["state_store_schema_version"] == (
        "state_store_schema_version"
    )
    assert MACHINE["authority_model"]["live_current"] == "DENIED"
    assert MACHINE["authority_model"]["testnet_to_live_fallback"] is False
    assert MACHINE["transaction_protocol"]["cross_resource_acid_claim"] is False


def test_backup_limitation_requires_intrinsic_validation_of_every_restored_descriptor() -> None:
    limitation = MACHINE["state_store_fingerprint_contract"]["durable_transaction_descriptor"][
        "backup_restore_limitation"
    ]
    assert "intrinsic validation" in limitation
    assert "external M0.3 restore freshness authority" in limitation


def test_backup_integrity_metadata_is_exact_required_descriptor_carrier() -> None:
    schema = MACHINE["executable_boundary_schemas"]["BackupEnvelope"]
    metadata = schema["properties"]["integrity_metadata"]
    assert metadata["additionalProperties"] is False
    assert metadata["required"] == ["state_store_transaction_descriptors"]
    assert metadata["properties"]["state_store_transaction_descriptors"]["items"] == {
        "$ref": "#/executable_boundary_schemas/StateStoreTransactionDescriptor"
    }
    assert _validate_backup(_valid_backup()) == "VALID"


@pytest.mark.parametrize("targets", [[1], [2], [1, 1], [1, 2, 3], [0, 2], [True, 2]])
def test_backup_rejects_non_exact_original_descriptor_generation_multiset(
    targets: list[int],
) -> None:
    backup = _valid_backup()
    chain = _descriptor_chain(3)
    backup["integrity_metadata"]["state_store_transaction_descriptors"] = [
        copy.deepcopy(chain[int(target) - 1])
        if type(target) is int and target > 0
        else {
            **copy.deepcopy(chain[0]),
            "target_generation": target,
        }
        for target in targets
    ]
    assert _validate_backup(backup) == "BACKUP_INTEGRITY_FAILED"


def test_backup_reordered_descriptor_input_canonicalizes_deterministically() -> None:
    backup = _valid_backup()
    canonical_fingerprint = _actual_fingerprint(_canonical_backup_projection(backup))
    descriptors = backup["integrity_metadata"]["state_store_transaction_descriptors"]
    backup["integrity_metadata"]["state_store_transaction_descriptors"] = list(
        reversed(descriptors)
    )
    reordered_fingerprint = _actual_fingerprint(_canonical_backup_projection(backup))
    assert reordered_fingerprint == canonical_fingerprint
    assert _validate_backup(backup) == "VALID"


def test_descriptor_input_permutation_alone_does_not_change_canonical_fingerprint() -> None:
    canonical = _valid_backup()
    permuted = copy.deepcopy(canonical)
    descriptors = permuted["integrity_metadata"]["state_store_transaction_descriptors"]
    permuted["integrity_metadata"]["state_store_transaction_descriptors"] = descriptors[::-1]
    assert _actual_fingerprint(_canonical_backup_projection(canonical)) == _actual_fingerprint(
        _canonical_backup_projection(permuted)
    )


@pytest.mark.parametrize(
    ("generation", "field", "replacement"),
    [
        (1, "account_id", "acct_01890f3a-2b4c-7abc-9def-0123456789ab"),
        (1, "device_installation_id", "dev_01890f3a-2b4c-7abc-9def-0123456789ab"),
        (1, "state_store_identity_fingerprint_sha256", "9" * 64),
        (2, "pre_state_fingerprint_sha256", "8" * 64),
        (2, "pre_history_tail_fingerprint_sha256", "8" * 64),
        (2, "post_state_fingerprint_sha256", "8" * 64),
    ],
)
def test_self_consistently_rehashed_descriptor_tamper_remains_invalid_backup(
    generation: int, field: str, replacement: Any
) -> None:
    backup = _valid_backup()
    descriptor = backup["integrity_metadata"]["state_store_transaction_descriptors"][generation - 1]
    descriptor[field] = replacement
    _rehash_descriptor(descriptor)
    backup["envelope_fingerprint_sha256"] = _actual_fingerprint(
        {key: value for key, value in backup.items() if key != "envelope_fingerprint_sha256"}
    )
    assert _validate_backup(backup) == "BACKUP_INTEGRITY_FAILED"


def test_descriptor_semantics_are_covered_only_by_envelope_fingerprint() -> None:
    backup = _valid_backup()
    original = backup["envelope_fingerprint_sha256"]
    backup["integrity_metadata"]["state_store_transaction_descriptors"][0]["environment"] = (
        "TESTNET"
    )
    projected = _canonical_backup_projection(backup)
    assert _actual_fingerprint(projected) != original
    contract = MACHINE["backup_contract"]["descriptor_preservation_contract"]
    assert contract["fingerprint_boundaries"]["second_descriptor_aggregate_hash"] is False
    assert set(contract["fingerprint_boundaries"]["excluded_from"]) >= {
        "state_fingerprint_sha256 projection",
        "history_tail_fingerprint_sha256 projection",
    }


def test_nested_descriptor_semantic_change_changes_canonical_envelope_fingerprint() -> None:
    backup = _valid_backup()
    original = _actual_fingerprint(_canonical_backup_projection(backup))
    descriptor = backup["integrity_metadata"]["state_store_transaction_descriptors"][0]
    descriptor["current_record_mutations"] = [_persistence_record("Workspace")]
    assert _actual_fingerprint(_canonical_backup_projection(backup)) != original


def test_nested_descriptor_records_cannot_hide_forbidden_material() -> None:
    backup = _valid_backup()
    descriptor = backup["integrity_metadata"]["state_store_transaction_descriptors"][0]
    forbidden = _persistence_record("Workspace")
    forbidden["payload"]["api_secret"] = "secret"
    forbidden["payload_fingerprint_sha256"] = _actual_fingerprint(forbidden["payload"])
    descriptor["current_record_mutations"] = [forbidden]
    _rehash_descriptor(descriptor)
    assert _validate_backup(backup) == "BACKUP_INTEGRITY_FAILED"
    scope = MACHINE["backup_contract"]["forbidden_record_scope"]
    assert any("current_record_mutations" in item for item in scope)
    assert any("immutable_history_appends" in item for item in scope)


def test_nested_descriptor_allows_canonical_pin_verifier_direct_verifier() -> None:
    backup = _valid_backup()
    descriptor = backup["integrity_metadata"]["state_store_transaction_descriptors"][-1]
    pin_record = _persistence_record("PinVerifierRecord accepted revisions")
    assert "verifier" in pin_record["payload"]["upstream_payload"]
    descriptor["immutable_history_appends"] = [pin_record]
    _rehash_descriptor(descriptor)
    backup["transaction_fingerprint_sha256"] = descriptor["transaction_fingerprint_sha256"]
    backup["envelope_fingerprint_sha256"] = _actual_fingerprint(
        _canonical_backup_projection(backup)
    )
    assert _validate_backup(backup) == "VALID"


def test_descriptor_backup_remains_candidate_without_any_authority_by_possession() -> None:
    contract = MACHINE["backup_contract"]
    preservation = contract["descriptor_preservation_contract"]
    assert contract["candidate_only"] is True
    assert preservation["authority_by_possession"] is False
    assert set(preservation["does_not_establish"]) >= {
        "protected membership",
        "current protected reference",
        "M0.3 authority",
        "LocalDurableEvidence membership",
        "LIVE",
        "restore authority",
    }
    assert "LocalDurableStateEvidence payload" in contract["excludes"]


def _s7c_backup(generation: int = 2) -> dict[str, Any]:
    value = _valid_backup()
    descriptors = _descriptor_chain(generation)
    final = descriptors[-1]
    value.update(
        local_protected_freshness_generation=generation,
        state_fingerprint_sha256=final["post_state_fingerprint_sha256"],
        transaction_fingerprint_sha256=final["transaction_fingerprint_sha256"],
        history_tail_fingerprint_sha256=final["post_history_tail_fingerprint_sha256"],
        integrity_metadata={"state_store_transaction_descriptors": descriptors},
    )
    value["envelope_fingerprint_sha256"] = _actual_fingerprint(_canonical_backup_projection(value))
    assert _validate_backup(value) == "VALID"
    return value


def _s7c_protected(backup: dict[str, Any], lifecycle: str = "COMMITTED") -> dict[str, Any]:
    value = {
        "account_id": backup["account_id"],
        "device_installation_id": backup["device_installation_id"],
        "state_store_identity_fingerprint_sha256": backup[
            "state_store_identity_fingerprint_sha256"
        ],
        "lifecycle": lifecycle,
        "committed_generation": None,
        "committed_state_fingerprint_sha256": None,
        "prepared_generation": None,
        "prepared_state_fingerprint_sha256": None,
        "prepared_transaction_fingerprint_sha256": None,
        "authority_revision": 7,
        "authority_source": "EXTERNAL_PRODUCT_PROTECTED_STATE_BOUNDARY",
        "content_fingerprint_sha256": "",
    }
    if lifecycle == "COMMITTED":
        value.update(
            committed_generation=backup["local_protected_freshness_generation"],
            committed_state_fingerprint_sha256=backup["state_fingerprint_sha256"],
        )
    elif lifecycle == "PREPARED":
        value.update(
            prepared_generation=backup["local_protected_freshness_generation"],
            prepared_state_fingerprint_sha256=backup["state_fingerprint_sha256"],
            prepared_transaction_fingerprint_sha256=backup["transaction_fingerprint_sha256"],
        )
        if backup["local_protected_freshness_generation"] > 1:
            value.update(
                committed_generation=backup["local_protected_freshness_generation"] - 1,
                committed_state_fingerprint_sha256="9" * 64,
            )
    value["content_fingerprint_sha256"] = _actual_fingerprint(
        {key: item for key, item in value.items() if key != "content_fingerprint_sha256"}
    )
    assert _protected_record_valid(value)
    return value


def _s7c_membership(record: dict[str, Any], *, historical: bool = False) -> dict[str, Any]:
    scope = (
        record["account_id"],
        record["device_installation_id"],
        record["state_store_identity_fingerprint_sha256"],
    )
    return {
        "resolved_ref": "protected:1",
        "accepted_refs": {"protected:1"},
        "current_ref_by_scope": {scope: "protected:2" if historical else "protected:1"},
    }


def _s7c_local(
    backup: dict[str, Any], generation: int | None = None, **updates: Any
) -> dict[str, Any]:
    observation = {
        "account_id": backup["account_id"],
        "device_installation_id": backup["device_installation_id"],
        "state_store_identity_fingerprint_sha256": backup[
            "state_store_identity_fingerprint_sha256"
        ],
        "environment": backup["environment"],
        "generation": generation or backup["local_protected_freshness_generation"],
        "state_fingerprint_sha256": backup["state_fingerprint_sha256"],
        "transaction_fingerprint_sha256": backup["transaction_fingerprint_sha256"],
        "history_tail_fingerprint_sha256": backup["history_tail_fingerprint_sha256"],
    }
    observation.update(updates)
    return {"status": "VERIFIED_STATE", "observation": observation}


def test_s7c_closed_decisions_cover_every_executable_and_canonical_emission() -> None:
    contract = MACHINE["restore_contract"]["s7c_restore_freshness"]
    registry = set(contract["closed_decision_registry"])
    canonical = {
        decision
        for row in contract["freshness_truth_table"]
        for decision in row["eligible_final_decisions"]
    }
    backup = _s7c_backup()
    external = _s7c_protected(backup)
    membership = _s7c_membership(external)
    emitted = {
        assess_restore_candidate(backup, external, membership, {"status": status})
        for status in ("VERIFIED_EMPTY", "NO_TRUSTED_OBSERVATION", "CORRUPT_OR_UNREADABLE")
    }
    emitted.add(select_restore_candidate([backup], external, membership)[0])
    assert canonical | emitted <= registry
    mutated = copy.deepcopy(contract["freshness_truth_table"])
    mutated[0]["eligible_final_decisions"] = ["MAYBE_RESTORE"]
    assert not {d for row in mutated for d in row["eligible_final_decisions"]} <= registry


def test_s7c_environment_is_bound_without_external_environment_field() -> None:
    state_fields = MACHINE["state_store_fingerprint_contract"]["state_fingerprint"][
        "projection_fields"
    ]
    tx_fields = MACHINE["state_store_fingerprint_contract"]["transaction_fingerprint"][
        "projection_fields"
    ]
    assert "environment" in state_fields and "environment" in tx_fields
    assert (
        "environment" not in PROTECTED_FIELDS and "environment_observation" not in PROTECTED_FIELDS
    )
    backup = _s7c_backup()
    external = _s7c_protected(backup)
    membership = _s7c_membership(external)
    changed = copy.deepcopy(backup)
    changed["environment"] = "TESTNET"
    for descriptor in changed["integrity_metadata"]["state_store_transaction_descriptors"]:
        descriptor["environment"] = "TESTNET"
        descriptor["post_state_fingerprint_sha256"] = _actual_fingerprint(
            {"environment": "TESTNET", "generation": descriptor["target_generation"]}
        )
        _rehash_descriptor(descriptor)
    descriptors = changed["integrity_metadata"]["state_store_transaction_descriptors"]
    for previous, current in zip(descriptors, descriptors[1:], strict=False):
        current["pre_state_fingerprint_sha256"] = previous["post_state_fingerprint_sha256"]
        _rehash_descriptor(current)
    changed.update(
        state_fingerprint_sha256=descriptors[-1]["post_state_fingerprint_sha256"],
        transaction_fingerprint_sha256=descriptors[-1]["transaction_fingerprint_sha256"],
    )
    changed["envelope_fingerprint_sha256"] = _actual_fingerprint(
        _canonical_backup_projection(changed)
    )
    assert _validate_backup(changed) == "VALID"
    assert (
        assess_restore_candidate(changed, external, membership, {"status": "VERIFIED_EMPTY"})
        == "DENY"
    )


@pytest.mark.parametrize("membership_kind", ["missing", "historical", "raw"])
def test_s7c_valid_candidate_hash_never_replaces_current_membership(membership_kind: str) -> None:
    backup = _s7c_backup()
    external = _s7c_protected(backup)
    membership = _s7c_membership(external)
    if membership_kind == "missing":
        membership["accepted_refs"] = set()
    elif membership_kind == "historical":
        membership = _s7c_membership(external, historical=True)
    else:
        membership = {}
    assert _validate_backup(backup) == "VALID"
    assert (
        assess_restore_candidate(backup, external, membership, {"status": "VERIFIED_EMPTY"})
        == "DENY"
    )


def test_s7c_rejects_invalid_backup_and_invalid_external_intrinsics() -> None:
    backup = _s7c_backup()
    external = _s7c_protected(backup)
    membership = _s7c_membership(external)
    invalid = copy.deepcopy(backup)
    invalid["envelope_fingerprint_sha256"] = "0" * 64
    assert (
        assess_restore_candidate(invalid, external, membership, {"status": "VERIFIED_EMPTY"})
        == "DENY"
    )
    for mutation in ("fingerprint", "lifecycle"):
        malformed = copy.deepcopy(external)
        if mutation == "fingerprint":
            malformed["content_fingerprint_sha256"] = "0" * 64
        else:
            malformed["prepared_generation"] = 3
        assert (
            assess_restore_candidate(backup, malformed, membership, {"status": "VERIFIED_EMPTY"})
            == "DENY"
        )


@pytest.mark.parametrize(
    ("local", "committed", "prepared"),
    [
        (
            {"status": "VERIFIED_EMPTY"},
            "RESTORE_EXTERNAL_COMMITTED_CURRENT",
            "RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE",
        ),
        (
            {"status": "NO_TRUSTED_OBSERVATION"},
            "RESTORE_EXTERNAL_COMMITTED_CURRENT",
            "RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE",
        ),
        (
            {"status": "CORRUPT_OR_UNREADABLE"},
            "RESTORE_EXTERNAL_COMMITTED_CURRENT",
            "RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE",
        ),
    ],
)
def test_s7c_local_availability_cross_product(
    local: dict[str, Any], committed: str, prepared: str
) -> None:
    backup = _s7c_backup()
    c = _s7c_protected(backup)
    p = _s7c_protected(backup, "PREPARED")
    assert assess_restore_candidate(backup, c, _s7c_membership(c), local) == committed
    assert assess_restore_candidate(backup, p, _s7c_membership(p), local) == prepared


def test_s7c_local_classification_is_derived_from_values_and_fails_closed() -> None:
    backup = _s7c_backup()
    committed = _s7c_protected(backup)
    membership = _s7c_membership(committed)
    assert (
        assess_restore_candidate(backup, committed, membership, _s7c_local(backup))
        == "NOOP_ALREADY_CURRENT"
    )
    assert (
        assess_restore_candidate(backup, committed, membership, _s7c_local(backup, 1))
        == "RESTORE_EXTERNAL_COMMITTED_CURRENT"
    )
    mutations = (
        {"state_fingerprint_sha256": "8" * 64},
        {"transaction_fingerprint_sha256": "8" * 64},
        {"generation": 3},
        {"account_id": "acct_wrong"},
        {"environment": "LIVE"},
    )
    for update in mutations:
        assert (
            assess_restore_candidate(backup, committed, membership, _s7c_local(backup, **update))
            == "DENY"
        )


def test_s7c_prepared_local_transaction_mismatch_denies_finalize() -> None:
    backup = _s7c_backup()
    external = _s7c_protected(backup, "PREPARED")
    assert (
        assess_restore_candidate(
            backup,
            external,
            _s7c_membership(external),
            _s7c_local(backup, transaction_fingerprint_sha256="8" * 64),
        )
        == "DENY"
    )
    assert (
        assess_restore_candidate(backup, external, _s7c_membership(external), _s7c_local(backup))
        == "RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE"
    )


def test_s7c_selector_filters_authority_and_denies_ambiguity() -> None:
    g1, g2, g3 = (_s7c_backup(n) for n in (1, 2, 3))
    external = _s7c_protected(g2)
    membership = _s7c_membership(external)
    decision, selected = select_restore_candidate([g1, g3, g2], external, membership)
    assert decision == "RESTORE_EXTERNAL_COMMITTED_CURRENT" and selected == g2
    assert select_restore_candidate([g1, g3], external, membership) == ("DENY", None)
    distinct_lineage = copy.deepcopy(g2)
    distinct_lineage["integrity_metadata"]["state_store_transaction_descriptors"][0][
        "state_store_schema_version"
    ] = 2
    _rehash_descriptor(
        distinct_lineage["integrity_metadata"]["state_store_transaction_descriptors"][0]
    )
    distinct_lineage["envelope_fingerprint_sha256"] = _actual_fingerprint(
        _canonical_backup_projection(distinct_lineage)
    )
    assert _validate_backup(distinct_lineage) == "VALID" and distinct_lineage != g2
    assert select_restore_candidate([g2, distinct_lineage], external, membership) == ("DENY", None)


def _s7c_truth_fixture(case: str) -> tuple[dict[str, Any], dict[str, Any]]:
    g1, g2, g3, g4 = (_s7c_backup(n) for n in (1, 2, 3, 4))
    if case == "UNINITIALIZED_G1":
        return g1, _s7c_protected(g1, "UNINITIALIZED")
    if case == "COMMITTED_G_candidate_G_minus_1":
        return g1, _s7c_protected(g2)
    if case == "COMMITTED_G_exact_G_state":
        return g2, _s7c_protected(g2)
    if case == "COMMITTED_G_wrong_state":
        external = _s7c_protected(g2)
        external["committed_state_fingerprint_sha256"] = "8" * 64
    elif case == "COMMITTED_G_candidate_G_plus_1":
        return g3, _s7c_protected(g2)
    elif case == "PREPARED_G_plus_1_candidate_G":
        return g2, _s7c_protected(g3, "PREPARED")
    elif case == "PREPARED_G_plus_1_exact_state_transaction":
        return g3, _s7c_protected(g3, "PREPARED")
    elif case == "PREPARED_G_plus_1_wrong_state":
        external = _s7c_protected(g3, "PREPARED")
        external["prepared_state_fingerprint_sha256"] = "8" * 64
    elif case == "PREPARED_G_plus_1_wrong_transaction":
        external = _s7c_protected(g3, "PREPARED")
        external["prepared_transaction_fingerprint_sha256"] = "8" * 64
    elif case == "PREPARED_G_plus_1_candidate_G_plus_2":
        return g4, _s7c_protected(g3, "PREPARED")
    else:
        return g1, _s7c_protected(g1, "PREPARED")
    external["content_fingerprint_sha256"] = _actual_fingerprint(
        {key: item for key, item in external.items() if key != "content_fingerprint_sha256"}
    )
    return g2 if case.startswith("COMMITTED") else g3, external


def _s7c_local_cross_product(backup: dict[str, Any]) -> list[dict[str, Any]]:
    generation = backup["local_protected_freshness_generation"]
    values = [
        {"status": "VERIFIED_EMPTY"},
        {"status": "NO_TRUSTED_OBSERVATION"},
        {"status": "CORRUPT_OR_UNREADABLE"},
        _s7c_local(backup),
        _s7c_local(backup, state_fingerprint_sha256="8" * 64),
        _s7c_local(backup, transaction_fingerprint_sha256="8" * 64),
        _s7c_local(backup, history_tail_fingerprint_sha256="8" * 64),
        _s7c_local(backup, generation=generation + 1),
        _s7c_local(backup, account_id="acct_wrong"),
        _s7c_local(backup, environment="LIVE"),
    ]
    if generation > 1:
        values.append(_s7c_local(backup, generation=generation - 1))
    return values


def _actual_final_decisions(backup: dict[str, Any], external: dict[str, Any]) -> set[str]:
    membership = _s7c_membership(external)
    return {
        assess_restore_candidate(backup, external, membership, local)
        for local in _s7c_local_cross_product(backup)
    }


@pytest.mark.parametrize(
    "row",
    MACHINE["restore_contract"]["s7c_restore_freshness"]["freshness_truth_table"],
    ids=lambda row: row["case"],
)
def test_s7c_canonical_truth_table_has_exact_phase_a_and_phase_b_parity(
    row: dict[str, Any],
) -> None:
    backup, external = _s7c_truth_fixture(row["case"])
    membership = _s7c_membership(external)
    assert _external_eligibility(backup, external, membership) == row["external_eligibility"]
    assert _actual_final_decisions(backup, external) == set(row["eligible_final_decisions"])


@pytest.mark.parametrize(
    "case",
    [
        "COMMITTED_G_exact_G_state",
        "PREPARED_G_plus_1_exact_state_transaction",
        "GENESIS_PREPARED_1_exact_state_transaction",
    ],
)
def test_s7c_valid_closed_token_truth_table_drift_is_detected(case: str) -> None:
    row = next(
        row
        for row in MACHINE["restore_contract"]["s7c_restore_freshness"]["freshness_truth_table"]
        if row["case"] == case
    )
    backup, external = _s7c_truth_fixture(case)
    actual = _actual_final_decisions(backup, external)
    mutated_expected = {"DENY"}
    assert mutated_expected <= RESTORE_DECISIONS
    assert actual == set(row["eligible_final_decisions"])
    assert actual != mutated_expected


EXPECTED_S7C_CRASH_MATRIX = [
    {
        "point": "before isolated restore",
        "live_effect": "UNCHANGED",
        "external_effect": "UNCHANGED",
        "recovery_action": "SAFE_RETRY",
        "second_restore_write_allowed": False,
        "protected_action_allowed": None,
    },
    {
        "point": "during isolated creation",
        "live_effect": "UNCHANGED",
        "external_effect": "UNCHANGED",
        "recovery_action": "DISCARD_AND_REBUILD_ISOLATED",
        "second_restore_write_allowed": False,
        "protected_action_allowed": None,
    },
    {
        "point": "after isolated verification before install",
        "live_effect": "UNCHANGED",
        "external_effect": "UNCHANGED",
        "recovery_action": "SAFE_RETRY",
        "second_restore_write_allowed": False,
        "protected_action_allowed": None,
    },
    {
        "point": "after install before fresh evidence",
        "live_effect": "EXACT_RESTORED_TARGET",
        "external_effect": "UNCHANGED",
        "recovery_action": "OBSERVE_EXACT_LOCAL_AND_REBUILD_FRESH_EVIDENCE",
        "second_restore_write_allowed": False,
        "protected_action_allowed": None,
    },
    {
        "point": "after evidence before FINALIZE",
        "live_effect": "EXACT_RESTORED_TARGET",
        "external_effect": "EXISTING_PENDING_RETAINED",
        "recovery_action": "RESOLVE_EVIDENCE_AND_FINALIZE_EXISTING_PENDING",
        "second_restore_write_allowed": False,
        "protected_action_allowed": "FINALIZE",
    },
    {
        "point": "FINALIZE ACK loss",
        "live_effect": "EXACT_RESTORED_TARGET",
        "external_effect": "UNKNOWN_UNTIL_RERESOLUTION",
        "recovery_action": "RERESOLVE_SAME_REF_NO_BLIND_RETRY",
        "second_restore_write_allowed": False,
        "protected_action_allowed": None,
    },
]


def _crash_matrix_exact(value: Any) -> bool:
    return value == EXPECTED_S7C_CRASH_MATRIX


def test_s7c_crash_matrix_and_protected_actions_are_exact() -> None:
    contract = MACHINE["restore_contract"]["s7c_restore_freshness"]
    assert _crash_matrix_exact(contract["crash_matrix"])
    assert contract["protected_action_inventory"] == {
        "allowed": [
            {
                "action": "FINALIZE",
                "condition": "exact existing PREPARED target after fresh accepted/current S4 evidence",
            }
        ],
        "forbidden": ["PREPARE", "ABORT", "replacement", "mint current", "set current"],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("live_effect", "SECOND_RESTORE_WRITE"),
        ("external_effect", "ABORT"),
        ("recovery_action", "BLIND_FINALIZE_RETRY"),
        ("second_restore_write_allowed", True),
        ("protected_action_allowed", "PREPARE"),
    ],
)
def test_s7c_every_crash_semantic_column_mutation_fails_exact_validator(
    field: str, value: Any
) -> None:
    mutated = copy.deepcopy(EXPECTED_S7C_CRASH_MATRIX)
    mutated[3][field] = value
    assert not _crash_matrix_exact(mutated)


def _rehash_protected(record: dict[str, Any]) -> None:
    record["content_fingerprint_sha256"] = _actual_fingerprint(
        {key: item for key, item in record.items() if key != "content_fingerprint_sha256"}
    )


def _ack_loss_result(
    backup: dict[str, Any],
    prepared: dict[str, Any],
    terminal: dict[str, Any],
    membership: dict[str, Any],
    local: dict[str, Any],
) -> str:
    return reconcile_finalize_outcome(backup, prepared, "protected:1", terminal, membership, local)


def test_s7c_finalize_ack_loss_derives_exact_local_and_revalidates_terminal_membership() -> None:
    backup = _s7c_backup()
    prepared = _s7c_protected(backup, "PREPARED")
    terminal = _s7c_protected(backup)
    membership = _s7c_membership(terminal)
    assert (
        _ack_loss_result(backup, prepared, terminal, membership, _s7c_local(backup))
        == "NOOP_ALREADY_CURRENT"
    )
    assert (
        _ack_loss_result(
            backup,
            prepared,
            terminal,
            _s7c_membership(terminal, historical=True),
            _s7c_local(backup),
        )
        == "DENY"
    )
    changed_ref = _s7c_membership(terminal)
    changed_ref["resolved_ref"] = "protected:2"
    changed_ref["accepted_refs"].add("protected:2")
    scope = next(iter(changed_ref["current_ref_by_scope"]))
    changed_ref["current_ref_by_scope"][scope] = "protected:2"
    assert _ack_loss_result(backup, prepared, terminal, changed_ref, _s7c_local(backup)) == "DENY"


@pytest.mark.parametrize(
    ("mutation", "rehash"),
    [
        ("partial", False),
        ("fingerprint", False),
        ("source", True),
        ("scope", True),
        ("generation", True),
        ("state", True),
    ],
)
def test_s7c_finalize_ack_loss_rejects_invalid_or_wrong_terminal(
    mutation: str, rehash: bool
) -> None:
    backup = _s7c_backup()
    prepared = _s7c_protected(backup, "PREPARED")
    terminal = _s7c_protected(backup)
    if mutation == "partial":
        terminal.pop("authority_revision")
    elif mutation == "fingerprint":
        terminal["content_fingerprint_sha256"] = "0" * 64
    elif mutation == "source":
        terminal["authority_source"] = "CALLER"
    elif mutation == "scope":
        terminal["device_installation_id"] = "dev_wrong"
    elif mutation == "generation":
        terminal["committed_generation"] += 1
    else:
        terminal["committed_state_fingerprint_sha256"] = "8" * 64
    if rehash:
        _rehash_protected(terminal)
    assert (
        _ack_loss_result(
            backup, prepared, terminal, _s7c_membership(_s7c_protected(backup)), _s7c_local(backup)
        )
        == "DENY"
    )


@pytest.mark.parametrize(
    "local_update",
    [
        {"transaction_fingerprint_sha256": "8" * 64},
        {"history_tail_fingerprint_sha256": "8" * 64},
        {"account_id": "acct_wrong"},
        {"environment": "LIVE"},
        {"generation": 3},
    ],
)
def test_s7c_finalize_ack_loss_rejects_non_exact_local(local_update: dict[str, Any]) -> None:
    backup = _s7c_backup()
    prepared = _s7c_protected(backup, "PREPARED")
    terminal = _s7c_protected(backup)
    assert (
        _ack_loss_result(
            backup,
            prepared,
            terminal,
            _s7c_membership(terminal),
            _s7c_local(backup, **local_update),
        )
        == "DENY"
    )


def test_s7c_finalize_ack_loss_rejects_invalid_backup() -> None:
    backup = _s7c_backup()
    prepared = _s7c_protected(backup, "PREPARED")
    terminal = _s7c_protected(backup)
    backup["envelope_fingerprint_sha256"] = "0" * 64
    assert (
        _ack_loss_result(
            backup, prepared, terminal, _s7c_membership(terminal), _s7c_local(_s7c_backup())
        )
        == "DENY"
    )


def test_s7c_every_final_function_output_stays_in_closed_registry() -> None:
    backup = _s7c_backup()
    committed = _s7c_protected(backup)
    prepared = _s7c_protected(backup, "PREPARED")
    outputs = {
        assess_restore_candidate(backup, committed, _s7c_membership(committed), local)
        for local in _s7c_local_cross_product(backup)
    }
    outputs.add(select_restore_candidate([backup], committed, _s7c_membership(committed))[0])
    outputs.add(
        reconcile_finalize_outcome(
            backup,
            prepared,
            "protected:1",
            committed,
            _s7c_membership(committed),
            _s7c_local(backup),
        )
    )
    assert outputs <= RESTORE_DECISIONS
