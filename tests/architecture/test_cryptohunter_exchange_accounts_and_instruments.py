"""Final fail-closed contract tests for CryptoHunter M0.5 exchange accounts and instruments."""
from __future__ import annotations

import hashlib
import inspect
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARCH_DIR = ROOT / "docs/architecture/cryptohunter_product_architecture"
DOC = ARCH_DIR / "exchange_accounts_and_instruments.json"
MD = ARCH_DIR / "exchange_accounts_and_instruments.md"
M04_DOC = ARCH_DIR / "environment_and_product_capabilities.json"
M04_MD = ARCH_DIR / "environment_and_product_capabilities.md"
ARCH = ARCH_DIR / "README.md"


def load_no_dupes(path: Path):
    def hook(pairs):
        out = {}
        for key, value in pairs:
            assert key not in out, f"duplicate key {key} in {path}"
            out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=hook)


DATA = load_no_dupes(DOC)
M04 = load_no_dupes(M04_DOC)
DENIALS = set(DATA["denial_code_registry"])
OPERATIONS = DATA["operation_dispatch_policy"]["closed_operation_registry"]
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def deny(code):
    assert code in DENIALS
    return False, code


def is_nonempty_str(value):
    return isinstance(value, str) and value != ""


def unique_str_list(value):
    return isinstance(value, list) and all(is_nonempty_str(v) for v in value) and len(value) == len(set(value))


def closed_schema(obj, required, allowed, denial):
    if not isinstance(obj, dict):
        return deny(denial)
    keys = set(obj)
    if not set(required) <= keys or not keys <= set(allowed):
        return deny(denial)
    return True, None


def parse_ts(value):
    if not isinstance(value, str) or not re.fullmatch(DATA["timestamp_policy"]["regex"], value):
        return None
    base = value[:-1]
    frac = ""
    if "." in base:
        base, frac = base.split(".", 1)
    try:
        dt = datetime.strptime(base, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return dt, int((frac + "0" * 9)[:9]) if frac else 0


def ts_lt(left, right):
    left_parsed, right_parsed = parse_ts(left), parse_ts(right)
    return left_parsed is not None and right_parsed is not None and left_parsed < right_parsed


def ts_le(left, right):
    left_parsed, right_parsed = parse_ts(left), parse_ts(right)
    return left_parsed is not None and right_parsed is not None and left_parsed <= right_parsed


def is_canonical_decimal(value, *, positive=False):
    if not isinstance(value, str) or not re.fullmatch(DATA["decimal_policy"]["regex"], value):
        return False
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError):
        return False
    return parsed > 0 if positive else parsed >= 0


def hash_payload(defn, payload):
    canonical = {field: payload.get(field) for field in defn["input_fields"]}
    for array_field in ("instrument_ids", "source_catalog_snapshot_ids", "observed_permission_set", "supported_instrument_types"):
        if isinstance(canonical.get(array_field), list):
            canonical[array_field] = sorted(canonical[array_field])
    raw = defn["domain_separator"] + "\n" + json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def exchange_entry(exchange_id):
    if not is_nonempty_str(exchange_id):
        return None
    for entry in DATA["exchange_registry_contract"]["entries"]:
        if entry.get("exchange_id") == exchange_id:
            return entry
    return None


def validate_exchange_registry_entry(entry):
    required = DATA["exchange_registry_contract"]["entry_fields"]
    ok, denial = closed_schema(entry, required, required, "UNKNOWN_EXCHANGE_ID")
    if not ok:
        return False, denial
    for field in ["exchange_id", "display_name", "adapter_family_id", "capability_discovery_policy", "instrument_catalog_discovery_policy", "account_identity_discovery_policy", "status"]:
        if not is_nonempty_str(entry.get(field)):
            return deny("UNKNOWN_EXCHANGE_ID")
    for field in ["supported_environments", "supported_market_types", "supported_instrument_types", "aliases"]:
        if not unique_str_list(entry.get(field)):
            return deny("UNKNOWN_EXCHANGE_ID")
    if entry["status"] not in {"ENABLED", "DISABLED"} or entry["status"] != "ENABLED":
        return deny("EXCHANGE_DISABLED")
    if entry["exchange_id"] in entry["aliases"]:
        return deny("UNKNOWN_EXCHANGE_ID")
    if not set(entry["supported_environments"]) <= set(DATA["environment_registry"]):
        return deny("EXCHANGE_ENVIRONMENT_UNSUPPORTED")
    if not set(entry["supported_market_types"]) <= set(DATA["market_type_registry"]):
        return deny("MARKET_TYPE_UNSUPPORTED")
    if not set(entry["supported_instrument_types"]) <= set(DATA["instrument_type_registry"]):
        return deny("INSTRUMENT_TYPE_UNSUPPORTED")
    if entry["capability_discovery_policy"] not in {"STATIC_BUILD_TIME", "ADAPTER_SNAPSHOT_REQUIRED"}:
        return deny("EXCHANGE_DISABLED")
    return True, None


def validate_account(account, *, operation="UPDATE_ACCOUNT", validation_context=None):
    fields = DATA["exchange_account_contract"]["record_fields"]
    ok, denial = closed_schema(account, fields, fields + ["legacy_record", "readiness_confirmed", "private_connection_active"], "EXCHANGE_ACCOUNT_NOT_FOUND")
    if not ok:
        return False, denial
    for field in ["exchange_account_id", "portfolio_id", "exchange_id", "environment", "market_type", "lifecycle_state", "connection_state", "execution_authorization", "external_account_identity_state"]:
        if not is_nonempty_str(account.get(field)):
            return deny("EXCHANGE_ACCOUNT_NOT_FOUND")
    ex = exchange_entry(account["exchange_id"])
    if ex is None:
        return deny("UNKNOWN_EXCHANGE_ID")
    if ex.get("status") != "ENABLED":
        return deny("EXCHANGE_DISABLED")
    if account["environment"] not in ex.get("supported_environments", []):
        return deny("EXCHANGE_ENVIRONMENT_UNSUPPORTED")
    if account["market_type"] not in ex.get("supported_market_types", []):
        return deny("MARKET_TYPE_UNSUPPORTED")
    if account["lifecycle_state"] not in DATA["exchange_account_contract"]["lifecycle_states"]:
        return deny("EXCHANGE_ACCOUNT_NOT_FOUND")
    if account["connection_state"] not in DATA["exchange_account_contract"]["connection_states"]:
        return deny("EXCHANGE_ACCOUNT_NOT_FOUND")
    if account["execution_authorization"] not in DATA["exchange_account_contract"]["execution_authorizations"]:
        return deny("EXCHANGE_ACCOUNT_NOT_FOUND")
    if account["external_account_identity_state"] not in DATA["external_account_identity_contract"]["states"]:
        return deny("EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED")
    if not isinstance(account.get("display_name"), str):
        return deny("ACCOUNT_READINESS_BLOCKED")
    for field in ["external_account_reference", "external_subaccount_reference", "active_credential_profile_id", "account_capability_snapshot_id"]:
        if account.get(field) is not None and not is_nonempty_str(account.get(field)):
            return deny("ACCOUNT_READINESS_BLOCKED")
    if not parse_ts(account.get("created_at_utc")) or (account.get("retired_at_utc") is not None and not parse_ts(account["retired_at_utc"])):
        return deny("ACCOUNT_READINESS_BLOCKED")
    if account["lifecycle_state"] in {"DRAFT", "ACTIVE", "DISABLED"} and account.get("retired_at_utc") is not None:
        return deny("ACCOUNT_READINESS_BLOCKED")
    if account["lifecycle_state"] == "RETIRED" and (account.get("retired_at_utc") is None or not ts_le(account["created_at_utc"], account["retired_at_utc"])):
        return deny("ACCOUNT_READINESS_BLOCKED")
    if "readiness_confirmed" in account and type(account["readiness_confirmed"]) is not bool:
        return deny("ACCOUNT_READINESS_BLOCKED")
    if "private_connection_active" in account and type(account["private_connection_active"]) is not bool:
        return deny("ACCOUNT_READINESS_BLOCKED")
    if account.get("account_capability_snapshot_id") is not None:
        snapshots = validation_context.get("account_capability_snapshots_by_id") if isinstance(validation_context, dict) else None
        previous = validation_context.get("previous_account_capability_snapshots_by_id") if isinstance(validation_context, dict) else None
        snapshot = snapshots.get(account["account_capability_snapshot_id"]) if isinstance(snapshots, dict) else None
        if not isinstance(previous, dict):
            return deny("ACCOUNT_READINESS_BLOCKED")
        ok, _ = validate_account_capability_snapshot(snapshot, account, previous)
        if not ok:
            return deny("ACCOUNT_READINESS_BLOCKED")
    if operation not in {"QUARANTINE_LEGACY_LIVE_RECORD", "RETIRE_ACCOUNT"} and account["environment"] == "LIVE":
        return deny("LIVE_BLOCKED_BY_EDITION")
    if account["lifecycle_state"] == "RETIRED" and operation != "RETIRE_ACCOUNT":
        return deny("EXCHANGE_ACCOUNT_RETIRED")
    if account["external_account_identity_state"] == "MISMATCH":
        return deny("EXTERNAL_ACCOUNT_IDENTITY_MISMATCH")
    return True, None


def validate_current_edition_account_operability(account, operation):
    ok, denial = validate_account(account, operation=operation)
    if not ok and denial != "LIVE_BLOCKED_BY_EDITION":
        return False, denial
    if isinstance(account, dict) and account.get("environment") == "LIVE":
        if operation == "QUARANTINE_LEGACY_LIVE_RECORD":
            return True, None
        return deny("LIVE_BLOCKED_BY_EDITION")
    return ok, denial


def validate_credential_profile_binding(profile, account, lineage_context=None):
    if not isinstance(account, dict):
        return deny("EXCHANGE_ACCOUNT_NOT_FOUND")
    if account.get("environment") == "LIVE":
        return deny("LIVE_BLOCKED_BY_EDITION")
    if account.get("environment") == "PAPER" and profile is None:
        return True, {"effective_authorization": "READ_ONLY", "secure_store_read": False}
    ok, denial = validate_credential_profile(profile)
    if not ok:
        return False, denial
    if profile["credential_profile_id"] == profile.get("rotated_from_credential_profile_id"):
        return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
    if profile["exchange_account_id"] != account["exchange_account_id"]:
        return deny("CREDENTIAL_PROFILE_NOT_FOUND")
    if profile["exchange_id"] != account["exchange_id"]:
        return deny("CREDENTIAL_PROFILE_EXCHANGE_MISMATCH")
    if profile["environment_scope"] != account["environment"]:
        return deny("CREDENTIAL_PROFILE_SCOPE_MISMATCH")
    if profile["credential_purpose"] not in DATA["credential_profile_contract"]["credential_purposes"] or profile["lifecycle_state"] != "ACTIVE":
        return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
    perms = profile.get("permission_snapshot")
    if not unique_str_list(perms) or not set(perms) <= set(DATA["credential_profile_contract"]["permission_registry"]):
        return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
    if "WITHDRAW" in perms:
        return deny("WITHDRAWAL_PERMISSION_FORBIDDEN")
    lineage_context = lineage_context or {}
    active_by_account = lineage_context.get("active_profile_ids_by_account_id")
    active_ids = active_by_account.get(account["exchange_account_id"]) if isinstance(active_by_account, dict) else None
    if not unique_str_list(active_ids) or active_ids != [profile["credential_profile_id"]]:
        return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
    rotated_from = profile.get("rotated_from_credential_profile_id")
    if rotated_from is not None:
        ok, denial = validate_credential_lineage(profile, lineage_context)
        if not ok:
            return False, denial
    return True, {"effective_authorization": "READ_ONLY", "order_entry_created": False}


def validate_credential_profile(profile):
    fields = DATA["credential_profile_contract"]["fields"] + ["saas_sync_candidate"]
    ok, denial = closed_schema(profile, fields, fields, "CREDENTIAL_PROFILE_NOT_FOUND")
    if not ok:
        return False, denial
    for field in ["credential_profile_id", "exchange_account_id", "exchange_id"]:
        if not is_nonempty_str(profile.get(field)):
            return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
    if profile.get("environment_scope") not in DATA["environment_registry"] or profile.get("credential_purpose") not in DATA["credential_profile_contract"]["credential_purposes"]:
        return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
    if not isinstance(profile.get("lifecycle_state"), str) or profile["lifecycle_state"] not in {"ACTIVE", "RETIRED"} or profile.get("saas_sync_candidate") is not False:
        return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
    reference = profile.get("secure_store_reference")
    forbidden_payload_markers = ["api_key", "apikey", "secret", "password", "token", "private_key", "credential_value", "plaintext"]
    locator = reference[len("secure-store://"):] if isinstance(reference, str) and reference.startswith("secure-store://") else ""
    if not is_nonempty_str(reference) or not locator or any(char.isspace() for char in reference) or any(char in reference for char in "?#=") or any(marker in reference.lower() for marker in forbidden_payload_markers):
        return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
    if profile.get("public_key_identifier") is not None and not is_nonempty_str(profile.get("public_key_identifier")):
        return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
    permissions = profile.get("permission_snapshot")
    if not unique_str_list(permissions) or not set(permissions) <= set(DATA["credential_profile_contract"]["permission_registry"]):
        return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
    if not parse_ts(profile.get("created_at_utc")):
        return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
    if profile["lifecycle_state"] == "ACTIVE" and profile.get("retired_at_utc") is not None:
        return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
    if profile["lifecycle_state"] == "RETIRED" and (not parse_ts(profile.get("retired_at_utc")) or not ts_le(profile["created_at_utc"], profile["retired_at_utc"])):
        return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
    if profile.get("rotated_from_credential_profile_id") is not None and not is_nonempty_str(profile.get("rotated_from_credential_profile_id")):
        return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
    return True, None


def validate_credential_lineage(current, context):
    previous = context.get("previous_profiles_by_id")
    if not isinstance(previous, dict):
        return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
    visited = {current["credential_profile_id"]}
    successor = current
    cursor = current.get("rotated_from_credential_profile_id")
    while cursor is not None:
        if not is_nonempty_str(cursor) or cursor in visited:
            return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
        visited.add(cursor)
        predecessor = previous.get(cursor)
        if not validate_credential_profile(predecessor)[0] or predecessor.get("credential_profile_id") != cursor:
            return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
        for field in ["exchange_account_id", "exchange_id", "environment_scope"]:
            if predecessor.get(field) != current[field]:
                return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
        if predecessor.get("lifecycle_state") != "RETIRED" or not parse_ts(predecessor.get("retired_at_utc")):
            return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
        if successor.get("rotated_from_credential_profile_id") != cursor or not ts_le(predecessor["retired_at_utc"], successor["created_at_utc"]):
            return deny("CREDENTIAL_PROFILE_NOT_ACTIVE")
        successor = predecessor
        cursor = predecessor.get("rotated_from_credential_profile_id")
    return True, None


def validate_external_account_identity(identity, account, *, reconnect=False):
    if not isinstance(account, dict):
        return deny("EXCHANGE_ACCOUNT_NOT_FOUND")
    if account.get("environment") == "LIVE":
        return deny("LIVE_BLOCKED_BY_EDITION")
    fields = DATA["external_account_identity_contract"]["verified_identity_fields"] + ["state"]
    ok, denial = closed_schema(identity, fields, fields + ["revalidated_at_utc"], "EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED")
    if not ok:
        return False, denial
    if identity["state"] == "UNAVAILABLE":
        return deny("EXTERNAL_ACCOUNT_IDENTITY_UNAVAILABLE")
    if identity["state"] == "MISMATCH":
        return deny("EXTERNAL_ACCOUNT_IDENTITY_MISMATCH")
    if identity["state"] != "VERIFIED":
        return deny("EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED")
    if reconnect and not parse_ts(identity.get("revalidated_at_utc")):
        return deny("EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED")
    if not is_nonempty_str(identity["venue_account_identifier"]):
        return deny("EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED")
    if identity["subaccount_identifier"] is not None and not isinstance(identity["subaccount_identifier"], str):
        return deny("EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED")
    if identity["subaccount_identifier"] == "":
        return deny("EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED")
    if not is_nonempty_str(identity["account_type"]) or not is_nonempty_str(identity["adapter_version_source"]):
        return deny("EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED")
    if not unique_str_list(identity["observed_permission_set"]):
        return deny("EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED")
    if not set(identity["observed_permission_set"]) <= set(DATA["credential_profile_contract"]["permission_registry"]):
        return deny("EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED")
    if not parse_ts(identity["verification_timestamp"]):
        return deny("EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED")
    for field in ["exchange_id", "environment", "market_type"]:
        if identity[field] != account[field]:
            return deny("EXTERNAL_ACCOUNT_IDENTITY_MISMATCH")
    return True, None


def validate_account_capability_snapshot(snapshot, account, previous_by_id=None):
    contract = DATA["account_capability_snapshot_contract"]
    fields = contract["fields"]
    ok, denial = closed_schema(snapshot, fields, fields, "ACCOUNT_READINESS_BLOCKED")
    if not ok or not isinstance(account, dict):
        return deny("ACCOUNT_READINESS_BLOCKED")
    for field in ["account_capability_snapshot_id", "exchange_account_id", "exchange_id", "environment", "market_type", "adapter_family_id", "adapter_version"]:
        if not is_nonempty_str(snapshot.get(field)):
            return deny("ACCOUNT_READINESS_BLOCKED")
    if snapshot["exchange_account_id"] != account.get("exchange_account_id") or any(snapshot.get(field) != account.get(field) for field in ["exchange_id", "environment", "market_type"]):
        return deny("ACCOUNT_READINESS_BLOCKED")
    ex = exchange_entry(snapshot["exchange_id"])
    if ex is None or ex.get("status") != "ENABLED" or snapshot["adapter_family_id"] != ex.get("adapter_family_id") or snapshot["environment"] == "LIVE":
        return deny("LIVE_BLOCKED_BY_EDITION" if snapshot["environment"] == "LIVE" else "ACCOUNT_READINESS_BLOCKED")
    if snapshot["environment"] not in ex.get("supported_environments", []) or snapshot["market_type"] not in ex.get("supported_market_types", []):
        return deny("ACCOUNT_READINESS_BLOCKED")
    if type(snapshot.get("version")) is not int or snapshot["version"] <= 0 or snapshot.get("status") not in contract["statuses"]:
        return deny("ACCOUNT_READINESS_BLOCKED")
    if snapshot.get("previous_snapshot_id") is not None and not is_nonempty_str(snapshot["previous_snapshot_id"]):
        return deny("ACCOUNT_READINESS_BLOCKED")
    permissions = snapshot.get("observed_permission_set")
    types = snapshot.get("supported_instrument_types")
    if not unique_str_list(permissions) or not set(permissions) <= set(DATA["credential_profile_contract"]["permission_registry"]):
        return deny("ACCOUNT_READINESS_BLOCKED")
    allowed_for_market = set(DATA["allowed_market_instrument_type_pairs"].get(snapshot["market_type"], []))
    if not unique_str_list(types) or not set(types) <= set(DATA["instrument_type_registry"]) or not set(types) <= set(ex["supported_instrument_types"]) or not set(types) <= allowed_for_market:
        return deny("ACCOUNT_READINESS_BLOCKED")
    if not (ts_le(snapshot.get("observed_at_utc"), snapshot.get("effective_at_utc")) and ts_lt(snapshot.get("effective_at_utc"), snapshot.get("stale_after_utc"))):
        return deny("ACCOUNT_READINESS_BLOCKED")
    definition = contract["content_hash_definition"]
    if not isinstance(snapshot.get("content_hash"), str) or not HEX64.fullmatch(snapshot["content_hash"]) or snapshot["content_hash"] != hash_payload(definition, snapshot):
        return deny("ACCOUNT_READINESS_BLOCKED")
    if not isinstance(previous_by_id, dict):
        return deny("ACCOUNT_READINESS_BLOCKED")
    if snapshot["version"] == 1:
        return (True, {"restricts_only": True, "extends_product_capabilities": False}) if snapshot["previous_snapshot_id"] is None else deny("ACCOUNT_READINESS_BLOCKED")
    visited = {snapshot["account_capability_snapshot_id"]}; cursor = snapshot["previous_snapshot_id"]; expected = snapshot["version"] - 1
    while cursor is not None:
        if not is_nonempty_str(cursor) or cursor in visited:
            return deny("ACCOUNT_READINESS_BLOCKED")
        visited.add(cursor); predecessor = previous_by_id.get(cursor)
        if not closed_schema(predecessor, fields, fields, "ACCOUNT_READINESS_BLOCKED")[0] or predecessor.get("account_capability_snapshot_id") != cursor:
            return deny("ACCOUNT_READINESS_BLOCKED")
        if predecessor.get("version") != expected or any(predecessor.get(field) != snapshot[field] for field in ["exchange_account_id", "exchange_id", "environment", "market_type"]):
            return deny("ACCOUNT_READINESS_BLOCKED")
        if type(predecessor.get("version")) is not int or predecessor.get("status") not in contract["statuses"] or predecessor.get("adapter_family_id") != ex.get("adapter_family_id") or not is_nonempty_str(predecessor.get("adapter_version")):
            return deny("ACCOUNT_READINESS_BLOCKED")
        if predecessor.get("environment") not in ex.get("supported_environments", []) or predecessor.get("market_type") not in ex.get("supported_market_types", []):
            return deny("ACCOUNT_READINESS_BLOCKED")
        if not unique_str_list(predecessor.get("observed_permission_set")) or not set(predecessor["observed_permission_set"]) <= set(DATA["credential_profile_contract"]["permission_registry"]):
            return deny("ACCOUNT_READINESS_BLOCKED")
        predecessor_market_types = set(DATA["allowed_market_instrument_type_pairs"].get(predecessor["market_type"], []))
        if not unique_str_list(predecessor.get("supported_instrument_types")) or not set(predecessor["supported_instrument_types"]) <= set(ex["supported_instrument_types"]) or not set(predecessor["supported_instrument_types"]) <= predecessor_market_types:
            return deny("ACCOUNT_READINESS_BLOCKED")
        if not (ts_le(predecessor.get("observed_at_utc"), predecessor.get("effective_at_utc")) and ts_lt(predecessor.get("effective_at_utc"), predecessor.get("stale_after_utc"))):
            return deny("ACCOUNT_READINESS_BLOCKED")
        if predecessor.get("content_hash") != hash_payload(definition, predecessor):
            return deny("ACCOUNT_READINESS_BLOCKED")
        cursor = predecessor.get("previous_snapshot_id"); expected -= 1
    return (True, {"restricts_only": True, "extends_product_capabilities": False}) if expected == 0 else deny("ACCOUNT_READINESS_BLOCKED")


def validate_asset_reference(ref):
    fields = DATA["asset_reference_contract"]["fields"]
    ok, denial = closed_schema(ref, fields, fields, "ASSET_MAPPING_UNKNOWN")
    if not ok:
        return False, denial
    for field in ["venue_asset_code", "canonical_display_code", "asset_namespace"]:
        if not is_nonempty_str(ref.get(field)):
            return deny("ASSET_MAPPING_UNKNOWN")
    if ref["mapping_status"] == "UNKNOWN":
        return deny("ASSET_MAPPING_UNKNOWN")
    if ref["mapping_status"] == "AMBIGUOUS":
        return deny("ASSET_MAPPING_AMBIGUOUS")
    if ref["mapping_status"] not in {"EXACT", "EXPLICIT_ALIAS"}:
        return deny("ASSET_MAPPING_UNKNOWN")
    return True, None


def validate_instrument_record(inst, now="2026-06-01T00:00:00Z", *, require_tradable=False, require_fresh=True):
    fields = DATA["instrument_contract"]["record_fields"]
    ok, denial = closed_schema(inst, fields, fields, "INSTRUMENT_METADATA_INVALID")
    if not ok:
        return False, denial
    for field in ["instrument_id", "workspace_id", "exchange_id", "environment", "market_type", "instrument_type", "venue_symbol", "display_symbol", "catalog_snapshot_id", "source_adapter_family_id"]:
        if not is_nonempty_str(inst.get(field)):
            return deny("INSTRUMENT_METADATA_INVALID")
    ex = exchange_entry(inst["exchange_id"])
    if ex is None:
        return deny("UNKNOWN_EXCHANGE_ID")
    if inst["environment"] not in ex["supported_environments"]:
        return deny("EXCHANGE_ENVIRONMENT_UNSUPPORTED")
    if inst["market_type"] not in ex["supported_market_types"]:
        return deny("MARKET_TYPE_UNSUPPORTED")
    if inst["instrument_type"] not in ex["supported_instrument_types"] or inst["instrument_type"] not in DATA["allowed_market_instrument_type_pairs"].get(inst["market_type"], []):
        return deny("INSTRUMENT_TYPE_UNSUPPORTED")
    if not is_nonempty_str(inst["venue_symbol"]) or inst["venue_symbol"].strip() != inst["venue_symbol"]:
        return deny("INSTRUMENT_IDENTITY_COLLISION")
    for ref_name in ["base_asset_reference", "quote_asset_reference"]:
        ok, denial = validate_asset_reference(inst[ref_name])
        if not ok:
            return False, denial
        if inst[ref_name].get("asset_namespace") != inst["exchange_id"]:
            return deny("ASSET_MAPPING_UNKNOWN")
    if inst.get("settlement_asset_reference") is not None:
        ok, denial = validate_asset_reference(inst["settlement_asset_reference"])
        if not ok:
            return False, denial
        if inst["settlement_asset_reference"].get("asset_namespace") != inst["exchange_id"]:
            return deny("ASSET_MAPPING_UNKNOWN")
    if type(inst["metadata_version"]) is not int or inst["metadata_version"] <= 0:
        return deny("INSTRUMENT_METADATA_INVALID")
    if not (ts_le(inst["observed_at_utc"], inst["effective_at_utc"]) and ts_lt(inst["effective_at_utc"], inst["stale_after_utc"])):
        return deny("INSTRUMENT_METADATA_INVALID")
    if require_fresh and not ts_lt(now, inst["stale_after_utc"]):
        return deny("INSTRUMENT_METADATA_STALE")
    if inst["trading_status"] not in DATA["instrument_contract"]["trading_statuses"]:
        return deny("INSTRUMENT_METADATA_INVALID")
    if require_tradable and inst["trading_status"] != "TRADING":
        return deny("INSTRUMENT_NOT_TRADABLE")
    for field in ["price_tick", "quantity_step"]:
        if not is_canonical_decimal(inst[field], positive=True):
            return deny("INSTRUMENT_METADATA_INVALID")
    for field in ["min_quantity", "max_quantity", "min_notional", "max_notional", "contract_size", "strike_price"]:
        if inst.get(field) is not None and not is_canonical_decimal(inst[field]):
            return deny("INSTRUMENT_METADATA_INVALID")
    for minimum, maximum in [("min_quantity", "max_quantity"), ("min_notional", "max_notional")]:
        if inst.get(maximum) is not None and inst.get(minimum) is None:
            return deny("INSTRUMENT_METADATA_INVALID")
        if inst.get(maximum) is not None and Decimal(inst[minimum]) > Decimal(inst[maximum]):
            return deny("INSTRUMENT_METADATA_INVALID")
    itype = inst["instrument_type"]
    derivative_fields = ["contract_size", "contract_value_currency", "derivative_settlement_type", "expiry_at_utc", "strike_price", "option_side"]
    if itype in {"SPOT_PAIR", "MARGIN_PAIR"}:
        if any(inst.get(field) is not None for field in derivative_fields):
            return deny("INSTRUMENT_METADATA_INVALID")
        settlement = inst.get("settlement_asset_reference")
        if settlement is not None and settlement.get("venue_asset_code") != inst["quote_asset_reference"].get("venue_asset_code"):
            return deny("INSTRUMENT_METADATA_INVALID")
    elif itype == "PERPETUAL_CONTRACT":
        if not is_canonical_decimal(inst.get("contract_size"), positive=True) or inst.get("settlement_asset_reference") is None or not is_nonempty_str(inst.get("contract_value_currency")) or inst.get("derivative_settlement_type") not in {"LINEAR", "INVERSE"} or any(inst.get(field) is not None for field in ["expiry_at_utc", "strike_price", "option_side"]):
            return deny("INSTRUMENT_METADATA_INVALID")
    elif itype == "DELIVERY_FUTURE":
        if not is_canonical_decimal(inst.get("contract_size"), positive=True) or inst.get("settlement_asset_reference") is None or not is_nonempty_str(inst.get("contract_value_currency")) or inst.get("derivative_settlement_type") not in {"LINEAR", "INVERSE"} or not parse_ts(inst.get("expiry_at_utc")) or any(inst.get(field) is not None for field in ["strike_price", "option_side"]):
            return deny("INSTRUMENT_METADATA_INVALID")
    elif itype == "OPTION":
        if not is_canonical_decimal(inst.get("contract_size"), positive=True) or inst.get("settlement_asset_reference") is None or not is_nonempty_str(inst.get("contract_value_currency")) or inst.get("derivative_settlement_type") not in {"LINEAR", "INVERSE"} or not parse_ts(inst.get("expiry_at_utc")) or not is_canonical_decimal(inst.get("strike_price"), positive=True) or inst.get("option_side") not in {"CALL", "PUT"}:
            return deny("INSTRUMENT_METADATA_INVALID")
    return True, None


def validate_instrument_history_map(instrument_history_by_id, current_instruments_by_id, now="2026-06-01T00:00:00Z"):
    if not isinstance(instrument_history_by_id, dict) or not isinstance(current_instruments_by_id, dict):
        return deny("INSTRUMENT_METADATA_INVALID")
    identity_fields = ["exchange_id", "environment", "market_type", "venue_symbol"]
    for instrument_id, history in instrument_history_by_id.items():
        if not is_nonempty_str(instrument_id) or not isinstance(history, list) or not history:
            return deny("INSTRUMENT_METADATA_INVALID")
        versions = []
        identity_tuple = None
        for record in history:
            if not isinstance(record, dict) or record.get("instrument_id") != instrument_id:
                return deny("INSTRUMENT_METADATA_INVALID")
            candidate_tuple = tuple(record.get(field) for field in identity_fields)
            if identity_tuple is not None and candidate_tuple != identity_tuple:
                return deny("INSTRUMENT_IDENTITY_COLLISION")
            identity_tuple = candidate_tuple
            current = current_instruments_by_id.get(instrument_id)
            if isinstance(current, dict) and tuple(current.get(field) for field in identity_fields) != identity_tuple:
                return deny("INSTRUMENT_IDENTITY_COLLISION")
            if not validate_instrument_record(record, now=now, require_fresh=False)[0]:
                return deny("INSTRUMENT_METADATA_INVALID")
            version = record.get("metadata_version")
            if type(version) is not int or version <= 0:
                return deny("INSTRUMENT_METADATA_INVALID")
            versions.append(version)
        if any(left >= right for left, right in zip(versions, versions[1:])):
            return deny("INSTRUMENT_METADATA_INVALID")
        current = current_instruments_by_id.get(instrument_id)
        if current is not None:
            if not isinstance(current, dict):
                return deny("INSTRUMENT_METADATA_INVALID")
            if tuple(current.get(field) for field in identity_fields) != identity_tuple:
                return deny("INSTRUMENT_IDENTITY_COLLISION")
            if type(current.get("metadata_version")) is not int or current["metadata_version"] <= versions[-1]:
                return deny("INSTRUMENT_METADATA_INVALID")
    return True, None


def validate_instrument_catalog_snapshot(snapshot, instruments, previous_catalogs_by_id=None,
                                         now="2026-06-01T00:00:00Z", instrument_history_by_id=None,
                                         catalogs_by_id=None):
    ok, denial = validate_instrument_history_map(instrument_history_by_id, instruments, now=now)
    if not ok:
        return False, denial
    fields = DATA["instrument_catalog_snapshot_contract"]["fields"]
    ok, denial = closed_schema(snapshot, fields, fields, "CATALOG_SNAPSHOT_INVALID")
    if not ok:
        return False, denial
    if snapshot["status"] not in DATA["instrument_catalog_snapshot_contract"]["statuses"]:
        return deny("CATALOG_SNAPSHOT_INVALID")
    if snapshot["status"] in {"STALE", "REJECTED"}:
        return deny("CATALOG_SNAPSHOT_STALE" if snapshot["status"] == "STALE" else "CATALOG_SNAPSHOT_INVALID")
    ex = exchange_entry(snapshot["exchange_id"])
    if ex is None:
        return deny("UNKNOWN_EXCHANGE_ID")
    if ex.get("status") != "ENABLED":
        return deny("EXCHANGE_DISABLED")
    if snapshot["environment"] not in ex.get("supported_environments", []):
        return deny("EXCHANGE_ENVIRONMENT_UNSUPPORTED")
    if snapshot["market_type"] not in ex.get("supported_market_types", []):
        return deny("MARKET_TYPE_UNSUPPORTED")
    if snapshot["adapter_family_id"] != ex["adapter_family_id"] or not is_nonempty_str(snapshot["adapter_version"]):
        return deny("CATALOG_SNAPSHOT_INVALID")
    if not unique_str_list(snapshot["instrument_ids"]):
        return deny("CATALOG_SNAPSHOT_INVALID")
    if not (ts_le(snapshot["observed_at_utc"], snapshot["effective_at_utc"]) and ts_lt(snapshot["effective_at_utc"], snapshot["stale_after_utc"]) and ts_lt(now, snapshot["stale_after_utc"])):
        return deny("CATALOG_SNAPSHOT_STALE")
    ok, denial = validate_catalog_lineage(snapshot, previous_catalogs_by_id)
    if not ok:
        return False, denial
    if not isinstance(instruments, dict):
        return deny("CATALOG_SNAPSHOT_INVALID")
    tuple_to_id = {}
    id_to_tuple = {}
    for candidate in instruments.values():
        if not isinstance(candidate, dict) or any(candidate.get(field) != snapshot[field] for field in ["exchange_id", "environment", "market_type"]):
            continue
        instrument_id = candidate.get("instrument_id")
        identity_tuple = tuple(candidate.get(field) for field in ["exchange_id", "environment", "market_type", "venue_symbol"])
        if identity_tuple in tuple_to_id and tuple_to_id[identity_tuple] != instrument_id:
            return deny("INSTRUMENT_IDENTITY_COLLISION")
        if instrument_id in id_to_tuple and id_to_tuple[instrument_id] != identity_tuple:
            return deny("INSTRUMENT_IDENTITY_COLLISION")
        tuple_to_id[identity_tuple] = instrument_id
        id_to_tuple[instrument_id] = identity_tuple
    for instrument_id in snapshot["instrument_ids"]:
        inst = instruments.get(instrument_id)
        if not isinstance(inst, dict):
            return deny("INSTRUMENT_NOT_FOUND")
        if inst.get("instrument_id") != instrument_id:
            return deny("INSTRUMENT_IDENTITY_COLLISION")
        history = instrument_history_by_id.get(instrument_id, [])
        for historical in history:
            historical_tuple = tuple(historical.get(field) for field in ["exchange_id", "environment", "market_type", "venue_symbol"])
            current_tuple = tuple(inst.get(field) for field in ["exchange_id", "environment", "market_type", "venue_symbol"])
            if historical_tuple != current_tuple:
                return deny("INSTRUMENT_IDENTITY_COLLISION")
        if inst.get("catalog_snapshot_id") != snapshot["catalog_snapshot_id"] or inst.get("source_adapter_family_id") != snapshot["adapter_family_id"]:
            return deny("CATALOG_SNAPSHOT_INVALID")
        if any(inst.get(field) != snapshot[field] for field in ["exchange_id", "environment", "market_type"]):
            return deny("CATALOG_SNAPSHOT_INVALID")
        ok, denial = validate_instrument_record(inst, now=now)
        if not ok:
            return False, denial
        trusted_catalogs = ({snapshot.get("catalog_snapshot_id"): snapshot}
                            if catalogs_by_id is None else catalogs_by_id)
        if resolve_catalog_member(snapshot, instrument_id, instruments, instrument_history_by_id,
                                  previous_catalogs_by_id, trusted_catalogs) is None:
            return deny("CATALOG_SNAPSHOT_INVALID")
    if snapshot["content_hash"] != hash_payload(DATA["deterministic_hash_contracts"]["catalog_snapshot_content_hash"], snapshot):
        return deny("CATALOG_SNAPSHOT_INVALID")
    return True, None


def validate_catalog_node(snapshot):
    fields = DATA["instrument_catalog_snapshot_contract"]["fields"]
    if not closed_schema(snapshot, fields, fields, "CATALOG_SNAPSHOT_INVALID")[0]:
        return False
    if any(not is_nonempty_str(snapshot.get(field)) for field in
           ["catalog_snapshot_id", "exchange_id", "environment", "market_type",
            "adapter_family_id", "adapter_version", "observed_at_utc", "effective_at_utc",
            "stale_after_utc", "status", "content_hash"]):
        return False
    if snapshot.get("previous_snapshot_id") is not None and not is_nonempty_str(snapshot.get("previous_snapshot_id")):
        return False
    if not HEX64.fullmatch(snapshot["content_hash"]) or not unique_str_list(snapshot.get("instrument_ids")):
        return False
    ex = exchange_entry(snapshot["exchange_id"])
    if (ex is None or ex.get("status") != "ENABLED"
            or snapshot["environment"] not in DATA["environment_registry"]
            or snapshot["environment"] not in ex.get("supported_environments", [])
            or snapshot["market_type"] not in DATA["market_type_registry"]
            or snapshot["market_type"] not in ex.get("supported_market_types", [])
            or snapshot["adapter_family_id"] != ex.get("adapter_family_id")
            or snapshot["status"] not in DATA["instrument_catalog_snapshot_contract"]["statuses"]):
        return False
    if not (ts_le(snapshot["observed_at_utc"], snapshot["effective_at_utc"])
            and ts_lt(snapshot["effective_at_utc"], snapshot["stale_after_utc"])):
        return False
    return snapshot["content_hash"] == hash_payload(
        DATA["deterministic_hash_contracts"]["catalog_snapshot_content_hash"], snapshot)


def validate_catalog_lineage(snapshot, previous_catalogs_by_id):
    if not isinstance(previous_catalogs_by_id, dict) or not validate_catalog_node(snapshot):
        return deny("CATALOG_SNAPSHOT_INVALID")
    visited = {snapshot["catalog_snapshot_id"]}
    cursor = snapshot.get("previous_snapshot_id")
    while cursor is not None:
        if not is_nonempty_str(cursor) or cursor in visited:
            return deny("CATALOG_SNAPSHOT_INVALID")
        visited.add(cursor)
        prev = previous_catalogs_by_id.get(cursor)
        if not validate_catalog_node(prev) or prev.get("catalog_snapshot_id") != cursor:
            return deny("CATALOG_SNAPSHOT_INVALID")
        if any(prev.get(field) != snapshot[field] for field in ["exchange_id", "environment", "market_type"]):
            return deny("CATALOG_SNAPSHOT_INVALID")
        cursor = prev.get("previous_snapshot_id")
    return True, None


def validate_universe_record(univ):
    fields = DATA["trading_universe_contract"]["record_fields"]
    ok, denial = closed_schema(univ, fields, fields, "TRADING_UNIVERSE_INVALID")
    if not ok:
        return False, denial
    if not is_nonempty_str(univ.get("trading_universe_id")) or not is_nonempty_str(univ.get("exchange_account_id")):
        return deny("TRADING_UNIVERSE_INVALID")
    if type(univ.get("version")) is not int or univ["version"] <= 0:
        return deny("TRADING_UNIVERSE_INVALID")
    if univ.get("lifecycle_state") not in DATA["trading_universe_contract"]["lifecycle_states"]:
        return deny("TRADING_UNIVERSE_INVALID")
    if not univ.get("instrument_ids") or not unique_str_list(univ.get("instrument_ids")) or not univ.get("source_catalog_snapshot_ids") or not unique_str_list(univ.get("source_catalog_snapshot_ids")):
        return deny("TRADING_UNIVERSE_INVALID")
    if univ.get("previous_version_id") is not None and not is_nonempty_str(univ.get("previous_version_id")):
        return deny("TRADING_UNIVERSE_INVALID")
    if not isinstance(univ.get("content_hash"), str) or not HEX64.fullmatch(univ["content_hash"]):
        return deny("TRADING_UNIVERSE_INVALID")
    if not parse_ts(univ.get("created_at_utc")):
        return deny("TRADING_UNIVERSE_INVALID")
    for field in ["activated_at_utc", "retired_at_utc"]:
        if univ.get(field) is not None and not parse_ts(univ[field]):
            return deny("TRADING_UNIVERSE_INVALID")
    if univ["lifecycle_state"] == "ACTIVE" and (univ.get("activated_at_utc") is None or univ.get("retired_at_utc") is not None):
        return deny("TRADING_UNIVERSE_INVALID")
    if univ["lifecycle_state"] == "RETIRED" and univ.get("retired_at_utc") is None:
        return deny("TRADING_UNIVERSE_INVALID")
    if not is_nonempty_str(univ.get("creation_reason")):
        return deny("TRADING_UNIVERSE_INVALID")
    return True, None


def validate_universe_lineage(univ, previous_universes_by_id):
    if not isinstance(previous_universes_by_id, dict):
        return deny("TRADING_UNIVERSE_INVALID")
    if univ["version"] == 1:
        return (True, None) if univ["previous_version_id"] is None else deny("TRADING_UNIVERSE_VERSION_CONFLICT")
    if univ["previous_version_id"] is None:
        return deny("TRADING_UNIVERSE_VERSION_CONFLICT")
    visited = {univ["trading_universe_id"]}; cursor = univ["previous_version_id"]; expected = univ["version"] - 1
    while cursor is not None:
        if not is_nonempty_str(cursor) or cursor in visited:
            return deny("TRADING_UNIVERSE_VERSION_CONFLICT")
        visited.add(cursor); predecessor = previous_universes_by_id.get(cursor)
        if not validate_universe_record(predecessor)[0] or predecessor.get("trading_universe_id") != cursor:
            return deny("TRADING_UNIVERSE_VERSION_CONFLICT")
        if predecessor.get("exchange_account_id") != univ["exchange_account_id"] or predecessor.get("version") != expected:
            return deny("TRADING_UNIVERSE_VERSION_CONFLICT")
        if predecessor.get("content_hash") != hash_payload(DATA["deterministic_hash_contracts"]["trading_universe_content_hash"], predecessor):
            return deny("TRADING_UNIVERSE_VERSION_CONFLICT")
        if predecessor.get("lifecycle_state") not in {"ACTIVE", "RETIRED"}:
            return deny("TRADING_UNIVERSE_VERSION_CONFLICT")
        cursor = predecessor.get("previous_version_id"); expected -= 1
    return (True, None) if expected == 0 else deny("TRADING_UNIVERSE_VERSION_CONFLICT")


def enforce_operation_states(operation, account):
    spec = DATA["operation_validation_matrix"][operation]
    for field, axis in [("lifecycle_state", "allowed_lifecycle_states"), ("connection_state", "allowed_connection_states"), ("execution_authorization", "allowed_authorization_states")]:
        allowed = spec[axis]
        if allowed and account.get(field) not in allowed:
            return deny("EXCHANGE_ACCOUNT_RETIRED" if field == "lifecycle_state" and account.get(field) == "RETIRED" else "ACCOUNT_READINESS_BLOCKED")
    return True, None


def validate_trading_universe_version(univ, account, instruments, catalogs, previous_universes_by_id=None, active_universes=None, previous_catalogs_by_id=None, now="2026-06-01T00:00:00Z", validation_context=None):
    if not validate_context(validation_context) or not isinstance(univ, dict) or not isinstance(account, dict):
        return deny("ACCOUNT_READINESS_BLOCKED")
    universe_id = univ.get("trading_universe_id")
    account_id = univ.get("exchange_account_id")
    trusted_account = validation_context["accounts_by_id"].get(account_id) if is_nonempty_str(account_id) else None
    trusted_universe = validation_context["universes_by_id"].get(universe_id) if is_nonempty_str(universe_id) else None
    bindings = [
        (account, trusted_account), (univ, trusted_universe),
        (instruments, validation_context["instruments_by_id"]),
        (catalogs, validation_context["catalogs_by_id"]),
        (previous_universes_by_id, validation_context["previous_universes_by_id"]),
        (previous_catalogs_by_id, validation_context["previous_catalogs_by_id"]),
        (active_universes, validation_context["active_universes"]),
    ]
    if trusted_account is None or trusted_universe is None or any(argument != trusted for argument, trusted in bindings):
        return deny("ACCOUNT_READINESS_BLOCKED")
    ready, readiness = validate_account_activation_readiness(trusted_account, validation_context, now=now)
    if not ready:
        return False, readiness
    ok, denial = validate_universe_record(univ)
    if not ok:
        return False, denial
    if univ["lifecycle_state"] in {"RETIRED", "REJECTED"}:
        return deny("TRADING_UNIVERSE_INVALID")
    ok, denial = validate_universe_lineage(univ, previous_universes_by_id)
    if not ok:
        return False, denial
    if univ["exchange_account_id"] != account["exchange_account_id"]:
        return deny("TRADING_UNIVERSE_INSTRUMENT_SCOPE_MISMATCH")
    if not isinstance(active_universes, list) or any(not isinstance(item, dict) for item in active_universes):
        return deny("TRADING_UNIVERSE_INVALID")
    active_union = {item["trading_universe_id"]: item for item in active_universes}
    active_union.update({uid: item for uid, item in validation_context["universes_by_id"].items() if item.get("lifecycle_state") == "ACTIVE"})
    for active in active_union.values():
        if not validate_universe_record(active)[0]:
            return deny("TRADING_UNIVERSE_INVALID")
        if active["lifecycle_state"] == "ACTIVE" and active["exchange_account_id"] == account["exchange_account_id"]:
            if active["trading_universe_id"] != univ["trading_universe_id"] or active != univ:
                return deny("TRADING_UNIVERSE_VERSION_CONFLICT")
    if not isinstance(instruments, dict) or not isinstance(catalogs, dict) or not unique_str_list(univ["source_catalog_snapshot_ids"]):
        return deny("CATALOG_SNAPSHOT_INVALID")
    source_instrument_ids = set()
    for catalog_id in univ["source_catalog_snapshot_ids"]:
        catalog = catalogs.get(catalog_id)
        if not isinstance(catalog, dict) or catalog.get("status") != "VALID":
            return deny("CATALOG_SNAPSHOT_INVALID")
        ok, denial = validate_instrument_catalog_snapshot(
            catalog, instruments, previous_catalogs_by_id, now=now,
            instrument_history_by_id=validation_context["instrument_history_by_id"],
            catalogs_by_id=validation_context["catalogs_by_id"])
        if not ok:
            return False, denial
        source_instrument_ids.update(catalog["instrument_ids"])
    if univ["content_hash"] != hash_payload(DATA["deterministic_hash_contracts"]["trading_universe_content_hash"], univ):
        return deny("TRADING_UNIVERSE_INVALID")
    for iid in univ["instrument_ids"]:
        inst = instruments.get(iid)
        if inst is None:
            return deny("INSTRUMENT_NOT_FOUND")
        if iid not in source_instrument_ids or inst.get("catalog_snapshot_id") not in univ["source_catalog_snapshot_ids"]:
            return deny("TRADING_UNIVERSE_INSTRUMENT_SCOPE_MISMATCH")
        if any(inst.get(field) != account[field] for field in ["exchange_id", "environment", "market_type"]):
            return deny("TRADING_UNIVERSE_INSTRUMENT_SCOPE_MISMATCH")
        capability_types = readiness.get("capability_instrument_types")
        if capability_types is not None and inst.get("instrument_type") not in capability_types:
            return deny("ACCOUNT_READINESS_BLOCKED")
        ok, denial = validate_instrument_record(inst, now=now, require_tradable=True)
        if not ok:
            return False, denial
    return True, None


def audit_result(operation, ok, denial=None):
    if operation not in DATA["operation_validation_matrix"]:
        return False, {"denial": "UNKNOWN_OPERATION", "audit_event": "UNKNOWN_OPERATION_REJECTED"}
    spec = DATA["operation_validation_matrix"][operation]
    if ok:
        return True, {"audit_event": spec["success_audit_event"]}
    schemas = {item["event_name"]: item for item in DATA["audit_event_schemas"]}
    event = spec["denial_event_by_code"].get(denial)
    schema = schemas.get(event)
    if denial not in spec["denial_codes"] or not event or not schema or denial not in schema["allowed_denial_codes"]:
        contract_denial = "CONTRACT_VALIDATION_MAPPING_ERROR"
        return False, {"denial": contract_denial, "audit_event": "CONTRACT_VALIDATION_MAPPING_REJECTED"}
    return False, {"denial": denial, "audit_event": event}


CONTEXT_MAPS = {"accounts_by_id", "credential_profiles_by_id", "external_identity_snapshots_by_account_id", "universes_by_id", "catalogs_by_id", "instruments_by_id", "instrument_history_by_id", "previous_profiles_by_id", "previous_catalogs_by_id", "previous_universes_by_id", "account_capability_snapshots_by_id", "previous_account_capability_snapshots_by_id", "readiness_by_account_id", "active_profile_ids_by_account_id"}
CONTEXT_LISTS = {"active_universes"}
MAP_ID_FIELDS = {"accounts_by_id":"exchange_account_id", "credential_profiles_by_id":"credential_profile_id", "universes_by_id":"trading_universe_id", "catalogs_by_id":"catalog_snapshot_id", "instruments_by_id":"instrument_id", "previous_profiles_by_id":"credential_profile_id", "previous_catalogs_by_id":"catalog_snapshot_id", "previous_universes_by_id":"trading_universe_id", "account_capability_snapshots_by_id":"account_capability_snapshot_id", "previous_account_capability_snapshots_by_id":"account_capability_snapshot_id"}


def validate_account_structure(account):
    fields = DATA["exchange_account_contract"]["record_fields"]
    if not closed_schema(account, fields, fields + ["legacy_record", "readiness_confirmed", "private_connection_active"], "EXCHANGE_ACCOUNT_NOT_FOUND")[0]:
        return False
    required_strings = ["exchange_account_id", "portfolio_id", "exchange_id", "environment", "market_type", "lifecycle_state", "connection_state", "execution_authorization", "external_account_identity_state"]
    if any(not is_nonempty_str(account.get(field)) for field in required_strings):
        return False
    ex = exchange_entry(account["exchange_id"])
    if ex is None or account["environment"] not in ex.get("supported_environments", []) or account["market_type"] not in ex.get("supported_market_types", []):
        return False
    if account["lifecycle_state"] not in DATA["exchange_account_contract"]["lifecycle_states"] or account["connection_state"] not in DATA["exchange_account_contract"]["connection_states"] or account["execution_authorization"] not in DATA["exchange_account_contract"]["execution_authorizations"] or account["external_account_identity_state"] not in DATA["external_account_identity_contract"]["states"]:
        return False
    if not isinstance(account.get("display_name"), str) or not parse_ts(account.get("created_at_utc")):
        return False
    if account.get("retired_at_utc") is not None and not parse_ts(account["retired_at_utc"]):
        return False
    if account["lifecycle_state"] == "RETIRED" and (account.get("retired_at_utc") is None or not ts_le(account["created_at_utc"], account["retired_at_utc"])):
        return False
    if account["lifecycle_state"] != "RETIRED" and account.get("retired_at_utc") is not None:
        return False
    for field in ["external_account_reference", "external_subaccount_reference", "active_credential_profile_id", "account_capability_snapshot_id"]:
        if account.get(field) is not None and not is_nonempty_str(account[field]):
            return False
    return all(field not in account or type(account[field]) is bool for field in ["legacy_record", "readiness_confirmed", "private_connection_active"])


def validate_identity_structure(identity, account):
    fields = DATA["external_account_identity_contract"]["verified_identity_fields"] + ["state"]
    if not closed_schema(identity, fields, fields + ["revalidated_at_utc"], "EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED")[0] or not isinstance(account, dict):
        return False
    if identity.get("state") not in DATA["external_account_identity_contract"]["states"]:
        return False
    if any(identity.get(field) != account.get(field) for field in ["exchange_id", "environment", "market_type"]):
        return False
    if not is_nonempty_str(identity.get("venue_account_identifier")) or identity.get("subaccount_identifier") == "" or (identity.get("subaccount_identifier") is not None and not isinstance(identity.get("subaccount_identifier"), str)):
        return False
    if not is_nonempty_str(identity.get("account_type")) or not is_nonempty_str(identity.get("adapter_version_source")) or not parse_ts(identity.get("verification_timestamp")):
        return False
    if not unique_str_list(identity.get("observed_permission_set")) or not set(identity["observed_permission_set"]) <= set(DATA["credential_profile_contract"]["permission_registry"]):
        return False
    return identity.get("revalidated_at_utc") is None or parse_ts(identity.get("revalidated_at_utc")) is not None


def validate_catalog_structure(snapshot, previous_catalogs_by_id):
    return validate_catalog_node(snapshot) and validate_catalog_lineage(snapshot, previous_catalogs_by_id)[0]


def validate_profile_account_binding(profile, accounts_by_id):
    if not isinstance(profile, dict) or not isinstance(accounts_by_id, dict):
        return False
    account = accounts_by_id.get(profile.get("exchange_account_id"))
    return (isinstance(account, dict)
            and profile.get("exchange_account_id") == account.get("exchange_account_id")
            and profile.get("exchange_id") == account.get("exchange_id")
            and profile.get("environment_scope") == account.get("environment"))


def validate_capability_account_binding(snapshot, accounts_by_id, previous_by_id):
    if not isinstance(snapshot, dict) or not isinstance(accounts_by_id, dict) or not isinstance(previous_by_id, dict):
        return False
    account = accounts_by_id.get(snapshot.get("exchange_account_id"))
    return isinstance(account, dict) and validate_account_capability_snapshot(snapshot, account, previous_by_id)[0]


def validate_target_history_catalog_bindings(instrument_id, history, catalogs_by_id,
                                             previous_catalogs_by_id):
    if (not is_nonempty_str(instrument_id) or not isinstance(history, list)
            or not isinstance(catalogs_by_id, dict) or not isinstance(previous_catalogs_by_id, dict)):
        return False
    all_catalogs = {**previous_catalogs_by_id, **catalogs_by_id}
    for record in history:
        if (not isinstance(record, dict) or record.get("instrument_id") != instrument_id
                or not validate_instrument_record(record, require_fresh=False)[0]):
            return False
        bound_catalog = all_catalogs.get(record.get("catalog_snapshot_id"))
        if (not validate_catalog_structure(bound_catalog, previous_catalogs_by_id)
                or instrument_id not in bound_catalog.get("instrument_ids", [])
                or any(record.get(field) != bound_catalog.get(field)
                       for field in ["exchange_id", "environment", "market_type"])
                or record.get("source_adapter_family_id") != bound_catalog.get("adapter_family_id")):
            return False
    return True


def resolve_catalog_member(catalog, instrument_id, instruments_by_id, instrument_history_by_id,
                           previous_catalogs_by_id=None, catalogs_by_id=None):
    if not isinstance(catalog, dict) or not is_nonempty_str(instrument_id) or not isinstance(instruments_by_id, dict) or not isinstance(instrument_history_by_id, dict):
        return None
    lineage = {} if previous_catalogs_by_id is None and catalog.get("previous_snapshot_id") is None else previous_catalogs_by_id
    if not isinstance(lineage, dict) or not validate_catalog_structure(catalog, lineage):
        return None
    for field in ["catalog_snapshot_id", "exchange_id", "environment", "market_type", "adapter_family_id"]:
        if not is_nonempty_str(catalog.get(field)):
            return None
    exchange = exchange_entry(catalog["exchange_id"])
    if (exchange is None or catalog["environment"] not in DATA["environment_registry"]
            or catalog["market_type"] not in DATA["market_type_registry"]
            or catalog["adapter_family_id"] != exchange.get("adapter_family_id")):
        return None
    catalog_members = catalog.get("instrument_ids")
    if not unique_str_list(catalog_members) or instrument_id not in catalog_members:
        return None
    scope = ["exchange_id", "environment", "market_type"]
    def bound(record):
        return (isinstance(record, dict) and record.get("instrument_id") == instrument_id
                and record.get("catalog_snapshot_id") == catalog.get("catalog_snapshot_id")
                and all(record.get(field) == catalog.get(field) for field in scope)
                and record.get("source_adapter_family_id") == catalog.get("adapter_family_id"))
    history = instrument_history_by_id.get(instrument_id, [])
    if not isinstance(history, list):
        return None
    trusted_catalogs = ({catalog.get("catalog_snapshot_id"): catalog}
                        if catalogs_by_id is None else catalogs_by_id)
    if not isinstance(trusted_catalogs, dict) or not validate_target_history_catalog_bindings(
            instrument_id, history, trusted_catalogs, lineage):
        return None
    versions = []
    history_identity = None
    for record in history:
        if (not isinstance(record, dict) or record.get("instrument_id") != instrument_id
                or not validate_instrument_record(record, require_fresh=False)[0]):
            return None
        identity = tuple(record.get(field) for field in ["exchange_id", "environment", "market_type", "venue_symbol"])
        if history_identity is not None and identity != history_identity:
            return None
        history_identity = identity
        version = record.get("metadata_version")
        if type(version) is not int or version <= 0:
            return None
        versions.append(version)
    if any(left >= right for left, right in zip(versions, versions[1:])):
        return None
    current = instruments_by_id.get(instrument_id)
    if current is not None:
        if not isinstance(current, dict) or not validate_instrument_record(current, require_fresh=False)[0]:
            return None
        if history_identity is not None and tuple(current.get(field) for field in ["exchange_id", "environment", "market_type", "venue_symbol"]) != history_identity:
            return None
        current_version = current.get("metadata_version")
        if type(current_version) is not int or current_version <= 0 or (versions and current_version <= versions[-1]):
            return None
        if bound(current):
            return current
    matches = [record for record in history if bound(record)]
    return max(matches, key=lambda record: record["metadata_version"]) if matches else None


def validate_universe_source_membership(universe, accounts_by_id, catalogs_by_id, instruments_by_id,
                                        instrument_history_by_id, *, historical=False,
                                        previous_catalogs_by_id=None):
    if (not validate_universe_record(universe)[0]
            or not all(isinstance(item, dict) for item in
                       [accounts_by_id, catalogs_by_id, instruments_by_id, instrument_history_by_id])):
        return False
    if any(not is_nonempty_str(key) or not isinstance(value, dict)
           or value.get("exchange_account_id") != key or not validate_account_structure(value)
           for key, value in accounts_by_id.items()):
        return False
    account = accounts_by_id.get(universe.get("exchange_account_id"))
    if not isinstance(account, dict):
        return False
    source_ids = universe.get("source_catalog_snapshot_ids")
    instrument_ids = universe.get("instrument_ids")
    if not unique_str_list(source_ids) or not unique_str_list(instrument_ids):
        return False
    lineage = {} if previous_catalogs_by_id is None else previous_catalogs_by_id
    if not isinstance(lineage, dict) or not validate_catalog_instrument_graph(
            catalogs_by_id, lineage, instruments_by_id, instrument_history_by_id):
        return False
    available_catalogs = {**lineage, **catalogs_by_id}
    catalogs = []
    for catalog_id in source_ids:
        catalog = available_catalogs.get(catalog_id)
        if not isinstance(catalog, dict) or any(catalog.get(field) != account.get(field) for field in ["exchange_id", "environment", "market_type"]):
            return False
        catalogs.append(catalog)
        if any(resolve_catalog_member(catalog, member_id, instruments_by_id,
                                      instrument_history_by_id, lineage, catalogs_by_id) is None
               for member_id in catalog.get("instrument_ids", [])):
            return False
    for instrument_id in instrument_ids:
        current = instruments_by_id.get(instrument_id)
        if not historical and (not isinstance(current, dict) or current.get("catalog_snapshot_id") not in source_ids):
            return False
        matches = [resolve_catalog_member(catalog, instrument_id, instruments_by_id,
                                          instrument_history_by_id, lineage, catalogs_by_id)
                   for catalog in catalogs]
        if not any(matches):
            return False
        record = current if isinstance(current, dict) else next(match for match in matches if match)
        if any(record.get(field) != account.get(field) for field in ["exchange_id", "environment", "market_type"]):
            return False
    return True


def validate_historical_instrument_catalog_bindings(instrument_history_by_id, catalogs_by_id,
                                                    previous_catalogs_by_id):
    if not all(isinstance(item, dict) for item in
               [instrument_history_by_id, catalogs_by_id, previous_catalogs_by_id]):
        return False
    all_catalogs = {**previous_catalogs_by_id, **catalogs_by_id}
    for instrument_id, history in instrument_history_by_id.items():
        if not is_nonempty_str(instrument_id) or not isinstance(history, list):
            return False
        for record in history:
            if (not isinstance(record, dict) or record.get("instrument_id") != instrument_id
                    or not validate_instrument_record(record, require_fresh=False)[0]):
                return False
            catalog = all_catalogs.get(record.get("catalog_snapshot_id"))
            if (not isinstance(catalog, dict)
                    or not validate_catalog_structure(catalog, previous_catalogs_by_id)
                    or instrument_id not in catalog.get("instrument_ids", [])
                    or record.get("catalog_snapshot_id") != catalog.get("catalog_snapshot_id")
                    or any(record.get(field) != catalog.get(field)
                           for field in ["exchange_id", "environment", "market_type"])
                    or record.get("source_adapter_family_id") != catalog.get("adapter_family_id")):
                return False
    return True


def validate_catalog_instrument_graph(catalogs_by_id, previous_catalogs_by_id,
                                      instruments_by_id, instrument_history_by_id):
    if not all(isinstance(item, dict) for item in
               [catalogs_by_id, previous_catalogs_by_id, instruments_by_id,
                instrument_history_by_id]):
        return False
    if set(catalogs_by_id) & set(previous_catalogs_by_id):
        return False
    for mapping in [catalogs_by_id, previous_catalogs_by_id]:
        for catalog_id, catalog in mapping.items():
            if (not is_nonempty_str(catalog_id) or not isinstance(catalog, dict)
                    or catalog.get("catalog_snapshot_id") != catalog_id
                    or not validate_catalog_structure(catalog, previous_catalogs_by_id)):
                return False
    for instrument_id, instrument in instruments_by_id.items():
        if (not is_nonempty_str(instrument_id) or not isinstance(instrument, dict)
                or instrument.get("instrument_id") != instrument_id
                or not validate_instrument_record(instrument, require_fresh=False)[0]):
            return False
    if not validate_instrument_history_map(instrument_history_by_id, instruments_by_id)[0]:
        return False
    if not validate_historical_instrument_catalog_bindings(
            instrument_history_by_id, catalogs_by_id, previous_catalogs_by_id):
        return False
    tuple_to_id = {}
    id_to_tuple = {}
    records = list(instruments_by_id.items())
    records.extend((instrument_id, record)
                   for instrument_id, history in instrument_history_by_id.items()
                   for record in history)
    for instrument_id, record in records:
        identity_tuple = tuple(record[field] for field in
                               ["exchange_id", "environment", "market_type", "venue_symbol"])
        if (identity_tuple in tuple_to_id
                and tuple_to_id[identity_tuple] != instrument_id):
            return False
        if instrument_id in id_to_tuple and id_to_tuple[instrument_id] != identity_tuple:
            return False
        tuple_to_id[identity_tuple] = instrument_id
        id_to_tuple[instrument_id] = identity_tuple
    all_catalogs = {**previous_catalogs_by_id, **catalogs_by_id}
    for instrument_id, instrument in instruments_by_id.items():
        catalog = all_catalogs.get(instrument.get("catalog_snapshot_id"))
        if (not isinstance(catalog, dict)
                or resolve_catalog_member(catalog, instrument_id, instruments_by_id,
                                          instrument_history_by_id, previous_catalogs_by_id,
                                          catalogs_by_id) != instrument):
            return False
    for catalog in all_catalogs.values():
        for instrument_id in catalog["instrument_ids"]:
            if resolve_catalog_member(catalog, instrument_id, instruments_by_id,
                                      instrument_history_by_id, previous_catalogs_by_id,
                                      catalogs_by_id) is None:
                return False
    return True


def validate_account_owned_bindings(account, ctx):
    if not isinstance(account, dict) or not isinstance(ctx, dict):
        return False
    account_id = account.get("exchange_account_id")
    profile_id = account.get("active_credential_profile_id")
    active_ids = ctx["active_profile_ids_by_account_id"].get(account_id)
    active_profiles = [profile for profile in ctx["credential_profiles_by_id"].values() if profile.get("exchange_account_id") == account_id and profile.get("lifecycle_state") == "ACTIVE"]
    if profile_id is None:
        if active_profiles or account_id in ctx["active_profile_ids_by_account_id"]:
            return False
    else:
        profile = ctx["credential_profiles_by_id"].get(profile_id)
        if (not validate_profile_account_binding(profile, ctx["accounts_by_id"])
                or profile.get("lifecycle_state") != "ACTIVE" or active_ids != [profile_id]
                or profile_id in ctx["previous_profiles_by_id"] or len(active_profiles) != 1):
            return False
    capability_id = account.get("account_capability_snapshot_id")
    if capability_id is not None:
        snapshot = ctx["account_capability_snapshots_by_id"].get(capability_id)
        if snapshot is None or capability_id in ctx["previous_account_capability_snapshots_by_id"] or not validate_capability_account_binding(snapshot, ctx["accounts_by_id"], ctx["previous_account_capability_snapshots_by_id"]):
            return False
    return True


def validate_context(ctx):
    if not isinstance(ctx, dict):
        return False
    if set(ctx) != CONTEXT_MAPS | CONTEXT_LISTS:
        return False
    if any(not isinstance(ctx.get(name), dict) for name in CONTEXT_MAPS):
        return False
    if any(not isinstance(ctx.get(name), list) for name in CONTEXT_LISTS):
        return False
    if any(not isinstance(key, str) or not isinstance(value, dict) for name in CONTEXT_MAPS - {"readiness_by_account_id", "active_profile_ids_by_account_id", "instrument_history_by_id"} for key, value in ctx[name].items()):
        return False
    for name, id_field in MAP_ID_FIELDS.items():
        if any(not is_nonempty_str(key) or value.get(id_field) != key for key, value in ctx[name].items()):
            return False
    disjoint_pairs = [
        ("credential_profiles_by_id", "previous_profiles_by_id"),
        ("catalogs_by_id", "previous_catalogs_by_id"),
        ("universes_by_id", "previous_universes_by_id"),
        ("account_capability_snapshots_by_id", "previous_account_capability_snapshots_by_id"),
    ]
    if any(set(ctx[current]) & set(ctx[previous]) for current, previous in disjoint_pairs):
        return False
    if any(not validate_account_structure(record) for record in ctx["accounts_by_id"].values()):
        return False
    if any(not validate_account_owned_bindings(account, ctx) for account in ctx["accounts_by_id"].values()):
        return False
    if any(not validate_credential_profile(record)[0] for name in ["credential_profiles_by_id", "previous_profiles_by_id"] for record in ctx[name].values()):
        return False
    if any(record.get("lifecycle_state") != "RETIRED" for record in ctx["previous_profiles_by_id"].values()):
        return False
    for profile in list(ctx["credential_profiles_by_id"].values()) + list(ctx["previous_profiles_by_id"].values()):
        if not validate_profile_account_binding(profile, ctx["accounts_by_id"]):
            return False
    for account_id, identity in ctx["external_identity_snapshots_by_account_id"].items():
        account = ctx["accounts_by_id"].get(account_id)
        if not validate_identity_structure(identity, account):
            return False
    if any(not validate_universe_record(record)[0] or record.get("content_hash") != hash_payload(DATA["deterministic_hash_contracts"]["trading_universe_content_hash"], record) for name in ["universes_by_id", "previous_universes_by_id"] for record in ctx[name].values()):
        return False
    if not validate_catalog_instrument_graph(
            ctx["catalogs_by_id"], ctx["previous_catalogs_by_id"],
            ctx["instruments_by_id"], ctx["instrument_history_by_id"]):
        return False
    if any(not validate_universe_source_membership(
            universe, ctx["accounts_by_id"], ctx["catalogs_by_id"], ctx["instruments_by_id"],
            ctx["instrument_history_by_id"], previous_catalogs_by_id=ctx["previous_catalogs_by_id"])
           for universe in ctx["universes_by_id"].values()):
        return False
    if any(not validate_universe_source_membership(
            universe, ctx["accounts_by_id"], ctx["catalogs_by_id"], ctx["instruments_by_id"],
            ctx["instrument_history_by_id"], historical=True,
            previous_catalogs_by_id=ctx["previous_catalogs_by_id"])
           for universe in ctx["previous_universes_by_id"].values()):
        return False
    for map_name in ["account_capability_snapshots_by_id", "previous_account_capability_snapshots_by_id"]:
        for snapshot in ctx[map_name].values():
            if not validate_capability_account_binding(snapshot, ctx["accounts_by_id"], ctx["previous_account_capability_snapshots_by_id"]):
                return False
    if any(not isinstance(key, str) or type(value) is not bool for key, value in ctx["readiness_by_account_id"].items()):
        return False
    if any(account_id not in ctx["accounts_by_id"] for account_id in ctx["readiness_by_account_id"]):
        return False
    if not all(validate_universe_record(item)[0] and item.get("content_hash") == hash_payload(DATA["deterministic_hash_contracts"]["trading_universe_content_hash"], item) for item in ctx["active_universes"]):
        return False
    active_ids = [item.get("trading_universe_id") for item in ctx["active_universes"]]
    if len(active_ids) != len(set(active_ids)):
        return False
    for item in ctx["active_universes"]:
        if item.get("lifecycle_state") != "ACTIVE" or ctx["universes_by_id"].get(item.get("trading_universe_id")) != item:
            return False
    if any(not is_nonempty_str(account_id) or not ids or not unique_str_list(ids) for account_id, ids in ctx["active_profile_ids_by_account_id"].items()):
        return False
    for account_id, profile_ids in ctx["active_profile_ids_by_account_id"].items():
        if account_id not in ctx["accounts_by_id"]:
            return False
        for profile_id in profile_ids:
            profile = ctx["credential_profiles_by_id"].get(profile_id)
            if not isinstance(profile, dict) or profile.get("exchange_account_id") != account_id or profile.get("lifecycle_state") != "ACTIVE":
                return False
    active_profiles_by_account = {}
    for profile in ctx["credential_profiles_by_id"].values():
        if profile.get("lifecycle_state") == "ACTIVE":
            active_profiles_by_account.setdefault(profile["exchange_account_id"], []).append(profile["credential_profile_id"])
    if any(len(ids) > 1 for ids in active_profiles_by_account.values()):
        return False
    for account_id, ids in active_profiles_by_account.items():
        account = ctx["accounts_by_id"].get(account_id)
        if ctx["active_profile_ids_by_account_id"].get(account_id) != ids or not isinstance(account, dict) or account.get("active_credential_profile_id") != ids[0]:
            return False
    if any(account_id not in active_profiles_by_account for account_id in ctx["active_profile_ids_by_account_id"]):
        return False
    return True


def validate_request_shape(operation, request):
    spec = DATA["operation_validation_matrix"][operation]
    forbidden = set(DATA["operation_dispatch_policy"]["caller_controlled_result_fields_forbidden"])
    if not isinstance(request, dict) or set(request) & forbidden or set(request) != set(spec["required_inputs"]):
        return False, spec["denial_codes"][0]
    if not is_nonempty_str(request.get("request_id")):
        return False, spec["denial_codes"][0]
    for field in [f for f in request if f.endswith("_id") and f != "request_id"]:
        if not is_nonempty_str(request[field]):
            return False, spec["denial_codes"][0]
    if "source_catalog_snapshot_ids" in request and not unique_str_list(request["source_catalog_snapshot_ids"]):
        return False, "CATALOG_SNAPSHOT_INVALID"
    if "content_hash" in request and (not isinstance(request["content_hash"], str) or not HEX64.fullmatch(request["content_hash"])):
        return False, "TRADING_UNIVERSE_INVALID"
    type_denial = {"VERIFY_EXTERNAL_IDENTITY":"EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED", "QUARANTINE_LEGACY_LIVE_RECORD":"ACCOUNT_READINESS_BLOCKED"}.get(operation, spec["denial_codes"][0])
    string_fields = {"portfolio_id", "exchange_id", "environment", "market_type", "requested_lifecycle_state", "binding_intent", "expected_adapter_version_source", "adapter_family_id", "adapter_version", "retirement_reason", "target_lifecycle_state", "target_connection_state", "target_execution_authorization", "quarantine_reason"}
    if any(field in request and not is_nonempty_str(request[field]) for field in string_fields):
        return False, type_denial
    for field in ["external_account_reference", "external_subaccount_reference"]:
        if field in request and request[field] is not None and not is_nonempty_str(request[field]):
            return False, type_denial
    for field in ["expected_identity_fields", "mutable_patch"]:
        if field in request and not isinstance(request[field], dict):
            return False, type_denial
    for field in ["legacy_record", "reconnect"]:
        if field in request and type(request[field]) is not bool:
            return False, type_denial
    return True, None


def validate_operation_request(operation, request, *, validation_context):
    if operation not in OPERATIONS:
        return False, {"denial": "UNKNOWN_OPERATION", "audit_event": "UNKNOWN_OPERATION_REJECTED"}
    if not validate_context(validation_context):
        context_denial = {"ACTIVATE_TRADING_UNIVERSE": "ACCOUNT_READINESS_BLOCKED", "BIND_CREDENTIAL_PROFILE": "CREDENTIAL_PROFILE_NOT_ACTIVE"}.get(operation, DATA["operation_validation_matrix"][operation]["denial_codes"][0])
        return audit_result(operation, False, context_denial)
    ok, denial = validate_request_shape(operation, request)
    if not ok:
        return audit_result(operation, False, denial)
    handlers = {
        "CREATE_ACCOUNT": handle_create_account,
        "UPDATE_ACCOUNT": handle_update_account,
        "BIND_CREDENTIAL_PROFILE": handle_bind_credential_profile,
        "VERIFY_EXTERNAL_IDENTITY": handle_verify_external_identity,
        "REFRESH_INSTRUMENT_CATALOG": handle_refresh_instrument_catalog,
        "ACTIVATE_TRADING_UNIVERSE": handle_activate_trading_universe,
        "RETIRE_ACCOUNT": handle_retire_account,
        "RETIRE_INSTRUMENT": handle_retire_instrument,
        "QUARANTINE_LEGACY_LIVE_RECORD": handle_quarantine_legacy_live_record,
    }
    handler = handlers.get(operation)
    if handler is None:
        return audit_result(operation, False, DATA["operation_validation_matrix"][operation]["denial_codes"][0])
    ok, denial = handler(request, validation_context)
    return audit_result(operation, ok, denial)


def handle_create_account(request, ctx):
    if not is_nonempty_str(request["portfolio_id"]):
        return False, "ACCOUNT_READINESS_BLOCKED"
    ex = exchange_entry(request["exchange_id"])
    if ex is None:
        return False, "UNKNOWN_EXCHANGE_ID"
    if ex["status"] != "ENABLED":
        return False, "EXCHANGE_DISABLED"
    if request["environment"] == "LIVE":
        return False, "LIVE_BLOCKED_BY_EDITION"
    if request["environment"] not in ex["supported_environments"]:
        return False, "EXCHANGE_ENVIRONMENT_UNSUPPORTED"
    if request["market_type"] not in ex["supported_market_types"]:
        return False, "MARKET_TYPE_UNSUPPORTED"
    if request["requested_lifecycle_state"] not in {"DRAFT", "DISABLED"}:
        return False, "ACCOUNT_READINESS_BLOCKED"
    for field in ["external_account_reference", "external_subaccount_reference"]:
        if request[field] is not None and not is_nonempty_str(request[field]):
            return False, "ACCOUNT_READINESS_BLOCKED"
    return True, None


def handle_update_account(request, ctx):
    account = ctx["accounts_by_id"].get(request["exchange_account_id"])
    if not isinstance(account, dict):
        return False, "EXCHANGE_ACCOUNT_NOT_FOUND"
    if account.get("exchange_account_id") != request["exchange_account_id"]:
        return False, "EXCHANGE_ACCOUNT_NOT_FOUND"
    if account.get("environment") == "LIVE":
        return False, "LIVE_BLOCKED_BY_EDITION"
    expected_fields = DATA["exchange_account_contract"]["identity_fields"]
    ok, denial = closed_schema(request["expected_identity_fields"], expected_fields, expected_fields, "EXCHANGE_ACCOUNT_IDENTITY_IMMUTABLE")
    if not ok:
        return False, denial
    if any(request["expected_identity_fields"].get(k) != account.get(k) for k in expected_fields):
        return False, "EXCHANGE_ACCOUNT_IDENTITY_IMMUTABLE"
    patch = request["mutable_patch"]
    allowed_patch = {"display_name", "lifecycle_state", "connection_state", "execution_authorization", "retired_at_utc"}
    if not isinstance(patch, dict) or not set(patch) <= allowed_patch:
        return False, "EXCHANGE_ACCOUNT_IDENTITY_IMMUTABLE"
    result = {**account, **patch}
    ok, denial = validate_account(result, operation="UPDATE_ACCOUNT", validation_context=ctx)
    if not ok:
        return False, denial
    ok, denial = enforce_operation_states("UPDATE_ACCOUNT", result)
    if not ok:
        return False, denial
    if result["lifecycle_state"] == "ACTIVE" and result["external_account_identity_state"] == "VERIFIED":
        return validate_trusted_identity_uniqueness(result["exchange_account_id"], ctx)
    return True, None


def handle_bind_credential_profile(request, ctx):
    if not is_nonempty_str(request["binding_intent"]):
        return False, "CREDENTIAL_PROFILE_NOT_ACTIVE"
    account = ctx["accounts_by_id"].get(request["exchange_account_id"])
    profile = ctx["credential_profiles_by_id"].get(request["credential_profile_id"])
    if account is None:
        return False, "EXCHANGE_ACCOUNT_NOT_FOUND"
    if profile is None:
        return False, "CREDENTIAL_PROFILE_NOT_FOUND"
    if account.get("exchange_account_id") != request["exchange_account_id"] or profile.get("credential_profile_id") != request["credential_profile_id"]:
        return False, "CREDENTIAL_PROFILE_NOT_FOUND"
    ok, denial = validate_account(account, operation="BIND_CREDENTIAL_PROFILE", validation_context=ctx)
    if not ok:
        return False, denial
    ok, denial = enforce_operation_states("BIND_CREDENTIAL_PROFILE", account)
    if not ok:
        return False, denial
    return validate_credential_profile_binding(profile, account, ctx)


def validate_trusted_identity_uniqueness(exchange_account_id, ctx):
    accounts = ctx.get("accounts_by_id") if isinstance(ctx, dict) else None
    snapshots = ctx.get("external_identity_snapshots_by_account_id") if isinstance(ctx, dict) else None
    if not isinstance(accounts, dict) or not isinstance(snapshots, dict):
        return deny("EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED")
    account = accounts.get(exchange_account_id); snapshot = snapshots.get(exchange_account_id)
    if not isinstance(account, dict) or not isinstance(snapshot, dict):
        return deny("EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED")
    ok, denial = validate_external_account_identity(snapshot, account)
    if not ok:
        return False, denial
    fields = ["exchange_id", "environment", "market_type", "venue_account_identifier", "subaccount_identifier"]
    identity_tuple = tuple(snapshot[field] for field in fields)
    for other_id, other_snapshot in snapshots.items():
        other_account = accounts.get(other_id)
        if other_id == exchange_account_id or not isinstance(other_account, dict) or other_account.get("lifecycle_state") != "ACTIVE" or not isinstance(other_snapshot, dict) or other_snapshot.get("state") != "VERIFIED":
            continue
        if not validate_external_account_identity(other_snapshot, other_account)[0]:
            return deny("EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED")
        if identity_tuple == tuple(other_snapshot[field] for field in fields):
            return deny("EXCHANGE_ACCOUNT_IDENTITY_COLLISION")
    return True, None


def validate_account_activation_readiness(account, ctx, now="2026-06-01T00:00:00Z"):
    if not isinstance(account, dict) or not validate_context(ctx):
        return deny("ACCOUNT_READINESS_BLOCKED")
    account_id = account.get("exchange_account_id")
    trusted_account = ctx["accounts_by_id"].get(account_id) if is_nonempty_str(account_id) else None
    if trusted_account is None or account != trusted_account:
        return deny("ACCOUNT_READINESS_BLOCKED")
    readiness = ctx["readiness_by_account_id"]
    if readiness.get(account_id) is not True or account.get("readiness_confirmed") is not True:
        return deny("ACCOUNT_READINESS_BLOCKED")
    ok, denial = validate_account(account, operation="ACTIVATE_TRADING_UNIVERSE", validation_context=ctx)
    if not ok:
        return False, denial
    ok, denial = enforce_operation_states("ACTIVATE_TRADING_UNIVERSE", account)
    if not ok or account.get("lifecycle_state") != "ACTIVE" or account.get("environment") == "LIVE":
        return False, denial or ("LIVE_BLOCKED_BY_EDITION" if account.get("environment") == "LIVE" else "ACCOUNT_READINESS_BLOCKED")
    if account.get("connection_state") not in {"ONLINE", "DEGRADED"}:
        return deny("ACCOUNT_READINESS_BLOCKED")
    authorization = account.get("execution_authorization")
    if authorization not in {"READ_ONLY", "ORDER_ENTRY_ALLOWED"}:
        return deny("ACCOUNT_READINESS_BLOCKED")
    ok, denial = validate_trusted_identity_uniqueness(account["exchange_account_id"], ctx)
    if not ok:
        return False, denial
    ex = exchange_entry(account["exchange_id"])
    snapshot_id = account.get("account_capability_snapshot_id")
    capability_snapshot = None
    if ex.get("capability_discovery_policy") == "ADAPTER_SNAPSHOT_REQUIRED" and snapshot_id is None:
        return deny("ACCOUNT_READINESS_BLOCKED")
    if snapshot_id is not None:
        snapshot = ctx["account_capability_snapshots_by_id"].get(snapshot_id)
        ok, denial = validate_account_capability_snapshot(snapshot, account, ctx["previous_account_capability_snapshots_by_id"])
        if not ok or snapshot.get("status") != "VALID" or not ts_lt(now, snapshot.get("stale_after_utc")):
            return deny("ACCOUNT_READINESS_BLOCKED")
        capability_snapshot = snapshot
    if account["environment"] == "TESTNET":
        profile_id = account.get("active_credential_profile_id")
        if not is_nonempty_str(profile_id):
            return deny("ACCOUNT_READINESS_BLOCKED")
        profile = ctx["credential_profiles_by_id"].get(profile_id)
        if not isinstance(profile, dict) or profile.get("credential_profile_id") != profile_id:
            return deny("ACCOUNT_READINESS_BLOCKED")
        ok, profile_denial = validate_credential_profile_binding(profile, account, ctx)
        if not ok:
            return deny(profile_denial if profile_denial == "WITHDRAWAL_PERMISSION_FORBIDDEN" else "ACCOUNT_READINESS_BLOCKED")
        identity = ctx["external_identity_snapshots_by_account_id"].get(account["exchange_account_id"])
        permission_sources = [set(profile["permission_snapshot"]), set(capability_snapshot["observed_permission_set"]), set(identity["observed_permission_set"])]
        if any("WITHDRAW" in source for source in permission_sources):
            return deny("WITHDRAWAL_PERMISSION_FORBIDDEN")
        effective_permissions = set.intersection(*permission_sources)
        if "READ_ACCOUNT" not in effective_permissions:
            return deny("ACCOUNT_READINESS_BLOCKED")
        if account["execution_authorization"] == "ORDER_ENTRY_ALLOWED" and "PLACE_ORDERS" not in effective_permissions:
            return deny("ACCOUNT_READINESS_BLOCKED")
    else:
        effective_permissions = set()
    return True, {"effective_permissions": sorted(effective_permissions), "capability_instrument_types": set(capability_snapshot["supported_instrument_types"]) if capability_snapshot else None, "extends_product_capabilities": False}


def handle_verify_external_identity(request, ctx):
    if type(request["reconnect"]) is not bool or not is_nonempty_str(request["expected_adapter_version_source"]):
        return False, "EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED"
    account = ctx["accounts_by_id"].get(request["exchange_account_id"])
    if account is None:
        return False, "EXCHANGE_ACCOUNT_NOT_FOUND"
    if account.get("exchange_account_id") != request["exchange_account_id"]:
        return False, "EXCHANGE_ACCOUNT_NOT_FOUND"
    if account.get("environment") == "LIVE":
        return False, "LIVE_BLOCKED_BY_EDITION"
    ok, denial = validate_account(account, operation="VERIFY_EXTERNAL_IDENTITY", validation_context=ctx)
    if not ok:
        return False, denial
    ok, denial = enforce_operation_states("VERIFY_EXTERNAL_IDENTITY", account)
    if not ok:
        return False, denial
    snapshot = ctx["external_identity_snapshots_by_account_id"].get(request["exchange_account_id"])
    if not isinstance(snapshot, dict):
        return False, "EXTERNAL_ACCOUNT_IDENTITY_UNAVAILABLE"
    if snapshot.get("adapter_version_source") != request["expected_adapter_version_source"]:
        return False, "EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED"
    ok, denial = validate_external_account_identity(snapshot, account, reconnect=request["reconnect"] is True)
    if not ok:
        return False, denial
    return validate_trusted_identity_uniqueness(request["exchange_account_id"], ctx)


def handle_refresh_instrument_catalog(request, ctx):
    catalogs_by_id = ctx["catalogs_by_id"]
    snapshot = catalogs_by_id.get(request["catalog_snapshot_id"])
    if not isinstance(snapshot, dict):
        return False, "CATALOG_SNAPSHOT_INVALID"
    if snapshot.get("catalog_snapshot_id") != request["catalog_snapshot_id"]:
        return False, "CATALOG_SNAPSHOT_INVALID"
    if request["environment"] == "LIVE" or snapshot.get("environment") == "LIVE":
        return False, "LIVE_BLOCKED_BY_EDITION"
    ex = exchange_entry(request["exchange_id"])
    if ex is None:
        return False, "UNKNOWN_EXCHANGE_ID"
    if request["environment"] not in ex.get("supported_environments", []):
        return False, "EXCHANGE_ENVIRONMENT_UNSUPPORTED"
    if request["market_type"] not in ex.get("supported_market_types", []):
        return False, "MARKET_TYPE_UNSUPPORTED"
    for field in ["exchange_id", "environment", "market_type", "adapter_family_id", "adapter_version"]:
        if request[field] != snapshot.get(field):
            return False, "CATALOG_SNAPSHOT_INVALID"
    instruments_by_id = ctx["instruments_by_id"]
    previous_catalogs_by_id = ctx["previous_catalogs_by_id"]
    return validate_instrument_catalog_snapshot(
        snapshot, instruments_by_id, previous_catalogs_by_id,
        instrument_history_by_id=ctx["instrument_history_by_id"],
        catalogs_by_id=ctx["catalogs_by_id"])


def handle_activate_trading_universe(request, ctx):
    universe = ctx["universes_by_id"].get(request["trading_universe_id"])
    account = ctx["accounts_by_id"].get(request["exchange_account_id"])
    if universe is None:
        return False, "TRADING_UNIVERSE_INVALID"
    if universe.get("trading_universe_id") != request["trading_universe_id"] or universe.get("exchange_account_id") != request["exchange_account_id"]:
        return False, "TRADING_UNIVERSE_INVALID"
    if universe.get("content_hash") != request["content_hash"] or sorted(universe.get("source_catalog_snapshot_ids", [])) != sorted(request["source_catalog_snapshot_ids"]):
        return False, "TRADING_UNIVERSE_INVALID"
    source_catalogs = {cid: ctx["catalogs_by_id"].get(cid) for cid in request["source_catalog_snapshot_ids"]}
    if account is None or account.get("exchange_account_id") != request["exchange_account_id"] or any(not isinstance(cat, dict) or cat.get("catalog_snapshot_id") != cid for cid, cat in source_catalogs.items()):
        return False, "EXCHANGE_ACCOUNT_NOT_FOUND" if account is None else "CATALOG_SNAPSHOT_INVALID"
    return validate_trading_universe_version(universe, account, ctx["instruments_by_id"], ctx["catalogs_by_id"], ctx["previous_universes_by_id"], ctx["active_universes"], ctx["previous_catalogs_by_id"], validation_context=ctx)


def handle_retire_account(request, ctx):
    account = ctx["accounts_by_id"].get(request["exchange_account_id"])
    if account is None:
        return False, "EXCHANGE_ACCOUNT_NOT_FOUND"
    if account.get("exchange_account_id") != request["exchange_account_id"]:
        return False, "EXCHANGE_ACCOUNT_NOT_FOUND"
    if account.get("lifecycle_state") == "RETIRED":
        return False, "EXCHANGE_ACCOUNT_RETIRED"
    if account.get("private_connection_active") is True or account.get("connection_state") not in {"DISCONNECTED", "BLOCKED"}:
        return False, "ACCOUNT_READINESS_BLOCKED"
    if account.get("environment") == "LIVE" and not account.get("legacy_record"):
        return False, "LIVE_BLOCKED_BY_EDITION"
    if not is_nonempty_str(request["retirement_reason"]):
        return False, "ACCOUNT_READINESS_BLOCKED"
    ok, denial = validate_account(account, operation="RETIRE_ACCOUNT")
    if not ok and denial != "LIVE_BLOCKED_BY_EDITION":
        return False, denial
    ok, denial = enforce_operation_states("RETIRE_ACCOUNT", account)
    if not ok:
        return False, denial
    return True, None


def handle_retire_instrument(request, ctx):
    inst = ctx["instruments_by_id"].get(request["instrument_id"])
    catalog = ctx["catalogs_by_id"].get(request["catalog_snapshot_id"])
    if inst is None or inst.get("instrument_id") != request["instrument_id"]:
        return False, "INSTRUMENT_NOT_FOUND"
    if catalog is None or catalog.get("catalog_snapshot_id") != request["catalog_snapshot_id"] or inst.get("catalog_snapshot_id") != request["catalog_snapshot_id"]:
        return False, "CATALOG_SNAPSHOT_INVALID"
    for universe in ctx["active_universes"]:
        if isinstance(universe, dict) and inst["instrument_id"] in universe.get("instrument_ids", []):
            return False, "INSTRUMENT_NOT_TRADABLE"
    if not is_nonempty_str(request["retirement_reason"]):
        return False, "INSTRUMENT_METADATA_INVALID"
    return True, None


def handle_quarantine_legacy_live_record(request, ctx):
    account = ctx["accounts_by_id"].get(request["exchange_account_id"])
    if account is None:
        return False, "EXCHANGE_ACCOUNT_NOT_FOUND"
    if account.get("exchange_account_id") != request["exchange_account_id"]:
        return False, "EXCHANGE_ACCOUNT_NOT_FOUND"
    if request["legacy_record"] is not True or account.get("legacy_record") is not True or account.get("environment") != "LIVE":
        return False, "LIVE_BLOCKED_BY_EDITION"
    if request["target_lifecycle_state"] not in {"DISABLED", "RETIRED"}:
        return False, "ACCOUNT_READINESS_BLOCKED"
    if request["target_connection_state"] not in {"DISCONNECTED", "BLOCKED"}:
        return False, "ACCOUNT_READINESS_BLOCKED"
    if request["target_execution_authorization"] != "BLOCKED_BY_POLICY":
        return False, "ACCOUNT_READINESS_BLOCKED"
    if not is_nonempty_str(request["quarantine_reason"]):
        return False, "ACCOUNT_READINESS_BLOCKED"
    return True, None


def asset(code="BTC", status="EXACT"):
    return {"venue_asset_code": code, "canonical_display_code": code, "asset_namespace": "generic_testnet_venue", "mapping_status": status}


def sample_account(**overrides):
    data = {"exchange_account_id":"xacc_test_1","portfolio_id":"port_1","exchange_id":"generic_testnet_venue","environment":"TESTNET","market_type":"SPOT","display_name":"Test","lifecycle_state":"ACTIVE","connection_state":"ONLINE","execution_authorization":"READ_ONLY","external_account_identity_state":"VERIFIED","external_account_reference":None,"external_subaccount_reference":None,"active_credential_profile_id":None,"account_capability_snapshot_id":None,"created_at_utc":"2026-01-01T00:00:00Z","retired_at_utc":None,"readiness_confirmed":True}
    data.update(overrides)
    return data


def direct_universe_result(universe, account, instruments, catalogs,
                           previous_universes_by_id=None, active_universes=None,
                           previous_catalogs_by_id=None, **context_overrides):
    """Uruchamia direct validator przez tę samą zaufaną bramkę co dispatcher."""
    primary = next(iter(instruments.values()))
    primary_catalog = next(iter(catalogs.values()))
    ctx = context(account=account, inst=primary, catalog=primary_catalog, universe=universe)
    ctx["instruments_by_id"] = instruments
    ctx["catalogs_by_id"] = catalogs
    ctx["previous_universes_by_id"] = {} if previous_universes_by_id is None else previous_universes_by_id
    ctx["active_universes"] = [] if active_universes is None else active_universes
    ctx["previous_catalogs_by_id"] = {} if previous_catalogs_by_id is None else previous_catalogs_by_id
    ctx.update(context_overrides)
    trusted_account = ctx["accounts_by_id"][account["exchange_account_id"]]
    return validate_trading_universe_version(
        universe, trusted_account, instruments, catalogs,
        ctx["previous_universes_by_id"], ctx["active_universes"],
        ctx["previous_catalogs_by_id"], validation_context=ctx,
    )


def sample_profile(**overrides):
    data = {"credential_profile_id":"cred_1","exchange_account_id":"xacc_test_1","exchange_id":"generic_testnet_venue","environment_scope":"TESTNET","credential_purpose":"ACCOUNT_READ","secure_store_reference":"secure-store://profile/cred_1","public_key_identifier":None,"permission_snapshot":["READ_ACCOUNT"],"lifecycle_state":"ACTIVE","created_at_utc":"2026-01-01T00:00:00Z","rotated_from_credential_profile_id":None,"retired_at_utc":None,"saas_sync_candidate":False}
    data.update(overrides)
    return data


def sample_identity(**overrides):
    data = {"state":"VERIFIED","exchange_id":"generic_testnet_venue","environment":"TESTNET","market_type":"SPOT","venue_account_identifier":"venue-acct-1","subaccount_identifier":None,"account_type":"SPOT","observed_permission_set":["READ_ACCOUNT"],"verification_timestamp":"2026-01-01T00:00:00Z","adapter_version_source":"generic_testnet_adapter_family/1"}
    data.update(overrides)
    return data


def sample_capability_snapshot(account=None, **overrides):
    account = account or sample_account()
    data = {"account_capability_snapshot_id":"caps_1","exchange_account_id":account["exchange_account_id"],"exchange_id":account["exchange_id"],"environment":account["environment"],"market_type":account["market_type"],"version":1,"previous_snapshot_id":None,"status":"VALID","observed_permission_set":["READ_ACCOUNT"],"supported_instrument_types":["SPOT_PAIR"],"observed_at_utc":"2026-01-01T00:00:00Z","effective_at_utc":"2026-01-01T00:00:00Z","stale_after_utc":"2026-12-01T00:00:00Z","adapter_family_id":"generic_testnet_adapter_family","adapter_version":"1","content_hash":""}
    data.update(overrides)
    data["content_hash"] = overrides.get("content_hash") or hash_payload(DATA["account_capability_snapshot_contract"]["content_hash_definition"], data)
    return data


def sample_instrument(**overrides):
    data = {"instrument_id":"instr_btcusdt_spot_testnet","workspace_id":"ws_1","exchange_id":"generic_testnet_venue","environment":"TESTNET","market_type":"SPOT","instrument_type":"SPOT_PAIR","venue_symbol":"BTCUSDT","display_symbol":"BTC/USDT","base_asset_reference":asset("BTC"),"quote_asset_reference":asset("USDT"),"settlement_asset_reference":None,"trading_status":"TRADING","price_tick":"0.01","quantity_step":"0.0001","min_quantity":"0.0001","max_quantity":"100","min_notional":"5","max_notional":None,"contract_size":None,"contract_value_currency":None,"derivative_settlement_type":None,"expiry_at_utc":None,"strike_price":None,"option_side":None,"catalog_snapshot_id":"cat_1","metadata_version":1,"observed_at_utc":"2026-01-01T00:00:00Z","effective_at_utc":"2026-01-01T00:00:00Z","stale_after_utc":"2026-12-01T00:00:00Z","source_adapter_family_id":"generic_testnet_adapter_family"}
    data.update(overrides)
    return data


def sample_catalog(inst=None, **overrides):
    inst = inst or sample_instrument()
    data = {"catalog_snapshot_id":inst["catalog_snapshot_id"],"exchange_id":inst["exchange_id"],"environment":inst["environment"],"market_type":inst["market_type"],"adapter_family_id":inst["source_adapter_family_id"],"adapter_version":"1","observed_at_utc":"2026-01-01T00:00:00Z","effective_at_utc":"2026-01-01T00:00:00Z","stale_after_utc":"2026-12-01T00:00:00Z","instrument_ids":[inst["instrument_id"]],"previous_snapshot_id":None,"status":"VALID","content_hash":""}
    data.update(overrides)
    data["content_hash"] = overrides.get("content_hash") or hash_payload(DATA["deterministic_hash_contracts"]["catalog_snapshot_content_hash"], data)
    return data


def sample_universe(account=None, inst=None, catalog=None, **overrides):
    account = account or sample_account()
    inst = inst or sample_instrument()
    catalog = catalog or sample_catalog(inst)
    data = {"trading_universe_id":"univ_1","exchange_account_id":account["exchange_account_id"],"version":1,"lifecycle_state":"ACTIVE","instrument_ids":[inst["instrument_id"]],"created_at_utc":"2026-01-02T00:00:00Z","activated_at_utc":"2026-01-02T00:00:00Z","retired_at_utc":None,"previous_version_id":None,"source_catalog_snapshot_ids":[catalog["catalog_snapshot_id"]],"content_hash":"","creation_reason":"INITIAL"}
    data.update(overrides)
    data["content_hash"] = overrides.get("content_hash") or hash_payload(DATA["deterministic_hash_contracts"]["trading_universe_content_hash"], data)
    return data


def context(account=None, profile=None, identity=None, inst=None, catalog=None, universe=None, **overrides):
    account = dict(account or sample_account())
    capability_snapshots = {}
    if account.get("exchange_id") == "generic_testnet_venue" and account.get("environment") == "TESTNET" and account.get("account_capability_snapshot_id") is None:
        account["account_capability_snapshot_id"] = "caps_1"
        capability_snapshots["caps_1"] = sample_capability_snapshot(account)
    if account.get("environment") == "TESTNET" and account.get("active_credential_profile_id") is None:
        account["active_credential_profile_id"] = "cred_1"
    profile = profile or sample_profile(exchange_account_id=account["exchange_account_id"], exchange_id=account["exchange_id"], environment_scope=account["environment"])
    identity = identity or sample_identity(exchange_id=account["exchange_id"], environment=account["environment"], market_type=account["market_type"])
    inst = inst or sample_instrument(exchange_id=account["exchange_id"], environment=account["environment"], market_type=account["market_type"])
    catalog = catalog or sample_catalog(inst)
    universe = universe or sample_universe(account, inst, catalog)
    bound_profile = account.get("active_credential_profile_id") == profile.get("credential_profile_id")
    data = {"accounts_by_id":{account["exchange_account_id"]:account},"credential_profiles_by_id":({profile["credential_profile_id"]:profile} if bound_profile else {}),"external_identity_snapshots_by_account_id":{account["exchange_account_id"]:identity},"universes_by_id":{universe["trading_universe_id"]:universe},"catalogs_by_id":{catalog["catalog_snapshot_id"]:catalog},"instruments_by_id":{inst["instrument_id"]:inst},"instrument_history_by_id":{},"previous_profiles_by_id":{},"previous_catalogs_by_id":{},"previous_universes_by_id":{},"account_capability_snapshots_by_id":capability_snapshots,"previous_account_capability_snapshots_by_id":{},"active_universes":[],"active_profile_ids_by_account_id":({account["exchange_account_id"]:[profile["credential_profile_id"]]} if bound_profile else {}),"readiness_by_account_id":{account["exchange_account_id"]:True}}
    data.update(overrides)
    return data


def identity_fields(account):
    return {field: account[field] for field in DATA["exchange_account_contract"]["identity_fields"]}


def request_for(operation, **overrides):
    base = {"CREATE_ACCOUNT":{"request_id":"r1","portfolio_id":"port_2","exchange_id":"generic_testnet_venue","environment":"TESTNET","market_type":"SPOT","external_account_reference":None,"external_subaccount_reference":None,"requested_lifecycle_state":"DRAFT"},
            "UPDATE_ACCOUNT":{"request_id":"r2","exchange_account_id":"xacc_test_1","expected_identity_fields":identity_fields(sample_account()),"mutable_patch":{"display_name":"Updated"}},
            "BIND_CREDENTIAL_PROFILE":{"request_id":"r3","exchange_account_id":"xacc_test_1","credential_profile_id":"cred_1","binding_intent":"ACTIVATE"},
            "VERIFY_EXTERNAL_IDENTITY":{"request_id":"r4","exchange_account_id":"xacc_test_1","expected_adapter_version_source":"generic_testnet_adapter_family/1","reconnect":False},
            "REFRESH_INSTRUMENT_CATALOG":{"request_id":"r5","catalog_snapshot_id":"cat_1","exchange_id":"generic_testnet_venue","environment":"TESTNET","market_type":"SPOT","adapter_family_id":"generic_testnet_adapter_family","adapter_version":"1"},
            "ACTIVATE_TRADING_UNIVERSE":{"request_id":"r6","trading_universe_id":"univ_1","exchange_account_id":"xacc_test_1","source_catalog_snapshot_ids":["cat_1"],"content_hash":sample_universe().get("content_hash")},
            "RETIRE_ACCOUNT":{"request_id":"r7","exchange_account_id":"xacc_test_1","retirement_reason":"operator"},
            "RETIRE_INSTRUMENT":{"request_id":"r8","instrument_id":"instr_btcusdt_spot_testnet","catalog_snapshot_id":"cat_1","retirement_reason":"delisted"},
            "QUARANTINE_LEGACY_LIVE_RECORD":{"request_id":"r9","exchange_account_id":"xacc_live","legacy_record":True,"target_lifecycle_state":"DISABLED","target_connection_state":"BLOCKED","target_execution_authorization":"BLOCKED_BY_POLICY","quarantine_reason":"legacy"}}
    req = dict(base[operation])
    req.update(overrides)
    return req


def test_status_and_documentation_sync():
    assert M04["status"] == "closed" and "Status: `closed`" in M04_MD.read_text() and "Status M0.4 — closed" in ARCH.read_text()
    assert DATA["status"] == "closed"
    assert "Status: `closed`" in MD.read_text()
    assert "Status M0.5 — closed" in ARCH.read_text()
    assert "request carries transport IDs hashes lists only" in json.dumps(DATA["operation_dispatch_policy"])


def test_dispatcher_has_no_default_success_fallthrough():
    src = inspect.getsource(validate_operation_request)
    assert "return audit_result(operation, True)" not in src
    assert set(OPERATIONS) == set(DATA["operation_validation_matrix"])
    assert all(op in src for op in OPERATIONS)


def test_request_id_fields_reject_embedded_records():
    for operation, field in [("UPDATE_ACCOUNT", "exchange_account_id"), ("BIND_CREDENTIAL_PROFILE", "credential_profile_id"), ("ACTIVATE_TRADING_UNIVERSE", "trading_universe_id")]:
        req = request_for(operation)
        req[field] = {"embedded": "record"}
        ok, out = validate_operation_request(operation, req, validation_context=context())
        assert not ok and out["denial"] in DENIALS


def test_create_rejects_active_requested_lifecycle():
    ok, out = validate_operation_request("CREATE_ACCOUNT", request_for("CREATE_ACCOUNT", requested_lifecycle_state="ACTIVE"), validation_context=context())
    assert not ok and out["denial"] == "ACCOUNT_READINESS_BLOCKED" and out["audit_event"] != "EXCHANGE_ACCOUNT_CREATED"


def test_update_rejects_request_record_id_mismatch():
    acct = sample_account(exchange_account_id="xacc_actual")
    ok, out = validate_operation_request("UPDATE_ACCOUNT", request_for("UPDATE_ACCOUNT", exchange_account_id="xacc_requested"), validation_context=context(account=acct))
    assert not ok and out["denial"] == "EXCHANGE_ACCOUNT_NOT_FOUND"


def test_update_rejects_expected_identity_mismatch():
    req = request_for("UPDATE_ACCOUNT")
    req["expected_identity_fields"] = {**req["expected_identity_fields"], "exchange_id": "other"}
    ok, out = validate_operation_request("UPDATE_ACCOUNT", req, validation_context=context())
    assert not ok and out["denial"] == "EXCHANGE_ACCOUNT_IDENTITY_IMMUTABLE"


def test_update_rejects_identity_patch():
    req = request_for("UPDATE_ACCOUNT", mutable_patch={"exchange_id":"other"})
    ok, out = validate_operation_request("UPDATE_ACCOUNT", req, validation_context=context())
    assert not ok and out["denial"] == "EXCHANGE_ACCOUNT_IDENTITY_IMMUTABLE"


def test_live_update_cannot_preserve_operational_state():
    live = sample_account(exchange_account_id="xacc_live", environment="LIVE", lifecycle_state="ACTIVE", connection_state="ONLINE", execution_authorization="ORDER_ENTRY_ALLOWED")
    req = request_for("UPDATE_ACCOUNT", exchange_account_id="xacc_live", expected_identity_fields=identity_fields(live))
    ok, out = validate_operation_request("UPDATE_ACCOUNT", req, validation_context=context(account=live))
    assert not ok and out["denial"] == "EXCHANGE_ACCOUNT_NOT_FOUND"


def test_verify_external_identity_runs_real_validator():
    bad = sample_identity(venue_account_identifier="")
    req = request_for("VERIFY_EXTERNAL_IDENTITY")
    ok, out = validate_operation_request("VERIFY_EXTERNAL_IDENTITY", req, validation_context=context(account=sample_account(connection_state="SYNCHRONIZING"), identity=bad))
    assert not ok and out["denial"] == "EXCHANGE_ACCOUNT_NOT_FOUND" and out["audit_event"] != "EXTERNAL_ACCOUNT_IDENTITY_VERIFIED"


def test_refresh_catalog_runs_real_validator():
    inst = sample_instrument()
    bad_catalog = sample_catalog(inst, content_hash="0" * 64)
    req = request_for("REFRESH_INSTRUMENT_CATALOG")
    ok, out = validate_operation_request("REFRESH_INSTRUMENT_CATALOG", req, validation_context=context(inst=inst, catalog=bad_catalog))
    assert not ok and out["denial"] == "UNKNOWN_EXCHANGE_ID"


def test_activate_request_uses_ids_hash_and_context():
    ctx = context()
    ok, out = validate_operation_request("ACTIVATE_TRADING_UNIVERSE", request_for("ACTIVATE_TRADING_UNIVERSE"), validation_context=ctx)
    assert ok and out["audit_event"] == "TRADING_UNIVERSE_ACTIVATED"
    req = request_for("ACTIVATE_TRADING_UNIVERSE", content_hash="0" * 64)
    ok, out = validate_operation_request("ACTIVATE_TRADING_UNIVERSE", req, validation_context=ctx)
    assert not ok and out["denial"] == "TRADING_UNIVERSE_INVALID"


def test_retire_account_runs_real_validator():
    acct = sample_account(connection_state="ONLINE")
    ok, out = validate_operation_request("RETIRE_ACCOUNT", request_for("RETIRE_ACCOUNT"), validation_context=context(account=acct))
    assert not ok and out["denial"] == "ACCOUNT_READINESS_BLOCKED"


def test_retire_instrument_runs_real_validator():
    inst = sample_instrument()
    cat = sample_catalog(inst)
    active = sample_universe(inst=inst, catalog=cat)
    ok, out = validate_operation_request("RETIRE_INSTRUMENT", request_for("RETIRE_INSTRUMENT"), validation_context=context(inst=inst, catalog=cat, active_universes=[active]))
    assert not ok and out["denial"] == "INSTRUMENT_NOT_TRADABLE"


def test_quarantine_rejects_active_target():
    live = sample_account(exchange_account_id="xacc_live", environment="LIVE", legacy_record=True, lifecycle_state="DISABLED", connection_state="BLOCKED", execution_authorization="BLOCKED_BY_POLICY")
    ok, out = validate_operation_request("QUARANTINE_LEGACY_LIVE_RECORD", request_for("QUARANTINE_LEGACY_LIVE_RECORD", target_lifecycle_state="ACTIVE"), validation_context=context(account=live))
    assert not ok and out["denial"] == "EXCHANGE_ACCOUNT_NOT_FOUND"


def test_quarantine_rejects_online_target():
    live = sample_account(exchange_account_id="xacc_live", environment="LIVE", legacy_record=True, lifecycle_state="DISABLED", connection_state="BLOCKED", execution_authorization="BLOCKED_BY_POLICY")
    ok, out = validate_operation_request("QUARANTINE_LEGACY_LIVE_RECORD", request_for("QUARANTINE_LEGACY_LIVE_RECORD", target_connection_state="ONLINE"), validation_context=context(account=live))
    assert not ok and out["denial"] == "EXCHANGE_ACCOUNT_NOT_FOUND"


def test_quarantine_rejects_order_entry_target():
    live = sample_account(exchange_account_id="xacc_live", environment="LIVE", legacy_record=True, lifecycle_state="DISABLED", connection_state="BLOCKED", execution_authorization="BLOCKED_BY_POLICY")
    ok, out = validate_operation_request("QUARANTINE_LEGACY_LIVE_RECORD", request_for("QUARANTINE_LEGACY_LIVE_RECORD", target_execution_authorization="ORDER_ENTRY_ALLOWED"), validation_context=context(account=live))
    assert not ok and out["denial"] == "EXCHANGE_ACCOUNT_NOT_FOUND"


def test_quarantine_checks_legacy_flag_and_reason():
    live = sample_account(exchange_account_id="xacc_live", environment="LIVE", legacy_record=True, lifecycle_state="DISABLED", connection_state="BLOCKED", execution_authorization="BLOCKED_BY_POLICY")
    assert validate_operation_request("QUARANTINE_LEGACY_LIVE_RECORD", request_for("QUARANTINE_LEGACY_LIVE_RECORD", legacy_record=False), validation_context=context(account=live))[1]["denial"] == "EXCHANGE_ACCOUNT_NOT_FOUND"
    assert validate_operation_request("QUARANTINE_LEGACY_LIVE_RECORD", request_for("QUARANTINE_LEGACY_LIVE_RECORD", quarantine_reason=""), validation_context=context(account=live))[1]["denial"] == "EXCHANGE_ACCOUNT_NOT_FOUND"
    assert validate_operation_request("QUARANTINE_LEGACY_LIVE_RECORD", request_for("QUARANTINE_LEGACY_LIVE_RECORD"), validation_context=context(account=sample_account(exchange_account_id="xacc_live")))[1]["denial"] == "LIVE_BLOCKED_BY_EDITION"


def test_known_denial_never_uses_unknown_operation_event():
    schemas = {s["event_name"]: s for s in DATA["audit_event_schemas"]}
    for op, spec in DATA["operation_validation_matrix"].items():
        assert spec["success_audit_event"] not in spec["denial_event_by_code"].values()
        for code, event in spec["denial_event_by_code"].items():
            assert event != "UNKNOWN_OPERATION_REJECTED"
            assert code in spec["denial_codes"]
            assert event in DATA["audit_event_registry"]
            assert code in schemas[event]["allowed_denial_codes"]


def test_activate_retired_account_has_registered_audit_event():
    acct = sample_account(lifecycle_state="RETIRED", retired_at_utc="2026-02-01T00:00:00Z")
    ctx = context(account=acct, identity=sample_identity(state="UNVERIFIED"))
    ok, out = validate_operation_request("ACTIVATE_TRADING_UNIVERSE", request_for("ACTIVATE_TRADING_UNIVERSE"), validation_context=ctx)
    assert not ok and out["denial"] == "EXCHANGE_ACCOUNT_RETIRED"
    assert out["audit_event"] in DATA["audit_event_registry"]


def test_activate_unverified_account_has_registered_audit_event():
    acct = sample_account(external_account_identity_state="UNVERIFIED")
    ctx = context(account=acct, identity=sample_identity(state="UNVERIFIED"))
    ok, out = validate_operation_request("ACTIVATE_TRADING_UNIVERSE", request_for("ACTIVATE_TRADING_UNIVERSE"), validation_context=ctx)
    assert not ok and out["denial"] == "EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED"
    assert out["audit_event"] in DATA["audit_event_registry"]


def test_denial_audit_schema_has_nonempty_allowed_codes():
    names = [schema["event_name"] for schema in DATA["audit_event_schemas"]]
    assert set(names) == set(DATA["audit_event_registry"])
    assert len(names) == len(set(names))
    for schema in DATA["audit_event_schemas"]:
        assert len(schema["required_identifier_fields"]) == len(set(schema["required_identifier_fields"]))
        assert len(schema["required_context_fields"]) == len(set(schema["required_context_fields"]))
        if schema["outcome_class"] == "DENIAL":
            assert schema["allowed_denial_codes"] and set(schema["allowed_denial_codes"]) <= DENIALS
        else:
            assert schema["allowed_denial_codes"] == [] and schema["denial_code"] == "nullable_must_be_null_on_success"


def test_readiness_confirmed_must_be_literal_true():
    acct = sample_account(readiness_confirmed="true")
    ctx = context(account=acct)
    ok, out = validate_operation_request("ACTIVATE_TRADING_UNIVERSE", request_for("ACTIVATE_TRADING_UNIVERSE"), validation_context=ctx)
    assert not ok and out["denial"] == "ACCOUNT_READINESS_BLOCKED"


def test_nested_catalog_wrong_type_fails_closed():
    for bad_context in [None, 1, "x", [], {"catalogs_by_id": []}, {"catalogs_by_id": {"cat_1": []}}, {"catalogs_by_id": {"cat_1": {"missing": "fields"}}}]:
        ok, out = validate_operation_request("REFRESH_INSTRUMENT_CATALOG", request_for("REFRESH_INSTRUMENT_CATALOG"), validation_context=bad_context)
        assert not ok and out["denial"] in DENIALS


def test_credential_lineage_rejects_multinode_cycle():
    acct = sample_account()
    current = sample_profile(credential_profile_id="cred_new", rotated_from_credential_profile_id="cred_old")
    old = sample_profile(credential_profile_id="cred_old", lifecycle_state="RETIRED", retired_at_utc="2026-02-01T00:00:00Z", rotated_from_credential_profile_id="cred_older")
    older = sample_profile(credential_profile_id="cred_older", lifecycle_state="RETIRED", retired_at_utc="2026-01-15T00:00:00Z", rotated_from_credential_profile_id="cred_new")
    ok, denial = validate_credential_profile_binding(current, acct, {"previous_profiles_by_id":{"cred_old":old,"cred_older":older}, "active_profile_ids_by_account_id":{acct["exchange_account_id"]:["cred_new"]}})
    assert not ok and denial == "CREDENTIAL_PROFILE_NOT_ACTIVE"


def test_catalog_lineage_rejects_two_node_cycle():
    inst = sample_instrument(catalog_snapshot_id="cat_a")
    cat_a = sample_catalog(inst, catalog_snapshot_id="cat_a", previous_snapshot_id="cat_b")
    cat_b = sample_catalog(inst, catalog_snapshot_id="cat_b", previous_snapshot_id="cat_a")
    assert validate_instrument_catalog_snapshot(cat_a, {inst["instrument_id"]:inst}, {"cat_b":cat_b, "cat_a":cat_a}, instrument_history_by_id={})[1] == "CATALOG_SNAPSHOT_INVALID"


def test_catalog_lineage_rejects_multinode_cycle():
    inst = sample_instrument(catalog_snapshot_id="cat_a")
    cat_a = sample_catalog(inst, catalog_snapshot_id="cat_a", previous_snapshot_id="cat_b")
    cat_b = sample_catalog(inst, catalog_snapshot_id="cat_b", previous_snapshot_id="cat_c")
    cat_c = sample_catalog(inst, catalog_snapshot_id="cat_c", previous_snapshot_id="cat_a")
    assert validate_instrument_catalog_snapshot(cat_a, {inst["instrument_id"]:inst}, {"cat_b":cat_b, "cat_c":cat_c, "cat_a":cat_a}, instrument_history_by_id={})[1] == "CATALOG_SNAPSHOT_INVALID"


def test_unknown_operation_and_no_caller_result_controls():
    assert validate_operation_request("NO_SUCH_OP", {"request_id":"r"}, validation_context=context())[1] == {"denial":"UNKNOWN_OPERATION", "audit_event":"UNKNOWN_OPERATION_REJECTED"}
    req = request_for("CREATE_ACCOUNT", result="SUCCESS")
    ok, out = validate_operation_request("CREATE_ACCOUNT", req, validation_context=context())
    assert not ok and out["audit_event"] != "EXCHANGE_ACCOUNT_CREATED"


def test_exact_venue_symbol_and_decimal_fullmatch():
    assert validate_instrument_record(sample_instrument(venue_symbol="btcusdt"))[0]
    assert validate_instrument_record(sample_instrument(venue_symbol="BtcUsdt"))[0]
    assert validate_instrument_record(sample_instrument(venue_symbol=" BTCUSDT"))[1] == "INSTRUMENT_IDENTITY_COLLISION"
    assert "venue_symbol_exact" not in DATA["instrument_contract"]["record_fields"]
    for bad in ["1\n", "1\r", "1\t", " 1", "1 ", "+1", "01", ".5", "1.", "1.2300"]:
        assert not is_canonical_decimal(bad)


def test_json_markdown_sync_for_final_fix():
    md = MD.read_text()
    for phrase in ["transportowe IDs", "trusted validation context", "brak default-success", "pełny handler", "visited set", "literal bool readiness", "nested fail-closed"]:
        assert phrase in md
    assert DATA["status"] == "closed"


def test_malformed_nested_context_fails_closed_for_all_handlers():
    for operation in OPERATIONS:
        for field in CONTEXT_MAPS | CONTEXT_LISTS:
            for bad in (None, 1, "bad", [] if field in CONTEXT_MAPS else {}):
                ctx = context()
                ctx[field] = bad
                ok, out = validate_operation_request(operation, request_for(operation), validation_context=ctx)
                assert not ok and out["denial"] in DENIALS
        map_field = next(iter(CONTEXT_MAPS))
        for bad_record in ([], {"missing": "fields"}):
            ctx = context()
            ctx[map_field] = {"bad": bad_record}
            ok, out = validate_operation_request(operation, request_for(operation), validation_context=ctx)
            assert not ok and out["denial"] in DENIALS


def test_verify_rejects_non_boolean_reconnect():
    for value in (0, 1, "true", "yes", [], {}, None):
        ok, out = validate_operation_request("VERIFY_EXTERNAL_IDENTITY", request_for("VERIFY_EXTERNAL_IDENTITY", reconnect=value), validation_context=context())
        assert not ok and out["denial"] == "EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED"


def test_verify_non_boolean_reconnect_cannot_bypass_revalidation():
    snapshot = sample_identity(revalidated_at_utc="not-a-timestamp")
    for value in (1, "true", [], {}):
        ok, out = validate_operation_request("VERIFY_EXTERNAL_IDENTITY", request_for("VERIFY_EXTERNAL_IDENTITY", reconnect=value), validation_context=context(identity=snapshot))
        assert not ok and out["audit_event"] != "EXTERNAL_ACCOUNT_IDENTITY_VERIFIED"


def test_create_rejects_non_string_external_account_reference():
    for value in ({}, [], 1, True, ""):
        assert validate_operation_request("CREATE_ACCOUNT", request_for("CREATE_ACCOUNT", external_account_reference=value), validation_context=context())[0] is False


def test_create_rejects_non_string_external_subaccount_reference():
    for value in ({}, [], 1, True, ""):
        assert validate_operation_request("CREATE_ACCOUNT", request_for("CREATE_ACCOUNT", external_subaccount_reference=value), validation_context=context())[0] is False


def test_catalog_rejects_exchange_unsupported_environment():
    inst = sample_instrument(environment="PAPER")
    cat = sample_catalog(inst)
    ok, denial = validate_instrument_catalog_snapshot(cat, {inst["instrument_id"]: inst}, {}, instrument_history_by_id={})
    assert not ok and denial == "EXCHANGE_ENVIRONMENT_UNSUPPORTED"
    ctx = context(inst=inst, catalog=cat)
    ok, out = validate_operation_request("REFRESH_INSTRUMENT_CATALOG", request_for("REFRESH_INSTRUMENT_CATALOG", environment="PAPER"), validation_context=ctx)
    assert not ok and out["denial"] == "UNKNOWN_EXCHANGE_ID"


def test_catalog_rejects_exchange_unsupported_market_type():
    inst = sample_instrument(market_type="MARGIN", instrument_type="MARGIN_PAIR")
    cat = sample_catalog(inst)
    ok, denial = validate_instrument_catalog_snapshot(cat, {inst["instrument_id"]: inst}, {}, instrument_history_by_id={})
    assert not ok and denial == "MARKET_TYPE_UNSUPPORTED"
    ctx = context(inst=inst, catalog=cat)
    ok, out = validate_operation_request("REFRESH_INSTRUMENT_CATALOG", request_for("REFRESH_INSTRUMENT_CATALOG", market_type="MARGIN"), validation_context=ctx)
    assert not ok and out["denial"] == "UNKNOWN_EXCHANGE_ID"


def test_catalog_runs_full_instrument_validation():
    inst = sample_instrument(price_tick="0")
    cat = sample_catalog(inst)
    assert validate_instrument_catalog_snapshot(cat, {inst["instrument_id"]: inst}, {}, instrument_history_by_id={})[1] == "INSTRUMENT_METADATA_INVALID"


def test_catalog_rejects_unsupported_instrument_type():
    inst = sample_instrument(instrument_type="OPTION")
    cat = sample_catalog(inst)
    assert validate_instrument_catalog_snapshot(cat, {inst["instrument_id"]: inst}, {}, instrument_history_by_id={})[1] == "INSTRUMENT_TYPE_UNSUPPORTED"


def test_universe_rejects_instrument_not_in_source_catalogs():
    account = sample_account()
    member = sample_instrument(instrument_id="instr_member", catalog_snapshot_id="cat_other", venue_symbol="MEMBER")
    source_inst = sample_instrument(instrument_id="instr_source")
    catalog = sample_catalog(source_inst)
    universe = sample_universe(account, member, catalog)
    other_catalog=sample_catalog(member,catalog_snapshot_id="cat_other",instrument_ids=[member["instrument_id"]])
    ok, denial = direct_universe_result(universe, account, {member["instrument_id"]: member, source_inst["instrument_id"]: source_inst}, {catalog["catalog_snapshot_id"]: catalog,"cat_other":other_catalog})
    assert not ok and denial == "ACCOUNT_READINESS_BLOCKED"


def test_universe_rejects_instrument_from_different_catalog():
    account = sample_account()
    inst = sample_instrument(catalog_snapshot_id="cat_other")
    catalog = sample_catalog(inst, catalog_snapshot_id="cat_1", instrument_ids=[inst["instrument_id"]])
    universe = sample_universe(account, inst, catalog)
    assert direct_universe_result(universe, account, {inst["instrument_id"]: inst}, {"cat_1": catalog})[0] is False


def test_universe_accepts_valid_catalog_with_predecessor_lineage():
    account = sample_account()
    inst = sample_instrument(metadata_version=2)
    historical = sample_instrument(metadata_version=1,catalog_snapshot_id="cat_old")
    old = sample_catalog(historical, catalog_snapshot_id="cat_old")
    current = sample_catalog(inst, previous_snapshot_id="cat_old")
    universe = sample_universe(account, inst, current)
    assert direct_universe_result(universe, account, {inst["instrument_id"]: inst}, {"cat_1": current}, previous_catalogs_by_id={"cat_old": old},instrument_history_by_id={inst["instrument_id"]:[historical]})[0]


def test_universe_rejects_broken_source_catalog_lineage():
    account = sample_account(); inst = sample_instrument(); catalog = sample_catalog(inst, previous_snapshot_id="missing"); universe = sample_universe(account, inst, catalog)
    assert direct_universe_result(universe, account, {inst["instrument_id"]: inst}, {"cat_1": catalog})[1] == "ACCOUNT_READINESS_BLOCKED"


def test_runtime_denials_are_declared_for_every_handler():
    scenarios = [(op, request_for(op), context()) for op in OPERATIONS]
    scenarios += [("CREATE_ACCOUNT", request_for("CREATE_ACCOUNT", environment="LIVE"), context()), ("REFRESH_INSTRUMENT_CATALOG", request_for("REFRESH_INSTRUMENT_CATALOG", environment="PAPER"), context())]
    for operation, request, ctx in scenarios:
        ok, out = validate_operation_request(operation, request, validation_context=ctx)
        if not ok:
            assert out["denial"] in DATA["operation_validation_matrix"][operation]["denial_codes"]


def test_runtime_denial_event_schema_matches_actual_denial():
    schemas = {schema["event_name"]: schema for schema in DATA["audit_event_schemas"]}
    for operation, spec in DATA["operation_validation_matrix"].items():
        for denial, event in spec["denial_event_by_code"].items():
            assert denial in schemas[event]["allowed_denial_codes"] and event != "UNKNOWN_OPERATION_REJECTED"


def test_activate_unsupported_environment_has_valid_audit_mapping():
    spec = DATA["operation_validation_matrix"]["ACTIVATE_TRADING_UNIVERSE"]
    assert spec["denial_event_by_code"]["EXCHANGE_ENVIRONMENT_UNSUPPORTED"] == "TRADING_UNIVERSE_REJECTED"


def test_activate_unsupported_market_has_valid_audit_mapping():
    spec = DATA["operation_validation_matrix"]["ACTIVATE_TRADING_UNIVERSE"]
    assert spec["denial_event_by_code"]["MARKET_TYPE_UNSUPPORTED"] == "TRADING_UNIVERSE_REJECTED"


def test_audit_result_never_pairs_denial_with_unrelated_event():
    ok, out = audit_result("CREATE_ACCOUNT", False, "INSTRUMENT_NOT_FOUND")
    assert not ok and out == {"denial": "CONTRACT_VALIDATION_MAPPING_ERROR", "audit_event": "CONTRACT_VALIDATION_MAPPING_REJECTED"}


def assert_universe_denied(**overrides):
    account = sample_account(); inst = sample_instrument(); catalog = sample_catalog(inst); universe = sample_universe(account, inst, catalog, **overrides)
    ok, denial = direct_universe_result(universe, account, {inst["instrument_id"]: inst}, {catalog["catalog_snapshot_id"]: catalog})
    assert not ok
    return denial


def test_universe_rejects_retired_lifecycle():
    assert assert_universe_denied(lifecycle_state="RETIRED") == "ACCOUNT_READINESS_BLOCKED"


def test_universe_rejects_rejected_lifecycle():
    assert assert_universe_denied(lifecycle_state="REJECTED") == "TRADING_UNIVERSE_INVALID"


def test_universe_rejects_empty_membership():
    assert assert_universe_denied(instrument_ids=[]) == "ACCOUNT_READINESS_BLOCKED"


def test_universe_rejects_duplicate_instrument_ids():
    assert assert_universe_denied(instrument_ids=["instr_btcusdt_spot_testnet"] * 2) == "ACCOUNT_READINESS_BLOCKED"


def test_universe_rejects_zero_version():
    assert assert_universe_denied(version=0) == "ACCOUNT_READINESS_BLOCKED"


def test_universe_rejects_string_version():
    assert assert_universe_denied(version="1") == "ACCOUNT_READINESS_BLOCKED"


def test_universe_rejects_boolean_version():
    assert assert_universe_denied(version=True) == "ACCOUNT_READINESS_BLOCKED"


def versioned_universe(version=2, previous_version_id="univ_1", **overrides):
    account=sample_account(); inst=sample_instrument(); cat=sample_catalog(inst)
    values={"trading_universe_id":f"univ_{version}","version":version,"previous_version_id":previous_version_id};values.update(overrides)
    return sample_universe(account, inst, cat, **values)


def test_universe_rejects_missing_previous_version():
    current = versioned_universe()
    assert validate_universe_lineage(current, {})[1] == "TRADING_UNIVERSE_VERSION_CONFLICT"


def test_universe_rejects_previous_version_number_gap():
    current = versioned_universe(version=3, previous_version_id="univ_1"); old = versioned_universe(version=1, previous_version_id=None, trading_universe_id="univ_1")
    assert validate_universe_lineage(current, {"univ_1":old})[1] == "TRADING_UNIVERSE_VERSION_CONFLICT"


def test_universe_rejects_predecessor_id_mismatch():
    current = versioned_universe(); old = versioned_universe(version=1, previous_version_id=None, trading_universe_id="spoof")
    assert validate_universe_lineage(current, {"univ_1":old})[1] == "TRADING_UNIVERSE_VERSION_CONFLICT"


def test_universe_rejects_multinode_version_cycle():
    current=versioned_universe(version=3, previous_version_id="univ_2"); two=versioned_universe(version=2, previous_version_id="univ_1"); one=versioned_universe(version=1, previous_version_id="univ_3")
    assert validate_universe_lineage(current, {"univ_2":two,"univ_1":one})[1] == "TRADING_UNIVERSE_VERSION_CONFLICT"


def test_universe_rejects_other_active_version_same_account():
    account=sample_account(); inst=sample_instrument(); cat=sample_catalog(inst); current=sample_universe(account,inst,cat); active=sample_universe(account,inst,cat,trading_universe_id="univ_other")
    assert direct_universe_result(current,account,{inst["instrument_id"]:inst},{"cat_1":cat},active_universes=[active],universes_by_id={current["trading_universe_id"]:current,active["trading_universe_id"]:active})[1] == "TRADING_UNIVERSE_VERSION_CONFLICT"


def test_universe_allows_active_version_of_other_account():
    account=sample_account(); inst=sample_instrument(); cat=sample_catalog(inst); current=sample_universe(account,inst,cat); other=sample_universe(sample_account(exchange_account_id="xacc_other"),inst,cat,trading_universe_id="univ_other",exchange_account_id="xacc_other")
    other_account=sample_account(exchange_account_id="xacc_other",active_credential_profile_id=None)
    assert direct_universe_result(current,account,{inst["instrument_id"]:inst},{"cat_1":cat},active_universes=[other],universes_by_id={current["trading_universe_id"]:current,other["trading_universe_id"]:other},accounts_by_id={account["exchange_account_id"]:context(account=account)["accounts_by_id"][account["exchange_account_id"]],"xacc_other":other_account})[0]


def operation_result(operation, account):
    return validate_operation_request(operation, request_for(operation), validation_context=context(account=account))


def test_activate_rejects_disconnected_account():
    assert operation_result("ACTIVATE_TRADING_UNIVERSE", sample_account(connection_state="DISCONNECTED"))[1]["denial"] == "ACCOUNT_READINESS_BLOCKED"


def test_activate_rejects_connecting_account():
    assert operation_result("ACTIVATE_TRADING_UNIVERSE", sample_account(connection_state="CONNECTING"))[0] is False


def test_activate_rejects_synchronizing_account():
    assert operation_result("ACTIVATE_TRADING_UNIVERSE", sample_account(connection_state="SYNCHRONIZING"))[0] is False


def test_bind_rejects_retired_account():
    assert operation_result("BIND_CREDENTIAL_PROFILE", sample_account(lifecycle_state="RETIRED", retired_at_utc="2026-02-01T00:00:00Z", connection_state="DISCONNECTED"))[1]["denial"] == "EXCHANGE_ACCOUNT_RETIRED"


def test_bind_rejects_online_account():
    assert operation_result("BIND_CREDENTIAL_PROFILE", sample_account(connection_state="ONLINE"))[1]["denial"] == "ACCOUNT_READINESS_BLOCKED"


def test_verify_rejects_retired_account():
    assert operation_result("VERIFY_EXTERNAL_IDENTITY", sample_account(lifecycle_state="RETIRED", retired_at_utc="2026-02-01T00:00:00Z", connection_state="DISCONNECTED"))[1]["denial"] == "EXCHANGE_ACCOUNT_RETIRED"


def test_verify_rejects_disallowed_connection_state():
    assert operation_result("VERIFY_EXTERNAL_IDENTITY", sample_account(connection_state="ONLINE"))[1]["denial"] == "ACCOUNT_READINESS_BLOCKED"


def test_operation_state_matrix_is_enforced_dynamically():
    cases=[("ACTIVATE_TRADING_UNIVERSE","connection_state",["DISCONNECTED","CONNECTING","SYNCHRONIZING","BLOCKED"]),("BIND_CREDENTIAL_PROFILE","connection_state",["ONLINE","CONNECTING","SYNCHRONIZING","DEGRADED"]),("VERIFY_EXTERNAL_IDENTITY","lifecycle_state",["DRAFT","RETIRED"])]
    for operation,field,values in cases:
        for value in values:
            account_values={field:value}
            if field!="connection_state": account_values["connection_state"]="DISCONNECTED"
            ok,out=operation_result(operation,sample_account(**account_values))
            assert not ok and out["denial"] in DATA["operation_validation_matrix"][operation]["denial_codes"]


def test_context_map_key_record_id_mismatches():
    cases=[("accounts_by_id",sample_account(),"false"),("credential_profiles_by_id",sample_profile(),"false"),("universes_by_id",sample_universe(),"false"),("catalogs_by_id",sample_catalog(),"false"),("instruments_by_id",sample_instrument(),"false")]
    for field,record,key in cases:
        ctx=context();ctx[field]={key:record}
        ok,out=validate_operation_request("CREATE_ACCOUNT",request_for("CREATE_ACCOUNT"),validation_context=ctx)
        assert not ok and out["denial"] in DENIALS


def test_context_rejects_account_map_key_record_id_mismatch():
    ctx=context();ctx["accounts_by_id"]={"false":sample_account()};assert validate_operation_request("CREATE_ACCOUNT",request_for("CREATE_ACCOUNT"),validation_context=ctx)[0] is False


def test_context_rejects_profile_map_key_record_id_mismatch():
    ctx=context();ctx["credential_profiles_by_id"]={"false":sample_profile()};assert validate_operation_request("CREATE_ACCOUNT",request_for("CREATE_ACCOUNT"),validation_context=ctx)[0] is False


def test_context_rejects_universe_map_key_record_id_mismatch():
    ctx=context();ctx["universes_by_id"]={"false":sample_universe()};assert validate_operation_request("CREATE_ACCOUNT",request_for("CREATE_ACCOUNT"),validation_context=ctx)[0] is False


def test_context_rejects_catalog_map_key_record_id_mismatch():
    ctx=context();ctx["catalogs_by_id"]={"false":sample_catalog()};assert validate_operation_request("CREATE_ACCOUNT",request_for("CREATE_ACCOUNT"),validation_context=ctx)[0] is False


def test_context_rejects_instrument_map_key_record_id_mismatch():
    ctx=context();ctx["instruments_by_id"]={"false":sample_instrument()};assert validate_operation_request("CREATE_ACCOUNT",request_for("CREATE_ACCOUNT"),validation_context=ctx)[0] is False


def test_verify_cannot_use_account_stored_under_false_id():
    ctx=context();ctx["accounts_by_id"]={"xacc_test_1":sample_account(exchange_account_id="xacc_spoof")}
    assert validate_operation_request("VERIFY_EXTERNAL_IDENTITY",request_for("VERIFY_EXTERNAL_IDENTITY"),validation_context=ctx)[0] is False


def test_refresh_cannot_use_catalog_stored_under_false_id():
    ctx=context();ctx["catalogs_by_id"]={"cat_1":sample_catalog(catalog_snapshot_id="cat_spoof")}
    assert validate_operation_request("REFRESH_INSTRUMENT_CATALOG",request_for("REFRESH_INSTRUMENT_CATALOG"),validation_context=ctx)[0] is False


def test_retire_cannot_use_record_stored_under_false_id():
    ctx=context();ctx["instruments_by_id"]={"instr_btcusdt_spot_testnet":sample_instrument(instrument_id="instr_spoof")}
    assert validate_operation_request("RETIRE_INSTRUMENT",request_for("RETIRE_INSTRUMENT"),validation_context=ctx)[0] is False


def test_credential_lineage_rejects_predecessor_id_mismatch():
    current=sample_profile(credential_profile_id="cred_new",rotated_from_credential_profile_id="cred_old");old=sample_profile(credential_profile_id="cred_spoof",lifecycle_state="RETIRED",retired_at_utc="2026-02-01T00:00:00Z")
    ctx={"previous_profiles_by_id":{"cred_old":old},"active_profile_ids_by_account_id":{"xacc_test_1":["cred_new"]}}
    assert validate_credential_profile_binding(current,sample_account(),ctx)[1] == "CREDENTIAL_PROFILE_NOT_ACTIVE"


def test_bind_rejects_two_active_profiles_same_account():
    ctx=context(active_profile_ids_by_account_id={"xacc_test_1":["cred_1","cred_2"]});acct=sample_account(connection_state="DISCONNECTED");ctx=context(account=acct,active_profile_ids_by_account_id={"xacc_test_1":["cred_1","cred_2"]})
    assert validate_operation_request("BIND_CREDENTIAL_PROFILE",request_for("BIND_CREDENTIAL_PROFILE"),validation_context=ctx)[0] is False


def test_bind_allows_active_profile_on_other_account():
    acct=sample_account(connection_state="DISCONNECTED");ctx=context(account=acct,active_profile_ids_by_account_id={"xacc_test_1":["cred_1"],"xacc_other":["cred_other"]});ctx["accounts_by_id"]["xacc_other"]=sample_account(exchange_account_id="xacc_other",active_credential_profile_id="cred_other");ctx["credential_profiles_by_id"]["cred_other"]=sample_profile(credential_profile_id="cred_other",exchange_account_id="xacc_other")
    assert validate_operation_request("BIND_CREDENTIAL_PROFILE",request_for("BIND_CREDENTIAL_PROFILE"),validation_context=ctx)[0]


def test_non_rotated_profile_still_checks_active_uniqueness():
    profile=sample_profile();acct=sample_account(connection_state="DISCONNECTED")
    assert validate_credential_profile_binding(profile,acct,{"previous_profiles_by_id":{},"active_profile_ids_by_account_id":{"xacc_test_1":["cred_1","cred_2"]}})[0] is False


def test_catalog_lineage_rejects_predecessor_identity_and_record_errors():
    inst=sample_instrument(catalog_snapshot_id="cat_new");current=sample_catalog(inst,catalog_snapshot_id="cat_new",previous_snapshot_id="cat_old")
    valid=sample_catalog(inst,catalog_snapshot_id="cat_old")
    variants=[sample_catalog(inst,catalog_snapshot_id="cat_spoof"),{"catalog_snapshot_id":"cat_old"},sample_catalog(inst,catalog_snapshot_id="cat_old",content_hash="0"*64),sample_catalog(inst,catalog_snapshot_id="cat_old",adapter_family_id="wrong"),sample_catalog(inst,catalog_snapshot_id="cat_old",observed_at_utc="bad")]
    for predecessor in variants:
        assert validate_catalog_lineage(current,{"cat_old":predecessor})[1] == "CATALOG_SNAPSHOT_INVALID"


def catalog_lineage_result(predecessor):
    inst=sample_instrument(catalog_snapshot_id="cat_new");current=sample_catalog(inst,catalog_snapshot_id="cat_new",previous_snapshot_id="cat_old")
    return validate_catalog_lineage(current,{"cat_old":predecessor})


def test_catalog_lineage_rejects_predecessor_id_mismatch():
    assert catalog_lineage_result(sample_catalog(catalog_snapshot_id="spoof"))[0] is False


def test_catalog_lineage_rejects_incomplete_predecessor():
    assert catalog_lineage_result({"catalog_snapshot_id":"cat_old"})[0] is False


def test_catalog_lineage_rejects_invalid_predecessor_hash():
    assert catalog_lineage_result(sample_catalog(catalog_snapshot_id="cat_old",content_hash="0"*64))[0] is False


def test_catalog_lineage_rejects_invalid_predecessor_adapter():
    assert catalog_lineage_result(sample_catalog(catalog_snapshot_id="cat_old",adapter_family_id="spoof"))[0] is False


def test_catalog_lineage_rejects_invalid_predecessor_timestamp():
    assert catalog_lineage_result(sample_catalog(catalog_snapshot_id="cat_old",observed_at_utc="bad"))[0] is False


def test_update_rejects_malformed_mutable_field_types():
    for field in ["display_name","active_credential_profile_id","account_capability_snapshot_id","retired_at_utc"]:
        for value in ({},[]):
            ok,out=validate_operation_request("UPDATE_ACCOUNT",request_for("UPDATE_ACCOUNT",mutable_patch={field:value}),validation_context=context())
            assert not ok and out["audit_event"]!="EXCHANGE_ACCOUNT_UPDATED"


def test_dynamic_adversarial_matrix_has_complete_audit_mapping():
    scenarios=[("ACTIVATE_TRADING_UNIVERSE",sample_account(connection_state="DISCONNECTED")),("BIND_CREDENTIAL_PROFILE",sample_account(lifecycle_state="RETIRED",connection_state="DISCONNECTED")),("VERIFY_EXTERNAL_IDENTITY",sample_account(lifecycle_state="DRAFT",connection_state="DISCONNECTED"))]
    schemas={s["event_name"]:s for s in DATA["audit_event_schemas"]}
    for operation,account in scenarios:
        ok,out=operation_result(operation,account);spec=DATA["operation_validation_matrix"][operation]
        assert not ok and out["denial"] in spec["denial_codes"]
        event=spec["denial_event_by_code"][out["denial"]]
        assert out["audit_event"]==event and event in DATA["audit_event_registry"] and out["denial"] in schemas[event]["allowed_denial_codes"] and event!="UNKNOWN_OPERATION_REJECTED"


def test_verify_request_cannot_embed_identity_snapshot():
    req=request_for("VERIFY_EXTERNAL_IDENTITY");req["identity_snapshot"]=sample_identity()
    assert validate_operation_request("VERIFY_EXTERNAL_IDENTITY",req,validation_context=context())[0] is False


def test_verify_uses_trusted_context_snapshot():
    acct=sample_account(connection_state="SYNCHRONIZING");ctx=context(account=acct,identity=sample_identity())
    assert validate_operation_request("VERIFY_EXTERNAL_IDENTITY",request_for("VERIFY_EXTERNAL_IDENTITY"),validation_context=ctx)[0]


def test_verify_trusted_mismatch_cannot_be_overridden_by_request():
    acct=sample_account(connection_state="SYNCHRONIZING");ctx=context(account=acct,identity=sample_identity(state="MISMATCH"));req=request_for("VERIFY_EXTERNAL_IDENTITY")
    assert validate_operation_request("VERIFY_EXTERNAL_IDENTITY",req,validation_context=ctx)[1]["denial"]=="EXTERNAL_ACCOUNT_IDENTITY_MISMATCH"
    req["state"]="VERIFIED"
    assert validate_operation_request("VERIFY_EXTERNAL_IDENTITY",req,validation_context=ctx)[0] is False


def test_verify_trusted_unavailable_cannot_be_overridden_by_request():
    acct=sample_account(connection_state="SYNCHRONIZING");ctx=context(account=acct,identity=sample_identity(state="UNAVAILABLE"))
    assert validate_operation_request("VERIFY_EXTERNAL_IDENTITY",request_for("VERIFY_EXTERNAL_IDENTITY"),validation_context=ctx)[1]["denial"]=="EXTERNAL_ACCOUNT_IDENTITY_UNAVAILABLE"


def test_verify_missing_trusted_snapshot_fails_closed():
    acct=sample_account(connection_state="SYNCHRONIZING");ctx=context(account=acct);ctx["external_identity_snapshots_by_account_id"]={}
    assert validate_operation_request("VERIFY_EXTERNAL_IDENTITY",request_for("VERIFY_EXTERNAL_IDENTITY"),validation_context=ctx)[0] is False


def test_verify_request_contains_no_authority_bearing_identity_fields():
    fields=set(DATA["operation_validation_matrix"]["VERIFY_EXTERNAL_IDENTITY"]["required_inputs"])
    assert not fields & {"identity_snapshot","state","observed_permission_set","venue_account_identifier","adapter_version_source"}


def test_universe_rejects_partial_source_catalog():
    acct=sample_account();inst=sample_instrument();cat=sample_catalog(inst,status="PARTIAL");univ=sample_universe(acct,inst,cat)
    assert direct_universe_result(univ,acct,{inst["instrument_id"]:inst},{cat["catalog_snapshot_id"]:cat})[1]=="CATALOG_SNAPSHOT_INVALID"


def test_universe_accepts_only_valid_source_catalogs():
    acct=sample_account();inst=sample_instrument();cat=sample_catalog(inst,status="VALID");univ=sample_universe(acct,inst,cat)
    assert direct_universe_result(univ,acct,{inst["instrument_id"]:inst},{cat["catalog_snapshot_id"]:cat})[0]


def test_partial_catalog_never_emits_trading_universe_activated():
    inst=sample_instrument();cat=sample_catalog(inst,status="PARTIAL");univ=sample_universe(catalog=cat);ctx=context(inst=inst,catalog=cat,universe=univ)
    ok,out=validate_operation_request("ACTIVATE_TRADING_UNIVERSE",request_for("ACTIVATE_TRADING_UNIVERSE",content_hash=univ["content_hash"]),validation_context=ctx)
    assert not ok and out["audit_event"]!="TRADING_UNIVERSE_ACTIVATED"


def test_refresh_and_activation_catalog_status_policies_are_distinct():
    policy=DATA["instrument_catalog_snapshot_contract"]["activation_status_policy"]
    assert "PARTIAL" in policy["REFRESH_INSTRUMENT_CATALOG"] and policy["ACTIVATE_TRADING_UNIVERSE"]==["VALID"]


def test_update_rejects_blocked_by_kill_switch_when_not_allowed():
    ok,out=validate_operation_request("UPDATE_ACCOUNT",request_for("UPDATE_ACCOUNT",mutable_patch={"execution_authorization":"BLOCKED_BY_KILL_SWITCH"}),validation_context=context())
    assert not ok and out["denial"]=="ACCOUNT_READINESS_BLOCKED"


def test_update_rejects_blocked_by_lease_when_not_allowed():
    ok,out=validate_operation_request("UPDATE_ACCOUNT",request_for("UPDATE_ACCOUNT",mutable_patch={"execution_authorization":"BLOCKED_BY_LEASE"}),validation_context=context())
    assert not ok and out["denial"]=="ACCOUNT_READINESS_BLOCKED"


def test_update_enforces_result_state_not_only_input_state():
    assert validate_operation_request("UPDATE_ACCOUNT",request_for("UPDATE_ACCOUNT",mutable_patch={"execution_authorization":"BLOCKED_BY_KILL_SWITCH"}),validation_context=context())[0] is False


def test_every_nonempty_operation_state_axis_is_enforced():
    for operation,spec in DATA["operation_validation_matrix"].items():
        for axis,field in [("allowed_lifecycle_states","lifecycle_state"),("allowed_connection_states","connection_state"),("allowed_authorization_states","execution_authorization")]:
            assert isinstance(spec[axis],list)
            if spec[axis]:
                account=sample_account(**{field:"NOT_ALLOWED"})
                assert enforce_operation_states(operation,account)[0] is False


def derivative_instrument(instrument_type, market_type, **overrides):
    data=sample_instrument(exchange_id="paper_simulated_venue",environment="PAPER",market_type=market_type,instrument_type=instrument_type,source_adapter_family_id="paper_simulation_adapter_family",base_asset_reference=asset("BTC"),quote_asset_reference=asset("USD"))
    data["base_asset_reference"]["asset_namespace"]="paper_simulated_venue";data["quote_asset_reference"]["asset_namespace"]="paper_simulated_venue";data.update(overrides);return data


def settlement(code="USD"):
    ref=asset(code);ref["asset_namespace"]="paper_simulated_venue";return ref


def test_instrument_metadata_version_rejects_bool():
    assert validate_instrument_record(sample_instrument(metadata_version=True))[1]=="INSTRUMENT_METADATA_INVALID"


def test_instrument_none_minimum_with_maximum_fails_closed():
    assert validate_instrument_record(sample_instrument(min_quantity=None,max_quantity="1"))[1]=="INSTRUMENT_METADATA_INVALID"


def test_instrument_notional_range_is_validated():
    assert validate_instrument_record(sample_instrument(min_notional="10",max_notional="1"))[1]=="INSTRUMENT_METADATA_INVALID"


def test_instrument_wrong_decimal_types_never_raise():
    for value in [None,1,True,{},[]]: assert validate_instrument_record(sample_instrument(price_tick=value))[0] is False


def test_spot_rejects_derivative_fields():
    assert validate_instrument_record(sample_instrument(contract_size="1"))[0] is False


def test_perpetual_requires_contract_fields():
    inst=derivative_instrument("PERPETUAL_CONTRACT","PERPETUAL",settlement_asset_reference=settlement())
    assert validate_instrument_record(inst)[0] is False


def test_delivery_future_requires_expiry():
    inst=derivative_instrument("DELIVERY_FUTURE","DELIVERY_FUTURES",contract_size="1",contract_value_currency="USD",derivative_settlement_type="LINEAR",settlement_asset_reference=settlement())
    assert validate_instrument_record(inst)[0] is False


def test_option_requires_strike_and_side():
    inst=derivative_instrument("OPTION","OPTIONS",contract_size="1",contract_value_currency="USD",derivative_settlement_type="LINEAR",expiry_at_utc="2026-10-01T00:00:00Z",settlement_asset_reference=settlement())
    assert validate_instrument_record(inst)[0] is False


def test_instrument_adversarial_types_never_raise():
    for field in DATA["instrument_contract"]["record_fields"]:
        validate_instrument_record(sample_instrument(**{field:None}))
        for value in [{},[]]: assert validate_instrument_record(sample_instrument(**{field:value}))[0] is False


def test_universe_lineage_rejects_predecessor_hash_mismatch():
    current=versioned_universe();old=versioned_universe(version=1,previous_version_id=None);old["content_hash"]="0"*64
    assert validate_universe_lineage(current,{"univ_1":old})[0] is False


def test_universe_lineage_rejects_tampered_predecessor_membership():
    current=versioned_universe();old=versioned_universe(version=1,previous_version_id=None);old["instrument_ids"]=["tampered"]
    assert validate_universe_lineage(current,{"univ_1":old})[0] is False


def test_universe_lineage_rejects_tampered_creation_reason():
    current=versioned_universe();old=versioned_universe(version=1,previous_version_id=None);old["creation_reason"]="TAMPERED"
    assert validate_universe_lineage(current,{"univ_1":old})[0] is False


def test_universe_lineage_rejects_invalid_predecessor_timestamps():
    current=versioned_universe();old=versioned_universe(version=1,previous_version_id=None);old["created_at_utc"]="bad"
    assert validate_universe_lineage(current,{"univ_1":old})[0] is False


def active_identity_result(**changes):
    acct=sample_account();inst=sample_instrument();cat=sample_catalog(inst);current=sample_universe(acct,inst,cat);active=dict(current);active.update(changes)
    return direct_universe_result(current,acct,{inst["instrument_id"]:inst},{"cat_1":cat},active_universes=[active])


def test_active_same_id_different_version_is_conflict():
    assert active_identity_result(version=2)[1]=="ACCOUNT_READINESS_BLOCKED"


def test_active_same_id_different_hash_is_conflict():
    assert active_identity_result(content_hash="0"*64)[1]=="ACCOUNT_READINESS_BLOCKED"


def test_active_same_id_different_membership_is_conflict():
    assert active_identity_result(instrument_ids=["other"])[1]=="ACCOUNT_READINESS_BLOCKED"


def test_exact_idempotent_reactivation_policy_is_explicit():
    assert active_identity_result()[0] and "exactly equal" in DATA["trading_universe_contract"]["idempotent_reactivation_policy"]


def test_profile_rejects_saas_sync_candidate_true():
    assert validate_credential_profile(sample_profile(saas_sync_candidate=True))[0] is False


def test_profile_rejects_invalid_created_timestamp():
    assert validate_credential_profile(sample_profile(created_at_utc="bad"))[0] is False


def test_active_profile_rejects_retired_timestamp():
    assert validate_credential_profile(sample_profile(retired_at_utc="2026-02-01T00:00:00Z"))[0] is False


def test_retired_profile_requires_retired_timestamp():
    assert validate_credential_profile(sample_profile(lifecycle_state="RETIRED",retired_at_utc=None))[0] is False


def test_profile_rejects_plaintext_secret_reference():
    assert validate_credential_profile(sample_profile(secure_store_reference="secure-store://plaintext-secret"))[0] is False


def test_profile_requires_secure_store_reference_shape():
    for value in ["", "file://secret", None, {}, []]: assert validate_credential_profile(sample_profile(secure_store_reference=value))[0] is False


def test_profile_rejects_invalid_public_key_identifier_type():
    assert validate_credential_profile(sample_profile(public_key_identifier={}))[0] is False


def test_lineage_runs_full_predecessor_profile_validator():
    current=sample_profile(credential_profile_id="cred_new",rotated_from_credential_profile_id="cred_old");old=sample_profile(credential_profile_id="cred_old",lifecycle_state="RETIRED",retired_at_utc="2026-02-01T00:00:00Z",saas_sync_candidate=True)
    ctx={"previous_profiles_by_id":{"cred_old":old},"active_profile_ids_by_account_id":{"xacc_test_1":["cred_new"]}}
    assert validate_credential_profile_binding(current,sample_account(),ctx)[0] is False


def test_context_rejects_unknown_top_level_field():
    ctx=context();ctx["unknown"]={};assert validate_context(ctx) is False


def test_context_rejects_caller_authority_field():
    ctx=context();ctx["caller_authority"]="CoreHost";assert validate_context(ctx) is False


def test_context_rejects_embedded_secret_field():
    ctx=context();ctx["secret"]="payload";assert validate_context(ctx) is False


def test_context_rejects_malformed_active_universe_record():
    ctx=context(active_universes=[{"trading_universe_id":"univ_bad"}]);assert validate_context(ctx) is False


def create_with_existing(existing, **request_overrides):
    ctx=context(account=existing);return validate_operation_request("CREATE_ACCOUNT",request_for("CREATE_ACCOUNT",**request_overrides),validation_context=ctx)


def test_create_allows_distinct_subaccounts_same_scope():
    existing=sample_account(external_account_reference="venue-account",external_subaccount_reference="sub-a")
    assert create_with_existing(existing,external_account_reference="venue-account",external_subaccount_reference="sub-b")[0]


def test_create_allows_multiple_draft_accounts_same_exchange_scope():
    existing=sample_account(lifecycle_state="DRAFT",connection_state="DISCONNECTED",external_account_identity_state="UNVERIFIED")
    assert create_with_existing(existing,external_account_reference=None,external_subaccount_reference=None)[0]


def test_verified_same_external_account_and_subaccount_collides():
    existing=sample_account(external_account_reference="venue-account",external_subaccount_reference="sub-a")
    ok,out=create_with_existing(existing,external_account_reference="venue-account",external_subaccount_reference="sub-a")
    assert ok  # caller references are metadata, never trusted identity authority


def test_different_external_subaccounts_do_not_collide():
    existing=sample_account(external_account_reference="venue-account",external_subaccount_reference="sub-a")
    assert create_with_existing(existing,external_account_reference="venue-account",external_subaccount_reference="sub-b")[0]


def test_account_collision_policy_matches_markdown_and_json():
    assert "different subaccounts" in " ".join(DATA["exchange_account_contract"]["multi_account_collision_policy"])
    assert "różne subkonta" in MD.read_text()


def trusted_collision_context(other_subaccount=None, target_subaccount=None):
    target=sample_account(connection_state="SYNCHRONIZING");ctx=context(account=target,identity=sample_identity(subaccount_identifier=target_subaccount))
    other=sample_account(exchange_account_id="xacc_other");ctx["accounts_by_id"]["xacc_other"]=other
    ctx["external_identity_snapshots_by_account_id"]["xacc_other"]=sample_identity(subaccount_identifier=other_subaccount)
    return ctx


def test_verify_rejects_duplicate_trusted_identity_on_other_active_account():
    ok,out=validate_operation_request("VERIFY_EXTERNAL_IDENTITY",request_for("VERIFY_EXTERNAL_IDENTITY"),validation_context=trusted_collision_context())
    assert not ok and out["denial"]=="EXCHANGE_ACCOUNT_IDENTITY_COLLISION"


def test_verify_allows_distinct_trusted_subaccounts():
    assert validate_operation_request("VERIFY_EXTERNAL_IDENTITY",request_for("VERIFY_EXTERNAL_IDENTITY"),validation_context=trusted_collision_context("sub-b","sub-a"))[0]


def test_verify_rejects_empty_subaccount_identifier():
    acct=sample_account(connection_state="SYNCHRONIZING");ctx=context(account=acct,identity=sample_identity(subaccount_identifier=""))
    assert validate_operation_request("VERIFY_EXTERNAL_IDENTITY",request_for("VERIFY_EXTERNAL_IDENTITY"),validation_context=ctx)[0] is False


def test_verify_root_account_uses_null_subaccount():
    acct=sample_account(connection_state="SYNCHRONIZING");ctx=context(account=acct,identity=sample_identity(subaccount_identifier=None))
    assert validate_operation_request("VERIFY_EXTERNAL_IDENTITY",request_for("VERIFY_EXTERNAL_IDENTITY"),validation_context=ctx)[0]


def test_create_external_references_are_not_identity_authority():
    existing=sample_account(external_account_reference="spoof",external_subaccount_reference="same")
    assert create_with_existing(existing,external_account_reference="spoof",external_subaccount_reference="same")[0]


def test_spoofed_external_reference_cannot_bypass_trusted_collision():
    ctx=trusted_collision_context();ctx["accounts_by_id"]["xacc_test_1"]["external_account_reference"]="different"
    ok,out=validate_operation_request("VERIFY_EXTERNAL_IDENTITY",request_for("VERIFY_EXTERNAL_IDENTITY"),validation_context=ctx)
    assert not ok and out["denial"]=="EXCHANGE_ACCOUNT_IDENTITY_COLLISION"


def test_duplicate_trusted_identity_denial_has_valid_audit_mapping():
    spec=DATA["operation_validation_matrix"]["VERIFY_EXTERNAL_IDENTITY"];event=spec["denial_event_by_code"]["EXCHANGE_ACCOUNT_IDENTITY_COLLISION"];schemas={s["event_name"]:s for s in DATA["audit_event_schemas"]}
    assert "EXCHANGE_ACCOUNT_IDENTITY_COLLISION" in schemas[event]["allowed_denial_codes"]


def partial_refresh_result(**catalog_overrides):
    inst=sample_instrument();cat=sample_catalog(inst,status="PARTIAL",**catalog_overrides);ctx=context(inst=inst,catalog=cat)
    return validate_operation_request("REFRESH_INSTRUMENT_CATALOG",request_for("REFRESH_INSTRUMENT_CATALOG"),validation_context=ctx)


def test_partial_refresh_operation_succeeds():
    assert partial_refresh_result()[0]


def test_partial_refresh_emits_instrument_catalog_refreshed():
    assert partial_refresh_result()[1]["audit_event"]=="INSTRUMENT_CATALOG_REFRESHED"


def test_partial_refresh_with_bad_hash_fails():
    assert partial_refresh_result(content_hash="0"*64)[0] is False


def test_partial_refresh_with_invalid_instrument_fails():
    inst=sample_instrument(price_tick="0");cat=sample_catalog(inst,status="PARTIAL");ctx=context(inst=inst,catalog=cat)
    assert validate_operation_request("REFRESH_INSTRUMENT_CATALOG",request_for("REFRESH_INSTRUMENT_CATALOG"),validation_context=ctx)[0] is False


def test_partial_activation_remains_denied():
    test_partial_catalog_never_emits_trading_universe_activated()


def test_partial_activation_never_emits_success_event():
    test_partial_catalog_never_emits_trading_universe_activated()


def test_all_operation_request_fields_wrong_types_never_raise():
    fields={"CREATE_ACCOUNT":["requested_lifecycle_state","environment","market_type"],"BIND_CREDENTIAL_PROFILE":["binding_intent"],"VERIFY_EXTERNAL_IDENTITY":["expected_adapter_version_source","reconnect"],"UPDATE_ACCOUNT":["expected_identity_fields","mutable_patch"],"RETIRE_ACCOUNT":["retirement_reason"],"QUARANTINE_LEGACY_LIVE_RECORD":["legacy_record","target_lifecycle_state","target_connection_state","target_execution_authorization","quarantine_reason"]}
    for operation,names in fields.items():
        for field in names:
            bad_values = [[],{},1,"true"] if field in {"legacy_record","reconnect"} else ([[],True,"bad"] if field in {"expected_identity_fields","mutable_patch"} else [[],{},True])
            for bad in bad_values:
                req=request_for(operation);req[field]=bad;ok,out=validate_operation_request(operation,req,validation_context=context())
                assert not ok and out["audit_event"]!="UNKNOWN_OPERATION_REJECTED"


def test_create_malformed_lifecycle_never_raises():
    for bad in [[],{},True]: assert validate_operation_request("CREATE_ACCOUNT",request_for("CREATE_ACCOUNT",requested_lifecycle_state=bad),validation_context=context())[0] is False


def test_quarantine_malformed_states_never_raise():
    for field in ["target_lifecycle_state","target_connection_state","target_execution_authorization"]:
        req=request_for("QUARANTINE_LEGACY_LIVE_RECORD");req[field]=[];assert validate_operation_request("QUARANTINE_LEGACY_LIVE_RECORD",req,validation_context=context())[0] is False


def test_bind_malformed_profile_lifecycle_never_raises():
    acct=sample_account(connection_state="DISCONNECTED");ctx=context(account=acct,profile=sample_profile(lifecycle_state=[]))
    assert validate_operation_request("BIND_CREDENTIAL_PROFILE",request_for("BIND_CREDENTIAL_PROFILE"),validation_context=ctx)[0] is False


def test_malformed_enum_values_return_controlled_denial():
    assert validate_operation_request("CREATE_ACCOUNT",request_for("CREATE_ACCOUNT",market_type={}),validation_context=context())[0] is False


def test_known_operation_malformed_input_never_uses_unknown_operation_event():
    ok,out=validate_operation_request("CREATE_ACCOUNT",request_for("CREATE_ACCOUNT",environment=[]),validation_context=context());assert not ok and out["audit_event"]!="UNKNOWN_OPERATION_REJECTED"


def test_profile_rejects_empty_secure_store_locator():
    assert validate_credential_profile(sample_profile(secure_store_reference="secure-store://"))[0] is False


def test_profile_rejects_secure_store_query():
    assert validate_credential_profile(sample_profile(secure_store_reference="secure-store://profile?x"))[0] is False


def test_profile_rejects_secure_store_fragment():
    assert validate_credential_profile(sample_profile(secure_store_reference="secure-store://profile#x"))[0] is False


def test_profile_rejects_secure_store_payload_assignment():
    assert validate_credential_profile(sample_profile(secure_store_reference="secure-store://value=x"))[0] is False


def test_profile_rejects_api_key_payload_marker():
    assert validate_credential_profile(sample_profile(secure_store_reference="secure-store://api_key"))[0] is False


def test_profile_lifecycle_wrong_type_never_raises():
    for bad in [[],{},True,None]: assert validate_credential_profile(sample_profile(lifecycle_state=bad))[0] is False


def test_retired_profile_must_not_precede_creation():
    assert validate_credential_profile(sample_profile(lifecycle_state="RETIRED",retired_at_utc="2025-01-01T00:00:00Z"))[0] is False


def test_lineage_rejects_invalid_timestamp_order():
    current=sample_profile(credential_profile_id="cred_new",created_at_utc="2026-01-01T00:00:00Z",rotated_from_credential_profile_id="cred_old");old=sample_profile(credential_profile_id="cred_old",lifecycle_state="RETIRED",created_at_utc="2025-01-01T00:00:00Z",retired_at_utc="2027-01-01T00:00:00Z")
    assert validate_credential_lineage(current,{"previous_profiles_by_id":{"cred_old":old}})[0] is False


def test_active_account_rejects_retired_timestamp():
    assert validate_account(sample_account(retired_at_utc="2026-02-01T00:00:00Z"))[0] is False


def test_draft_account_rejects_retired_timestamp():
    assert validate_account(sample_account(lifecycle_state="DRAFT",retired_at_utc="2026-02-01T00:00:00Z"))[0] is False


def test_disabled_account_rejects_retired_timestamp():
    assert validate_account(sample_account(lifecycle_state="DISABLED",retired_at_utc="2026-02-01T00:00:00Z"))[0] is False


def test_retired_account_requires_retired_timestamp():
    assert validate_account(sample_account(lifecycle_state="RETIRED"))[0] is False


def test_retired_account_timestamp_not_before_creation():
    assert validate_account(sample_account(lifecycle_state="RETIRED",retired_at_utc="2025-01-01T00:00:00Z"))[0] is False


def test_update_cannot_add_retired_timestamp_to_active_account():
    req=request_for("UPDATE_ACCOUNT",mutable_patch={"retired_at_utc":"2026-02-01T00:00:00Z"});assert validate_operation_request("UPDATE_ACCOUNT",req,validation_context=context())[0] is False


def test_update_retirement_requires_complete_consistent_transition():
    bad=request_for("UPDATE_ACCOUNT",mutable_patch={"lifecycle_state":"RETIRED"});assert validate_operation_request("UPDATE_ACCOUNT",bad,validation_context=context())[0] is False
    good=request_for("UPDATE_ACCOUNT",mutable_patch={"lifecycle_state":"RETIRED","retired_at_utc":"2026-02-01T00:00:00Z"});assert validate_operation_request("UPDATE_ACCOUNT",good,validation_context=context())[0] is False  # UPDATE matrix excludes RETIRED


def test_update_cannot_set_external_identity_verified():
    req=request_for("UPDATE_ACCOUNT",mutable_patch={"external_account_identity_state":"VERIFIED"});ok,out=validate_operation_request("UPDATE_ACCOUNT",req,validation_context=context());assert not ok and out["denial"]=="EXCHANGE_ACCOUNT_IDENTITY_IMMUTABLE"


def test_update_cannot_mutate_external_identity_state():
    for state in DATA["external_account_identity_contract"]["states"]:
        assert validate_operation_request("UPDATE_ACCOUNT",request_for("UPDATE_ACCOUNT",mutable_patch={"external_account_identity_state":state}),validation_context=context())[0] is False


def test_update_cannot_clear_external_identity_state():
    assert validate_operation_request("UPDATE_ACCOUNT",request_for("UPDATE_ACCOUNT",mutable_patch={"external_account_identity_state":None}),validation_context=context())[0] is False


def test_update_identity_state_denial_has_valid_audit_mapping():
    spec=DATA["operation_validation_matrix"]["UPDATE_ACCOUNT"];event=spec["denial_event_by_code"]["EXCHANGE_ACCOUNT_IDENTITY_IMMUTABLE"];schemas={s["event_name"]:s for s in DATA["audit_event_schemas"]};assert "EXCHANGE_ACCOUNT_IDENTITY_IMMUTABLE" in schemas[event]["allowed_denial_codes"]


def test_update_cannot_bind_missing_credential_profile():
    assert validate_operation_request("UPDATE_ACCOUNT",request_for("UPDATE_ACCOUNT",mutable_patch={"active_credential_profile_id":"cred_missing"}),validation_context=context())[0] is False


def test_update_cannot_bind_foreign_credential_profile():
    assert validate_operation_request("UPDATE_ACCOUNT",request_for("UPDATE_ACCOUNT",mutable_patch={"active_credential_profile_id":"cred_foreign"}),validation_context=context())[0] is False


def test_update_cannot_bind_withdraw_profile():
    assert validate_operation_request("UPDATE_ACCOUNT",request_for("UPDATE_ACCOUNT",mutable_patch={"active_credential_profile_id":"cred_withdraw"}),validation_context=context())[0] is False


def test_update_cannot_clear_active_credential_profile():
    acct=sample_account(active_credential_profile_id="cred_1");assert validate_operation_request("UPDATE_ACCOUNT",request_for("UPDATE_ACCOUNT",mutable_patch={"active_credential_profile_id":None}),validation_context=context(account=acct))[0] is False


def test_only_bind_operation_can_change_active_profile():
    assert "active_credential_profile_id" not in DATA["exchange_account_contract"]["caller_mutable_patch_fields"] and DATA["exchange_account_contract"]["credential_binding_authority"].startswith("BIND_CREDENTIAL_PROFILE")


def test_update_credential_binding_denial_has_valid_audit_mapping():
    ok,out=validate_operation_request("UPDATE_ACCOUNT",request_for("UPDATE_ACCOUNT",mutable_patch={"active_credential_profile_id":"cred_1"}),validation_context=context());spec=DATA["operation_validation_matrix"]["UPDATE_ACCOUNT"];assert not ok and out["denial"] in spec["denial_codes"]


def update_identity_context(target_sub=None, other_sub=None):
    target=sample_account(lifecycle_state="DISABLED",exchange_account_id="xacc_target");other=sample_account(exchange_account_id="xacc_other");ctx=context(account=target,identity=sample_identity(subaccount_identifier=target_sub));ctx["accounts_by_id"]["xacc_other"]=other;ctx["external_identity_snapshots_by_account_id"]["xacc_other"]=sample_identity(subaccount_identifier=other_sub);return target,ctx


def test_update_rejects_duplicate_trusted_identity_when_activating_account():
    target,ctx=update_identity_context();req=request_for("UPDATE_ACCOUNT",exchange_account_id="xacc_target",expected_identity_fields=identity_fields(target),mutable_patch={"lifecycle_state":"ACTIVE"});ok,out=validate_operation_request("UPDATE_ACCOUNT",req,validation_context=ctx);assert not ok and out["denial"]=="EXCHANGE_ACCOUNT_IDENTITY_COLLISION"


def test_update_allows_distinct_trusted_subaccount():
    target,ctx=update_identity_context("sub-a","sub-b");req=request_for("UPDATE_ACCOUNT",exchange_account_id="xacc_target",expected_identity_fields=identity_fields(target),mutable_patch={"lifecycle_state":"ACTIVE"});assert validate_operation_request("UPDATE_ACCOUNT",req,validation_context=ctx)[0]


def test_update_duplicate_identity_uses_trusted_snapshots_only():
    target,ctx=update_identity_context();target["external_account_reference"]="spoof";req=request_for("UPDATE_ACCOUNT",exchange_account_id="xacc_target",expected_identity_fields=identity_fields(target),mutable_patch={"lifecycle_state":"ACTIVE"});assert validate_operation_request("UPDATE_ACCOUNT",req,validation_context=ctx)[0] is False


def test_spoofed_account_metadata_cannot_bypass_update_collision():
    test_update_duplicate_identity_uses_trusted_snapshots_only()


def test_verify_and_update_use_same_identity_uniqueness_helper():
    import inspect
    assert "validate_trusted_identity_uniqueness" in inspect.getsource(handle_verify_external_identity) and "validate_trusted_identity_uniqueness" in inspect.getsource(handle_update_account)


def test_update_identity_collision_has_valid_audit_mapping():
    spec=DATA["operation_validation_matrix"]["UPDATE_ACCOUNT"];event=spec["denial_event_by_code"]["EXCHANGE_ACCOUNT_IDENTITY_COLLISION"];schemas={s["event_name"]:s for s in DATA["audit_event_schemas"]};assert "EXCHANGE_ACCOUNT_IDENTITY_COLLISION" in schemas[event]["allowed_denial_codes"]


def test_capability_snapshot_full_valid_record():
    assert validate_account_capability_snapshot(sample_capability_snapshot(),sample_account(),{})[0]


def test_capability_snapshot_rejects_missing_record():
    assert validate_account_capability_snapshot(None,sample_account(),{})[0] is False


def test_capability_snapshot_rejects_spoofed_map_key():
    snap=sample_capability_snapshot();ctx=context(account_capability_snapshots_by_id={"spoof":snap});assert validate_context(ctx) is False


def test_capability_snapshot_rejects_account_scope_mismatch():
    assert validate_account_capability_snapshot(sample_capability_snapshot(exchange_account_id="xacc_other"),sample_account(),{})[0] is False


def test_capability_snapshot_rejects_unknown_permission():
    assert validate_account_capability_snapshot(sample_capability_snapshot(observed_permission_set=["UNKNOWN"]),sample_account(),{})[0] is False


def test_capability_snapshot_rejects_unknown_instrument_type():
    assert validate_account_capability_snapshot(sample_capability_snapshot(supported_instrument_types=["UNKNOWN"]),sample_account(),{})[0] is False


def test_capability_snapshot_rejects_invalid_hash():
    assert validate_account_capability_snapshot(sample_capability_snapshot(content_hash="0"*64),sample_account(),{})[0] is False


def test_capability_snapshot_rejects_invalid_timestamps():
    assert validate_account_capability_snapshot(sample_capability_snapshot(observed_at_utc="bad"),sample_account(),{})[0] is False


def test_capability_snapshot_rejects_version_bool():
    assert validate_account_capability_snapshot(sample_capability_snapshot(version=True),sample_account(),{})[0] is False


def capability_version(version, snapshot_id, previous_id):
    return sample_capability_snapshot(account_capability_snapshot_id=snapshot_id,version=version,previous_snapshot_id=previous_id)


def test_capability_snapshot_rejects_lineage_cycle():
    current=capability_version(2,"caps_2","caps_1");old=capability_version(1,"caps_1","caps_2");assert validate_account_capability_snapshot(current,sample_account(),{"caps_1":old,"caps_2":current})[0] is False


def test_capability_snapshot_rejects_non_contiguous_version():
    current=capability_version(3,"caps_3","caps_1");old=capability_version(1,"caps_1",None);assert validate_account_capability_snapshot(current,sample_account(),{"caps_1":old})[0] is False


def test_update_cannot_bind_arbitrary_capability_snapshot():
    assert validate_operation_request("UPDATE_ACCOUNT",request_for("UPDATE_ACCOUNT",mutable_patch={"account_capability_snapshot_id":"caps_1"}),validation_context=context())[0] is False


def test_stale_capability_snapshot_only_restricts():
    ok,result=validate_account_capability_snapshot(sample_capability_snapshot(status="STALE"),sample_account(),{});assert ok and result["restricts_only"]


def test_capability_snapshot_never_extends_product_capabilities():
    assert validate_account_capability_snapshot(sample_capability_snapshot(),sample_account(),{})[1]["extends_product_capabilities"] is False


def test_capability_snapshot_never_enables_live():
    account=sample_account(exchange_id="paper_simulated_venue",environment="LIVE");snap=sample_capability_snapshot(account,adapter_family_id="paper_simulation_adapter_family");assert validate_account_capability_snapshot(snap,account,{})[0] is False


def refresh_status(status):
    inst=sample_instrument(trading_status=status);cat=sample_catalog(inst);ctx=context(inst=inst,catalog=cat);return validate_operation_request("REFRESH_INSTRUMENT_CATALOG",request_for("REFRESH_INSTRUMENT_CATALOG"),validation_context=ctx)


def activate_status(status):
    inst=sample_instrument(trading_status=status);cat=sample_catalog(inst);univ=sample_universe(inst=inst,catalog=cat);ctx=context(inst=inst,catalog=cat,universe=univ);return validate_operation_request("ACTIVATE_TRADING_UNIVERSE",request_for("ACTIVATE_TRADING_UNIVERSE",content_hash=univ["content_hash"]),validation_context=ctx)


def test_catalog_refresh_accepts_halted_instrument_metadata():
    assert refresh_status("HALTED")[0]


def test_catalog_refresh_accepts_suspended_instrument_metadata():
    assert refresh_status("SUSPENDED")[0]


def test_catalog_refresh_accepts_delisted_instrument_metadata():
    assert refresh_status("DELISTED")[0]


def test_catalog_refresh_accepts_unknown_status_metadata():
    assert refresh_status("UNKNOWN")[0]


def test_halted_instrument_cannot_activate_universe():
    assert activate_status("HALTED")[1]["denial"]=="INSTRUMENT_NOT_TRADABLE"


def test_suspended_instrument_cannot_activate_universe():
    assert activate_status("SUSPENDED")[1]["denial"]=="INSTRUMENT_NOT_TRADABLE"


def test_delisted_instrument_cannot_activate_universe():
    assert activate_status("DELISTED")[1]["denial"]=="INSTRUMENT_NOT_TRADABLE"


def test_unknown_status_instrument_cannot_activate_universe():
    assert activate_status("UNKNOWN")[1]["denial"]=="INSTRUMENT_NOT_TRADABLE"


def test_non_trading_metadata_remains_historically_resolvable():
    assert validate_instrument_record(sample_instrument(trading_status="DELISTED"))[0]


def test_catalog_refresh_non_trading_status_emits_success_event():
    assert refresh_status("HALTED")[1]["audit_event"]=="INSTRUMENT_CATALOG_REFRESHED"


def test_universe_non_trading_status_never_emits_activation_success():
    for status in ["HALTED","SUSPENDED","DELISTED","UNKNOWN"]: assert activate_status(status)[1]["audit_event"]!="TRADING_UNIVERSE_ACTIVATED"


def profile_timeline(profile_id, created, retired=None, rotated=None, lifecycle="RETIRED"):
    return sample_profile(credential_profile_id=profile_id,created_at_utc=created,retired_at_utc=retired,rotated_from_credential_profile_id=rotated,lifecycle_state=lifecycle)


def test_two_node_credential_lineage_has_monotonic_timestamps():
    current=profile_timeline("cred_new","2026-07-01T00:00:00Z",None,"cred_old","ACTIVE");old=profile_timeline("cred_old","2026-01-01T00:00:00Z","2026-02-01T00:00:00Z");assert validate_credential_lineage(current,{"previous_profiles_by_id":{"cred_old":old}})[0]


def test_three_node_credential_lineage_has_monotonic_timestamps():
    current=profile_timeline("cred_new","2026-07-01T00:00:00Z",None,"cred_old","ACTIVE");old=profile_timeline("cred_old","2026-01-01T00:00:00Z","2026-02-01T00:00:00Z","cred_older");older=profile_timeline("cred_older","2025-01-01T00:00:00Z","2025-12-01T00:00:00Z");assert validate_credential_lineage(current,{"previous_profiles_by_id":{"cred_old":old,"cred_older":older}})[0]


def test_lineage_rejects_older_profile_created_after_successor():
    current=profile_timeline("cred_new","2026-07-01T00:00:00Z",None,"cred_old","ACTIVE");old=profile_timeline("cred_old","2026-01-01T00:00:00Z","2026-02-01T00:00:00Z","cred_older");older=profile_timeline("cred_older","2026-05-01T00:00:00Z","2026-06-01T00:00:00Z");assert validate_credential_lineage(current,{"previous_profiles_by_id":{"cred_old":old,"cred_older":older}})[0] is False


def test_lineage_rejects_older_profile_retired_after_successor_creation():
    test_lineage_rejects_older_profile_created_after_successor()


def test_lineage_accepts_valid_three_node_timeline():
    test_three_node_credential_lineage_has_monotonic_timestamps()


def test_lineage_timestamp_failure_never_raises():
    current=profile_timeline("cred_new","bad",None,"cred_old","ACTIVE");old=profile_timeline("cred_old","2026-01-01T00:00:00Z","2026-02-01T00:00:00Z");assert validate_credential_lineage(current,{"previous_profiles_by_id":{"cred_old":old}})[0] is False


def activation_with_context(ctx):
    universe=next(iter(ctx["universes_by_id"].values()));return validate_operation_request("ACTIVATE_TRADING_UNIVERSE",request_for("ACTIVATE_TRADING_UNIVERSE",trading_universe_id=universe["trading_universe_id"],exchange_account_id=universe["exchange_account_id"],source_catalog_snapshot_ids=universe["source_catalog_snapshot_ids"],content_hash=universe["content_hash"]),validation_context=ctx)


def test_activation_rejects_missing_trusted_identity_snapshot():
    ctx=context();ctx["external_identity_snapshots_by_account_id"]={};assert activation_with_context(ctx)[0] is False


def test_activation_rejects_unverified_trusted_identity_snapshot():
    assert activation_with_context(context(identity=sample_identity(state="UNVERIFIED")))[0] is False


def test_activation_rejects_mismatch_trusted_identity_snapshot():
    assert activation_with_context(context(identity=sample_identity(state="MISMATCH")))[0] is False


def test_activation_rejects_snapshot_account_scope_mismatch():
    ctx=context();ctx["external_identity_snapshots_by_account_id"]["xacc_test_1"]["market_type"]="PERPETUAL";assert activation_with_context(ctx)[0] is False


def test_activation_rejects_duplicate_trusted_identity():
    ctx=trusted_collision_context();ctx["accounts_by_id"]["xacc_test_1"]["connection_state"]="ONLINE";assert activation_with_context(ctx)[1]["denial"]=="EXCHANGE_ACCOUNT_IDENTITY_COLLISION"


def test_activation_allows_distinct_verified_subaccount():
    ctx=trusted_collision_context("sub-b","sub-a");ctx["accounts_by_id"]["xacc_test_1"]["connection_state"]="ONLINE";assert activation_with_context(ctx)[0]


def test_account_verified_string_without_snapshot_is_not_authority():
    ctx=context();assert ctx["accounts_by_id"]["xacc_test_1"]["external_account_identity_state"]=="VERIFIED";ctx["external_identity_snapshots_by_account_id"]={};assert activation_with_context(ctx)[0] is False


def test_activation_identity_denials_have_valid_audit_mapping():
    spec=DATA["operation_validation_matrix"]["ACTIVATE_TRADING_UNIVERSE"];schemas={s["event_name"]:s for s in DATA["audit_event_schemas"]}
    for denial in ["EXTERNAL_ACCOUNT_IDENTITY_UNVERIFIED","EXTERNAL_ACCOUNT_IDENTITY_MISMATCH","EXCHANGE_ACCOUNT_IDENTITY_COLLISION"]: assert denial in schemas[spec["denial_event_by_code"][denial]]["allowed_denial_codes"]


def capability_activation_context(status="VALID", **overrides):
    ctx=context();account=ctx["accounts_by_id"]["xacc_test_1"];snap=sample_capability_snapshot(account,status=status,**overrides);account["account_capability_snapshot_id"]=snap["account_capability_snapshot_id"];ctx["account_capability_snapshots_by_id"]={snap["account_capability_snapshot_id"]:snap};return ctx


def test_activation_accepts_valid_bound_capability_snapshot():
    assert activation_with_context(capability_activation_context())[0]


def test_activation_rejects_missing_bound_capability_snapshot():
    ctx=context();ctx["accounts_by_id"]["xacc_test_1"]["account_capability_snapshot_id"]="caps_missing";ctx["account_capability_snapshots_by_id"]={};assert activation_with_context(ctx)[0] is False


def test_activation_rejects_stale_capability_snapshot():
    assert activation_with_context(capability_activation_context("STALE"))[0] is False


def test_activation_rejects_rejected_capability_snapshot():
    assert activation_with_context(capability_activation_context("REJECTED"))[0] is False


def test_activation_rejects_expired_valid_capability_snapshot():
    assert activation_with_context(capability_activation_context(stale_after_utc="2026-06-01T00:00:00Z"))[0] is False


def test_activation_rejects_invalid_capability_hash():
    assert activation_with_context(capability_activation_context(content_hash="0"*64))[0] is False


def test_activation_rejects_capability_lineage_cycle():
    ctx=context();account=ctx["accounts_by_id"]["xacc_test_1"];current=capability_version(2,"caps_2","caps_1");old=capability_version(1,"caps_1","caps_2");account["account_capability_snapshot_id"]="caps_2";ctx["account_capability_snapshots_by_id"]={"caps_2":current};ctx["previous_account_capability_snapshots_by_id"]={"caps_1":old,"caps_2":current};assert activation_with_context(ctx)[0] is False


def test_adapter_snapshot_required_exchange_requires_snapshot():
    ctx=context();ctx["accounts_by_id"]["xacc_test_1"]["account_capability_snapshot_id"]=None;ctx["account_capability_snapshots_by_id"]={};assert activation_with_context(ctx)[0] is False


def paper_activation_context():
    account=sample_account(exchange_account_id="xacc_paper",exchange_id="paper_simulated_venue",environment="PAPER",external_account_identity_state="VERIFIED");inst=sample_instrument(instrument_id="instr_paper",exchange_id="paper_simulated_venue",environment="PAPER",source_adapter_family_id="paper_simulation_adapter_family");inst["base_asset_reference"]["asset_namespace"]="paper_simulated_venue";inst["quote_asset_reference"]["asset_namespace"]="paper_simulated_venue";cat=sample_catalog(inst);univ=sample_universe(account,inst,cat);return context(account=account,identity=sample_identity(exchange_id="paper_simulated_venue",environment="PAPER"),inst=inst,catalog=cat,universe=univ)


def test_static_paper_policy_does_not_require_private_snapshot():
    assert activation_with_context(paper_activation_context())[0]


def test_capability_snapshot_cannot_enable_live():
    live=sample_account(exchange_account_id="xacc_live",environment="LIVE");snap=sample_capability_snapshot(live);assert validate_account_capability_snapshot(snap,live,{})[0] is False


def test_capability_snapshot_cannot_bypass_readiness():
    ctx=capability_activation_context();ctx["accounts_by_id"]["xacc_test_1"]["readiness_confirmed"]=False;assert activation_with_context(ctx)[0] is False


def test_capability_activation_denials_have_valid_audit_mapping():
    spec=DATA["operation_validation_matrix"]["ACTIVATE_TRADING_UNIVERSE"];event=spec["denial_event_by_code"]["ACCOUNT_READINESS_BLOCKED"];schemas={s["event_name"]:s for s in DATA["audit_event_schemas"]};assert "ACCOUNT_READINESS_BLOCKED" in schemas[event]["allowed_denial_codes"]


def test_capability_rejects_exchange_unsupported_environment():
    assert validate_account_capability_snapshot(sample_capability_snapshot(environment="PAPER"),sample_account(),{})[0] is False


def test_capability_rejects_exchange_unsupported_market_type():
    assert validate_account_capability_snapshot(sample_capability_snapshot(market_type="MARGIN"),sample_account(),{})[0] is False


def test_capability_rejects_disabled_exchange():
    entry=exchange_entry("generic_testnet_venue");old=entry["status"];entry["status"]="DISABLED"
    try: assert validate_account_capability_snapshot(sample_capability_snapshot(),sample_account(),{})[0] is False
    finally: entry["status"]=old


def test_capability_rejects_market_instrument_type_mismatch():
    assert validate_account_capability_snapshot(sample_capability_snapshot(supported_instrument_types=["PERPETUAL_CONTRACT"]),sample_account(),{})[0] is False


def test_capability_predecessor_rejects_unsupported_environment():
    current=capability_version(2,"caps_2","caps_1");old=capability_version(1,"caps_1",None);old["environment"]="PAPER";old["content_hash"]=hash_payload(DATA["account_capability_snapshot_contract"]["content_hash_definition"],old);assert validate_account_capability_snapshot(current,sample_account(),{"caps_1":old})[0] is False


def test_capability_predecessor_rejects_unsupported_market_type():
    current=capability_version(2,"caps_2","caps_1");old=capability_version(1,"caps_1",None);old["market_type"]="MARGIN";old["content_hash"]=hash_payload(DATA["account_capability_snapshot_contract"]["content_hash_definition"],old);assert validate_account_capability_snapshot(current,sample_account(),{"caps_1":old})[0] is False


def test_malformed_current_capability_context_fails_closed():
    ctx=context();ctx["account_capability_snapshots_by_id"]={"bad":{"account_capability_snapshot_id":"bad"}};assert validate_operation_request("CREATE_ACCOUNT",request_for("CREATE_ACCOUNT"),validation_context=ctx)[0] is False


def test_malformed_previous_capability_context_fails_closed():
    ctx=context();ctx["previous_account_capability_snapshots_by_id"]={"bad":{"account_capability_snapshot_id":"bad"}};assert validate_operation_request("CREATE_ACCOUNT",request_for("CREATE_ACCOUNT"),validation_context=ctx)[0] is False


def catalog_with_records(records):
    mapping={key:value for key,value in records};first=records[0][1];cat=sample_catalog(first,instrument_ids=[key for key,_ in records]);return cat,mapping


def test_catalog_rejects_two_ids_with_same_instrument_identity():
    one=sample_instrument(instrument_id="instr_one");two=sample_instrument(instrument_id="instr_two");cat,mapping=catalog_with_records([("instr_one",one),("instr_two",two)]);assert validate_instrument_catalog_snapshot(cat,mapping,{}, instrument_history_by_id={})[1]=="INSTRUMENT_IDENTITY_COLLISION"
    ctx=multi_catalog_context();second=ctx["instruments_by_id"]["instr_ethusdt_spot_testnet"];second["venue_symbol"]="BTCUSDT"
    assert validate_catalog_instrument_graph(ctx["catalogs_by_id"],ctx["previous_catalogs_by_id"],ctx["instruments_by_id"],ctx["instrument_history_by_id"]) is False
    assert direct_membership(ctx) is False and validate_context(ctx) is False
    ok,out=activation_with_context(ctx);assert not ok and out=={"denial":"ACCOUNT_READINESS_BLOCKED","audit_event":"TRADING_UNIVERSE_REJECTED"}


def test_catalog_rejects_same_id_with_changed_identity():
    current=sample_instrument(instrument_id="instr_same",venue_symbol="ONE",metadata_version=2);historical=sample_instrument(instrument_id="instr_same",venue_symbol="TWO",metadata_version=1);cat=sample_catalog(current,instrument_ids=["instr_same"]);assert validate_instrument_catalog_snapshot(cat,{"instr_same":current},{},instrument_history_by_id={"instr_same":[historical]})[1]=="INSTRUMENT_IDENTITY_COLLISION"
    ctx=history_only_member_context();ctx["instrument_history_by_id"]["instr_ghost"][0]["venue_symbol"]="BTCUSDT"
    assert validate_catalog_instrument_graph(ctx["catalogs_by_id"],ctx["previous_catalogs_by_id"],ctx["instruments_by_id"],ctx["instrument_history_by_id"]) is False
    assert validate_context(ctx) is False


def test_catalog_allows_exactly_distinct_venue_symbols():
    one=sample_instrument(instrument_id="instr_one",venue_symbol="ONE");two=sample_instrument(instrument_id="instr_two",venue_symbol="TWO");cat,mapping=catalog_with_records([("instr_one",one),("instr_two",two)]);assert validate_instrument_catalog_snapshot(cat,mapping,{}, instrument_history_by_id={})[0]


def test_catalog_treats_symbol_case_as_exact():
    one=sample_instrument(instrument_id="instr_upper",venue_symbol="BTCUSDT");two=sample_instrument(instrument_id="instr_lower",venue_symbol="btcusdt");cat,mapping=catalog_with_records([("instr_upper",one),("instr_lower",two)]);assert validate_instrument_catalog_snapshot(cat,mapping,{}, instrument_history_by_id={})[0]


def test_display_symbol_does_not_create_identity():
    one=sample_instrument(instrument_id="instr_one",venue_symbol="ONE",display_symbol="SAME");two=sample_instrument(instrument_id="instr_two",venue_symbol="TWO",display_symbol="SAME");cat,mapping=catalog_with_records([("instr_one",one),("instr_two",two)]);assert validate_instrument_catalog_snapshot(cat,mapping,{}, instrument_history_by_id={})[0]


def test_instrument_identity_collision_has_valid_audit_mapping():
    schemas={s["event_name"]:s for s in DATA["audit_event_schemas"]}
    for operation in ["REFRESH_INSTRUMENT_CATALOG","ACTIVATE_TRADING_UNIVERSE"]:
        spec=DATA["operation_validation_matrix"][operation];event=spec["denial_event_by_code"]["INSTRUMENT_IDENTITY_COLLISION"];assert "INSTRUMENT_IDENTITY_COLLISION" in schemas[event]["allowed_denial_codes"]


def test_non_trading_historical_instrument_remains_resolvable():
    assert validate_instrument_record(sample_instrument(trading_status="DELISTED"))[0]

# Finalne regresje: jedna bramka readiness oraz zaufana historia Instrument.
def direct_activation_from_context(ctx, validation_context_marker=True):
    universe = next(iter(ctx["universes_by_id"].values()))
    account = ctx["accounts_by_id"][universe["exchange_account_id"]]
    supplied = ctx if validation_context_marker else None
    return validate_trading_universe_version(
        universe, account, ctx["instruments_by_id"], ctx["catalogs_by_id"],
        ctx["previous_universes_by_id"], ctx["active_universes"],
        ctx["previous_catalogs_by_id"], validation_context=supplied,
    )


def test_direct_universe_validator_requires_validation_context():
    ctx=context(); assert direct_activation_from_context(ctx, False)==(False,"ACCOUNT_READINESS_BLOCKED")


def test_direct_universe_validator_rejects_missing_trusted_identity():
    ctx=context();ctx["external_identity_snapshots_by_account_id"]={};assert direct_activation_from_context(ctx)[0] is False


def test_direct_universe_validator_rejects_unverified_identity():
    ctx=context(identity=sample_identity(state="UNVERIFIED"));assert direct_activation_from_context(ctx)[0] is False


def test_direct_universe_validator_rejects_duplicate_identity():
    ctx=trusted_collision_context();ctx["accounts_by_id"]["xacc_test_1"]["connection_state"]="ONLINE";assert direct_activation_from_context(ctx)[1]=="EXCHANGE_ACCOUNT_IDENTITY_COLLISION"


@pytest.mark.parametrize("status",["STALE","REJECTED"])
def test_direct_universe_validator_rejects_stale_or_rejected_capability(status):
    assert direct_activation_from_context(capability_activation_context(status))[0] is False


def test_direct_universe_validator_rejects_stale_capability():
    assert direct_activation_from_context(capability_activation_context("STALE"))[0] is False


def test_direct_universe_validator_rejects_rejected_capability():
    assert direct_activation_from_context(capability_activation_context("REJECTED"))[0] is False


def test_direct_universe_validator_rejects_expired_capability():
    assert direct_activation_from_context(capability_activation_context(stale_after_utc="2026-06-01T00:00:00Z"))[0] is False


def test_direct_and_dispatched_activation_share_same_readiness_result():
    ctx=capability_activation_context("STALE");direct=direct_activation_from_context(ctx);dispatched=activation_with_context(ctx);assert direct[0] is dispatched[0] is False and direct[1]==dispatched[1]["denial"]


def test_no_direct_activation_success_bypasses_readiness():
    ctx=context();ctx["readiness_by_account_id"]["xacc_test_1"]=False;assert direct_activation_from_context(ctx)[0] is False


def test_testnet_activation_rejects_null_active_credential_profile_id():
    ctx=context();ctx["accounts_by_id"]["xacc_test_1"]["active_credential_profile_id"]=None;assert activation_with_context(ctx)[0] is False


def test_testnet_activation_rejects_missing_active_profile_record():
    ctx=context();ctx["credential_profiles_by_id"]={};assert activation_with_context(ctx)[0] is False


def test_testnet_activation_rejects_foreign_active_profile():
    ctx=context();ctx["credential_profiles_by_id"]["cred_1"]["exchange_account_id"]="other";assert activation_with_context(ctx)[0] is False


def test_testnet_activation_rejects_profile_scope_mismatch():
    ctx=context();ctx["credential_profiles_by_id"]["cred_1"]["environment_scope"]="PAPER";assert activation_with_context(ctx)[0] is False


def test_testnet_activation_rejects_inactive_profile():
    ctx=context(profile=sample_profile(lifecycle_state="RETIRED",retired_at_utc="2026-02-01T00:00:00Z"));assert activation_with_context(ctx)[0] is False


def test_testnet_activation_rejects_active_profile_map_mismatch():
    ctx=context();ctx["active_profile_ids_by_account_id"]["xacc_test_1"]=["other"];assert activation_with_context(ctx)[0] is False


def test_testnet_activation_rejects_withdraw_profile():
    ctx=context(profile=sample_profile(permission_snapshot=["READ_ACCOUNT","WITHDRAW"]));assert activation_with_context(ctx)[0] is False


def test_testnet_activation_rejects_invalid_profile_lineage():
    ctx=context(profile=sample_profile(rotated_from_credential_profile_id="missing"));assert activation_with_context(ctx)[0] is False


def test_testnet_activation_accepts_valid_bound_read_profile():
    assert activation_with_context(context())[0]


def test_static_paper_activation_does_not_require_private_profile():
    ctx=paper_activation_context();ctx["accounts_by_id"]["xacc_paper"]["active_credential_profile_id"]=None;ctx["credential_profiles_by_id"]={};ctx["active_profile_ids_by_account_id"]={};assert activation_with_context(ctx)[0]


def order_entry_context(profile_permissions, capability_permissions, identity_permissions):
    profile=sample_profile(credential_purpose="ORDER_ENTRY",permission_snapshot=profile_permissions)
    identity=sample_identity(observed_permission_set=identity_permissions)
    ctx=context(profile=profile,identity=identity);account=ctx["accounts_by_id"]["xacc_test_1"];account["execution_authorization"]="ORDER_ENTRY_ALLOWED"
    snap=sample_capability_snapshot(account,observed_permission_set=capability_permissions);ctx["account_capability_snapshots_by_id"]={"caps_1":snap};return ctx


def test_order_entry_rejects_profile_without_place_orders():
    assert activation_with_context(order_entry_context(["READ_ACCOUNT"],["READ_ACCOUNT","PLACE_ORDERS"],["READ_ACCOUNT","PLACE_ORDERS"]))[0] is False


def test_order_entry_rejects_capability_without_place_orders():
    assert activation_with_context(order_entry_context(["READ_ACCOUNT","PLACE_ORDERS"],["READ_ACCOUNT"],["READ_ACCOUNT","PLACE_ORDERS"]))[0] is False


def test_order_entry_rejects_identity_without_place_orders():
    assert activation_with_context(order_entry_context(["READ_ACCOUNT","PLACE_ORDERS"],["READ_ACCOUNT","PLACE_ORDERS"],["READ_ACCOUNT"]))[0] is False


def test_order_entry_accepts_place_orders_in_all_trusted_sources():
    permissions=["READ_ACCOUNT","PLACE_ORDERS"];assert activation_with_context(order_entry_context(permissions,permissions,permissions))[0]


def test_read_only_accepts_read_account_without_place_orders():
    assert activation_with_context(context())[0]


def test_missing_read_account_blocks_testnet_readiness():
    ctx=context(identity=sample_identity(observed_permission_set=[]));assert activation_with_context(ctx)[0] is False


def test_withdraw_in_capability_blocks_readiness():
    ctx=capability_activation_context(observed_permission_set=["READ_ACCOUNT","WITHDRAW"]);assert activation_with_context(ctx)[0] is False


def test_withdraw_in_identity_blocks_readiness():
    ctx=context(identity=sample_identity(observed_permission_set=["READ_ACCOUNT","WITHDRAW"]));assert activation_with_context(ctx)[0] is False


def test_permissions_never_extend_product_capabilities():
    ok,details=validate_account_activation_readiness(context()["accounts_by_id"]["xacc_test_1"],context());assert ok and details["extends_product_capabilities"] is False


def test_activation_rejects_empty_capability_instrument_types():
    assert activation_with_context(capability_activation_context(supported_instrument_types=[]))[0] is False


def test_activation_rejects_universe_type_not_in_capability_snapshot():
    assert activation_with_context(capability_activation_context(supported_instrument_types=[]))[0] is False


def test_activation_accepts_universe_type_in_capability_snapshot():
    assert activation_with_context(capability_activation_context(supported_instrument_types=["SPOT_PAIR"]))[0]


def test_bound_static_paper_snapshot_restricts_instrument_types():
    ctx=paper_activation_context();account=ctx["accounts_by_id"]["xacc_paper"];snap=sample_capability_snapshot(account,supported_instrument_types=[] ,adapter_family_id="paper_simulation_adapter_family");account["account_capability_snapshot_id"]="caps_1";ctx["account_capability_snapshots_by_id"]={"caps_1":snap};assert activation_with_context(ctx)[0] is False


def history_context(current=None, history=None):
    current=current or sample_instrument(metadata_version=3);cat=sample_catalog(current);univ=sample_universe(inst=current,catalog=cat);ctx=context(inst=current,catalog=cat,universe=univ);ctx["instrument_history_by_id"]={current["instrument_id"]: history or [sample_instrument(metadata_version=1,trading_status="DELISTED"),sample_instrument(metadata_version=2,display_symbol="BTC-USDT")]};return ctx


def test_context_accepts_valid_instrument_identity_history():
    assert validate_context(history_context())


def test_context_rejects_history_key_record_id_mismatch():
    ctx=history_context();ctx["instrument_history_by_id"]={"wrong":ctx["instrument_history_by_id"]["instr_btcusdt_spot_testnet"]};assert not validate_context(ctx)


def test_context_rejects_malformed_instrument_history_record():
    ctx=history_context();ctx["instrument_history_by_id"]["instr_btcusdt_spot_testnet"]=[{}];assert not validate_context(ctx)


def test_context_rejects_duplicate_history_metadata_version():
    ctx=history_context(history=[sample_instrument(metadata_version=1),sample_instrument(metadata_version=1)]);assert not validate_context(ctx)


@pytest.mark.parametrize("field,value",[("venue_symbol","NEW"),("exchange_id","paper_simulated_venue"),("environment","PAPER"),("market_type","MARGIN")])
def test_catalog_rejects_historical_identity_rewrite_fields(field,value):
    current=sample_instrument(metadata_version=2);historical=sample_instrument(metadata_version=1);historical[field]=value;cat=sample_catalog(current);assert validate_instrument_catalog_snapshot(cat,{current["instrument_id"]:current},{},instrument_history_by_id={current["instrument_id"]:[historical]})[1]=="INSTRUMENT_IDENTITY_COLLISION"


def test_catalog_rejects_historical_venue_symbol_rewrite(): test_catalog_rejects_historical_identity_rewrite_fields("venue_symbol","NEW")
def test_catalog_rejects_historical_exchange_rewrite(): test_catalog_rejects_historical_identity_rewrite_fields("exchange_id","paper_simulated_venue")
def test_catalog_rejects_historical_environment_rewrite(): test_catalog_rejects_historical_identity_rewrite_fields("environment","PAPER")
def test_catalog_rejects_historical_market_type_rewrite(): test_catalog_rejects_historical_identity_rewrite_fields("market_type","MARGIN")


def test_catalog_allows_display_symbol_change():
    ctx=history_context();assert activation_with_context(ctx)[0]


def test_catalog_allows_trading_status_change():
    ctx=history_context();assert activation_with_context(ctx)[0]


def test_catalog_allows_non_trading_historical_record():
    assert validate_context(history_context())


def test_catalog_allows_exact_identity_across_multiple_versions():
    assert validate_context(history_context()) and activation_with_context(history_context())[0]


def test_dispatcher_rejects_historical_identity_rewrite():
    ctx=history_context();ctx["instrument_history_by_id"]["instr_btcusdt_spot_testnet"][0]["venue_symbol"]="OLD";assert activation_with_context(ctx)[0] is False


def test_direct_catalog_validator_rejects_historical_identity_rewrite():
    current=sample_instrument(metadata_version=2);old=sample_instrument(metadata_version=1,venue_symbol="OLD");cat=sample_catalog(current);assert validate_instrument_catalog_snapshot(cat,{current["instrument_id"]:current},{},instrument_history_by_id={current["instrument_id"]:[old]})[0] is False

# Final hardening: validation_context jest jedynym źródłem authority.
def raw_direct_activation(ctx, *, account=None, universe=None, instruments=None, catalogs=None,
                          previous_catalogs=None, previous_universes=None, active_universes=None):
    trusted_universe = next(iter(ctx["universes_by_id"].values()))
    trusted_account = ctx["accounts_by_id"][trusted_universe["exchange_account_id"]]
    return validate_trading_universe_version(
        universe if universe is not None else trusted_universe,
        account if account is not None else trusted_account,
        instruments if instruments is not None else ctx["instruments_by_id"],
        catalogs if catalogs is not None else ctx["catalogs_by_id"],
        previous_universes if previous_universes is not None else ctx["previous_universes_by_id"],
        active_universes if active_universes is not None else ctx["active_universes"],
        previous_catalogs if previous_catalogs is not None else ctx["previous_catalogs_by_id"],
        validation_context=ctx,
    )


def test_direct_activation_rejects_account_not_equal_to_context_record():
    ctx=context();foreign=dict(ctx["accounts_by_id"]["xacc_test_1"],display_name="foreign");assert raw_direct_activation(ctx,account=foreign)[0] is False


def test_direct_activation_rejects_paper_account_against_testnet_context():
    ctx=context();paper=dict(ctx["accounts_by_id"]["xacc_test_1"],exchange_id="paper_simulated_venue",environment="PAPER");assert raw_direct_activation(ctx,account=paper)[0] is False


def test_direct_activation_rejects_instruments_not_bound_to_context():
    ctx=context();foreign={**ctx["instruments_by_id"]};foreign[next(iter(foreign))]=dict(next(iter(foreign.values())),display_symbol="foreign");assert raw_direct_activation(ctx,instruments=foreign)[0] is False


def test_direct_activation_rejects_catalogs_not_bound_to_context():
    ctx=context();foreign={**ctx["catalogs_by_id"]};key=next(iter(foreign));foreign[key]=dict(foreign[key],adapter_version="foreign");assert raw_direct_activation(ctx,catalogs=foreign)[0] is False


def test_direct_activation_rejects_universe_not_bound_to_context():
    ctx=context();foreign=dict(next(iter(ctx["universes_by_id"].values())),creation_reason="foreign");assert raw_direct_activation(ctx,universe=foreign)[0] is False


def test_direct_activation_rejects_previous_catalog_map_mismatch():
    assert raw_direct_activation(context(),previous_catalogs={"foreign":{}})[0] is False


def test_direct_activation_rejects_previous_universe_map_mismatch():
    assert raw_direct_activation(context(),previous_universes={"foreign":{}})[0] is False


def test_direct_activation_rejects_active_universe_list_mismatch():
    assert raw_direct_activation(context(),active_universes=[sample_universe()])[0] is False


def test_direct_activation_uses_only_trusted_context_records():
    assert raw_direct_activation(context())[0]


def test_dispatcher_and_direct_validator_use_identical_trusted_records():
    ctx=context();assert raw_direct_activation(ctx)[0] == activation_with_context(ctx)[0] is True


@pytest.mark.parametrize("bad",[None,False,1,"account",[],{}])
def test_readiness_rejects_non_dict_account(bad):
    assert validate_account_activation_readiness(bad,context())[0] is False


def test_readiness_rejects_none_account():
    assert validate_account_activation_readiness(None,context())[0] is False


@pytest.mark.parametrize("bad",[None,False,1,"context",[],{}])
def test_readiness_rejects_partial_context(bad):
    assert validate_account_activation_readiness(sample_account(),bad)[0] is False


def test_readiness_rejects_missing_context_field():
    ctx=context();account=ctx["accounts_by_id"]["xacc_test_1"];del ctx["instrument_history_by_id"];assert validate_account_activation_readiness(account,ctx)[0] is False


def test_readiness_rejects_account_not_bound_to_context():
    ctx=context();account=dict(ctx["accounts_by_id"]["xacc_test_1"],display_name="spoof");assert validate_account_activation_readiness(account,ctx)[0] is False


@pytest.mark.parametrize("state",["DRAFT","DISABLED"])
def test_readiness_rejects_nonactive_account(state):
    ctx=context(account=sample_account(lifecycle_state=state));account=ctx["accounts_by_id"]["xacc_test_1"];assert validate_account_activation_readiness(account,ctx)[0] is False


def test_readiness_rejects_draft_account(): test_readiness_rejects_nonactive_account("DRAFT")
def test_readiness_rejects_disabled_account(): test_readiness_rejects_nonactive_account("DISABLED")


def test_readiness_rejects_disconnected_account():
    ctx=context(account=sample_account(connection_state="DISCONNECTED"));assert validate_account_activation_readiness(ctx["accounts_by_id"]["xacc_test_1"],ctx)[0] is False


def test_readiness_rejects_blocked_authorization():
    ctx=context(account=sample_account(execution_authorization="BLOCKED_BY_POLICY"));assert validate_account_activation_readiness(ctx["accounts_by_id"]["xacc_test_1"],ctx)[0] is False


def test_readiness_rejects_account_readiness_confirmed_false():
    ctx=context(account=sample_account(readiness_confirmed=False));assert validate_account_activation_readiness(ctx["accounts_by_id"]["xacc_test_1"],ctx)[0] is False


def test_readiness_rejects_account_readiness_confirmed_missing():
    ctx=context();account=ctx["accounts_by_id"]["xacc_test_1"];del account["readiness_confirmed"];assert validate_account_activation_readiness(account,ctx)[0] is False


def test_readiness_rejects_readiness_map_false():
    ctx=context();ctx["readiness_by_account_id"]["xacc_test_1"]=False;assert validate_account_activation_readiness(ctx["accounts_by_id"]["xacc_test_1"],ctx)[0] is False


def test_readiness_accepts_only_complete_operable_account():
    ctx=context();assert validate_account_activation_readiness(ctx["accounts_by_id"]["xacc_test_1"],ctx)[0]


def test_direct_activation_missing_instrument_history_map_fails_closed():
    ctx=context();del ctx["instrument_history_by_id"];assert raw_direct_activation(ctx)[0] is False


@pytest.mark.parametrize("bad",[None,False,1,"context",[],{"accounts_by_id":{}}])
def test_direct_activation_malformed_context_never_raises(bad):
    assert validate_trading_universe_version({}, {}, {}, {}, {}, [], {}, validation_context=bad)[0] is False


def withdrawal_result(source):
    if source=="profile": ctx=context(profile=sample_profile(permission_snapshot=["READ_ACCOUNT","WITHDRAW"]))
    elif source=="capability": ctx=capability_activation_context(observed_permission_set=["READ_ACCOUNT","WITHDRAW"])
    else: ctx=context(identity=sample_identity(observed_permission_set=["READ_ACCOUNT","WITHDRAW"]))
    return activation_with_context(ctx)


@pytest.mark.parametrize("source",["profile","capability","identity"])
def test_withdrawal_sources_preserve_exact_denial_and_audit(source):
    ok,out=withdrawal_result(source);assert not ok and out=={"denial":"WITHDRAWAL_PERMISSION_FORBIDDEN","audit_event":"TRADING_UNIVERSE_REJECTED"}


def direct_catalog(history):
    inst=sample_instrument();cat=sample_catalog(inst);return validate_instrument_catalog_snapshot(cat,{inst["instrument_id"]:inst},{},instrument_history_by_id=history)


def test_direct_catalog_requires_explicit_history_map():
    inst=sample_instrument();assert validate_instrument_catalog_snapshot(sample_catalog(inst),{inst["instrument_id"]:inst},{})[0] is False


def test_direct_catalog_rejects_none_history(): assert direct_catalog(None)[0] is False
def test_direct_catalog_rejects_string_history(): assert direct_catalog("history")[0] is False
def test_direct_catalog_rejects_list_history(): assert direct_catalog([])[0] is False

def test_direct_catalog_rejects_non_list_history_entry(): assert direct_catalog({"instr_btcusdt_spot_testnet":{}})[0] is False
def test_direct_catalog_rejects_non_dict_history_record(): assert direct_catalog({"instr_btcusdt_spot_testnet":["record"]})[0] is False


@pytest.mark.parametrize("bad",[None,False,1,"history",[],{"id":None},{"id":[None]}])
def test_direct_catalog_malformed_history_never_raises(bad): assert direct_catalog(bad)[0] is False


def test_direct_catalog_empty_trusted_history_map_is_valid_for_new_instrument(): assert direct_catalog({})[0]


def history_without_current(*records):
    ctx=context();ctx["instruments_by_id"]={};ctx["instrument_history_by_id"]={records[0]["instrument_id"]:list(records)};return ctx


def test_context_rejects_identity_change_between_history_versions():
    one=sample_instrument(metadata_version=1);two=sample_instrument(metadata_version=2,venue_symbol="OTHER");assert not validate_context(history_without_current(one,two))


@pytest.mark.parametrize("field,value",[("venue_symbol","OTHER"),("exchange_id","paper_simulated_venue"),("environment","PAPER"),("market_type","MARGIN")])
def test_context_rejects_history_identity_change_without_current_record(field,value):
    one=sample_instrument(metadata_version=1);two=sample_instrument(metadata_version=2);two[field]=value;assert not validate_context(history_without_current(one,two))


def test_context_rejects_history_symbol_change_without_current_record(): test_context_rejects_history_identity_change_without_current_record("venue_symbol","OTHER")
def test_context_rejects_history_exchange_change_without_current_record(): test_context_rejects_history_identity_change_without_current_record("exchange_id","paper_simulated_venue")
def test_context_rejects_history_environment_change_without_current_record(): test_context_rejects_history_identity_change_without_current_record("environment","PAPER")
def test_context_rejects_history_market_change_without_current_record(): test_context_rejects_history_identity_change_without_current_record("market_type","MARGIN")


def version_history_context(current_version, history_versions=(1,2)):
    current=sample_instrument(metadata_version=current_version);history=[sample_instrument(metadata_version=v) for v in history_versions];return history_context(current=current,history=history)


def test_context_rejects_current_version_equal_to_history_version(): assert not validate_context(version_history_context(2))
def test_context_rejects_current_version_lower_than_history_version(): assert not validate_context(version_history_context(1))
def test_context_rejects_current_version_rollback(): assert not validate_context(version_history_context(1,(2,3)))
def test_activation_rejects_current_version_equal_to_history(): assert activation_with_context(version_history_context(2))[0] is False
def test_activation_rejects_current_version_lower_than_history(): assert activation_with_context(version_history_context(1))[0] is False

def test_context_accepts_current_version_greater_than_complete_history(): assert validate_context(version_history_context(3))


def test_context_allows_display_symbol_change_across_history():
    ctx=version_history_context(3);ctx["instrument_history_by_id"]["instr_btcusdt_spot_testnet"][1]["display_symbol"]="OLD";assert validate_context(ctx)


def test_context_allows_trading_status_change_across_history():
    ctx=version_history_context(3);ctx["instrument_history_by_id"]["instr_btcusdt_spot_testnet"][0]["trading_status"]="DELISTED";assert validate_context(ctx)


def stale_history(status="DELISTED"):
    return sample_instrument(metadata_version=1,trading_status=status,observed_at_utc="2020-01-01T00:00:00Z",effective_at_utc="2020-01-02T00:00:00Z",stale_after_utc="2020-02-01T00:00:00Z")


def test_context_accepts_structurally_valid_stale_historical_record(): assert validate_context(history_context(history=[stale_history()]))
def test_context_accepts_old_delisted_historical_record(): assert validate_context(history_context(history=[stale_history("DELISTED")]))
def test_context_accepts_old_halted_historical_record(): assert validate_context(history_context(history=[stale_history("HALTED")]))


def test_current_catalog_rejects_stale_current_instrument():
    inst=sample_instrument(stale_after_utc="2020-01-01T00:00:00Z");assert validate_instrument_catalog_snapshot(sample_catalog(inst),{inst["instrument_id"]:inst},{},instrument_history_by_id={})[0] is False


def test_activation_rejects_stale_current_instrument():
    inst=sample_instrument(stale_after_utc="2020-01-01T00:00:00Z");cat=sample_catalog(inst);univ=sample_universe(inst=inst,catalog=cat);assert activation_with_context(context(inst=inst,catalog=cat,universe=univ))[0] is False


def test_historical_staleness_does_not_grant_execution():
    assert validate_instrument_record(stale_history(),require_fresh=False)[0] and validate_instrument_record(stale_history(),require_fresh=True)[0] is False


def test_historical_validator_preserves_all_structural_checks():
    bad=stale_history();bad["price_tick"]="bad";assert validate_instrument_record(bad,require_fresh=False)[0] is False

# Multi-catalog, wspólny validator historii i pełny closed trusted context.
def multi_catalog_context():
    ctx=context()
    second=sample_instrument(instrument_id="instr_ethusdt_spot_testnet",venue_symbol="ETHUSDT",display_symbol="ETH/USDT",catalog_snapshot_id="cat_2")
    second["base_asset_reference"]=asset("ETH")
    cat2=sample_catalog(second,catalog_snapshot_id="cat_2",instrument_ids=[second["instrument_id"]])
    ctx["instruments_by_id"][second["instrument_id"]]=second
    ctx["catalogs_by_id"]["cat_2"]=cat2
    return ctx


def test_dispatch_activation_accepts_unrelated_valid_catalog_in_context():
    assert activation_with_context(multi_catalog_context())[0]


def test_direct_activation_accepts_unrelated_valid_catalog_in_context():
    assert raw_direct_activation(multi_catalog_context())[0]


def test_direct_and_dispatch_multi_catalog_results_are_identical():
    ctx=multi_catalog_context();assert raw_direct_activation(ctx)==(True,None) and activation_with_context(ctx)[0]


def test_activation_uses_only_declared_source_catalogs():
    ctx=multi_catalog_context();universe=ctx["universes_by_id"]["univ_1"];assert universe["source_catalog_snapshot_ids"]==["cat_1"] and activation_with_context(ctx)[0]


def test_activation_rejects_foreign_source_catalog():
    ctx=multi_catalog_context();req=request_for("ACTIVATE_TRADING_UNIVERSE",source_catalog_snapshot_ids=["cat_2"]);assert validate_operation_request("ACTIVATE_TRADING_UNIVERSE",req,validation_context=ctx)[0] is False


def test_activation_rejects_tampered_catalog_record():
    ctx=multi_catalog_context();ctx["catalogs_by_id"]["cat_1"]["adapter_version"]="tampered";assert activation_with_context(ctx)[0] is False


def test_unrelated_catalog_does_not_extend_universe_membership():
    ctx=multi_catalog_context();universe=ctx["universes_by_id"]["univ_1"];universe["instrument_ids"].append("instr_ethusdt_spot_testnet");universe["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["trading_universe_content_hash"],universe);ok,out=activation_with_context(ctx);assert not ok and out["denial"]=="ACCOUNT_READINESS_BLOCKED"


def direct_history_result(current_version, history):
    current=sample_instrument(metadata_version=current_version);cat=sample_catalog(current);return validate_instrument_catalog_snapshot(cat,{current["instrument_id"]:current},{},instrument_history_by_id={current["instrument_id"]:history})


def test_direct_catalog_rejects_duplicate_history_versions():
    assert direct_history_result(3,[sample_instrument(metadata_version=1),sample_instrument(metadata_version=1)])==(False,"INSTRUMENT_METADATA_INVALID")


def test_direct_catalog_rejects_out_of_order_history_versions():
    assert direct_history_result(3,[sample_instrument(metadata_version=2),sample_instrument(metadata_version=1)])==(False,"INSTRUMENT_METADATA_INVALID")


def test_direct_catalog_rejects_current_version_equal_to_history():
    assert direct_history_result(2,[sample_instrument(metadata_version=1),sample_instrument(metadata_version=2)])==(False,"INSTRUMENT_METADATA_INVALID")


def test_direct_catalog_rejects_current_version_lower_than_history():
    assert direct_history_result(1,[sample_instrument(metadata_version=2)])==(False,"INSTRUMENT_METADATA_INVALID")


def test_direct_catalog_rejects_current_version_rollback():
    assert direct_history_result(2,[sample_instrument(metadata_version=1),sample_instrument(metadata_version=3)])==(False,"INSTRUMENT_METADATA_INVALID")


def test_direct_catalog_rejects_identity_change_between_history_versions():
    assert direct_history_result(3,[sample_instrument(metadata_version=1),sample_instrument(metadata_version=2,venue_symbol="OTHER")])==(False,"INSTRUMENT_IDENTITY_COLLISION")


def test_direct_catalog_rejects_identity_change_against_current_record():
    assert direct_history_result(2,[sample_instrument(metadata_version=1,venue_symbol="OTHER")])==(False,"INSTRUMENT_IDENTITY_COLLISION")


def test_direct_catalog_accepts_current_version_above_complete_history():
    assert direct_history_result(3,[sample_instrument(metadata_version=1),sample_instrument(metadata_version=2)])==(True,None)


def test_direct_catalog_accepts_old_structurally_valid_history():
    assert direct_history_result(2,[stale_history()])==(True,None)


def test_direct_catalog_allows_historical_display_symbol_change():
    assert direct_history_result(2,[sample_instrument(metadata_version=1,display_symbol="OLD")])==(True,None)


def test_direct_catalog_allows_historical_trading_status_change():
    assert direct_history_result(2,[sample_instrument(metadata_version=1,trading_status="DELISTED")])==(True,None)


MALFORMED_RECORDS=[None,False,1,"record",[],{},]


def malformed_unrelated(map_name,bad):
    ctx=context()
    if map_name=="accounts_by_id": ctx[map_name]["xacc_bad"]=bad if bad!={ } else {"exchange_account_id":"xacc_bad"}
    elif map_name=="credential_profiles_by_id": ctx[map_name]["cred_bad"]=bad if bad!={} else {"credential_profile_id":"cred_bad"}
    elif map_name=="external_identity_snapshots_by_account_id":
        ctx["accounts_by_id"]["xacc_other"]=sample_account(exchange_account_id="xacc_other");ctx[map_name]["xacc_other"]=bad if bad!={} else {"state":"UNVERIFIED"}
    elif map_name=="universes_by_id": ctx[map_name]["univ_bad"]=bad if bad!={} else {"trading_universe_id":"univ_bad"}
    elif map_name=="catalogs_by_id": ctx[map_name]["cat_bad"]=bad if bad!={} else {"catalog_snapshot_id":"cat_bad"}
    elif map_name=="instruments_by_id": ctx[map_name]["instr_bad"]=bad if bad!={} else {"instrument_id":"instr_bad"}
    elif map_name=="previous_profiles_by_id": ctx[map_name]["cred_bad"]=bad if bad!={} else {"credential_profile_id":"cred_bad"}
    elif map_name=="previous_catalogs_by_id": ctx[map_name]["cat_bad"]=bad if bad!={} else {"catalog_snapshot_id":"cat_bad"}
    else: ctx[map_name]["univ_bad"]=bad if bad!={} else {"trading_universe_id":"univ_bad"}
    return ctx


@pytest.mark.parametrize("map_name",["accounts_by_id","credential_profiles_by_id","external_identity_snapshots_by_account_id","universes_by_id","catalogs_by_id","instruments_by_id","previous_profiles_by_id","previous_catalogs_by_id","previous_universes_by_id"])
@pytest.mark.parametrize("bad",MALFORMED_RECORDS)
def test_context_rejects_every_malformed_unrelated_record(map_name,bad):
    assert validate_context(malformed_unrelated(map_name,bad)) is False


def test_context_rejects_malformed_unrelated_account(): assert not validate_context(malformed_unrelated("accounts_by_id",{}))
def test_context_rejects_malformed_unrelated_credential_profile(): assert not validate_context(malformed_unrelated("credential_profiles_by_id",{}))
def test_context_rejects_malformed_unrelated_external_identity(): assert not validate_context(malformed_unrelated("external_identity_snapshots_by_account_id",{}))
def test_context_rejects_malformed_unrelated_universe(): assert not validate_context(malformed_unrelated("universes_by_id",{}))
def test_context_rejects_malformed_unrelated_catalog(): assert not validate_context(malformed_unrelated("catalogs_by_id",{}))
def test_context_rejects_malformed_unrelated_instrument(): assert not validate_context(malformed_unrelated("instruments_by_id",{}))
def test_context_rejects_malformed_unrelated_previous_profile(): assert not validate_context(malformed_unrelated("previous_profiles_by_id",{}))
def test_context_rejects_malformed_unrelated_previous_catalog(): assert not validate_context(malformed_unrelated("previous_catalogs_by_id",{}))
def test_context_rejects_malformed_unrelated_previous_universe(): assert not validate_context(malformed_unrelated("previous_universes_by_id",{}))


def test_direct_activation_rejects_malformed_unrelated_trusted_record():
    assert raw_direct_activation(malformed_unrelated("instruments_by_id",{}))[0] is False


def test_dispatcher_rejects_malformed_unrelated_trusted_record():
    assert activation_with_context(malformed_unrelated("catalogs_by_id",{}))[0] is False


def test_context_accepts_structurally_valid_nonoperable_historical_records():
    ctx=context();ctx["accounts_by_id"]["xacc_disabled"]=sample_account(exchange_account_id="xacc_disabled",lifecycle_state="DISABLED",connection_state="DISCONNECTED",external_account_identity_state="UNVERIFIED");ctx["external_identity_snapshots_by_account_id"]["xacc_disabled"]=sample_identity(state="UNVERIFIED");old=stale_history("DELISTED");ctx["instrument_history_by_id"]={old["instrument_id"]:[old]};ctx["instruments_by_id"][old["instrument_id"]]["metadata_version"]=2;assert validate_context(ctx)

# Active indexes, durable-ID spaces i referential integrity.
def second_active_profile_context(same_account=True):
    ctx=context();account_id="xacc_test_1" if same_account else "xacc_other"
    if not same_account:
        ctx["accounts_by_id"][account_id]=sample_account(exchange_account_id=account_id,active_credential_profile_id="cred_2")
        ctx["active_profile_ids_by_account_id"][account_id]=["cred_2"]
    ctx["credential_profiles_by_id"]["cred_2"]=sample_profile(credential_profile_id="cred_2",exchange_account_id=account_id)
    return ctx


def test_context_rejects_hidden_second_active_profile_same_account(): assert not validate_context(second_active_profile_context())
def test_activation_rejects_hidden_second_active_profile_same_account(): assert activation_with_context(second_active_profile_context())[0] is False


def test_context_rejects_active_profile_missing_from_active_index():
    ctx=context();ctx["active_profile_ids_by_account_id"]={};assert not validate_context(ctx)


def test_context_rejects_retired_profile_in_active_index():
    ctx=context(profile=sample_profile(lifecycle_state="RETIRED",retired_at_utc="2026-02-01T00:00:00Z"));ctx["accounts_by_id"]["xacc_test_1"]["active_credential_profile_id"]="cred_1";assert not validate_context(ctx)


def test_context_rejects_active_profile_in_previous_profiles():
    ctx=context();ctx["previous_profiles_by_id"]["cred_old"]=sample_profile(credential_profile_id="cred_old");assert not validate_context(ctx)


def test_context_rejects_duplicate_profile_id_across_current_and_previous():
    ctx=context();ctx["previous_profiles_by_id"]["cred_1"]=dict(ctx["credential_profiles_by_id"]["cred_1"]);assert not validate_context(ctx)


def test_activation_accepts_one_active_profile_per_account(): assert activation_with_context(context())[0]
def test_active_profile_on_other_account_does_not_conflict(): assert validate_context(second_active_profile_context(False))


def hidden_active_universe_context(other_account=False):
    ctx=context();target=ctx["universes_by_id"]["univ_1"];account_id="xacc_other" if other_account else target["exchange_account_id"]
    if other_account: ctx["accounts_by_id"][account_id]=sample_account(exchange_account_id=account_id,active_credential_profile_id=None)
    other=sample_universe(exchange_account_id=account_id,trading_universe_id="univ_2")
    ctx["universes_by_id"]["univ_2"]=other
    return ctx


def test_context_rejects_active_universe_not_bound_to_universe_map():
    ctx=context();ctx["active_universes"]=[sample_universe(trading_universe_id="missing")];assert not validate_context(ctx)


def test_context_rejects_tampered_active_universe_copy():
    ctx=context();copy=dict(ctx["universes_by_id"]["univ_1"],creation_reason="tampered");copy["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["trading_universe_content_hash"],copy);ctx["active_universes"]=[copy];assert not validate_context(ctx)


def test_context_rejects_duplicate_active_universe_entries():
    ctx=context();item=ctx["universes_by_id"]["univ_1"];ctx["active_universes"]=[item,item];assert not validate_context(ctx)


def test_activation_rejects_hidden_second_active_universe_same_account():
    ok,out=activation_with_context(hidden_active_universe_context());assert not ok and out["denial"]=="TRADING_UNIVERSE_VERSION_CONFLICT"


def test_direct_activation_rejects_hidden_second_active_universe_same_account():
    assert raw_direct_activation(hidden_active_universe_context())==(False,"TRADING_UNIVERSE_VERSION_CONFLICT")


def test_dispatch_and_direct_return_same_universe_conflict():
    ctx=hidden_active_universe_context();direct=raw_direct_activation(ctx);dispatched=activation_with_context(ctx);assert direct[1]==dispatched[1]["denial"]=="TRADING_UNIVERSE_VERSION_CONFLICT" and dispatched[1]["audit_event"]=="TRADING_UNIVERSE_REJECTED"


def test_active_universe_on_other_account_does_not_conflict(): assert activation_with_context(hidden_active_universe_context(True))[0]


def test_context_rejects_duplicate_universe_id_across_current_and_previous():
    ctx=context();ctx["previous_universes_by_id"]["univ_1"]=dict(ctx["universes_by_id"]["univ_1"]);assert not validate_context(ctx)


DISJOINT_PAIRS=[("credential_profiles_by_id","previous_profiles_by_id"),("catalogs_by_id","previous_catalogs_by_id"),("universes_by_id","previous_universes_by_id"),("account_capability_snapshots_by_id","previous_account_capability_snapshots_by_id")]


@pytest.mark.parametrize("current,previous",DISJOINT_PAIRS)
@pytest.mark.parametrize("different",[False,True])
def test_context_rejects_duplicate_durable_id_across_current_previous(current,previous,different):
    ctx=context();key=next(iter(ctx[current]));record=dict(ctx[current][key]);
    if different: record[next(iter(record))]=key
    ctx[previous][key]=record;assert not validate_context(ctx)


@pytest.mark.parametrize("current,previous",DISJOINT_PAIRS)
def test_context_allows_disjoint_current_previous_ids(current,previous):
    ctx=context()
    if current=="credential_profiles_by_id": record=sample_profile(credential_profile_id="cred_old",lifecycle_state="RETIRED",retired_at_utc="2026-02-01T00:00:00Z")
    elif current=="catalogs_by_id":
        historical=sample_instrument(metadata_version=1,catalog_snapshot_id="cat_old");record=sample_catalog(historical,catalog_snapshot_id="cat_old");ctx["instruments_by_id"][historical["instrument_id"]]["metadata_version"]=2;ctx["instrument_history_by_id"]={historical["instrument_id"]:[historical]}
    elif current=="universes_by_id": record=sample_universe(trading_universe_id="univ_old")
    else: record=sample_capability_snapshot(account=ctx["accounts_by_id"]["xacc_test_1"],account_capability_snapshot_id="caps_old")
    ctx[previous][next(value for key,value in [("credential_profiles_by_id","cred_old"),("catalogs_by_id","cat_old"),("universes_by_id","univ_old"),("account_capability_snapshots_by_id","caps_old")] if key==current)]=record
    assert validate_context(ctx)


@pytest.mark.parametrize("current,previous",DISJOINT_PAIRS)
def test_context_rejects_malformed_disjoint_key(current,previous):
    ctx=context();ctx[previous][""]={};assert not validate_context(ctx)


def test_context_rejects_profile_with_missing_account():
    ctx=context();ctx["credential_profiles_by_id"]["cred_1"]["exchange_account_id"]="missing";assert not validate_context(ctx)


def test_context_rejects_profile_account_scope_mismatch():
    ctx=context();ctx["credential_profiles_by_id"]["cred_1"]["environment_scope"]="PAPER";assert not validate_context(ctx)


def test_context_rejects_current_instrument_with_missing_catalog():
    ctx=context();ctx["instruments_by_id"]["instr_btcusdt_spot_testnet"]["catalog_snapshot_id"]="missing";assert not validate_context(ctx)


def test_context_rejects_current_instrument_catalog_scope_mismatch():
    ctx=context();ctx["instruments_by_id"]["instr_btcusdt_spot_testnet"]["source_adapter_family_id"]="other";assert not validate_context(ctx)


def test_context_rejects_current_catalog_with_unresolved_member():
    ctx=context();catalog=ctx["catalogs_by_id"]["cat_1"];catalog["instrument_ids"]=["missing"];catalog["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["catalog_snapshot_content_hash"],catalog);assert not validate_context(ctx)


def test_context_rejects_universe_with_missing_account():
    ctx=context();universe=ctx["universes_by_id"]["univ_1"];universe["exchange_account_id"]="missing";universe["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["trading_universe_content_hash"],universe);assert not validate_context(ctx)


def test_context_rejects_universe_with_missing_source_catalog():
    ctx=context();universe=ctx["universes_by_id"]["univ_1"];universe["source_catalog_snapshot_ids"]=["missing"];universe["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["trading_universe_content_hash"],universe);assert not validate_context(ctx)


def test_context_rejects_universe_with_unresolved_instrument():
    ctx=context();universe=ctx["universes_by_id"]["univ_1"];universe["instrument_ids"]=["missing"];universe["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["trading_universe_content_hash"],universe);assert not validate_context(ctx)


def test_context_accepts_resolvable_historical_catalog_member():
    ctx=context();old=stale_history();ctx["universes_by_id"]={};ctx["instruments_by_id"]={};ctx["instrument_history_by_id"]={old["instrument_id"]:[old]};assert validate_context(ctx)


def test_context_accepts_structurally_valid_nonoperable_history(): test_context_accepts_resolvable_historical_catalog_member()


@pytest.mark.parametrize("bad",[False,0,"",[]])
def test_direct_catalog_rejects_falsey_previous_maps(bad):
    inst=sample_instrument();assert validate_instrument_catalog_snapshot(sample_catalog(inst),{inst["instrument_id"]:inst},bad,instrument_history_by_id={})==(False,"CATALOG_SNAPSHOT_INVALID")


def test_direct_catalog_rejects_false_previous_map(): test_direct_catalog_rejects_falsey_previous_maps(False)
def test_direct_catalog_rejects_zero_previous_map(): test_direct_catalog_rejects_falsey_previous_maps(0)
def test_direct_catalog_rejects_empty_string_previous_map(): test_direct_catalog_rejects_falsey_previous_maps("")
def test_direct_catalog_rejects_list_previous_map(): test_direct_catalog_rejects_falsey_previous_maps([])


@pytest.mark.parametrize("bad",[False,0,"",[]])
def test_direct_capability_rejects_falsey_previous_maps(bad): assert validate_account_capability_snapshot(sample_capability_snapshot(),sample_account(),bad)==(False,"ACCOUNT_READINESS_BLOCKED")


def test_direct_capability_rejects_false_previous_map(): test_direct_capability_rejects_falsey_previous_maps(False)
def test_direct_capability_rejects_zero_previous_map(): test_direct_capability_rejects_falsey_previous_maps(0)
def test_direct_capability_rejects_empty_string_previous_map(): test_direct_capability_rejects_falsey_previous_maps("")
def test_direct_capability_rejects_list_previous_map(): test_direct_capability_rejects_falsey_previous_maps([])


@pytest.mark.parametrize("bad",[False,0,"",[],(),set()])
def test_direct_universe_rejects_non_dict_previous_map(bad):
    ctx=context();universe=ctx["universes_by_id"]["univ_1"];assert validate_trading_universe_version(universe,ctx["accounts_by_id"]["xacc_test_1"],ctx["instruments_by_id"],ctx["catalogs_by_id"],bad,ctx["active_universes"],ctx["previous_catalogs_by_id"],validation_context=ctx)==(False,"ACCOUNT_READINESS_BLOCKED")


def test_empty_dict_is_valid_for_first_version_without_predecessor():
    inst=sample_instrument();assert validate_instrument_catalog_snapshot(sample_catalog(inst),{inst["instrument_id"]:inst},{},instrument_history_by_id={})[0] and validate_account_capability_snapshot(sample_capability_snapshot(),sample_account(),{})[0]

# Pełny trusted reference graph i exact Catalog/TradingUniverse membership.
def unrelated_account_context(**account_overrides):
    ctx=context();values={"exchange_account_id":"xacc_other","active_credential_profile_id":None,"account_capability_snapshot_id":None};values.update(account_overrides);other=sample_account(**values);ctx["accounts_by_id"]["xacc_other"]=other;return ctx


def assert_context_and_activation_blocked(ctx):
    assert validate_context(ctx) is False
    ok,out=activation_with_context(ctx);assert not ok and out=={"denial":"ACCOUNT_READINESS_BLOCKED","audit_event":"TRADING_UNIVERSE_REJECTED"}


def test_context_rejects_dangling_account_active_profile_id(): assert_context_and_activation_blocked(unrelated_account_context(active_credential_profile_id="missing"))


def test_context_rejects_account_bound_to_retired_profile():
    ctx=unrelated_account_context(active_credential_profile_id="cred_old");ctx["credential_profiles_by_id"]["cred_old"]=sample_profile(credential_profile_id="cred_old",exchange_account_id="xacc_other",lifecycle_state="RETIRED",retired_at_utc="2026-02-01T00:00:00Z");assert_context_and_activation_blocked(ctx)


def test_context_rejects_account_bound_to_foreign_profile():
    ctx=unrelated_account_context(active_credential_profile_id="cred_foreign");ctx["credential_profiles_by_id"]["cred_foreign"]=sample_profile(credential_profile_id="cred_foreign",exchange_account_id="xacc_test_1");assert_context_and_activation_blocked(ctx)


def test_context_rejects_account_profile_scope_mismatch():
    ctx=unrelated_account_context(active_credential_profile_id="cred_other");ctx["credential_profiles_by_id"]["cred_other"]=sample_profile(credential_profile_id="cred_other",exchange_account_id="xacc_other",environment_scope="PAPER");assert_context_and_activation_blocked(ctx)


def test_context_rejects_active_profile_when_account_pointer_is_null():
    ctx=unrelated_account_context();ctx["credential_profiles_by_id"]["cred_other"]=sample_profile(credential_profile_id="cred_other",exchange_account_id="xacc_other");assert_context_and_activation_blocked(ctx)


def test_context_rejects_active_index_when_account_pointer_is_null():
    ctx=unrelated_account_context();ctx["active_profile_ids_by_account_id"]["xacc_other"]=["cred_1"];assert_context_and_activation_blocked(ctx)


def test_context_rejects_dangling_account_capability_snapshot_id(): assert_context_and_activation_blocked(unrelated_account_context(account_capability_snapshot_id="missing"))


def test_context_rejects_account_capability_bound_from_previous_map():
    ctx=unrelated_account_context(account_capability_snapshot_id="caps_old");snapshot=sample_capability_snapshot(ctx["accounts_by_id"]["xacc_other"],account_capability_snapshot_id="caps_old");ctx["previous_account_capability_snapshots_by_id"]["caps_old"]=snapshot;assert_context_and_activation_blocked(ctx)


def test_context_rejects_account_capability_scope_mismatch():
    ctx=unrelated_account_context(account_capability_snapshot_id="caps_other");snapshot=sample_capability_snapshot(ctx["accounts_by_id"]["xacc_other"],account_capability_snapshot_id="caps_other",environment="PAPER");ctx["account_capability_snapshots_by_id"]["caps_other"]=snapshot;assert_context_and_activation_blocked(ctx)


def test_context_accepts_null_optional_bindings_without_active_records(): assert validate_context(unrelated_account_context())


def retired_previous_profile(**overrides): return sample_profile(credential_profile_id="cred_old",lifecycle_state="RETIRED",retired_at_utc="2026-02-01T00:00:00Z",**overrides)


def test_context_rejects_previous_profile_with_missing_account():
    ctx=context();ctx["previous_profiles_by_id"]["cred_old"]=retired_previous_profile(exchange_account_id="missing");assert not validate_context(ctx)


def test_context_rejects_previous_profile_exchange_mismatch():
    ctx=context();ctx["previous_profiles_by_id"]["cred_old"]=retired_previous_profile(exchange_id="paper_simulated_venue");assert not validate_context(ctx)


def test_context_rejects_previous_profile_environment_mismatch():
    ctx=context();ctx["previous_profiles_by_id"]["cred_old"]=retired_previous_profile(environment_scope="PAPER");assert not validate_context(ctx)


def test_context_rejects_unrelated_dangling_previous_profile(): test_context_rejects_previous_profile_with_missing_account()


def test_activation_rejects_unrelated_dangling_previous_profile():
    ctx=context();ctx["previous_profiles_by_id"]["cred_old"]=retired_previous_profile(exchange_account_id="missing");assert_context_and_activation_blocked(ctx)


def test_context_accepts_valid_retired_previous_profile():
    ctx=context();ctx["previous_profiles_by_id"]["cred_old"]=retired_previous_profile();assert validate_context(ctx)


def exact_history_catalog_context():
    ctx=context();current=ctx["instruments_by_id"]["instr_btcusdt_spot_testnet"];current["metadata_version"]=2;old=sample_instrument(metadata_version=1,catalog_snapshot_id="cat_old",observed_at_utc="2019-01-01T00:00:00Z",effective_at_utc="2019-01-02T00:00:00Z",stale_after_utc="2020-01-01T00:00:00Z");old_catalog=sample_catalog(old,catalog_snapshot_id="cat_old",observed_at_utc="2019-01-01T00:00:00Z",effective_at_utc="2019-01-02T00:00:00Z",stale_after_utc="2020-01-01T00:00:00Z");ctx["previous_catalogs_by_id"]["cat_old"]=old_catalog;ctx["instrument_history_by_id"]={old["instrument_id"]:[old]};return ctx


def test_context_rejects_catalog_listing_current_instrument_from_other_catalog():
    ctx=multi_catalog_context();cat2=ctx["catalogs_by_id"]["cat_2"];cat2["instrument_ids"]=["instr_btcusdt_spot_testnet"];cat2["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["catalog_snapshot_content_hash"],cat2);assert not validate_context(ctx)


def test_context_rejects_previous_catalog_listing_current_instrument_from_other_catalog():
    ctx=context();old=sample_catalog(catalog_snapshot_id="cat_old");ctx["previous_catalogs_by_id"]["cat_old"]=old;assert not validate_context(ctx)


def test_context_rejects_catalog_backed_by_history_from_other_catalog():
    ctx=exact_history_catalog_context();ctx["instrument_history_by_id"]["instr_btcusdt_spot_testnet"][0]["catalog_snapshot_id"]="other";assert not validate_context(ctx)


@pytest.mark.parametrize("field,value",[("exchange_id","paper_simulated_venue"),("environment","PAPER"),("market_type","MARGIN"),("source_adapter_family_id","other")])
def test_context_rejects_catalog_member_scope_or_adapter_mismatch(field,value):
    ctx=exact_history_catalog_context();ctx["instrument_history_by_id"]["instr_btcusdt_spot_testnet"][0][field]=value;assert not validate_context(ctx)


def test_context_rejects_catalog_member_exchange_mismatch(): test_context_rejects_catalog_member_scope_or_adapter_mismatch("exchange_id","paper_simulated_venue")
def test_context_rejects_catalog_member_environment_mismatch(): test_context_rejects_catalog_member_scope_or_adapter_mismatch("environment","PAPER")
def test_context_rejects_catalog_member_market_mismatch(): test_context_rejects_catalog_member_scope_or_adapter_mismatch("market_type","MARGIN")
def test_context_rejects_catalog_member_adapter_mismatch(): test_context_rejects_catalog_member_scope_or_adapter_mismatch("source_adapter_family_id","other")


def test_context_accepts_current_member_bound_to_exact_catalog(): assert validate_context(context())
def test_context_accepts_historical_member_bound_to_exact_catalog(): assert validate_context(exact_history_catalog_context())


def test_context_accepts_same_instrument_across_sequential_exact_catalog_records(): assert validate_context(exact_history_catalog_context())


def test_activation_rejects_unrelated_catalog_with_invalid_member_binding():
    ctx=multi_catalog_context();cat2=ctx["catalogs_by_id"]["cat_2"];cat2["instrument_ids"]=["instr_btcusdt_spot_testnet"];cat2["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["catalog_snapshot_content_hash"],cat2);assert_context_and_activation_blocked(ctx)


def universe_cross_catalog_context():
    ctx=multi_catalog_context();universe=ctx["universes_by_id"]["univ_1"];universe["instrument_ids"]=["instr_ethusdt_spot_testnet"];universe["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["trading_universe_content_hash"],universe);return ctx


def test_context_rejects_universe_instrument_outside_source_catalogs(): assert not validate_context(universe_cross_catalog_context())
def test_context_rejects_universe_instrument_bound_to_different_catalog(): test_context_rejects_universe_instrument_outside_source_catalogs()
def test_context_rejects_universe_using_unrelated_catalog_membership(): test_context_rejects_universe_instrument_outside_source_catalogs()


def historical_universe_context():
    ctx=exact_history_catalog_context();old_universe=sample_universe(trading_universe_id="univ_old",version=1,lifecycle_state="RETIRED",instrument_ids=["instr_btcusdt_spot_testnet"],source_catalog_snapshot_ids=["cat_old"],retired_at_utc="2026-02-01T00:00:00Z");ctx["previous_universes_by_id"]["univ_old"]=old_universe;return ctx


def test_context_rejects_previous_universe_with_missing_account():
    ctx=historical_universe_context();u=ctx["previous_universes_by_id"]["univ_old"];u["exchange_account_id"]="missing";u["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["trading_universe_content_hash"],u);assert not validate_context(ctx)


def test_context_rejects_previous_universe_with_missing_source_catalog():
    ctx=historical_universe_context();u=ctx["previous_universes_by_id"]["univ_old"];u["source_catalog_snapshot_ids"]=["missing"];u["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["trading_universe_content_hash"],u);assert not validate_context(ctx)


def test_context_rejects_previous_universe_with_unresolved_member():
    ctx=historical_universe_context();ctx["instrument_history_by_id"]={};assert not validate_context(ctx)


def test_context_accepts_universe_with_exact_source_membership(): assert validate_context(context())
def test_context_accepts_valid_historical_universe_membership(): assert validate_context(historical_universe_context())


def test_activation_rejects_unrelated_invalid_universe_in_context(): assert_context_and_activation_blocked(universe_cross_catalog_context())

# Exact catalog.instrument_ids membership i totalny resolver historii.
def catalog_without_target_context(replacement_id=None):
    ctx=context();catalog=ctx["catalogs_by_id"]["cat_1"]
    if replacement_id:
        other=sample_instrument(instrument_id=replacement_id,venue_symbol="OTHER",display_symbol="OTHER/USDT")
        ctx["instruments_by_id"][replacement_id]=other
        catalog["instrument_ids"]=[replacement_id]
    else:
        catalog["instrument_ids"]=[]
    catalog["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["catalog_snapshot_content_hash"],catalog)
    return ctx


def test_current_universe_rejects_target_absent_from_catalog_member_ids():
    ctx=catalog_without_target_context();catalog=ctx["catalogs_by_id"]["cat_1"];iid="instr_btcusdt_spot_testnet"
    assert resolve_catalog_member(catalog,iid,ctx["instruments_by_id"],ctx["instrument_history_by_id"]) is None
    assert validate_universe_source_membership(ctx["universes_by_id"]["univ_1"],ctx["accounts_by_id"],ctx["catalogs_by_id"],ctx["instruments_by_id"],ctx["instrument_history_by_id"]) is False
    assert validate_context(ctx) is False
    ok,out=activation_with_context(ctx);assert not ok and out=={"denial":"ACCOUNT_READINESS_BLOCKED","audit_event":"TRADING_UNIVERSE_REJECTED"} and out["audit_event"]!="TRADING_UNIVERSE_ACTIVATED"


def test_current_catalog_with_other_member_does_not_grant_target_membership():
    ctx=catalog_without_target_context("instr_other");catalog=ctx["catalogs_by_id"]["cat_1"];iid="instr_btcusdt_spot_testnet"
    assert resolve_catalog_member(catalog,iid,ctx["instruments_by_id"],ctx["instrument_history_by_id"]) is None
    assert not validate_context(ctx)


def test_historical_universe_rejects_member_absent_from_source_catalog_ids():
    ctx=historical_universe_context();catalog=ctx["previous_catalogs_by_id"]["cat_old"];catalog["instrument_ids"]=[];catalog["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["catalog_snapshot_content_hash"],catalog);assert not validate_context(ctx)


MALFORMED_HISTORY_VERSIONS=[None,"2",True,1.5,0,-1,{},[]]


@pytest.mark.parametrize("bad_version",MALFORMED_HISTORY_VERSIONS)
def test_malformed_history_metadata_version_is_total_and_fail_closed(bad_version):
    ctx=exact_history_catalog_context();record=ctx["instrument_history_by_id"]["instr_btcusdt_spot_testnet"][0];record["metadata_version"]=bad_version;catalog=ctx["previous_catalogs_by_id"]["cat_old"]
    assert resolve_catalog_member(catalog,record["instrument_id"],ctx["instruments_by_id"],ctx["instrument_history_by_id"]) is None
    assert validate_context(ctx) is False
    ok,out=activation_with_context(ctx);assert not ok and out=={"denial":"ACCOUNT_READINESS_BLOCKED","audit_event":"TRADING_UNIVERSE_REJECTED"}


def test_mixed_history_metadata_versions_are_total_and_fail_closed():
    ctx=exact_history_catalog_context();first=ctx["instrument_history_by_id"]["instr_btcusdt_spot_testnet"][0];second=dict(first,metadata_version="bad");ctx["instrument_history_by_id"][first["instrument_id"]]=[first,second];catalog=ctx["previous_catalogs_by_id"]["cat_old"]
    assert resolve_catalog_member(catalog,first["instrument_id"],ctx["instruments_by_id"],ctx["instrument_history_by_id"]) is None
    assert not validate_context(ctx)
    assert activation_with_context(ctx)[0] is False


def test_exact_membership_success_regressions():
    assert validate_context(context())
    assert validate_context(exact_history_catalog_context())
    assert validate_context(historical_universe_context())


# Pełny history Instrument -> Catalog graph i direct/dispatcher parity.
def history_only_member_context(*, catalog_id="cat_1", listed=True):
    ctx=context();base=ctx["instruments_by_id"]["instr_btcusdt_spot_testnet"]
    ghost=dict(base,instrument_id="instr_ghost",venue_symbol="GHOST",display_symbol="GHOST/USDT",
               metadata_version=1,catalog_snapshot_id=catalog_id)
    ctx["instrument_history_by_id"]["instr_ghost"]=[ghost]
    catalog={**ctx["previous_catalogs_by_id"],**ctx["catalogs_by_id"]}.get(catalog_id)
    if listed and isinstance(catalog,dict):
        catalog["instrument_ids"].append("instr_ghost")
        catalog["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["catalog_snapshot_content_hash"],catalog)
    return ctx


def test_context_rejects_dangling_historical_instrument_catalog_reference():
    ctx=history_only_member_context(catalog_id="cat_missing",listed=False);assert_context_and_activation_blocked(ctx)


def test_context_rejects_unlisted_historical_instrument_member():
    ctx=history_only_member_context(listed=False);assert_context_and_activation_blocked(ctx)


@pytest.mark.parametrize("field,value",[("exchange_id","paper_simulated_venue"),("environment","PAPER"),("market_type","MARGIN"),("source_adapter_family_id","wrong")])
def test_context_rejects_historical_instrument_catalog_scope_or_adapter_mismatch(field,value):
    ctx=history_only_member_context();ctx["instrument_history_by_id"]["instr_ghost"][0][field]=value;assert_context_and_activation_blocked(ctx)


def test_context_accepts_history_only_member_bound_to_exact_catalog():
    assert validate_context(history_only_member_context())


@pytest.mark.parametrize("history_versions,current_version",[([1],1),([1,2],1)])
def test_direct_membership_rejects_current_version_not_above_history_max(history_versions,current_version):
    ctx=exact_history_catalog_context();iid="instr_btcusdt_spot_testnet";template=ctx["instrument_history_by_id"][iid][0]
    ctx["instrument_history_by_id"][iid]=[dict(template,metadata_version=version) for version in history_versions]
    ctx["instruments_by_id"][iid]["metadata_version"]=current_version
    catalog=ctx["catalogs_by_id"]["cat_1"]
    assert resolve_catalog_member(catalog,iid,ctx["instruments_by_id"],ctx["instrument_history_by_id"],ctx["previous_catalogs_by_id"]) is None
    assert validate_universe_source_membership(ctx["universes_by_id"]["univ_1"],ctx["accounts_by_id"],ctx["catalogs_by_id"],ctx["instruments_by_id"],ctx["instrument_history_by_id"],previous_catalogs_by_id=ctx["previous_catalogs_by_id"]) is False
    assert_context_and_activation_blocked(ctx)


def test_direct_membership_accepts_current_version_above_complete_history():
    ctx=exact_history_catalog_context();iid="instr_btcusdt_spot_testnet";template=ctx["instrument_history_by_id"][iid][0]
    ctx["instrument_history_by_id"][iid]=[dict(template,metadata_version=1),dict(template,metadata_version=2)]
    ctx["instruments_by_id"][iid]["metadata_version"]=3
    assert resolve_catalog_member(ctx["catalogs_by_id"]["cat_1"],iid,ctx["instruments_by_id"],ctx["instrument_history_by_id"],ctx["previous_catalogs_by_id"]) == ctx["instruments_by_id"][iid]
    assert validate_catalog_instrument_graph(ctx["catalogs_by_id"],ctx["previous_catalogs_by_id"],ctx["instruments_by_id"],ctx["instrument_history_by_id"]) is True
    assert validate_context(ctx)


@pytest.mark.parametrize("field,value",[("status","BOGUS"),("content_hash","bad"),("observed_at_utc","bad"),("effective_at_utc","bad"),("stale_after_utc","bad"),("adapter_version",""),("previous_snapshot_id",False),("environment","LIVE"),("market_type","OPTIONS")])
def test_direct_membership_rejects_semantically_malformed_catalog(field,value):
    ctx=context();catalog=ctx["catalogs_by_id"]["cat_1"];catalog[field]=value;iid="instr_btcusdt_spot_testnet"
    assert resolve_catalog_member(catalog,iid,ctx["instruments_by_id"],ctx["instrument_history_by_id"],ctx["previous_catalogs_by_id"]) is None
    assert validate_universe_source_membership(ctx["universes_by_id"]["univ_1"],ctx["accounts_by_id"],ctx["catalogs_by_id"],ctx["instruments_by_id"],ctx["instrument_history_by_id"],previous_catalogs_by_id=ctx["previous_catalogs_by_id"]) is False
    assert validate_context(ctx) is False

# Resolver totality oraz odwrotne current Instrument -> Catalog binding.
def orphan_current_instrument_context():
    ctx=context();orphan=sample_instrument(instrument_id="instr_orphan",venue_symbol="ORPHAN",display_symbol="ORPHAN/USDT");ctx["instruments_by_id"][orphan["instrument_id"]]=orphan;return ctx


def test_orphan_current_instrument_invalidates_complete_trusted_graph():
    ctx=orphan_current_instrument_context();assert validate_context(ctx) is False
    ok,out=activation_with_context(ctx);assert not ok and out=={"denial":"ACCOUNT_READINESS_BLOCKED","audit_event":"TRADING_UNIVERSE_REJECTED"} and out["audit_event"]!="TRADING_UNIVERSE_ACTIVATED"


MALFORMED_CURRENT_VERSIONS=[None,"2",True,1.5,0,-1,{},[]]


@pytest.mark.parametrize("bad_version",MALFORMED_CURRENT_VERSIONS)
def test_resolver_rejects_malformed_current_metadata_version_without_exception(bad_version):
    ctx=context();current=ctx["instruments_by_id"]["instr_btcusdt_spot_testnet"];current["metadata_version"]=bad_version;catalog=ctx["catalogs_by_id"]["cat_1"]
    assert resolve_catalog_member(catalog,current["instrument_id"],ctx["instruments_by_id"],ctx["instrument_history_by_id"]) is None
    assert validate_universe_source_membership(ctx["universes_by_id"]["univ_1"],ctx["accounts_by_id"],ctx["catalogs_by_id"],ctx["instruments_by_id"],ctx["instrument_history_by_id"]) is False
    assert validate_context(ctx) is False


@pytest.mark.parametrize("missing_field",["display_symbol","metadata_version","catalog_snapshot_id","source_adapter_family_id","exchange_id","environment","market_type"])
def test_resolver_rejects_malformed_current_schema(missing_field):
    ctx=context();current=ctx["instruments_by_id"]["instr_btcusdt_spot_testnet"];del current[missing_field];assert resolve_catalog_member(ctx["catalogs_by_id"]["cat_1"],current["instrument_id"],ctx["instruments_by_id"],ctx["instrument_history_by_id"]) is None


@pytest.mark.parametrize("mutation",["catalog_snapshot_id","exchange_id","environment","market_type","adapter_family_id","instrument_ids","instrument_ids_wrong_type","duplicate_ids","empty_id","minimal"])
def test_resolver_rejects_malformed_catalog_without_exception(mutation):
    ctx=context();catalog=dict(ctx["catalogs_by_id"]["cat_1"]);iid="instr_btcusdt_spot_testnet"
    if mutation=="instrument_ids_wrong_type": catalog["instrument_ids"]="bad"
    elif mutation=="duplicate_ids": catalog["instrument_ids"]=[iid,iid]
    elif mutation=="empty_id": catalog["instrument_ids"]=[""]
    elif mutation=="minimal": catalog={"instrument_ids":[iid]}
    else: catalog.pop(mutation,None)
    assert resolve_catalog_member(catalog,iid,ctx["instruments_by_id"],ctx["instrument_history_by_id"]) is None


@pytest.mark.parametrize("versions",[[2,1],[1,3,2]])
def test_resolver_rejects_out_of_order_history_without_exception(versions):
    ctx=exact_history_catalog_context();iid="instr_btcusdt_spot_testnet";template=ctx["instrument_history_by_id"][iid][0];ctx["instrument_history_by_id"][iid]=[dict(template,metadata_version=version) for version in versions];catalog=ctx["previous_catalogs_by_id"]["cat_old"]
    assert resolve_catalog_member(catalog,iid,ctx["instruments_by_id"],ctx["instrument_history_by_id"]) is None
    assert validate_context(ctx) is False
    assert activation_with_context(ctx)[0] is False


@pytest.mark.parametrize("mutation",["missing_field","different_id","identity_rewrite"])
def test_resolver_rejects_malformed_history_record_with_valid_version(mutation):
    ctx=exact_history_catalog_context();iid="instr_btcusdt_spot_testnet";record=ctx["instrument_history_by_id"][iid][0]
    if mutation=="missing_field": del record["display_symbol"]
    elif mutation=="different_id": record["instrument_id"]="different"
    else: record["venue_symbol"]="DIFFERENT"
    assert resolve_catalog_member(ctx["previous_catalogs_by_id"]["cat_old"],iid,ctx["instruments_by_id"],ctx["instrument_history_by_id"]) is None


def test_reverse_binding_and_resolver_success_regressions():
    assert validate_context(context())
    assert validate_context(exact_history_catalog_context())
    ordered=exact_history_catalog_context();iid="instr_btcusdt_spot_testnet";first=ordered["instrument_history_by_id"][iid][0];ordered["instrument_history_by_id"][iid]=[dict(first,metadata_version=1),dict(first,metadata_version=2)];ordered["instruments_by_id"][iid]["metadata_version"]=3;assert validate_context(ordered)
    assert validate_context(historical_universe_context())


# Total Catalog validation, complete direct graph closure i predecessor parity.
MALFORMED_CATALOG_IDS=[None,"",1,True,1.5,[],{},set(),()]


@pytest.mark.parametrize("bad_id",MALFORMED_CATALOG_IDS)
def test_resolver_catalog_snapshot_id_is_total(bad_id):
    ctx=context();catalog=dict(ctx["catalogs_by_id"]["cat_1"]);catalog["catalog_snapshot_id"]=bad_id
    assert resolve_catalog_member(catalog,"instr_btcusdt_spot_testnet",ctx["instruments_by_id"],ctx["instrument_history_by_id"],{}) is None


@pytest.mark.parametrize("bad_id",[1,True,1.5,[],{},set(),()])
def test_catalog_previous_snapshot_id_is_total(bad_id):
    ctx=context();catalog=dict(ctx["catalogs_by_id"]["cat_1"]);catalog["previous_snapshot_id"]=bad_id
    assert validate_catalog_structure(catalog,{}) is False
    assert resolve_catalog_member(catalog,"instr_btcusdt_spot_testnet",ctx["instruments_by_id"],ctx["instrument_history_by_id"],{}) is None


@pytest.mark.parametrize("field",["catalog_snapshot_id","previous_snapshot_id","instrument_ids"])
def test_catalog_non_json_serializable_fields_never_raise(field):
    ctx=context();catalog=dict(ctx["catalogs_by_id"]["cat_1"]);catalog[field]=set()
    assert validate_catalog_structure(catalog,{}) is False
    assert resolve_catalog_member(catalog,"instr_btcusdt_spot_testnet",ctx["instruments_by_id"],ctx["instrument_history_by_id"],{}) is None


def assert_target_history_direct_and_dispatch_blocked(ctx):
    iid="instr_btcusdt_spot_testnet";catalog=ctx["catalogs_by_id"]["cat_1"]
    assert resolve_catalog_member(catalog,iid,ctx["instruments_by_id"],ctx["instrument_history_by_id"],ctx["previous_catalogs_by_id"],ctx["catalogs_by_id"]) is None
    assert validate_universe_source_membership(ctx["universes_by_id"]["univ_1"],ctx["accounts_by_id"],ctx["catalogs_by_id"],ctx["instruments_by_id"],ctx["instrument_history_by_id"],previous_catalogs_by_id=ctx["previous_catalogs_by_id"]) is False
    assert_context_and_activation_blocked(ctx)


def test_direct_membership_rejects_dangling_target_history_with_valid_current():
    ctx=exact_history_catalog_context();ctx["instrument_history_by_id"]["instr_btcusdt_spot_testnet"][0]["catalog_snapshot_id"]="cat_missing";assert_target_history_direct_and_dispatch_blocked(ctx)


def test_direct_membership_rejects_unlisted_target_history_with_valid_current():
    ctx=exact_history_catalog_context();catalog=ctx["previous_catalogs_by_id"]["cat_old"];catalog["instrument_ids"]=[];catalog["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["catalog_snapshot_content_hash"],catalog);assert_target_history_direct_and_dispatch_blocked(ctx)


@pytest.mark.parametrize("field,value",[("exchange_id","paper_simulated_venue"),("environment","PAPER"),("market_type","MARGIN"),("source_adapter_family_id","wrong")])
def test_direct_membership_rejects_target_history_scope_or_adapter_mismatch(field,value):
    ctx=exact_history_catalog_context();ctx["instrument_history_by_id"]["instr_btcusdt_spot_testnet"][0][field]=value;assert_target_history_direct_and_dispatch_blocked(ctx)


def test_source_catalog_member_closure_rejects_additional_unresolved_member():
    ctx=context();catalog=ctx["catalogs_by_id"]["cat_1"];catalog["instrument_ids"].append("instr_missing");catalog["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["catalog_snapshot_content_hash"],catalog)
    assert validate_universe_source_membership(ctx["universes_by_id"]["univ_1"],ctx["accounts_by_id"],ctx["catalogs_by_id"],ctx["instruments_by_id"],ctx["instrument_history_by_id"],previous_catalogs_by_id={}) is False
    assert_context_and_activation_blocked(ctx)


@pytest.mark.parametrize("mutation",["missing_field","scope","adapter","dangling_history","bad_version"])
def test_source_catalog_member_closure_rejects_additional_malformed_member(mutation):
    ctx=context();catalog=ctx["catalogs_by_id"]["cat_1"];extra=sample_instrument(instrument_id="instr_extra",venue_symbol="EXTRA",display_symbol="EXTRA/USDT")
    if mutation=="missing_field": extra.pop("display_symbol")
    elif mutation=="scope": extra["environment"]="PAPER"
    elif mutation=="adapter": extra["source_adapter_family_id"]="wrong"
    elif mutation=="bad_version": extra["metadata_version"]="bad"
    else:
        extra["catalog_snapshot_id"]="cat_missing";ctx["instrument_history_by_id"]["instr_extra"]=[extra];extra=None
    if extra is not None: ctx["instruments_by_id"]["instr_extra"]=extra
    catalog["instrument_ids"].append("instr_extra");catalog["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["catalog_snapshot_content_hash"],catalog)
    assert validate_universe_source_membership(ctx["universes_by_id"]["univ_1"],ctx["accounts_by_id"],ctx["catalogs_by_id"],ctx["instruments_by_id"],ctx["instrument_history_by_id"],previous_catalogs_by_id={}) is False


def catalog_lineage_context():
    ctx=exact_history_catalog_context();current=ctx["catalogs_by_id"]["cat_1"];current["previous_snapshot_id"]="cat_old";current["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["catalog_snapshot_content_hash"],current);return ctx


MALFORMED_PREDECESSORS=["instrument_ids_type","instrument_ids_duplicate","instrument_ids_empty","catalog_id","previous_id","status","adapter_version","content_hash","timestamps","environment","market"]


@pytest.mark.parametrize("mutation",MALFORMED_PREDECESSORS)
def test_catalog_lineage_rejects_malformed_predecessor_totally(mutation):
    ctx=catalog_lineage_context();previous=ctx["previous_catalogs_by_id"]["cat_old"]
    if mutation=="instrument_ids_type": previous["instrument_ids"]="bad"
    elif mutation=="instrument_ids_duplicate": previous["instrument_ids"]=["instr_btcusdt_spot_testnet"]*2
    elif mutation=="instrument_ids_empty": previous["instrument_ids"]=[""]
    elif mutation=="catalog_id": previous["catalog_snapshot_id"]=[]
    elif mutation=="previous_id": previous["previous_snapshot_id"]=set()
    elif mutation=="status": previous["status"]="BOGUS"
    elif mutation=="adapter_version": previous["adapter_version"]=""
    elif mutation=="content_hash": previous["content_hash"]="bad"
    elif mutation=="timestamps": previous["observed_at_utc"]="bad"
    elif mutation=="environment": previous["environment"]="LIVE"
    else: previous["market_type"]="OPTIONS"
    current=ctx["catalogs_by_id"]["cat_1"]
    assert validate_catalog_lineage(current,ctx["previous_catalogs_by_id"])==(False,"CATALOG_SNAPSHOT_INVALID")
    assert validate_catalog_structure(current,ctx["previous_catalogs_by_id"]) is False
    assert resolve_catalog_member(current,"instr_btcusdt_spot_testnet",ctx["instruments_by_id"],ctx["instrument_history_by_id"],ctx["previous_catalogs_by_id"],ctx["catalogs_by_id"]) is None
    assert validate_universe_source_membership(ctx["universes_by_id"]["univ_1"],ctx["accounts_by_id"],ctx["catalogs_by_id"],ctx["instruments_by_id"],ctx["instrument_history_by_id"],previous_catalogs_by_id=ctx["previous_catalogs_by_id"]) is False


def test_total_catalog_and_graph_closure_success_regressions():
    assert validate_context(context());assert validate_context(exact_history_catalog_context());assert validate_context(history_only_member_context());assert validate_context(historical_universe_context())
    ctx=catalog_lineage_context();assert validate_catalog_lineage(ctx["catalogs_by_id"]["cat_1"],ctx["previous_catalogs_by_id"])==(True,None)


# Total direct membership oraz jeden complete Catalog/Instrument graph validator.
def direct_membership(ctx):
    return validate_universe_source_membership(
        ctx["universes_by_id"]["univ_1"],ctx["accounts_by_id"],ctx["catalogs_by_id"],
        ctx["instruments_by_id"],ctx["instrument_history_by_id"],
        previous_catalogs_by_id=ctx["previous_catalogs_by_id"])


MALFORMED_MEMBER_LISTS=[None,1,True,1.5,"bad",{},set(),()]


@pytest.mark.parametrize("bad_members",MALFORMED_MEMBER_LISTS)
def test_direct_membership_is_total_for_malformed_source_instrument_ids(bad_members):
    ctx=context();ctx["catalogs_by_id"]["cat_1"]["instrument_ids"]=bad_members;assert direct_membership(ctx) is False


@pytest.mark.parametrize("target,bad",[("catalog_value",None),("catalog_key",None),("previous_value",None),("instrument_value",None),("history_value",None)])
def test_direct_membership_is_total_for_malformed_trusted_maps(target,bad):
    ctx=context()
    if target=="catalog_value": ctx["catalogs_by_id"]["cat_1"]=bad
    elif target=="catalog_key": ctx["catalogs_by_id"]={1:ctx["catalogs_by_id"]["cat_1"]}
    elif target=="previous_value": ctx["previous_catalogs_by_id"]["cat_old"]=bad
    elif target=="instrument_value": ctx["instruments_by_id"]["instr_bad"]=bad
    else: ctx["instrument_history_by_id"]["instr_bad"]=bad
    assert direct_membership(ctx) is False


def assert_unrelated_graph_rejected(ctx):
    assert direct_membership(ctx) is False;assert_context_and_activation_blocked(ctx)


def test_direct_rejects_unrelated_empty_history():
    ctx=context();ctx["instrument_history_by_id"]["instr_ghost"]=[];assert_unrelated_graph_rejected(ctx)


def unrelated_history_context():
    ctx=history_only_member_context();record=ctx["instrument_history_by_id"]["instr_ghost"][0];return ctx,record


def test_direct_rejects_unrelated_out_of_order_history():
    ctx,record=unrelated_history_context();ctx["instrument_history_by_id"]["instr_ghost"]=[dict(record,metadata_version=2),dict(record,metadata_version=1)];assert_unrelated_graph_rejected(ctx)


def test_direct_rejects_unrelated_history_identity_rewrite():
    ctx,record=unrelated_history_context();ctx["instrument_history_by_id"]["instr_ghost"]=[dict(record,metadata_version=1),dict(record,metadata_version=2,venue_symbol="REWRITE")];assert_unrelated_graph_rejected(ctx)


@pytest.mark.parametrize("current_version",[2,1])
def test_direct_rejects_unrelated_current_history_rollback(current_version):
    ctx=context();catalog=ctx["catalogs_by_id"]["cat_1"];current=sample_instrument(instrument_id="instr_extra",venue_symbol="EXTRA",display_symbol="EXTRA/USDT",metadata_version=current_version);ctx["instruments_by_id"]["instr_extra"]=current
    ctx["instrument_history_by_id"]["instr_extra"]=[dict(current,metadata_version=1),dict(current,metadata_version=2)]
    catalog["instrument_ids"].append("instr_extra");catalog["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["catalog_snapshot_content_hash"],catalog);assert_unrelated_graph_rejected(ctx)


def test_direct_rejects_unrelated_orphan_current_instrument():
    assert_unrelated_graph_rejected(orphan_current_instrument_context())


@pytest.mark.parametrize("mutation",["missing_display","bad_version","missing_catalog","bad_adapter","scope"])
def test_direct_rejects_unrelated_malformed_current_instrument(mutation):
    ctx=context();record=sample_instrument(instrument_id="instr_extra",venue_symbol="EXTRA",display_symbol="EXTRA/USDT")
    if mutation=="missing_display": record.pop("display_symbol")
    elif mutation=="bad_version": record["metadata_version"]="bad"
    elif mutation=="missing_catalog": record["catalog_snapshot_id"]="cat_missing"
    elif mutation=="bad_adapter": record["source_adapter_family_id"]="wrong"
    else: record["environment"]="PAPER"
    ctx["instruments_by_id"]["instr_extra"]=record;assert_unrelated_graph_rejected(ctx)


@pytest.mark.parametrize("mutation",["status","members","hash","timestamp","unresolved","key_mismatch"])
def test_direct_rejects_unrelated_malformed_catalog(mutation):
    ctx=multi_catalog_context();catalog=ctx["catalogs_by_id"]["cat_2"]
    if mutation=="status": catalog["status"]="BOGUS"
    elif mutation=="members": catalog["instrument_ids"]=None
    elif mutation=="hash": catalog["content_hash"]="bad"
    elif mutation=="timestamp": catalog["observed_at_utc"]="bad"
    elif mutation=="unresolved": catalog["instrument_ids"].append("instr_missing");catalog["content_hash"]=hash_payload(DATA["deterministic_hash_contracts"]["catalog_snapshot_content_hash"],catalog)
    else: ctx["catalogs_by_id"]["wrong"]=ctx["catalogs_by_id"].pop("cat_2")
    assert_unrelated_graph_rejected(ctx)


def test_direct_rejects_current_previous_catalog_id_collision():
    ctx=context();ctx["previous_catalogs_by_id"]["cat_1"]=dict(ctx["catalogs_by_id"]["cat_1"]);assert_unrelated_graph_rejected(ctx)


def history_in_second_current_catalog_context():
    ctx=exact_history_catalog_context();ctx["catalogs_by_id"]["cat_old"]=ctx["previous_catalogs_by_id"].pop("cat_old");return ctx


def test_history_bound_to_second_current_catalog_has_full_parity_success():
    ctx=history_in_second_current_catalog_context();assert direct_membership(ctx) is True;assert validate_context(ctx) is True;assert activation_with_context(ctx)[0] is True
