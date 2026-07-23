"""Contract tests for CryptoHunter M0.4 environment and ProductCapabilities."""
from __future__ import annotations

import ast
import base64
import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/architecture/cryptohunter_product_architecture/environment_and_product_capabilities.json"
MD = ROOT / "docs/architecture/cryptohunter_product_architecture/environment_and_product_capabilities.md"
ARCH = ROOT / "docs/architecture/cryptohunter_product_architecture/README.md"
VOCAB = ROOT / "docs/architecture/cryptohunter_product_architecture/canonical_domain_vocabulary.json"
PROC = ROOT / "docs/architecture/cryptohunter_product_architecture/process_topology_and_lifecycle.json"
MAIN = ROOT / "README.md"
DATA = json.loads(DOC.read_text())

REQ_GATES = [
    "UI_MODE_SELECTOR", "CONFIG_DESERIALIZATION", "PERSISTED_STATE_RESTORE", "IPC_COMMAND_VALIDATION",
    "PRODUCT_CAPABILITY_POLICY", "LIVE_ACCESS_GRANT_POLICY", "CREDENTIAL_PROFILE_RESOLUTION",
    "SECRET_READ_OR_DECRYPT", "ENDPOINT_RESOLUTION", "ADAPTER_FACTORY", "PRIVATE_CONNECTION_CREATION",
    "RUNTIME_ENVIRONMENT_ACTIVATION", "STRATEGY_ROUTE_BINDING", "ORDER_INTENT_ACCEPTANCE",
    "EXECUTION_COMMAND_SUBMISSION", "RECONNECT", "RECOVERY", "RECONCILIATION_WITH_EXTERNAL_VENUE",
]
GATE_AUDIT = {
    "UI_MODE_SELECTOR": "LIVE_ACTIVATION_BLOCKED_BY_EDITION",
    "CONFIG_DESERIALIZATION": "CONFIG_ENVIRONMENT_REJECTED",
    "PERSISTED_STATE_RESTORE": "PERSISTED_ENVIRONMENT_REJECTED",
    "IPC_COMMAND_VALIDATION": "IPC_ENVIRONMENT_COMMAND_REJECTED",
    "PRODUCT_CAPABILITY_POLICY": "LIVE_ACTIVATION_BLOCKED_BY_EDITION",
    "LIVE_ACCESS_GRANT_POLICY": "LIVE_ACTIVATION_BLOCKED_BY_EDITION",
    "CREDENTIAL_PROFILE_RESOLUTION": "CREDENTIAL_SCOPE_REJECTED",
    "SECRET_READ_OR_DECRYPT": "CREDENTIAL_READ_REJECTED",
    "ENDPOINT_RESOLUTION": "ENDPOINT_POLICY_REJECTED",
    "ADAPTER_FACTORY": "ADAPTER_FACTORY_REJECTED",
    "PRIVATE_CONNECTION_CREATION": "PRIVATE_CONNECTION_REJECTED",
    "RUNTIME_ENVIRONMENT_ACTIVATION": "RUNTIME_ENVIRONMENT_ACTIVATION_REJECTED",
    "STRATEGY_ROUTE_BINDING": "EXECUTION_ROUTE_REJECTED",
    "ORDER_INTENT_ACCEPTANCE": "ORDER_INTENT_REJECTED",
    "EXECUTION_COMMAND_SUBMISSION": "EXECUTION_COMMAND_REJECTED",
    "RECONNECT": "RECONNECT_ENVIRONMENT_REJECTED",
    "RECOVERY": "RECOVERY_ENVIRONMENT_REJECTED",
    "RECONCILIATION_WITH_EXTERNAL_VENUE": "RECONCILIATION_ENVIRONMENT_REJECTED",
}
CANONICAL_PAYLOAD = ["schema_version", "capabilities_id", "edition_id", "issued_at_utc", "expires_at_utc", "non_expiring", "capability_set", "environment_capabilities", "feature_flags", "capability_set_hash", "source", "fail_closed_policy", "revocation_reference"]
HEADER = ["signature_schema_version", "signature_algorithm_id", "key_id"]
ENVELOPE = ["capability_payload", "signature_header", "signed_payload_hash", "signature"]
SIG_INPUT = ["signature_header", "signed_payload_hash"]
PAYLOAD_VERSION = "cryptohunter.product_capabilities.payload.v1"
SIG_VERSION = "cryptohunter.product_capabilities.signature.v1"
DEFAULT_VERSION_REGISTRY = {"payload": {PAYLOAD_VERSION}, "signature": {SIG_VERSION}}
STAGE_BY_RESULT = {
    "STRUCTURE_ACCEPTED":"DOCUMENT_STRUCTURE_VALIDATION", "STRUCTURE_REJECTED":"DOCUMENT_STRUCTURE_VALIDATION",
    "HASHES_ACCEPTED":"CANONICAL_HASH_VALIDATION", "HASHES_REJECTED":"CANONICAL_HASH_VALIDATION", "HASH_VALIDATION_NOT_REACHED":"CANONICAL_HASH_VALIDATION",
    "REGISTRY_ACCEPTED":"REGISTRY_POLICY_VALIDATION", "REGISTRY_REJECTED":"REGISTRY_POLICY_VALIDATION", "REGISTRY_VALIDATION_NOT_REACHED":"REGISTRY_POLICY_VALIDATION",
    "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED":"CRYPTOGRAPHIC_SIGNATURE_VERIFICATION", "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED":"CRYPTOGRAPHIC_SIGNATURE_VERIFICATION", "CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4":"CRYPTOGRAPHIC_SIGNATURE_VERIFICATION", "CRYPTOGRAPHIC_VERIFICATION_NOT_REACHED":"CRYPTOGRAPHIC_SIGNATURE_VERIFICATION",
    "EXPIRY_REVOCATION_ACCEPTED":"EXPIRY_AND_REVOCATION_VALIDATION", "EXPIRY_REVOCATION_REJECTED":"EXPIRY_AND_REVOCATION_VALIDATION", "EXPIRY_REVOCATION_NOT_REACHED":"EXPIRY_AND_REVOCATION_VALIDATION",
    "SNAPSHOT_CREATED":"VALIDATED_SNAPSHOT_CREATION", "SNAPSHOT_NOT_CREATED":"VALIDATED_SNAPSHOT_CREATION",
}
SPECIFIC = {
    "UI_MODE_SELECTOR": ["selected_mode", "mode_visibility", "mode_locked_state"],
    "CONFIG_DESERIALIZATION": ["serialized_requested_environment", "config_schema_version"],
    "PERSISTED_STATE_RESTORE": ["persisted_active_environment", "persisted_last_successful_environment", "persisted_capabilities_identity"],
    "IPC_COMMAND_VALIDATION": ["command_type", "client_role", "authorization_context"],
    "PRODUCT_CAPABILITY_POLICY": ["capability_trust_state", "capability_set", "environment_capability"],
    "LIVE_ACCESS_GRANT_POLICY": ["live_access_grant_id_or_absence", "live_access_grant_validation_state"],
    "CREDENTIAL_PROFILE_RESOLUTION": ["credential_profile_id", "credential_scope", "exchange_account_id"],
    "SECRET_READ_OR_DECRYPT": ["credential_profile_id", "credential_scope", "secret_access_purpose"],
    "ENDPOINT_RESOLUTION": ["requested_endpoint_class", "exchange_capabilities", "exchange_account_id"],
    "ADAPTER_FACTORY": ["requested_adapter_class", "exchange_capabilities", "requested_endpoint_class"],
    "PRIVATE_CONNECTION_CREATION": ["exchange_account_id", "credential_profile_id", "resolved_endpoint_class"],
    "RUNTIME_ENVIRONMENT_ACTIVATION": ["environment_readiness_state", "kill_switch_state", "previous_active_environment"],
    "STRATEGY_ROUTE_BINDING": ["strategy_instance_id", "execution_route_id", "exchange_account_id"],
    "ORDER_INTENT_ACCEPTANCE": ["order_intent_id", "execution_route_id", "intended_environment"],
    "EXECUTION_COMMAND_SUBMISSION": ["execution_command_id", "order_id_or_client_order_id", "intended_environment"],
    "RECONNECT": ["previous_connection_environment", "exchange_account_id", "credential_scope", "endpoint_class"],
    "RECOVERY": ["persisted_environment", "persisted_execution_routes", "recovery_checkpoint_id"],
    "RECONCILIATION_WITH_EXTERNAL_VENUE": ["exchange_account_id", "reconciliation_environment", "credential_scope", "endpoint_class"],
}
TS = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?Z$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ACCEPTED_RESULTS = {"STRUCTURE_ACCEPTED", "HASHES_ACCEPTED", "REGISTRY_ACCEPTED", "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED", "EXPIRY_REVOCATION_ACCEPTED", "SNAPSHOT_CREATED"}
_ACCEPTED_FAIL_CLOSED = {
    "STRUCTURE_ACCEPTED": "STRUCTURE_REJECTED",
    "HASHES_ACCEPTED": "HASHES_REJECTED",
    "REGISTRY_ACCEPTED": "REGISTRY_REJECTED",
    "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED": "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED",
    "EXPIRY_REVOCATION_ACCEPTED": "EXPIRY_REVOCATION_REJECTED",
    "SNAPSHOT_CREATED": "SNAPSHOT_NOT_CREATED",
}
_EXPECTED_ACCEPTED_RESULT = {
    "DOCUMENT_STRUCTURE_VALIDATION": "STRUCTURE_ACCEPTED",
    "CANONICAL_HASH_VALIDATION": "HASHES_ACCEPTED",
    "REGISTRY_POLICY_VALIDATION": "REGISTRY_ACCEPTED",
    "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION": "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED",
    "EXPIRY_AND_REVOCATION_VALIDATION": "EXPIRY_REVOCATION_ACCEPTED",
    "VALIDATED_SNAPSHOT_CREATION": "SNAPSHOT_CREATED",
}
_PREDECESSOR_STAGE = {
    "DOCUMENT_STRUCTURE_VALIDATION": None,
    "CANONICAL_HASH_VALIDATION": "DOCUMENT_STRUCTURE_VALIDATION",
    "REGISTRY_POLICY_VALIDATION": "CANONICAL_HASH_VALIDATION",
    "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION": "REGISTRY_POLICY_VALIDATION",
    "EXPIRY_AND_REVOCATION_VALIDATION": "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION",
    "VALIDATED_SNAPSHOT_CREATION": "EXPIRY_AND_REVOCATION_VALIDATION",
}

@dataclass(frozen=True, order=True)
class PreciseTimestamp:
    utc_datetime_at_whole_second: datetime
    fractional_nanoseconds: int


class _IssuedStageEvidence(Mapping):
    __slots__ = ("_public_fields",)

    def __init__(self, public_fields: Mapping | None = None, **kwargs):
        if kwargs:
            raise TypeError("issued evidence constructor does not accept attestation authority")
        object.__setattr__(self, "_public_fields", dict(public_fields or {}))

    def __getitem__(self, key):
        return self._public_fields[key]

    def __iter__(self):
        return iter(self._public_fields)

    def __len__(self):
        return len(self._public_fields)

    def _immutable(self, *args, **kwargs):
        raise TypeError("issued stage evidence public representation is read-only")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __setattr__(self, name, value):
        raise TypeError("issued stage evidence public representation is read-only")

    def __delattr__(self, name):
        raise TypeError("issued stage evidence public representation is read-only")

    def __copy__(self):
        return dict(self)

    def __deepcopy__(self, memo):
        return copy.deepcopy(dict(self), memo)



def parse_ts(value: object) -> PreciseTimestamp:
    if not isinstance(value, str):
        raise ValueError("timestamp must be string")
    match = TS.fullmatch(value)
    if not match:
        raise ValueError("timestamp profile mismatch")
    year, month, day, hour, minute, second, fraction = match.groups()
    if second == "60":
        raise ValueError("leap seconds are rejected")
    whole_second = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), tzinfo=timezone.utc)
    return PreciseTimestamp(whole_second, int((fraction or "0").ljust(9, "0")))


def validate_expiry(doc: dict, current_utc: object) -> str:
    try:
        if not isinstance(doc.get("non_expiring"), bool):
            return "UNSUPPORTED_SCHEMA"
        if "expires_at_utc" not in doc:
            return "UNSUPPORTED_SCHEMA"
        if "issued_at_utc" not in doc:
            return "UNSUPPORTED_SCHEMA"
        issued = parse_ts(doc["issued_at_utc"])
        current = parse_ts(current_utc)
        if doc["non_expiring"]:
            return "VALID_EXPIRY" if doc["expires_at_utc"] is None else "UNSUPPORTED_SCHEMA"
        if doc["expires_at_utc"] is None:
            return "UNSUPPORTED_SCHEMA"
        expires = parse_ts(doc["expires_at_utc"])
        if expires <= issued:
            return "UNSUPPORTED_SCHEMA"
        return "EXPIRED" if current >= expires else "VALID_EXPIRY"
    except Exception:
        return "UNSUPPORTED_SCHEMA"


def _escape_string(text: str) -> str:
    out = ['"']
    for ch in text:
        code = ord(ch)
        if ch == '"': out.append('\\"')
        elif ch == "\\": out.append('\\\\')
        elif ch == "\b": out.append('\\b')
        elif ch == "\t": out.append('\\t')
        elif ch == "\n": out.append('\\n')
        elif ch == "\f": out.append('\\f')
        elif ch == "\r": out.append('\\r')
        elif code < 0x20: out.append(f"\\u{code:04x}")
        else: out.append(ch)
    out.append('"')
    return "".join(out)


def utf16_code_unit_sort_key(value: str) -> bytes:
    return value.encode("utf-16-be")


def jcs_current_schema_canonicalize(obj: object) -> bytes:
    def emit(value: object) -> str:
        if value is None: return "null"
        if value is True: return "true"
        if value is False: return "false"
        if isinstance(value, str): return _escape_string(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            raise TypeError("numbers are forbidden in current schema canonicalizer")
        if isinstance(value, list): return "[" + ",".join(emit(v) for v in value) + "]"
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise TypeError("JCS object keys must be strings")
            ordered = sorted(value, key=utf16_code_unit_sort_key)
            return "{" + ",".join(_escape_string(k) + ":" + emit(value[k]) for k in ordered) + "}"
        raise TypeError(f"unsupported JSON type: {type(value)!r}")
    return emit(obj).encode("utf-8")

def reject_numbers_in_signed_payload(value: object) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        raise ValueError("numbers are forbidden in current signed payload schema")
    if isinstance(value, list):
        for item in value: reject_numbers_in_signed_payload(item)
        return
    if isinstance(value, dict):
        for item in value.values(): reject_numbers_in_signed_payload(item)
        return
    raise ValueError("unsupported signed payload type")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def capability_set_hash(capability_set: list[str]) -> str:
    return sha_bytes(jcs_current_schema_canonicalize(sorted(capability_set, key=utf16_code_unit_sort_key)))


def base64url_no_padding(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def validate_base64url(value: object) -> bool:
    if not isinstance(value, str) or not value or "=" in value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return False
    if len(value) % 4 == 1:
        return False
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * ((4 - len(value) % 4) % 4))
    except Exception:
        return False
    return bool(decoded) and base64url_no_padding(decoded) == value


def build_envelope(payload: dict, algorithm: str = "alg", key: str = "key") -> dict:
    payload = copy.deepcopy(payload)
    payload["capability_set_hash"] = capability_set_hash(payload["capability_set"])
    header = {"signature_schema_version": SIG_VERSION, "signature_algorithm_id": algorithm, "key_id": key}
    signed_payload_hash = sha_bytes(jcs_current_schema_canonicalize(payload))
    signature_input = {"signature_header": header, "signed_payload_hash": signed_payload_hash}
    structural_signature_fixture = base64url_no_padding(hashlib.sha256(jcs_current_schema_canonicalize(signature_input)).digest())
    return {"capability_payload": payload, "signature_header": header, "signed_payload_hash": signed_payload_hash, "signature": structural_signature_fixture}


def _validate_parsed_document_structure(envelope: dict, version_registry=DEFAULT_VERSION_REGISTRY) -> dict:
    schemas = DATA["ProductCapabilities"]["document_schemas"]
    try:
        if set(envelope) != set(ENVELOPE): return _stage_result("STRUCTURE_REJECTED")
        payload, header = envelope["capability_payload"], envelope["signature_header"]
        if "schema_registry" in payload: return _stage_result("STRUCTURE_REJECTED", "PAYLOAD_SCHEMA_REGISTRY_OVERRIDE_FORBIDDEN", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA")
        if set(payload) != set(schemas["capability_payload_schema"]["required_fields"]): return _stage_result("STRUCTURE_REJECTED")
        if set(header) != set(schemas["signature_header_schema"]["required_fields"]): return _stage_result("STRUCTURE_REJECTED")
        if "trust_state" in envelope or "trust_state" in payload or "signature" in payload: return _stage_result("STRUCTURE_REJECTED")
        if not validate_base64url(envelope["signature"]): return _stage_result("STRUCTURE_REJECTED")
        if not HEX64.fullmatch(envelope["signed_payload_hash"]): return _stage_result("STRUCTURE_REJECTED")
        if not version_registry or not version_registry.get("payload") or not version_registry.get("signature"):
            return _stage_result("STRUCTURE_REJECTED", "MISSING_SCHEMA_VERSION_REGISTRY", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA")
        if payload["schema_version"] not in version_registry["payload"]: return _stage_result("STRUCTURE_REJECTED", "UNKNOWN_CAPABILITY_PAYLOAD_SCHEMA_VERSION", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA")
        if header["signature_schema_version"] not in version_registry["signature"]: return _stage_result("STRUCTURE_REJECTED", "UNKNOWN_SIGNATURE_SCHEMA_VERSION", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA")
        reject_numbers_in_signed_payload(payload)
        for field, contract in schemas["capability_payload_schema"]["field_contracts"].items():
            if not validate_field(payload[field], contract): return _stage_result("STRUCTURE_REJECTED")
        for field, contract in schemas["signature_header_schema"]["field_contracts"].items():
            if not validate_field(header[field], contract): return _stage_result("STRUCTURE_REJECTED")
        if not validate_nested_environment_capabilities(payload["environment_capabilities"]): return _stage_result("STRUCTURE_REJECTED")
        if not validate_feature_flags(payload["feature_flags"]): return _stage_result("STRUCTURE_REJECTED")
        edition_policy = DATA["current_edition_signed_payload_policy"]
        if not _current_edition_policy_consistent(envelope): return _stage_result("STRUCTURE_REJECTED", "EDITION_POLICY_MISMATCH", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA")
        return _EVIDENCE_RUNTIME.structure_validated(_context_from_envelope(envelope))
    except Exception:
        return _stage_result("STRUCTURE_REJECTED")


def validate_canonical_hashes(envelope: dict, structure_result: Mapping) -> dict:
    try:
        if not isinstance(structure_result, Mapping) or structure_result.get("stage_result") != "STRUCTURE_ACCEPTED":
            return _stage_result("HASH_VALIDATION_NOT_REACHED")
        context = _context_from_envelope(envelope)
        if not _complete_clean_evidence(structure_result, "DOCUMENT_STRUCTURE_VALIDATION", "STRUCTURE_ACCEPTED", context):
            return _stage_result("HASHES_REJECTED", "PREVIOUS_STAGE_EVIDENCE_INVALID", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA", context, structure_result.get("stage_evidence_id") if isinstance(structure_result, Mapping) else None)
        payload = envelope["capability_payload"]
        if payload["capability_set"] != sorted(payload["capability_set"], key=utf16_code_unit_sort_key): return _stage_result("HASHES_REJECTED", context=context, predecessor_stage_evidence_id=structure_result.get("stage_evidence_id"))
        if payload["capability_set_hash"] != capability_set_hash(payload["capability_set"]): return _stage_result("HASHES_REJECTED", context=context, predecessor_stage_evidence_id=structure_result.get("stage_evidence_id"))
        if envelope["signed_payload_hash"] != sha_bytes(jcs_current_schema_canonicalize(payload)): return _stage_result("HASHES_REJECTED", context=context, predecessor_stage_evidence_id=structure_result.get("stage_evidence_id"))
        return _EVIDENCE_RUNTIME.hashes_validated(context, structure_result)
    except Exception:
        return _stage_result("HASHES_REJECTED", "PREVIOUS_STAGE_EVIDENCE_INVALID", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA")


def cryptographic_signature_verification(envelope: dict, registry_result: Mapping) -> dict:
    try:
        if not isinstance(registry_result, Mapping) or registry_result.get("stage_result") != "REGISTRY_ACCEPTED":
            return _stage_result("CRYPTOGRAPHIC_VERIFICATION_NOT_REACHED")
        context = _context_from_envelope(envelope)
        if not _complete_clean_evidence(registry_result, "REGISTRY_POLICY_VALIDATION", "REGISTRY_ACCEPTED", context):
            return _stage_result("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED", "PREVIOUS_STAGE_EVIDENCE_INVALID", "INVALID_SIGNATURE", "PRODUCT_CAPABILITIES_INVALID_SIGNATURE", context, registry_result.get("stage_evidence_id") if isinstance(registry_result, Mapping) else None)
        return _stage_result("CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4", context=context, predecessor_stage_evidence_id=registry_result["stage_evidence_id"])
    except Exception:
        return _stage_result("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED", "PREVIOUS_STAGE_EVIDENCE_INVALID", "INVALID_SIGNATURE", "PRODUCT_CAPABILITIES_INVALID_SIGNATURE")


def create_validated_snapshot(envelope: dict, structure_result: Mapping, hash_result: Mapping, registry_result: Mapping, crypto_result: Mapping, expiry_revocation_result: Mapping) -> dict:
    try:
        context = _context_from_envelope(envelope)
        evidences = [structure_result, hash_result, registry_result, crypto_result, expiry_revocation_result]
        expected = [
            ("DOCUMENT_STRUCTURE_VALIDATION", "STRUCTURE_ACCEPTED"),
            ("CANONICAL_HASH_VALIDATION", "HASHES_ACCEPTED"),
            ("REGISTRY_POLICY_VALIDATION", "REGISTRY_ACCEPTED"),
            ("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION", "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED"),
            ("EXPIRY_AND_REVOCATION_VALIDATION", "EXPIRY_REVOCATION_ACCEPTED"),
        ]
        predecessors = [None, structure_result, hash_result, registry_result, crypto_result]
        for evidence, (stage_id, result), predecessor in zip(evidences, expected, predecessors):
            if not _complete_clean_evidence(evidence, stage_id, result, context, predecessor):
                return _stage_result("SNAPSHOT_NOT_CREATED", "PREVIOUS_STAGE_EVIDENCE_INVALID", context=context)
        if crypto_result.get("fixture_availability") != DATA["stage_evidence_chain_contract"]["future_crypto_fixture_marker"]:
            return _stage_result("SNAPSHOT_NOT_CREATED", "PREVIOUS_STAGE_EVIDENCE_INVALID", context=context)
        if not _current_edition_policy_consistent(envelope) or not _canonical_hashes_revalidated(envelope):
            return _stage_result("SNAPSHOT_NOT_CREATED", "DOCUMENT_IDENTITY_MISMATCH", context=context)
        return _EVIDENCE_RUNTIME.snapshot_validated(context, expiry_revocation_result)
    except Exception:
        return _stage_result("SNAPSHOT_NOT_CREATED", "PREVIOUS_STAGE_EVIDENCE_INVALID")


def _valid_registry_container(value: object) -> bool:
    return isinstance(value, (set, frozenset, list, tuple)) and all(isinstance(item, str) and item for item in value)


def validate_registry_policy(envelope: dict, hash_result: Mapping, recognized_algorithm_ids: set[str] | None, recognized_key_ids: set[str] | None) -> dict:
    contract = DATA["signature_contract"]["signature_verification_registry_contract"]
    def rejected(reason: str, context: dict | None = None) -> dict:
        return _stage_result(contract["rejected_stage_result"], reason, contract["reason_code_to_trust_state"].get(reason), contract["reason_code_to_denial_code"].get(reason), context, hash_result.get("stage_evidence_id") if isinstance(hash_result, Mapping) else None)
    try:
        if not isinstance(hash_result, Mapping) or hash_result.get("stage_result") != "HASHES_ACCEPTED":
            return _stage_result("REGISTRY_VALIDATION_NOT_REACHED")
        if hash_result.get("stage_id") != "CANONICAL_HASH_VALIDATION":
            return _stage_result("REGISTRY_REJECTED", "PREVIOUS_STAGE_EVIDENCE_INVALID", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA")
        if not isinstance(envelope, dict):
            return rejected("REGISTRY_INPUT_DOCUMENT_INVALID")
        header = envelope.get("signature_header")
        payload = envelope.get("capability_payload")
        context = None
        if not isinstance(header, dict) or not isinstance(payload, dict):
            return rejected("REGISTRY_INPUT_DOCUMENT_INVALID")
        if "signature_algorithm_id" in payload or "key_id" in payload:
            return rejected("REGISTRY_INPUT_DOCUMENT_INVALID")
        algorithm_id = header.get("signature_algorithm_id")
        key_id = header.get("key_id")
        if not isinstance(algorithm_id, str) or not algorithm_id or not isinstance(key_id, str) or not key_id:
            return rejected("REGISTRY_REFERENCE_TYPE_INVALID")
        context = _context_from_envelope(envelope)
        if not _complete_clean_evidence(hash_result, "CANONICAL_HASH_VALIDATION", "HASHES_ACCEPTED", context):
            return rejected("PREVIOUS_STAGE_EVIDENCE_INVALID", context)
    except Exception:
        return rejected("REGISTRY_INPUT_DOCUMENT_INVALID")
    if recognized_algorithm_ids is None:
        return rejected("MISSING_SIGNATURE_ALGORITHM_REGISTRY", context)
    if recognized_key_ids is None:
        return rejected("MISSING_VERIFICATION_KEY_REGISTRY", context)
    if not _valid_registry_container(recognized_algorithm_ids) or not _valid_registry_container(recognized_key_ids):
        return rejected("REGISTRY_CONTAINER_INVALID", context)
    if len(recognized_algorithm_ids) == 0:
        return rejected("MISSING_SIGNATURE_ALGORITHM_REGISTRY", context)
    if len(recognized_key_ids) == 0:
        return rejected("MISSING_VERIFICATION_KEY_REGISTRY", context)
    if algorithm_id not in set(recognized_algorithm_ids):
        return rejected("UNKNOWN_SIGNATURE_ALGORITHM", context)
    if key_id not in set(recognized_key_ids):
        return rejected("UNKNOWN_KEY_ID", context)
    return _EVIDENCE_RUNTIME.registry_validated(context, hash_result)


def steps(flow=None):
    return {step["step_id"]: step for step in (flow or DATA["live_activation_dialog_flow"])["steps"]}


def run_path(events: list[str], flow=None):
    flow = flow or DATA["live_activation_dialog_flow"]
    sid = flow["entry_step_id"]
    audits, props, used = [], [], 0
    while True:
        step = steps(flow)[sid]
        props.append(step)
        if "audit_event" in step:
            audits.append(step["audit_event"])
        if step["step_kind"] == "TERMINAL":
            if used != len(events):
                raise ValueError("unused decision events")
            return step["terminal_outcome"], audits, props
        transitions = step["transitions"]
        if step["step_kind"] == "ACTION":
            if len(transitions) != 1: raise ValueError("ambiguous action")
            event = transitions[0]["event"]
        else:
            if used >= len(events): raise ValueError("missing decision event")
            event = events[used]; used += 1
        matches = [t for t in transitions if t["event"] == event]
        if len(matches) != 1: raise ValueError("unknown or ambiguous event")
        sid = matches[0]["destination_step_id"]


def validate_graph(flow):
    ids = [step["step_id"] for step in flow["steps"]]
    assert len(ids) == len(set(ids)) and flow["entry_step_id"] in ids
    terminal_counts = {outcome: 0 for outcome in flow["terminal_outcomes"]}
    def dfs(sid, stack=()):
        assert sid not in stack
        step = steps(flow)[sid]
        if step["step_kind"] == "TERMINAL": return {sid}
        reachable = {sid}
        for transition in step["transitions"]:
            reachable |= dfs(transition["destination_step_id"], stack + (sid,))
        return reachable
    for step in flow["steps"]:
        assert step["step_kind"] in {"ACTION", "DECISION", "TERMINAL"}
        if step["step_kind"] == "ACTION": assert len(step["transitions"]) == 1
        elif step["step_kind"] == "DECISION":
            events = [transition.get("event") for transition in step["transitions"]]
            assert len(events) >= 2 and len(events) == len(set(events)) and all(events)
            assert all(transition["destination_step_id"] in ids for transition in step["transitions"])
        else:
            assert "transitions" not in step
            assert step.get("sets_active_environment_live") is not True
            assert step["terminal_outcome"] in terminal_counts
            terminal_counts[step["terminal_outcome"]] += 1
    assert set(flow["terminal_outcomes"]) == {"CANCELLED", "AUTHENTICATION_FAILED", "LIVE_BLOCKED_BY_EDITION"}
    assert all(count == 1 for count in terminal_counts.values())
    assert dfs(flow["entry_step_id"]) == set(ids)


def gates():
    return {gate["gate_id"]: gate for gate in DATA["live_enforcement_gates"]}


def test_scope_and_basic_contract():
    assert DOC.exists() and MD.exists() and DATA["status"] == "under audit"
    assert "environment_and_product_capabilities" not in MAIN.read_text()
    assert "## Status M0.4 — under audit" in ARCH.read_text()
    assert [env["environment_id"] for env in DATA["execution_environments"]] == ["PAPER", "TESTNET", "LIVE"]


def test_timestamp_profile_preserves_nanoseconds_and_rejects_bad_formats():
    policy = DATA["ProductCapabilities"]["expiry_policy"]
    assert {key: policy[key] for key in ["timestamp_format", "timestamp_profile_id", "timezone_policy", "fractional_seconds_policy", "leap_second_policy"]} == {
        "timestamp_format": "RFC3339_UTC", "timestamp_profile_id": "RFC3339_UTC_ZULU_SECONDS", "timezone_policy": "UTC_Z_REQUIRED", "fractional_seconds_policy": "OPTIONAL_1_TO_9_DIGITS", "leap_second_policy": "REJECT",
    }
    assert parse_ts("2025-01-01T00:00:00Z").fractional_nanoseconds == 0
    assert parse_ts("2025-01-01T00:00:00.1Z").fractional_nanoseconds == 100000000
    assert parse_ts("2025-01-01T00:00:00.000001Z").fractional_nanoseconds == 1000
    assert parse_ts("2025-01-01T00:00:00.000000001Z").fractional_nanoseconds == 1
    assert parse_ts("2025-01-01T00:00:00.123456789Z").fractional_nanoseconds == 123456789
    for value in ["2025-01-01 00:00:00Z", "20250101T000000Z", "2025-W01-3T00:00:00Z", "2025-01-01T00:00Z", "2025-01-01T00:00:00,123Z", "2025-01-01t00:00:00Z", "2025-01-01T00:00:00z", "2025-01-01T00:00:00+00:00", "2025-02-30T00:00:00Z", "2025-01-01T00:00:60Z", 1, None, ""]:
        try: parse_ts(value); raise AssertionError(value)
        except ValueError: pass


def test_expiry_interpreter_positive_negative_and_nanosecond_boundaries():
    assert parse_ts("2025-01-01T00:00:00.000000002Z") > parse_ts("2025-01-01T00:00:00.000000001Z")
    base = {"non_expiring": False, "issued_at_utc": "2025-01-01T00:00:00.000000001Z", "expires_at_utc": "2025-01-01T00:00:00.000000002Z"}
    assert validate_expiry(base, "2025-01-01T00:00:00.000000001Z") == "VALID_EXPIRY"
    assert validate_expiry(base, "2025-01-01T00:00:00.000000002Z") == "EXPIRED"
    assert validate_expiry(base, "2025-01-01T00:00:00.000000003Z") == "EXPIRED"
    assert validate_expiry({"non_expiring": True, "expires_at_utc": None, "issued_at_utc": "2025-01-01T00:00:00Z"}, "2026-01-01T00:00:00Z") == "VALID_EXPIRY"
    assert validate_expiry({"non_expiring": False, "expires_at_utc": "2027-01-01T00:00:00Z", "issued_at_utc": "2025-01-01T00:00:00Z"}, "2026-01-01T00:00:00Z") == "VALID_EXPIRY"
    malformed = [
        {"non_expiring": True, "expires_at_utc": "2027-01-01T00:00:00Z", "issued_at_utc": "2025-01-01T00:00:00Z"},
        {"non_expiring": False, "expires_at_utc": None, "issued_at_utc": "2025-01-01T00:00:00Z"},
        {"expires_at_utc": None, "issued_at_utc": "2025-01-01T00:00:00Z"},
        {"non_expiring": True, "issued_at_utc": "2025-01-01T00:00:00Z"},
        {"non_expiring": True, "expires_at_utc": None},
        {"non_expiring": "true", "expires_at_utc": None, "issued_at_utc": "2025-01-01T00:00:00Z"},
        {"non_expiring": False, "expires_at_utc": 1, "issued_at_utc": "2025-01-01T00:00:00Z"},
        {"non_expiring": False, "expires_at_utc": "2027-01-01T00:00:00Z", "issued_at_utc": "bad"},
        {"non_expiring": False, "expires_at_utc": "bad", "issued_at_utc": "2025-01-01T00:00:00Z"},
        {"non_expiring": False, "expires_at_utc": "2027-01-01T00:00:00+00:00", "issued_at_utc": "2025-01-01T00:00:00Z"},
        {"non_expiring": False, "expires_at_utc": "2025-01-01T00:00:00Z", "issued_at_utc": "2025-01-01T00:00:00Z"},
        {"non_expiring": False, "expires_at_utc": "2024-01-01T00:00:00Z", "issued_at_utc": "2025-01-01T00:00:00Z"},
    ]
    for doc in malformed:
        assert validate_expiry(doc, "2026-01-01T00:00:00Z") == "UNSUPPORTED_SCHEMA"


def test_current_schema_jcs_vectors_and_signed_payload_type_policy():
    assert DATA["signature_contract"]["signed_payload_json_type_policy"]["number"] == "FORBIDDEN_IN_CURRENT_SCHEMA"
    for numeric in [{"n": 1}, {"n": 1.0}, {"nested": {"n": 1.5}}, {"a": ["x", 2]}]:
        try: jcs_current_schema_canonicalize(numeric); raise AssertionError(numeric)
        except TypeError: pass
        try: reject_numbers_in_signed_payload(numeric); raise AssertionError(numeric)
        except ValueError: pass
    assert jcs_current_schema_canonicalize({"\ue000": "bmp", "\U00010000": "supplementary"}) == '{"𐀀":"supplementary","":"bmp"}'.encode("utf-8")
    assert jcs_current_schema_canonicalize({"s": "line\nquote\"backslash\\"}) == b'{"s":"line\\nquote\\"backslash\\\\"}'
    assert jcs_current_schema_canonicalize({"b": [True, None, "x"], "a": "x"}) == jcs_current_schema_canonicalize({"a": "x", "b": [True, None, "x"]})


def test_single_authoritative_document_schema_registry_and_recursive_no_duplicates():
    schemas = DATA["ProductCapabilities"]["document_schemas"]
    assert schemas["capability_payload_schema"]["required_fields"] == CANONICAL_PAYLOAD
    assert schemas["signature_header_schema"]["required_fields"] == HEADER
    assert schemas["signed_document_envelope_schema"]["required_fields"] == ENVELOPE
    assert schemas["signature_input_schema"]["required_fields"] == SIG_INPUT
    assert schemas["signature_input_schema"]["canonical_object_shape"] == {"signature_header": "<full_signature_header>", "signed_payload_hash": "<signed_payload_hash>"}
    required_paths, shape_paths, full_copies = [], [], {"payload": 0, "header": 0, "envelope": 0, "sig_input": 0}
    def walk(value, path=()):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "required_fields": required_paths.append(path + (key,))
                if key == "canonical_object_shape": shape_paths.append(path + (key,))
                walk(child, path + (key,))
        elif isinstance(value, list):
            if value == CANONICAL_PAYLOAD: full_copies["payload"] += 1
            if value == HEADER: full_copies["header"] += 1
            if value == ENVELOPE: full_copies["envelope"] += 1
            if value == SIG_INPUT: full_copies["sig_input"] += 1
            for index, child in enumerate(value): walk(child, path + (str(index),))
    walk(DATA)
    assert all(path[:2] == ("ProductCapabilities", "document_schemas") for path in required_paths)
    assert full_copies == {"payload": 1, "header": 1, "envelope": 1, "sig_input": 1}
    assert shape_paths == [("ProductCapabilities", "document_schemas", "signature_input_schema", "canonical_object_shape")]
    for container in [DATA["ProductCapabilities"]["contract_schema"], DATA["signature_contract"]]:
        for ref in ["capability_payload_schema_ref", "signature_header_schema_ref", "signed_document_envelope_schema_ref", "signature_input_schema_ref"]:
            assert container[ref] in schemas
        assert "required_fields" not in container


def test_base64url_no_padding_validator():
    assert validate_base64url(base64url_no_padding(b"fixture"))
    for value in ["", "A", "AA=", "AA+", "AA/", "AA ", "AB", 1, None]:
        assert not validate_base64url(value)


def _payload(capabilities_id: str = "cap"):
    policy = DATA["ProductCapabilities"]["current_edition_capability_policy"]
    return {"schema_version": PAYLOAD_VERSION, "capabilities_id": capabilities_id, "edition_id": policy["edition_id"], "issued_at_utc": "2025-01-01T00:00:00Z", "expires_at_utc": None, "non_expiring": True, "capability_set": list(policy["capability_set"]), "environment_capabilities": copy.deepcopy(policy["environment_capabilities"]), "feature_flags": copy.deepcopy(policy["feature_flags"]), "capability_set_hash": "", "source": policy["source"], "fail_closed_policy": "SAFE_LOCAL_ONLY", "revocation_reference": "caprev_test1234"}


def validate_expiry_revocation_stage(envelope: dict, crypto_result: Mapping, current_utc: str, revocation_registry: dict | None) -> dict:
    try:
        if not isinstance(crypto_result, Mapping) or crypto_result.get("stage_result") != "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED":
            return _stage_result("EXPIRY_REVOCATION_NOT_REACHED")
        context = _context_from_envelope(envelope)
        if not _complete_clean_evidence(crypto_result, "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION", "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED", context):
            return _stage_result("EXPIRY_REVOCATION_REJECTED", "PREVIOUS_STAGE_EVIDENCE_INVALID", "INVALID_SIGNATURE", "PRODUCT_CAPABILITIES_INVALID_SIGNATURE", context)
        policy_decision = _evaluate_expiry_and_revocation_policy(envelope, current_utc, revocation_registry)
        return _EVIDENCE_RUNTIME.expiry_validated(context, crypto_result) if policy_decision["policy_result"] == "EXPIRY_REVOCATION_ACCEPTED" else _stage_result(policy_decision["policy_result"], policy_decision.get("diagnostic_reason_code"), policy_decision.get("mapped_trust_state"), policy_decision.get("mapped_denial_code"), context, crypto_result["stage_evidence_id"])
    except Exception:
        return _stage_result("EXPIRY_REVOCATION_REJECTED", "PREVIOUS_STAGE_EVIDENCE_INVALID", "INVALID_SIGNATURE", "PRODUCT_CAPABILITIES_INVALID_SIGNATURE")


def _evaluate_expiry_and_revocation_policy(envelope: dict, current_utc: str, revocation_registry: dict | None) -> dict:
    policy = DATA["capability_revocation_policy"]
    context = _context_from_envelope(envelope)
    payload = envelope["capability_payload"]
    expiry = validate_expiry(payload, current_utc)
    if expiry == "EXPIRED":
        return {"policy_result": "EXPIRY_REVOCATION_REJECTED", "diagnostic_reason_code": "EXPIRED", "mapped_trust_state": "EXPIRED", "mapped_denial_code": "PRODUCT_CAPABILITIES_EXPIRED", "audit_event": "PRODUCT_CAPABILITIES_REJECTED"}
    if expiry != "VALID_EXPIRY":
        return {"policy_result": "EXPIRY_REVOCATION_REJECTED", "diagnostic_reason_code": "UNSUPPORTED_EXPIRY", "mapped_trust_state": "UNSUPPORTED_SCHEMA", "mapped_denial_code": "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA", "audit_event": "PRODUCT_CAPABILITIES_REJECTED"}
    if revocation_registry is None:
        mapping = policy["status_mappings"]["REGISTRY_UNAVAILABLE"]
        return {"policy_result": mapping["stage_result"], "diagnostic_reason_code": mapping["diagnostic_reason_code"], "mapped_trust_state": mapping["mapped_trust_state"], "mapped_denial_code": mapping["mapped_denial_code"], "audit_event": mapping.get("audit_event")}
    if "revocation_status" in payload:
        mapping = policy["status_mappings"]["STATUS_UNKNOWN"]
        return {"policy_result": mapping["stage_result"], "diagnostic_reason_code": "DOCUMENT_SUPPLIED_REVOCATION_STATUS_FORBIDDEN", "mapped_trust_state": mapping["mapped_trust_state"], "mapped_denial_code": mapping["mapped_denial_code"], "audit_event": mapping.get("audit_event")}
    reference = payload["revocation_reference"] or payload["capabilities_id"]
    if not revocation_registry:
        mapping = policy["status_mappings"]["REGISTRY_UNAVAILABLE"]
        return {"policy_result": mapping["stage_result"], "diagnostic_reason_code": mapping["diagnostic_reason_code"], "mapped_trust_state": mapping["mapped_trust_state"], "mapped_denial_code": mapping["mapped_denial_code"], "audit_event": mapping.get("audit_event")}
    if reference not in revocation_registry:
        status = "STATUS_UNKNOWN"
    else:
        status = revocation_registry.get(reference)
    if status not in policy["statuses"]:
        status = policy["invalid_status_result"]
    mapping = policy["status_mappings"][status]
    return {"policy_result": mapping["stage_result"], "diagnostic_reason_code": mapping.get("diagnostic_reason_code"), "mapped_trust_state": mapping["mapped_trust_state"], "mapped_denial_code": mapping["mapped_denial_code"], "audit_event": mapping.get("audit_event")}


def test_separated_pipeline_hash_registry_crypto_and_version_regressions():
    envelope = build_envelope(_payload())
    structure = _validate_parsed_document_structure(envelope)
    assert structure["stage_result"] == "STRUCTURE_ACCEPTED"
    arbitrary = copy.deepcopy(envelope); arbitrary["signature"] = base64url_no_padding(b"arbitrary-asymmetric-signature-shaped-fixture")
    assert _validate_parsed_document_structure(arbitrary)["stage_result"] == "STRUCTURE_ACCEPTED"
    assert validate_canonical_hashes(arbitrary, _validate_parsed_document_structure(arbitrary))["stage_result"] == "HASHES_ACCEPTED"
    assert cryptographic_signature_verification(arbitrary, validate_registry_policy(arbitrary, validate_canonical_hashes(arbitrary, _validate_parsed_document_structure(arbitrary)), {"alg"}, {"key"}))["stage_result"] == "CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4"
    bad_hash = copy.deepcopy(envelope); bad_hash["capability_payload"]["capability_set_hash"] = "0" * 64
    assert _validate_parsed_document_structure(bad_hash)["stage_result"] == "STRUCTURE_ACCEPTED"
    assert validate_canonical_hashes(bad_hash, _validate_parsed_document_structure(bad_hash))["stage_result"] == "HASHES_REJECTED"
    registry = validate_registry_policy(envelope, validate_canonical_hashes(envelope, structure), {"other"}, {"key"})
    assert registry["stage_result"] == "REGISTRY_REJECTED" and registry["diagnostic_reason_code"] == "UNKNOWN_SIGNATURE_ALGORITHM"
    assert create_validated_snapshot(envelope, structure, validate_canonical_hashes(envelope, structure), _stage_result("REGISTRY_ACCEPTED"), cryptographic_signature_verification(envelope, _stage_result("REGISTRY_ACCEPTED")), _stage_result("EXPIRY_REVOCATION_NOT_REACHED"))["stage_result"] != "SNAPSHOT_CREATED"
    for target, field, reason in [("capability_payload", "schema_version", "UNKNOWN_CAPABILITY_PAYLOAD_SCHEMA_VERSION"), ("signature_header", "signature_schema_version", "UNKNOWN_SIGNATURE_SCHEMA_VERSION")]:
        bad = copy.deepcopy(envelope); bad[target][field] = "unknown.version"
        result = _validate_parsed_document_structure(bad)
        assert result["stage_result"] == "STRUCTURE_REJECTED" and result["diagnostic_reason_code"] == reason
    bad = copy.deepcopy(envelope); bad["capability_payload"]["schema_registry"] = "self supplied"
    assert _validate_parsed_document_structure(bad)["stage_result"] == "STRUCTURE_REJECTED"

def test_dialog_graph_determinism_and_exactly_once_audits():
    flow = DATA["live_activation_dialog_flow"]
    validate_graph(flow)
    outcome, audits, props = run_path(["CONTINUE", "SUCCEEDED"])
    assert outcome == "LIVE_BLOCKED_BY_EDITION"
    assert audits.count("LIVE_ACTIVATION_AUTH_SUCCEEDED") == 1
    assert audits.count("LIVE_ACTIVATION_BLOCKED_BY_EDITION") == 1
    assert all(step.get("sets_active_environment_live") is not True for step in props)
    assert run_path(["CANCEL"])[0] == "CANCELLED"
    assert run_path(["CONTINUE", "FAILED"])[0] == "AUTHENTICATION_FAILED"
    for events in [["UNKNOWN"], [], ["CANCEL", "EXTRA"]]:
        try: run_path(events); raise AssertionError(events)
        except ValueError: pass
    for mutate in ["action_two", "decision_duplicate", "duplicate_terminal", "missing_terminal"]:
        bad = copy.deepcopy(flow)
        if mutate == "action_two": bad["steps"][0]["transitions"].append({"event": "X", "destination_step_id": "desktop_shell_opens_activation_dialog"})
        elif mutate == "decision_duplicate": bad["steps"][2]["transitions"][1]["event"] = "CANCEL"
        elif mutate == "duplicate_terminal": bad["steps"][6]["terminal_outcome"] = "CANCELLED"
        elif mutate == "missing_terminal": bad["terminal_outcomes"].remove("CANCELLED")
        try: validate_graph(bad); raise AssertionError(mutate)
        except AssertionError: pass


def test_gate_mapping_no_bypass_and_defensive_validation():
    assert DATA["gate_execution_invariants"] == ["NO_OR_BYPASS_BETWEEN_GATES", "ALL_APPLICABLE_GATES_REQUIRED", "EARLIER_DENIAL_FORBIDS_LATER_SIDE_EFFECTS", "DIRECT_LAYER_INVOCATION_REVALIDATES_OWN_GATE", "LATER_LAYERS_DO_NOT_ASSUME_EARLIER_VALIDATION", "DENIED_ENVIRONMENT_NEVER_COMMITS_ACTIVE_ENVIRONMENT", "CURRENT_EDITION_LIVE_DENIED_AT_EVERY_DEFENSIVE_GATE"]
    all_gates = gates()
    assert set(GATE_AUDIT) == set(REQ_GATES) == set(all_gates) == set(DATA["live_gate_audit_event_map"])
    assert len(all_gates) == 18
    for gate_id, gate in all_gates.items():
        assert gate["current_edition_result"] == "DENY_LIVE"
        assert gate["denial_code"] in DATA["denial_codes"]
        assert gate["required_audit_event"] == GATE_AUDIT[gate_id]
        assert gate["required_audit_event"] in DATA["audit_events"]
        assert "set_active_environment_live" in gate["side_effects_forbidden_on_denial"]
        assert set(SPECIFIC[gate_id]) <= set(gate["gate_specific_required_inputs"])
        assert gate["direct_invocation_revalidates_own_gate"] is True
        assert gate["may_skip_own_validation_due_to_previous_gate"] is False


def test_preserved_environment_capability_trust_endpoint_and_cross_contracts():
    assert DATA["trust_state_outcomes"]["VALID"]["live_execution_decision"] == "DENY"
    for state, outcome in DATA["trust_state_outcomes"].items():
        if state != "VALID":
            assert outcome["fallback_policy"] == "SAFE_LOCAL_ONLY"
            assert outcome["private_testnet_execution_decision"] == "DENY"
    assert DATA["endpoint_policy"]["testnet_to_live_fallback_allowed"] is False
    assert DATA["credential_policy"]["current_edition_allowed_scopes"] == ["TESTNET"]
    assert DATA["credential_policy"]["current_edition_forbidden_read_or_decrypt_scopes"] == ["LIVE"]
    assert any(entity["canonical_name"] == "LiveAccessGrant" for entity in json.loads(VOCAB.read_text())["entity_kinds"])
    assert DATA["cross_contract_references"]["active_environment_mutation_authority"] == "CoreHost"
    assert any("only CoreHost applies mutable trading state mutations" in invariant for invariant in json.loads(PROC.read_text())["invariants"])
    for group in [DATA["audit_payload_contract"]["required_field_ids"], DATA["audit_payload_contract"]["optional_fields"], DATA["audit_payload_contract"]["result_fields"], DATA["audit_payload_contract"]["forbidden_fields"], DATA["persistence_and_recovery"]["persistable_fields"]]:
        for identifier in group:
            assert re.fullmatch(r"^[a-z][a-z0-9_]*$", identifier), identifier
    assert re.search(r"AKIA[0-9A-Z]{16}|BEGIN .*PRIVATE KEY|sk-[A-Za-z0-9]{20,}", json.dumps(DATA)) is None


def _document_fingerprint(envelope: dict) -> str:
    return sha_bytes(jcs_current_schema_canonicalize(envelope))


def _context_from_envelope(envelope: dict) -> dict:
    payload = envelope["capability_payload"]
    header = envelope["signature_header"]
    fingerprint = _document_fingerprint(envelope)
    return {
        "validation_context_id": sha_bytes(("cryptohunter.validation_context.v1:" + fingerprint).encode()),
        "document_fingerprint": fingerprint,
        "capabilities_id": payload["capabilities_id"],
        "edition_id": payload["edition_id"],
        "signed_payload_hash": envelope["signed_payload_hash"],
        "capability_set_hash": payload["capability_set_hash"],
        "payload_schema_version": payload["schema_version"],
        "signature_schema_version": header["signature_schema_version"],
    }


def _evidence_id(evidence: dict) -> str:
    fields = DATA["stage_evidence_chain_contract"]["evidence_id_fields"]
    return sha_bytes(jcs_current_schema_canonicalize({field: evidence.get(field) for field in fields}))


def _base_stage_evidence(result_code: str, reason: str | None = None, trust: str | None = None, denial: str | None = None, context: dict | None = None, predecessor_stage_evidence_id: str | None = None, fixture_availability: str | None = None) -> dict:
    evidence = {
        "evidence_schema_version": DATA["stage_evidence_chain_contract"]["evidence_schema_version"],
        "stage_evidence_id": None,
        "predecessor_stage_evidence_id": predecessor_stage_evidence_id,
        "stage_id": STAGE_BY_RESULT.get(result_code),
        "stage_result": result_code,
        "validation_context_id": None,
        "document_fingerprint": None,
        "capabilities_id": None,
        "edition_id": None,
        "signed_payload_hash": None,
        "capability_set_hash": None,
        "payload_schema_version": None,
        "signature_schema_version": None,
        "fixture_availability": fixture_availability,
        "diagnostic_reason_code": reason,
        "mapped_trust_state": trust,
        "mapped_denial_code": denial,
    }
    if context:
        for key in DATA["validation_context_contract"]["context_fields"]:
            evidence[key] = context.get(key)
    evidence["stage_evidence_id"] = _evidence_id(evidence)
    return evidence


def _stage_result(result_code: str, reason: str | None = None, trust: str | None = None, denial: str | None = None, context: dict | None = None, predecessor_stage_evidence_id: str | None = None, fixture_token: object | None = None) -> dict:
    if result_code in _ACCEPTED_RESULTS:
        return _base_stage_evidence(_ACCEPTED_FAIL_CLOSED[result_code], "ACCEPTED_RESULT_REQUIRES_STAGE_ISSUER", trust or "UNSUPPORTED_SCHEMA", denial or "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA", context, predecessor_stage_evidence_id)
    return _base_stage_evidence(result_code, reason, trust, denial, context, predecessor_stage_evidence_id)


def _issued_chain_valid(evidence, context: dict | None = None) -> bool:
    return False


def _complete_clean_evidence(evidence: Mapping, expected_stage_id: str, expected_result: str, context: dict, predecessor=None) -> bool:
    return False



def _same_context(evidence: Mapping, context: dict) -> bool:
    return isinstance(evidence, Mapping) and all(evidence.get(key) == context.get(key) for key in DATA["validation_context_contract"]["context_fields"])


def _evidence_id_is_valid(evidence: Mapping) -> bool:
    return isinstance(evidence, Mapping) and isinstance(evidence.get("stage_evidence_id"), str) and HEX64.fullmatch(evidence["stage_evidence_id"]) and evidence["stage_evidence_id"] == _evidence_id(dict(evidence))


def _issued_chain_valid(evidence, context: dict | None = None) -> bool:
    return False


def _complete_clean_evidence(evidence: Mapping, expected_stage_id: str, expected_result: str, context: dict, predecessor=None) -> bool:
    return False



def _current_edition_policy_consistent(envelope: dict) -> bool:
    try:
        payload = envelope["capability_payload"]
        edition_policy = DATA["current_edition_signed_payload_policy"]
        pc_policy = DATA["ProductCapabilities"]["current_edition_capability_policy"]
        return (
            payload["edition_id"] == edition_policy["edition_id"]
            and payload["capability_set"] == edition_policy["capability_set"]
            and payload["environment_capabilities"] == pc_policy["environment_capabilities"]
            and payload["feature_flags"] == pc_policy["feature_flags"]
            and payload["fail_closed_policy"] == edition_policy["fail_closed_policy"]
            and payload["source"] == edition_policy["source"]
            and payload["schema_version"] == PAYLOAD_VERSION
            and envelope["signature_header"]["signature_schema_version"] == SIG_VERSION
        )
    except Exception:
        return False


def _canonical_hashes_revalidated(envelope: dict) -> bool:
    try:
        payload = envelope["capability_payload"]
        return payload["capability_set_hash"] == capability_set_hash(payload["capability_set"]) and envelope["signed_payload_hash"] == sha_bytes(jcs_current_schema_canonicalize(payload))
    except Exception:
        return False


def validate_field(value: object, contract: dict) -> bool:
    typ = contract["type"]
    if typ == "STRING":
        if not isinstance(value, str): return False
        if contract.get("non_empty") and value == "": return False
        if contract.get("format") == "LOWERCASE_HEX" and not HEX64.fullmatch(value): return False
        if "exact_length" in contract and len(value) != contract["exact_length"]: return False
        if "allowed_values" in contract and value not in contract["allowed_values"]: return False
        if "timestamp_profile_ref" in contract:
            try: parse_ts(value)
            except ValueError: return False
        if contract.get("format") == "BASE64URL_NO_PADDING" and not validate_base64url(value): return False
        return True
    if typ == "STRING_OR_NULL":
        if value is None: return True
        if not isinstance(value, str): return False
        if contract.get("non_empty_when_string") and value == "": return False
        if contract.get("format") == "CAPABILITY_REVOCATION_REFERENCE_OR_NULL" and not re.fullmatch(r"caprev_[A-Za-z0-9_-]{8,64}", value): return False
        return True
    if typ == "BOOLEAN": return isinstance(value, bool)
    if typ == "ARRAY":
        if not isinstance(value, list) or (contract.get("non_empty") and not value): return False
        if contract.get("item_type") == "STRING" and not all(isinstance(item, str) for item in value): return False
        if contract.get("unique") and len(set(value)) != len(value): return False
        if contract.get("canonical_sort_order_ref") == "UTF16_CODE_UNIT_LEXICOGRAPHIC" and value != sorted(value, key=utf16_code_unit_sort_key): return False
        return True
    if typ == "OBJECT":
        if not isinstance(value, dict): return False
        keys = contract.get("required_environment_keys")
        if keys and set(value) != set(keys): return False
        if contract.get("key_type") == "STRING" and not all(isinstance(k, str) for k in value): return False
        if contract.get("value_type") == "BOOLEAN" and not all(isinstance(v, bool) for v in value.values()): return False
        return True
    return False



def validate_nested_environment_capabilities(value: object) -> bool:
    if not isinstance(value, dict): return False
    schemas = DATA["ProductCapabilities"]["document_schemas"]
    refs = schemas["capability_payload_schema"]["nested_environment_capability_schema_refs"]
    if set(value) != set(refs): return False
    for env, schema_ref in refs.items():
        schema = schemas[schema_ref]
        if not isinstance(value[env], dict) or set(value[env]) != set(schema["required_fields"]): return False
        for field, contract in schema["field_contracts"].items():
            if not validate_field(value[env][field], contract): return False
            if "current_edition_value" in contract and value[env][field] != contract["current_edition_value"]: return False
    return True


def validate_feature_flags(value: object) -> bool:
    registry = DATA["ProductCapabilities"]["document_schemas"]["current_schema_feature_flag_registry"]
    return isinstance(value, dict) and set(value) == set(registry["required_flags"]) and all(isinstance(v, bool) for v in value.values()) and all(value[k] == registry["current_edition_values"][k] for k in registry["required_flags"])


def parse_raw_json_without_duplicates(raw: str | bytes):
    def no_duplicates(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate object member: {key}")
            out[key] = value
        return out
    if not isinstance(raw, (str, bytes)):
        raise ValueError("raw document type not supported")
    if isinstance(raw, bytes):
        try: raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc: raise ValueError("malformed utf-8") from exc
    allowed_ws = {" ", "\t", "\n", "\r"}
    start = 0
    while start < len(raw) and raw[start] in allowed_ws:
        start += 1
    decoder = json.JSONDecoder(object_pairs_hook=no_duplicates, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    obj, idx = decoder.raw_decode(raw, start)
    if any(ch not in allowed_ws for ch in raw[idx:]):
        raise ValueError("trailing data")
    return obj



def validate_raw_document_structure(raw_document: str | bytes, version_registry=DEFAULT_VERSION_REGISTRY) -> dict:
    if not isinstance(raw_document, (str, bytes)):
        return _stage_result("STRUCTURE_REJECTED", "RAW_DOCUMENT_TYPE_NOT_SUPPORTED", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA")
    try:
        parsed = parse_raw_json_without_duplicates(raw_document)
    except ValueError:
        return _stage_result("STRUCTURE_REJECTED")
    return _validate_parsed_document_structure(parsed, version_registry)

def raw_document(envelope: dict) -> str:
    return jcs_current_schema_canonicalize(envelope).decode("utf-8")


def test_signature_validation_stages_and_registry_mapping_are_data_driven():
    pipeline = DATA["signature_validation_pipeline"]
    assert pipeline["validation_stages"] == ["DOCUMENT_STRUCTURE_VALIDATION", "CANONICAL_HASH_VALIDATION", "REGISTRY_POLICY_VALIDATION", "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION", "EXPIRY_AND_REVOCATION_VALIDATION", "VALIDATED_SNAPSHOT_CREATION"]
    assert "CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4" in pipeline["stage_results"]
    assert pipeline["structural_signature_fixture_may_create_valid_trust_state"] is False
    contract = DATA["signature_contract"]["signature_verification_registry_contract"]
    for reason in contract["diagnostic_reason_codes"]:
        expected = ("UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA") if reason in {"REGISTRY_INPUT_DOCUMENT_INVALID", "REGISTRY_REFERENCE_TYPE_INVALID", "REGISTRY_CONTAINER_INVALID"} else ("INVALID_SIGNATURE", "PRODUCT_CAPABILITIES_INVALID_SIGNATURE")
        assert contract["reason_code_to_trust_state"][reason] == expected[0]
        assert contract["reason_code_to_denial_code"][reason] == expected[1]
    env = build_envelope(_payload())
    accepted = validate_registry_policy(env, validate_canonical_hashes(env, _validate_parsed_document_structure(env)), {"alg"}, {"key"})
    assert accepted["stage_result"] == "REGISTRY_ACCEPTED"
    assert accepted["mapped_trust_state"] is None
    for algs, keys, reason in [
        (set(), {"key"}, "MISSING_SIGNATURE_ALGORITHM_REGISTRY"),
        ({"alg"}, set(), "MISSING_VERIFICATION_KEY_REGISTRY"),
        ({"other"}, {"key"}, "UNKNOWN_SIGNATURE_ALGORITHM"),
        ({"alg"}, {"other"}, "UNKNOWN_KEY_ID"),
    ]:
        outcome = validate_registry_policy(env, validate_canonical_hashes(env, _validate_parsed_document_structure(env)), algs, keys)
        assert outcome["stage_result"] == "REGISTRY_REJECTED"
        assert outcome["diagnostic_reason_code"] == reason
        assert outcome["mapped_trust_state"] == "INVALID_SIGNATURE"
        assert outcome["mapped_denial_code"] == "PRODUCT_CAPABILITIES_INVALID_SIGNATURE"


def test_utf16_capability_set_order_and_fixed_hash_regression():
    assert DATA["signature_contract"]["capability_set_sort_order_id"] == "UTF16_CODE_UNIT_LEXICOGRAPHIC"
    capability_set = ["\U00010000", "\ue000"]
    assert sorted(["\ue000", "\U00010000"], key=utf16_code_unit_sort_key) == capability_set
    assert sorted(["\ue000", "\U00010000"]) != capability_set
    assert capability_set_hash(capability_set) == "e8bdee294d4a756532cd1660a49d7d99325bb04ec58c236f78b94ff2718d31de"


def test_field_contracts_are_complete_and_malformed_types_rejected():
    schemas = DATA["ProductCapabilities"]["document_schemas"]
    assert set(schemas["capability_payload_schema"]["field_contracts"]) == set(CANONICAL_PAYLOAD)
    assert set(schemas["signature_header_schema"]["field_contracts"]) == set(HEADER)
    assert set(schemas["signed_document_envelope_schema"]["field_contracts"]) == set(ENVELOPE)
    assert set(schemas["signature_input_schema"]["field_contracts"]) == {"signature_input", "signature_header", "signed_payload_hash"}
    valid = build_envelope(_payload())
    assert _validate_parsed_document_structure(valid)["stage_result"] == "STRUCTURE_ACCEPTED"
    mutations = {
        "capabilities_id_array": lambda e: e["capability_payload"].__setitem__("capabilities_id", []),
        "edition_id_null": lambda e: e["capability_payload"].__setitem__("edition_id", None),
        "issued_bool": lambda e: e["capability_payload"].__setitem__("issued_at_utc", True),
        "non_expiring_string": lambda e: e["capability_payload"].__setitem__("non_expiring", "true"),
        "capability_set_string": lambda e: e["capability_payload"].__setitem__("capability_set", "A"),
        "capability_set_non_string": lambda e: e["capability_payload"].__setitem__("capability_set", ["A", 1]),
        "env_caps_null": lambda e: e["capability_payload"].__setitem__("environment_capabilities", None),
        "env_caps_array": lambda e: e["capability_payload"].__setitem__("environment_capabilities", []),
        "missing_environment": lambda e: e["capability_payload"].__setitem__("environment_capabilities", {"PAPER": {}, "TESTNET": {}}),
        "extra_environment": lambda e: e["capability_payload"].__setitem__("environment_capabilities", {"PAPER": {}, "TESTNET": {}, "LIVE": {}, "DEMO": {}}),
        "feature_flags_array": lambda e: e["capability_payload"].__setitem__("feature_flags", []),
        "feature_flag_string": lambda e: e["capability_payload"].__setitem__("feature_flags", {"x": "true"}),
        "hash_non_string": lambda e: e["capability_payload"].__setitem__("capability_set_hash", 1),
        "source_empty": lambda e: e["capability_payload"].__setitem__("source", ""),
        "fail_closed_other": lambda e: e["capability_payload"].__setitem__("fail_closed_policy", "OPEN"),
        "algorithm_array": lambda e: e["signature_header"].__setitem__("signature_algorithm_id", []),
        "key_null": lambda e: e["signature_header"].__setitem__("key_id", None),
        "signed_hash_non_string": lambda e: e.__setitem__("signed_payload_hash", 1),
        "signature_non_string": lambda e: e.__setitem__("signature", 1),
    }
    for name, mutate in mutations.items():
        bad = copy.deepcopy(valid); mutate(bad)
        assert _validate_parsed_document_structure(bad)["stage_result"] == "STRUCTURE_REJECTED", name


def test_raw_json_parse_policy_rejects_duplicates_and_trailing_data():
    policy = DATA["raw_json_parse_policy"]
    assert policy["duplicate_object_member_names"] == "REJECT_BEFORE_CANONICALIZATION"
    cases = [
        '{"capability_payload":{"edition_id":"a","edition_id":"b"}}',
        '{"capability_payload":{"environment_capabilities":{"LIVE":{},"LIVE":{}}}}',
        '{"signature_header":{"key_id":"a","key_id":"b"}}',
        '{"signed_payload_hash":"a","signed_payload_hash":"b"}',
        '{"capability_payload":{"feature_flags":{"x":true,"x":false}}}',
        '{} {}',
    ]
    for raw in cases:
        try: parse_raw_json_without_duplicates(raw); raise AssertionError(raw)
        except ValueError: pass


def test_execution_environment_matrix_endpoint_trust_safe_local_and_recovery_contracts():
    envs = {e["environment_id"]: e for e in DATA["execution_environments"]}
    assert set(envs) == {"PAPER", "TESTNET", "LIVE"}
    assert not ({"DEMO", "PREVIEW", "SIMULATION", "SANDBOX"} & set(envs))
    edition = DATA["current_product_edition"]["environment_capabilities"]
    assert edition["PAPER"] == {"visible": True, "selectable": True, "executable": True, "locked": False}
    assert edition["TESTNET"]["visible"] and edition["TESTNET"]["selectable"] and edition["TESTNET"]["executable"] and edition["TESTNET"]["requires_readiness_gates"] and not edition["TESTNET"]["locked"]
    assert edition["LIVE"]["visible"] and edition["LIVE"]["locked"] and not edition["LIVE"]["selectable"] and not edition["LIVE"]["executable"] and edition["LIVE"]["activation_dialog_available"] and edition["LIVE"]["denial_code"] == "LIVE_BLOCKED_BY_EDITION"
    pc_policy = DATA["ProductCapabilities"]["current_edition_capability_policy"]
    assert pc_policy["live_allowed_in_current_edition"] is False and pc_policy["environment_capabilities"]["LIVE"]["private_execution_allowed"] is False
    assert envs["PAPER"]["endpoint_classes"] == ["LOCAL_SIMULATION"]
    assert envs["TESTNET"]["endpoint_classes"] == ["PUBLIC_TESTNET", "PRIVATE_TESTNET"]
    assert envs["LIVE"]["endpoint_classes"] == ["PUBLIC_LIVE", "PRIVATE_LIVE"]
    assert all(cls in DATA["endpoint_policy"]["endpoint_classes"] for env in envs.values() for cls in env["endpoint_classes"])
    assert DATA["endpoint_policy"]["testnet_to_live_fallback_allowed"] is False
    expected_trust = ["VALID", "MISSING", "INVALID_SIGNATURE", "EXPIRED", "UNSUPPORTED_SCHEMA", "REVOKED"]
    assert DATA["ProductCapabilities"]["trust_state_registry"] == expected_trust
    keys = set(DATA["trust_state_outcomes"]["VALID"])
    for state, outcome in DATA["trust_state_outcomes"].items():
        assert set(outcome) == keys and outcome["live_execution_decision"] == "DENY"
        if state != "VALID": assert outcome["fallback_policy"] == "SAFE_LOCAL_ONLY"
    safe = DATA["fail_closed_fallbacks"]["SAFE_LOCAL_ONLY"]
    assert safe["read_or_decrypt_exchange_secrets"] is False and safe["testnet_private_execution_allowed"] is False and safe["live_allowed"] is False and safe["paper_local_simulation_may_remain_available"] is True
    rules = DATA["runtime_environment_activation_gate_order"]["rules"]
    assert any("requested_environment enum validation precedes edition" in rule for rule in rules)
    assert any("before credential read" in rule for rule in rules)
    assert "core_commits_active_environment" in DATA["runtime_environment_activation_gate_order"]["order"]
    assert DATA["cross_contract_references"]["DesktopShell_runtime_mutation_allowed"] is False
    recovery_rules = "\n".join(DATA["persistence_and_recovery"]["rules"])
    for text in ["persisted LIVE", "last_successful_environment", "Live reconnect", "kill switch"]:
        assert text in recovery_rules


def test_audit_denial_registries_unique_and_full_safety_preserved():
    assert len(DATA["denial_codes"]) == len(set(DATA["denial_codes"]))
    assert len(DATA["audit_events"]) == len(set(DATA["audit_events"]))
    for gate in DATA["live_enforcement_gates"]:
        assert gate["denial_code"] in DATA["denial_codes"]
        assert gate["required_audit_event"] in DATA["audit_events"]
    for identifier_group in [DATA["audit_payload_contract"]["required_field_ids"], DATA["audit_payload_contract"]["optional_fields"], DATA["audit_payload_contract"]["result_fields"]]:
        for identifier in identifier_group:
            assert re.fullmatch(r"^[a-z][a-z0-9_]*$", identifier)
    for forbidden in ["secret", "pin", "password", "biometric_template"]:
        assert forbidden in DATA["audit_payload_contract"]["forbidden_fields"]
    test_gate_mapping_no_bypass_and_defensive_validation()
    test_dialog_graph_determinism_and_exactly_once_audits()


def test_closed_environment_capability_schemas_and_feature_flag_registry():
    valid = build_envelope(_payload())
    env_mutations = {
        "empty_paper": lambda e: e["capability_payload"]["environment_capabilities"].__setitem__("PAPER", {}),
        "empty_testnet": lambda e: e["capability_payload"]["environment_capabilities"].__setitem__("TESTNET", {}),
        "empty_live": lambda e: e["capability_payload"]["environment_capabilities"].__setitem__("LIVE", {}),
        "missing_nested": lambda e: e["capability_payload"]["environment_capabilities"]["LIVE"].pop("private_execution_allowed"),
        "extra_nested": lambda e: e["capability_payload"]["environment_capabilities"]["LIVE"].__setitem__("unknown_live_unlock", True),
        "string_boolean": lambda e: e["capability_payload"]["environment_capabilities"]["PAPER"].__setitem__("local_execution_allowed", "true"),
        "live_private_true": lambda e: e["capability_payload"]["environment_capabilities"]["LIVE"].__setitem__("private_execution_allowed", True),
    }
    for name, mutate in env_mutations.items():
        bad = copy.deepcopy(valid); mutate(bad)
        assert _validate_parsed_document_structure(bad)["stage_result"] == "STRUCTURE_REJECTED", name
    registry = DATA["ProductCapabilities"]["document_schemas"]["current_schema_feature_flag_registry"]
    assert registry["required_flags"] == ["live_activation_dialog", "live_execution", "testnet_execution"]
    assert registry["current_edition_values"] == {"live_activation_dialog": True, "live_execution": False, "testnet_execution": True}
    flag_mutations = {
        "fictional_x": lambda e: e["capability_payload"].__setitem__("feature_flags", {"x": True}),
        "unknown_live_unlock": lambda e: e["capability_payload"]["feature_flags"].__setitem__("unknown_live_unlock", True),
        "live_execution_true": lambda e: e["capability_payload"]["feature_flags"].__setitem__("live_execution", True),
        "missing_live_execution": lambda e: e["capability_payload"]["feature_flags"].pop("live_execution"),
        "string_boolean": lambda e: e["capability_payload"]["feature_flags"].__setitem__("testnet_execution", "true"),
    }
    for name, mutate in flag_mutations.items():
        bad = copy.deepcopy(valid); mutate(bad)
        assert _validate_parsed_document_structure(bad)["stage_result"] == "STRUCTURE_REJECTED", name


def test_version_policy_and_revocation_interpreter():
    policy = DATA["capability_document_version_policy"]
    assert policy["supported_payload_schema_versions"] == [PAYLOAD_VERSION]
    assert policy["supported_signature_schema_versions"] == [SIG_VERSION]
    assert policy["mapped_trust_state"] == "UNSUPPORTED_SCHEMA"
    assert policy["mapped_denial_code"] == "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA"
    envelope = build_envelope(_payload())
    for target, field in [("capability_payload", "schema_version"), ("signature_header", "signature_schema_version")]:
        for value in ["", "unknown", [], None]:
            bad = copy.deepcopy(envelope); bad[target][field] = value
            assert _validate_parsed_document_structure(bad)["stage_result"] == "STRUCTURE_REJECTED"
    assert _evaluate_expiry_and_revocation_policy(envelope, "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"})["policy_result"] == "EXPIRY_REVOCATION_ACCEPTED"
    revoked = _evaluate_expiry_and_revocation_policy(envelope, "2026-01-01T00:00:00Z", {"caprev_test1234": "REVOKED"})
    assert (revoked["policy_result"], revoked["mapped_trust_state"], revoked["mapped_denial_code"]) == ("EXPIRY_REVOCATION_REJECTED", "REVOKED", "PRODUCT_CAPABILITIES_REVOKED")
    unknown_ref = _evaluate_expiry_and_revocation_policy(envelope, "2026-01-01T00:00:00Z", {"other": "NOT_REVOKED"})
    assert (unknown_ref["policy_result"], unknown_ref["diagnostic_reason_code"]) == ("EXPIRY_REVOCATION_REJECTED", "REVOCATION_STATUS_UNKNOWN")
    unavailable = _evaluate_expiry_and_revocation_policy(envelope, "2026-01-01T00:00:00Z", None)
    assert (unavailable["policy_result"], unavailable["diagnostic_reason_code"]) == ("EXPIRY_REVOCATION_REJECTED", "REVOCATION_REGISTRY_UNAVAILABLE")
    empty_registry = _evaluate_expiry_and_revocation_policy(envelope, "2026-01-01T00:00:00Z", {})
    assert (empty_registry["policy_result"], empty_registry["diagnostic_reason_code"]) == ("EXPIRY_REVOCATION_REJECTED", "REVOCATION_REGISTRY_UNAVAILABLE")
    injected = copy.deepcopy(envelope); injected["capability_payload"]["revocation_status"] = "NOT_REVOKED"
    assert _evaluate_expiry_and_revocation_policy(injected, "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"})["diagnostic_reason_code"] == "DOCUMENT_SUPPLIED_REVOCATION_STATUS_FORBIDDEN"
    expired = build_envelope({**_payload(), "non_expiring": False, "expires_at_utc": "2025-01-01T00:00:01Z"})
    expired_outcome = _evaluate_expiry_and_revocation_policy(expired, "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"})
    assert (expired_outcome["policy_result"], expired_outcome["diagnostic_reason_code"], expired_outcome["mapped_trust_state"], expired_outcome["mapped_denial_code"]) == ("EXPIRY_REVOCATION_REJECTED", "EXPIRED", "EXPIRED", "PRODUCT_CAPABILITIES_EXPIRED")


def test_raw_json_policy_rejects_malformed_utf8_nonfinite_duplicates_and_trailing_data():
    bad_inputs = [
        b"\xff",
        '{"x": NaN}',
        '{"x": Infinity}',
        '{"x": -Infinity}',
        '{} trailing',
        '{"outer":{"dup":1,"dup":2}}',
        '{"capability_payload":{},"capability_payload":{}}',
    ]
    for raw in bad_inputs:
        try:
            parse_raw_json_without_duplicates(raw)
            raise AssertionError(raw)
        except ValueError:
            assert DATA["raw_json_parse_policy"]["parse_error_stage_result"] == "STRUCTURE_REJECTED"
            assert DATA["raw_json_parse_policy"]["canonicalization_after_parse_error_allowed"] is False


def test_pipeline_regression_hash_registry_expiry_revocation_and_snapshot_boundaries():
    envelope = build_envelope(_payload())
    structure = _validate_parsed_document_structure(envelope)
    hashes = validate_canonical_hashes(envelope, structure)
    assert structure["stage_result"] == "STRUCTURE_ACCEPTED" and hashes["stage_result"] == "HASHES_ACCEPTED"
    bad_hash = copy.deepcopy(envelope); bad_hash["signed_payload_hash"] = "0" * 64
    assert _validate_parsed_document_structure(bad_hash)["stage_result"] == "STRUCTURE_ACCEPTED"
    assert validate_canonical_hashes(bad_hash, _validate_parsed_document_structure(bad_hash))["stage_result"] == "HASHES_REJECTED"
    registry = validate_registry_policy(envelope, hashes, {"other"}, {"key"})
    assert registry["stage_result"] == "REGISTRY_REJECTED"
    crypto = cryptographic_signature_verification(envelope, validate_registry_policy(envelope, hashes, {"alg"}, {"key"}))
    assert crypto["stage_result"] == "CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4"
    assert create_validated_snapshot(envelope, structure, hashes, _stage_result("REGISTRY_ACCEPTED"), crypto, _stage_result("EXPIRY_REVOCATION_NOT_REACHED"))["stage_result"] != "SNAPSHOT_CREATED"
    assert DATA["signature_validation_pipeline"]["snapshot_created_reachable_in_m0_4"] is False
    assert DATA["signature_validation_pipeline"]["test_functions_may_return_VALID_trust_state"] is False


def _future_crypto_accepted_evidence(registry_evidence: Mapping) -> dict:
    context = {key: registry_evidence.get(key) for key in DATA["validation_context_contract"]["context_fields"]} if isinstance(registry_evidence, Mapping) else {}
    if not _complete_clean_evidence(registry_evidence, "REGISTRY_POLICY_VALIDATION", "REGISTRY_ACCEPTED", context):
        return _stage_result("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED", "PREVIOUS_STAGE_EVIDENCE_INVALID", "INVALID_SIGNATURE", "PRODUCT_CAPABILITIES_INVALID_SIGNATURE", context if context else None, registry_evidence.get("stage_evidence_id") if isinstance(registry_evidence, Mapping) else None)
    return _EVIDENCE_RUNTIME.future_fixture_validated(context, registry_evidence)


def run_pipeline(raw_document, *, version_registry=DEFAULT_VERSION_REGISTRY, algorithms={"alg"}, keys={"key"}, current_utc="2026-01-01T00:00:00Z", revocation_registry=None):
    structure = validate_raw_document_structure(raw_document, version_registry)
    document = parse_raw_json_without_duplicates(raw_document) if structure.get("stage_result") == "STRUCTURE_ACCEPTED" else None
    hashes = validate_canonical_hashes(document or {}, structure)
    registry = validate_registry_policy(document or {}, hashes, algorithms, keys)
    crypto = cryptographic_signature_verification(document or {}, registry)
    expiry = validate_expiry_revocation_stage(document or {}, crypto, current_utc, revocation_registry)
    snapshot = create_validated_snapshot(document or {}, structure, hashes, registry, crypto, expiry)
    return {"structure": structure, "hashes": hashes, "registry": registry, "crypto": crypto, "expiry_revocation": expiry, "snapshot": snapshot}


def _run_future_contract_pipeline(raw_document, *, version_registry=DEFAULT_VERSION_REGISTRY, algorithms={"alg"}, keys={"key"}, current_utc="2026-01-01T00:00:00Z", revocation_registry=None):
    structure = validate_raw_document_structure(raw_document, version_registry)
    document = parse_raw_json_without_duplicates(raw_document) if structure.get("stage_result") == "STRUCTURE_ACCEPTED" else None
    hashes = validate_canonical_hashes(document or {}, structure)
    registry = validate_registry_policy(document or {}, hashes, algorithms, keys)
    crypto = _future_crypto_accepted_evidence(registry) if registry["stage_result"] == "REGISTRY_ACCEPTED" else cryptographic_signature_verification(document or {}, registry)
    expiry = validate_expiry_revocation_stage(document or {}, crypto, current_utc, revocation_registry)
    snapshot = create_validated_snapshot(document or {}, structure, hashes, registry, crypto, expiry)
    return {"structure": structure, "hashes": hashes, "registry": registry, "crypto": crypto, "expiry_revocation": expiry, "snapshot": snapshot}


def test_raw_document_entrypoint_is_the_only_safe_pipeline_entry():
    envelope = build_envelope(_payload())
    raw = raw_document(envelope)
    assert validate_raw_document_structure(raw)["stage_result"] == "STRUCTURE_ACCEPTED"
    assert validate_raw_document_structure(raw.encode("utf-8"))["stage_result"] == "STRUCTURE_ACCEPTED"
    for bad in [
        f'{{"capability_payload":{{"edition_id":"a","edition_id":"b"}},"signature_header":{{}},"signed_payload_hash":"{"a"*64}","signature":"YWJj"}}',
        '{"capability_payload":{"feature_flags":{"x":true,"x":false}}}',
        '{"signed_payload_hash":"a","signed_payload_hash":"b"}',
        b"\xff",
        '{"x": NaN}',
        '{"x": Infinity}',
        '{"x": -Infinity}',
        '{} {}',
    ]:
        assert validate_raw_document_structure(bad)["stage_result"] == "STRUCTURE_REJECTED"
    parsed_by_last_key_wins = json.loads('{"signed_payload_hash":"a","signed_payload_hash":"b"}')
    assert validate_raw_document_structure(parsed_by_last_key_wins)["stage_result"] == "STRUCTURE_REJECTED"


def test_stage_result_registry_is_closed_and_all_outputs_registered():
    registry = DATA["signature_validation_pipeline"]["stage_result_registry"]
    required = {"STRUCTURE_ACCEPTED", "STRUCTURE_REJECTED", "HASHES_ACCEPTED", "HASHES_REJECTED", "HASH_VALIDATION_NOT_REACHED", "REGISTRY_ACCEPTED", "REGISTRY_REJECTED", "REGISTRY_VALIDATION_NOT_REACHED", "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED", "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED", "CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4", "CRYPTOGRAPHIC_VERIFICATION_NOT_REACHED", "EXPIRY_REVOCATION_ACCEPTED", "EXPIRY_REVOCATION_REJECTED", "EXPIRY_REVOCATION_NOT_REACHED", "SNAPSHOT_CREATED", "SNAPSHOT_NOT_CREATED"}
    assert set(registry) == required
    for outputs in DATA["signature_validation_pipeline"]["stage_outputs"].values():
        assert set(outputs) <= set(registry)
    source = Path(__file__).read_text()
    returned_literals = set(re.findall(r'stage_result\("([A-Z0-9_]+)"', source)) | set(re.findall(r'"stage_result": "([A-Z0-9_]+)"', source))
    assert returned_literals <= set(registry)


def test_stage_prerequisites_not_reached_and_snapshot_prerequisites():
    envelope = build_envelope(_payload())
    assert validate_canonical_hashes(envelope, _stage_result("STRUCTURE_REJECTED"))["stage_result"] == "HASH_VALIDATION_NOT_REACHED"
    assert validate_registry_policy(envelope, _stage_result("HASHES_REJECTED"), {"alg"}, {"key"})["stage_result"] == "REGISTRY_VALIDATION_NOT_REACHED"
    assert cryptographic_signature_verification(envelope, _stage_result("REGISTRY_REJECTED"))["stage_result"] == "CRYPTOGRAPHIC_VERIFICATION_NOT_REACHED"
    assert validate_expiry_revocation_stage(envelope, _stage_result("CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4"), "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"})["stage_result"] == "EXPIRY_REVOCATION_NOT_REACHED"
    structure, hashes, registry, crypto, accepted = _accepted_future_evidence_chain(envelope)
    rejected = validate_expiry_revocation_stage(envelope, crypto, "2026-01-01T00:00:00Z", {"caprev_test1234": "REVOKED"})
    assert create_validated_snapshot(envelope, structure, hashes, registry, crypto, accepted)["stage_result"] == "SNAPSHOT_CREATED"
    for combo in [
        (structure, hashes, registry, crypto, rejected),
        (structure, _stage_result("HASHES_REJECTED", context=_context_from_envelope(envelope), predecessor_stage_evidence_id=structure["stage_evidence_id"]), registry, crypto, accepted),
        (structure, hashes, _stage_result("REGISTRY_REJECTED", context=_context_from_envelope(envelope), predecessor_stage_evidence_id=hashes["stage_evidence_id"]), crypto, accepted),
        (structure, hashes, registry, cryptographic_signature_verification(envelope, registry), accepted),
    ]:
        assert create_validated_snapshot(envelope, *combo)["stage_result"] == "SNAPSHOT_NOT_CREATED"


def test_revocation_reference_format_and_invalid_registry_status_fail_closed():
    envelope = build_envelope(_payload())
    for value in [None, "caprev_12345678", "caprev_ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789-abcd"]:
        good = copy.deepcopy(envelope); good["capability_payload"]["revocation_reference"] = value; good = build_envelope(good["capability_payload"])
        assert _validate_parsed_document_structure(good)["stage_result"] == "STRUCTURE_ACCEPTED"
    for value in ["", "bad_12345678", "caprev_short", "caprev_" + "a" * 65, "caprev_bad space", 1, True, {}]:
        bad = copy.deepcopy(envelope); bad["capability_payload"]["revocation_reference"] = value
        assert _validate_parsed_document_structure(bad)["stage_result"] == "STRUCTURE_REJECTED"
    for status in ["BROKEN", None, True, {}, []]:
        outcome = _evaluate_expiry_and_revocation_policy(envelope, "2026-01-01T00:00:00Z", {"caprev_test1234": status})
        assert outcome["policy_result"] == "EXPIRY_REVOCATION_REJECTED"
        assert outcome["diagnostic_reason_code"] == "INVALID_REVOCATION_REGISTRY_STATUS"
        assert outcome["mapped_trust_state"] == "INVALID_SIGNATURE"
        assert outcome["mapped_denial_code"] == "PRODUCT_CAPABILITIES_INVALID_SIGNATURE"
    assert _evaluate_expiry_and_revocation_policy(envelope, "2026-01-01T00:00:00Z", {"other": "NOT_REVOKED"})["diagnostic_reason_code"] == "REVOCATION_STATUS_UNKNOWN"
    assert DATA["capability_revocation_policy"]["status_mappings"]["NOT_REVOKED"].get("audit_event") == "PRODUCT_CAPABILITIES_REVOCATION_CHECK_PASSED"
    assert DATA["capability_revocation_policy"]["status_mappings"]["NOT_REVOKED"].get("audit_event") != "PRODUCT_CAPABILITIES_VALIDATED"


def test_missing_version_registry_and_state_machine_paths():
    envelope = build_envelope(_payload())
    raw = raw_document(envelope)
    assert validate_raw_document_structure(raw, DEFAULT_VERSION_REGISTRY)["stage_result"] == "STRUCTURE_ACCEPTED"
    assert validate_raw_document_structure(raw, {"payload": {"unknown"}, "signature": {SIG_VERSION}})["diagnostic_reason_code"] == "UNKNOWN_CAPABILITY_PAYLOAD_SCHEMA_VERSION"
    assert validate_raw_document_structure(raw, None) == _stage_result("STRUCTURE_REJECTED", "MISSING_SCHEMA_VERSION_REGISTRY", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA")
    assert validate_raw_document_structure(raw, {"payload": set(), "signature": set()})["diagnostic_reason_code"] == "MISSING_SCHEMA_VERSION_REGISTRY"
    override = copy.deepcopy(envelope); override["capability_payload"]["schema_registry"] = {"payload": [PAYLOAD_VERSION]}
    assert _validate_parsed_document_structure(override)["diagnostic_reason_code"] == "PAYLOAD_SCHEMA_REGISTRY_OVERRIDE_FORBIDDEN"
    happy = run_pipeline(raw, revocation_registry={"caprev_test1234": "NOT_REVOKED"})
    assert [happy[k]["stage_result"] for k in ["structure", "hashes", "registry", "crypto", "expiry_revocation", "snapshot"]] == ["STRUCTURE_ACCEPTED", "HASHES_ACCEPTED", "REGISTRY_ACCEPTED", "CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4", "EXPIRY_REVOCATION_NOT_REACHED", "SNAPSHOT_NOT_CREATED"]
    bad_raw = run_pipeline('{"x": NaN}')
    assert [bad_raw[k]["stage_result"] for k in ["structure", "hashes", "registry", "crypto", "expiry_revocation", "snapshot"]] == ["STRUCTURE_REJECTED", "HASH_VALIDATION_NOT_REACHED", "REGISTRY_VALIDATION_NOT_REACHED", "CRYPTOGRAPHIC_VERIFICATION_NOT_REACHED", "EXPIRY_REVOCATION_NOT_REACHED", "SNAPSHOT_NOT_CREATED"]
    bad_hash_doc = copy.deepcopy(envelope); bad_hash_doc["signed_payload_hash"] = "0" * 64
    bad_hash = run_pipeline(raw_document(bad_hash_doc))
    assert [bad_hash[k]["stage_result"] for k in ["structure", "hashes", "registry", "crypto", "expiry_revocation", "snapshot"]] == ["STRUCTURE_ACCEPTED", "HASHES_REJECTED", "REGISTRY_VALIDATION_NOT_REACHED", "CRYPTOGRAPHIC_VERIFICATION_NOT_REACHED", "EXPIRY_REVOCATION_NOT_REACHED", "SNAPSHOT_NOT_CREATED"]
    unknown_alg = run_pipeline(raw, algorithms={"other"})
    assert unknown_alg["registry"]["stage_result"] == "REGISTRY_REJECTED" and unknown_alg["crypto"]["stage_result"] == "CRYPTOGRAPHIC_VERIFICATION_NOT_REACHED"
    future_ok = _run_future_contract_pipeline(raw, revocation_registry={"caprev_test1234": "NOT_REVOKED"})
    assert future_ok["snapshot"]["stage_result"] == "SNAPSHOT_CREATED"
    future_revoked = _run_future_contract_pipeline(raw, revocation_registry={"caprev_test1234": "REVOKED"})
    assert future_revoked["expiry_revocation"]["stage_result"] == "EXPIRY_REVOCATION_REJECTED" and future_revoked["snapshot"]["stage_result"] == "SNAPSHOT_NOT_CREATED"


def _stage_outputs(stage_id: str) -> set[str]:
    return set(DATA["signature_validation_pipeline"]["stage_outputs"][stage_id])


def test_raw_entrypoint_rejects_every_non_raw_type_with_stable_reason():
    class CustomObject:
        pass

    unsupported_inputs = [
        {}, [], (), 1, 1.5, True, None, bytearray(b"{}"), memoryview(b"{}"), {"x"}, CustomObject()
    ]
    for raw_document in unsupported_inputs:
        result = validate_raw_document_structure(raw_document)
        assert result == _stage_result(
            "STRUCTURE_REJECTED",
            "RAW_DOCUMENT_TYPE_NOT_SUPPORTED",
            "UNSUPPORTED_SCHEMA",
            "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA",
        )
        try:
            parse_raw_json_without_duplicates(raw_document)
            raise AssertionError(type(raw_document).__name__)
        except ValueError:
            pass
    policy = DATA["raw_json_parse_policy"]
    assert policy["accepted_raw_document_types"] == ["str", "bytes"]
    assert policy["unsupported_type_result"] == "STRUCTURE_REJECTED"
    assert policy["unsupported_type_diagnostic_reason"] == "RAW_DOCUMENT_TYPE_NOT_SUPPORTED"


def test_no_public_dict_bypass_for_structure_validation_pipeline():
    pipeline = DATA["signature_validation_pipeline"]
    assert pipeline["raw_document_entrypoint"] == "validate_raw_document_structure"
    assert pipeline["production_path_accepts_dict_directly"] is False
    assert "data_driven_structure_validation" not in globals()
    public_dict_structure_helpers = [
        name for name, value in globals().items()
        if callable(value)
        and not name.startswith("_")
        and not name.startswith("test_")
        and "structure" in name
        and name != pipeline["raw_document_entrypoint"]
    ]
    assert public_dict_structure_helpers == []


def test_registry_stage_never_returns_structure_stage_result_for_malformed_input():
    malformed_documents = [
        {"capability_payload": {"signature_algorithm_id": "alg"}, "signature_header": {"signature_algorithm_id": "alg", "key_id": "key"}},
        {"capability_payload": {}, "signature_header": None},
        {"capability_payload": []},
        {},
    ]
    issued_hash = validate_canonical_hashes(build_envelope(_payload()), _validate_parsed_document_structure(build_envelope(_payload())))
    for malformed in malformed_documents:
        result = validate_registry_policy(malformed, issued_hash, {"alg"}, {"key"})
        assert result["stage_result"] == "REGISTRY_REJECTED"
        assert result["diagnostic_reason_code"] == "REGISTRY_INPUT_DOCUMENT_INVALID"
        assert result["mapped_trust_state"] == "UNSUPPORTED_SCHEMA"
        assert result["mapped_denial_code"] == "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA"
        assert result["stage_result"] != "STRUCTURE_REJECTED"


def test_each_stage_helper_owns_only_its_declared_stage_outputs():
    envelope = build_envelope(_payload())
    raw = raw_document(envelope)
    accepted_structure = validate_raw_document_structure(raw)
    bad_version_registry = {"payload": {"unknown"}, "signature": {SIG_VERSION}}
    structure_results = [
        accepted_structure,
        validate_raw_document_structure('{"x": NaN}'),
        validate_raw_document_structure({}),
        validate_raw_document_structure(raw, None),
        validate_raw_document_structure(raw, bad_version_registry),
    ]
    assert {result["stage_result"] for result in structure_results} <= _stage_outputs("DOCUMENT_STRUCTURE_VALIDATION")

    accepted_hashes = validate_canonical_hashes(envelope, accepted_structure)
    bad_hash_doc = copy.deepcopy(envelope); bad_hash_doc["signed_payload_hash"] = "0" * 64
    hash_results = [
        accepted_hashes,
        validate_canonical_hashes(bad_hash_doc, validate_raw_document_structure(raw_document(bad_hash_doc))),
        validate_canonical_hashes(envelope, _stage_result("STRUCTURE_REJECTED")),
    ]
    assert {result["stage_result"] for result in hash_results} <= _stage_outputs("CANONICAL_HASH_VALIDATION")

    registry_results = [
        validate_registry_policy(envelope, accepted_hashes, {"alg"}, {"key"}),
        validate_registry_policy(envelope, accepted_hashes, {"other"}, {"key"}),
        validate_registry_policy(envelope, accepted_hashes, {"alg"}, {"other"}),
        validate_registry_policy(envelope, accepted_hashes, None, {"key"}),
        validate_registry_policy({"capability_payload": {"key_id": "payload-key"}, "signature_header": {}}, accepted_hashes, {"alg"}, {"key"}),
        validate_registry_policy(envelope, _stage_result("HASHES_REJECTED"), {"alg"}, {"key"}),
    ]
    assert {result["stage_result"] for result in registry_results} <= _stage_outputs("REGISTRY_POLICY_VALIDATION")
    assert "STRUCTURE_REJECTED" not in {result["stage_result"] for result in registry_results}

    crypto_results = [
        cryptographic_signature_verification(envelope, _stage_result("REGISTRY_ACCEPTED")),
        cryptographic_signature_verification(envelope, _stage_result("REGISTRY_REJECTED")),
        _stage_result("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED"),
        _stage_result("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED"),
    ]
    assert {result["stage_result"] for result in crypto_results} <= _stage_outputs("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION")

    expiry_results = [
        validate_expiry_revocation_stage(envelope, _stage_result("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED"), "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"}),
        validate_expiry_revocation_stage(envelope, _stage_result("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED"), "2026-01-01T00:00:00Z", {"caprev_test1234": "REVOKED"}),
        validate_expiry_revocation_stage(envelope, _stage_result("CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4"), "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"}),
    ]
    assert {result["stage_result"] for result in expiry_results} <= _stage_outputs("EXPIRY_AND_REVOCATION_VALIDATION")

    snapshot_results = [
        create_validated_snapshot(envelope, _stage_result("STRUCTURE_ACCEPTED"), _stage_result("HASHES_ACCEPTED"), _stage_result("REGISTRY_ACCEPTED"), _stage_result("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED"), _stage_result("EXPIRY_REVOCATION_ACCEPTED")),
        create_validated_snapshot(envelope, _stage_result("STRUCTURE_ACCEPTED"), _stage_result("HASHES_REJECTED"), _stage_result("REGISTRY_ACCEPTED"), _stage_result("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED"), _stage_result("EXPIRY_REVOCATION_ACCEPTED")),
    ]
    assert {result["stage_result"] for result in snapshot_results} <= _stage_outputs("VALIDATED_SNAPSHOT_CREATION")


def test_stage_result_registry_has_exact_global_and_per_stage_ownership():
    pipeline = DATA["signature_validation_pipeline"]
    registry = set(pipeline["stage_result_registry"])
    assert set(pipeline["stage_results"]) == registry
    output_union = set().union(*(set(outputs) for outputs in pipeline["stage_outputs"].values()))
    assert output_union == registry
    for outputs in pipeline["stage_outputs"].values():
        assert set(outputs) <= registry
    normal = run_pipeline(raw_document(build_envelope(_payload())), revocation_registry={"caprev_test1234": "NOT_REVOKED"})
    normal_results = {normal[key]["stage_result"] for key in ["structure", "hashes", "registry", "crypto", "expiry_revocation", "snapshot"]}
    future_only = {result for result, spec in pipeline["stage_result_registry"].items() if spec.get("availability") == "FUTURE_ONLY"}
    assert normal_results.isdisjoint(future_only)
    assert [normal[key]["stage_result"] for key in ["structure", "hashes", "registry", "crypto", "expiry_revocation", "snapshot"]] == [
        "STRUCTURE_ACCEPTED",
        "HASHES_ACCEPTED",
        "REGISTRY_ACCEPTED",
        "CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4",
        "EXPIRY_REVOCATION_NOT_REACHED",
        "SNAPSHOT_NOT_CREATED",
    ]


def test_current_edition_signed_payload_policy_and_capability_registry_binding():
    signed_policy = DATA["current_edition_signed_payload_policy"]
    capability_registry = DATA["capability_id_registry"]
    pc_policy = DATA["ProductCapabilities"]["current_edition_capability_policy"]
    assert signed_policy["edition_id"] == DATA["current_product_edition"]["edition_id"] == pc_policy["edition_id"] == "CRYPTOHUNTER_TESTNET_EDITION"
    assert signed_policy["capability_set"] == pc_policy["capability_set"] == capability_registry["current_schema_allowed_capability_ids"]
    assert signed_policy["environment_capabilities_ref"] == "ProductCapabilities.current_edition_capability_policy.environment_capabilities"
    assert signed_policy["feature_flags_ref"] == "ProductCapabilities.current_edition_capability_policy.feature_flags"
    assert signed_policy["fail_closed_policy"] == "SAFE_LOCAL_ONLY"
    assert capability_registry["closed_registry"] is True
    assert capability_registry["document_may_extend_registry"] is False
    valid = build_envelope(_payload())
    assert _validate_parsed_document_structure(valid)["stage_result"] == "STRUCTURE_ACCEPTED"
    mutations = {
        "bad_edition": lambda p: p.__setitem__("edition_id", "ed"),
        "missing_edition": lambda p: p.pop("edition_id"),
        "arbitrary_ab": lambda p: p.__setitem__("capability_set", ["A", "B"]),
        "unknown_capability": lambda p: p.__setitem__("capability_set", ["LIVE_VISIBLE_LOCKED_ONLY", "PAPER_LOCAL_SIMULATION", "UNKNOWN_CAPABILITY"]),
        "additional_capability": lambda p: p["capability_set"].append("LIVE_PRIVATE_EXECUTION"),
        "missing_capability": lambda p: p.__setitem__("capability_set", p["capability_set"][:-1]),
        "future_live_capability": lambda p: p.__setitem__("capability_set", ["LIVE_PRIVATE_EXECUTION", "PAPER_LOCAL_SIMULATION", "TESTNET_PRIVATE_EXECUTION_AFTER_READINESS"]),
        "wrong_order": lambda p: p.__setitem__("capability_set", list(reversed(p["capability_set"]))),
        "duplicate": lambda p: p.__setitem__("capability_set", p["capability_set"] + [p["capability_set"][0]]),
    }
    for name, mutate in mutations.items():
        payload = _payload()
        mutate(payload)
        envelope = build_envelope(payload) if set(payload) == set(CANONICAL_PAYLOAD) else {"capability_payload": payload, "signature_header": valid["signature_header"], "signed_payload_hash": valid["signed_payload_hash"], "signature": valid["signature"]}
        assert _validate_parsed_document_structure(envelope)["stage_result"] == "STRUCTURE_REJECTED", name


def test_document_bound_validation_context_blocks_cross_document_evidence_mixing():
    envelope_a = build_envelope(_payload("cap-A"))
    envelope_b = build_envelope(_payload("cap-B"))
    raw_a, raw_b = raw_document(envelope_a), raw_document(envelope_b)
    structure_a = validate_raw_document_structure(raw_a)
    structure_b = validate_raw_document_structure(raw_b)
    hashes_a = validate_canonical_hashes(envelope_a, structure_a)
    hashes_b = validate_canonical_hashes(envelope_b, structure_b)
    registry_a = validate_registry_policy(envelope_a, hashes_a, {"alg"}, {"key"})
    registry_b = validate_registry_policy(envelope_b, hashes_b, {"alg"}, {"key"})
    crypto_a = _accepted_future_evidence_chain(envelope_a)[3]
    crypto_b = _accepted_future_evidence_chain(envelope_b)[3]
    expiry_a = validate_expiry_revocation_stage(envelope_a, crypto_a, "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"})
    expiry_b = validate_expiry_revocation_stage(envelope_b, crypto_b, "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"})
    assert validate_canonical_hashes(envelope_a, structure_b)["stage_result"] == "HASHES_REJECTED"
    assert validate_registry_policy(envelope_b, hashes_a, {"alg"}, {"key"})["stage_result"] == "REGISTRY_REJECTED"
    assert cryptographic_signature_verification(envelope_b, registry_a)["stage_result"] == "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED"
    assert validate_expiry_revocation_stage(envelope_b, crypto_a, "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"})["stage_result"] == "EXPIRY_REVOCATION_REJECTED"
    assert create_validated_snapshot(envelope_b, structure_a, hashes_a, registry_a, crypto_a, expiry_a)["stage_result"] == "SNAPSHOT_NOT_CREATED"
    for field in ["validation_context_id", "capabilities_id", "signed_payload_hash"]:
        tampered = copy.deepcopy(hashes_a)
        tampered[field] = "tampered"
        assert validate_registry_policy(envelope_a, tampered, {"alg"}, {"key"})["stage_result"] == "REGISTRY_REJECTED"
    assert create_validated_snapshot(envelope_a, structure_a, hashes_a, registry_a, crypto_a, expiry_b)["stage_result"] == "SNAPSHOT_NOT_CREATED"


def test_future_crypto_fixture_is_private_context_bound_and_not_public_pipeline_switch():
    import inspect
    assert "force_future_crypto" not in inspect.signature(run_pipeline).parameters
    assert DATA["signature_validation_pipeline"]["normal_pipeline_allows_force_future_crypto"] is False
    assert DATA["signature_validation_pipeline"]["future_crypto_fixture"] == "closure-local in _run_future_contract_pipeline"
    envelope = build_envelope(_payload())
    normal = run_pipeline(raw_document(envelope), revocation_registry={"caprev_test1234": "NOT_REVOKED"})
    assert normal["crypto"]["stage_result"] == "CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4"
    assert normal["snapshot"]["stage_result"] == "SNAPSHOT_NOT_CREATED"
    future = _run_future_contract_pipeline(raw_document(envelope), revocation_registry={"caprev_test1234": "NOT_REVOKED"})
    assert future["crypto"].get("fixture_availability") == "TEST_FIXTURE_FUTURE_ONLY"
    assert future["snapshot"]["stage_result"] == "SNAPSHOT_CREATED"
    other = build_envelope(_payload("other-cap"))
    assert validate_expiry_revocation_stage(other, future["crypto"], "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"})["stage_result"] == "EXPIRY_REVOCATION_REJECTED"


def test_snapshot_rejects_any_stage_trust_denial_diagnostic_or_context_mismatch():
    envelope = build_envelope(_payload())
    context = _context_from_envelope(envelope)
    accepted = list(_accepted_future_evidence_chain(envelope))
    assert create_validated_snapshot(envelope, *accepted)["stage_result"] == "SNAPSHOT_CREATED"
    regressions = [
        (0, _stage_result("STRUCTURE_ACCEPTED", None, "UNSUPPORTED_SCHEMA", None, context)),
        (1, _stage_result("HASHES_ACCEPTED", None, None, "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA", context)),
        (2, _stage_result("REGISTRY_ACCEPTED", None, "INVALID_SIGNATURE", None, context)),
        (3, _stage_result("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED", None, None, "PRODUCT_CAPABILITIES_INVALID_SIGNATURE", context)),
        (4, _stage_result("EXPIRY_REVOCATION_ACCEPTED", None, "REVOKED", None, context)),
        (4, _stage_result("EXPIRY_REVOCATION_ACCEPTED", "FAIL_CLOSED", None, None, context)),
    ]
    for index, bad in regressions:
        evidence = list(accepted); evidence[index] = bad
        assert create_validated_snapshot(envelope, *evidence)["stage_result"] == "SNAPSHOT_NOT_CREATED"
    mismatched = copy.deepcopy(accepted)
    mismatched[2] = {**mismatched[2], "validation_context_id": "different"}
    assert create_validated_snapshot(envelope, *mismatched)["stage_result"] == "SNAPSHOT_NOT_CREATED"
    mismatched_hash = copy.deepcopy(accepted)
    mismatched_hash[1] = {**mismatched_hash[1], "signed_payload_hash": "0" * 64}
    assert create_validated_snapshot(envelope, *mismatched_hash)["stage_result"] == "SNAPSHOT_NOT_CREATED"


def test_registry_policy_fail_closed_for_malformed_input_types_without_exceptions():
    envelope = build_envelope(_payload())
    structure = _validate_parsed_document_structure(envelope)
    hashes = validate_canonical_hashes(envelope, structure)
    malformed_cases = [
        lambda e: e["signature_header"].__setitem__("signature_algorithm_id", []),
        lambda e: e["signature_header"].__setitem__("signature_algorithm_id", {}),
        lambda e: e["signature_header"].__setitem__("key_id", []),
        lambda e: e["signature_header"].__setitem__("key_id", None),
        lambda e: e.__setitem__("signature_header", []),
        lambda e: e.pop("signature_header"),
        lambda e: e.pop("capability_payload"),
    ]
    for mutate in malformed_cases:
        bad = copy.deepcopy(envelope); mutate(bad)
        result = validate_registry_policy(bad, hashes, {"alg"}, {"key"})
        assert result["stage_result"] == "REGISTRY_REJECTED"
        assert result["diagnostic_reason_code"] in {"REGISTRY_INPUT_DOCUMENT_INVALID", "REGISTRY_REFERENCE_TYPE_INVALID"}
    for algorithms, keys in [(1, {"key"}), ({"alg"}, True), ({"alg": True}, {"key"}), ([[]], {"key"})]:
        result = validate_registry_policy(envelope, hashes, algorithms, keys)
        assert result["stage_result"] == "REGISTRY_REJECTED"
        assert result["diagnostic_reason_code"] in {"REGISTRY_CONTAINER_INVALID", "MISSING_SIGNATURE_ALGORITHM_REGISTRY", "MISSING_VERIFICATION_KEY_REGISTRY"}


def test_defensive_stage_inputs_and_json_surrounding_whitespace_policy():
    envelope = build_envelope(_payload())
    raw = raw_document(envelope)
    for wrapped in ["   " + raw, "\n\t" + raw, raw + "   ", " \n" + raw + "\t\n"]:
        assert validate_raw_document_structure(wrapped)["stage_result"] == "STRUCTURE_ACCEPTED"
    assert validate_raw_document_structure(raw + " {} ")["stage_result"] == "STRUCTURE_REJECTED"
    policy = DATA["raw_json_parse_policy"]
    assert policy["leading_json_whitespace"] == "ALLOW"
    assert policy["trailing_json_whitespace"] == "ALLOW"
    assert policy["trailing_non_whitespace_data"] == "REJECT"
    assert validate_canonical_hashes(envelope, {"stage_result": "STRUCTURE_ACCEPTED"})["stage_result"] == "HASHES_REJECTED"
    assert validate_registry_policy(envelope, {"stage_result": "HASHES_ACCEPTED"}, {"alg"}, {"key"})["stage_result"] == "REGISTRY_REJECTED"
    assert cryptographic_signature_verification(envelope, {"stage_result": "REGISTRY_ACCEPTED"})["stage_result"] == "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED"
    assert validate_expiry_revocation_stage(envelope, {"stage_result": "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED"}, "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"})["stage_result"] == "EXPIRY_REVOCATION_REJECTED"
    assert create_validated_snapshot(envelope, {"stage_result": "STRUCTURE_ACCEPTED"}, {}, {}, {}, {})["stage_result"] == "SNAPSHOT_NOT_CREATED"


def _accepted_future_evidence_chain(envelope: dict):
    future = _run_future_contract_pipeline(raw_document(envelope), revocation_registry={"caprev_test1234": "NOT_REVOKED"})
    return future["structure"], future["hashes"], future["registry"], future["crypto"], future["expiry_revocation"]


def test_same_identity_mutations_change_document_fingerprint_and_reject_prior_evidence():
    envelope_a = build_envelope(_payload())
    structure_a, hashes_a, registry_a, crypto_a, expiry_a = _accepted_future_evidence_chain(envelope_a)
    fingerprint_a = _context_from_envelope(envelope_a)["document_fingerprint"]
    mutations = {
        "feature_flags_live_execution": lambda e: e["capability_payload"]["feature_flags"].__setitem__("live_execution", True),
        "live_private_execution": lambda e: e["capability_payload"]["environment_capabilities"]["LIVE"].__setitem__("private_execution_allowed", True),
        "capability_set": lambda e: e["capability_payload"].__setitem__("capability_set", ["LIVE_VISIBLE_LOCKED_ONLY", "PAPER_LOCAL_SIMULATION", "LIVE_PRIVATE_EXECUTION"]),
        "fail_closed_policy": lambda e: e["capability_payload"].__setitem__("fail_closed_policy", "OPEN"),
        "source": lambda e: e["capability_payload"].__setitem__("source", "signed"),
        "revocation_reference": lambda e: e["capability_payload"].__setitem__("revocation_reference", "caprev_changed1"),
        "signature_header_key_id": lambda e: e["signature_header"].__setitem__("key_id", "other-key"),
        "signature": lambda e: e.__setitem__("signature", base64url_no_padding(b"changed-signature")),
    }
    for name, mutate in mutations.items():
        envelope_b = copy.deepcopy(envelope_a)
        mutate(envelope_b)
        assert _context_from_envelope(envelope_b)["document_fingerprint"] != fingerprint_a, name
        assert validate_registry_policy(envelope_b, hashes_a, {"alg"}, {"key"})["stage_result"] == "REGISTRY_REJECTED", name
        assert cryptographic_signature_verification(envelope_b, registry_a)["stage_result"] == "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED", name
        assert validate_expiry_revocation_stage(envelope_b, crypto_a, "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"})["stage_result"] == "EXPIRY_REVOCATION_REJECTED", name
        assert create_validated_snapshot(envelope_b, structure_a, hashes_a, registry_a, crypto_a, expiry_a)["stage_result"] == "SNAPSHOT_NOT_CREATED", name


def test_every_transition_rejects_dirty_or_malformed_previous_evidence_metadata():
    envelope = build_envelope(_payload())
    structure, hashes, registry, crypto, _expiry = _accepted_future_evidence_chain(envelope)
    transitions = [
        (lambda ev: validate_canonical_hashes(envelope, ev), structure, "HASHES_REJECTED"),
        (lambda ev: validate_registry_policy(envelope, ev, {"alg"}, {"key"}), hashes, "REGISTRY_REJECTED"),
        (lambda ev: cryptographic_signature_verification(envelope, ev), registry, "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED"),
        (lambda ev: validate_expiry_revocation_stage(envelope, ev, "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"}), crypto, "EXPIRY_REVOCATION_REJECTED"),
    ]
    for call, accepted, rejected_result in transitions:
        dirty_cases = []
        for key, value in [
            ("diagnostic_reason_code", "FAIL_CLOSED"),
            ("mapped_trust_state", "UNSUPPORTED_SCHEMA"),
            ("mapped_denial_code", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA"),
            ("stage_id", "WRONG_STAGE"),
            ("document_fingerprint", "0" * 64),
            ("document_fingerprint", 7),
        ]:
            bad = copy.deepcopy(accepted); bad[key] = value; dirty_cases.append(bad)
        missing = copy.deepcopy(accepted); missing.pop("document_fingerprint"); dirty_cases.append(missing)
        for bad in dirty_cases:
            assert call(bad)["stage_result"] == rejected_result


def test_no_public_future_crypto_acceptance_producers_and_snapshot_prerequisites_exact():
    required = [
        "STRUCTURE_ACCEPTED", "HASHES_ACCEPTED", "REGISTRY_ACCEPTED", "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED", "EXPIRY_REVOCATION_ACCEPTED", "complete_stage_evidence", "direct_stage_chain", "same_validation_context_id", "same_document_fingerprint", "same_capabilities_id", "same_edition_id", "same_signed_payload_hash", "same_capability_set_hash", "same_payload_schema_version", "same_signature_schema_version", "no_diagnostic_reason", "no_mapped_non_valid_trust_state", "no_denial_code", "current_edition_policy_consistent", "canonical_hashes_revalidated", "final_document_unchanged", "verified_stage_evidence_id_chain", "crypto_acceptance_fixture_bound_to_registry_evidence", "expiry_evidence_bound_to_crypto_evidence", "evidence_ids_recalculated", "no_fabricated_accepted_stage_evidence", "stage_specific_private_issuer_attestation", "private_predecessor_object_chain_verified", "public_and_private_predecessor_links_match", "serialized_evidence_cannot_preserve_issuer_attestation", "future_crypto_private_fixture_attestation_verified", "fixture_marker_is_diagnostic_not_authoritative", "fully_rehashed_fabricated_chain_rejected", "immutable_issued_stage_evidence", "recursively_validated_predecessor_results", "recursively_validated_predecessor_contexts", "recursively_validated_predecessor_clean_metadata", "stage_specific_issuer_only", "no_generic_accepted_evidence_factory", "direct_wrapper_construction_rejected", "mutated_issued_predecessor_chain_rejected", "issued_evidence_mapping_implementation_composition_not_dict_subclass", "generic_sealer_disallowed", "direct_attestation_constructor_disallowed", "runtime_authority_objects_not_global", "raw_stage_specific_issuers_not_global", "closure_local_attestation_registry", "public_mapping_digest_bound_to_attestation", "object_attribute_tampering_fails_closed", "input_public_dict_copied_before_issue", "no_module_global_evidence_runtime", "no_module_global_future_fixture_token", "no_authority_bundle_factory_global", "no_global_stage_issuer_methods", "strong_current_evidence_ref_identity", "numeric_object_id_not_attestation", "future_fixture_authority_closure_local_unreturned", "no_standalone_future_crypto_issuer_global", "initializer_removed_from_globals_after_install", "future_crypto_acceptance_only_inside_future_pipeline_closure", "future_fixture_authority_not_runtime_argument_or_token",
    ]
    assert DATA["signature_validation_pipeline"]["snapshot_prerequisites"] == required
    public_future_producers = [name for name, value in globals().items() if callable(value) and not name.startswith("_") and not name.startswith("test_") and "future" in name and "crypto" in name]
    assert public_future_producers == []
    assert DATA["signature_validation_pipeline"]["public_future_crypto_acceptance_producers_allowed"] is False
    assert "_future_crypto_accepted_evidence" not in globals()


def test_strict_json_whitespace_rejects_non_json_surrounding_whitespace():
    envelope = build_envelope(_payload())
    raw = raw_document(envelope)
    for prefix in [" ", "\t", "\n", "\r", " \t\r\n"]:
        assert validate_raw_document_structure(prefix + raw + prefix)["stage_result"] == "STRUCTURE_ACCEPTED"
    for bad_prefix in ["\v", "\f", "\u00a0", "\u2003", "\ufeff"]:
        assert validate_raw_document_structure(bad_prefix + raw)["stage_result"] == "STRUCTURE_REJECTED"
    assert validate_raw_document_structure(raw + " \n {}") ["stage_result"] == "STRUCTURE_REJECTED"
    policy = DATA["raw_json_parse_policy"]
    assert policy["legal_surrounding_whitespace_codepoints"] == ["U+0020", "U+0009", "U+000A", "U+000D"]


def test_fabricated_accepted_stage_dicts_cannot_create_snapshot():
    envelope = build_envelope(_payload())
    structure, hashes, registry, crypto, expiry = _accepted_future_evidence_chain(envelope)
    context = _context_from_envelope(envelope)

    fabricated_structure = _stage_result("STRUCTURE_ACCEPTED", context=context)
    fabricated_hashes = _stage_result("HASHES_ACCEPTED", context=context, predecessor_stage_evidence_id="f" * 64)
    fabricated_registry = _stage_result("REGISTRY_ACCEPTED", context=context, predecessor_stage_evidence_id="e" * 64)
    fabricated_crypto = dict(crypto)
    fabricated_crypto.pop("fixture_availability")
    fabricated_expiry = _stage_result("EXPIRY_REVOCATION_ACCEPTED", context=context, predecessor_stage_evidence_id="d" * 64)
    variants = [
        (fabricated_structure, fabricated_hashes, fabricated_registry, fabricated_crypto, fabricated_expiry),
        (structure, {**hashes, "predecessor_stage_evidence_id": "a" * 64}, registry, crypto, expiry),
        (structure, hashes, {**registry, "predecessor_stage_evidence_id": "b" * 64}, crypto, expiry),
        (structure, hashes, registry, {**crypto, "predecessor_stage_evidence_id": None}, expiry),
        (structure, hashes, registry, {k: v for k, v in crypto.items() if k != "fixture_availability"}, expiry),
        (structure, hashes, registry, crypto, {**expiry, "predecessor_stage_evidence_id": None}),
        (structure, {**hashes, "stage_evidence_id": registry["stage_evidence_id"]}, registry, crypto, expiry),
        (structure, hashes, registry, {**crypto, "stage_result": "CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4"}, expiry),
    ]
    for chain in variants:
        assert create_validated_snapshot(envelope, *chain)["stage_result"] == "SNAPSHOT_NOT_CREATED"


def test_stage_evidence_chain_contract_matches_interpreter_and_pure_policy_is_not_stage_evidence():
    chain = DATA["stage_evidence_chain_contract"]
    assert chain["direct_predecessor_map"] == {
        "DOCUMENT_STRUCTURE_VALIDATION": None,
        "CANONICAL_HASH_VALIDATION": "DOCUMENT_STRUCTURE_VALIDATION",
        "REGISTRY_POLICY_VALIDATION": "CANONICAL_HASH_VALIDATION",
        "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION": "REGISTRY_POLICY_VALIDATION",
        "EXPIRY_AND_REVOCATION_VALIDATION": "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION",
        "VALIDATED_SNAPSHOT_CREATION": "EXPIRY_AND_REVOCATION_VALIDATION",
    }
    envelope = build_envelope(_payload())
    structure, hashes, registry, crypto, expiry = _accepted_future_evidence_chain(envelope)
    assert structure["predecessor_stage_evidence_id"] is None
    assert hashes["predecessor_stage_evidence_id"] == structure["stage_evidence_id"]
    assert registry["predecessor_stage_evidence_id"] == hashes["stage_evidence_id"]
    assert crypto["predecessor_stage_evidence_id"] == registry["stage_evidence_id"]
    assert crypto["fixture_availability"] == chain["future_crypto_fixture_marker"]
    assert expiry["predecessor_stage_evidence_id"] == crypto["stage_evidence_id"]
    for evidence in [structure, hashes, registry, crypto, expiry]:
        assert evidence["stage_evidence_id"] == _evidence_id(evidence)
    mutated = dict(hashes); mutated["stage_result"] = "HASHES_REJECTED"
    assert create_validated_snapshot(envelope, structure, mutated, registry, crypto, expiry)["stage_result"] == "SNAPSHOT_NOT_CREATED"
    skipped = create_validated_snapshot(envelope, structure, hashes, crypto, registry, expiry)
    assert skipped["stage_result"] == "SNAPSHOT_NOT_CREATED"
    pure = _evaluate_expiry_and_revocation_policy(envelope, "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"})
    assert set(pure) == {"policy_result", "diagnostic_reason_code", "mapped_trust_state", "mapped_denial_code", "audit_event"}
    assert not {"stage_id", "stage_result", "stage_evidence_id", "predecessor_stage_evidence_id"} & set(pure)


def test_no_public_arbitrary_stage_result_or_future_acceptance_producer():
    import inspect
    assert "stage_result" not in globals()
    allowed_public_interpreters = {"run_pipeline", "validate_raw_document_structure", "validate_canonical_hashes", "validate_registry_policy", "cryptographic_signature_verification", "validate_expiry_revocation_stage", "create_validated_snapshot"}
    public_functions = {
        name: value
        for name, value in globals().items()
        if inspect.isfunction(value) and not name.startswith("_") and not name.startswith("test_")
    }
    assert allowed_public_interpreters <= set(public_functions)
    assert all("stage_result" not in inspect.signature(fn).parameters for fn in public_functions.values())
    assert "future_crypto_accepted" not in public_functions
    assert "future_crypto_rejected" not in public_functions
    envelope = build_envelope(_payload())
    normal = run_pipeline(raw_document(envelope), revocation_registry={"caprev_test1234": "NOT_REVOKED"})
    assert all(result["stage_result"] != "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED" for result in normal.values())
    context = _context_from_envelope(envelope)
    assert _stage_result("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED", context=context)["stage_result"] == "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED"


def _fabricated_public_evidence(result_code: str, context: dict, predecessor_id: str | None, fixture_availability: str | None = None) -> dict:
    evidence = _base_stage_evidence(result_code, context=context, predecessor_stage_evidence_id=predecessor_id, fixture_availability=fixture_availability)
    evidence["stage_evidence_id"] = _evidence_id(evidence)
    return evidence


def test_fully_rehashed_fabricated_chain_cannot_create_snapshot():
    envelope = build_envelope(_payload())
    context = _context_from_envelope(envelope)
    structure = _fabricated_public_evidence("STRUCTURE_ACCEPTED", context, None)
    hashes = _fabricated_public_evidence("HASHES_ACCEPTED", context, structure["stage_evidence_id"])
    registry = _fabricated_public_evidence("REGISTRY_ACCEPTED", context, hashes["stage_evidence_id"])
    crypto = _fabricated_public_evidence("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED", context, registry["stage_evidence_id"], "TEST_FIXTURE_FUTURE_ONLY")
    expiry = _fabricated_public_evidence("EXPIRY_REVOCATION_ACCEPTED", context, crypto["stage_evidence_id"])
    for evidence in [structure, hashes, registry, crypto, expiry]:
        assert evidence["diagnostic_reason_code"] is None
        assert evidence["mapped_trust_state"] is None
        assert evidence["mapped_denial_code"] is None
        assert evidence["document_fingerprint"] == context["document_fingerprint"]
        assert evidence["stage_evidence_id"] == _evidence_id(evidence)
    assert crypto["fixture_availability"] == "TEST_FIXTURE_FUTURE_ONLY"
    assert create_validated_snapshot(envelope, structure, hashes, registry, crypto, expiry)["stage_result"] == "SNAPSHOT_NOT_CREATED"


def test_copy_serialization_and_manual_rehash_do_not_preserve_issuer_attestation():
    envelope = build_envelope(_payload())
    issued = list(_accepted_future_evidence_chain(envelope))
    assert create_validated_snapshot(envelope, *issued)["stage_result"] == "SNAPSHOT_CREATED"
    variants = [
        dict(issued[0]),
        copy.copy(issued[1]),
        copy.deepcopy(issued[2]),
        json.loads(json.dumps(dict(issued[3]))),
        {key: issued[4][key] for key in issued[4]},
    ]
    for index, public_copy in enumerate(variants):
        public_copy["stage_evidence_id"] = _evidence_id(public_copy)
        chain = list(issued)
        chain[index] = public_copy
        assert create_validated_snapshot(envelope, *chain)["stage_result"] == "SNAPSHOT_NOT_CREATED"


def test_transition_level_private_predecessor_attestation_is_required():
    envelope = build_envelope(_payload())
    structure, hashes, registry, crypto, expiry = _accepted_future_evidence_chain(envelope)
    context = _context_from_envelope(envelope)
    forged_structure = _fabricated_public_evidence("STRUCTURE_ACCEPTED", context, None)
    assert validate_canonical_hashes(envelope, forged_structure)["stage_result"] == "HASHES_REJECTED"
    bad_structure = structure
    try:
        bad_structure["predecessor_stage_evidence_id"] = "a" * 64
        raise AssertionError("issued structure evidence mutation was not blocked")
    except TypeError:
        pass
    forged_bad_structure = dict(bad_structure)
    forged_bad_structure["predecessor_stage_evidence_id"] = "a" * 64
    forged_bad_structure["stage_evidence_id"] = _evidence_id(forged_bad_structure)
    assert validate_canonical_hashes(envelope, forged_bad_structure)["stage_result"] == "HASHES_REJECTED"
    forged_hash = _fabricated_public_evidence("HASHES_ACCEPTED", context, "b" * 64)
    assert validate_registry_policy(envelope, forged_hash, {"alg"}, {"key"})["stage_result"] == "REGISTRY_REJECTED"
    forged_registry = _fabricated_public_evidence("REGISTRY_ACCEPTED", context, "c" * 64)
    assert cryptographic_signature_verification(envelope, forged_registry)["stage_result"] == "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED"
    assert validate_expiry_revocation_stage(envelope, dict(crypto), "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"})["stage_result"] == "EXPIRY_REVOCATION_REJECTED"
    wrong_private_ref = registry
    forged_private_ref = dict(wrong_private_ref)
    assert cryptographic_signature_verification(envelope, forged_private_ref)["stage_result"] == "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED"
    wrong_public_id = registry
    try:
        wrong_public_id["predecessor_stage_evidence_id"] = structure["stage_evidence_id"]
        raise AssertionError("public predecessor mutation was not blocked")
    except TypeError:
        pass
    forged_public_id = dict(wrong_public_id)
    forged_public_id["predecessor_stage_evidence_id"] = structure["stage_evidence_id"]
    forged_public_id["stage_evidence_id"] = _evidence_id(forged_public_id)
    assert cryptographic_signature_verification(envelope, forged_public_id)["stage_result"] == "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED"
    other = build_envelope(_payload("cap-other"))
    assert create_validated_snapshot(other, structure, hashes, registry, crypto, expiry)["stage_result"] == "SNAPSHOT_NOT_CREATED"


def test_generic_helper_cannot_issue_any_accepted_stage_evidence_and_public_api_hides_attestation_inputs():
    import inspect
    context = _context_from_envelope(build_envelope(_payload()))
    for accepted_result in _ACCEPTED_RESULTS:
        evidence = _stage_result(accepted_result, context=context)
        assert not isinstance(evidence, _IssuedStageEvidence)
        assert evidence["stage_result"] != accepted_result
    public_functions = [
        run_pipeline,
        validate_raw_document_structure,
        validate_canonical_hashes,
        validate_registry_policy,
        cryptographic_signature_verification,
        validate_expiry_revocation_stage,
        create_validated_snapshot,
    ]
    forbidden = {"issuer_token", "producer_attestation", "predecessor_evidence_ref", "fixture_token", "force_future_crypto"}
    for fn in public_functions:
        assert forbidden.isdisjoint(inspect.signature(fn).parameters)
    normal = run_pipeline(raw_document(build_envelope(_payload())), revocation_registry={"caprev_test1234": "NOT_REVOKED"})
    assert normal["crypto"]["stage_result"] == "CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4"
    assert normal["snapshot"]["stage_result"] == "SNAPSHOT_NOT_CREATED"


def test_mutated_issued_structure_chain_cannot_reach_registry_accepted():
    envelope = build_envelope(_payload())
    structure = _validate_parsed_document_structure(envelope)
    hashes = validate_canonical_hashes(envelope, structure)
    for key, value in [
        ("stage_result", "STRUCTURE_REJECTED"),
        ("diagnostic_reason_code", "FORGED"),
        ("mapped_trust_state", "UNSUPPORTED_SCHEMA"),
        ("mapped_denial_code", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA"),
    ]:
        try:
            structure[key] = value
            raise AssertionError("issued evidence mutation was not blocked")
        except TypeError:
            pass
    forged_structure = dict(structure)
    forged_hashes = dict(hashes)
    forged_structure["stage_result"] = "STRUCTURE_REJECTED"
    forged_structure["diagnostic_reason_code"] = "FORGED"
    forged_structure["mapped_trust_state"] = "UNSUPPORTED_SCHEMA"
    forged_structure["mapped_denial_code"] = "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA"
    forged_structure["stage_evidence_id"] = _evidence_id(forged_structure)
    forged_hashes["predecessor_stage_evidence_id"] = forged_structure["stage_evidence_id"]
    forged_hashes["stage_evidence_id"] = _evidence_id(forged_hashes)
    assert validate_registry_policy(envelope, forged_hashes, {"alg"}, {"key"})["stage_result"] == "REGISTRY_REJECTED"


def test_mutated_issued_predecessor_context_cannot_reach_successor_acceptance():
    envelope = build_envelope(_payload())
    structure, hashes, registry, crypto, expiry = _accepted_future_evidence_chain(envelope)
    for stage_call, evidence, rejected in [
        (lambda ev: validate_canonical_hashes(envelope, ev), structure, "HASHES_REJECTED"),
        (lambda ev: validate_registry_policy(envelope, ev, {"alg"}, {"key"}), hashes, "REGISTRY_REJECTED"),
        (lambda ev: cryptographic_signature_verification(envelope, ev), registry, "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED"),
        (lambda ev: validate_expiry_revocation_stage(envelope, ev, "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"}), crypto, "EXPIRY_REVOCATION_REJECTED"),
    ]:
        forged = dict(evidence)
        forged["capabilities_id"] = "forged-capability"
        forged["stage_evidence_id"] = _evidence_id(forged)
        assert stage_call(forged)["stage_result"] == rejected
    forged_expiry = dict(expiry)
    forged_expiry["edition_id"] = "FORGED_EDITION"
    forged_expiry["stage_evidence_id"] = _evidence_id(forged_expiry)
    assert create_validated_snapshot(envelope, structure, hashes, registry, crypto, forged_expiry)["stage_result"] == "SNAPSHOT_NOT_CREATED"


def test_mutated_issued_predecessor_result_cannot_reach_successor_acceptance():
    envelope = build_envelope(_payload())
    structure, hashes, registry, crypto, expiry = _accepted_future_evidence_chain(envelope)
    for stage_call, evidence, rejected_result in [
        (lambda ev: validate_canonical_hashes(envelope, ev), structure, "HASHES_REJECTED"),
        (lambda ev: validate_registry_policy(envelope, ev, {"alg"}, {"key"}), hashes, "REGISTRY_REJECTED"),
        (lambda ev: cryptographic_signature_verification(envelope, ev), registry, "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED"),
        (lambda ev: validate_expiry_revocation_stage(envelope, ev, "2026-01-01T00:00:00Z", {"caprev_test1234": "NOT_REVOKED"}), crypto, "EXPIRY_REVOCATION_REJECTED"),
    ]:
        forged = dict(evidence)
        forged["stage_result"] = _ACCEPTED_FAIL_CLOSED[evidence["stage_result"]]
        forged["stage_evidence_id"] = _evidence_id(forged)
        assert stage_call(forged)["stage_result"] != evidence["stage_result"]
    forged_expiry = dict(expiry)
    forged_expiry["stage_result"] = "EXPIRY_REVOCATION_REJECTED"
    forged_expiry["stage_evidence_id"] = _evidence_id(forged_expiry)
    assert create_validated_snapshot(envelope, structure, hashes, registry, crypto, forged_expiry)["stage_result"] == "SNAPSHOT_NOT_CREATED"


def test_mutated_issued_predecessor_diagnostics_cannot_reach_successor_acceptance():
    envelope = build_envelope(_payload())
    structure, hashes, registry, crypto, expiry = _accepted_future_evidence_chain(envelope)
    dirty_fields = [
        ("diagnostic_reason_code", "FORGED_DIAGNOSTIC"),
        ("mapped_trust_state", "UNSUPPORTED_SCHEMA"),
        ("mapped_denial_code", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA"),
    ]
    for key, value in dirty_fields:
        forged = dict(registry)
        forged[key] = value
        forged["stage_evidence_id"] = _evidence_id(forged)
        assert cryptographic_signature_verification(envelope, forged)["stage_result"] == "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED"
    forged_expiry = dict(expiry)
    forged_expiry["diagnostic_reason_code"] = "FORGED_DIAGNOSTIC"
    forged_expiry["stage_evidence_id"] = _evidence_id(forged_expiry)
    assert create_validated_snapshot(envelope, structure, hashes, registry, crypto, forged_expiry)["stage_result"] == "SNAPSHOT_NOT_CREATED"


def test_all_issued_evidence_mapping_mutators_are_blocked():
    evidence = _validate_parsed_document_structure(build_envelope(_payload()))
    mutators = [
        lambda: evidence.__setitem__("stage_result", "STRUCTURE_REJECTED"),
        lambda: evidence.__delitem__("stage_result"),
        lambda: evidence.clear(),
        lambda: evidence.pop("stage_result"),
        lambda: evidence.popitem(),
        lambda: evidence.setdefault("x", True),
        lambda: evidence.update({"stage_result": "STRUCTURE_REJECTED"}),
        lambda: evidence.__ior__({"stage_result": "STRUCTURE_REJECTED"}),
    ]
    for mutate in mutators:
        try:
            mutate()
            raise AssertionError("mapping mutator was not blocked")
        except TypeError:
            pass


def test_private_provenance_attributes_are_immutable():
    evidence = _validate_parsed_document_structure(build_envelope(_payload()))
    for name, value in [
        ("_issuer_token", object()),
        ("_producer_stage_id", "FORGED"),
        ("_predecessor_evidence_ref", evidence),
        ("_sealed", False),
    ]:
        try:
            setattr(evidence, name, value)
            raise AssertionError(f"private provenance field {name} was not blocked")
        except TypeError:
            pass


def test_no_generic_accepted_evidence_issuer_exists():
    import inspect
    forbidden_globals = {
        "_issue_stage_evidence",
        "_seal_stage_evidence",
        "_issuer_token_for_stage",
        "_WRAPPER_CONSTRUCTION_AUTHORITY",
        "_issue_structure_accepted",
        "_issue_hashes_accepted",
        "_issue_registry_accepted",
        "_issue_future_crypto_accepted",
        "_issue_expiry_revocation_accepted",
        "_issue_snapshot_created",
    }
    assert forbidden_globals.isdisjoint(globals())
    assert not any(name.endswith("_ISSUER_TOKEN") for name in globals())
    forbidden_params = {"stage_result", "issuer_token", "construction_authority", "producer_stage_id", "predecessor_evidence_ref"}
    for value in globals().values():
        if inspect.isfunction(value):
            assert forbidden_params.isdisjoint(inspect.signature(value).parameters)


def test_no_module_global_runtime_issuer_token_registry_exists():
    assert "_STAGE_ISSUER_TOKENS" not in globals()
    assert DATA["stage_evidence_producer_attestation_contract"]["issuer_token_values_exposed_in_json"] is False
    assert "issuer_id_registry" in DATA["stage_evidence_producer_attestation_contract"]


def test_direct_wrapper_construction_cannot_create_valid_attestation():
    public = _fabricated_public_evidence("STRUCTURE_ACCEPTED", _context_from_envelope(build_envelope(_payload())), None)
    try:
        _IssuedStageEvidence(public, issuer_token=object(), producer_stage_id="DOCUMENT_STRUCTURE_VALIDATION", predecessor_evidence_ref=None)
        raise AssertionError("direct wrapper construction was not blocked")
    except TypeError:
        pass
    assert not _issued_chain_valid(public)


def test_full_recursive_chain_validation_checks_results_context_and_clean_metadata():
    envelope = build_envelope(_payload())
    structure, hashes, registry, crypto, expiry = _accepted_future_evidence_chain(envelope)
    for evidence in [structure, hashes, registry, crypto, expiry]:
        assert _issued_chain_valid(evidence, _context_from_envelope(envelope))
    for key, value in [
        ("stage_id", "FORGED_STAGE"),
        ("capability_set_hash", "0" * 64),
        ("fixture_availability", "TEST_FIXTURE_FUTURE_ONLY"),
        ("diagnostic_reason_code", "FORGED"),
        ("mapped_trust_state", "UNSUPPORTED_SCHEMA"),
        ("mapped_denial_code", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA"),
    ]:
        forged = dict(hashes)
        forged[key] = value
        forged["stage_evidence_id"] = _evidence_id(forged)
        assert not _issued_chain_valid(forged, _context_from_envelope(envelope))


def test_generic_sealer_and_module_authorities_cannot_fabricate_snapshot():
    import inspect
    forbidden_names = {
        "_seal_stage_evidence",
        "_issuer_token_for_stage",
        "_WRAPPER_CONSTRUCTION_AUTHORITY",
    }
    assert forbidden_names.isdisjoint(globals())
    assert not any(name.endswith("_ISSUER_TOKEN") for name in globals())
    forbidden_params = {"stage_result", "issuer_token", "construction_authority", "producer_stage_id"}
    for name, value in globals().items():
        if inspect.isfunction(value):
            assert forbidden_params.isdisjoint(inspect.signature(value).parameters), name
    envelope = build_envelope(_payload())
    context = _context_from_envelope(envelope)
    structure = _fabricated_public_evidence("STRUCTURE_ACCEPTED", context, None)
    hashes = _fabricated_public_evidence("HASHES_ACCEPTED", context, structure["stage_evidence_id"])
    registry = _fabricated_public_evidence("REGISTRY_ACCEPTED", context, hashes["stage_evidence_id"])
    crypto = _fabricated_public_evidence("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED", context, registry["stage_evidence_id"], "TEST_FIXTURE_FUTURE_ONLY")
    expiry = _fabricated_public_evidence("EXPIRY_REVOCATION_ACCEPTED", context, crypto["stage_evidence_id"])
    assert create_validated_snapshot(envelope, structure, hashes, registry, crypto, expiry)["stage_result"] == "SNAPSHOT_NOT_CREATED"


def test_issued_evidence_is_not_a_dict_subclass():
    evidence = _validate_parsed_document_structure(build_envelope(_payload()))
    assert isinstance(evidence, _IssuedStageEvidence)
    assert isinstance(evidence, Mapping)
    assert not isinstance(evidence, dict)


def test_dict_base_class_mutation_cannot_modify_issued_evidence():
    evidence = _validate_parsed_document_structure(build_envelope(_payload()))
    before = dict(evidence)
    for mutate in [
        lambda: dict.__setitem__(evidence, "stage_result", "STRUCTURE_REJECTED"),
        lambda: dict.__delitem__(evidence, "stage_result"),
        lambda: dict.update(evidence, {"stage_result": "STRUCTURE_REJECTED"}),
        lambda: dict.clear(evidence),
    ]:
        try:
            mutate()
            raise AssertionError("dict base-class mutator unexpectedly accepted Mapping evidence")
        except TypeError:
            pass
    assert dict(evidence) == before


def test_object_setattr_tampering_fails_closed():
    envelope = build_envelope(_payload())
    issued = list(_accepted_future_evidence_chain(envelope))
    assert create_validated_snapshot(envelope, *issued)["stage_result"] == "SNAPSHOT_CREATED"
    tampered = issued[0]
    forged_public = dict(tampered)
    forged_public["stage_result"] = "STRUCTURE_REJECTED"
    forged_public["stage_evidence_id"] = _evidence_id(forged_public)
    object.__setattr__(tampered, "_public_fields", forged_public)
    assert not _issued_chain_valid(tampered, _context_from_envelope(envelope))
    assert create_validated_snapshot(envelope, *issued)["stage_result"] == "SNAPSHOT_NOT_CREATED"


def test_input_public_dict_is_copied_before_issue():
    public = _fabricated_public_evidence("STRUCTURE_ACCEPTED", _context_from_envelope(build_envelope(_payload())), None)
    evidence = _IssuedStageEvidence(public)
    before = dict(evidence)
    public["stage_result"] = "STRUCTURE_REJECTED"
    public["stage_evidence_id"] = _evidence_id(public)
    assert dict(evidence) == before
    assert not _issued_chain_valid(evidence)


def test_accepted_chain_cannot_be_built_by_calling_raw_issuers():
    raw_issuer_names = {
        "_issue_structure_accepted",
        "_issue_hashes_accepted",
        "_issue_registry_accepted",
        "_issue_future_crypto_accepted",
        "_issue_expiry_revocation_accepted",
        "_issue_snapshot_created",
    }
    assert raw_issuer_names.isdisjoint(globals())
    assert "_run_future_contract_pipeline" in globals()
    envelope = build_envelope(_payload())
    context = _context_from_envelope(envelope)
    chain = [
        _fabricated_public_evidence("STRUCTURE_ACCEPTED", context, None),
    ]
    chain.append(_fabricated_public_evidence("HASHES_ACCEPTED", context, chain[-1]["stage_evidence_id"]))
    chain.append(_fabricated_public_evidence("REGISTRY_ACCEPTED", context, chain[-1]["stage_evidence_id"]))
    chain.append(_fabricated_public_evidence("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED", context, chain[-1]["stage_evidence_id"], "TEST_FIXTURE_FUTURE_ONLY"))
    chain.append(_fabricated_public_evidence("EXPIRY_REVOCATION_ACCEPTED", context, chain[-1]["stage_evidence_id"]))
    assert create_validated_snapshot(envelope, *chain)["stage_result"] == "SNAPSHOT_NOT_CREATED"


def test_direct_wrapper_construction_variants_do_not_create_attestation():
    public = _fabricated_public_evidence("STRUCTURE_ACCEPTED", _context_from_envelope(build_envelope(_payload())), None)
    direct = _IssuedStageEvidence(public)
    assert not _issued_chain_valid(direct)
    assert not hasattr(_IssuedStageEvidence, "_create")
    assert not hasattr(_IssuedStageEvidence, "create")
    assert not hasattr(_IssuedStageEvidence, "from_fields")
    try:
        dict.__new__(_IssuedStageEvidence)
        raise AssertionError("dict.__new__ should not construct Mapping evidence")
    except TypeError:
        pass
    raw = object.__new__(_IssuedStageEvidence)
    object.__setattr__(raw, "_public_fields", dict(public))
    assert not _issued_chain_valid(raw)
    object.__setattr__(raw, "_public_fields", {**dict(public), "issuer_token": object(), "construction_authority": object()})
    assert not _issued_chain_valid(raw)


def _initialize_closure_local_interpreter() -> None:
    issued_registry: dict[int, dict] = {}
    def public_digest(evidence: Mapping) -> str:
        return _evidence_id(dict(evidence))

    def seal(public: dict, predecessor):
        evidence = _IssuedStageEvidence(public)
        context = {key: evidence.get(key) for key in DATA["validation_context_contract"]["context_fields"]}
        issued_registry[id(evidence)] = {
            "evidence_ref": evidence,
            "diagnostic_identity": id(evidence),
            "stage_id": evidence.get("stage_id"),
            "stage_result": evidence.get("stage_result"),
            "context": context,
            "predecessor_ref": predecessor,
            "public_digest": public_digest(evidence),
        }
        return evidence

    def issued_chain_valid(evidence, context: dict | None = None) -> bool:
        if not isinstance(evidence, _IssuedStageEvidence) or isinstance(evidence, dict):
            return False
        meta = issued_registry.get(id(evidence))
        if meta is None or meta["evidence_ref"] is not evidence:
            return False
        public = dict(evidence)
        if meta["public_digest"] != public_digest(evidence):
            return False
        required = set(DATA["validation_context_contract"]["stage_evidence_fields"])
        if set(public) != required:
            return False
        stage_id = public.get("stage_id")
        expected_result = _EXPECTED_ACCEPTED_RESULT.get(stage_id)
        if expected_result is None or public.get("stage_result") != expected_result:
            return False
        if meta["stage_id"] != stage_id or meta["stage_result"] != expected_result:
            return False
        if public.get("evidence_schema_version") != DATA["stage_evidence_chain_contract"]["evidence_schema_version"]:
            return False
        if not _evidence_id_is_valid(public):
            return False
        public_context = {key: public.get(key) for key in DATA["validation_context_contract"]["context_fields"]}
        if context is None:
            context = public_context
        if meta["context"] != public_context or public_context != {key: context.get(key) for key in DATA["validation_context_contract"]["context_fields"]}:
            return False
        if public.get("diagnostic_reason_code") is not None or public.get("mapped_trust_state") is not None or public.get("mapped_denial_code") is not None:
            return False
        if not isinstance(public.get("document_fingerprint"), str) or not HEX64.fullmatch(public["document_fingerprint"]):
            return False
        for key in ["validation_context_id", "capabilities_id", "edition_id", "signed_payload_hash", "capability_set_hash", "payload_schema_version", "signature_schema_version"]:
            if not isinstance(public.get(key), str) or not public.get(key):
                return False
        marker = public.get("fixture_availability")
        if stage_id == "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION":
            if marker != DATA["stage_evidence_chain_contract"]["future_crypto_fixture_marker"]:
                return False
        elif marker is not None:
            return False
        predecessor_stage = _PREDECESSOR_STAGE.get(stage_id)
        predecessor_ref = meta["predecessor_ref"]
        if predecessor_stage is None:
            return predecessor_ref is None and public.get("predecessor_stage_evidence_id") is None
        if not isinstance(predecessor_ref, _IssuedStageEvidence):
            return False
        return (
            predecessor_ref.get("stage_id") == predecessor_stage
            and predecessor_ref.get("stage_result") == _EXPECTED_ACCEPTED_RESULT[predecessor_stage]
            and public.get("predecessor_stage_evidence_id") == predecessor_ref.get("stage_evidence_id")
            and issued_chain_valid(predecessor_ref, context)
        )

    def complete_clean_evidence(evidence, expected_stage_id: str, expected_result: str, context: dict, predecessor=None) -> bool:
        if not issued_chain_valid(evidence, context):
            return False
        meta = issued_registry.get(id(evidence))
        if dict(evidence).get("stage_id") != expected_stage_id or dict(evidence).get("stage_result") != expected_result:
            return False
        if predecessor is None:
            if expected_stage_id == "DOCUMENT_STRUCTURE_VALIDATION":
                return meta["predecessor_ref"] is None and evidence.get("predecessor_stage_evidence_id") is None
            return True
        return (
            isinstance(predecessor, _IssuedStageEvidence)
            and meta["predecessor_ref"] is predecessor
            and evidence.get("predecessor_stage_evidence_id") == predecessor.get("stage_evidence_id")
            and issued_chain_valid(predecessor, context)
        )

    def issue_structure(context: dict):
        return seal(_base_stage_evidence("STRUCTURE_ACCEPTED", context=context, predecessor_stage_evidence_id=None), None)

    def issue_hashes(context: dict, structure_evidence):
        if not complete_clean_evidence(structure_evidence, "DOCUMENT_STRUCTURE_VALIDATION", "STRUCTURE_ACCEPTED", context, predecessor=None):
            raise ValueError("hash accepted evidence requires issued structure predecessor")
        return seal(_base_stage_evidence("HASHES_ACCEPTED", context=context, predecessor_stage_evidence_id=structure_evidence["stage_evidence_id"]), structure_evidence)

    def issue_registry(context: dict, hash_evidence):
        if not complete_clean_evidence(hash_evidence, "CANONICAL_HASH_VALIDATION", "HASHES_ACCEPTED", context):
            raise ValueError("registry accepted evidence requires issued hash predecessor chain")
        return seal(_base_stage_evidence("REGISTRY_ACCEPTED", context=context, predecessor_stage_evidence_id=hash_evidence["stage_evidence_id"]), hash_evidence)

    def issue_future_crypto(context: dict, registry_evidence):
        if not complete_clean_evidence(registry_evidence, "REGISTRY_POLICY_VALIDATION", "REGISTRY_ACCEPTED", context):
            raise ValueError("future crypto accepted evidence requires issued registry predecessor chain")
        return seal(_base_stage_evidence("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED", context=context, predecessor_stage_evidence_id=registry_evidence["stage_evidence_id"], fixture_availability="TEST_FIXTURE_FUTURE_ONLY"), registry_evidence)

    def issue_expiry(context: dict, crypto_evidence):
        if not complete_clean_evidence(crypto_evidence, "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION", "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED", context):
            raise ValueError("expiry accepted evidence requires issued crypto predecessor chain")
        return seal(_base_stage_evidence("EXPIRY_REVOCATION_ACCEPTED", context=context, predecessor_stage_evidence_id=crypto_evidence["stage_evidence_id"]), crypto_evidence)

    def issue_snapshot(context: dict, expiry_evidence):
        if not complete_clean_evidence(expiry_evidence, "EXPIRY_AND_REVOCATION_VALIDATION", "EXPIRY_REVOCATION_ACCEPTED", context):
            raise ValueError("snapshot evidence requires issued expiry predecessor chain")
        return seal(_base_stage_evidence("SNAPSHOT_CREATED", context=context, predecessor_stage_evidence_id=expiry_evidence["stage_evidence_id"]), expiry_evidence)

    def parsed_structure(envelope: dict, version_registry=DEFAULT_VERSION_REGISTRY, schemas=None):
        schemas = schemas or DATA["ProductCapabilities"]["document_schemas"]
        try:
            if not isinstance(envelope, dict) or set(envelope) != set(ENVELOPE): return _stage_result("STRUCTURE_REJECTED")
            payload, header = envelope["capability_payload"], envelope["signature_header"]
            if not isinstance(payload, dict) or not isinstance(header, dict): return _stage_result("STRUCTURE_REJECTED")
            if "schema_registry" in payload or "schema_registry" in envelope: return _stage_result("STRUCTURE_REJECTED", "PAYLOAD_SCHEMA_REGISTRY_OVERRIDE_FORBIDDEN", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA")
            if set(payload) != set(CANONICAL_PAYLOAD) or set(header) != set(HEADER): return _stage_result("STRUCTURE_REJECTED")
            if not isinstance(version_registry, dict) or set(version_registry) != {"payload", "signature"} or not version_registry["payload"] or not version_registry["signature"]:
                return _stage_result("STRUCTURE_REJECTED", "MISSING_SCHEMA_VERSION_REGISTRY", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA")
            if payload["schema_version"] not in version_registry["payload"]: return _stage_result("STRUCTURE_REJECTED", "UNKNOWN_CAPABILITY_PAYLOAD_SCHEMA_VERSION", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA")
            if header["signature_schema_version"] not in version_registry["signature"]: return _stage_result("STRUCTURE_REJECTED", "UNKNOWN_SIGNATURE_SCHEMA_VERSION", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA")
            reject_numbers_in_signed_payload(payload)
            for field, contract in schemas["capability_payload_schema"]["field_contracts"].items():
                if not validate_field(payload[field], contract): return _stage_result("STRUCTURE_REJECTED")
            for field, contract in schemas["signature_header_schema"]["field_contracts"].items():
                if not validate_field(header[field], contract): return _stage_result("STRUCTURE_REJECTED")
            if not validate_nested_environment_capabilities(payload["environment_capabilities"]): return _stage_result("STRUCTURE_REJECTED")
            if not validate_feature_flags(payload["feature_flags"]): return _stage_result("STRUCTURE_REJECTED")
            if not _current_edition_policy_consistent(envelope): return _stage_result("STRUCTURE_REJECTED", "EDITION_POLICY_MISMATCH", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA")
            return issue_structure(_context_from_envelope(envelope))
        except Exception:
            return _stage_result("STRUCTURE_REJECTED")

    def raw_structure(raw_document, version_registry=DEFAULT_VERSION_REGISTRY):
        if not isinstance(raw_document, (str, bytes)):
            return _stage_result("STRUCTURE_REJECTED", "RAW_DOCUMENT_TYPE_NOT_SUPPORTED", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA")
        try:
            return parsed_structure(parse_raw_json_without_duplicates(raw_document), version_registry)
        except Exception:
            return _stage_result("STRUCTURE_REJECTED")

    def canonical_hashes(envelope: dict, structure_result: Mapping) -> dict:
        try:
            if not isinstance(structure_result, Mapping) or structure_result.get("stage_result") != "STRUCTURE_ACCEPTED":
                return _stage_result("HASH_VALIDATION_NOT_REACHED")
            context = _context_from_envelope(envelope)
            if not complete_clean_evidence(structure_result, "DOCUMENT_STRUCTURE_VALIDATION", "STRUCTURE_ACCEPTED", context):
                return _stage_result("HASHES_REJECTED", "PREVIOUS_STAGE_EVIDENCE_INVALID", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA", context, structure_result.get("stage_evidence_id") if isinstance(structure_result, Mapping) else None)
            payload = envelope["capability_payload"]
            if payload["capability_set"] != sorted(payload["capability_set"], key=utf16_code_unit_sort_key): return _stage_result("HASHES_REJECTED", context=context, predecessor_stage_evidence_id=structure_result.get("stage_evidence_id"))
            if payload["capability_set_hash"] != capability_set_hash(payload["capability_set"]): return _stage_result("HASHES_REJECTED", context=context, predecessor_stage_evidence_id=structure_result.get("stage_evidence_id"))
            if envelope["signed_payload_hash"] != sha_bytes(jcs_current_schema_canonicalize(payload)): return _stage_result("HASHES_REJECTED", context=context, predecessor_stage_evidence_id=structure_result.get("stage_evidence_id"))
            return issue_hashes(context, structure_result)
        except Exception:
            return _stage_result("HASHES_REJECTED", "PREVIOUS_STAGE_EVIDENCE_INVALID", "UNSUPPORTED_SCHEMA", "PRODUCT_CAPABILITIES_UNSUPPORTED_SCHEMA")

    def registry_policy(envelope: dict, hash_result: Mapping, recognized_algorithm_ids: set[str] | None, recognized_key_ids: set[str] | None) -> dict:
        contract = DATA["signature_contract"]["signature_verification_registry_contract"]
        def rejected(reason: str, context: dict | None = None) -> dict:
            return _stage_result(contract["rejected_stage_result"], reason, contract["reason_code_to_trust_state"].get(reason), contract["reason_code_to_denial_code"].get(reason), context, hash_result.get("stage_evidence_id") if isinstance(hash_result, Mapping) else None)
        try:
            if not isinstance(hash_result, Mapping) or hash_result.get("stage_result") != "HASHES_ACCEPTED":
                return _stage_result("REGISTRY_VALIDATION_NOT_REACHED")
            if hash_result.get("stage_id") != "CANONICAL_HASH_VALIDATION":
                return rejected("PREVIOUS_STAGE_EVIDENCE_INVALID")
            if not isinstance(envelope, dict):
                return rejected("REGISTRY_INPUT_DOCUMENT_INVALID")
            header = envelope.get("signature_header"); payload = envelope.get("capability_payload")
            if not isinstance(header, dict) or not isinstance(payload, dict):
                return rejected("REGISTRY_INPUT_DOCUMENT_INVALID")
            algorithm_id = header.get("signature_algorithm_id"); key_id = header.get("key_id")
            if not isinstance(algorithm_id, str) or not algorithm_id or not isinstance(key_id, str) or not key_id:
                return rejected("REGISTRY_REFERENCE_TYPE_INVALID")
            context = _context_from_envelope(envelope)
            if not complete_clean_evidence(hash_result, "CANONICAL_HASH_VALIDATION", "HASHES_ACCEPTED", context):
                return rejected("PREVIOUS_STAGE_EVIDENCE_INVALID", context)
        except Exception:
            return rejected("REGISTRY_INPUT_DOCUMENT_INVALID")
        if recognized_algorithm_ids is None or recognized_key_ids is None:
            return rejected("REGISTRY_INPUT_DOCUMENT_INVALID", context)
        if not _valid_registry_container(recognized_algorithm_ids) or not _valid_registry_container(recognized_key_ids):
            return rejected("REGISTRY_CONTAINER_INVALID", context)
        if not recognized_algorithm_ids: return rejected("MISSING_SIGNATURE_ALGORITHM_REGISTRY", context)
        if not recognized_key_ids: return rejected("MISSING_VERIFICATION_KEY_REGISTRY", context)
        if algorithm_id not in set(recognized_algorithm_ids): return rejected("UNKNOWN_SIGNATURE_ALGORITHM", context)
        if key_id not in set(recognized_key_ids): return rejected("UNKNOWN_KEY_ID", context)
        return issue_registry(context, hash_result)

    def crypto_verify(envelope: dict, registry_result: Mapping) -> dict:
        try:
            if not isinstance(registry_result, Mapping) or registry_result.get("stage_result") != "REGISTRY_ACCEPTED":
                return _stage_result("CRYPTOGRAPHIC_VERIFICATION_NOT_REACHED")
            context = _context_from_envelope(envelope)
            if not complete_clean_evidence(registry_result, "REGISTRY_POLICY_VALIDATION", "REGISTRY_ACCEPTED", context):
                return _stage_result("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED", "PREVIOUS_STAGE_EVIDENCE_INVALID", "INVALID_SIGNATURE", "PRODUCT_CAPABILITIES_INVALID_SIGNATURE", context, registry_result.get("stage_evidence_id") if isinstance(registry_result, Mapping) else None)
            return _stage_result("CRYPTOGRAPHIC_VERIFICATION_NOT_IMPLEMENTED_IN_M0_4", context=context, predecessor_stage_evidence_id=registry_result["stage_evidence_id"])
        except Exception:
            return _stage_result("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED", "PREVIOUS_STAGE_EVIDENCE_INVALID", "INVALID_SIGNATURE", "PRODUCT_CAPABILITIES_INVALID_SIGNATURE")

    def future_crypto(registry_evidence: Mapping) -> dict:
        context = {key: registry_evidence.get(key) for key in DATA["validation_context_contract"]["context_fields"]} if isinstance(registry_evidence, Mapping) else {}
        if not complete_clean_evidence(registry_evidence, "REGISTRY_POLICY_VALIDATION", "REGISTRY_ACCEPTED", context):
            return _stage_result("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_REJECTED", "PREVIOUS_STAGE_EVIDENCE_INVALID", "INVALID_SIGNATURE", "PRODUCT_CAPABILITIES_INVALID_SIGNATURE", context if context else None, registry_evidence.get("stage_evidence_id") if isinstance(registry_evidence, Mapping) else None)
        return issue_future_crypto(context, registry_evidence)

    def expiry_stage(envelope: dict, crypto_result: Mapping, current_utc: str, revocation_registry: dict | None) -> dict:
        try:
            if not isinstance(crypto_result, Mapping) or crypto_result.get("stage_result") != "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED":
                return _stage_result("EXPIRY_REVOCATION_NOT_REACHED")
            context = _context_from_envelope(envelope)
            if not complete_clean_evidence(crypto_result, "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION", "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED", context):
                return _stage_result("EXPIRY_REVOCATION_REJECTED", "PREVIOUS_STAGE_EVIDENCE_INVALID", "INVALID_SIGNATURE", "PRODUCT_CAPABILITIES_INVALID_SIGNATURE", context)
            decision = _evaluate_expiry_and_revocation_policy(envelope, current_utc, revocation_registry)
            return issue_expiry(context, crypto_result) if decision["policy_result"] == "EXPIRY_REVOCATION_ACCEPTED" else _stage_result(decision["policy_result"], decision.get("diagnostic_reason_code"), decision.get("mapped_trust_state"), decision.get("mapped_denial_code"), context, crypto_result["stage_evidence_id"])
        except Exception:
            return _stage_result("EXPIRY_REVOCATION_REJECTED", "PREVIOUS_STAGE_EVIDENCE_INVALID", "INVALID_SIGNATURE", "PRODUCT_CAPABILITIES_INVALID_SIGNATURE")

    def snapshot(envelope: dict, structure_result: Mapping, hash_result: Mapping, registry_result: Mapping, crypto_result: Mapping, expiry_revocation_result: Mapping) -> dict:
        try:
            context = _context_from_envelope(envelope)
            evidences = [structure_result, hash_result, registry_result, crypto_result, expiry_revocation_result]
            expected = [("DOCUMENT_STRUCTURE_VALIDATION", "STRUCTURE_ACCEPTED"), ("CANONICAL_HASH_VALIDATION", "HASHES_ACCEPTED"), ("REGISTRY_POLICY_VALIDATION", "REGISTRY_ACCEPTED"), ("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION", "CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED"), ("EXPIRY_AND_REVOCATION_VALIDATION", "EXPIRY_REVOCATION_ACCEPTED")]
            predecessors = [None, structure_result, hash_result, registry_result, crypto_result]
            for evidence, (stage_id, result), predecessor in zip(evidences, expected, predecessors):
                if not complete_clean_evidence(evidence, stage_id, result, context, predecessor):
                    return _stage_result("SNAPSHOT_NOT_CREATED", "PREVIOUS_STAGE_EVIDENCE_INVALID", context=context)
            if crypto_result.get("fixture_availability") != DATA["stage_evidence_chain_contract"]["future_crypto_fixture_marker"]:
                return _stage_result("SNAPSHOT_NOT_CREATED", "PREVIOUS_STAGE_EVIDENCE_INVALID", context=context)
            if not _current_edition_policy_consistent(envelope) or not _canonical_hashes_revalidated(envelope):
                return _stage_result("SNAPSHOT_NOT_CREATED", "DOCUMENT_IDENTITY_MISMATCH", context=context)
            return issue_snapshot(context, expiry_revocation_result)
        except Exception:
            return _stage_result("SNAPSHOT_NOT_CREATED", "PREVIOUS_STAGE_EVIDENCE_INVALID")

    def pipeline(raw_document, *, version_registry=DEFAULT_VERSION_REGISTRY, algorithms={"alg"}, keys={"key"}, current_utc="2026-01-01T00:00:00Z", revocation_registry=None):
        structure = raw_structure(raw_document, version_registry)
        document = parse_raw_json_without_duplicates(raw_document) if structure.get("stage_result") == "STRUCTURE_ACCEPTED" else None
        hashes = canonical_hashes(document or {}, structure)
        registry = registry_policy(document or {}, hashes, algorithms, keys)
        crypto = crypto_verify(document or {}, registry)
        expiry = expiry_stage(document or {}, crypto, current_utc, revocation_registry)
        snap = snapshot(document or {}, structure, hashes, registry, crypto, expiry)
        return {"structure": structure, "hashes": hashes, "registry": registry, "crypto": crypto, "expiry_revocation": expiry, "snapshot": snap}

    def future_pipeline(raw_document, *, version_registry=DEFAULT_VERSION_REGISTRY, algorithms={"alg"}, keys={"key"}, current_utc="2026-01-01T00:00:00Z", revocation_registry=None):
        structure = raw_structure(raw_document, version_registry)
        document = parse_raw_json_without_duplicates(raw_document) if structure.get("stage_result") == "STRUCTURE_ACCEPTED" else None
        hashes = canonical_hashes(document or {}, structure)
        registry = registry_policy(document or {}, hashes, algorithms, keys)
        crypto = future_crypto(registry) if registry.get("stage_result") == "REGISTRY_ACCEPTED" else crypto_verify(document or {}, registry)
        expiry = expiry_stage(document or {}, crypto, current_utc, revocation_registry)
        snap = snapshot(document or {}, structure, hashes, registry, crypto, expiry)
        return {"structure": structure, "hashes": hashes, "registry": registry, "crypto": crypto, "expiry_revocation": expiry, "snapshot": snap}

    globals().update({
        "_issued_chain_valid": issued_chain_valid,
        "_complete_clean_evidence": complete_clean_evidence,
        "_validate_parsed_document_structure": parsed_structure,
        "validate_raw_document_structure": raw_structure,
        "validate_canonical_hashes": canonical_hashes,
        "validate_registry_policy": registry_policy,
        "cryptographic_signature_verification": crypto_verify,
        "validate_expiry_revocation_stage": expiry_stage,
        "create_validated_snapshot": snapshot,
        "run_pipeline": pipeline,
        "_run_future_contract_pipeline": future_pipeline,
    })


_initialize_closure_local_interpreter()
globals().pop("_initialize_closure_local_interpreter", None)
globals().pop("_future_crypto_accepted_evidence", None)


def test_no_global_evidence_runtime_token_or_callable_issuer_attributes_escape():
    forbidden_global_names = {"_EVIDENCE_RUNTIME", "_make_evidence_runtime", "_FUTURE_CRYPTO_FIXTURE_TOKEN", "_future_crypto_accepted_evidence", "future_crypto", "issue_future_crypto", "_initialize_closure_local_interpreter"}
    assert forbidden_global_names.isdisjoint(globals())
    forbidden_callable_attrs = {
        "structure_validated",
        "hashes_validated",
        "registry_validated",
        "future_fixture_validated",
        "expiry_validated",
        "snapshot_validated",
        "seal",
        "issue",
        "issue_accepted",
    }
    for name, value in globals().items():
        if name.startswith("__"):
            continue
        for attr in forbidden_callable_attrs:
            assert not callable(getattr(value, attr, None)), (name, attr)
    envelope = build_envelope(_payload())
    context = _context_from_envelope(envelope)
    fabricated = [_fabricated_public_evidence("STRUCTURE_ACCEPTED", context, None)]
    fabricated.append(_fabricated_public_evidence("HASHES_ACCEPTED", context, fabricated[-1]["stage_evidence_id"]))
    fabricated.append(_fabricated_public_evidence("REGISTRY_ACCEPTED", context, fabricated[-1]["stage_evidence_id"]))
    fabricated.append(_fabricated_public_evidence("CRYPTOGRAPHIC_SIGNATURE_VERIFICATION_ACCEPTED", context, fabricated[-1]["stage_evidence_id"], "TEST_FIXTURE_FUTURE_ONLY"))
    fabricated.append(_fabricated_public_evidence("EXPIRY_REVOCATION_ACCEPTED", context, fabricated[-1]["stage_evidence_id"]))
    assert create_validated_snapshot(envelope, *fabricated)["stage_result"] == "SNAPSHOT_NOT_CREATED"


def test_markdown_describes_final_tokenless_closure_registry_model():
    markdown = MD.read_text()
    required_fragments = [
        "read-only `Mapping` composition object",
        "private copy of the public representation in",
        "`_public_fields`",
        "does not store an issuer token",
        "does not store a producer\nstage ID",
        "does not store a predecessor reference",
        "closure-local registry",
        "exact strong `evidence_ref`",
        "diagnostic `id(evidence)`",
        "`stage_id`",
        "`stage_result`",
        "full validation context",
        "exact `predecessor_ref`",
        "canonical `public_digest`",
        '`meta["evidence_ref"] is evidence`',
        "matching canonical public digest",
        "full recursive predecessor-chain validation",
        "inside the full\n`_run_future_contract_pipeline` closure",
        "`issue_future_crypto(context, registry_evidence)`",
        "No sentinel\ntoken, runtime authority object, or authority argument is involved",
        "`fixture_availability=TEST_FIXTURE_FUTURE_ONLY` remains only a diagnostic marker",
    ]
    for fragment in required_fragments:
        assert fragment in markdown
    forbidden_fragments = [
        "module-local sentinel token",
        "`_issuer_token`, `_producer_stage_id`, and `_predecessor_evidence_ref`",
        "Each accepted stage has its own private issuer token",
        "private future-fixture issuer token",
        "checks the private issuer token",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in markdown


def test_ast_nested_issuer_functions_do_not_accept_authority_arguments():
    tree = ast.parse(Path(__file__).read_text())
    forbidden = {"authority", "issuer_token", "construction_authority", "producer_stage_id", "fixture_token", "force_future_crypto"}
    found_issue_future_crypto = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [arg.arg for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
            if node.name == "issue_future_crypto":
                found_issue_future_crypto = True
                assert params == ["context", "registry_evidence"]
            if node.name.startswith(("issue", "seal")) or "issuer" in node.name:
                assert forbidden.isdisjoint(params), (node.name, params)
    assert found_issue_future_crypto


def test_strong_evidence_reference_and_no_numeric_id_only_attestation():
    source = Path(__file__).read_text()
    assert '"evidence_ref": evidence' in source
    assert 'meta["evidence_ref"] is not evidence' in source
    assert '"diagnostic_identity": id(evidence)' in source
    envelope = build_envelope(_payload())
    issued = list(_accepted_future_evidence_chain(envelope))
    assert create_validated_snapshot(envelope, *issued)["stage_result"] == "SNAPSHOT_CREATED"
    structure_clone = _IssuedStageEvidence(dict(issued[0]))
    assert structure_clone["stage_evidence_id"] == issued[0]["stage_evidence_id"]
    assert not _issued_chain_valid(structure_clone, _context_from_envelope(envelope))
    public_copy = dict(issued[1])
    public_copy["stage_evidence_id"] = _evidence_id(public_copy)
    assert public_copy["stage_evidence_id"] == issued[1]["stage_evidence_id"]
    assert create_validated_snapshot(envelope, issued[0], public_copy, issued[2], issued[3], issued[4])["stage_result"] == "SNAPSHOT_NOT_CREATED"
    deep = copy.deepcopy(issued[2]); deep["stage_evidence_id"] = _evidence_id(deep)
    roundtrip = json.loads(json.dumps(dict(issued[3]))); roundtrip["stage_evidence_id"] = _evidence_id(roundtrip)
    assert create_validated_snapshot(envelope, issued[0], issued[1], deep, issued[3], issued[4])["stage_result"] == "SNAPSHOT_NOT_CREATED"
    assert create_validated_snapshot(envelope, issued[0], issued[1], issued[2], roundtrip, issued[4])["stage_result"] == "SNAPSHOT_NOT_CREATED"
    tampered = issued[0]
    public = dict(tampered); public["stage_result"] = "STRUCTURE_REJECTED"; public["stage_evidence_id"] = _evidence_id(public)
    object.__setattr__(tampered, "_public_fields", public)
    assert not _issued_chain_valid(tampered, _context_from_envelope(envelope))
