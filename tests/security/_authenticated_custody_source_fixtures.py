"""Synthetic test-only source originals, policy and signing key.

The close and installed identities are explicit fixture claims, never qualified
PostgreSQL or installed-authority evidence. No production module imports this
helper and its test signatures do not attest execution or prior durability.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bot_core import authenticated_custody_mutation_source as source
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.root_proof_issuer_substrate import public_key_material_identity


def _b64(raw):
    return base64.b64encode(raw).decode("ascii")


def _set_original(document, name, raw):
    document[name + "_b64"] = _b64(raw)
    document[name + "_sha256"] = hashlib.sha256(raw).hexdigest()


def build_source_case(descriptor, journal, *, signing_key=None):
    """Make isolated fixture bytes; synthetic close remains NOT_QUALIFIED."""
    operation, record = json.loads(descriptor), json.loads(journal)
    before, after = record["before_metadata"], record["after_metadata"]
    private = signing_key or Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    role = (
        source.ROOT_SOURCE_ROLE
        if operation["semantic_role"] == "ROOT_PROOF_SIGNING"
        else source.HISTORY_SOURCE_ROLE
    )
    executor = {
        "authority_id": "test-source-authority-" + role,
        "service_id": "test-source-service-" + role,
        "host_id": "test-isolated-linux-host",
        "os_uid": 1005,
        "os_gid": 2005,
        "endpoint": "/run/test-custody-source/" + role + ".sock",
        "provider_namespace": "test-source-namespace-" + role,
        "credential_identity": "test-source-credential-" + role,
        "semantic_role": role,
        "public_key_b64": _b64(public),
        "public_material_identity": public_key_material_identity(public),
        "key_version": 1,
        "lifecycle_generation": 1,
        "lifecycle_state": "ACTIVE",
        "trust_epoch": 1,
    }
    completion_public = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    )
    completion = {
        "authority_id": "test-completion-authority",
        "service_id": "test-completion-service",
        "host_id": "test-isolated-linux-host",
        "os_uid": 1006,
        "os_gid": 2006,
        "endpoint": "/run/test-completion-authority/authority.sock",
        "provider_namespace": "test-completion-namespace",
        "credential_identity": "test-completion-credential",
        "semantic_role": "CUSTODY_COMPLETION_ATTESTATION_V1",
        "public_key_b64": _b64(completion_public),
        "public_material_identity": public_key_material_identity(completion_public),
        "key_version": 1,
        "lifecycle_generation": 1,
        "lifecycle_state": "ACTIVE",
        "trust_epoch": 1,
    }
    installation = {
        "installation_id": operation["database_identity"]["instance_id"],
        "environment_id": "isolated-authenticated-source-test",
        "profile": "PRODUCTION_LOCAL",
        "trust_domain": before["trust_domain"],
        "installation_sha256": "a" * 64,
        "trust_pin_id": "test-independent-installation-pin",
        "trust_pin_version": 1,
        "trust_pin_sha256": "b" * 64,
    }
    database = {
        **operation["database_identity"],
        "authority_identity": "test-claimed-production-lifecycle-authority",
        "schema_name": "test_claimed_lifecycle_schema",
        "schema_version": 1,
        "catalog_sha256": "c" * 64,
    }
    request = {
        "operation_id": operation["operation_id"],
        "semantic_role": operation["semantic_role"],
        "identity": before["credential_id"],
        "generation": operation["lifecycle_generation"],
        "predecessor_state": operation["predecessor_state"],
        "target_state": operation["target_state"],
    }
    # Preserve the separate noncanonical PostgreSQL16 jsonb-text representation.
    close = {
        "operation_id": operation["operation_id"],
        "request_format": "PostgreSQL16/jsonb-text/UTF8",
        "close_xid": "100",
        "phase": "PENDING_CLOSED",
    }
    _set_original(close, "request", json.dumps(request, sort_keys=True).encode("utf-8"))
    observation = {
        "reader_role": "test_claimed_read_only_source_reader",
        "reader_role_oid": 100,
        "observation_xid": "101",
        "qualified_catalog_sha256": database["catalog_sha256"],
    }
    originals = {}
    for name, raw in (
        ("operation_descriptor", descriptor),
        ("complete_journal", journal),
        ("before_metadata", canonical_json_bytes(before)),
        ("after_metadata", canonical_json_bytes(after)),
    ):
        _set_original(originals, name, raw)
    execution_id = "test-execution-" + operation["operation_id"]
    position = {
        "namespace": executor["provider_namespace"],
        "sequence": 1,
        "previous_entry_sha256": "0" * 64,
        "entry_id": execution_id,
    }
    payload = {
        "schema": source.SOURCE_SCHEMA,
        "version": 1,
        "domain": source.SOURCE_DOMAIN[:-1].decode("ascii"),
        "installation": installation,
        "executor": executor,
        "database": database,
        "custody": {
            "semantic_role": operation["semantic_role"],
            "authority_identity": before["provider_namespace"],
            "credential_identity": before["credential_id"],
            "key_handle_or_version": operation["credential_identity"]["key_handle_or_version"],
            "custody_lifecycle_namespace": before["lifecycle_namespace"],
            "public_key_b64": before["public_key_b64"],
            "public_material_identity": operation["credential_identity"]["key_material_identity"],
            "key_version": before["key_version"],
            "predecessor_generation": before["lifecycle_generation"],
            "successor_generation": after["lifecycle_generation"],
            "predecessor_state": before["lifecycle_state"],
            "target_state": after["lifecycle_state"],
        },
        "close": close,
        "close_observation": observation,
        "originals": originals,
        "execution_id": execution_id,
        "source_log_position": position,
        "execution_result": "EXACT_LIFECYCLE_TRANSITION_COMPLETED",
        "durability_result": "METADATA_AND_COMPLETE_SYNCED",
    }
    policy_document = deepcopy(
        {
            "schema": source.SOURCE_POLICY_SCHEMA,
            "version": 1,
            "installation": installation,
            "executor": executor,
            "database": database,
            "completion_authority": completion,
            "operation_descriptor_b64": _b64(descriptor),
            "close": close,
            "close_observation": observation,
            "execution_id": execution_id,
            "source_log_position": position,
        }
    )
    payload_bytes = canonical_json_bytes(payload)
    envelope = canonical_json_bytes(
        {
            "schema": source.SOURCE_ENVELOPE_SCHEMA,
            "version": 1,
            "payload_b64": _b64(payload_bytes),
            "credential_id": executor["credential_identity"],
            "key_version": executor["key_version"],
            "signature_b64": _b64(private.sign(source.SOURCE_DOMAIN + payload_bytes)),
        }
    )
    return SimpleNamespace(
        payload=payload,
        payload_bytes=payload_bytes,
        policy_document=policy_document,
        private_key=private,
        envelope=envelope,
        event=source.parse_custody_mutation_execution(envelope),
        policy=source.parse_source_verification_policy(canonical_json_bytes(policy_document)),
    )
