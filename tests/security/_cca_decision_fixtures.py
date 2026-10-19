"""Synthetic TEST CCA bytes and keys, never installed authority or execution proof."""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bot_core import authenticated_custody_mutation_source as source
from bot_core import independent_cca_decision as cca
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.root_proof_issuer_substrate import ProviderRole, public_key_material_identity
from tests.security._authenticated_custody_source_fixtures import build_source_case


def b64(raw):
    return base64.b64encode(raw).decode("ascii")


def sign_cca_payload(case, payload=None, *, payload_raw=None, key=None, domain=None):
    """Sign test-only original fixture bytes; never imported by production code."""
    if payload_raw is None:
        payload_raw = canonical_json_bytes(case.payload if payload is None else payload)
    authority = (
        case.payload["completion_authority"] if payload is None else payload["completion_authority"]
    )
    return canonical_json_bytes(
        {
            "schema": cca.CCA_ENVELOPE_SCHEMA,
            "version": 1,
            "payload_b64": b64(payload_raw),
            "credential_id": authority["credential_identity"],
            "key_version": authority["key_version"],
            "signature_b64": b64(
                (case.private_key if key is None else key).sign(
                    (cca.CCA_DOMAIN if domain is None else domain) + payload_raw
                )
            ),
        }
    )


def build_cca_case(descriptor, journal, *, signing_key=None):
    """Build a valid synthetic protocol sample; all reports remain NOT_QUALIFIED."""
    source_case = build_source_case(descriptor, journal)
    private = signing_key or Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    source_policy = deepcopy(source_case.policy_document)
    completion = source_policy["completion_authority"]
    completion["public_key_b64"] = b64(public)
    completion["public_material_identity"] = public_key_material_identity(public)
    observation = {
        "reader_role": "test_claimed_independent_cca_reader",
        "reader_role_oid": 101,
        "observation_xid": "102",
        "qualified_catalog_sha256": source_case.payload["database"]["catalog_sha256"],
    }
    decision_id = "test-cca-decision-" + source_case.payload["close"]["operation_id"]
    position = {
        "namespace": completion["provider_namespace"],
        "sequence": 1,
        "previous_entry_sha256": "0" * 64,
        "entry_id": decision_id,
    }
    payload = {
        "schema": cca.CCA_SCHEMA,
        "version": 1,
        "domain": cca.CCA_DOMAIN[:-1].decode("ascii"),
        **{
            name: deepcopy(source_case.payload[name])
            for name in ("installation", "database", "custody", "close", "originals")
        },
        "completion_authority": deepcopy(completion),
        "close_observation": deepcopy(observation),
        "source_event_envelope_b64": b64(source_case.envelope),
        "source_event_envelope_sha256": hashlib.sha256(source_case.envelope).hexdigest(),
        "decision_id": decision_id,
        "decision_log_position": position,
        "source_authentication_result": "QUALIFIED_INSTALLED_SOURCE",
        "execution_result": "EXACT_LIFECYCLE_TRANSITION_COMPLETED",
        "durability_result": "SOURCE_AND_CCA_RECORDS_SYNCED",
        "completion_result": "AUTHENTICATED_CUSTODY_COMPLETION_RETAINED",
    }
    records = []
    for authority in (source_case.payload["executor"], completion):
        records.append(
            {
                name: authority[name]
                for name in (
                    "semantic_role",
                    "os_uid",
                    "os_gid",
                    "provider_namespace",
                    "credential_identity",
                    "public_material_identity",
                )
            }
            | {"supplementary_gids": []}
        )
    operation = json.loads(descriptor)
    participating_role = operation["semantic_role"]
    other_source_role = (
        source.HISTORY_SOURCE_ROLE
        if source_case.payload["executor"]["semantic_role"] == source.ROOT_SOURCE_ROLE
        else source.ROOT_SOURCE_ROLE
    )
    for role, uid in (
        (other_source_role, 1007),
        (ProviderRole.ROOT_PROOF_SIGNING.value, 1008),
        (ProviderRole.HISTORY_ATTESTATION_SIGNING.value, 1009),
    ):
        identity = (
            operation["credential_identity"]
            if role == participating_role
            else {
                "provider_namespace": "test-other-namespace-" + role,
                "credential_identity": "test-other-credential-" + role,
                "key_material_identity": public_key_material_identity(
                    Ed25519PrivateKey.generate()
                    .public_key()
                    .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
                ),
            }
        )
        records.append(
            {
                "semantic_role": role,
                "os_uid": uid,
                "os_gid": uid + 1000,
                "supplementary_gids": [],
                "provider_namespace": identity["provider_namespace"],
                "credential_identity": identity["credential_identity"],
                "public_material_identity": identity["key_material_identity"],
            }
        )
    policy_document = {
        "schema": cca.CCA_POLICY_SCHEMA,
        "version": 1,
        "source_verification_policy_b64": b64(canonical_json_bytes(source_policy)),
        "source_event_envelope_sha256": payload["source_event_envelope_sha256"],
        "completion_authority": deepcopy(completion),
        "close_observation": deepcopy(observation),
        "decision_id": decision_id,
        "decision_log_position": deepcopy(position),
        "principal_inventory": records,
    }
    case = SimpleNamespace(
        payload=payload,
        source_case=source_case,
        source_policy=source_policy,
        policy_document=policy_document,
        private_key=private,
    )
    case.payload_bytes = canonical_json_bytes(payload)
    case.envelope = sign_cca_payload(case)
    case.decision = cca.parse_independent_cca_decision(case.envelope)
    case.policy = cca.parse_cca_verification_policy(canonical_json_bytes(policy_document))
    case.immutable_binding = canonical_json_bytes(
        {
            name: value
            for name, value in payload.items()
            if name not in {"decision_id", "decision_log_position"}
        }
    )
    return case
