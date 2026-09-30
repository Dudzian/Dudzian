"""Executable freeze guard for the Stage 9 external provisioning architecture."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
CONTRACT_PATH = DOCS / "stage9_external_provisioning_architecture_contract.json"
FREEZE_PATH = DOCS / "stage9_external_provisioning_architecture_freeze.json"
REQUIRED_COVERAGE = {
    "account_id authority ownership",
    "first-account/first-device ordering",
    "provisioning_subject_id",
    "operation identity",
    "provisioning membership",
    "PDSA trust domain",
    "protected freshness ownership",
    "secret resource ownership",
    "post-install ordering",
    "Windows handoff contract",
    "production material dependency",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _validate(value: dict[str, Any]) -> None:
    assert value["status"] == "FROZEN"
    assert value["implementation_authorized"] is True
    assert value["production_provisioning_ready"] is False
    assert value["stage_status"] == {
        "stage_9": "IN_PROGRESS",
        "windows_0_14": "9/15 DONE = 60.0%",
        "windows_production_ready": "NOT_READY",
        "stage_10": "NOT_STARTED",
        "windows_tpm_activation_bridge": "LIVE_PASS",
        "production_root_material": "PROVISIONED_LOCALLY",
        "production_ceremony": "NOT_STARTED",
    }

    owners = value["authority_ownership"]
    assert owners["account_genesis"] == {
        "owner": "CryptoHunterAccountAuthority",
        "owner_count": 1,
        "role": "FINAL_ACCOUNT_GENESIS_DECISION_OWNER",
        "pdsa_is_owner": False,
        "lppi_is_owner": False,
    }
    assert owners["account_id"]["mint_and_reservation_owner"] == "CryptoHunterAccountAuthority"
    assert owners["account_id"]["owner_count"] == 1
    assert owners["provisioning"]["is_account_genesis_owner"] is False

    ordering = value["atomic_account_plus_first_device"]
    assert ordering["ordering"] == "ATOMIC_ACCOUNT_PLUS_FIRST_DEVICE"
    assert ordering["atomicity"] == "LOGICAL_PROTOCOL_NOT_DISTRIBUTED_SQL_TRANSACTION"
    assert ordering["durable_progression"] == [
        "ISSUED",
        "ENROLLMENT_ACCEPTED",
        "PREPARED",
        "ACCOUNT_COMMITTED",
        "FIRST_DEVICE_COMMITTED",
        "MEMBERSHIP_COMMITTED",
        "MATERIALIZED",
        "CONSUMED",
    ]
    assert ordering["second_operation_after_winner"] == "REJECT"

    subject = value["provisioning_subject_id"]
    assert subject["owner_issuer"] == "ProductDeploymentSecurityAuthority"
    assert subject["caller_selectable"] is False
    assert subject["stable_across_retry"] is subject["stable_across_restart"] is True
    assert subject["purpose_domain"] and subject["canonical_encoding"]
    assert "timestamp component" in subject["issuance"]
    assert "CSPRNG" in subject["issuance"]
    assert "durably reserves" in subject["issuance"]

    request = value["pre_enrollment_request"]
    assert request["artifact"] == "PreEnrollmentRequestV1"
    assert "TPMEnrollmentRequestV1" in request["relationship_to_existing_contracts"]
    assert request["canonical_digest"] == {
        "field": "pre_enrollment_request_digest_sha256",
        "derivation": "SHA256(exact RFC8785 JCS UTF-8 canonical payload bytes)",
    }
    required_request_bindings = {
        "pdsa_challenge_id",
        "pdsa_challenge_digest_sha256",
        "pdsa_challenge_nonce_digest_sha256",
        "tpm_enrollment_request_digest_sha256",
        "tpm_enrollment_challenge_digest_sha256",
        "tpm_enrollment_response_digest_sha256",
        "verified_tpm_exchange_reference",
        "verified_tpm_public_projection_id",
        "ek_public_digest",
        "ak_public_digest",
        "tpm_attestation_evidence_reference",
        "pre_enrollment_public_key_algorithm_profile",
        "pre_enrollment_public_key_canonical_bytes",
        "pre_enrollment_public_key_fingerprint_sha256",
        "release_policy_digest_sha256",
        "release_policy_generation",
        "request_nonce_hex",
    }
    assert required_request_bindings <= set(request["canonical_payload_fields"])
    assert request["authentication"]["mutable_device_fields_after_authentication"] is False
    request_key = request["pre_enrollment_key"]
    assert request_key["algorithm_profile"] == "ECDSA-P256-SHA256"
    assert (request_key["algorithm"], request_key["curve"], request_key["hash"]) == (
        "ECDSA",
        "NIST_P256",
        "SHA256",
    )
    assert request_key["public_key_encoding"] == "SEC1_UNCOMPRESSED_P256_65_BYTES"
    assert request_key["public_key_bytes"] == "0x04 || X32 || Y32"
    assert request_key["fingerprint_derivation"] == (
        "lowercase hex SHA256(exact 65 canonical SEC1 uncompressed public-key bytes)"
    )
    assert request_key["fingerprint_length_and_alphabet"] == (
        "exactly 64 lowercase hexadecimal characters"
    )
    assert request_key["accepted_alternative_algorithms"] == []
    assert request_key["forbidden_fingerprint_inputs"] == [
        "SubjectPublicKeyInfo DER",
        "CNG ECCPUBLIC_BLOB",
        "PEM",
        "JSON",
        "base64",
        "TPMT_PUBLIC",
    ]
    assert request_key["custody_profile"] == (
        "WINDOWS_PLATFORM_CRYPTO_PROVIDER_TPM_ECDSA_P256_PRE_ENROLLMENT_V1"
    )
    assert request_key["custody_requirements"]["private_key_exportable"] is False
    assert request_key["custody_requirements"]["private_key_in_package"] is False
    request_signature = request["authentication"]["request_signature_profile"]
    assert request_signature["algorithm"] == "ECDSA-P256-SHA256"
    assert request_signature["domain"] == "CryptoHunter.Stage9.PreEnrollmentRequest.v1"
    assert request_signature["verification_key"].startswith(
        "exact pre_enrollment_public_key_canonical_bytes"
    )
    assert request_signature["signature_encoding"] == (
        "strict minimal ASN.1 DER SEQUENCE(INTEGER r, INTEGER low-S s)"
    )

    package = value["pdsa_enrollment_authorization_package"]
    assert package["issuer"] == "ProductDeploymentSecurityAuthority"
    assert package["contains_provisioning_operation_id"] is False
    assert package["invariant"] == (
        "PDSA_ENROLLMENT_PACKAGE_CONTAINS_PROVISIONING_OPERATION_ID = FALSE"
    )
    assert package["canonical_payload_fields"] == [
        "schema_version",
        "environment",
        "pdsa_trust_domain",
        "provisioning_subject_id",
        "enrollment_reference",
        "pdsa_challenge_id",
        "pdsa_challenge_digest_sha256",
        "pre_enrollment_request_digest_sha256",
        "verified_tpm_exchange_reference",
        "verified_tpm_public_projection_id",
        "target_tpm_ek_public_digest",
        "target_tpm_ak_public_digest",
        "pre_enrollment_public_key_algorithm_profile",
        "pre_enrollment_public_key_fingerprint_sha256",
        "authorization_generation",
        "authorization_version",
        "issued_at_utc",
        "expires_at_utc",
        "release_policy_digest_sha256",
        "release_policy_generation",
        "product_profile",
        "predecessor_package_digest_or_null",
        "lineage_generation",
    ]
    assert "provisioning_operation_id" not in package["canonical_payload_fields"]
    assert package["signature_profile"] == {
        "algorithm": "Ed25519",
        "threshold": "2_OF_3_PDSA_KEYS",
        "signed_bytes": (
            'UTF8("CryptoHunter.Stage9.PDSAEnrollmentAuthorization.v1") || 0x00 || '
            "SHA256(JCS_UTF8(canonical_payload))"
        ),
        "key_source": "verified production release-policy PDSA key set only",
    }
    retained = package["retained_issuance_record"]
    assert retained["same_request_retry"] == "RETURN_EXACT_SAME_PACKAGE_BYTES"
    assert retained["same_challenge_different_request"] == ("REJECT_CHALLENGE_REPLAY_CONFLICT")
    assert {
        "pdsa_challenge_id",
        "pdsa_challenge_digest_sha256",
        "pre_enrollment_request_digest_sha256",
        "pdsa_package_digest_sha256",
        "provisioning_subject_id",
        "enrollment_reference",
    } == set(retained["fields"])
    comparisons = package["request_and_device_binding"]["exact_target_comparisons"]
    assert [item["package_field"] for item in comparisons] == [
        "pre_enrollment_request_digest_sha256",
        "pdsa_challenge_id",
        "pdsa_challenge_digest_sha256",
        "verified_tpm_exchange_reference",
        "verified_tpm_public_projection_id",
        "target_tpm_ek_public_digest",
        "target_tpm_ak_public_digest",
        "pre_enrollment_public_key_fingerprint_sha256",
    ]
    assert package["request_and_device_binding"]["required_live_proofs"] == [
        "TPM_ATTESTATION_EXCHANGE_REVERIFIED",
        "PRE_ENROLLMENT_REQUEST_KEY_PROOF_OF_POSSESSION_VERIFIED",
    ]

    lifecycle = value["lppi_authority_key_lifecycle"]
    assert lifecycle["selection"] == "SUCCESSOR_KEY_AFTER_PACKAGE_VERIFICATION"
    authority_binding = lifecycle["binding_artifact"]
    assert authority_binding["artifact"] == "LPPIAuthorityKeyBindingV1"
    assert {
        "pdsa_package_digest_sha256",
        "pre_enrollment_request_digest_sha256",
        "pre_enrollment_public_key_fingerprint_sha256",
        "lppi_authority_public_key_fingerprint_sha256",
        "custody_profile",
        "lppi_authority_key_tpm_name",
        "lppi_authority_tpmt_public_sha256",
        "verified_tpm_public_projection_id",
        "verified_tpm_exchange_reference",
        "provisioning_subject_id",
        "enrollment_reference",
        "generation",
    } <= set(authority_binding["canonical_payload_fields"])
    profile = lifecycle["authority_key_profile"]
    assert profile["invariant"] == "LPPI_AUTHORITY_KEY_ALGORITHM = ECDSA-P256-SHA256"
    assert profile["algorithm"] == "ECDSA-P256-SHA256"
    assert profile["accepted_alternative_algorithms"] == []
    assert profile["public_key_encoding"] == (
        "canonical TPMT_PUBLIC bytes for the TPM-backed CNG key"
    )
    assert "strict minimal ASN.1 DER" in profile["signature_encoding"]
    custody = lifecycle["custody"]
    assert custody["custody_profile_enum"] == ["WINDOWS_PLATFORM_CRYPTO_PROVIDER_TPM_ECDSA_P256_V1"]
    assert custody["selected_custody_profile"] == custody["custody_profile_enum"][0]
    assert authority_binding["required_values"] == {
        "lppi_authority_public_key_algorithm_profile": "ECDSA-P256-SHA256",
        "custody_profile": "WINDOWS_PLATFORM_CRYPTO_PROVIDER_TPM_ECDSA_P256_V1",
        "cng_provider_name": "Microsoft Platform Crypto Provider",
    }
    continuity = authority_binding["continuity_signature_profile"]
    assert continuity["algorithm"] == "ECDSA-P256-SHA256"
    assert continuity["domain"] == "CryptoHunter.Stage9.LPPIAuthorityKeyContinuity.v1"
    assert continuity["signing_key_fingerprint_source"].endswith(
        ".pre_enrollment_public_key_fingerprint_sha256"
    )
    assert "strict minimal ASN.1 DER" in continuity["signature_encoding"]
    pop = authority_binding["authority_key_pop_profile"]
    assert pop["algorithm"] == "ECDSA-P256-SHA256"
    assert pop["domain"] == "CryptoHunter.Stage9.LPPIAuthorityKeyPoP.v1"
    assert pop["signing_key_fingerprint_source"].endswith(
        ".lppi_authority_public_key_fingerprint_sha256"
    )
    assert pop["challenge_fields"] == [
        "lppi_authority_key_binding_digest_sha256",
        "pdsa_package_digest_sha256",
        "pre_enrollment_request_digest_sha256",
        "lppi_authority_public_key_fingerprint_sha256",
        "verified_tpm_public_projection_id",
        "verified_tpm_exchange_reference",
    ]
    assert authority_binding["status_progression"] == [
        "CANDIDATE",
        "CONTINUITY_SIGNATURE_VERIFIED",
        "AUTHORITY_KEY_POP_VERIFIED",
        "CUSTODY_EVIDENCE_VERIFIED",
        "VERIFIED_CONTINUITY",
        "ACTIVE",
    ]
    assert lifecycle["trust_gate"].startswith(
        "ProvisioningMembershipBinding and LPPIAuthenticatedProvisioningOperationBinding"
    )

    operation = value["operation_identity"]
    assert operation["status"] == "FROZEN"
    assert (
        operation["provisioning_operation_id"]["owner_issuer"] == "LocalProductProvisioningIssuer"
    )
    assert operation["provisioning_operation_id"]["caller_selectable"] is False
    assert operation["cha_logical_operation_id"]["owner_issuer"] == "CryptoHunterAccountAuthority"
    assert operation["mapping"]["cardinality"] == "ONE_TO_ONE_BIJECTION"
    assert operation["mapping"]["same_operation_replay"] == "RETURN_SAME_RESULT_OR_DURABLY_RESUME"
    assert operation["lppi_authenticated_operation_binding"]["canonical_payload_fields"] == [
        "schema_version",
        "environment",
        "pdsa_trust_domain",
        "pdsa_package_digest_sha256",
        "provisioning_subject_id",
        "enrollment_reference",
        "provisioning_operation_id",
        "binding_generation",
        "created_at_utc",
    ]
    operation_signature = operation["lppi_authenticated_operation_binding"]["signature_profile"]
    assert operation_signature["algorithm"] == profile["algorithm"]
    assert operation_signature["domain"] == (
        "CryptoHunter.Stage9.LPPIAuthenticatedProvisioningOperationBinding.v1"
    )
    assert "strict minimal ASN.1 DER" in operation_signature["signature_encoding"]
    assert (
        operation_signature["active_key_match_fields"]
        == lifecycle["active_key_record"]["consumer_match_fields"]
    )
    assert operation["protocol_ordering"] == [
        "PDSA_CHALLENGE_ISSUED",
        "TPM_BOUND_PRE_ENROLLMENT_REQUEST_AUTHENTICATED",
        "PDSA_CHALLENGE_AND_REQUEST_VERIFIED",
        "PDSA_CHALLENGE_CONSUMED_WITH_REQUEST_DIGEST",
        "PDSA_DEVICE_REQUEST_BOUND_PACKAGE_SIGNED",
        "PDSA_PACKAGE_VERIFIED_ON_EXACT_TARGET",
        "LPPI_AUTHORITY_KEY_BINDING_VERIFIED",
        "LPPI_PRVOP_DURABLY_RESERVED",
        "LPPI_AUTHENTICATED_OPERATION_BINDING_COMMITTED",
        "CHA_AGO_ASSIGNED",
        "PRVOP_AGO_BIJECTION_COMMITTED",
    ]
    package_requires_prvop = (
        package["contains_provisioning_operation_id"]
        or "provisioning_operation_id" in package["canonical_payload_fields"]
    )
    lppi_issues_after_package_verification = operation["protocol_ordering"].index(
        "PDSA_PACKAGE_VERIFIED_ON_EXACT_TARGET"
    ) < operation["protocol_ordering"].index("LPPI_PRVOP_DURABLY_RESERVED")
    assert not (package_requires_prvop and lppi_issues_after_package_verification)

    membership = value["provisioning_membership"]
    assert membership["authentication"]["algorithm"] == profile["algorithm"]
    assert membership["authentication"]["domain"] == (
        "CryptoHunter.Stage9.ProvisioningMembershipBinding.v1"
    )
    assert "strict minimal ASN.1 DER" in membership["authentication"]["signature_encoding"]
    assert (
        membership["authentication"]["active_key_match_fields"]
        == lifecycle["active_key_record"]["consumer_match_fields"]
    )
    assert membership["authentication"]["sha256_alone_is_authentication"] is False
    assert {
        "account_id",
        "device_installation_id",
        "provisioning_subject_id",
        "provisioning_operation_id",
        "logical_operation_id",
    } <= set(membership["canonical_payload"])
    for field in (
        "registry",
        "durability",
        "lookup",
        "expiration",
        "revocation",
        "rollback_resistance",
    ):
        assert membership[field]

    pdsa = value["pdsa_trust_domain"]
    assert (pdsa["threshold"], pdsa["key_count"], pdsa["algorithm"]) == (2, 3, "Ed25519")
    assert pdsa["production_accepts_test_or_dev_keys"] is False
    assert pdsa["substitute_production_keys_forbidden"] is True

    protected = value["protected_freshness_authority"]
    secret = value["secret_resource_authority"]
    assert protected["port"] == "ProtectedFreshnessAuthorityPort"
    assert protected["separate_from_provisioning"] is True
    assert protected["tpm_nv"]["hierarchy"] == "TPM_RH_OWNER"
    assert protected["tpm_nv"]["definition_hierarchy"]["value"] == "TPM_RH_OWNER"
    assert protected["tpm_nv"]["index_type"] == "TPMA_NV_COUNTER"
    expected_attributes = {
        "TPMA_NV_COUNTER": "SET",
        "TPMA_NV_POLICYWRITE": "SET",
        "TPMA_NV_POLICYREAD": "CLEAR",
        "TPMA_NV_OWNERWRITE": "CLEAR",
        "TPMA_NV_OWNERREAD": "CLEAR",
        "TPMA_NV_AUTHWRITE": "CLEAR",
        "TPMA_NV_AUTHREAD": "SET",
        "TPMA_NV_NO_DA": "SET",
        "TPMA_NV_PLATFORMCREATE": "CLEAR",
    }
    attributes = protected["tpm_nv"]["attributes"]
    assert {name: setting["state"] for name, setting in attributes.items()} == expected_attributes
    assert all(setting["reason"] for setting in attributes.values())
    write = protected["tpm_nv"]["write_authorization"]
    assert write["command"] == "TPM2_NV_Increment"
    assert write["required_path"] == "TPMA_NV_POLICYWRITE with the exact frozen authPolicy"
    assert write["owner_auth_bypass"] is False
    assert write["auth_value_or_hmac_bypass"] is False
    read = protected["tpm_nv"]["read_authorization"]
    assert read["command"] == "TPM2_NV_Read"
    assert read["required_path"] == (
        "TPMA_NV_AUTHREAD with the frozen empty NV authValue using an ordinary password session"
    )
    assert read["less_restricted_than_increment"] is True
    assert read["may_authorize_increment"] is False
    assert {"TPM_RH_PLATFORM", "TPMA_NV_PLATFORMCREATE", "TPM_CLEAR"} <= set(
        protected["tpm_nv"]["forbidden"]
    )
    assert secret["port"] == "SecretExternalResourcePort"
    assert secret["separate_from_provisioning"] is True
    assert secret["test_secret_port_is_production_provider"] is False
    assert set(secret["operations"]) == {"begin", "reconcile", "cleanup"}

    windows = value["windows_post_install"]
    assert windows["mode"] == "POST_INSTALL_FIRST_RUN_ENROLLMENT"
    assert windows["backend_before_accepted_terminal_transition"] == "DEMAND_START_STOPPED"
    assert windows["corehost_before_accepted_provisioning_transition"] == "MUST_NOT_START"
    assert {
        "MINT_ACCOUNT_IDENTITY",
        "MINT_PROVISIONING_MEMBERSHIP",
        "SYNTHESIZE_FIRST_RUN_BOOTSTRAP_CLAIM",
        "BECOME_ROOT_AUTHORITY",
    } == set(windows["msi_must_not"])

    handoff = value["windows_handoff"]
    assert handoff["missing_real_ceremony_package"] == "WindowsProvisioningAdapterUnavailable"
    assert handoff["silent_fallback"] is False
    assert handoff["test_or_dev_trust_in_production"] is False
    assert "supply independent ProtectedFreshnessAuthorityPort" in handoff["production_contract"]
    assert "supply independent SecretExternalResourcePort" in handoff["production_contract"]

    material = value["production_material_dependency"]
    assert material["ceremony_performed"] is False
    assert material["absence_behavior"] == "FAIL_CLOSED"
    assert material["production_activation_authorized"] is False
    assert material["message"] == "PRODUCTION MATERIAL REQUIRED — DO NOT GENERATE SUBSTITUTE KEYS."
    assert not any(item["current_value"] == "DESIGN_BLOCKED" for item in value["supersession"])


def test_frozen_topology_and_all_negative_guards() -> None:
    _validate(_load(CONTRACT_PATH))


def test_pdsa_to_lppi_to_cha_protocol_has_executable_acyclic_ordering() -> None:
    value = _load(CONTRACT_PATH)
    package = value["pdsa_enrollment_authorization_package"]
    operation = value["operation_identity"]
    assert package["contains_provisioning_operation_id"] is False
    assert "provisioning_operation_id" not in package["canonical_payload_fields"]
    assert operation["protocol_ordering"] == [
        "PDSA_CHALLENGE_ISSUED",
        "TPM_BOUND_PRE_ENROLLMENT_REQUEST_AUTHENTICATED",
        "PDSA_CHALLENGE_AND_REQUEST_VERIFIED",
        "PDSA_CHALLENGE_CONSUMED_WITH_REQUEST_DIGEST",
        "PDSA_DEVICE_REQUEST_BOUND_PACKAGE_SIGNED",
        "PDSA_PACKAGE_VERIFIED_ON_EXACT_TARGET",
        "LPPI_AUTHORITY_KEY_BINDING_VERIFIED",
        "LPPI_PRVOP_DURABLY_RESERVED",
        "LPPI_AUTHENTICATED_OPERATION_BINDING_COMMITTED",
        "CHA_AGO_ASSIGNED",
        "PRVOP_AGO_BIJECTION_COMMITTED",
    ]
    assert operation["mapping"]["precondition"] == (
        "device/request-bound PDSA package and LPPIAuthorityKeyBindingV1 are verified, then LPPI "
        "authenticated operation binding is committed before CHA assignment"
    )


def test_cycle_is_rejected_when_pdsa_package_requires_later_lppi_prvop() -> None:
    value = copy.deepcopy(_load(CONTRACT_PATH))
    value["pdsa_enrollment_authorization_package"]["canonical_payload_fields"].append(
        "provisioning_operation_id"
    )
    with pytest.raises(AssertionError):
        _validate(value)


def _verify_package_target(
    contract: dict[str, Any],
    package: dict[str, str],
    request: dict[str, str],
    enrolling_host: dict[str, str],
) -> None:
    """Executable projection of the frozen exact-target acceptance relation."""
    binding = contract["pdsa_enrollment_authorization_package"]["request_and_device_binding"]
    for comparison in binding["exact_target_comparisons"]:
        request_value = request[comparison["request_field"]]
        assert package[comparison["package_field"]] == request_value
        if comparison["host_field"] is not None:
            assert enrolling_host[comparison["host_field"]] == request_value
    assert set(enrolling_host["verified_live_proofs"]) == set(binding["required_live_proofs"])


def test_package_cloning_to_other_tpm_and_request_key_is_rejected() -> None:
    contract = _load(CONTRACT_PATH)
    request_a = {
        "request_digest": "request-a",
        "challenge_id": "challenge-a",
        "challenge_digest": "challenge-digest-a",
        "exchange_reference": "exchange-a",
        "projection_id": "tpm-a",
        "ek_digest": "ek-a",
        "ak_digest": "ak-a",
        "key_fingerprint": "request-key-a",
    }
    package_a = {
        "pre_enrollment_request_digest_sha256": "request-a",
        "pdsa_challenge_id": "challenge-a",
        "pdsa_challenge_digest_sha256": "challenge-digest-a",
        "verified_tpm_exchange_reference": "exchange-a",
        "verified_tpm_public_projection_id": "tpm-a",
        "target_tpm_ek_public_digest": "ek-a",
        "target_tpm_ak_public_digest": "ak-a",
        "pre_enrollment_public_key_fingerprint_sha256": "request-key-a",
    }
    host_a = {
        "projection_id": "tpm-a",
        "ek_digest": "ek-a",
        "ak_digest": "ak-a",
        "key_fingerprint": "request-key-a",
        "verified_live_proofs": [
            "TPM_ATTESTATION_EXCHANGE_REVERIFIED",
            "PRE_ENROLLMENT_REQUEST_KEY_PROOF_OF_POSSESSION_VERIFIED",
        ],
    }
    _verify_package_target(contract, package_a, request_a, host_a)

    host_b = {
        "projection_id": "tpm-b",
        "ek_digest": "ek-b",
        "ak_digest": "ak-b",
        "key_fingerprint": "request-key-b",
        "verified_live_proofs": [
            "TPM_ATTESTATION_EXCHANGE_REVERIFIED",
            "PRE_ENROLLMENT_REQUEST_KEY_PROOF_OF_POSSESSION_VERIFIED",
        ],
    }
    with pytest.raises(AssertionError):
        _verify_package_target(contract, package_a, request_a, host_b)


P256_PRIME = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
P256_B = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
P256_ORDER = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
P256_GENERATOR_SEC1 = bytes.fromhex(
    "04"
    "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296"
    "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5"
)


def _canonical_pre_enrollment_key_fingerprint(canonical_hex: str) -> str:
    assert re.fullmatch(r"[0-9a-f]{130}", canonical_hex)
    key = bytes.fromhex(canonical_hex)
    assert len(key) == 65 and key[0] == 0x04
    x, y = int.from_bytes(key[1:33], "big"), int.from_bytes(key[33:], "big")
    assert x < P256_PRIME and y < P256_PRIME
    assert (pow(y, 2, P256_PRIME) - (pow(x, 3, P256_PRIME) - 3 * x + P256_B)) % P256_PRIME == 0
    return hashlib.sha256(key).hexdigest()


def _verify_strict_low_s_p256_der(signature: bytes) -> None:
    assert len(signature) >= 8 and signature[0] == 0x30
    assert signature[1] < 0x80 and signature[1] == len(signature) - 2
    offset = 2
    values: list[int] = []
    for _ in range(2):
        assert signature[offset] == 0x02
        length = signature[offset + 1]
        assert 0 < length < 0x80
        encoded = signature[offset + 2 : offset + 2 + length]
        assert len(encoded) == length
        assert encoded[0] & 0x80 == 0
        assert not (length > 1 and encoded[0] == 0 and encoded[1] & 0x80 == 0)
        values.append(int.from_bytes(encoded, "big"))
        offset += 2 + length
    assert offset == len(signature)
    r, s = values
    assert 1 <= r < P256_ORDER
    assert 1 <= s <= P256_ORDER // 2


def _verify_pre_enrollment_key_relationship(
    request: dict[str, str], package: dict[str, str], continuity: dict[str, str]
) -> None:
    fingerprint = _canonical_pre_enrollment_key_fingerprint(
        request["pre_enrollment_public_key_canonical_bytes"]
    )
    assert request["pre_enrollment_public_key_algorithm_profile"] == "ECDSA-P256-SHA256"
    assert request["pre_enrollment_public_key_fingerprint_sha256"] == fingerprint
    assert package["pre_enrollment_public_key_fingerprint_sha256"] == fingerprint
    assert continuity["continuity_signer_fingerprint"] == fingerprint
    _verify_strict_low_s_p256_der(bytes.fromhex(request["request_signature_der_hex"]))


def test_canonical_pre_enrollment_key_fingerprint_and_signature_relation() -> None:
    canonical_hex = P256_GENERATOR_SEC1.hex()
    fingerprint = hashlib.sha256(P256_GENERATOR_SEC1).hexdigest()
    request = {
        "pre_enrollment_public_key_algorithm_profile": "ECDSA-P256-SHA256",
        "pre_enrollment_public_key_canonical_bytes": canonical_hex,
        "pre_enrollment_public_key_fingerprint_sha256": fingerprint,
        "request_signature_der_hex": "3006020101020101",
    }
    package = {"pre_enrollment_public_key_fingerprint_sha256": fingerprint}
    continuity = {"continuity_signer_fingerprint": fingerprint}
    _verify_pre_enrollment_key_relationship(request, package, continuity)


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("fingerprint", "0" * 64),
        ("spki", None),
        ("algorithm", "Ed25519"),
        ("short_key", "04" + "00" * 63),
        ("wrong_prefix", "03" + P256_GENERATOR_SEC1.hex()[2:]),
        ("non_minimal_der", "300702020001020101"),
        ("high_s", None),
    ],
)
def test_noncanonical_pre_enrollment_key_or_signature_is_rejected(
    mutation: str, value: str | None
) -> None:
    canonical_hex = P256_GENERATOR_SEC1.hex()
    fingerprint = hashlib.sha256(P256_GENERATOR_SEC1).hexdigest()
    request = {
        "pre_enrollment_public_key_algorithm_profile": "ECDSA-P256-SHA256",
        "pre_enrollment_public_key_canonical_bytes": canonical_hex,
        "pre_enrollment_public_key_fingerprint_sha256": fingerprint,
        "request_signature_der_hex": "3006020101020101",
    }
    if mutation == "fingerprint":
        request["pre_enrollment_public_key_fingerprint_sha256"] = str(value)
    elif mutation == "spki":
        spki = (
            bytes.fromhex("3059301306072a8648ce3d020106082a8648ce3d030107034200")
            + P256_GENERATOR_SEC1
        )
        request["pre_enrollment_public_key_canonical_bytes"] = spki.hex()
        request["pre_enrollment_public_key_fingerprint_sha256"] = hashlib.sha256(spki).hexdigest()
    elif mutation == "algorithm":
        request["pre_enrollment_public_key_algorithm_profile"] = str(value)
    elif mutation in {"short_key", "wrong_prefix"}:
        request["pre_enrollment_public_key_canonical_bytes"] = str(value)
        request["pre_enrollment_public_key_fingerprint_sha256"] = hashlib.sha256(
            bytes.fromhex(str(value))
        ).hexdigest()
    elif mutation == "non_minimal_der":
        request["request_signature_der_hex"] = str(value)
    else:
        high_s = P256_ORDER - 1
        encoded_s = high_s.to_bytes(32, "big")
        request["request_signature_der_hex"] = (
            bytes([0x30, 38, 0x02, 1, 1, 0x02, 33, 0]) + encoded_s
        ).hex()
    with pytest.raises(AssertionError):
        _verify_pre_enrollment_key_relationship(
            request,
            {"pre_enrollment_public_key_fingerprint_sha256": fingerprint},
            {"continuity_signer_fingerprint": fingerprint},
        )


def _verify_lppi_key_activation(
    contract: dict[str, Any], binding: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, Any]:
    lifecycle = contract["lppi_authority_key_lifecycle"]
    frozen = lifecycle["binding_artifact"]
    assert all(binding.get(key) == value for key, value in frozen["required_values"].items())
    constraints = lifecycle["custody"]["field_constraints"]
    for field, constraint in constraints.items():
        value = binding.get(field)
        assert isinstance(value, str)
        if constraint.startswith("exact:"):
            assert value == constraint.removeprefix("exact:")
        else:
            assert re.fullmatch(constraint, value)
    assert evidence["continuity_signature_present"] is True
    assert (
        evidence["continuity_signer_fingerprint"]
        == binding["pre_enrollment_public_key_fingerprint_sha256"]
    )
    assert evidence["authority_key_pop_present"] is True
    assert (
        evidence["authority_key_pop_signer_fingerprint"]
        == binding["lppi_authority_public_key_fingerprint_sha256"]
    )
    assert evidence["custody_evidence_verified"] is True
    return {
        "status": "ACTIVE",
        **{
            field: binding[field]
            for field in lifecycle["active_key_record"]["consumer_match_fields"]
        },
    }


def _verify_lppi_signed_artifact(
    contract: dict[str, Any], active: dict[str, Any], artifact: dict[str, Any], kind: str
) -> None:
    if kind == "membership":
        signature = contract["provisioning_membership"]["authentication"]
    else:
        signature = contract["operation_identity"]["lppi_authenticated_operation_binding"][
            "signature_profile"
        ]
    assert active["status"] == "ACTIVE"
    assert artifact["signature_present"] is True
    assert artifact["signature_algorithm"] == signature["algorithm"]
    for field in signature["active_key_match_fields"]:
        assert artifact[field] == active[field]


def _valid_lppi_binding_and_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    hex64 = "a" * 64
    binding = {
        "lppi_authority_public_key_algorithm_profile": "ECDSA-P256-SHA256",
        "lppi_authority_public_key_fingerprint_sha256": hex64,
        "pre_enrollment_public_key_fingerprint_sha256": "b" * 64,
        "custody_profile": "WINDOWS_PLATFORM_CRYPTO_PROVIDER_TPM_ECDSA_P256_V1",
        "lppi_authority_key_tpm_name": "000b" + "c" * 64,
        "lppi_authority_tpmt_public_sha256": "d" * 64,
        "cng_provider_name": "Microsoft Platform Crypto Provider",
        "cng_key_name": "CryptoHunter.LPPI.Authority.1",
        "cng_key_unique_name": "machine\\lppi-authority-1",
        "tpm_creation_attestation_sha256": "e" * 64,
        "tpm_public_projection_id": "f" * 64,
    }
    evidence = {
        "continuity_signature_present": True,
        "continuity_signer_fingerprint": "b" * 64,
        "authority_key_pop_present": True,
        "authority_key_pop_signer_fingerprint": hex64,
        "custody_evidence_verified": True,
    }
    return binding, evidence


def test_lppi_crypto_profile_and_exact_active_key_relations() -> None:
    contract = _load(CONTRACT_PATH)
    binding, evidence = _valid_lppi_binding_and_evidence()
    active = _verify_lppi_key_activation(contract, binding, evidence)
    for kind in ("membership", "operation"):
        artifact = {
            "signature_present": True,
            "signature_algorithm": "ECDSA-P256-SHA256",
            **{
                field: active[field]
                for field in contract["lppi_authority_key_lifecycle"]["active_key_record"][
                    "consumer_match_fields"
                ]
            },
        }
        _verify_lppi_signed_artifact(contract, active, artifact, kind)


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("algorithm", "Ed25519"),
        ("continuity_signer", "0" * 64),
        ("pop_missing", False),
        ("custody_profile", "CALLER_STRING"),
        ("tpm_name", "not-a-tpm-name"),
    ],
)
def test_invalid_lppi_authority_crypto_or_custody_is_rejected(mutation: str, value: Any) -> None:
    contract = _load(CONTRACT_PATH)
    binding, evidence = _valid_lppi_binding_and_evidence()
    if mutation == "algorithm":
        binding["lppi_authority_public_key_algorithm_profile"] = value
    elif mutation == "continuity_signer":
        evidence["continuity_signer_fingerprint"] = value
    elif mutation == "pop_missing":
        evidence["authority_key_pop_present"] = value
    elif mutation == "custody_profile":
        binding["custody_profile"] = value
    else:
        binding["lppi_authority_key_tpm_name"] = value
    with pytest.raises(AssertionError):
        _verify_lppi_key_activation(contract, binding, evidence)


@pytest.mark.parametrize("kind", ["membership", "operation"])
def test_non_active_or_fingerprint_mismatched_lppi_signing_key_is_rejected(kind: str) -> None:
    contract = _load(CONTRACT_PATH)
    binding, evidence = _valid_lppi_binding_and_evidence()
    active = _verify_lppi_key_activation(contract, binding, evidence)
    artifact = {
        "signature_present": True,
        "signature_algorithm": "ECDSA-P256-SHA256",
        "lppi_authority_public_key_algorithm_profile": "ECDSA-P256-SHA256",
        "lppi_authority_public_key_fingerprint_sha256": "0" * 64,
        "custody_profile": "WINDOWS_PLATFORM_CRYPTO_PROVIDER_TPM_ECDSA_P256_V1",
    }
    with pytest.raises(AssertionError):
        _verify_lppi_signed_artifact(contract, active, artifact, kind)


def test_freeze_manifest_guards_exact_contract_bytes_and_coverage() -> None:
    freeze = _load(FREEZE_PATH)
    assert freeze["status"] == "FROZEN"
    assert freeze["implementation_architecturally_authorized"] is True
    assert freeze["production_provisioning_ready"] is False
    assert freeze["production_ceremony"] == "NOT_STARTED"
    assert len(freeze["artifacts"]) == 1
    artifact = freeze["artifacts"][0]
    assert set(artifact["covers"]) == REQUIRED_COVERAGE
    assert artifact["canonical_schema_version"] == _load(CONTRACT_PATH)["schema_version"]
    assert artifact["sha256"] == hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("path", "bad"),
    [
        (("authority_ownership", "account_genesis", "owner_count"), 2),
        (("authority_ownership", "account_genesis", "pdsa_is_owner"), True),
        (("provisioning_subject_id", "owner_issuer"), ""),
        (("provisioning_subject_id", "caller_selectable"), True),
        (("pdsa_enrollment_authorization_package", "contains_provisioning_operation_id"), True),
        (("operation_identity", "status"), "DESIGN_BLOCKED"),
        (("provisioning_membership", "authentication", "sha256_alone_is_authentication"), True),
        (("windows_post_install", "corehost_before_accepted_provisioning_transition"), "MAY_START"),
        (("pdsa_trust_domain", "production_accepts_test_or_dev_keys"), True),
        (("protected_freshness_authority", "separate_from_provisioning"), False),
        (
            (
                "protected_freshness_authority",
                "tpm_nv",
                "attributes",
                "TPMA_NV_COUNTER",
                "state",
            ),
            "CLEAR",
        ),
        (
            (
                "protected_freshness_authority",
                "tpm_nv",
                "attributes",
                "TPMA_NV_OWNERWRITE",
                "state",
            ),
            "SET",
        ),
        (
            (
                "protected_freshness_authority",
                "tpm_nv",
                "attributes",
                "TPMA_NV_PLATFORMCREATE",
                "state",
            ),
            "SET",
        ),
        (("secret_resource_authority", "separate_from_provisioning"), False),
        (("windows_handoff", "silent_fallback"), True),
    ],
)
def test_architecture_mutations_fail_closed(path: tuple[str, ...], bad: Any) -> None:
    value = copy.deepcopy(_load(CONTRACT_PATH))
    target: Any = value
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = bad
    with pytest.raises(AssertionError):
        _validate(value)


@pytest.mark.parametrize(
    "removed_field",
    [
        "pre_enrollment_request_digest_sha256",
        "verified_tpm_public_projection_id",
        "target_tpm_ek_public_digest",
        "target_tpm_ak_public_digest",
        "pre_enrollment_public_key_fingerprint_sha256",
        "pdsa_challenge_id",
        "pdsa_challenge_digest_sha256",
    ],
)
def test_removing_request_device_key_or_challenge_binding_fails_closed(
    removed_field: str,
) -> None:
    value = copy.deepcopy(_load(CONTRACT_PATH))
    value["pdsa_enrollment_authorization_package"]["canonical_payload_fields"].remove(removed_field)
    with pytest.raises(AssertionError):
        _validate(value)


def test_msi_source_does_not_mint_authority_facts() -> None:
    source = (ROOT / "deployment/windows_installer/provision.py").read_text(encoding="utf-8")
    forbidden = (
        "FirstRunBootstrapClaim(",
        "ProvisioningMembershipBinding(",
        "acct_",
        "psub_",
        "prvop_",
    )
    assert not any(token in source for token in forbidden)


def test_production_loader_remains_unconditionally_fail_closed() -> None:
    source = (ROOT / "deployment/windows_installer/corehost_composition.py").read_text(
        encoding="utf-8"
    )
    body = source.split("def load_windows_external_provisioning_handoff", 1)[1].split(
        "def materialize_canonical_pre_state", 1
    )[0]
    assert "raise WindowsProvisioningAdapterUnavailable" in body
    assert "fallback" not in body.lower()
