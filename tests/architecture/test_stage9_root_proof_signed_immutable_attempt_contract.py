"""Child freeze, historical byte identity, current status and forbidden scope."""

import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from bot_core import (
    cha_attempt_signatures,
    cha_attempt_store,
    cha_issuance_request,
    cha_root_proof_signing_custody,
)
from bot_core.licensing import cha_root_proof_signed_attempt

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
CONTRACT_PATH = DOCS / "stage9_root_proof_signed_immutable_attempt_contract.json"
FREEZE_PATH = DOCS / "stage9_root_proof_signed_immutable_attempt_freeze.json"
CONTRACT = json.loads(CONTRACT_PATH.read_bytes())
FREEZE = json.loads(FREEZE_PATH.read_bytes())


def test_child_freeze_and_historical_parent_bytes():
    assert FREEZE["contract_bytes_sha256"] == hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    assert FREEZE["parent_sha256"] == CONTRACT["frozen_parent_sha256"]
    for parent in FREEZE["parent_sha256"]:
        digest = hashlib.sha256((DOCS / parent["path"]).read_bytes()).hexdigest()
        assert digest == parent["bytes_sha256"]
    assert FREEZE["boundary"] == CONTRACT["boundary"] == "SIGNED_IMMUTABLE_DURABLE_NOT_SENT"


def test_exact_request_fields_domains_profiles_and_authority_namespace():
    parent = json.loads(
        (DOCS / "m05_account_genesis_independent_root_proof_issuer_contract.json").read_bytes()
    )
    assert (
        list(cha_issuance_request.REQUEST_FIELDS)
        == CONTRACT["request"]["signed_payload_fields"]
        == parent["issuance_request"]["signed_payload_fields"]
    )
    assert (
        CONTRACT["request"]["issuer_target_namespace"]
        == cha_issuance_request.ISSUER_TARGET_NAMESPACE
    )
    for role in cha_issuance_request.IssuanceSigningRole:
        contract = CONTRACT["signatures"][
            "requester"
            if role is cha_issuance_request.IssuanceSigningRole.REQUESTER
            else "claimant"
        ]
        assert contract["domain"] == role.domain
        assert contract["profile"] == role.profile
        assert (
            contract["preimage"] == "ASCII(domain) || 0x00 || exact same UTF-8 JCS signed payload"
        )
        assert contract["algorithm"] == "Ed25519"
    assert (
        cha_attempt_store._ATTEMPT_DOMAIN
        == b"CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_ATTEMPT_IDENTITY_V1\0"
    )
    assert CONTRACT["request"]["attempt_digest_in_request"] is False
    assert CONTRACT["request"]["namespace_caller_selected"] is False


def test_schema_and_immutable_signing_tables():
    assert cha_attempt_store._SCHEMA_VERSION == CONTRACT["durability"]["schema_version"] == 6
    assert list(cha_attempt_signatures.TABLES) == CONTRACT["durability"]["tables"][:3]
    assert CONTRACT["durability"]["migration"]["from_version"] == 5
    assert CONTRACT["durability"]["resigning_allowed"] is False
    assert (
        CONTRACT["durability"]["reference"]
        == "immutable:req:sha256:<lowercase SHA256 of exact canonical bytes>"
    )
    assert CONTRACT["durability"]["states"] == [
        cha_attempt_store.AttemptState.RESERVED_AWAITING_SIGNATURES.value,
        cha_attempt_store.AttemptState.REQUEST_SIGNED_BY_REQUESTER.value,
        cha_attempt_store.AttemptState.CLAIMANT_AUTHORIZED.value,
        cha_attempt_store.AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT.value,
    ]


def assert_status(current):
    assert CONTRACT["current_status"] == FREEZE["current_status"]
    for key, value in CONTRACT["current_status"].items():
        assert type(current[key]) is type(value) and current[key] == value


def test_current_status_keeps_implementation_and_live_provisioning_separate():
    current = json.loads((ROOT / "deployment/stage9_current_status.json").read_bytes())
    assert_status(current)
    assert current["signed_immutable_attempt"] == "IMPLEMENTED"
    assert (
        current["signed_immutable_attempt_live_availability"]
        == "BLOCKED_UNTIL_PRODUCTION_CREDENTIAL_PROVISIONING"
    )
    assert current["production_preaccount_credentials"] == "NOT_PROVISIONED"
    assert current["production_provisioning_ready"] is False
    assert current["windows_0_14"] == "10/15 DONE"
    assert current["root_proof"] == "NOT_ISSUED"
    assert current["prepared"] == "NOT_STARTED"


@pytest.mark.parametrize("field", CONTRACT["current_status"])
def test_status_conflicts_fail_closed(field):
    current = json.loads((ROOT / "deployment/stage9_current_status.json").read_bytes())
    current[field] = "CONTRADICTORY"
    with pytest.raises(AssertionError):
        assert_status(current)


def test_public_api_has_only_existing_upstream_capability_inputs():
    assert list(
        inspect.signature(cha_root_proof_signed_attempt.sign_root_proof_issuance_attempt).parameters
    ) == ["reservation"]
    assert list(
        inspect.signature(
            cha_root_proof_signed_attempt.resume_root_proof_issuance_attempt
        ).parameters
    ) == ["binding", "authorization"]
    assert (
        CONTRACT["public_api"][
            "caller_selected_private_key_path_dsn_signer_key_version_attempt_id_reference_signature_namespace"
        ]
        is False
    )


@pytest.mark.parametrize(
    "module",
    [
        cha_issuance_request,
        cha_attempt_signatures,
        cha_root_proof_signing_custody,
        cha_root_proof_signed_attempt,
    ],
)
def test_child_scope_contains_no_transport_bind_or_proof_authority(module):
    tree = ast.parse(inspect.getsource(module))
    imports = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not imports & {"requests", "httpx", "grpc", "socket", "aiohttp", "urllib"}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not (attrs | names) & {
        "compare_and_swap_bind",
        "sign_root_proof",
        "sign_history_head",
        "send",
        "send_issuance_request",
        "sign_freshness_proposal",
        "sign_finalization",
        "LocalRootProofSigningProvider",
        "RootProofIssuer",
        "admit_root_proof",
        "MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN",
        "EXACT_BOUND_RECOVERED",
        "PREPARED",
    }
    assert not any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("rpf_")
        for node in ast.walk(tree)
    )


def test_custody_runtime_ports_do_not_expose_provisioning_or_private_export():
    for cls, expected in (
        (cha_root_proof_signing_custody.LocalCHARequesterSigningCustody, "sign_issuance_request"),
        (
            cha_root_proof_signing_custody.LocalPreaccountClaimantAuthorizationCustody,
            "authorize_entitlement_claim",
        ),
    ):
        public = {name for name in dir(cls) if not name.startswith("_")}
        assert public == {"identity", expected}
    assert (
        CONTRACT["custody"]["runtime_can_generate_rotate_delete_export_or_change_lifecycle"]
        is False
    )
    assert CONTRACT["provisioning"]["distributed_atomicity"] is False
