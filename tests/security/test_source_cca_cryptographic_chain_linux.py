"""S9-RPI-19: validate both real retained source and synthetic CCA signatures.

The TEST CCA public key is bound before worker-created source policy retention.
CCA private key never crosses into the worker; source private key never leaves it.
All authority/close/durability assertions stay synthetic and non-authorizing.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from bot_core import authenticated_custody_mutation_source as source
from bot_core import independent_cca_decision as cca
from bot_core import test_only_cca_decision_log as log_module
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.root_proof_issuer_substrate import ProviderRole
from tests.security._cca_decision_fixtures import build_cca_case, sign_cca_payload
from tests.security._custody_source_demo import start_process
from tests.security.test_retained_source_real_linux_disk_bridge import (
    _LinuxFixtureOriginalReader,
    _REQUEST,
    _inspect,
    _receive,
)
from tests.security.test_retained_source_test_floor_witness import (
    _FloorGuardedReader,
    _once_provision_fixture_floor,
)
from tests.security.test_source_custody_shared_lease_linux import (
    hold_synthetic_source_read_lease,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform != "linux", reason="real retained Linux source required"),
]


@pytest.fixture
def verified_chain(tmp_path):
    # Separate CCA signing key stays in the test orchestrator, never in source
    # worker config, process arguments, filesystem namespace or source policy.
    cca_key = Ed25519PrivateKey.generate()
    cca_public = cca_key.public_key().public_bytes_raw()
    root = tmp_path / "physical-test-source"
    root.mkdir(mode=0o700)
    status, config = _receive(
        *start_process(
            root=str(root),
            role=ProviderRole.ROOT_PROOF_SIGNING.value,
            completion_public_key=cca_public,
        )
    )
    assert status == "ok"
    assert config["completion_public_key"] == cca_public
    assert "private_key" not in config and "signing_key" not in config
    status, execution = _receive(*start_process(config=config, request=_REQUEST))
    assert status == "ok"
    assert execution["status"] == "NOT_QUALIFIED"
    assert execution["production_emission"] == "BLOCKED"
    original = execution["event"]
    reader = _LinuxFixtureOriginalReader(config, original)
    with _inspect(reader, original) as evidence:
        assert evidence.status == "NOT_QUALIFIED"
        assert evidence.production_authorization == "BLOCKED"
    raw_policy = reader._read("public-policy")
    policy = source.parse_source_verification_policy(raw_policy)
    assert base64.b64decode(policy.document["completion_authority"]["public_key_b64"]) == cca_public
    floor_dir = tmp_path / "same-uid-independent-test-source-floor"
    floor_dir.mkdir(mode=0o700)
    floor_pin = _once_provision_fixture_floor(floor_dir, original, raw_policy)
    return config, original, raw_policy, floor_pin, cca_key


def _prepare_decision(original, raw_policy, signing_key):
    event = source.parse_custody_mutation_execution(original)
    descriptor = base64.b64decode(event.payload["originals"]["operation_descriptor_b64"])
    complete = base64.b64decode(event.payload["originals"]["complete_journal_b64"])
    case = build_cca_case(descriptor, complete, signing_key=signing_key)
    source_policy = source.parse_source_verification_policy(raw_policy).document
    assert case.payload["completion_authority"] == source_policy["completion_authority"]
    for name in ("installation", "database", "custody", "close", "originals"):
        assert case.payload[name] == event.payload[name]

    case.payload["source_event_envelope_b64"] = base64.b64encode(original).decode("ascii")
    case.payload["source_event_envelope_sha256"] = hashlib.sha256(original).hexdigest()
    policy_document = deepcopy(case.policy_document)
    policy_document["source_verification_policy_b64"] = base64.b64encode(raw_policy).decode("ascii")
    policy_document["source_event_envelope_sha256"] = hashlib.sha256(original).hexdigest()
    policy_document["principal_inventory"][0]["public_material_identity"] = source_policy[
        "executor"
    ]["public_material_identity"]
    assert policy_document["principal_inventory"][0]["semantic_role"] == source_policy[
        "executor"
    ]["semantic_role"]
    return case, policy_document


def _cca_policy_for_retained_decision(base_policy, original_decision):
    position = original_decision.payload["decision_log_position"]
    document = deepcopy(base_policy)
    document["decision_id"] = position["entry_id"]
    document["decision_log_position"] = position
    return cca.parse_cca_verification_policy(canonical_json_bytes(document))


def _record_and_verify(config, original, raw_policy, floor_pin, key, tmp_path):
    case, policy_document = _prepare_decision(original, raw_policy, key)
    log_dir, floor_dir = tmp_path / "cca-log", tmp_path / "cca-floor"
    log_dir.mkdir(mode=0o700)
    floor_dir.mkdir(mode=0o700)
    pin = log_module.provision_test_only_cca_log(
        log_dir,
        floor_dir,
        installation_id=case.payload["installation"]["installation_id"],
        namespace=case.payload["completion_authority"]["provider_namespace"],
    )
    binding = cca.validate_cca_immutable_binding(
        canonical_json_bytes(
            {key: value for key, value in case.payload.items() if key not in {
                "decision_id", "decision_log_position"
            }}
        )
    )
    # The TEST log accepts exact immutable bytes, not the decoded object.
    immutable = canonical_json_bytes(binding)
    log = log_module.TestOnlyCCADecisionLog(pin)
    reader = _LinuxFixtureOriginalReader(config, original)
    guarded = _FloorGuardedReader(reader, floor_pin)
    with hold_synthetic_source_read_lease(config) as lease:
        assert lease.status == "NOT_QUALIFIED"
        assert lease.production_authorization == "BLOCKED"
        with _inspect(guarded, original) as evidence:
            assert evidence.production_authorization == "BLOCKED"

            def sign_at_position(position):
                payload = deepcopy(case.payload)
                payload["decision_id"] = position.entry_id
                payload["decision_log_position"] = position.document
                return sign_cca_payload(case, payload)

            retained = log.decide(
                case.payload["close"]["operation_id"], immutable, create_decision=sign_at_position
            )
            decision = cca.parse_independent_cca_decision(retained)
            policy = _cca_policy_for_retained_decision(policy_document, decision)
            verification = cca.verify_independent_cca_decision(
                decision, policy_loader=lambda: policy
            )
            assert verification.status == "NOT_QUALIFIED"
            assert verification.production_emission == "BLOCKED"
            assert verification.signature == "MATCHED_EXPLICIT_POLICY"
            assert verification.source_signature == "MATCHED_EXPLICIT_POLICY"
            assert verification.source_authorization == "NOT_QUALIFIED"
            assert verification.committed_close == "NOT_QUALIFIED"
            assert verification.durability == "NOT_ATTESTED"
            assert base64.b64decode(decision.payload["source_event_envelope_b64"]) == original
            assert guarded.load_policy().canonical_bytes == raw_policy
            assert log.exact_lookup(case.payload["close"]["operation_id"], immutable) == retained
        assert reader.lookups == 3
    return pin, immutable, case, retained, policy


def test_real_source_and_cca_ed25519_chain_under_held_lease(verified_chain, tmp_path):
    config, original, raw_policy, floor_pin, key = verified_chain
    pin, binding, case, retained, policy = _record_and_verify(
        config, original, raw_policy, floor_pin, key, tmp_path
    )
    assert (
        log_module.TestOnlyCCADecisionLog(pin).exact_lookup(
            case.payload["close"]["operation_id"], binding
        )
        == retained
    )
    status, restart = _receive(*start_process(config=config, request=_REQUEST))
    assert status == "ok" and restart["recovered"] is True
    assert restart["event"] == original
    # A retry never calls a new CCA signer.
    def forbidden(_position):
        pytest.fail("retained CCA decision may not be signed twice")
    with hold_synthetic_source_read_lease(config):
        assert (
            log_module.TestOnlyCCADecisionLog(pin).decide(
                case.payload["close"]["operation_id"],
                binding,
                create_decision=forbidden,
            )
            == retained
        )
    assert cca.verify_independent_cca_decision(
        cca.parse_independent_cca_decision(retained),
        policy_loader=lambda: policy,
    ).production_emission == "BLOCKED"


def test_forged_cca_signature_denied_against_retained_source_policy(verified_chain, tmp_path):
    config, original, raw_policy, floor_pin, key = verified_chain
    _pin, _binding, _case, retained, policy = _record_and_verify(
        config, original, raw_policy, floor_pin, key, tmp_path
    )
    altered = json.loads(retained)
    altered["signature_b64"] = base64.b64encode(b"\x00" * 64).decode("ascii")
    forged = cca.parse_independent_cca_decision(canonical_json_bytes(altered))
    with pytest.raises(cca.IndependentCCAVerificationError, match="VERIFICATION_DENIED"):
        cca.verify_independent_cca_decision(forged, policy_loader=lambda: policy)


def test_policy_snapshot_change_after_valid_signatures_is_denied(verified_chain, tmp_path):
    config, original, raw_policy, floor_pin, key = verified_chain
    _pin, _binding, _case, retained, policy = _record_and_verify(
        config, original, raw_policy, floor_pin, key, tmp_path
    )
    other = deepcopy(policy.document)
    other["decision_id"] = "test-another-cca-decision"
    other["decision_log_position"]["entry_id"] = other["decision_id"]
    alternate = cca.parse_cca_verification_policy(canonical_json_bytes(other))
    calls = []

    def loader():
        calls.append(True)
        return policy if len(calls) == 1 else alternate

    with pytest.raises(cca.IndependentCCAVerificationError, match="VERIFICATION_DENIED"):
        cca.verify_independent_cca_decision(
            cca.parse_independent_cca_decision(retained), policy_loader=loader
        )
    assert len(calls) == 2


def test_source_policy_tamper_denied_even_with_valid_retained_cca_signature(
    verified_chain, tmp_path
):
    config, original, raw_policy, floor_pin, key = verified_chain
    _pin, _binding, _case, retained, policy = _record_and_verify(
        config, original, raw_policy, floor_pin, key, tmp_path
    )
    path = Path(config["root"]) / "events" / "public-policy"
    path.write_bytes(b"{}")
    reader = _LinuxFixtureOriginalReader(config, original)
    guarded = _FloorGuardedReader(reader, floor_pin)
    with hold_synthetic_source_read_lease(config):
        with pytest.raises(ValueError, match="retained TEST inode or metadata changed"):
            guarded.load_policy()
    # The static CCA signature still verifies against the earlier snapshot;
    # a live policy source recheck is separately mandatory for acceptance.
    assert cca.verify_independent_cca_decision(
        cca.parse_independent_cca_decision(retained), policy_loader=lambda: policy
    ).production_emission == "BLOCKED"
