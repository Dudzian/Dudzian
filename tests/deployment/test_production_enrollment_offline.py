"""Offline delivery simulations; real verifiers around isolated TEST_ONLY custody.

The reused harness explicitly substitutes installed trust, OS configuration,
native TPM/CNG and off-host signing. No production keys or TPM are modified.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Barrier

import pytest
from cryptography.hazmat.primitives.asymmetric import utils

from bot_core.licensing import (
    pdsa_enrollment_challenge as challenge,
    production_tpm_custody as custody,
)
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.enrollment_file_io import read_artifact_file, write_artifact_file
from bot_core.licensing.enrollment_transport import ArtifactEnvelope
from bot_core.licensing.tpm_attestation import TPMEnrollmentRequestV1
from deployment import (
    production_enrollment_issuer as issuer,
    production_enrollment_offline as offline,
    production_pdsa_signing as signing,
    windows_stage9_production_trust as trust,
)
from tests.licensing import test_pdsa_enrollment_authorization as authorization_tests

challenge_harness = authorization_tests.challenge_harness
integration = authorization_tests.integration
issuance = authorization_tests.issuance
installed_signing_boundary = issuer.ProductionEnrollmentIssuerContext.sign_enrollment_authorization


def _save(path: Path, raw: bytes) -> Path:
    write_artifact_file(path, raw)
    return path


@pytest.fixture
def exchange(issuance, tmp_path):
    item = issuance
    device = ArtifactEnvelope.create(
        "DEVICE_REQUEST",
        {
            "activation_request": item.activation.canonical_bytes,
            "tpm_request": item.tpm_request.canonical_bytes,
        },
    )
    item.device = device
    item.device_path = _save(tmp_path / "device.json", device.canonical_bytes)
    item.pdsa_path = _save(tmp_path / "pdsa.json", item.pdsa_raw)
    item.endorsement_path = _save(tmp_path / "endorsement.json", item.endorsement_raw)
    item.output_root = tmp_path
    return item


def _issue(item, *, output="challenge.json", **changes):
    return offline.issue_offline_tpm_challenge(
        item.device_path,
        pdsa_challenge_path=item.pdsa_path,
        endorsement_path=item.endorsement_path,
        output_path=item.output_root / output,
        **changes,
    )


def _prepare(item, tpm_challenge, **changes):
    artifacts = tpm_challenge.artifacts | {
        "tpm_response": item.tpm_response.canonical_bytes,
        "request": item.request.canonical_bytes,
        "request_signature": item.arguments["signature"],
        "custody_evidence": item.evidence.canonical_bytes,
    }
    return ArtifactEnvelope.create(
        "PRE_ENROLLMENT", artifacts | changes, parent_id=tpm_challenge.request_id
    )


def _authorize(item, envelope, *, name="package.json"):
    path = item.output_root / f"pre-{envelope.request_id}.json"
    _save(path, envelope.canonical_bytes)
    return offline.authorize_offline_pre_enrollment(path, output_path=item.output_root / name)


def test_complete_exchange_reuses_existing_crypto_and_retains_exact_audit(exchange):
    item = exchange
    tpm = _issue(item)
    assert tpm.parent_id == item.device.request_id
    assert tpm.artifacts["tpm_challenge"] == item.tpm_challenge.canonical_bytes
    prepared = _prepare(item, tpm)
    package = _authorize(item, prepared)
    assert package.parent_id == prepared.request_id
    assert package.kind == "PDSA_PACKAGE"
    assert len(item.sign_calls) == 1
    journal = item.authority.store.path.parent / "offline"
    audits = [json.loads(read_artifact_file(path)) for path in journal.glob("audit-*.json")]
    assert {entry["result"] for entry in audits} >= {"RETAINED", "COMMITTED_PACKAGE_RETAINED"}
    assert all(entry["legal_enrollment"] == "NOT_PERFORMED" for entry in audits)
    assert (journal / f"{prepared.request_id}.package.json").read_bytes() == package.canonical_bytes


def test_lost_response_and_restarted_operator_return_identical_bytes(exchange, monkeypatch):
    item = exchange
    first_tpm = _issue(item)
    (item.output_root / "challenge.json").unlink()
    assert _issue(item).canonical_bytes == first_tpm.canonical_bytes
    prepared = _prepare(item, first_tpm)
    first = _authorize(item, prepared)
    (item.output_root / "package.json").unlink()
    item.authority.issuer.close()
    monkeypatch.setattr(challenge, "_utc_now", lambda: authorization_tests.NOW + timedelta(days=10))
    monkeypatch.setattr(custody, "_utc_now", lambda: authorization_tests.NOW + timedelta(days=10))

    def cannot_authenticate(*args, **kwargs):
        raise AssertionError("terminal file delivery must not renew authorization")

    monkeypatch.setattr(offline, "authenticate_production_pre_enrollment", cannot_authenticate)
    assert _authorize(item, prepared).canonical_bytes == first.canonical_bytes
    assert len(item.sign_calls) == 1


def test_terminal_retry_rejects_changed_proof_for_same_signed_request(exchange):
    item = exchange
    tpm = _issue(item)
    _authorize(item, _prepare(item, tpm))
    replacement = item.key.sign_request(
        item.request, production_trust_context=item.authority.context
    )
    assert replacement != item.arguments["signature"]
    changed = _prepare(item, tpm, request_signature=replacement)
    with pytest.raises(ValueError, match="RETAINED_INPUT_CONFLICT"):
        _authorize(item, changed, name="substituted-package.json")
    assert len(item.sign_calls) == 1
    assert not (item.output_root / "substituted-package.json").exists()


def test_parallel_request_retries_share_exact_challenge(exchange, monkeypatch):
    item = exchange
    contenders = Barrier(2)
    ensure = offline.ensure_artifact_directory

    def simultaneous_initialization(path):
        contenders.wait(timeout=5)
        ensure(path)

    monkeypatch.setattr(offline, "ensure_artifact_directory", simultaneous_initialization)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_issue, item, output=f"challenge-{index}.json") for index in range(2)
        ]
        replies = [future.result().canonical_bytes for future in futures]
    assert replies[0] == replies[1]


def test_failed_parallel_delivery_releases_session_and_preserves_exact_challenge(
    exchange, monkeypatch
):
    item = exchange
    contenders = Barrier(2)
    ensure = offline.ensure_artifact_directory
    write = offline.write_artifact_file

    def simultaneous_initialization(path):
        contenders.wait(timeout=5)
        ensure(path)

    def lose_one_response(path, raw):
        if path == item.output_root / "failed-challenge.json":
            raise OSError("TEST_ONLY response delivery failed")
        write(path, raw)

    monkeypatch.setattr(offline, "ensure_artifact_directory", simultaneous_initialization)
    monkeypatch.setattr(offline, "write_artifact_file", lose_one_response)
    with ThreadPoolExecutor(max_workers=2) as pool:
        failed = pool.submit(_issue, item, output="failed-challenge.json")
        successful = pool.submit(_issue, item, output="successful-challenge.json")
        with pytest.raises(OSError, match="TEST_ONLY response delivery failed"):
            failed.result(timeout=5)
        response = successful.result(timeout=5)
    monkeypatch.setattr(offline, "ensure_artifact_directory", ensure)
    retry = _issue(item, output="retry-challenge.json")
    assert retry.canonical_bytes == response.canonical_bytes
    assert response.artifacts["tpm_challenge"] == item.tpm_challenge.canonical_bytes
    journal = item.authority.store.path.parent / "offline"
    assert (journal / f"{item.device.request_id}.tpm.json").read_bytes() == response.canonical_bytes
    assert not (item.output_root / "failed-challenge.json").exists()


def test_two_parallel_distinct_device_requests_remain_distinct(exchange):
    item = exchange
    other_request = TPMEnrollmentRequestV1.create(
        activation_request=item.activation,
        public_projection=item.projection,
        client_nonce=b"\x94" * 32,
    )
    other = ArtifactEnvelope.create(
        "DEVICE_REQUEST",
        {
            "activation_request": item.activation.canonical_bytes,
            "tpm_request": other_request.canonical_bytes,
        },
    )
    other_path = _save(item.output_root / "other-device.json", other.canonical_bytes)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_issue, item)
        second = pool.submit(
            offline.issue_offline_tpm_challenge,
            other_path,
            pdsa_challenge_path=item.pdsa_path,
            endorsement_path=item.endorsement_path,
            output_path=item.output_root / "other-challenge.json",
        )
        first_reply, second_reply = first.result(), second.result()
    assert first_reply.parent_id != second_reply.parent_id
    assert first_reply.artifacts["tpm_challenge"] != second_reply.artifacts["tpm_challenge"]


def test_reselection_of_pdsa_challenge_for_same_device_request_is_rejected(exchange):
    item = exchange
    _issue(item)
    second = item.authority.store.issue(item.authority.context, item.authority.sign)
    path = _save(item.output_root / "other-pdsa.json", second)
    with pytest.raises(ValueError, match="RETAINED_INPUT_CONFLICT"):
        offline.issue_offline_tpm_challenge(
            item.device_path,
            pdsa_challenge_path=path,
            endorsement_path=item.endorsement_path,
            output_path=item.output_root / "other-challenge.json",
        )


def test_other_tpm_endorsement_cannot_bind_or_poison_device_request(exchange):
    item = exchange
    document = json.loads(item.endorsement_raw)
    payload = document["payload"]
    payload["ek_public_digest"] = "00" * 32
    payload["ek_name"] = "000b" + "00" * 32
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    document["payload_digest_sha256"] = digest.hex()
    for record in document["signatures"]:
        record["signature_hex"] = (
            item.authority.keys[record["key_id"]].sign(custody.ENDORSEMENT_DOMAIN + digest).hex()
        )
    path = _save(item.output_root / "other-tpm-endorsement.json", canonical_json_bytes(document))
    with pytest.raises(ValueError, match="ENDORSEMENT"):
        offline.issue_offline_tpm_challenge(
            item.device_path,
            pdsa_challenge_path=item.pdsa_path,
            endorsement_path=path,
            output_path=item.output_root / "wrong-tpm.json",
        )
    assert not (item.output_root / "wrong-tpm.json").exists()
    assert _issue(item).parent_id == item.device.request_id


@pytest.mark.parametrize("substitution", ["signature", "response", "endorsement"])
def test_substituted_public_proofs_cannot_authorize(exchange, substitution):
    item = exchange
    tpm = _issue(item)
    changed = {}
    if substitution == "signature":
        changed["request_signature"] = utils.encode_dss_signature(1, 1)
    elif substitution == "response":
        value = json.loads(item.tpm_response.canonical_bytes)
        value["activated_credential_digest"] = "00" * 32
        changed["tpm_response"] = canonical_json_bytes(value)
    else:
        value = json.loads(item.endorsement_raw)
        value["signatures"][0]["signature_hex"] = "00" * 64
        changed["endorsement"] = canonical_json_bytes(value)
    with pytest.raises(ValueError):
        _authorize(item, _prepare(item, tpm, **changed))
    assert not item.sign_calls
    assert not (item.output_root / "package.json").exists()


def test_expired_challenge_cannot_start_authorization(exchange, monkeypatch):
    item = exchange
    prepared = _prepare(item, _issue(item))
    monkeypatch.setattr(challenge, "_utc_now", lambda: authorization_tests.NOW + timedelta(days=10))
    with pytest.raises(ValueError, match="EXPIRED"):
        _authorize(item, prepared)
    assert not item.sign_calls


def test_unavailable_signer_keeps_reservation_and_retry_resumes(exchange):
    item = exchange
    prepared = _prepare(item, _issue(item))
    item.available_signer_ids = set()
    with pytest.raises(ValueError, match="QUORUM_UNAVAILABLE"):
        _authorize(item, prepared)
    assert not (item.output_root / "package.json").exists()
    item.available_signer_ids = set(item.authority.keys)
    assert _authorize(item, prepared).kind == "PDSA_PACKAGE"


def test_test_only_signing_keys_cannot_be_used_by_production_boundary(exchange, monkeypatch):
    item = exchange
    prepared = _prepare(item, _issue(item))
    monkeypatch.setattr(
        issuer.ProductionEnrollmentIssuerContext,
        "sign_enrollment_authorization",
        installed_signing_boundary,
    )
    monkeypatch.setattr(
        signing,
        "require_current_production_trust_context",
        trust.require_verified_production_trust_context,
    )
    with pytest.raises(ValueError, match="PRODUCTION_PDSA_SIGNING_AUTHORITY_REQUIRED"):
        _authorize(item, prepared)
    assert not (item.output_root / "package.json").exists()


def test_preview_does_not_open_installed_authority(exchange, monkeypatch):
    def forbidden():
        raise AssertionError("preview cannot open production authority")

    monkeypatch.setattr(offline, "open_installed_production_enrollment_issuer", forbidden)
    preview = offline.preview_offline_artifact(exchange.device_path)
    assert preview["request_id"] == exchange.device.request_id
    assert preview["authorization"] == "NOT_ESTABLISHED"
    assert not (exchange.authority.store.path.parent / "offline").exists()


def test_unauthorized_operator_cannot_create_journal(exchange, monkeypatch):
    def reject():
        raise issuer.ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_DEPLOYMENT_REQUIRED")

    monkeypatch.setattr(issuer, "_installed_service_configuration", reject)
    with pytest.raises(ValueError, match="DEPLOYMENT_REQUIRED"):
        _issue(exchange)
    assert not (exchange.authority.store.path.parent / "offline").exists()


def test_export_only_returns_existing_live_retained_challenge(exchange):
    item = exchange
    path = item.output_root / "exported-pdsa.json"
    offline.export_retained_pdsa_challenge(
        item.pdsa.document["payload"]["challenge_id"], output_path=path
    )
    assert read_artifact_file(path) == item.pdsa_raw
    with pytest.raises(ValueError, match="RETAINED_PDSA_CHALLENGE_REQUIRED"):
        offline.export_retained_pdsa_challenge(
            "unknown", output_path=item.output_root / "missing.json"
        )


def test_authorize_requires_retained_offline_parent(exchange):
    item = exchange
    tpm = ArtifactEnvelope.create(
        "TPM_CHALLENGE",
        item.device.artifacts
        | {
            "pdsa_challenge": item.pdsa_raw,
            "endorsement": item.endorsement_raw,
            "tpm_challenge": item.tpm_challenge.canonical_bytes,
        },
        parent_id=item.device.request_id,
    )
    with pytest.raises(ValueError, match="RETAINED_TPM_CHALLENGE_REQUIRED"):
        _authorize(item, _prepare(item, tpm))
    assert not item.sign_calls
