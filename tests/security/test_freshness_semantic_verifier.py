from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import socket
import struct
import subprocess
import time

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from bot_core.freshness_semantic_verifier import (
    DOCUMENT_AUTHENTICATION_DOMAIN,
    DOCUMENT_DIGEST_DOMAIN,
    FINALIZATION_ROLE,
    PROPOSER_DOMAIN,
    RECEIPT_AUTHENTICATION_DOMAIN,
    FreshnessSemanticVerificationError,
    FreshnessSemanticVerifier,
    RetainedVerificationCredential,
    serve_local_unix_socket,
)
from bot_core.postgresql_freshness_authority import (
    PROPOSER_ROLE, canonical_json_bytes, complete_semantic_head_digest, parse_canonical_json,
)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


class Resolver:
    def __init__(self, proposer, finalizer):
        self.values = {(PROPOSER_ROLE, "p-key", 1): proposer,
                       (FINALIZATION_ROLE, "f-key", 1): finalizer}

    def resolve(self, **query):
        try:
            value = self.values[(query["semantic_role"], query["key_id"], query["key_version"])]
        except KeyError as exc:
            raise FreshnessSemanticVerificationError("unknown credential") from exc
        if (value.environment, value.trust_domain, value.authority_id) != (
                query["environment"], query["trust_domain"], query["authority_id"]):
            raise FreshnessSemanticVerificationError("wrong authority scope")
        return value

    def predecessor_complete_head_digest(self, **query):
        if query != {"environment": "PRODUCTION", "trust_domain": "td",
                     "authority_id": "auth", "generation": 0,
                     "document_digest": "0" * 64}:
            raise FreshnessSemanticVerificationError("unknown predecessor")
        return "1" * 64


class Writer:
    def __init__(self): self.calls = []
    def prepare(self, *values):
        self.calls.append(values)
        return "opaque-preparation"


def fixture_candidate():
    pkey, fkey = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    raw = lambda key: key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    proposer_credential = RetainedVerificationCredential(
        "PRODUCTION", "td", "auth", "proposer-credential",
        "CryptoHunterAccountAuthority", PROPOSER_ROLE, "p-key", 1, 7, raw(pkey))
    finalizer_credential = RetainedVerificationCredential(
        "PRODUCTION", "td", "auth", "finalizer-credential",
        "FreshnessAuthority", FINALIZATION_ROLE, "f-key", 1, 9, raw(fkey))
    heads = [{"digest": "2" * 64, "domain": "catalog"}]
    payload = {"schema_version": 1, "environment": "PRODUCTION", "trust_domain": "td",
               "authority_id": "auth", "generation": 1, "predecessor_generation": 0,
               "predecessor_document_digest": "0" * 64,
               "complete_semantic_head_set": heads, "freshness_authority_key_id": "f-key",
               "freshness_authority_key_version": 1, "finalization_request_id": "request-1"}
    digest = hashlib.sha256(DOCUMENT_DIGEST_DOMAIN + canonical_json_bytes(payload)).hexdigest()
    proposer_unsigned = {"schema_version": 1, "proposer_identity": "CryptoHunterAccountAuthority",
                         "environment": "PRODUCTION", "trust_domain": "td",
                         "proposer_key_id": "p-key", "proposer_key_version": 1,
                         "signed_or_authenticated_document_digest": digest}
    proposer = {**proposer_unsigned, "authentication_tag_or_signature":
                _b64(pkey.sign(PROPOSER_DOMAIN + canonical_json_bytes(proposer_unsigned)))}
    document_unsigned = {"payload": payload, "document_digest": digest}
    document = {**document_unsigned, "authentication_tag_or_signature":
                _b64(fkey.sign(DOCUMENT_AUTHENTICATION_DOMAIN + canonical_json_bytes(document_unsigned)))}
    receipt_unsigned = {"schema_version": 1, "environment": "PRODUCTION",
                        "trust_domain": "td", "authority_id": "auth",
                        "exact_predecessor_generation": 0,
                        "exact_predecessor_document_digest": "0" * 64,
                        "accepted_generation": 1, "accepted_document_digest": digest,
                        "complete_semantic_head_digest": complete_semantic_head_digest(heads),
                        "finalization_request_id": "request-1", "receipt_id": "receipt-1",
                        "freshness_authority_key_id": "f-key",
                        "freshness_authority_key_version": 1}
    receipt = {**receipt_unsigned, "authentication_tag_or_signature":
               _b64(fkey.sign(RECEIPT_AUTHENTICATION_DOMAIN + canonical_json_bytes(receipt_unsigned)))}
    writer = Writer()
    verifier = FreshnessSemanticVerifier(Resolver(proposer_credential, finalizer_credential),
                                         writer, "semantic-verifier-v1")
    return verifier, writer, proposer, document, receipt, pkey, fkey


def _run(values):
    verifier, _, proposer, document, receipt, *_ = values
    return verifier.verify_and_prepare(*(canonical_json_bytes(x) for x in (proposer, document, receipt)))


def test_three_real_ed25519_signatures_produce_one_derived_preparation():
    values = fixture_candidate()
    assert _run(values) == "opaque-preparation"
    assert len(values[1].calls) == 1
    preparation = values[1].calls[0][0]
    assert b'"verified"' not in preparation and b'"active"' not in preparation
    assert b'"original_decision_identity"' in preparation


@pytest.mark.parametrize("object_index", [2, 3, 4])
def test_one_bit_signature_mutation_rejects_without_preparation(object_index):
    values = list(fixture_candidate())
    obj = values[object_index]
    signature = bytearray(base64.urlsafe_b64decode(obj["authentication_tag_or_signature"] + "=="))
    signature[0] ^= 1
    obj["authentication_tag_or_signature"] = _b64(bytes(signature))
    with pytest.raises(FreshnessSemanticVerificationError, match="signature"):
        _run(values)
    assert values[1].calls == []


@pytest.mark.parametrize("object_index,domain", [
    (2, RECEIPT_AUTHENTICATION_DOMAIN),
    (3, RECEIPT_AUTHENTICATION_DOMAIN),
    (4, DOCUMENT_AUTHENTICATION_DOMAIN),
])
def test_correct_key_over_wrong_domain_is_rejected(object_index, domain):
    values = list(fixture_candidate())
    obj = values[object_index]
    unsigned = {k: v for k, v in obj.items() if k != "authentication_tag_or_signature"}
    key = values[5] if object_index == 2 else values[6]
    obj["authentication_tag_or_signature"] = _b64(key.sign(domain + canonical_json_bytes(unsigned)))
    with pytest.raises(FreshnessSemanticVerificationError, match="signature"):
        _run(values)
    assert values[1].calls == []


@pytest.mark.parametrize("target,path,value", [
    ("proposer", ("environment",), "TEST"),
    ("proposer", ("trust_domain",), "other"),
    ("proposer", ("proposer_identity",), "attacker"),
    ("proposer", ("proposer_key_version",), 2),
    ("document", ("payload", "authority_id"), "other"),
    ("document", ("payload", "generation"), 2),
    ("document", ("payload", "predecessor_generation"), 1),
    ("document", ("payload", "predecessor_document_digest"), "3" * 64),
    ("document", ("payload", "complete_semantic_head_set"), []),
    ("document", ("payload", "freshness_authority_key_id"), "other"),
    ("document", ("payload", "freshness_authority_key_version"), 2),
    ("document", ("payload", "finalization_request_id"), "other"),
    ("receipt", ("receipt_id",), "other"),
])
def test_signed_field_substitution_matrix_rejects(target, path, value):
    values = list(fixture_candidate())
    obj = {"proposer": values[2], "document": values[3], "receipt": values[4]}[target]
    cursor = obj
    for key in path[:-1]: cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(FreshnessSemanticVerificationError):
        _run(values)
    assert values[1].calls == []


def test_noncanonical_duplicate_and_caller_authority_fields_are_rejected():
    values = fixture_candidate()
    verifier, writer, proposer, document, receipt, *_ = values
    duplicate = canonical_json_bytes(proposer)[:-1] + b',"schema_version":1}'
    with pytest.raises(FreshnessSemanticVerificationError):
        verifier.verify_and_prepare(duplicate, canonical_json_bytes(document), canonical_json_bytes(receipt))
    proposer["trusted"] = True
    with pytest.raises(FreshnessSemanticVerificationError):
        _run(values)
    assert writer.calls == []


def test_cross_role_same_raw_key_rejected_before_database():
    values = list(fixture_candidate())
    proposer = values[0].credentials.values[(PROPOSER_ROLE, "p-key", 1)]
    finalizer = values[0].credentials.values[(FINALIZATION_ROLE, "f-key", 1)]
    object.__setattr__(finalizer, "public_key", proposer.public_key)
    with pytest.raises(FreshnessSemanticVerificationError, match="cross-role"):
        _run(values)
    assert values[1].calls == []


@pytest.mark.parametrize("target", ["proposer", "document", "receipt"])
def test_wrong_verification_public_key_rejected(target):
    values = list(fixture_candidate())
    credential = (values[0].credentials.values[(PROPOSER_ROLE, "p-key", 1)]
                  if target == "proposer" else
                  values[0].credentials.values[(FINALIZATION_ROLE, "f-key", 1)])
    wrong = Ed25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    object.__setattr__(credential, "public_key", wrong)
    with pytest.raises(FreshnessSemanticVerificationError, match="signature"):
        _run(values)
    assert values[1].calls == []


def test_unknown_selector_scope_role_and_bad_retained_key_fail_closed():
    values = list(fixture_candidate())
    values[2]["proposer_key_id"] = "unknown"
    with pytest.raises(FreshnessSemanticVerificationError, match="unknown"):
        _run(values)

    for field, replacement in (("authority_id", "other"),
                               ("semantic_role", FINALIZATION_ROLE),
                               ("public_key", b"short")):
        current = list(fixture_candidate())
        credential = current[0].credentials.values[(PROPOSER_ROLE, "p-key", 1)]
        object.__setattr__(credential, field, replacement)
        with pytest.raises(FreshnessSemanticVerificationError):
            _run(current)
        assert current[1].calls == []


def test_legal_proposer_rotation_changes_full_evidence_preparation_identity():
    values = list(fixture_candidate())
    assert _run(values) == "opaque-preparation"
    first_preparation = parse_canonical_json(values[1].calls[-1][0])
    new_key = Ed25519PrivateKey.generate()
    raw = new_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    old = values[0].credentials.values.pop((PROPOSER_ROLE, "p-key", 1))
    rotated = RetainedVerificationCredential(
        old.environment, old.trust_domain, old.authority_id, "proposer-credential-v2",
        old.semantic_identity, old.semantic_role, "p-key-2", 2, 8, raw,
    )
    values[0].credentials.values[(PROPOSER_ROLE, "p-key-2", 2)] = rotated
    unsigned = {k: v for k, v in values[2].items() if k != "authentication_tag_or_signature"}
    unsigned["proposer_key_id"] = "p-key-2"
    unsigned["proposer_key_version"] = 2
    values[2] = {**unsigned, "authentication_tag_or_signature":
                 _b64(new_key.sign(PROPOSER_DOMAIN + canonical_json_bytes(unsigned)))}
    assert _run(values) == "opaque-preparation"
    second_preparation = parse_canonical_json(values[1].calls[-1][0])
    assert first_preparation["preparation_id"] != second_preparation["preparation_id"]


def _ipc_exchange(path: Path, request: bytes, *, declared_length: int | None = None) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(path))
        client.sendall(struct.pack("!I", len(request) if declared_length is None else declared_length) + request)
        size = struct.unpack("!I", client.recv(4))[0]
        return json.loads(client.recv(size))


def _ipc_server(tmp_path: Path, verifier, requests: int = 1):
    tmp_path.chmod(0o755)
    tmp_path.parent.chmod(0o755)
    tmp_path.parent.parent.chmod(0o755)
    path = tmp_path / "verifier.sock"
    process = multiprocessing.get_context("fork").Process(
        target=serve_local_unix_socket,
        args=(verifier, str(path)),
        kwargs={"allowed_peer_uid": os.getuid(), "stop_after": requests},
    )
    process.start()
    for _ in range(100):
        if path.exists(): break
        time.sleep(0.01)
    return path, process


def _ipc_request(values, **extra) -> bytes:
    fields = dict(zip(("proposer_authentication", "authoritative_document",
                       "finalization_receipt"),
                      (_b64(canonical_json_bytes(x)) for x in values[2:5])))
    fields.update(extra)
    return canonical_json_bytes(fields)


def test_unix_ipc_process_accepts_only_closed_valid_request(tmp_path):
    values = list(fixture_candidate())
    path, process = _ipc_server(tmp_path, values[0])
    assert _ipc_exchange(path, _ipc_request(values)) == {"preparation_id": "opaque-preparation"}
    process.join(5)
    assert process.exitcode == 0 and not path.exists()


@pytest.mark.parametrize("request_factory", [
    lambda values: _ipc_request(values, method="verify"),
    lambda values: _ipc_request(values, provider="attacker"),
    lambda values: _ipc_request(values, db_role="freshness_admin"),
    lambda values: b"{not-json}",
    lambda values: canonical_json_bytes({"proposer_authentication": "%%%",
                                         "authoritative_document": "AA",
                                         "finalization_receipt": "AA"}),
])
def test_unix_ipc_process_rejects_injection_and_malformed_without_oracle(tmp_path, request_factory):
    values = list(fixture_candidate())
    path, process = _ipc_server(tmp_path, values[0])
    assert _ipc_exchange(path, request_factory(values)) == {"error": "REJECTED"}
    process.join(5)
    assert process.exitcode == 0


@pytest.mark.parametrize("declared", [1, 4 * 1024 * 1024 + 1])
def test_unix_ipc_process_rejects_bad_frame_lengths(tmp_path, declared):
    values = list(fixture_candidate())
    path, process = _ipc_server(tmp_path, values[0])
    assert _ipc_exchange(path, b"x", declared_length=declared) == {"error": "REJECTED"}
    process.join(5)
    assert process.exitcode == 0


def test_unix_ipc_wrong_peer_cannot_invoke_and_there_is_no_tcp_listener(tmp_path):
    values = list(fixture_candidate())
    path, process = _ipc_server(tmp_path, values[0])
    path.chmod(0o666)  # force the request through to the SO_PEERCRED gate
    attempt = subprocess.run(
        ["runuser", "-u", "nobody", "--", "/usr/bin/python3", "-c",
         "import socket,sys;s=socket.socket(socket.AF_UNIX);s.connect(sys.argv[1]);s.sendall(b'\\0\\0\\0\\2{}');print(s.recv(1).hex())", str(path)],
        text=True, capture_output=True,
    )
    assert attempt.returncode != 0
    assert process.is_alive()  # rejected peer did not count as a verifier operation
    assert _ipc_exchange(path, _ipc_request(values))["preparation_id"] == "opaque-preparation"
    process.join(5)
    assert process.exitcode == 0
