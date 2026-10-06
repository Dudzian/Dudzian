"""TEST_ONLY socket, NSS and crypto simulation; no production signer is created."""

from __future__ import annotations

import copy
import hashlib
import os
import stat
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

import bot_core.licensing.pdsa_enrollment_authorization as authorization
import bot_core.licensing.pdsa_enrollment_challenge as challenge
import deployment.production_enrollment_issuer as issuer
import deployment.production_pdsa_signing as signing
import deployment.windows_stage9_production_trust as trust
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.licensing.external_provisioning import PDSA_DOMAIN
from tests.licensing import test_production_pre_enrollment as pre_enrollment_tests


@pytest.fixture
def installed(monkeypatch, tmp_path, request):
    # TEST_ONLY software keys use simulated service labels so the production
    # rejection of the actual TestOnlyProvisioningAuthority stays active.
    prefix = getattr(request, "param", "pdsa-service")
    keys = {
        f"{prefix}-{index}": ed25519.Ed25519PrivateKey.from_private_bytes(bytes([index]) * 32)
        for index in range(1, 4)
    }
    release = SimpleNamespace(
        payload_digest=trust.RELEASE_PAYLOAD_DIGEST,
        release_version=1,
        pinned_root=SimpleNamespace(
            key_set_digest=trust.ROOT_KEY_SET_DIGEST, environment="PRODUCTION"
        ),
        pdsa_key_set_digest=trust.PDSA_KEY_SET_DIGEST,
        recovery_public_digest=trust.RECOVERY_PUBLIC_DIGEST,
        purpose="PRODUCTION",
        pdsa_threshold=2,
        pdsa_keys=keys,
    )
    monkeypatch.setattr(
        trust,
        "verify_final_package",
        lambda *args, **kwargs: SimpleNamespace(
            ceremony_id=trust.CEREMONY_ID, verified_release=release
        ),
    )

    def document(path):
        if path.name == "pdsa_public_bundle.json":
            return {
                "threshold": 2,
                "keys": [
                    {
                        "key_id": key_id,
                        "public_key_hex": key.public_key()
                        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
                        .hex(),
                    }
                    for key_id, key in keys.items()
                ],
            }
        return {"status": "PRODUCTION_ROOT_OF_TRUST_FROZEN"}

    monkeypatch.setattr(trust, "_canonical_document", document)
    context = trust.load_production_trust(tmp_path)
    monkeypatch.setattr(challenge, "_utc_now", lambda: pre_enrollment_tests.NOW)
    monkeypatch.setattr(authorization, "_utc_now", lambda: pre_enrollment_tests.NOW)
    monkeypatch.setattr(
        issuer,
        "_installed_service_configuration",
        lambda: issuer._InstalledIssuerConfiguration(tmp_path / "TEST_ONLY_ISSUER", context),
    )
    authority = issuer.open_installed_production_enrollment_issuer()

    def sign(message):
        return [
            {"key_id": key_id, "algorithm": "Ed25519", "signature_hex": key.sign(message).hex()}
            for key_id, key in sorted(keys.items())[:2]
        ]

    simulated_authority = SimpleNamespace(
        issuer=authority, context=context, keys=keys, store=authority.pdsa_store, sign=sign
    )
    composition = pre_enrollment_tests.integration.__wrapped__(
        simulated_authority, monkeypatch, tmp_path
    )
    item = next(composition)
    accepted = pre_enrollment_tests._authenticate(item)
    reservation = authorization._reserve_issuance(accepted)
    yield SimpleNamespace(
        issuer=authority,
        context=context,
        keys=keys,
        payload=parse_canonical(reservation.payload_raw),
        raw=reservation.payload_raw,
        accepted=accepted,
    )
    try:
        next(composition)
    except StopIteration:
        pass


@pytest.fixture
def socket_boundary(installed, monkeypatch):
    boundary = SimpleNamespace(
        requests=[],
        connections=[],
        retained={},
        peer_uid=0,
        fault=None,
        change_reply=lambda reply: reply,
        frame_size=None,
    )
    identity = signing._SocketIdentity(1, 2, 0, 123456, 0o660)
    monkeypatch.setattr(signing, "_installed_signing_socket_identity", lambda: identity)

    class TestOnlyQuorumSocket:
        def __init__(self, family, kind):
            assert family == signing.socket.AF_UNIX
            assert kind == signing.socket.SOCK_STREAM
            self.reply = b""
            boundary.connections.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def settimeout(self, timeout):
            assert 0 < timeout <= signing.RPC_TIMEOUT_SECONDS

        def connect(self, path):
            assert path == os.fspath(signing.SIGNING_SOCKET)
            if boundary.fault == "connect":
                raise ConnectionRefusedError("TEST_ONLY socket unavailable")

        def getsockopt(self, level, option, size):
            assert (level, option, size) == (
                signing.socket.SOL_SOCKET,
                signing.socket.SO_PEERCRED,
                12,
            )
            return struct.pack("3i", 100, boundary.peer_uid, 0)

        def sendall(self, frame):
            if boundary.fault == "send":
                raise TimeoutError("TEST_ONLY send timeout")
            length = struct.unpack("!I", frame[:4])[0]
            assert length == len(frame) - 4
            request = parse_canonical(frame[4:])
            boundary.requests.append(request)
            payload_raw = bytes.fromhex(request["payload_canonical_hex"])
            digest = hashlib.sha256(payload_raw).hexdigest()
            assert request["reservation_id"] == digest
            assert request["signature_domain"].encode() + b"\x00" == PDSA_DOMAIN
            assert request["required_signer_ids"] == sorted(installed.keys)[:2]
            if digest not in boundary.retained:
                boundary.retained[digest] = {
                    "schema_version": signing.REPLY_SCHEMA,
                    "reservation_id": digest,
                    "signatures": [
                        {
                            "key_id": key_id,
                            "algorithm": "Ed25519",
                            "signature_hex": installed.keys[key_id]
                            .sign(PDSA_DOMAIN + hashlib.sha256(payload_raw).digest())
                            .hex(),
                        }
                        for key_id in request["required_signer_ids"]
                    ],
                }
            reply = boundary.change_reply(copy.deepcopy(boundary.retained[digest]))
            raw = reply if type(reply) is bytes else canonical_json_bytes(reply)
            size = len(raw) if boundary.frame_size is None else boundary.frame_size
            self.reply = struct.pack("!I", size) + raw

        def recv(self, size):
            if boundary.fault == "read":
                raise TimeoutError("TEST_ONLY read timeout")
            if boundary.fault == "eof":
                return b""
            part, self.reply = self.reply[: min(size, 7)], self.reply[min(size, 7) :]
            return part

    monkeypatch.setattr(signing.socket, "socket", TestOnlyQuorumSocket)
    return boundary


def test_fixed_rpc_verifies_exact_quorum_and_retains_retry_after_restart(
    installed, socket_boundary
):
    result = installed.issuer.sign_enrollment_authorization(installed.raw)
    installed.issuer.close()
    restarted = issuer.open_installed_production_enrollment_issuer()
    assert restarted.sign_enrollment_authorization(installed.raw) == result
    assert len(result) == 2
    assert [record["key_id"] for record in result] == sorted(installed.keys)[:2]
    assert socket_boundary.requests[0] == socket_boundary.requests[1]
    assert len(socket_boundary.retained) == 1


@pytest.mark.parametrize("installed", ["TEST_ONLY_PDSA"], indirect=True)
def test_test_only_trust_never_enters_production_socket(installed, socket_boundary):
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="AUTHORITY_REQUIRED"):
        installed.issuer.sign_enrollment_authorization(installed.raw)
    assert socket_boundary.connections == []


@pytest.mark.parametrize("argument", ["signer", "callback", "socket", "backend", "path", "keys"])
def test_no_transport_selected_signer_or_configuration(installed, argument):
    with pytest.raises(TypeError):
        installed.issuer.sign_enrollment_authorization(installed.raw, **{argument: object()})
    with pytest.raises(TypeError):
        issuer.ProductionEnrollmentIssuerContext(**{argument: object()})


@pytest.mark.parametrize("fault", ["connect", "send", "read", "eof"])
def test_unavailable_or_partial_service_fails_closed(installed, socket_boundary, fault):
    socket_boundary.fault = fault
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="SIGNING_(UNAVAILABLE|REPLY)"):
        installed.issuer.sign_enrollment_authorization(installed.raw)


@pytest.mark.parametrize("frame_size", [0, signing.MAX_REPLY_BYTES + 1, 2**32 - 1])
def test_oversized_reply_rejected_before_body_read(installed, socket_boundary, frame_size):
    socket_boundary.frame_size = frame_size
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="REPLY_INVALID"):
        installed.issuer.sign_enrollment_authorization(installed.raw)


def test_peer_rejected_before_payload_sent(installed, socket_boundary):
    socket_boundary.peer_uid = 123456
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="PEER_REQUIRED"):
        installed.issuer.sign_enrollment_authorization(installed.raw)
    assert socket_boundary.requests == []


@pytest.mark.parametrize("replacement_call", [2, 3])
def test_socket_replacement_fails_closed(installed, socket_boundary, monkeypatch, replacement_call):
    calls = []

    def identity():
        calls.append(None)
        return signing._SocketIdentity(
            1, 3 if len(calls) == replacement_call else 2, 0, 123456, 0o660
        )

    monkeypatch.setattr(
        signing,
        "_installed_signing_socket_identity",
        identity,
    )
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="SOURCE_CHANGED"):
        installed.issuer.sign_enrollment_authorization(installed.raw)
    assert len(socket_boundary.requests) == (0 if replacement_call == 2 else 1)


def test_other_valid_quorum_cannot_change_reserved_signer_pair(installed, socket_boundary):
    def alternate(reply):
        digest = bytes.fromhex(reply["reservation_id"])
        key_id = sorted(installed.keys)[2]
        reply["signatures"][1] = {
            "key_id": key_id,
            "algorithm": "Ed25519",
            "signature_hex": installed.keys[key_id].sign(PDSA_DOMAIN + digest).hex(),
        }
        return reply

    socket_boundary.change_reply = alternate
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="REPLY_INVALID"):
        installed.issuer.sign_enrollment_authorization(installed.raw)


@pytest.mark.parametrize("change", ["missing", "payload", "signers", "committed"])
def test_only_exact_unpublished_reserved_payload_can_be_signed(installed, socket_boundary, change):
    with installed.issuer.pdsa_store._connect() as db:
        if change == "missing":
            db.execute("DELETE FROM pdsa_authorization_issuances")
        elif change == "payload":
            payload = dict(installed.payload)
            payload["provisioning_subject_id"] = "psub_019ba13c-5c00-7000-8000-000000000002"
            db.execute(
                "UPDATE pdsa_authorization_issuances SET payload_raw=?",
                (canonical_json_bytes(payload),),
            )
        elif change == "signers":
            db.execute(
                "UPDATE pdsa_authorization_issuances SET signer_ids_raw=?",
                (canonical_json_bytes(sorted(installed.keys)[1:]),),
            )
        else:
            # The SQL CHECK intentionally prevents a fabricated terminal state.
            # A genuine final commit is constructed through the full issuer flow.
            signatures = [
                {
                    "key_id": key_id,
                    "algorithm": "Ed25519",
                    "signature_hex": installed.keys[key_id]
                    .sign(PDSA_DOMAIN + hashlib.sha256(installed.raw).digest())
                    .hex(),
                }
                for key_id in sorted(installed.keys)[:2]
            ]
            package_raw = canonical_json_bytes(
                {"payload": installed.payload, "signatures": signatures}
            )
            authorization._persist_signed_package(installed.accepted, package_raw)
            authorization._finalize_issuance(installed.accepted)
        db.commit()
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="RESERVATION_REQUIRED"):
        installed.issuer.sign_enrollment_authorization(installed.raw)
    assert socket_boundary.connections == []


def test_total_rpc_deadline_cannot_be_extended_by_partial_reads(
    installed, socket_boundary, monkeypatch
):
    calls = iter([10.0, 10.0, 10.0, 16.0])
    monkeypatch.setattr(signing.time, "monotonic", lambda: next(calls))
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="UNAVAILABLE"):
        installed.issuer.sign_enrollment_authorization(installed.raw)


def test_oversized_request_rejected_before_connection(socket_boundary):
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="REQUEST_INVALID"):
        signing._exchange_request(b" " * (signing.MAX_REQUEST_BYTES + 1))
    assert socket_boundary.connections == []


@pytest.mark.parametrize(
    "mutation",
    [
        "one",
        "three",
        "duplicate",
        "unsorted",
        "unknown",
        "test",
        "bad_signature",
        "upper",
        "algorithm",
        "extra",
        "digest",
        "schema",
        "noncanonical",
    ],
)
def test_bad_quorum_or_rpc_reply_fails_closed(installed, socket_boundary, mutation):
    def mutate(reply):
        records = reply["signatures"]
        if mutation == "one":
            records.pop()
        elif mutation == "three":
            records.append(copy.deepcopy(records[0]))
        elif mutation == "duplicate":
            records[1] = copy.deepcopy(records[0])
        elif mutation == "unsorted":
            records.reverse()
        elif mutation == "unknown":
            records[0]["key_id"] = "unknown-pdsa"
        elif mutation == "test":
            records[0]["key_id"] = "TEST_ONLY_PDSA_1"
        elif mutation == "bad_signature":
            records[0]["signature_hex"] = "00" * 64
        elif mutation == "upper":
            records[0]["signature_hex"] = records[0]["signature_hex"].upper()
        elif mutation == "algorithm":
            records[0]["algorithm"] = "ECDSA-P256-SHA256"
        elif mutation == "extra":
            records[0]["caller_authority"] = "forbidden"
        elif mutation == "digest":
            reply["reservation_id"] = "00" * 32
        elif mutation == "schema":
            reply["schema_version"] = "v2"
        elif mutation == "noncanonical":
            return canonical_json_bytes(reply) + b"\n"
        return reply

    socket_boundary.change_reply = mutate
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="REPLY_INVALID"):
        installed.issuer.sign_enrollment_authorization(installed.raw)


@pytest.mark.parametrize("raw", [None, bytearray(b"{}"), b"{}", b"[]", b"{", b" " * 16_385])
def test_invalid_payload_rejected_before_rpc(installed, socket_boundary, raw):
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="PAYLOAD_INVALID"):
        installed.issuer.sign_enrollment_authorization(raw)
    assert socket_boundary.connections == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("environment", "TEST_ONLY"),
        ("schema_version", "PDSAEnrollmentPackageV1"),
        ("pdsa_trust_domain", "caller-domain"),
        ("product_profile", "caller-product"),
        ("release_policy_digest_sha256", "00" * 32),
        ("release_policy_generation", True),
    ],
)
def test_wrong_payload_authority_rejected_before_rpc(installed, socket_boundary, field, value):
    payload = dict(installed.payload, **{field: value})
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="PAYLOAD_INVALID"):
        installed.issuer.sign_enrollment_authorization(canonical_json_bytes(payload))
    assert socket_boundary.connections == []


def test_forged_closed_or_changed_issuer_cannot_sign(installed, socket_boundary):
    forged = object.__new__(issuer.ProductionEnrollmentIssuerContext)
    for authority in (forged, copy.copy(installed.issuer)):
        with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="CONTEXT_REQUIRED"):
            authority.sign_enrollment_authorization(installed.raw)
    installed.issuer.pdsa_store.path.chmod(0o666)
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="SOURCE_CHANGED"):
        installed.issuer.sign_enrollment_authorization(installed.raw)
    assert socket_boundary.connections == []


def test_trust_rechecked_after_service_reply(installed, socket_boundary, monkeypatch):
    original = signing.require_current_production_trust_context
    calls = []

    def changed(value):
        calls.append(value)
        if len(calls) == 2:
            raise trust.ProductionTrustUnavailable("TEST_ONLY current trust expired")
        return original(value)

    monkeypatch.setattr(signing, "require_current_production_trust_context", changed)
    with pytest.raises(trust.ProductionTrustUnavailable, match="expired"):
        installed.issuer.sign_enrollment_authorization(installed.raw)
    assert len(socket_boundary.requests) == 1


@pytest.mark.skipif(os.name != "posix", reason="fixed Linux issuer/socket permission boundary")
@pytest.mark.parametrize(
    "failure", [None, "owner", "group", "mode", "kind", "links", "principal", "ancestor"]
)
def test_fixed_installed_socket_permissions(monkeypatch, tmp_path, failure):
    import pwd

    fixed_path = tmp_path / "TEST_ONLY_quorum.sock"
    fixed_path.write_bytes(b"TEST_ONLY metadata fixture")
    lstat = Path.lstat
    principal = SimpleNamespace(pw_uid=123456, pw_gid=234567)
    inspected = []

    def metadata(path):
        source = lstat(path)
        if path != fixed_path:
            return source
        values = list(source)
        values[0] = (stat.S_IFREG if failure == "kind" else stat.S_IFSOCK) | (
            0o666 if failure == "mode" else 0o660
        )
        values[3] = 2 if failure == "links" else 1
        values[4] = 123456 if failure == "owner" else 0
        values[5] = 345678 if failure == "group" else principal.pw_gid
        return os.stat_result(values)

    def ancestors(path):
        inspected.append(path)
        if failure == "ancestor":
            raise issuer.ProductionEnrollmentIssuerError("PRODUCTION_ISSUER_DEPLOYMENT_REQUIRED")

    monkeypatch.setattr(signing.sys, "platform", "linux")
    monkeypatch.setattr(signing, "SIGNING_SOCKET", fixed_path)
    monkeypatch.setattr(Path, "lstat", metadata)
    monkeypatch.setattr(signing, "_require_protected_ancestors", ancestors)
    monkeypatch.setattr(pwd, "getpwnam", lambda name: principal)
    monkeypatch.setattr(
        os, "geteuid", lambda: 345678 if failure == "principal" else principal.pw_uid
    )
    monkeypatch.setattr(os, "getegid", lambda: principal.pw_gid)
    if failure is None:
        result = signing._installed_signing_socket_identity()
        assert result.uid == 0 and result.gid == principal.pw_gid and result.mode == 0o660
        assert inspected == [fixed_path.parent]
    else:
        with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="DEPLOYMENT_REQUIRED"):
            signing._installed_signing_socket_identity()
