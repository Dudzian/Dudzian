"""TEST_ONLY socket, NSS and crypto simulation; no production signer is created."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
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
        composition=item,
    )
    try:
        next(composition)
    except StopIteration:
        pass


class TestOnlyDurableQuorumService:
    """TEST_ONLY software signer with a disk-backed remote idempotency journal."""

    __test__ = False

    def __init__(self, path, keys):
        assert path.name.startswith("TEST_ONLY")
        self.path = path
        self.keys = keys
        self.available = set(keys)
        self.fault = None
        self.signature_calls = []
        with sqlite3.connect(path) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute(
                """CREATE TABLE IF NOT EXISTS test_only_quorum_reservations (
                    reservation_id TEXT PRIMARY KEY,
                    request_raw BLOB NOT NULL,
                    selected_signer_ids_raw BLOB NOT NULL,
                    reply_raw BLOB
                )"""
            )

    @property
    def retained(self):
        with sqlite3.connect(self.path) as db:
            return {
                digest: parse_canonical(raw)
                for digest, raw in db.execute(
                    "SELECT reservation_id,reply_raw FROM test_only_quorum_reservations "
                    "WHERE reply_raw IS NOT NULL"
                )
            }

    def handle(self, request_raw):
        request = parse_canonical(request_raw)
        assert set(request) == {
            "schema_version",
            "reservation_id",
            "payload_canonical_hex",
            "signature_domain",
            "authorized_signer_ids",
            "required_threshold",
        }
        assert request["schema_version"] == signing.REQUEST_SCHEMA
        payload_raw = bytes.fromhex(request["payload_canonical_hex"])
        digest = hashlib.sha256(payload_raw).hexdigest()
        assert request["reservation_id"] == digest
        assert request["signature_domain"].encode() + b"\x00" == PDSA_DOMAIN
        assert request["authorized_signer_ids"] == sorted(self.keys)
        assert len(request["authorized_signer_ids"]) == 3
        assert type(request["required_threshold"]) is int
        assert request["required_threshold"] == 2
        with sqlite3.connect(self.path) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT request_raw,selected_signer_ids_raw,reply_raw "
                "FROM test_only_quorum_reservations WHERE reservation_id=?",
                (digest,),
            ).fetchone()
            if row is None:
                selected = sorted(set(request["authorized_signer_ids"]) & self.available)[:2]
                if len(selected) != 2:
                    raise ConnectionError("TEST_ONLY available quorum below required threshold")
                db.execute(
                    "INSERT INTO test_only_quorum_reservations VALUES (?,?,?,NULL)",
                    (digest, request_raw, canonical_json_bytes(selected)),
                )
                db.commit()
                reply_raw = None
            else:
                retained_request, selected_raw, reply_raw = row
                assert retained_request == request_raw
                selected = json.loads(selected_raw)
        if reply_raw is not None:
            return reply_raw
        if self.fault == "after_selection":
            raise ConnectionAbortedError("TEST_ONLY service crash after durable selection")
        if not set(selected) <= self.available:
            raise ConnectionError("TEST_ONLY retained quorum currently unavailable")
        signatures = []
        for key_id in selected:
            self.signature_calls.append((digest, key_id))
            signatures.append(
                {
                    "key_id": key_id,
                    "algorithm": "Ed25519",
                    "signature_hex": self.keys[key_id]
                    .sign(PDSA_DOMAIN + bytes.fromhex(digest))
                    .hex(),
                }
            )
        if self.fault == "after_signing":
            raise ConnectionAbortedError("TEST_ONLY service crash before durable reply")
        reply_raw = canonical_json_bytes(
            {
                "schema_version": signing.REPLY_SCHEMA,
                "reservation_id": digest,
                "signatures": signatures,
            }
        )
        with sqlite3.connect(self.path) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE test_only_quorum_reservations SET reply_raw=? "
                "WHERE reservation_id=? AND reply_raw IS NULL",
                (reply_raw, digest),
            )
            retained_raw = db.execute(
                "SELECT reply_raw FROM test_only_quorum_reservations WHERE reservation_id=?",
                (digest,),
            ).fetchone()[0]
            db.commit()
        if self.fault == "after_reply_commit":
            raise ConnectionAbortedError("TEST_ONLY service crash after durable reply")
        return retained_raw


@pytest.fixture
def socket_boundary(installed, monkeypatch, tmp_path):
    boundary = SimpleNamespace(
        requests=[],
        connections=[],
        service=TestOnlyDurableQuorumService(tmp_path / "TEST_ONLY_QUORUM.sqlite", installed.keys),
        reply_raws=[],
        peer_uid=0,
        fault=None,
        change_reply=lambda reply: reply,
        frame_size=None,
    )
    identity = signing._SocketIdentity(1, 2, 0, 123456, 0o660)
    monkeypatch.setattr(signing, "_installed_signing_socket_identity", lambda: identity)
    # The production transport is Linux-only, but these are transport mechanics
    # tests and intentionally replace the real socket boundary. Keep the simulated
    # constants available on Windows/macOS where Python may not expose them.
    monkeypatch.setattr(
        signing.socket,
        "AF_UNIX",
        getattr(signing.socket, "AF_UNIX", 1),
        raising=False,
    )
    monkeypatch.setattr(
        signing.socket,
        "SO_PEERCRED",
        getattr(signing.socket, "SO_PEERCRED", 17),
        raising=False,
    )

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
            request_raw = frame[4:]
            request = parse_canonical(request_raw)
            boundary.requests.append(request)
            retained_raw = boundary.service.handle(request_raw)
            boundary.reply_raws.append(retained_raw)
            reply = boundary.change_reply(parse_canonical(retained_raw))
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


@pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2)])
def test_fixed_rpc_verifies_every_authorized_quorum(installed, socket_boundary, pair):
    selected = [sorted(installed.keys)[index] for index in pair]
    socket_boundary.service.available = set(selected)
    result = installed.issuer.sign_enrollment_authorization(installed.raw)
    assert [record["key_id"] for record in result] == selected
    request = socket_boundary.requests[0]
    assert request["authorized_signer_ids"] == sorted(installed.keys)
    assert request["required_threshold"] == 2
    assert "required_signer_ids" not in request


def test_fixed_rpc_retains_selected_quorum_and_exact_reply_after_both_restarts(
    installed, socket_boundary
):
    selected = sorted(installed.keys)[1:]
    socket_boundary.service.available = set(selected)
    result = installed.issuer.sign_enrollment_authorization(installed.raw)
    old_service = socket_boundary.service
    assert len(old_service.signature_calls) == 2
    installed.issuer.close()
    restarted = issuer.open_installed_production_enrollment_issuer()
    socket_boundary.service = TestOnlyDurableQuorumService(old_service.path, installed.keys)
    assert socket_boundary.service.available == set(installed.keys)
    assert restarted.sign_enrollment_authorization(installed.raw) == result
    assert len(result) == 2
    assert [record["key_id"] for record in result] == selected
    assert socket_boundary.requests[0] == socket_boundary.requests[1]
    assert socket_boundary.reply_raws[0] == socket_boundary.reply_raws[1]
    assert len(socket_boundary.service.retained) == 1
    assert socket_boundary.service.signature_calls == []
    restarted.close()


def test_one_available_signer_cannot_issue_a_quorum(installed, socket_boundary):
    socket_boundary.service.available = {sorted(installed.keys)[1]}
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="SIGNING_UNAVAILABLE"):
        authorization.issue_production_pdsa_enrollment_authorization(installed.accepted)
    assert socket_boundary.service.retained == {}
    assert socket_boundary.service.signature_calls == []
    with installed.issuer.pdsa_store._connect() as db:
        row = db.execute("SELECT state,signer_ids_raw FROM pdsa_authorization_issuances").fetchone()
        assert tuple(row) == ("RESERVED", None)


def test_first_signer_unavailable_still_issues_authorized_package(installed, socket_boundary):
    selected = sorted(installed.keys)[1:]
    socket_boundary.service.available = set(selected)
    raw = authorization.issue_production_pdsa_enrollment_authorization(installed.accepted)
    document = parse_canonical(raw)
    assert [entry["key_id"] for entry in document["signatures"]] == selected
    with installed.issuer.pdsa_store._connect() as db:
        row = db.execute(
            "SELECT state,signer_ids_raw,package_raw FROM pdsa_authorization_issuances"
        ).fetchone()
        assert tuple(row) == ("COMMITTED", canonical_json_bytes(selected), raw)


@pytest.mark.parametrize("cut", ["read", "after_reply_commit"])
def test_durable_reply_lost_before_local_signed_commit_retries_exact_package(
    installed, socket_boundary, cut
):
    selected = sorted(installed.keys)[1:]
    old_service = socket_boundary.service
    old_service.available = set(selected)
    if cut == "read":
        socket_boundary.fault = cut
    else:
        old_service.fault = cut
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="SIGNING_UNAVAILABLE"):
        authorization.issue_production_pdsa_enrollment_authorization(installed.accepted)
    digest = hashlib.sha256(installed.raw).hexdigest()
    first_reply = old_service.retained[digest]
    with sqlite3.connect(old_service.path) as db:
        retained_reply_raw = db.execute(
            "SELECT reply_raw FROM test_only_quorum_reservations WHERE reservation_id=?",
            (digest,),
        ).fetchone()[0]
    expected_raw = canonical_json_bytes(
        {"payload": installed.payload, "signatures": first_reply["signatures"]}
    )
    assert [entry["key_id"] for entry in first_reply["signatures"]] == selected
    with installed.issuer.pdsa_store._connect() as db:
        row = db.execute(
            "SELECT state,signer_ids_raw,package_raw FROM pdsa_authorization_issuances"
        ).fetchone()
        assert tuple(row) == ("RESERVED", None, None)
    installed.issuer.close()
    restarted = issuer.open_installed_production_enrollment_issuer()
    item = installed.composition
    item.authority.issuer = restarted
    item.authority.store = restarted.pdsa_store
    item.arguments["challenge_store"] = restarted.pdsa_store
    item.arguments["pending"] = restarted.tpm_store
    socket_boundary.service = TestOnlyDurableQuorumService(old_service.path, installed.keys)
    socket_boundary.fault = None
    accepted = pre_enrollment_tests._authenticate(item)
    raw = authorization.issue_production_pdsa_enrollment_authorization(accepted)
    assert raw == expected_raw
    assert socket_boundary.service.signature_calls == []
    assert socket_boundary.service.retained[digest] == first_reply
    assert socket_boundary.reply_raws[-1] == retained_reply_raw
    assert (
        authorization.retry_production_pdsa_enrollment_authorization(
            restarted, request_raw=accepted.request_raw, challenge_raw=accepted.challenge_raw
        )
        == expected_raw
    )
    with restarted.pdsa_store._connect() as db:
        row = db.execute(
            "SELECT state,signer_ids_raw,package_raw FROM pdsa_authorization_issuances"
        ).fetchone()
        assert tuple(row) == ("COMMITTED", canonical_json_bytes(selected), expected_raw)
    restarted.close()


@pytest.mark.parametrize("cut", ["after_selection", "after_signing"])
def test_service_crash_before_reply_cannot_reselect_reserved_quorum(
    installed, socket_boundary, cut
):
    selected = sorted(installed.keys)[1:]
    old_service = socket_boundary.service
    old_service.available = set(selected)
    old_service.fault = cut
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="SIGNING_UNAVAILABLE"):
        installed.issuer.sign_enrollment_authorization(installed.raw)
    assert old_service.retained == {}
    with sqlite3.connect(old_service.path) as db:
        selected_raw = db.execute(
            "SELECT selected_signer_ids_raw FROM test_only_quorum_reservations"
        ).fetchone()[0]
        assert json.loads(selected_raw) == selected
    socket_boundary.service = TestOnlyDurableQuorumService(old_service.path, installed.keys)
    result = installed.issuer.sign_enrollment_authorization(installed.raw)
    assert [record["key_id"] for record in result] == selected


@pytest.mark.parametrize("installed", ["TEST_ONLY_PDSA"], indirect=True)
def test_test_only_trust_never_enters_production_socket(installed, socket_boundary):
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="AUTHORITY_REQUIRED"):
        installed.issuer.sign_enrollment_authorization(installed.raw)
    assert socket_boundary.connections == []


@pytest.mark.parametrize(
    "argument",
    [
        "signer",
        "signer_ids",
        "authorized_signer_ids",
        "required_threshold",
        "threshold",
        "public_keys",
        "endpoint",
        "callback",
        "socket",
        "backend",
        "path",
        "keys",
    ],
)
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


def test_valid_signature_from_unauthorized_fourth_key_fails_closed(installed, socket_boundary):
    fourth_key = ed25519.Ed25519PrivateKey.from_private_bytes(b"\x04" * 32)

    def unauthorized(reply):
        digest = bytes.fromhex(reply["reservation_id"])
        reply["signatures"][1] = {
            "key_id": "pdsa-service-4",
            "algorithm": "Ed25519",
            "signature_hex": fourth_key.sign(PDSA_DOMAIN + digest).hex(),
        }
        return reply

    socket_boundary.change_reply = unauthorized
    with pytest.raises(issuer.ProductionEnrollmentIssuerError, match="REPLY_INVALID"):
        installed.issuer.sign_enrollment_authorization(installed.raw)


@pytest.mark.parametrize(
    "change", ["missing", "payload", "authorized_signers", "trust", "committed"]
)
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
        elif change == "authorized_signers":
            db.execute(
                "UPDATE pdsa_authorization_issuances SET authorized_signer_ids_raw=?",
                (canonical_json_bytes(sorted(installed.keys)[1:]),),
            )
        elif change == "trust":
            db.execute(
                "UPDATE pdsa_authorization_issuances SET production_trust_raw=?",
                (canonical_json_bytes({"ceremony_id": "different-authority"}),),
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
