"""End-to-end verifier proof over the frozen Unix peer-auth composition."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import psycopg
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bot_core.freshness_semantic_verifier import (
    DOCUMENT_AUTHENTICATION_DOMAIN, DOCUMENT_DIGEST_DOMAIN, PROPOSER_DOMAIN,
    RECEIPT_AUTHENTICATION_DOMAIN,
)
from bot_core.postgresql_freshness_authority import (
    FINALIZATION_ROLE, PROPOSER_ROLE, PostgreSQLConnectionConfig, canonical_json_bytes,
    complete_semantic_head_digest, provision_postgresql_freshness_authority,
)
from tests.security.test_freshness_semantic_verifier import fixture_candidate
from tests.security.test_postgresql_freshness_production_local_authentication import (
    _as, isolated_peer_cluster,
)

ROLES = ("freshness_reader", "freshness_runtime", "freshness_crypto_verifier",
         "freshness_admin", "freshness_function_owner", "freshness_schema_owner")


def _signature(key, value, domain):
    return base64.urlsafe_b64encode(
        key.sign(domain + canonical_json_bytes(value))).rstrip(b"=").decode()


def _resign(values, request_id: str, receipt_id: str):
    proposer_key, finalizer_key = values[5], values[6]
    payload = values[3]["payload"]
    payload["finalization_request_id"] = request_id
    digest = hashlib.sha256(
        DOCUMENT_DIGEST_DOMAIN + canonical_json_bytes(payload)).hexdigest()
    values[3]["document_digest"] = digest
    document_unsigned = {"payload": payload, "document_digest": digest}
    values[3]["authentication_tag_or_signature"] = _signature(
        finalizer_key, document_unsigned, DOCUMENT_AUTHENTICATION_DOMAIN)
    proposer_unsigned = {k: v for k, v in values[2].items()
                         if k != "authentication_tag_or_signature"}
    proposer_unsigned["signed_or_authenticated_document_digest"] = digest
    values[2] = {**proposer_unsigned, "authentication_tag_or_signature":
                 _signature(proposer_key, proposer_unsigned, PROPOSER_DOMAIN)}
    receipt_unsigned = {k: v for k, v in values[4].items()
                        if k != "authentication_tag_or_signature"}
    receipt_unsigned.update({
        "accepted_document_digest": digest,
        "complete_semantic_head_digest": complete_semantic_head_digest(
            payload["complete_semantic_head_set"]),
        "finalization_request_id": request_id, "receipt_id": receipt_id,
    })
    values[4] = {**receipt_unsigned, "authentication_tag_or_signature":
                 _signature(finalizer_key, receipt_unsigned, RECEIPT_AUTHENTICATION_DOMAIN)}
    return values


def _successor(values, generation: int, predecessor_digest: str,
               request_id: str, receipt_id: str):
    values[3]["payload"].update({
        "generation": generation, "predecessor_generation": generation - 1,
        "predecessor_document_digest": predecessor_digest,
    })
    values[4].update({
        "exact_predecessor_generation": generation - 1,
        "exact_predecessor_document_digest": predecessor_digest,
        "accepted_generation": generation,
    })
    return _resign(values, request_id, receipt_id)


def _reset_to_real_core(cluster, values):
    with psycopg.connect(cluster["admin"], autocommit=True) as connection:
        connection.execute("DROP SCHEMA freshness_authority CASCADE")
        for role in ROLES:
            connection.execute(f'DROP OWNED BY "{role}" CASCADE')
            connection.execute(f'DROP ROLE "{role}"')
    provision_postgresql_freshness_authority(PostgreSQLConnectionConfig(cluster["admin"]))
    verifier = values[0]
    proposer = verifier.credentials.values[(PROPOSER_ROLE, "p-key", 1)]
    finalizer = verifier.credentials.values[(FINALIZATION_ROLE, "f-key", 1)]
    with psycopg.connect(cluster["admin"], autocommit=True) as connection:
        connection.execute("SET SESSION AUTHORIZATION freshness_admin")
        connection.execute(
            "SELECT freshness_authority.provision_authority(%s,%s,%s,%s,%s)",
            ("PRODUCTION", "td", "auth", "0" * 64, "1" * 64),
        )
        for credential in (proposer, finalizer):
            connection.execute("SELECT freshness_authority.provision_credential(%s)",
                (json.dumps({"security_profile": "PRODUCTION_LOCAL",
                  "environment": credential.environment,
                  "trust_domain": credential.trust_domain,
                  "authority_id": credential.authority_id,
                  "credential_id": credential.credential_id,
                  "semantic_identity": credential.semantic_identity,
                  "semantic_role": credential.semantic_role,
                  "key_id": credential.key_id,
                  "key_version": credential.key_version,
                  "public_key_hex": credential.public_key.hex()}),))


def _candidate_file(cluster, tmp_path: Path, values) -> Path:
    tmp_path.chmod(0o755)
    tmp_path.parent.chmod(0o755)
    tmp_path.parent.parent.chmod(0o755)
    payload = {
        "socket": str(cluster["socket"]), "port": cluster["port"],
    }
    # The verifier inputs, not authority assertions, cross the process boundary.
    from bot_core.postgresql_freshness_authority import canonical_json_bytes
    payload["objects"] = [base64.urlsafe_b64encode(canonical_json_bytes(x)).decode()
                          for x in values[2:5]]
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(payload), encoding="ascii")
    path.chmod(0o644)
    return path


def _verify_process(cluster, path: Path, *, check=True, lose_after_commit=False):
    # The CI interpreter lives below /root; permit only directory traversal for
    # the dedicated no-login test principal, never file mutation.
    for parent in (Path("/root"), Path(sys.executable).parent,
                   Path(sys.executable).parent.parent, Path(sys.executable).parent.parent.parent):
        parent.chmod(parent.stat().st_mode | 0o111)
    code = r'''import base64,json,sys
from bot_core.freshness_semantic_verifier import FreshnessSemanticVerifier,PostgreSQLPreparationWriter,PostgreSQLRetainedCredentialResolver,PostgreSQLVerifierConnection,PreparationOutcomeUnknown
v=json.load(open(sys.argv[1])); c=PostgreSQLVerifierConnection(v["socket"],v["port"])
writer=PostgreSQLPreparationWriter(c)
if sys.argv[2]=="lose":
 delegate=writer
 class Lost:
  def prepare(self,*args): delegate.prepare(*args); raise PreparationOutcomeUnknown("simulated post-commit response loss")
 writer=Lost()
ver=FreshnessSemanticVerifier(PostgreSQLRetainedCredentialResolver(c),writer,"semantic-verifier-v1")
print(ver.verify_and_prepare(*(base64.urlsafe_b64decode(x) for x in v["objects"])))'''
    return _as("os_freshness_crypto_verifier", [sys.executable, "-c", code, str(path),
        "lose" if lose_after_commit else "normal"], check=check)


def _cas_process(cluster, preparation_id: str, binding, document: bytes, receipt: bytes, *, check=False):
    material = base64.urlsafe_b64encode(canonical_json_bytes({
        "preparation_id": preparation_id, "binding": binding,
        "document": base64.urlsafe_b64encode(document).decode(),
        "receipt": base64.urlsafe_b64encode(receipt).decode(),
    })).decode()
    code = r'''import base64,json,sys,psycopg
v=json.loads(base64.urlsafe_b64decode(sys.argv[1])); c=psycopg.connect(host=sys.argv[2],port=int(sys.argv[3]),dbname="freshness_gate",user="freshness_runtime",sslmode="disable")
c.execute("set transaction isolation level serializable")
r=c.execute("select * from freshness_authority.compare_and_advance(%s,%s,%s,%s)",(v["preparation_id"],json.dumps(v["binding"]),base64.urlsafe_b64decode(v["document"]),base64.urlsafe_b64decode(v["receipt"]))).fetchone();c.commit();print(r[0])'''
    return _as("os_freshness_runtime", [sys.executable, "-c", code, material,
        str(cluster["socket"]), str(cluster["port"])], check=check)


def _retained(cluster, preparation_id):
    with psycopg.connect(cluster["admin"]) as connection:
        return connection.execute(
            "select binding,canonical_document_bytes,canonical_receipt_bytes from freshness_authority.prepared_verifications where preparation_id=%s",
            (preparation_id,),
        ).fetchone()


def _raw_prepare_process(cluster, binding, document: bytes, receipt: bytes):
    material = base64.urlsafe_b64encode(canonical_json_bytes({
        "binding": binding, "document": base64.urlsafe_b64encode(document).decode(),
        "receipt": base64.urlsafe_b64encode(receipt).decode(),
    })).decode()
    code = r'''import base64,json,sys,psycopg
from bot_core.postgresql_freshness_authority import canonical_json_bytes
v=json.loads(base64.urlsafe_b64decode(sys.argv[1]));c=psycopg.connect(host=sys.argv[2],port=int(sys.argv[3]),dbname="freshness_gate",user="freshness_crypto_verifier",sslmode="disable")
r=c.execute("select freshness_authority.prepare_verified_freshness_candidate(%s,%s,%s)",(canonical_json_bytes(v["binding"]),base64.urlsafe_b64decode(v["document"]),base64.urlsafe_b64decode(v["receipt"]))).fetchone();c.commit();print(r[0])'''
    return _as("os_freshness_crypto_verifier", [sys.executable, "-c", code, material,
        str(cluster["socket"]), str(cluster["port"])], check=False)


def _resolve_head(cluster, generation: int, digest: str, *, environment="PRODUCTION",
                  trust_domain="td", authority_id="auth"):
    query = (
        "select freshness_authority.resolve_predecessor_head("
        f"'{environment}','{trust_domain}','{authority_id}',{generation},'{digest}')"
    )
    return _as("os_freshness_crypto_verifier", [
        str(cluster["bindir"] / "psql"), "-XAt", "-h", str(cluster["socket"]),
        "-p", str(cluster["port"]), "-d", "freshness_gate",
        "-U", "freshness_crypto_verifier", "-c", query,
    ], check=False)


def _restart(cluster):
    _as("postgres", [str(cluster["bindir"] / "pg_ctl"), "-D", str(cluster["data"]),
        "-m", "fast", "-w", "stop"])
    _as("postgres", [str(cluster["bindir"] / "pg_ctl"), "-D", str(cluster["data"]),
        "-l", str(cluster["data"].parent / "postgres.log"), "-w", "start"])


def test_real_peer_resolver_verifier_writer_and_exact_restart_recovery(isolated_peer_cluster, tmp_path):
    cluster = isolated_peer_cluster
    values = list(fixture_candidate())
    _reset_to_real_core(cluster, values)
    path = _candidate_file(cluster, tmp_path, values)

    lost = _verify_process(cluster, path, check=False, lose_after_commit=True)
    assert lost.returncode != 0 and "simulated post-commit response loss" in lost.stderr
    with psycopg.connect(cluster["admin"]) as connection:
        before = connection.execute(
            "SELECT binding,canonical_document_bytes,canonical_receipt_bytes FROM freshness_authority.prepared_verifications"
        ).fetchall()
    assert len(before) == 1
    preparation_id = before[0][0]["preparation_id"]

    # A new interpreter is a real verifier process restart and models recovery
    # after a committed response was lost. PostgreSQL is restarted as well,
    # proving that recovery is from durable retained evidence, not process state.
    _restart(cluster)
    recovered = _verify_process(cluster, path)
    assert recovered.stdout.strip() == preparation_id
    with psycopg.connect(cluster["admin"]) as connection:
        after = connection.execute(
            "SELECT binding,canonical_document_bytes,canonical_receipt_bytes FROM freshness_authority.prepared_verifications"
        ).fetchall()
    assert after == before
    conflicting = dict(before[0][0])
    conflicting["verifier_authority_identity"] = "collision-attempt"
    rejected = _raw_prepare_process(cluster, conflicting, bytes(before[0][1]), bytes(before[0][2]))
    assert rejected.returncode != 0 and "preparation identity conflict" in rejected.stderr
    with psycopg.connect(cluster["admin"]) as connection:
        retained = connection.execute(
            "SELECT binding,canonical_document_bytes,canonical_receipt_bytes FROM freshness_authority.prepared_verifications"
        ).fetchall()
    assert retained == before


def test_concurrent_exact_prepare_serializes_to_one_row(isolated_peer_cluster, tmp_path):
    cluster = isolated_peer_cluster
    values = list(fixture_candidate())
    _reset_to_real_core(cluster, values)
    path = _candidate_file(cluster, tmp_path, values)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: _verify_process(cluster, path).stdout.strip(), range(2)))
    assert results[0] == results[1]
    with psycopg.connect(cluster["admin"]) as connection:
        assert connection.execute("SELECT count(*) FROM freshness_authority.prepared_verifications").fetchone() == (1,)


def test_peer_gate_has_no_tcp_or_password_fallback(isolated_peer_cluster):
    cluster = isolated_peer_cluster
    failed = subprocess.run([
        str(cluster["bindir"] / "psql"), "-XAt", "-h", "127.0.0.1",
        "-p", str(cluster["port"]), "-d", "freshness_gate",
        "-U", "freshness_crypto_verifier", "-c", "SELECT 1",
    ], text=True, capture_output=True)
    assert failed.returncode != 0
    assert _as("os_freshness_runtime", [
        str(cluster["bindir"] / "psql"), "-XAt", "-h", str(cluster["socket"]),
        "-p", str(cluster["port"]), "-d", "freshness_gate",
        "-U", "freshness_crypto_verifier", "-c", "SELECT 1",
    ], check=False).returncode != 0


def test_real_verifier_preparation_then_runtime_cas_accepted(isolated_peer_cluster, tmp_path):
    cluster = isolated_peer_cluster
    values = list(fixture_candidate())
    _reset_to_real_core(cluster, values)
    preparation_id = _verify_process(cluster, _candidate_file(cluster, tmp_path, values)).stdout.strip()
    binding, document, receipt = _retained(cluster, preparation_id)
    assert _cas_process(cluster, preparation_id, binding, bytes(document), bytes(receipt)).stdout.strip() == "CAS_ACCEPTED"


@pytest.mark.parametrize("credential,target", [
    ("proposer-credential", "VERIFY_ONLY"),
    ("proposer-credential", "REVOKED"),
    ("finalizer-credential", "VERIFY_ONLY"),
    ("finalizer-credential", "REVOKED"),
])
def test_real_verifier_preparation_then_lifecycle_change_rejects_cas(
        isolated_peer_cluster, tmp_path, credential, target):
    cluster = isolated_peer_cluster
    values = list(fixture_candidate())
    _reset_to_real_core(cluster, values)
    preparation_id = _verify_process(cluster, _candidate_file(cluster, tmp_path, values)).stdout.strip()
    binding, document, receipt = _retained(cluster, preparation_id)
    with psycopg.connect(cluster["admin"], autocommit=True) as connection:
        connection.execute("SET SESSION AUTHORIZATION freshness_admin")
        connection.execute(
            "select freshness_authority.transition_credential(%s,%s,%s,%s,%s,%s,%s)",
            ("PRODUCTION", "td", "auth", credential, 1, target,
             json.dumps({"reason": "integration-race"})),
        )
    rejected = _cas_process(cluster, preparation_id, binding, bytes(document), bytes(receipt))
    assert rejected.returncode != 0 and "lifecycle divergence or inactive" in rejected.stderr


def test_real_retained_resolver_accepts_legal_new_proposer_but_old_cannot_cas(
        isolated_peer_cluster, tmp_path):
    cluster = isolated_peer_cluster
    values = list(fixture_candidate())
    _reset_to_real_core(cluster, values)
    old_id = _verify_process(cluster, _candidate_file(cluster, tmp_path, values)).stdout.strip()
    old_binding, old_document, old_receipt = _retained(cluster, old_id)
    new_key = Ed25519PrivateKey.generate()
    public = new_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    with psycopg.connect(cluster["admin"], autocommit=True) as connection:
        connection.execute("SET SESSION AUTHORIZATION freshness_admin")
        connection.execute(
            "select freshness_authority.transition_credential(%s,%s,%s,%s,%s,%s,%s)",
            ("PRODUCTION", "td", "auth", "proposer-credential", 1, "VERIFY_ONLY",
             json.dumps({"reason": "legal-rotation"})),
        )
        connection.execute("select freshness_authority.provision_credential(%s)",
            (json.dumps({"security_profile": "PRODUCTION_LOCAL", "environment": "PRODUCTION",
             "trust_domain": "td", "authority_id": "auth", "credential_id": "proposer-v2",
             "semantic_identity": "CryptoHunterAccountAuthority", "semantic_role": PROPOSER_ROLE,
             "key_id": "p-key-v2", "key_version": 2, "public_key_hex": public.hex()}),))
    unsigned = {k: v for k, v in values[2].items() if k != "authentication_tag_or_signature"}
    unsigned.update({"proposer_key_id": "p-key-v2", "proposer_key_version": 2})
    values[2] = {**unsigned, "authentication_tag_or_signature": base64.urlsafe_b64encode(
        new_key.sign(PROPOSER_DOMAIN + canonical_json_bytes(unsigned))).rstrip(b"=").decode()}
    new_id = _verify_process(cluster, _candidate_file(cluster, tmp_path, values)).stdout.strip()
    assert new_id != old_id
    rejected = _cas_process(cluster, old_id, old_binding, bytes(old_document), bytes(old_receipt))
    assert rejected.returncode != 0 and "lifecycle divergence or inactive" in rejected.stderr


def test_c2_wins_before_c1_verification_but_c1_crypto_prepares_then_cas_rejects(
        isolated_peer_cluster, tmp_path):
    cluster = isolated_peer_cluster
    c1 = list(fixture_candidate())
    _resign(c1, "request-c1", "receipt-c1")
    _reset_to_real_core(cluster, c1)
    # C2 uses the same authoritative retained keys but distinct, correctly
    # signed candidate semantics and wins the N=0 linearization point first.
    c2 = list(fixture_candidate())
    c2[5], c2[6] = c1[5], c1[6]
    _resign(c2, "request-c2", "receipt-c2")
    winner_id = _verify_process(cluster, _candidate_file(cluster, tmp_path, c2)).stdout.strip()
    winner_binding, winner_document, winner_receipt = _retained(cluster, winner_id)
    assert _cas_process(cluster, winner_id, winner_binding,
                        bytes(winner_document), bytes(winner_receipt)).stdout.strip() == "CAS_ACCEPTED"

    # C1 is verified only after current has moved to generation 1. Historical
    # generation 0 remains resolvable, so crypto preparation succeeds.
    stale_id = _verify_process(cluster, _candidate_file(cluster, tmp_path, c1)).stdout.strip()
    stale_binding, stale_document, stale_receipt = _retained(cluster, stale_id)
    rejected = _cas_process(cluster, stale_id, stale_binding,
                            bytes(stale_document), bytes(stale_receipt))
    assert rejected.returncode != 0 and "predecessor mismatch" in rejected.stderr
    with psycopg.connect(cluster["admin"]) as connection:
        assert connection.execute(
            "select current_generation from freshness_authority.authority_lineages"
        ).fetchone() == (1,)
        assert connection.execute(
            "select count(*) from freshness_authority.decisions"
        ).fetchone() == (1,)
        assert connection.execute(
            "select generation from freshness_authority.authority_generation_heads order by generation"
        ).fetchall() == [(0,), (1,)]


def test_generation_head_history_survives_two_durable_postgresql_restarts(
        isolated_peer_cluster, tmp_path):
    cluster = isolated_peer_cluster
    first = list(fixture_candidate())
    _resign(first, "durable-request-1", "durable-receipt-1")
    _reset_to_real_core(cluster, first)
    first_id = _verify_process(cluster, _candidate_file(cluster, tmp_path, first)).stdout.strip()
    first_binding, first_document, first_receipt = _retained(cluster, first_id)
    assert _cas_process(cluster, first_id, first_binding,
                        bytes(first_document), bytes(first_receipt)).stdout.strip() == "CAS_ACCEPTED"
    digest1 = first_binding["proposed_document_digest"]
    head1 = first[4]["complete_semantic_head_digest"]

    _restart(cluster)
    assert _resolve_head(cluster, 0, "0" * 64).stdout.strip() == "1" * 64
    assert _resolve_head(cluster, 1, digest1).stdout.strip() == head1

    second = list(fixture_candidate())
    second[5], second[6] = first[5], first[6]
    _successor(second, 2, digest1, "durable-request-2", "durable-receipt-2")
    second_id = _verify_process(cluster, _candidate_file(cluster, tmp_path, second)).stdout.strip()
    second_binding, second_document, second_receipt = _retained(cluster, second_id)
    assert _cas_process(cluster, second_id, second_binding,
                        bytes(second_document), bytes(second_receipt)).stdout.strip() == "CAS_ACCEPTED"
    digest2 = second_binding["proposed_document_digest"]
    head2 = second[4]["complete_semantic_head_digest"]

    _restart(cluster)
    assert _resolve_head(cluster, 0, "0" * 64).stdout.strip() == "1" * 64
    assert _resolve_head(cluster, 1, digest1).stdout.strip() == head1
    assert _resolve_head(cluster, 2, digest2).stdout.strip() == head2
