"""Hermetyczne dowody syntetyczne; żaden wynik nie kwalifikuje fizycznego hosta."""

from __future__ import annotations

import base64
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from deployment import linux_physical_evidence as evidence
from tests.deployment.test_linux_physical_collector import installation_fixture, payload_fixture


NOW = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
BEFORE_BOOT = "11111111-1111-1111-1111-111111111111"
AFTER_BOOT = "22222222-2222-2222-2222-222222222222"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def synthetic_transport() -> dict[str, Any]:
    """Materiał testu kryptografii, bez deklaracji wykonania natywnego testu."""
    private_key = Ed25519PrivateKey.generate()
    key = private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    installation = installation_fixture(Path("/tmp/stage9-synthetic-fixture"))
    materials = {name: _sha(name) for name in evidence.MATERIAL_KEYS}
    materials["source_revision"] = _sha("synthetic-source-revision")[:40]
    materials["installation_contract_sha256"] = evidence.digest(installation)
    materials["configuration_sha256"] = evidence.digest(
        {item["path"]: item["sha256"] for item in installation["configuration"]}
    )
    for material, phase in (
        ("artifact_sha256", "baseline"),
        ("update_artifact_sha256", "update"),
        ("failed_update_artifact_sha256", "failed_update"),
    ):
        materials[material] = installation["packages"][phase]["sha256"]
    host = {
        "host_id": _sha("synthetic-test-host"),
        "machine_id_sha256": _sha("synthetic-machine-id"),
        "dmi_product_uuid_sha256": _sha("synthetic-dmi-uuid"),
        "python_executable_sha256": _sha("synthetic-python-executable"),
        "public_key": base64.b64encode(public).decode("ascii"),
        "materials": materials,
        "installation": installation,
    }
    policy = {
        "schema_version": 1,
        "purpose": "LIVE_PHYSICAL_LINUX",
        "max_age_seconds": 3600,
        "hosts": [host],
    }
    challenge = evidence.issue_challenge(
        policy,
        source_revision=materials["source_revision"],
        ci_run_id="12345",
        ci_run_attempt="2",
        host_id=host["host_id"],
        now=NOW,
    )
    payloads = []
    for index, phase in enumerate(evidence.PHASES):
        payloads.append(
            {
                "schema_version": 1,
                "challenge_id": challenge["challenge_id"],
                "sequence": index,
                "phase": phase,
                "previous_sha256": "0" * 64,
                "observed_at_utc": _stamp(NOW + timedelta(seconds=index + 1)),
                "host": {
                    **{
                        name: host[name]
                        for name in (
                            "host_id",
                            "machine_id_sha256",
                            "dmi_product_uuid_sha256",
                            "python_executable_sha256",
                        )
                    },
                    "boot_id": BEFORE_BOOT if index < 4 else AFTER_BOOT,
                    "os_id": "ubuntu",
                    "os_version": "24.04",
                    "kernel": "6.8.0-synthetic-test",
                    "architecture": "x86_64",
                    "virtualization": "none",
                    "python_version": "3.11.13",
                },
                "materials": copy.deepcopy(materials),
                "observations": {},
            }
        )
    context = {"key": key, "host": host, "policy": policy, "challenge": challenge}
    context["bundle"] = _signed_bundle(context, payloads)
    return context


def _signed_bundle(context: dict[str, Any], payloads: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    previous = "0" * 64
    for payload in payloads:
        payload = copy.deepcopy(payload)
        payload["previous_sha256"] = previous
        record = evidence.sign_record(payload, context["key"])
        records.append(record)
        previous = evidence.digest(record)
    return {
        "schema_version": 1,
        "challenge_id": context["challenge"]["challenge_id"],
        "records": records,
    }


def _complete_transport(tmp_path: Path, *, current: bool = False) -> dict[str, Any]:
    context = synthetic_transport()
    installation = installation_fixture(tmp_path)
    context["host"]["installation"] = installation
    materials = context["host"]["materials"]
    for material, phase in (
        ("artifact_sha256", "baseline"),
        ("update_artifact_sha256", "update"),
        ("failed_update_artifact_sha256", "failed_update"),
    ):
        materials[material] = installation["packages"][phase]["sha256"]
    materials["configuration_sha256"] = evidence.digest(
        {item["path"]: item["sha256"] for item in installation["configuration"]}
    )
    materials["installation_contract_sha256"] = evidence.digest(installation)
    created = datetime.now(timezone.utc) - timedelta(minutes=1) if current else NOW
    context["challenge"] = evidence.issue_challenge(
        context["policy"],
        source_revision=materials["source_revision"],
        ci_run_id="12345",
        ci_run_attempt="2",
        host_id=context["host"]["host_id"],
        now=created,
    )
    payloads = _payloads(context)
    observations = payload_fixture(installation)
    for index, (payload, raw) in enumerate(zip(payloads, observations, strict=True)):
        payload["challenge_id"] = context["challenge"]["challenge_id"]
        payload["materials"] = copy.deepcopy(materials)
        payload["observed_at_utc"] = _stamp(created + timedelta(seconds=index + 1))
        payload["host"]["boot_id"] = raw["host"]["boot_id"]
        payload["observations"] = raw["observations"]
    _bind_state_to_challenge(payloads, context["challenge"]["challenge_id"], installation)
    context["bundle"] = _signed_bundle(context, payloads)
    return context


def _bind_state_to_challenge(
    payloads: list[dict[str, Any]], challenge_id: str, installation: dict[str, Any]
) -> None:
    def bind(value: object) -> None:
        if isinstance(value, dict):
            if "state" in value and isinstance(value["state"], dict):
                value["state"]["path"] = (
                    installation["state_directory"] + "/.stage9-evidence-" + challenge_id
                )
            for item in value.values():
                bind(item)
        elif isinstance(value, list):
            for item in value:
                bind(item)

    bind(payloads)


@pytest.fixture
def transport(tmp_path: Path) -> dict[str, Any]:
    return _complete_transport(tmp_path)


def _validate(context: dict[str, Any], *, complete: bool = True) -> list[dict[str, Any]]:
    return evidence.validate_record_chain(
        context["bundle"],
        context["challenge"],
        context["host"],
        complete=complete,
        now=NOW + timedelta(minutes=1),
    )


def _verify(context: dict[str, Any], **overrides: str) -> dict[str, Any]:
    identity = {
        "source_revision": context["challenge"]["source_revision"],
        "ci_run_id": context["challenge"]["ci_run_id"],
        "ci_run_attempt": context["challenge"]["ci_run_attempt"],
    }
    identity.update(overrides)
    return evidence.verify_bundle(
        context["bundle"],
        context["challenge"],
        context["policy"],
        now=NOW + timedelta(minutes=1),
        **identity,
    )


def _payloads(context: dict[str, Any]) -> list[dict[str, Any]]:
    return [copy.deepcopy(record["payload"]) for record in context["bundle"]["records"]]


def test_canonical_json_is_independent_of_field_order() -> None:
    assert evidence.canonical_bytes({"b": [1], "a": "ą"}) == evidence.canonical_bytes(
        {"a": "ą", "b": [1]}
    )
    assert evidence.digest({"x": 1}) != evidence.digest({"x": True})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), {"x": object()}])
def test_noncanonical_python_input_is_refused(value: object) -> None:
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.canonical_bytes(value)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"null",
        b"[]",
        b"{",
        b'{"x":1,"x":2}',
        b'{"x":{"nested":1,"nested":2}}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":-Infinity}',
        b'{"x":"\xff"}',
    ],
)
def test_malformed_or_ambiguous_json_is_refused(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "untrusted.json"
    path.write_bytes(raw)
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.read_json(path)


def test_missing_and_oversized_json_are_refused(tmp_path: Path) -> None:
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.read_json(tmp_path / "missing.json")
    path = tmp_path / "large.json"
    path.write_bytes(b" " * (evidence.MAX_JSON_BYTES + 1))
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.read_json(path)


def test_default_empty_trust_policy_does_not_issue_challenge() -> None:
    policy = evidence.read_json(evidence.DEFAULT_POLICY)
    assert policy["hosts"] == []
    with pytest.raises(evidence.LinuxEvidenceError, match="Host"):
        evidence.issue_challenge(
            policy,
            source_revision=_sha("source")[:40],
            ci_run_id="1",
            ci_run_attempt="1",
            host_id=_sha("not-approved"),
            now=NOW,
        )


def test_policy_requires_out_of_band_digest(tmp_path: Path, transport: dict[str, Any]) -> None:
    path = _write(tmp_path / "policy.json", transport["policy"])
    assert evidence.load_policy(path, evidence.digest(transport["policy"])) == transport["policy"]
    for digest in ("", "not-a-digest", "0" * 64):
        with pytest.raises(evidence.LinuxEvidenceError):
            evidence.load_policy(path, digest)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(schema_version=True),
        lambda p: p.update(purpose="TEST_ONLY"),
        lambda p: p.update(max_age_seconds=0),
        lambda p: p.update(max_age_seconds=86401),
        lambda p: p.update(max_age_seconds=True),
        lambda p: p.update(max_age_seconds=float("nan")),
        lambda p: p.update(hosts={}),
        lambda p: p.update(extra="PASS"),
        lambda p: p["hosts"].append(copy.deepcopy(p["hosts"][0])),
        lambda p: p["hosts"][0].update(public_key="not-base64"),
        lambda p: p["hosts"][0].update(public_key=base64.b64encode(b"short").decode()),
        lambda p: p["hosts"][0].update(public_key=[]),
        lambda p: p["hosts"][0].update(host_id="different-host"),
        lambda p: p["hosts"][0].update(installation={"status": "PASS"}),
        lambda p: p["hosts"][0]["materials"].pop("collector_sha256"),
        lambda p: p["hosts"][0]["materials"].update(source_revision="branch-name"),
        lambda p: p["hosts"][0]["materials"].update(artifact_sha256="not-a-hash"),
        lambda p: p["hosts"][0]["materials"].update(
            update_artifact_sha256=p["hosts"][0]["materials"]["artifact_sha256"]
        ),
    ],
)
def test_unapproved_or_malformed_policy_fails_closed(
    tmp_path: Path,
    transport: dict[str, Any],
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    policy = copy.deepcopy(transport["policy"])
    mutate(policy)
    path = _write(tmp_path / "policy.json", policy)
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.load_policy(path, evidence.digest(policy))


def test_nonce_is_unique_and_bound_to_challenge(transport: dict[str, Any]) -> None:
    original = transport["challenge"]
    second = evidence.issue_challenge(
        transport["policy"],
        source_revision=original["source_revision"],
        ci_run_id=original["ci_run_id"],
        ci_run_attempt=original["ci_run_attempt"],
        host_id=original["host_id"],
        now=NOW,
    )
    assert original["nonce"] != second["nonce"]
    assert original["challenge_id"] != second["challenge_id"]


@pytest.mark.parametrize(
    "change",
    [
        {"source_revision": "main"},
        {"source_revision": "a" * 40},
        {"ci_run_id": "0"},
        {"ci_run_id": True},
        {"ci_run_attempt": "-1"},
        {"nonce": "f" * 64},
        {"purpose": "TEST_ONLY"},
        {"repository": "attacker/repo"},
        {"schema_version": True},
        {"generated_status": "PASS"},
    ],
)
def test_challenge_tampering_or_self_attestation_is_refused(
    transport: dict[str, Any],
    change: dict[str, Any],
) -> None:
    challenge = copy.deepcopy(transport["challenge"])
    challenge.update(change)
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.validate_challenge(challenge, now=NOW)


@pytest.mark.parametrize("offset", [-1, 3601])
def test_future_and_expired_challenges_are_refused(transport: dict[str, Any], offset: int) -> None:
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.validate_challenge(transport["challenge"], now=NOW + timedelta(seconds=offset))


@pytest.mark.parametrize("duration", [0, -1, 86401])
def test_self_consistent_challenge_cannot_extend_expiration(
    transport: dict[str, Any],
    duration: int,
) -> None:
    challenge = copy.deepcopy(transport["challenge"])
    challenge["expires_at_utc"] = _stamp(NOW + timedelta(seconds=duration))
    challenge["challenge_id"] = evidence.digest(
        {k: v for k, v in challenge.items() if k != "challenge_id"}
    )
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.validate_challenge(challenge, now=NOW)


def test_valid_signed_transport_requires_all_phases_and_new_boot(transport: dict[str, Any]) -> None:
    payloads = _validate(transport)
    assert [item["phase"] for item in payloads] == list(evidence.PHASES)
    assert payloads[3]["host"]["boot_id"] != payloads[4]["host"]["boot_id"]


@pytest.mark.parametrize("missing", range(len(evidence.PHASES)))
def test_missing_phase_blocks_complete_verification(
    transport: dict[str, Any], missing: int
) -> None:
    del transport["bundle"]["records"][missing]
    with pytest.raises(evidence.LinuxEvidenceError):
        _validate(transport)


def test_partial_prefix_is_valid_only_for_collection_continuation(
    transport: dict[str, Any],
) -> None:
    transport["bundle"]["records"] = transport["bundle"]["records"][:3]
    assert len(_validate(transport, complete=False)) == 3
    with pytest.raises(evidence.LinuxEvidenceError):
        _validate(transport)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda b: b.update(schema_version=True),
        lambda b: b.update(challenge_id="0" * 64),
        lambda b: b.update(records={}),
        lambda b: b.update(status="PASS"),
        lambda b: b["records"].append(copy.deepcopy(b["records"][-1])),
        lambda b: b["records"].reverse(),
        lambda b: b["records"][0].update(status="PASS"),
        lambda b: b["records"][0].update(signature="not-base64"),
        lambda b: b["records"][0].update(signature=base64.b64encode(b"short").decode()),
        lambda b: b["records"][0].update(signature=base64.b64encode(b"\0" * 64).decode()),
        lambda b: b["records"][1]["payload"].update(observations={"status": "PASS"}),
        lambda b: b["records"][0]["payload"].update(source_revision="a" * 40),
    ],
)
def test_unsigned_tampering_or_manually_written_pass_is_refused(
    transport: dict[str, Any],
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    mutation(transport["bundle"])
    with pytest.raises(evidence.LinuxEvidenceError):
        _validate(transport)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p[2].update(sequence=True),
        lambda p: p[1].update(phase="post_reboot"),
        lambda p: p[2].update(challenge_id="0" * 64),
        lambda p: p[1]["materials"].update(source_revision="a" * 40),
        lambda p: p[1]["materials"].update(artifact_sha256="b" * 64),
        lambda p: p[0]["host"].update(host_id="0" * 64),
        lambda p: p[4]["host"].update(machine_id_sha256="0" * 64),
        lambda p: p[4]["host"].update(dmi_product_uuid_sha256="0" * 64),
        lambda p: p[4]["host"].update(python_executable_sha256="0" * 64),
        lambda p: p[0]["host"].update(virtualization="kvm"),
        lambda p: p[0]["host"].update(os_version="22.04"),
        lambda p: p[0]["host"].update(os_id="debian"),
        lambda p: p[0]["host"].update(architecture="aarch64"),
        lambda p: p[0]["host"].update(boot_id="not-a-boot-id"),
        lambda p: p[4]["host"].update(python_version="3.12.0"),
        lambda p: p[0]["host"].update(evidence_kind="TEST_ONLY"),
        lambda p: p[0].update(observations=None),
        lambda p: p[0].update(schema_version=True),
        lambda p: p[0].update(observed_at_utc=_stamp(NOW - timedelta(seconds=1))),
        lambda p: p[2].update(observed_at_utc=_stamp(NOW + timedelta(seconds=1))),
        lambda p: p[0].update(observed_at_utc="2026-11-01T12:00:01+00:00"),
        lambda p: p[0].update(observed_at_utc=_stamp(NOW + timedelta(minutes=2))),
    ],
)
def test_even_signed_claims_must_match_trusted_identity_materials_and_order(
    transport: dict[str, Any],
    mutation: Callable[[list[dict[str, Any]]], None],
) -> None:
    payloads = _payloads(transport)
    mutation(payloads)
    transport["bundle"] = _signed_bundle(transport, payloads)
    with pytest.raises(evidence.LinuxEvidenceError):
        _validate(transport)


def test_hash_chain_rejects_individually_signed_but_unlinked_records(
    transport: dict[str, Any],
) -> None:
    payload = copy.deepcopy(transport["bundle"]["records"][3]["payload"])
    payload["previous_sha256"] = "f" * 64
    transport["bundle"]["records"][3] = evidence.sign_record(payload, transport["key"])
    with pytest.raises(evidence.LinuxEvidenceError):
        _validate(transport)


@pytest.mark.parametrize("mode", ["no-reboot", "early-reboot", "late-reboot", "second-reboot"])
def test_real_reboot_requires_single_boot_change_at_correct_boundary(
    transport: dict[str, Any],
    mode: str,
) -> None:
    payloads = _payloads(transport)
    if mode == "no-reboot":
        for payload in payloads:
            payload["host"]["boot_id"] = BEFORE_BOOT
    elif mode == "early-reboot":
        payloads[3]["host"]["boot_id"] = AFTER_BOOT
    elif mode == "late-reboot":
        payloads[4]["host"]["boot_id"] = BEFORE_BOOT
    else:
        payloads[-1]["host"]["boot_id"] = "33333333-3333-3333-3333-333333333333"
    transport["bundle"] = _signed_bundle(transport, payloads)
    with pytest.raises(evidence.LinuxEvidenceError, match="restartu"):
        _validate(transport)


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_revision": "a" * 40},
        {"ci_run_id": "12346"},
        {"ci_run_attempt": "3"},
    ],
)
def test_signed_old_revision_or_different_ci_attempt_is_refused(
    transport: dict[str, Any],
    overrides: dict[str, str],
) -> None:
    with pytest.raises(evidence.LinuxEvidenceError, match="rewizji"):
        _verify(transport, **overrides)


def test_revoked_host_policy_cannot_be_replaced_by_bundle_claims(transport: dict[str, Any]) -> None:
    transport["policy"]["hosts"] = []
    with pytest.raises(evidence.LinuxEvidenceError):
        _verify(transport)


def test_unapproved_signature_key_cannot_authorize_known_host(transport: dict[str, Any]) -> None:
    replacement = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    )
    transport["host"]["public_key"] = base64.b64encode(replacement).decode("ascii")
    with pytest.raises(evidence.LinuxEvidenceError, match="producent"):
        _validate(transport)


def test_replay_ledger_refuses_reuse_and_changed_bundle(tmp_path: Path) -> None:
    path = tmp_path / "consumed.sqlite"
    challenge_id, bundle_id = _sha("challenge"), _sha("bundle")
    evidence.consume_challenge(path, challenge_id, bundle_id)
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.consume_challenge(path, challenge_id, bundle_id)
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.consume_challenge(path, challenge_id, _sha("replacement-bundle"))
    evidence.consume_challenge(path, _sha("new-challenge"), bundle_id)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM consumed").fetchone()[0] == 2


def test_replay_ledger_atomic_race_has_one_winner(tmp_path: Path) -> None:
    path = tmp_path / "consumed.sqlite"

    def consume(_: int) -> bool:
        try:
            evidence.consume_challenge(path, _sha("same-challenge"), _sha("same-bundle"))
            return True
        except evidence.LinuxEvidenceError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(consume, range(4))) == 1


@pytest.mark.parametrize("challenge,bundle", [("", "a" * 64), ("a" * 64, "PASS")])
def test_bad_ledger_identity_fails_closed(tmp_path: Path, challenge: str, bundle: str) -> None:
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.consume_challenge(tmp_path / "ledger.sqlite", challenge, bundle)


def test_unavailable_replay_ledger_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.consume_challenge(tmp_path / "missing" / "ledger.sqlite", _sha("c"), _sha("b"))


@pytest.mark.parametrize("size", [0, 31, 33, 64])
def test_signing_accepts_only_dedicated_raw_ed25519_key(size: int) -> None:
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.sign_record({}, bytes(size))


def test_issue_cli_refuses_missing_policy_and_does_not_write_pass(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    assert (
        evidence.main(
            [
                "issue-challenge",
                "--policy",
                str(tmp_path / "missing.json"),
                "--expected-policy-sha256",
                "a" * 64,
                "--source-revision",
                "a" * 40,
                "--ci-run-id",
                "1",
                "--ci-run-attempt",
                "1",
                "--host-id",
                "b" * 64,
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()


def test_full_verifier_derives_three_results_from_signed_raw_observations(
    transport: dict[str, Any],
) -> None:
    result = _verify(transport)
    assert [item["item"] for item in result["results"]] == list(evidence.LINUX_ITEMS)
    assert {item["status"] for item in result["results"]} == {"PASS"}
    assert result["linux_physical"]["bundle_sha256"] == evidence.digest(transport["bundle"])
    assert result["source_revision"] == transport["challenge"]["source_revision"]


@pytest.mark.parametrize(
    "mutation", ["pass", "root", "state", "state-replay", "failure", "postgres", "orphan"]
)
def test_valid_signature_does_not_replace_required_physical_observations(
    transport: dict[str, Any],
    mutation: str,
) -> None:
    payloads = _payloads(transport)
    if mutation == "pass":
        payloads[1]["observations"] = {"status": "PASS"}
    elif mutation == "root":
        payloads[1]["observations"]["snapshot"]["units"][0]["processes"][0]["uid"] = 0
    elif mutation == "state":
        payloads[4]["observations"]["snapshot"]["state"]["sha256"] = "c" * 64
    elif mutation == "state-replay":
        _bind_state_to_challenge(payloads, "c" * 64, transport["host"]["installation"])
    elif mutation == "failure":
        payloads[6]["observations"]["install_exit_code"] = 0
    elif mutation == "postgres":
        payloads[1]["observations"]["snapshot"]["postgres"]["system_user"] = "peer:caller-claim"
    else:
        payloads[2]["observations"]["stopped"][0]["old_processes_alive"] = [9000]
    transport["bundle"] = _signed_bundle(transport, payloads)
    with pytest.raises(evidence.LinuxEvidenceError):
        _verify(transport)


@pytest.mark.parametrize(
    "material",
    [
        "artifact_sha256",
        "update_artifact_sha256",
        "failed_update_artifact_sha256",
        "configuration_sha256",
    ],
)
def test_policy_material_must_match_pinned_installation_contract(
    transport: dict[str, Any],
    tmp_path: Path,
    material: str,
) -> None:
    policy = copy.deepcopy(transport["policy"])
    policy["hosts"][0]["materials"][material] = _sha("different-approved-claim")
    path = _write(tmp_path / "inconsistent-policy.json", policy)
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.load_policy(path, evidence.digest(policy))
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.issue_challenge(
            policy,
            source_revision=transport["challenge"]["source_revision"],
            ci_run_id="1",
            ci_run_attempt="1",
            host_id=transport["host"]["host_id"],
            now=NOW,
        )


def test_collection_prefix_rejects_unexpected_early_reboot(transport: dict[str, Any]) -> None:
    payloads = _payloads(transport)[:3]
    payloads[2]["host"]["boot_id"] = AFTER_BOOT
    transport["bundle"] = _signed_bundle(transport, payloads)
    with pytest.raises(evidence.LinuxEvidenceError):
        _validate(transport, complete=False)


def _current_files(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Path], Any, dict[str, Any]]:
    from deployment.platform_readiness import LinuxEvidenceContext

    context = _complete_transport(tmp_path, current=True)
    paths = {
        name: _write(tmp_path / (name + ".json"), context[name])
        for name in ("bundle", "challenge", "policy")
    }
    gate_context = LinuxEvidenceContext(
        paths["bundle"],
        paths["challenge"],
        paths["policy"],
        evidence.digest(context["policy"]),
        "2",
    )
    revision = context["challenge"]["source_revision"]
    verified = gate_context.verify(revision, "https://github.com", "12345")
    return context, paths, gate_context, verified


def _core_document(verified: dict[str, Any]) -> dict[str, Any]:
    core = {
        name: copy.deepcopy(value) for name, value in verified.items() if name != "linux_physical"
    }
    core["platform"] = "CROSS_PLATFORM_CORE"
    core["results"] = [
        {
            "item": "CORE_REQUIRED_SUITES",
            "status": "PASS",
            "evidence_class": "CROSS_OS_CI_MATRIX",
            "test_or_probe": "synthetic core fixture",
            "details": "TEST_ONLY; hermetyczny test bramki",
        }
    ]
    return core


def _gate_cli(
    context: dict[str, Any], paths: dict[str, Path], core_path: Path, ledger: Path
) -> list[str]:
    return [
        "--platform",
        "LINUX",
        "--current-revision",
        context["challenge"]["source_revision"],
        "--expected-ci-provider",
        "https://github.com",
        "--expected-ci-run-id",
        "12345",
        "--expected-ci-run-attempt",
        "2",
        "--evidence",
        str(core_path),
        "--linux-bundle",
        str(paths["bundle"]),
        "--linux-challenge",
        str(paths["challenge"]),
        "--linux-policy",
        str(paths["policy"]),
        "--linux-policy-sha256",
        evidence.digest(context["policy"]),
        "--linux-replay-ledger",
        str(ledger),
    ]


def test_gate_revalidates_raw_bundle_and_rejects_standalone_verified_json(tmp_path: Path) -> None:
    from deployment.platform_readiness import blocking_items, load_contract, production_ready

    context, paths, gate_context, verified = _current_files(tmp_path)
    revision = context["challenge"]["source_revision"]
    documents = [_core_document(verified), verified]
    assert production_ready(
        "LINUX", load_contract(), documents, revision, "https://github.com", "12345", gate_context
    )
    assert not production_ready(
        "LINUX", load_contract(), documents, revision, "https://github.com", "12345"
    )
    assert not production_ready(
        "LINUX", load_contract(), [verified], revision, "https://github.com", "12345", gate_context
    )
    tampered = copy.deepcopy(documents)
    tampered[-1]["results"][0]["details"] = "ręcznie wpisany PASS"
    assert not production_ready(
        "LINUX", load_contract(), tampered, revision, "https://github.com", "12345", gate_context
    )
    # Wynik niezależnego wcześniejszego verifiera nie zwalnia bramki z ponownego odczytu.
    changed_bundle = copy.deepcopy(context["bundle"])
    changed_bundle["records"][0]["signature"] = base64.b64encode(b"\0" * 64).decode()
    _write(paths["bundle"], changed_bundle)
    blockers = blocking_items(
        "LINUX", load_contract(), documents, revision, "https://github.com", "12345", gate_context
    )
    assert blockers and blockers[0].startswith("INVALID_EVIDENCE:")


def test_gate_cli_consumes_challenge_once_and_blocks_second_qualification(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from deployment.platform_readiness import main

    context, paths, _, verified = _current_files(tmp_path)
    core_path = _write(tmp_path / "core.json", _core_document(verified))
    ledger = tmp_path / "consumed.sqlite"
    command = _gate_cli(context, paths, core_path, ledger)
    assert main(command) == 0
    assert "LINUX_PRODUCTION_READY=PASS" in capsys.readouterr().out
    assert main(command) == 1
    output = capsys.readouterr().out
    assert "LINUX_PRODUCTION_READY=NOT_READY" in output
    assert "wykorzystany" in output


def test_incomplete_gate_does_not_consume_challenge(tmp_path: Path) -> None:
    from deployment.platform_readiness import main

    context, paths, _, verified = _current_files(tmp_path)
    core = _core_document(verified)
    core["results"][0]["status"] = "FAIL"
    core_path = _write(tmp_path / "core.json", core)
    ledger = tmp_path / "consumed.sqlite"
    command = _gate_cli(context, paths, core_path, ledger)
    assert main(command) == 1
    assert not ledger.exists()
    _write(core_path, _core_document(verified))
    assert main(command) == 0


@pytest.mark.parametrize(
    "option", ["--linux-policy", "--linux-replay-ledger", "--expected-ci-run-attempt"]
)
def test_gate_cli_requires_all_independent_inputs(tmp_path: Path, option: str) -> None:
    from deployment.platform_readiness import main

    context, paths, _, verified = _current_files(tmp_path)
    core_path = _write(tmp_path / "core.json", _core_document(verified))
    command = _gate_cli(context, paths, core_path, tmp_path / "ledger.sqlite")
    index = command.index(option)
    del command[index : index + 2]
    # Nie korzystaj przypadkowo z kontekstu wykonania testów CI.
    if option == "--expected-ci-run-attempt":
        command.extend(["--expected-ci-run-attempt", ""])
    assert main(command) == 1


def test_gate_cannot_promote_saved_json_without_raw_context(tmp_path: Path) -> None:
    from deployment.platform_readiness import main

    context, _, _, verified = _current_files(tmp_path)
    core_path = _write(tmp_path / "core.json", _core_document(verified))
    evidence_path = _write(tmp_path / "verified.json", verified)
    assert (
        main(
            [
                "--platform",
                "LINUX",
                "--current-revision",
                context["challenge"]["source_revision"],
                "--expected-ci-provider",
                "https://github.com",
                "--expected-ci-run-id",
                "12345",
                "--evidence",
                str(core_path),
                "--evidence",
                str(evidence_path),
            ]
        )
        == 1
    )


def _export_cli(context: dict[str, Any], paths: dict[str, Path], output: Path) -> list[str]:
    return [
        "linux-physical",
        "--source-revision",
        context["challenge"]["source_revision"],
        "--ci-run-id",
        "12345",
        "--ci-run-attempt",
        "2",
        "--bundle",
        str(paths["bundle"]),
        "--challenge",
        str(paths["challenge"]),
        "--policy",
        str(paths["policy"]),
        "--expected-policy-sha256",
        evidence.digest(context["policy"]),
        "--output",
        str(output),
    ]


def test_platform_evidence_exports_only_independently_verified_physical_bundle(
    tmp_path: Path,
) -> None:
    from deployment.platform_evidence import main

    context, paths, _, verified = _current_files(tmp_path)
    output = tmp_path / "linux-evidence.json"
    assert main(_export_cli(context, paths, output)) == 0
    assert evidence.read_json(output) == verified
    assert main(_export_cli(context, paths, output)) == 1  # istniejący raport nie jest nadpisywany


@pytest.mark.parametrize("mutation", ["missing", "signature", "revision", "attempt", "policy"])
def test_platform_export_failure_never_creates_pass(
    tmp_path: Path,
    mutation: str,
) -> None:
    from deployment.platform_evidence import main

    context, paths, _, _ = _current_files(tmp_path)
    output = tmp_path / "linux-evidence.json"
    command = _export_cli(context, paths, output)
    if mutation == "missing":
        paths["bundle"].unlink()
    elif mutation == "signature":
        context["bundle"]["records"][-1]["signature"] = "invalid"
        _write(paths["bundle"], context["bundle"])
    else:
        option = {
            "revision": "--source-revision",
            "attempt": "--ci-run-attempt",
            "policy": "--expected-policy-sha256",
        }[mutation]
        command[command.index(option) + 1] = (
            "7" if mutation == "attempt" else "a" * (40 if mutation == "revision" else 64)
        )
    assert main(command) == 1
    assert not output.exists()


def test_independent_verifier_cli_uses_same_raw_contract(tmp_path: Path) -> None:
    context, paths, _, verified = _current_files(tmp_path)
    output = tmp_path / "independently-verified.json"
    command = _export_cli(context, paths, output)
    command[0] = "verify"
    assert evidence.main(command) == 0
    assert evidence.read_json(output) == verified
    assert evidence.main(command) == 1


def test_issue_challenge_cli_does_not_claim_test_execution(tmp_path: Path) -> None:
    context, paths, _, _ = _current_files(tmp_path)
    output = tmp_path / "new-challenge.json"
    assert (
        evidence.main(
            [
                "issue-challenge",
                "--policy",
                str(paths["policy"]),
                "--expected-policy-sha256",
                evidence.digest(context["policy"]),
                "--source-revision",
                context["challenge"]["source_revision"],
                "--ci-run-id",
                "12345",
                "--ci-run-attempt",
                "2",
                "--host-id",
                context["host"]["host_id"],
                "--output",
                str(output),
            ]
        )
        == 0
    )
    challenge = evidence.load_challenge(output)
    assert "results" not in challenge
    assert challenge["challenge_id"] != context["challenge"]["challenge_id"]
