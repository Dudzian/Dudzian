"""Real Linux TEST mutation processes, with synthetic close and same-UID fixtures.

Selected by the existing Linux integration runner. They demonstrate process
crash/restart ordering, not installed principals or power-loss qualification.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from bot_core.authenticated_custody_mutation_source import (
    SourceMutationVerificationError,
    parse_custody_mutation_execution,
    parse_source_verification_policy,
    verify_custody_mutation_execution,
)
from bot_core.root_proof_issuer_substrate import ProviderRole
from tests.security._custody_source_demo import _before, _operation, start_process

pytestmark = pytest.mark.integration

REQUEST = {"operation_id": "controlled-operation-1", "target_state": "VERIFY_ONLY"}


def _receive(process, connection):
    assert connection.poll(15), "controlled TEST process did not respond"
    result = connection.recv()
    process.join(15)
    if process.is_alive():
        process.kill()
        process.join(5)
        pytest.fail("controlled TEST process did not stop")
    connection.close()
    assert process.exitcode == 0
    return result


def _run(config, request=None, **arguments):
    process, connection = start_process(
        config=config, request=REQUEST if request is None else request, **arguments
    )
    return _receive(process, connection)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "controlled-source"
    root.mkdir(mode=0o700)
    process, connection = start_process(root=str(root), role=ProviderRole.ROOT_PROOF_SIGNING.value)
    status, config = _receive(process, connection)
    assert status == "ok"
    assert config["public_key"] and "private_key" not in config and "seed" not in config
    return config


def _policy(config):
    return parse_source_verification_policy(
        (Path(config["root"]) / "events" / "public-policy").read_bytes()
    )


def test_controlled_operation_returns_original_authenticated_bytes_but_no_authority(source):
    status, response = _run(source)
    assert status == "ok"
    event = parse_custody_mutation_execution(response["event"])
    verified = verify_custody_mutation_execution(event, policy_loader=lambda: _policy(source))
    assert verified.status == "NOT_QUALIFIED"
    assert response["status"] == "NOT_QUALIFIED"
    assert response["production_emission"] == "BLOCKED"
    assert response["close"] == "SYNTHETIC_TEST_CLAIM_NOT_QUALIFIED_POSTGRESQL_CLOSE"
    root = Path(source["root"])
    assert response["event"] == (root / "events" / "original").read_bytes()
    assert (
        json.loads((root / "journal" / "operation").read_bytes())["completion_state"] == "COMPLETE"
    )


@pytest.mark.parametrize("stage", ["before_complete", "after_complete", "after_event"])
def test_crash_restart_recovers_exact_operation_and_retains_original_without_resigning(
    source, stage, monkeypatch
):
    process, connection = start_process(config=source, request=REQUEST, crash_at=stage)
    process.join(15)
    assert not process.is_alive() and process.exitcode == 73
    connection.close()
    root = Path(source["root"])
    journal = json.loads((root / "journal" / "operation").read_bytes())
    assert journal["completion_state"] == ("INTENT" if stage == "before_complete" else "COMPLETE")
    original = root / "events" / "original"
    assert original.exists() == (stage == "after_event")
    if stage == "after_event":
        retained, info = original.read_bytes(), original.stat()

        def forbidden_signing(*args, **kwargs):
            raise AssertionError("exact recovery attempted to construct/sign a new event")

        monkeypatch.setattr(
            "tests.security._authenticated_custody_source_fixtures.build_source_case",
            forbidden_signing,
        )
    status, response = _run(source)
    assert status == "ok" and response["status"] == "NOT_QUALIFIED"
    if stage == "after_event":
        assert response["recovered"] is True
        assert response["event"] == retained
        assert (original.stat().st_ino, original.stat().st_mtime_ns) == (
            info.st_ino,
            info.st_mtime_ns,
        )
    else:
        assert response["recovered"] is False
    assert (
        json.loads((root / "journal" / "operation").read_bytes())["completion_state"] == "COMPLETE"
    )


@pytest.mark.parametrize(
    "command",
    [
        {"operation": "SIGN", "payload": "arbitrary"},
        {**REQUEST, "operation_descriptor_b64": "e30="},
        {**REQUEST, "source_event_sha256": "a" * 64},
        {**REQUEST, "role": ProviderRole.HISTORY_ATTESTATION_SIGNING.value},
        {**REQUEST, "public_key_b64": "caller-selected"},
        {**REQUEST, "installation_sha256": "a" * 64},
        {**REQUEST, "trust_pin_sha256": "b" * 64},
        {**REQUEST, "target_state": "ACTIVE"},
    ],
)
def test_closed_request_cannot_sign_arbitrary_dto_or_supply_source_authority(source, command):
    root = Path(source["root"])
    before = (root / "metadata").read_bytes()
    response = _run(source, command)
    assert response[0] == "refused"
    assert not (root / "journal" / "operation").exists()
    assert not (root / "events" / "original").exists()
    assert (root / "metadata").read_bytes() == before


@pytest.mark.parametrize(
    "competitor",
    [
        REQUEST,
        {**REQUEST, "operation_id": "other-operation"},
        {**REQUEST, "target_state": "REVOKED"},
    ],
)
def test_real_flock_serializes_competitors_and_same_operation_conflicts(source, competitor):
    first, first_pipe = start_process(config=source, request=REQUEST, pause_at="locked")
    assert first_pipe.poll(15) and first_pipe.recv() == ("barrier", "locked")
    second, second_pipe = start_process(config=source, request=competitor)
    try:
        assert not second_pipe.poll(0.2), "second process crossed the live custody lock"
        assert second.is_alive()
        first_pipe.send("resume")
        status, result = _receive(first, first_pipe)
        assert status == "ok"
        other = _receive(second, second_pipe)
        if competitor == REQUEST:
            assert other[0] == "ok" and other[1]["event"] == result["event"]
            assert other[1]["recovered"] is True
        else:
            assert other[0] == "refused" and "conflict" in other[2]
    finally:
        for process in (first, second):
            if process.is_alive():
                process.kill()
                process.join(5)


@pytest.mark.parametrize(
    "attack",
    [
        "journal",
        "journal_symlink",
        "journal_directory",
        "credential_directory",
        "key_inode",
        "key_material",
        "revoke",
        "rotate",
        "permissions",
    ],
)
def test_mid_execution_replacement_and_source_key_lifecycle_refuse_positive_event(source, attack):
    process, connection = start_process(config=source, request=REQUEST, pause_at="after_complete")
    assert connection.poll(15) and connection.recv() == ("barrier", "after_complete")
    root = Path(source["root"])
    journal, seed = root / "journal" / "operation", root / "credential" / "seed"
    if attack == "journal":
        journal.write_bytes(b"{}")
    elif attack == "journal_symlink":
        journal.rename(journal.with_name("saved"))
        journal.symlink_to("saved")
    elif attack.endswith("directory"):
        directory = root / ("journal" if attack == "journal_directory" else "credential")
        directory.rename(directory.with_name(directory.name + "-saved"))
        directory.mkdir(mode=0o700)
    elif attack == "key_inode":
        temporary = seed.with_name("replacement")
        shutil.copyfile(seed, temporary)
        temporary.chmod(0o400)
        temporary.replace(seed)
    elif attack == "key_material":
        seed.chmod(0o600)
        seed.write_bytes(b"x" * 32)
        seed.chmod(0o400)
    elif attack in {"revoke", "rotate"}:
        (root / "credential" / "lifecycle").write_bytes(
            b"REVOKED:1:2" if attack == "revoke" else b"ACTIVE:2:2"
        )
    else:
        seed.chmod(0o444)
    connection.send("resume")
    assert _receive(process, connection)[0] == "refused"
    assert not (root / "events" / "original").exists()


def test_retained_event_tampering_never_becomes_recovery(source):
    assert _run(source)[0] == "ok"
    event = Path(source["root"]) / "events" / "original"
    event.write_bytes(b"{}")
    assert _run(source)[0] == "refused"


@pytest.mark.parametrize("phase", ["INTENT", "COMPLETE"])
def test_preinjected_journal_and_metadata_cannot_become_own_execution(source, phase):
    from dataclasses import replace

    from bot_core import local_signing_custody as custody
    from bot_core import local_signing_lifecycle_evidence as lifecycle

    root = Path(source["root"])
    record = _operation(root, _before(source["role"]), REQUEST)
    journal = root / "journal" / "operation"
    journal.write_bytes(lifecycle._document(replace(record, state=phase)))
    journal.chmod(0o600)
    (root / "metadata").write_bytes(custody._encode_record(record.after))
    assert _run(source)[0] == "refused"
    assert not (root / "events" / "original").exists()


def _replace_same_bytes(path):
    replacement = path.with_name(path.name + "-replacement")
    shutil.copyfile(path, replacement)
    replacement.chmod(0o600)
    replacement.replace(path)


def test_initial_metadata_inode_replacement_is_not_an_owned_predecessor(source):
    _replace_same_bytes(Path(source["root"]) / "metadata")
    assert _run(source)[0] == "refused"
    assert not (Path(source["root"]) / "journal" / "operation").exists()


@pytest.mark.parametrize("stage", ["before_complete", "after_complete", "after_event"])
@pytest.mark.parametrize("artifact", ["journal/operation", "metadata"])
def test_own_execution_rejects_same_content_inode_replacement(source, stage, artifact):
    process, connection = start_process(config=source, request=REQUEST, pause_at=stage)
    assert connection.poll(15) and connection.recv() == ("barrier", stage)
    _replace_same_bytes(Path(source["root"]) / artifact)
    connection.send("resume")
    assert _receive(process, connection)[0] == "refused"
    assert _run(source)[0] == "refused"


@pytest.mark.parametrize("artifact", ["events/original", "events/public-policy"])
def test_original_response_and_recovery_reject_event_artifact_inode_replacement(source, artifact):
    process, connection = start_process(config=source, request=REQUEST, pause_at="after_event")
    assert connection.poll(15) and connection.recv() == ("barrier", "after_event")
    _replace_same_bytes(Path(source["root"]) / artifact)
    connection.send("resume")
    assert _receive(process, connection)[0] == "refused"
    assert _run(source)[0] == "refused"


def test_journal_tampering_after_event_write_prevents_positive_response(source):
    process, connection = start_process(config=source, request=REQUEST, pause_at="after_event")
    assert connection.poll(15) and connection.recv() == ("barrier", "after_event")
    (Path(source["root"]) / "journal" / "operation").write_bytes(b"{}")
    connection.send("resume")
    assert _receive(process, connection)[0] == "refused"
    assert _run(source)[0] == "refused"


def test_root_event_cannot_be_replayed_under_history_policy(source, tmp_path):
    assert _run(source)[0] == "ok"
    root_event = parse_custody_mutation_execution(
        (Path(source["root"]) / "events" / "original").read_bytes()
    )
    history = tmp_path / "history"
    history.mkdir(mode=0o700)
    process, connection = start_process(
        root=str(history), role=ProviderRole.HISTORY_ATTESTATION_SIGNING.value
    )
    status, config = _receive(process, connection)
    assert status == "ok" and _run(config)[0] == "ok"
    with pytest.raises(SourceMutationVerificationError):
        verify_custody_mutation_execution(root_event, policy_loader=lambda: _policy(config))
