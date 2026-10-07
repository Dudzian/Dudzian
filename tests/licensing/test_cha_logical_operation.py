"""TEST_ONLY native ABI simulation through the real committed LPPI boundary.

The inherited fixture substitutes NCrypt/TBS and production trust context only.
CHA issuance, upstream provenance/current requalification, durable records and
real ECDSA signatures remain production code. No physical Windows qualification
or account reservation/genesis occurs here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from bot_core.licensing import (
    cha_logical_operation as operation,
    lppi_authenticated_operation as upstream_operation,
    pdsa_enrollment_authorization as authorization,
)
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.licensing.pre_enrollment import P256_ORDER, PDSA_TRUST_DOMAIN
from deployment import (
    windows_production_cha_operation as installed,
    windows_production_lppi_authority as lifecycle,
    windows_production_lppi_operation as upstream_installed,
)
from tests.deployment.test_windows_cng_pre_enrollment import wide
from tests.licensing import test_lppi_authenticated_operation as upstream_tests

challenge_harness = upstream_tests.challenge_harness
integration = upstream_tests.integration
package = upstream_tests.package
lppi = upstream_tests.lppi
active = upstream_tests.active
committed = upstream_tests.committed

ASSIGNMENT_NOW = upstream_tests.RESERVATION_NOW + timedelta(milliseconds=876)
STATE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "history",
        "mapping_generation",
        "pdsa_trust_domain",
        "provisioning_operation_id",
        "logical_operation_id",
        "assigned_at_utc",
        "lppi_operation_state_raw_hex",
    }
)


def _state():
    return installed._read(installed._state_path())


def _upstream_raw():
    return canonical_json_bytes(upstream_installed._read(upstream_installed._state_path()))


def _forbid_remint(*args, **kwargs):
    pytest.fail("retained CHA operation attempted new identity or assignment time")


def _changed_valid_upstream(committed, field):
    """Construct internally consistent transport; this never issues authority."""
    state = parse_canonical(_upstream_raw())
    original_binding = (
        upstream_operation.LPPIAuthenticatedProvisioningOperationBindingV1.from_canonical_bytes(
            bytes.fromhex(state["binding_raw_hex"])
        )
    )
    document = original_binding.document
    if field in document:
        if field == "pdsa_package_digest_sha256":
            document[field] = "cd" * 32
        else:
            previous = uuid.UUID(document[field].split("_", 1)[1])
            document[field] = (
                document[field].split("_", 1)[0] + "_" + str(uuid.UUID(int=previous.int ^ 1))
            )
        binding = upstream_operation.LPPIAuthenticatedProvisioningOperationBindingV1.from_mapping(
            document
        )
        signature = committed.lppi.successor.private.sign(
            upstream_operation.authenticated_operation_signed_bytes(binding.canonical_bytes),
            ec.ECDSA(hashes.SHA256()),
        )
        r, s = utils.decode_dss_signature(signature)
        state.update(
            **{field: document[field]},
            binding_raw_hex=binding.canonical_bytes.hex(),
            binding_digest_sha256=binding.digest_sha256,
            signature_hex=utils.encode_dss_signature(r, min(s, P256_ORDER - s)).hex(),
        )
        if field != "provisioning_operation_id":
            active_document = parse_canonical(
                bytes.fromhex(state["active_authority_binding_raw_hex"])
            )
            active_document[field] = document[field]
            active_raw = canonical_json_bytes(active_document)
            state["active_authority_binding_raw_hex"] = active_raw.hex()
            state["lppi_authority_key_binding_digest_sha256"] = hashlib.sha256(
                active_raw
            ).hexdigest()
            source = {name: state[name] for name in upstream_installed._SOURCE_FIELDS}
            state["reservation_digest_sha256"] = upstream_operation.reservation_digest(source)
    elif field == "signature_hex":
        binding = original_binding
        old = state[field]
        while state[field] == old:
            signature = committed.lppi.successor.private.sign(
                upstream_operation.authenticated_operation_signed_bytes(binding.canonical_bytes),
                ec.ECDSA(hashes.SHA256()),
            )
            r, s = utils.decode_dss_signature(signature)
            state[field] = utils.encode_dss_signature(r, min(s, P256_ORDER - s)).hex()
    else:
        raise AssertionError(field)
    upstream_installed._validate_state(state)
    return canonical_json_bytes(state)


@pytest.fixture
def cha(committed, monkeypatch):
    monkeypatch.setattr(installed, "_utc_now", lambda: ASSIGNMENT_NOW)
    return installed.establish_installed_cha_logical_operation(committed.value)


def test_assignment_uses_one_full_millisecond_instant_and_csprng(committed, monkeypatch):
    captured, random_calls = [], []
    original = authorization.secrets.randbits

    def clock():
        captured.append(ASSIGNMENT_NOW)
        return ASSIGNMENT_NOW

    def randomness(bits):
        value = original(bits)
        random_calls.append((bits, value))
        return value

    monkeypatch.setattr(installed, "_utc_now", clock)
    monkeypatch.setattr(authorization.secrets, "randbits", randomness)
    value = installed.establish_installed_cha_logical_operation(committed.value)
    state = _state()
    identity = uuid.UUID(value.logical_operation_id.removeprefix("ago_"))
    millis = authorization._reservation_epoch_milliseconds(ASSIGNMENT_NOW)
    assert type(value) is operation.VerifiedCHALogicalOperation
    assert identity.version == 7 and identity.variant == uuid.RFC_4122
    assert value.logical_operation_id == "ago_" + str(identity)
    assert identity.int >> 80 == millis
    assert millis % 1000 == 999
    assert state["assigned_at_utc"].endswith(".999Z")
    assert captured == [ASSIGNMENT_NOW]
    assert [bits for bits, _ in random_calls] == [12, 62]
    assert (identity.int >> 64) & ((1 << 12) - 1) == random_calls[0][1]
    assert identity.int & ((1 << 62) - 1) == random_calls[1][1]
    assert set(state) == STATE_FIELDS
    assert state["schema_version"] == "CHAInitialLogicalOperationV1"
    assert state["mapping_generation"] == 1
    assert state["pdsa_trust_domain"] == PDSA_TRUST_DOMAIN
    assert state["provisioning_operation_id"] == committed.value.provisioning_operation_id
    assert state["status"] == "PRVOP_AGO_BIJECTION_COMMITTED"
    assert state["history"] == list(installed.STATUSES)
    assert state["lppi_operation_state_raw_hex"] == _upstream_raw().hex()


def test_caller_cannot_select_identity_or_assignment_time(committed):
    for field in ["ago", "logical_operation_id", "uuid", "timestamp", "clock"]:
        with pytest.raises(TypeError):
            installed.establish_installed_cha_logical_operation(
                committed.value, **{field: "caller-controlled"}
            )
        assert _state() is None


def test_invalid_clock_fails_before_uuid_or_reservation(committed, monkeypatch):
    monkeypatch.setattr(authorization.secrets, "randbits", _forbid_remint)
    for instant in [
        datetime(1969, 12, 31, 23, 59, 59, 999000, tzinfo=timezone.utc),
        datetime(2026, 10, 6, 12, 0, 0),
        datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone(timedelta(hours=1))),
    ]:
        monkeypatch.setattr(installed, "_utc_now", lambda instant=instant: instant)
        with pytest.raises((ValueError, RuntimeError)):
            installed.establish_installed_cha_logical_operation(committed.value)
        assert _state() is None


def test_raw_copied_or_forged_upstream_cannot_issue_cha_authority(committed):
    value = committed.value
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT ? AS provisioning_operation_id", (value.provisioning_operation_id,)
        ).fetchone()
    sources = {
        "none": lambda: None,
        "binding": lambda: value.binding,
        "record": lambda: parse_canonical(_upstream_raw()),
        "json": _upstream_raw,
        "prvop": lambda: value.provisioning_operation_id,
        "signature": lambda: value.signature,
        "dto": lambda: {"provisioning_operation_id": value.provisioning_operation_id},
        "row": lambda: row,
        "copy": lambda: copy.copy(value),
        "deepcopy": lambda: copy.deepcopy(value),
        "new": lambda: object.__new__(type(value)),
        "subclass": lambda: object.__new__(type("TestOnlyForgedUpstream", (type(value),), {})),
    }
    for create in sources.values():
        source = create()
        for consume in (
            installed.establish_installed_cha_logical_operation,
            installed.load_installed_cha_logical_operation,
        ):
            with pytest.raises((ValueError, RuntimeError)):
                consume(source)
        assert _state() is None


def test_exact_commit_source_transport_and_capability_are_immutable(cha, committed):
    assert operation.require_verified_cha_logical_operation(cha) is cha
    assert cha.provisioning_operation_id == committed.value.provisioning_operation_id
    assert cha.source_binding.canonical_bytes == committed.value.binding.canonical_bytes
    document = cha.source_binding.document
    document["provisioning_operation_id"] = "caller-controlled"
    assert cha.source_binding.canonical_bytes == committed.value.binding.canonical_bytes
    source_tuple = cha.source_tuple
    assert (
        source_tuple["lppi_authenticated_operation_binding_digest_sha256"]
        == committed.value.binding.digest_sha256
    )
    source_tuple["provisioning_operation_id"] = "caller-controlled"
    assert (
        cha.source_tuple["provisioning_operation_id"] == committed.value.provisioning_operation_id
    )
    with pytest.raises(TypeError):
        cha.logical_operation_id = "ago_caller-controlled"
    with pytest.raises(TypeError):
        operation.VerifiedCHALogicalOperation()


def test_reconstructed_or_copied_cha_never_grants_authority(cha):
    forgeries = {
        "copy": lambda: copy.copy(cha),
        "deepcopy": lambda: copy.deepcopy(cha),
        "new": lambda: object.__new__(type(cha)),
        "subclass": lambda: object.__new__(type("TestOnlyForgedCHA", (type(cha),), {})),
        "record": _state,
        "json": lambda: canonical_json_bytes(_state()),
        "binding": lambda: cha.source_binding,
    }
    for create in forgeries.values():
        forged = create()
        with pytest.raises((ValueError, RuntimeError)):
            operation.require_verified_cha_logical_operation(forged)
    assert operation.require_verified_cha_logical_operation(cha) is cha


def test_retry_and_lost_response_retain_mapping_without_remint(cha, committed, monkeypatch):
    identity, prvop, source = (
        cha.logical_operation_id,
        cha.provisioning_operation_id,
        cha.source_binding.canonical_bytes,
    )
    retained = installed._state_path().read_bytes()
    monkeypatch.setattr(authorization.secrets, "randbits", _forbid_remint)
    monkeypatch.setattr(installed, "_utc_now", _forbid_remint)
    for retry in (
        installed.establish_installed_cha_logical_operation(committed.value),
        installed.load_installed_cha_logical_operation(committed.value),
    ):
        assert retry.logical_operation_id == identity
        assert retry.provisioning_operation_id == prvop
        assert retry.source_binding.canonical_bytes == source
    assert installed._state_path().read_bytes() == retained


def test_restart_requires_reverified_upstream_and_reconstructs_exact_mapping(
    cha, committed, monkeypatch
):
    identity, source = cha.logical_operation_id, cha.source_binding.canonical_bytes
    retained = installed._state_path().read_bytes()
    operation._ISSUED.clear()
    upstream_operation._ISSUED.clear()
    committed.active.close()
    with pytest.raises((ValueError, RuntimeError)):
        operation.require_verified_cha_logical_operation(cha)
    fresh_active = lifecycle.establish_installed_lppi_authority_key(committed.lppi.accept())
    try:
        fresh_upstream = upstream_installed.load_installed_lppi_authenticated_operation(
            fresh_active
        )
        monkeypatch.setattr(authorization.secrets, "randbits", _forbid_remint)
        monkeypatch.setattr(installed, "_utc_now", _forbid_remint)
        loaded = installed.load_installed_cha_logical_operation(fresh_upstream)
        assert loaded.logical_operation_id == identity
        assert loaded.source_binding.canonical_bytes == source
        assert operation.require_verified_cha_logical_operation(loaded) is loaded
        assert installed._state_path().read_bytes() == retained
    finally:
        fresh_active.close()


@pytest.mark.parametrize(
    "mutation",
    ["closed", "revoked", "lppi_record", "authority_record", "unique_name", "tpmt_public"],
)
def test_current_upstream_requalification_revokes_all_cha_accesses(cha, committed, mutation):
    if mutation == "closed":
        committed.active.close()
    elif mutation == "revoked":
        upstream_operation._ISSUED.pop(committed.value)
    elif mutation == "lppi_record":
        state = upstream_installed._read(upstream_installed._state_path())
        state["signature_hex"] = "00"
        upstream_installed._state_path().write_bytes(canonical_json_bytes(state))
    elif mutation == "authority_record":
        state = lifecycle._read(lifecycle._state_path())
        document = committed.active.binding.document
        document["custody_profile"] = "TEST_ONLY"
        state["binding_raw_hex"] = canonical_json_bytes(document).hex()
        lifecycle._write(lifecycle._state_path(), state)
    elif mutation == "unique_name":
        committed.lppi.successor.properties[(22, "Unique Name")] = wide("TEST_ONLY-foreign-unique")
    else:
        committed.lppi.successor_tbs.subject = bytes.fromhex("00" * 64)
    for consume in (
        lambda: operation.require_verified_cha_logical_operation(cha),
        lambda: cha.provisioning_operation_id,
        lambda: cha.logical_operation_id,
        lambda: cha.source_binding,
        lambda: cha.source_tuple,
        lambda: installed.load_installed_cha_logical_operation(committed.value),
        lambda: installed.establish_installed_cha_logical_operation(committed.value),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            consume()


def test_upstream_closed_after_guard_fails_before_reservation_or_lock(committed, monkeypatch):
    original = installed.require_verified_lppi_authenticated_operation

    def guard_then_close(value):
        verified = original(value)
        committed.active.close()
        return verified

    monkeypatch.setattr(
        installed, "require_verified_lppi_authenticated_operation", guard_then_close
    )
    monkeypatch.setattr(installed, "_locked_state", _forbid_remint)
    with pytest.raises((ValueError, RuntimeError)):
        installed.establish_installed_cha_logical_operation(committed.value)
    assert _state() is None


def test_live_upstream_requalification_runs_outside_cha_write_lock(committed, monkeypatch):
    original_source = installed._source
    observations = []

    def source(*args):
        with installed._locked_state():
            observations.append("qualified")
        return original_source(*args)

    monkeypatch.setattr(installed, "_source", source)
    value = installed.establish_installed_cha_logical_operation(committed.value)
    assert operation.require_verified_cha_logical_operation(value) is value
    assert len(observations) >= 2


def test_loader_never_mints_or_publishes_missing_or_reserved_mapping(committed, monkeypatch):
    original_randomness = authorization.secrets.randbits
    monkeypatch.setattr(authorization.secrets, "randbits", _forbid_remint)
    with pytest.raises(installed.CHAOperationError, match="CHA_LOGICAL_OPERATION_NOT_COMMITTED"):
        installed.load_installed_cha_logical_operation(committed.value)
    assert _state() is None
    monkeypatch.setattr(authorization.secrets, "randbits", original_randomness)
    with installed._locked_state() as path:
        reserved = installed._reserve(path, _upstream_raw())
    assert reserved["status"] == "AGO_RESERVED"
    monkeypatch.setattr(authorization.secrets, "randbits", _forbid_remint)
    with pytest.raises(installed.CHAOperationError, match="CHA_LOGICAL_OPERATION_NOT_COMMITTED"):
        installed.load_installed_cha_logical_operation(committed.value)
    assert _state() == reserved


@pytest.mark.parametrize(
    "cut", ["before_reserve", "after_reserve", "before_commit", "after_commit", "before_response"]
)
def test_crash_cuts_preserve_reserved_identity_and_never_publish_early(committed, monkeypatch, cut):
    original_write = installed._write
    original_issue = installed._issue_verified_cha_operation
    snapshots, issued = [], []

    def crash():
        raise RuntimeError("TEST_ONLY_CHA_CRASH")

    def write(path, state):
        reserved = state["status"] == "AGO_RESERVED"
        if (reserved and cut == "before_reserve") or (not reserved and cut == "before_commit"):
            crash()
        original_write(path, state)
        snapshots.append(dict(state))
        if (reserved and cut == "after_reserve") or (not reserved and cut == "after_commit"):
            crash()

    def issue(*args, **kwargs):
        assert _state()["status"] == "PRVOP_AGO_BIJECTION_COMMITTED"
        if cut == "before_response":
            crash()
        result = original_issue(*args, **kwargs)
        issued.append(result)
        return result

    monkeypatch.setattr(installed, "_write", write)
    monkeypatch.setattr(installed, "_issue_verified_cha_operation", issue)
    with pytest.raises(RuntimeError, match="TEST_ONLY_CHA_CRASH"):
        installed.establish_installed_cha_logical_operation(committed.value)
    assert not issued
    retained = _state()
    if cut == "before_reserve":
        assert retained is None
    else:
        assert retained is not None
        monkeypatch.setattr(authorization.secrets, "randbits", _forbid_remint)
        monkeypatch.setattr(installed, "_utc_now", _forbid_remint)
    monkeypatch.setattr(installed, "_write", original_write)
    monkeypatch.setattr(installed, "_issue_verified_cha_operation", original_issue)
    result = installed.establish_installed_cha_logical_operation(committed.value)
    if retained is not None:
        assert result.logical_operation_id == retained["logical_operation_id"]
        assert result.provisioning_operation_id == retained["provisioning_operation_id"]
        assert _state()["assigned_at_utc"] == retained["assigned_at_utc"]
        assert _state()["lppi_operation_state_raw_hex"] == retained["lppi_operation_state_raw_hex"]
    if cut in {"after_commit", "before_response"}:
        assert _state() == retained
    assert len({state["logical_operation_id"] for state in snapshots}) <= 1


@pytest.mark.parametrize("status", ["AGO_RESERVED", "PRVOP_AGO_BIJECTION_COMMITTED"])
def test_exact_source_conflict_cannot_claim_retained_mapping(committed, status):
    with installed._locked_state() as path:
        reserved = installed._reserve(path, _upstream_raw())
        if status == "PRVOP_AGO_BIJECTION_COMMITTED":
            installed._commit(path, canonical_json_bytes(reserved))
    retained = installed._state_path().read_bytes()
    for field in [
        "provisioning_operation_id",
        "pdsa_package_digest_sha256",
        "provisioning_subject_id",
        "enrollment_reference",
        "signature_hex",
    ]:
        changed = _changed_valid_upstream(committed, field)
        with installed._locked_state() as path:
            with pytest.raises(
                installed.CHAOperationError, match="^CHA_LOGICAL_OPERATION_CONFLICT$"
            ):
                installed._reserve(path, changed)
        assert installed._state_path().read_bytes() == retained


@pytest.mark.parametrize("status", ["AGO_RESERVED", "PRVOP_AGO_BIJECTION_COMMITTED"])
def test_immutable_record_rejects_another_valid_ago(committed, status):
    with installed._locked_state() as path:
        reserved = installed._reserve(path, _upstream_raw())
        state = (
            reserved
            if status == "AGO_RESERVED"
            else installed._commit(path, canonical_json_bytes(reserved))
        )
        identity = uuid.UUID(state["logical_operation_id"].removeprefix("ago_"))
        changed = state | {"logical_operation_id": "ago_" + str(uuid.UUID(int=identity.int ^ 1))}
        retained = path.read_bytes()
        with pytest.raises(installed.CHAOperationError, match="^CHA_LOGICAL_OPERATION_CONFLICT$"):
            installed._write(path, changed)
    assert installed._state_path().read_bytes() == retained


def test_reverse_mapping_cannot_rebind_valid_ago_to_second_authenticated_prvop(cha, committed):
    state = _state()
    changed_raw = _changed_valid_upstream(committed, "provisioning_operation_id")
    second_prvop = parse_canonical(changed_raw)["provisioning_operation_id"]
    assert second_prvop != state["provisioning_operation_id"]
    replacement = state | {
        "provisioning_operation_id": second_prvop,
        "lppi_operation_state_raw_hex": changed_raw.hex(),
    }
    retained = installed._state_path().read_bytes()
    with pytest.raises(installed.CHAOperationError, match="^CHA_LOGICAL_OPERATION_CONFLICT$"):
        installed._write(installed._state_path(), replacement)
    assert installed._state_path().read_bytes() == retained
    assert operation.require_verified_cha_logical_operation(cha) is cha


def test_structurally_valid_source_replacement_revokes_exact_retained_capability(cha, committed):
    retained = _state()
    for field in [
        "provisioning_operation_id",
        "pdsa_package_digest_sha256",
        "provisioning_subject_id",
        "enrollment_reference",
        "signature_hex",
    ]:
        state = dict(retained)
        changed_raw = _changed_valid_upstream(committed, field)
        state["lppi_operation_state_raw_hex"] = changed_raw.hex()
        state["provisioning_operation_id"] = parse_canonical(changed_raw)[
            "provisioning_operation_id"
        ]
        installed._state_path().write_bytes(canonical_json_bytes(state))
        for consume in (
            lambda: operation.require_verified_cha_logical_operation(cha),
            lambda: cha.source_binding,
            lambda: installed.load_installed_cha_logical_operation(committed.value),
            lambda: installed.establish_installed_cha_logical_operation(committed.value),
        ):
            with pytest.raises((ValueError, RuntimeError)):
                consume()


def test_every_missing_record_field_fails_closed(cha):
    retained = _state()
    for field in sorted(STATE_FIELDS):
        state = dict(retained)
        del state[field]
        installed._state_path().write_bytes(canonical_json_bytes(state))
        with pytest.raises((ValueError, RuntimeError)):
            operation.require_verified_cha_logical_operation(cha)


def test_record_profile_identity_generation_timestamp_or_source_mutation_fails_closed(
    cha, committed
):
    retained = _state()
    for field, replacement in [
        ("schema_version", "other-schema"),
        ("status", "AGO_RESERVED"),
        ("history", ["PRVOP_AGO_BIJECTION_COMMITTED"]),
        ("mapping_generation", True),
        ("mapping_generation", "1"),
        ("mapping_generation", 2),
        ("pdsa_trust_domain", "TEST_ONLY"),
        ("provisioning_operation_id", "prvop_caller"),
        ("logical_operation_id", "ago_caller"),
        ("logical_operation_id", "ago_019ba13c-5c00-4000-8000-000000000001"),
        ("logical_operation_id", "ago_019BA13C-5c00-7000-8000-000000000001"),
        ("logical_operation_id", "ago_019ba13c-5c00-7000-0000-000000000001"),
        ("logical_operation_id", "ago_019ba13c-5c00-7000-8000-000000000001\n"),
        ("assigned_at_utc", "2026-10-06T12:00:00Z"),
        ("assigned_at_utc", "2026-10-06T12:00:00.12Z"),
        ("assigned_at_utc", "2026-10-06T12:00:00.1234Z"),
        ("assigned_at_utc", "2026-10-06T12:00:00.123+00:00"),
        ("assigned_at_utc", "2026-02-30T12:00:00.123Z"),
        ("assigned_at_utc", None),
        ("lppi_operation_state_raw_hex", "00"),
        ("lppi_operation_state_raw_hex", ""),
    ]:
        state = retained | {field: replacement}
        installed._state_path().write_bytes(canonical_json_bytes(state))
        for consume in (
            lambda: operation.require_verified_cha_logical_operation(cha),
            lambda: installed.load_installed_cha_logical_operation(committed.value),
            lambda: installed.establish_installed_cha_logical_operation(committed.value),
        ):
            with pytest.raises((ValueError, RuntimeError)):
                consume()


def test_unknown_duplicate_or_noncanonical_record_fails_closed(cha):
    state = _state()
    canonical = canonical_json_bytes(state)
    corruptions = {
        "extra": lambda: canonical_json_bytes(state | {"account_id": "caller-controlled"}),
        "duplicate": lambda: b'{"schema_version":"CHAInitialLogicalOperationV1",' + canonical[1:],
        "space": lambda: json.dumps(state).encode(),
        "order": lambda: json.dumps(
            dict(reversed(list(state.items()))), separators=(",", ":")
        ).encode(),
        "newline": lambda: canonical + b"\n",
        "array": lambda: b"[]",
        "utf8": lambda: b"\xff",
    }
    for create in corruptions.values():
        malformed = create()
        assert malformed != canonical
        installed._state_path().write_bytes(malformed)
        with pytest.raises((ValueError, RuntimeError)):
            operation.require_verified_cha_logical_operation(cha)


@pytest.mark.skipif(os.name == "nt", reason="TEST_ONLY inherited native ABI requires fork")
def test_two_processes_keep_one_mapping_and_nonblocking_busy(committed, monkeypatch):
    context = multiprocessing.get_context("fork")
    released = context.Event()
    first_reader, first_writer = context.Pipe(duplex=False)
    second_reader, second_writer = context.Pipe(duplex=False)
    original = installed._write

    def reserved_then_pause(path, state):
        original(path, state)
        if state["status"] == "AGO_RESERVED":
            first_writer.send(("reserved", state["logical_operation_id"]))
            if not released.wait(15):
                raise RuntimeError("TEST_ONLY_CHA_RELEASE_TIMEOUT")

    def establish(pipe):
        try:
            value = installed.establish_installed_cha_logical_operation(committed.value)
            pipe.send(
                (
                    "ok",
                    value.logical_operation_id,
                    value.provisioning_operation_id,
                    value.source_binding.canonical_bytes,
                )
            )
        except (ValueError, RuntimeError) as exc:
            pipe.send(("error", str(exc)))
        finally:
            pipe.close()

    monkeypatch.setattr(installed, "_write", reserved_then_pause)
    first = context.Process(target=establish, args=(first_writer,))
    second = context.Process(target=establish, args=(second_writer,))
    first.start()
    try:
        assert first_reader.poll(10), "first process did not durably reserve"
        status, identity = first_reader.recv()
        assert status == "reserved"
        second.start()
        assert second_reader.poll(5), "second process blocked instead of returning stable BUSY"
        assert second_reader.recv() == ("error", "CHA_LOGICAL_OPERATION_BUSY")
        released.set()
        assert first_reader.poll(60), "first process did not commit after release"
        status, winner, prvop, source = first_reader.recv()
        assert status == "ok" and winner == identity
        first.join(10)
        second.join(10)
        assert first.exitcode == second.exitcode == 0
        monkeypatch.setattr(installed, "_write", original)
        monkeypatch.setattr(authorization.secrets, "randbits", _forbid_remint)
        retry = installed.establish_installed_cha_logical_operation(committed.value)
        assert retry.logical_operation_id == winner
        assert retry.provisioning_operation_id == prvop
        assert retry.source_binding.canonical_bytes == source
        assert _state()["logical_operation_id"] == winner
    finally:
        released.set()
        for process in (first, second):
            if process.pid is not None:
                process.join(10)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
        first_reader.close()
        second_reader.close()
