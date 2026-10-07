"""Hosted TEST_ONLY NCrypt/TBS ABI; real upstream guards, signing and persistence."""

from __future__ import annotations

import copy
import hashlib
import inspect
import uuid
from datetime import timedelta

import pytest

from bot_core import uuid7
from bot_core.licensing import (
    cha_account_reservation as capability,
    cha_logical_operation as cha_capability,
)
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from deployment import (
    windows_production_cha_account_reservation as installed,
    windows_production_cha_operation as cha_installed,
)
from tests.licensing import test_cha_logical_operation as upstream_tests

wide = upstream_tests.wide
challenge_harness = upstream_tests.challenge_harness
integration = upstream_tests.integration
package = upstream_tests.package
lppi = upstream_tests.lppi
active = upstream_tests.active
committed = upstream_tests.committed
cha = upstream_tests.cha
NOW = upstream_tests.ASSIGNMENT_NOW + timedelta(milliseconds=137)


@pytest.fixture
def reserved(cha, monkeypatch):
    monkeypatch.setattr(installed, "_utc_now", lambda: NOW)
    value = installed.establish_installed_account_initial_binding(cha)
    return cha, value, installed._state_path(), installed._read(installed._state_path())


def forbidden(*args, **kwargs):
    pytest.fail("retained reservation attempted mint/time")


def test_single_instant_integer_uuid7_entropy_and_exact_binding(cha, monkeypatch):
    clocks, random_calls = [], []
    original = uuid7.secrets.randbits

    def clock():
        clocks.append(NOW)
        return NOW

    def random(bits):
        result = original(bits)
        random_calls.append((bits, result))
        return result

    monkeypatch.setattr(installed, "_utc_now", clock)
    monkeypatch.setattr(uuid7.secrets, "randbits", random)
    value = installed.establish_installed_account_initial_binding(cha)
    identity = uuid.UUID(value.account_id.removeprefix("acct_"))
    assert value.account_id == "acct_" + str(identity)
    assert identity.version == 7 and identity.variant == uuid.RFC_4122
    assert identity.int >> 80 == uuid7.reservation_epoch_milliseconds(NOW)
    assert clocks == [NOW]
    assert [bits for bits, _ in random_calls] == [12, 62]
    assert (identity.int >> 64) & 4095 == random_calls[0][1]
    assert identity.int & ((1 << 62) - 1) == random_calls[1][1]
    state = installed._read(installed._state_path())
    assert state["status"] == "INITIAL_BINDING_COMMITTED"
    assert value.logical_operation_id == cha.logical_operation_id
    assert value.provisioning_operation_id == cha.provisioning_operation_id
    request = parse_canonical(bytes.fromhex(state["canonical_request_raw_hex"]))
    assert "account_id" not in request and "assigned_at_utc" not in request
    assert set(state) == installed._FIELDS
    assert state["cha_operation_state_raw_hex"] == cha_installed._state_path().read_bytes().hex()


def test_retry_lost_response_and_restart_never_remint(reserved, monkeypatch):
    upstream, value, path, state = reserved
    identity = value.account_id
    monkeypatch.setattr(installed, "mint_uuid7", forbidden)
    monkeypatch.setattr(installed, "_utc_now", forbidden)
    retried = installed.establish_installed_account_initial_binding(upstream)
    loaded = installed.load_installed_account_initial_binding(upstream)
    assert retried.account_id == loaded.account_id == identity
    assert path.read_bytes() == canonical_json_bytes(state)
    assert capability.require_verified_account_initial_binding(loaded) is loaded


def test_loader_empty_never_mints(cha, monkeypatch):
    monkeypatch.setattr(installed, "mint_uuid7", forbidden)
    with pytest.raises(installed.AccountReservationError, match="NOT_FOUND"):
        installed.load_installed_account_initial_binding(cha)


def test_only_upstream_argument_no_caller_ids_timestamps_paths_or_dto(cha):
    for function in (
        installed.establish_installed_account_initial_binding,
        installed.load_installed_account_initial_binding,
    ):
        assert list(inspect.signature(function).parameters) == ["upstream"]
        for name in ("account_id", "logical_operation_id", "assigned_at_utc", "path", "request"):
            with pytest.raises(TypeError):
                function(cha, **{name: "TEST_ONLY"})


def test_raw_copied_and_forged_upstream_never_authority(cha):
    for value in (
        {},
        "ago_TEST_ONLY",
        cha_installed._read(cha_installed._state_path()),
        object.__new__(cha_capability.VerifiedCHALogicalOperation),
        copy.copy(cha),
        copy.deepcopy(cha),
    ):
        with pytest.raises(installed.AccountReservationError, match="VERIFIED_CHA"):
            installed.establish_installed_account_initial_binding(value)
    assert not installed._state_path().exists()


def test_reservation_provenance_and_opaque_instance(reserved):
    _, value, _, state = reserved
    for raw in (
        state,
        canonical_json_bytes(state),
        value.account_id,
        {},
        parse_canonical(bytes.fromhex(state["canonical_request_raw_hex"])),
        object.__new__(capability.VerifiedAccountGenesisInitialBinding),
    ):
        with pytest.raises(capability.AccountInitialBindingError):
            capability.require_verified_account_initial_binding(raw)
    with pytest.raises(TypeError):
        capability.VerifiedAccountGenesisInitialBinding()
    for copier in (copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            copier(value)
    with pytest.raises(TypeError):
        type("Forged", (capability.VerifiedAccountGenesisInitialBinding,), {})
    with pytest.raises(TypeError):
        value.account_id = "acct_TEST_ONLY"
    with pytest.raises(AttributeError):
        object.__setattr__(value, "forged", True)


def test_current_upstream_revocation_invalidates_every_property(reserved, monkeypatch):
    _, value, _, _ = reserved

    def revoked(upstream):
        raise cha_capability.CHALogicalOperationError("TEST_ONLY-revoked")

    monkeypatch.setattr(installed, "require_verified_cha_logical_operation", revoked)
    for field in ("account_id", "logical_operation_id", "provisioning_operation_id"):
        with pytest.raises(installed.AccountReservationError, match="VERIFIED_CHA"):
            getattr(value, field)
    with pytest.raises(installed.AccountReservationError):
        capability.require_verified_account_initial_binding(value)


def test_all_retained_mutations_and_request_changes_fail_closed(reserved):
    upstream, value, path, state = reserved
    original = path.read_bytes()
    changes = {
        "account_id": "acct_" + str(uuid.UUID(int=uuid.UUID(state["account_id"][5:]).int ^ 1)),
        "logical_operation_id": "ago_"
        + str(uuid.UUID(int=uuid.UUID(state["logical_operation_id"][4:]).int ^ 1)),
        "provisioning_operation_id": "prvop_"
        + str(uuid.UUID(int=uuid.UUID(state["provisioning_operation_id"][6:]).int ^ 1)),
        "pdsa_trust_domain": "TEST_ONLY-other-domain",
        "assigned_at_utc": "2026-10-07T00:00:00.000Z",
        "canonical_request_raw_hex": canonical_json_bytes({"schema_version": "other"}).hex(),
        "canonical_request_sha256": "00" * 32,
        "cha_operation_state_raw_hex": canonical_json_bytes({}).hex(),
        "status": "NO_ACCOUNT_RESERVATION",
        "reservation_generation": True,
        "schema_version": "other",
        "initial_binding_sha256": "00" * 32,
    }
    for field, changed in changes.items():
        mutated = {**state, field: changed}
        path.write_bytes(canonical_json_bytes(mutated))
        for call in (
            lambda: value.account_id,
            lambda: installed.establish_installed_account_initial_binding(upstream),
            lambda: installed.load_installed_account_initial_binding(upstream),
        ):
            with pytest.raises((ValueError, RuntimeError)):
                call()
        path.write_bytes(original)
    request = parse_canonical(bytes.fromhex(state["canonical_request_raw_hex"]))
    request["unknown"] = "TEST_ONLY"
    raw = canonical_json_bytes(request)
    changed = {
        **state,
        "canonical_request_raw_hex": raw.hex(),
        "canonical_request_sha256": hashlib.sha256(raw).hexdigest(),
    }
    changed["initial_binding_sha256"] = installed._binding_integrity(changed)
    path.write_bytes(canonical_json_bytes(changed))
    with pytest.raises(installed.AccountReservationError):
        installed.load_installed_account_initial_binding(upstream)
    path.write_bytes(original)


def test_changed_complete_source_and_second_operation_conflict(reserved):
    _, _, path, state = reserved
    source = parse_canonical(bytes.fromhex(state["cha_operation_state_raw_hex"]))
    # Keep the source structurally valid; raw source alone does not issue authority.
    for field in ("logical_operation_id", "lppi_operation_state_raw_hex"):
        changed = dict(source)
        if field == "logical_operation_id":
            identity = uuid.UUID(changed[field][4:])
            changed[field] = "ago_" + str(uuid.UUID(int=identity.int ^ 1))
        else:
            lppi_state = parse_canonical(bytes.fromhex(changed[field]))
            lppi_state["signature_hex"] = lppi_state["signature_hex"][:-2] + (
                "00" if lppi_state["signature_hex"][-2:] != "00" else "01"
            )
            changed[field] = canonical_json_bytes(lppi_state).hex()
        raw = canonical_json_bytes(changed)
        # A different mapping or retained lineage must never mint a second candidate.
        with pytest.raises((installed.AccountReservationError, ValueError)):
            installed._reserve(path, raw)
        assert installed._read(path) == state


def test_restart_reconstructs_from_upstream_not_old_provenance(reserved, monkeypatch):
    upstream, value, path, state = reserved
    previous = cha_capability._logical_operation_snapshot(upstream)
    capability._ISSUED.clear()
    cha_capability._ISSUED.clear()
    with pytest.raises(capability.AccountInitialBindingError):
        capability.require_verified_account_initial_binding(value)
    reconstructed = cha_installed.load_installed_cha_logical_operation(previous.upstream_operation)
    monkeypatch.setattr(installed, "mint_uuid7", forbidden)
    monkeypatch.setattr(installed, "_utc_now", forbidden)
    loaded = installed.load_installed_account_initial_binding(reconstructed)
    assert loaded.account_id == state["account_id"]
    assert path.read_bytes() == canonical_json_bytes(state)


@pytest.mark.parametrize("milliseconds", [0, (1 << 48) - 1, 1 << 47])
def test_neutral_mint_full_uint48(milliseconds, monkeypatch):
    calls = []

    def random(bits):
        calls.append(bits)
        return (1 << bits) - 1

    monkeypatch.setattr(uuid7.secrets, "randbits", random)
    identity = uuid.UUID(installed.mint_uuid7("acct_", milliseconds)[5:])
    assert identity.int >> 80 == milliseconds
    assert identity.version == 7 and identity.variant == uuid.RFC_4122
    assert calls == [12, 62]


@pytest.mark.parametrize("milliseconds", [-1, 1 << 48, True, 1.5])
def test_neutral_mint_rejects_outside_uint48_without_masking(milliseconds):
    with pytest.raises(uuid7.UUID7Error):
        installed.mint_uuid7("acct_", milliseconds)


def test_upstream_subclass_cannot_create_reservation_authority():
    forged_type = type("TestOnlyForgedCHA", (cha_capability.VerifiedCHALogicalOperation,), {})
    forged = object.__new__(forged_type)
    for consume in (
        installed.establish_installed_account_initial_binding,
        installed.load_installed_account_initial_binding,
    ):
        with pytest.raises(installed.AccountReservationError, match="VERIFIED_CHA"):
            consume(forged)
