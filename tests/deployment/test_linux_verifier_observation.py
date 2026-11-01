"""Niezależna walidacja surowej obserwacji verifiera nie daje pełnej kwalifikacji."""

from __future__ import annotations

import copy
import os
import sys

import pytest

from bot_core.linux_service_ipc import PARTIAL_BLOCKERS
from deployment import linux_physical_ipc as ipc


CHALLENGE_ID = "a" * 64
INSTALLATION_ID = "b" * 64
SERVER_CGROUP = "/system.slice/cryptohunter-freshness-verifier.service"


def expected_verifier_process():
    return ipc.ExpectedPeerProcess(1234, 61002, 61002, 123456, "e" * 64)


def valid_verifier_observation():
    """Syntetyczne dane wyłącznie do testowania walidatora i podpisanej otoczki."""
    server = {
        "pid": 1234,
        "uid": 61002,
        "gid": 61002,
        "groups": [61002],
        "start_time_ticks": 123456,
        "exe_sha256": "e" * 64,
        "cgroup": SERVER_CGROUP,
    }
    client = {
        "pid": 1235,
        "uid": 0,
        "gid": 0,
        "groups": [0],
        "start_time_ticks": 123450,
        "exe_sha256": "f" * 64,
        "cgroup": "/system.slice/cryptohunter-collector.scope",
    }
    request = {
        "version": 1,
        "operation": "READONLY_QUALIFY",
        "installation_id": INSTALLATION_ID,
        "challenge_id": CHALLENGE_ID,
        "nonce": "c" * 64,
    }
    response = {
        **request,
        "status": "PARTIAL_VERIFIER_OBSERVATION",
        "production_qualification": "BLOCKED",
        "server_process": copy.deepcopy(server),
        "server_observed_client": {
            "peer_credentials": {key: client[key] for key in ("pid", "uid", "gid")},
            "process": copy.deepcopy(client),
        },
        "socket_bound_pidfd": True,
        "postgresql": None,
        "blockers": [*PARTIAL_BLOCKERS, "VERIFIER_POSTGRESQL_OBSERVER_UNAVAILABLE"],
    }
    return {
        "contract": ipc.IPC_CONTRACT,
        "status": "PARTIAL_VERIFIER_OBSERVATION",
        "production_qualification": "BLOCKED",
        "request": request,
        "response": response,
        "client_process": client,
        "server_peer_credentials": {key: server[key] for key in ("pid", "uid", "gid")},
        "server_process": server,
        "socket_bound_pidfd": True,
    }


def _validate(value):
    return ipc.validate_verifier_observation(
        value,
        expected=expected_verifier_process(),
        expected_cgroup=SERVER_CGROUP,
        challenge_id=CHALLENGE_ID,
        installation_id=INSTALLATION_ID,
    )


def _replace(value, path, replacement):
    target = value
    for item in path[:-1]:
        target = target[item]
    target[path[-1]] = replacement


def test_complete_partial_fixture_validates_without_becoming_full_qualification():
    value = valid_verifier_observation()
    _validate(value)
    with pytest.raises(ipc.LinuxIPCQualificationBlocked):
        ipc.require_ipc_qualification(value)


@pytest.mark.parametrize(
    "path,replacement",
    [
        (("contract",), "invented-contract"),
        (("status",), "PASS"),
        (("production_qualification",), "PASS"),
        (("socket_bound_pidfd",), False),
        (("socket_bound_pidfd",), 1),
        (("request", "version"), True),
        (("request", "operation"), "SIGN"),
        (("request", "challenge_id"), "d" * 64),
        (("request", "installation_id"), "d" * 64),
        (("request", "nonce"), "C" * 64),
        (("response", "version"), True),
        (("response", "operation"), "ADMIN"),
        (("response", "challenge_id"), "d" * 64),
        (("response", "installation_id"), "d" * 64),
        (("response", "nonce"), "d" * 64),
        (("response", "status"), "PASS"),
        (("response", "production_qualification"), "PASS"),
        (("response", "socket_bound_pidfd"), False),
        (("response", "blockers"), []),
        (("response", "blockers"), ["OPERATOR_SAYS_BLOCKED"]),
        (("response", "postgresql"), {"status": "PASS"}),
        (("server_peer_credentials", "pid"), 1236),
        (("server_peer_credentials", "uid"), 0),
        (("server_peer_credentials", "gid"), 0),
        (("server_process", "pid"), 1236),
        (("server_process", "uid"), 0),
        (("server_process", "gid"), 0),
        (("server_process", "start_time_ticks"), 123457),
        (("server_process", "exe_sha256"), "0" * 64),
        (("server_process", "cgroup"), "/system.slice/other.service"),
        (("response", "server_process", "pid"), 1236),
        (("response", "server_process", "start_time_ticks"), 123457),
        (("response", "server_process", "exe_sha256"), "0" * 64),
        (("client_process", "uid"), 1000),
        (("client_process", "gid"), 1000),
        (("client_process", "start_time_ticks"), 0),
        (("client_process", "exe_sha256"), None),
        (("client_process", "groups"), [True]),
        (("response", "server_observed_client", "peer_credentials", "pid"), 999),
        (("response", "server_observed_client", "peer_credentials", "uid"), 1000),
        (("response", "server_observed_client", "process", "start_time_ticks"), 123451),
        (("response", "server_observed_client", "process", "exe_sha256"), "0" * 64),
        (("response", "server_observed_client", "process", "cgroup"), "/other"),
    ],
)
def test_validator_refuses_raw_identity_challenge_application_and_pass_substitution(
    path, replacement
):
    value = valid_verifier_observation()
    _replace(value, path, replacement)
    with pytest.raises(ipc.LinuxIPCObservationError):
        _validate(value)


@pytest.mark.parametrize("value", [None, [], True, {}, {"status": "PASS"}])
def test_pass_json_and_metadata_alone_never_form_a_raw_observation(value):
    with pytest.raises(ipc.LinuxIPCObservationError):
        _validate(value)


@pytest.mark.parametrize("location", ["outer", "request", "response", "client", "server"])
def test_unversioned_fields_are_not_ignored(location):
    value = valid_verifier_observation()
    target = {
        "outer": value,
        "request": value["request"],
        "response": value["response"],
        "client": value["client_process"],
        "server": value["server_process"],
    }[location]
    target["caller_sql"] = "SELECT 1"
    with pytest.raises(ipc.LinuxIPCObservationError):
        _validate(value)


def test_different_expected_installation_or_challenge_refuses_same_bundle():
    for key in ("installation_id", "challenge_id"):
        arguments = {"installation_id": INSTALLATION_ID, "challenge_id": CHALLENGE_ID}
        arguments[key] = "d" * 64
        with pytest.raises(ipc.LinuxIPCObservationError):
            ipc.validate_verifier_observation(
                valid_verifier_observation(),
                expected=expected_verifier_process(),
                expected_cgroup=SERVER_CGROUP,
                **arguments,
            )


@pytest.mark.skipif(
    sys.platform != "linux" or getattr(os, "geteuid", lambda: 0)() == 0,
    reason="Sprawdza realną odmowę klientowi nie-root na Linux",
)
def test_real_nonroot_collector_refuses_before_any_socket_is_created(monkeypatch):
    monkeypatch.setattr(
        ipc.socket,
        "socket",
        lambda *_args, **_kwargs: pytest.fail("Nie-root nie może nawet utworzyć socketu sondy"),
    )
    with pytest.raises(ipc.LinuxIPCObservationError):
        ipc.collect_verifier_observation(
            "/nonexistent-verifier.sock",
            expected=expected_verifier_process(),
            expected_cgroup=SERVER_CGROUP,
            challenge_id=CHALLENGE_ID,
            installation_id=INSTALLATION_ID,
        )


def test_unsupported_platform_refuses_before_posix_identity_or_socket(monkeypatch):
    monkeypatch.setattr(ipc.sys, "platform", "win32")
    with pytest.raises(ipc.LinuxIPCObservationError):
        ipc.collect_verifier_observation(
            "/nonexistent-verifier.sock",
            expected=expected_verifier_process(),
            expected_cgroup=SERVER_CGROUP,
            challenge_id=CHALLENGE_ID,
            installation_id=INSTALLATION_ID,
        )
