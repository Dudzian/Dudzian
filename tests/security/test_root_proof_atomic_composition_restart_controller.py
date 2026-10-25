"""Hermetic controller checks; these do not qualify a PostgreSQL restart."""

import json
import subprocess

import pytest

from tests.security import test_root_proof_atomic_composition_integration as integration


@pytest.fixture(autouse=True)
def clear_restart_controllers(monkeypatch):
    monkeypatch.delenv("CRYPTOHUNTER_TEST_POSTGRES_CONTAINER", raising=False)
    monkeypatch.delenv("DUDZIAN_TEST_POSTGRES_RESTART_COMMAND", raising=False)


@pytest.fixture
def restart_calls(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(integration.subprocess, "run", run)
    return calls


def test_native_controller_preserves_argv_without_shell(monkeypatch, restart_calls):
    argv = ["/opt/PostgreSQL 16/bin/pg_ctl", "restart", "-D", "/tmp/stage9 cluster", "-w"]
    monkeypatch.setenv("DUDZIAN_TEST_POSTGRES_RESTART_COMMAND", json.dumps(argv))

    integration._restart_controlled_postgresql()

    assert restart_calls == [
        (argv, dict(check=True, capture_output=True, timeout=30, env=None, shell=False))
    ]


def test_docker_controller_uses_local_socket_and_sanitized_environment(monkeypatch, restart_calls):
    docker_variables = (
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_TLS",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
    )
    for name in docker_variables:
        monkeypatch.setenv(name, "untrusted-docker-configuration")
    monkeypatch.setenv("STAGE9_CONTROLLER_TEST_SENTINEL", "preserved")
    monkeypatch.setenv("CRYPTOHUNTER_TEST_POSTGRES_CONTAINER", "stage9-postgres")

    integration._restart_controlled_postgresql()

    assert len(restart_calls) == 1
    argv, kwargs = restart_calls[0]
    assert argv == ["docker", "--host=unix:///var/run/docker.sock", "restart", "stage9-postgres"]
    assert kwargs["check"] is True
    assert kwargs["capture_output"] is True
    assert kwargs["timeout"] == 30
    assert kwargs["shell"] is False
    assert all(name not in kwargs["env"] for name in docker_variables)
    assert kwargs["env"]["STAGE9_CONTROLLER_TEST_SENTINEL"] == "preserved"
    assert all(
        integration.os.environ[name] == "untrusted-docker-configuration"
        for name in docker_variables
    )


def test_missing_controller_fails_before_execution(restart_calls):
    with pytest.raises(pytest.fail.Exception, match="exactly one PostgreSQL restart controller"):
        integration._restart_controlled_postgresql()
    assert restart_calls == []


@pytest.mark.parametrize("native", ['["pg_ctl", "restart"]', "", "not-json"])
def test_two_controllers_fail_before_execution(monkeypatch, restart_calls, native):
    monkeypatch.setenv("CRYPTOHUNTER_TEST_POSTGRES_CONTAINER", "stage9-postgres")
    monkeypatch.setenv("DUDZIAN_TEST_POSTGRES_RESTART_COMMAND", native)
    with pytest.raises(pytest.fail.Exception, match="exactly one PostgreSQL restart controller"):
        integration._restart_controlled_postgresql()
    assert restart_calls == []


@pytest.mark.parametrize("container", ["", "   "])
def test_empty_docker_controller_fails_before_execution(monkeypatch, restart_calls, container):
    monkeypatch.setenv("CRYPTOHUNTER_TEST_POSTGRES_CONTAINER", container)
    with pytest.raises(pytest.fail.Exception, match="non-empty Docker container"):
        integration._restart_controlled_postgresql()
    assert restart_calls == []


@pytest.mark.parametrize("native", ["", "not-json", '["pg_ctl",'])
def test_invalid_native_json_fails_before_execution(monkeypatch, restart_calls, native):
    monkeypatch.setenv("DUDZIAN_TEST_POSTGRES_RESTART_COMMAND", native)
    with pytest.raises(pytest.fail.Exception, match="JSON argv array"):
        integration._restart_controlled_postgresql()
    assert restart_calls == []


@pytest.mark.parametrize(
    "argv",
    [None, {}, "pg_ctl", [], [""], ["   "], [1], ["pg_ctl", None], ["pg_ctl", False], [["pg_ctl"]]],
)
def test_invalid_native_argv_fails_before_execution(monkeypatch, restart_calls, argv):
    monkeypatch.setenv("DUDZIAN_TEST_POSTGRES_RESTART_COMMAND", json.dumps(argv))
    with pytest.raises(pytest.fail.Exception, match="non-empty JSON argv array"):
        integration._restart_controlled_postgresql()
    assert restart_calls == []


@pytest.mark.parametrize("controller", ["native", "docker"])
@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(7, ["pg_ctl", "restart"]),
        subprocess.TimeoutExpired(["pg_ctl", "restart"], 30),
        FileNotFoundError("restart executable unavailable"),
    ],
    ids=["nonzero-exit", "timeout", "missing-executable"],
)
def test_restart_errors_propagate_without_fallback(monkeypatch, controller, error):
    if controller == "native":
        monkeypatch.setenv("DUDZIAN_TEST_POSTGRES_RESTART_COMMAND", '["pg_ctl", "restart"]')
    else:
        monkeypatch.setenv("CRYPTOHUNTER_TEST_POSTGRES_CONTAINER", "stage9-postgres")
    calls = []

    def fail(argv, **kwargs):
        calls.append((argv, kwargs))
        assert kwargs["check"] is True
        raise error

    monkeypatch.setattr(integration.subprocess, "run", fail)
    with pytest.raises(type(error)) as raised:
        integration._restart_controlled_postgresql()
    assert raised.value is error
    assert len(calls) == 1
