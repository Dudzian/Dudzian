from __future__ import annotations

import ast
import ctypes
from pathlib import Path
import subprocess
import sys
import re
import threading
import time

import pytest
from psycopg.conninfo import conninfo_to_dict

import deployment.windows_stage8_postgresql_probe as probe
from deployment.platform_evidence import WINDOWS_LIVE_ITEMS, WINDOWS_STAGE8_ITEMS


@pytest.fixture(autouse=True)
def process_exit_boundary(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(probe, "_open_process_exit_witness", lambda _pid: object())
    monkeypatch.setattr(
        probe, "_wait_process_exit", lambda _process, timeout=probe.SERVICE_TIMEOUT: None
    )
    monkeypatch.setattr(probe, "_close_process_exit_witness", lambda _process: None)


def test_non_windows_cannot_emit_live_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "canonical_host_os", lambda: "Linux")
    with pytest.raises(probe.Stage8PostgreSQLProbeError) as failure:
        probe.run_probe(tmp_path)
    assert failure.value.item == "WINDOWS_POSTGRESQL_SUBSTRATE"


def _fake_bin(tmp_path: Path) -> Path:
    bindir = tmp_path / "pgbin"
    bindir.mkdir()
    for name in probe.REQUIRED_BINARIES:
        (bindir / name).write_bytes(b"binary")
    return bindir


def test_postgresql_below_16_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bindir = _fake_bin(tmp_path)
    monkeypatch.setattr(
        probe,
        "_run",
        lambda *_a, **_k: subprocess.CompletedProcess([], 0, "postgres (PostgreSQL) 15.8", ""),
    )
    with pytest.raises(probe.Stage8PostgreSQLProbeError, match="below required"):
        probe.discover_postgresql({"PGBIN": str(bindir)})


def test_missing_binaries_fail(tmp_path: Path) -> None:
    with pytest.raises(probe.Stage8PostgreSQLProbeError, match="were not found"):
        probe.discover_postgresql({"PGBIN": str(tmp_path)})


def test_final_hba_is_exact_ordered_loopback_sspi_then_reject() -> None:
    lines = probe.final_hba_lines()
    assert lines[:2] == (
        "host stage8_freshness freshness_crypto_verifier 127.0.0.1/32 sspi map=stage8_sspi include_realm=1",
        "host stage8_freshness freshness_runtime 127.0.0.1/32 sspi map=stage8_sspi include_realm=1",
    )
    assert all(method not in "\n".join(lines[:2]) for method in probe.FORBIDDEN_ONLINE_METHODS)
    assert all("reject" in line for line in lines[2:])


def test_ident_is_exact_and_principals_must_be_distinct() -> None:
    assert probe.ident_lines("HOST\\runtime", "HOST\\verifier") == (
        "stage8_sspi HOST\\runtime freshness_runtime",
        "stage8_sspi HOST\\verifier freshness_crypto_verifier",
    )
    with pytest.raises(probe.Stage8PostgreSQLProbeError, match="two distinct"):
        probe.ident_lines("HOST$@REALM", "host$@realm")
    with pytest.raises(probe.Stage8PostgreSQLProbeError, match="unsafe"):
        probe.ident_lines(".*", "HOST\\verifier")


def test_effective_hba_rows_require_parser_success_and_order() -> None:
    rows = [
        (
            1,
            "host",
            [probe.DATABASE],
            [probe.VERIFIER_ROLE],
            "127.0.0.1",
            "255.255.255.255",
            "sspi",
            ["map=stage8_sspi", "include_realm=1"],
            None,
        ),
        (
            2,
            "host",
            [probe.DATABASE],
            [probe.RUNTIME_ROLE],
            "127.0.0.1",
            "255.255.255.255",
            "sspi",
            ["map=stage8_sspi", "include_realm=1"],
            None,
        ),
        (
            3,
            "host",
            [probe.DATABASE],
            ["all"],
            "127.0.0.1",
            "255.255.255.255",
            "reject",
            None,
            None,
        ),
        (4, "host", ["all"], ["all"], "127.0.0.1", "255.255.255.255", "reject", None, None),
        (5, "host", ["all"], ["all"], "0.0.0.0", "0.0.0.0", "reject", None, None),
        (6, "host", ["all"], ["all"], "::", "::", "reject", None, None),
    ]
    probe.qualify_hba_rows(rows)
    rows[1] = (*rows[1][:-1], "parse error")
    with pytest.raises(probe.Stage8PostgreSQLProbeError):
        probe.qualify_hba_rows(rows)


def test_stage8_evidence_contract_is_last_live_slice() -> None:
    assert WINDOWS_STAGE8_ITEMS == (
        "WINDOWS_POSTGRESQL_SUBSTRATE",
        "WINDOWS_LOCAL_PRINCIPAL_AUTHENTICATION",
    )
    assert WINDOWS_LIVE_ITEMS[-2:] == WINDOWS_STAGE8_ITEMS


def test_authority_dsn_bounds_connect_statements_and_locks() -> None:
    dsn = probe._admin_dsn(5432, probe.DATABASE)
    parsed = conninfo_to_dict(dsn)
    assert parsed["connect_timeout"] == "5"
    assert parsed["options"] == "-c statement_timeout=60000 -c lock_timeout=5000"


def test_live_session_timeout_settings_require_exact_numeric_ms_values() -> None:
    probe._qualify_session_timeout_settings(
        [("statement_timeout", "60000", "ms"), ("lock_timeout", "5000", "ms")]
    )
    with pytest.raises(probe.Stage8PostgreSQLProbeError) as failure:
        probe._qualify_session_timeout_settings(
            [("statement_timeout", "0", "ms"), ("lock_timeout", "5000", "ms")]
        )
    assert failure.value.item == probe.SUBSTRATE
    assert failure.value.label == "POSTGRESQL_DURABILITY"


def test_worker_publishes_terminal_success_only_after_run_probe_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status = tmp_path / "status.json"
    events: list[str] = []
    result = {probe.SUBSTRATE: "PASS", probe.PRINCIPAL_AUTH: "PASS"}

    def completed_probe(_scratch: Path, *, status_file: Path):
        probe._status_file = status_file
        probe._status = {"last_phase": "CLEANUP_SCRATCH"}
        events.extend(("cleanup-finished", "run-probe-returned"))
        return result

    original_write_status = probe._write_status

    def observed_write_status(**updates: object) -> None:
        if updates.get("completed") is True:
            events.append("terminal-success-written")
        original_write_status(**updates)

    monkeypatch.setattr(probe, "run_probe", completed_probe)
    monkeypatch.setattr(probe, "_write_status", observed_write_status)
    assert probe.main(["--scratch-parent", str(tmp_path), "--status-file", str(status)]) == 0
    assert events == ["cleanup-finished", "run-probe-returned", "terminal-success-written"]
    terminal = probe._read_json(status)
    assert terminal is not None
    assert terminal["completed"] is True
    assert terminal["results"] == result


def test_stage8_phase_markers_flush_and_retain_failed_last_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        probe,
        "print",
        lambda message, **kwargs: calls.append((message, kwargs.get("flush") is True)),
        raising=False,
    )
    probe._status_file = tmp_path / "status.json"
    probe._status = {}
    with pytest.raises(RuntimeError, match="synthetic hang"):
        with probe._phase("MATRIX_RUNTIME"):
            raise RuntimeError("synthetic hang")
    assert [message.split()[1:3] for message, _flush in calls] == [
        ["START", "MATRIX_RUNTIME"],
        ["FAIL", "MATRIX_RUNTIME"],
    ]
    assert all(flush for _message, flush in calls)
    assert probe._read_json(tmp_path / "status.json")["last_phase"] == "MATRIX_RUNTIME"


def test_probe_source_keeps_ownership_token_and_production_boundaries() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert "tempfile.mkdtemp" in source and "shutil.rmtree(root)" in source
    assert "OpenProcessToken" in source and "TokenUser" in source
    assert "provision_postgresql_freshness_authority(authority)" in source
    assert "qualify_postgresql_freshness_authority(authority)" in source
    assert "pg_hba_file_rules" in source
    assert "timeout=" in source
    assert 'os.environ["PGDATA"]' not in source


def _completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_stale_ready_and_result_are_not_accepted(tmp_path: Path) -> None:
    path = tmp_path / "message.json"
    path.write_text('{"invocation_token":"old","pid":10}', encoding="utf-8")
    with pytest.raises(TimeoutError):
        probe._wait_invocation_message(path, "fresh", timeout=0.01)
    path.write_text('{"invocation_token":"fresh","pid":10}', encoding="utf-8")
    with pytest.raises(TimeoutError):
        probe._wait_invocation_message(path, "fresh", expected_pid=11, timeout=0.01)


def test_each_invocation_uses_fresh_files_and_old_files_cannot_satisfy_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(probe.secrets, "token_hex", lambda _size: next(tokens))
    monkeypatch.setattr(probe, "_sc", lambda *_a, **_k: _completed())
    monkeypatch.setattr(probe, "_wait_service_state", lambda *_a, **_k: 123)
    monkeypatch.setattr(probe, "_assert_expected_token", lambda *_a: ("sid", "account"))

    def messages(path: Path, token: str, *, expected_pid=None, timeout=probe.SERVICE_TIMEOUT):
        if path.name == "ready.json":
            return {"invocation_token": token, "pid": 123, "ready": True}
        return {"invocation_token": token, "pid": 123, "ok": True, "session_user": "role"}

    monkeypatch.setattr(probe, "_wait_invocation_message", messages)
    pointer = tmp_path / "current.json"
    probe._invoke_service(tmp_path, pointer, "service", "account", 1, "role")
    first = pointer.read_text(encoding="utf-8")
    probe._invoke_service(tmp_path, pointer, "service", "account", 1, "role")
    second = pointer.read_text(encoding="utf-8")
    assert '"invocation_token":"' + "a" * 64 in first
    assert '"invocation_token":"' + "b" * 64 in second
    assert (tmp_path / "invocations" / ("a" * 64) / "result.json") != (
        tmp_path / "invocations" / ("b" * 64) / "result.json"
    )


def test_parent_binds_ready_pid_and_proves_token_before_fresh_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    monkeypatch.setattr(probe.secrets, "token_hex", lambda _size: "c" * 64)
    monkeypatch.setattr(probe, "_sc", lambda *_a, **_k: _completed())
    monkeypatch.setattr(probe, "_wait_service_state", lambda *_a, **_k: 321)

    def messages(path: Path, token: str, *, expected_pid=None, timeout=probe.SERVICE_TIMEOUT):
        events.append(path.name)
        if path.name == "ready.json":
            return {"invocation_token": token, "pid": 321, "ready": True}
        assert expected_pid == 321
        return {"invocation_token": token, "pid": 321, "ok": True}

    monkeypatch.setattr(probe, "_wait_invocation_message", messages)
    monkeypatch.setattr(
        probe,
        "_assert_expected_token",
        lambda *_a: events.append("token-proof") or ("sid", "account"),
    )
    original_write = probe._atomic_json

    def write(path: Path, value: dict[str, object]) -> None:
        if path.name == "go.json":
            events.append("go")
        original_write(path, value)

    monkeypatch.setattr(probe, "_atomic_json", write)
    probe._invoke_service(tmp_path, tmp_path / "request.json", "service", "account", 1, "role")
    assert events.index("token-proof") < events.index("go") < events.index("result.json")


def test_ready_pid_must_equal_scm_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe.secrets, "token_hex", lambda _size: "d" * 64)
    monkeypatch.setattr(probe, "_sc", lambda *_a, **_k: _completed())
    monkeypatch.setattr(probe, "_wait_service_state", lambda *_a, **_k: 2)
    monkeypatch.setattr(
        probe,
        "_wait_invocation_message",
        lambda *_a, **_k: {"invocation_token": "d" * 64, "pid": 1, "ready": True},
    )
    with pytest.raises(RuntimeError, match="READY PID does not match SCM PID"):
        probe._invoke_service(tmp_path, tmp_path / "request.json", "service", "account", 1, "role")


def test_child_requires_fresh_go_before_connecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deployment import windows_stage8_postgresql_service as child

    token = "e" * 64
    request = tmp_path / "request.json"
    ready, go, result = (tmp_path / name for name in ("ready", "go", "result"))
    request.write_text(
        __import__("json").dumps(
            {
                "invocation_token": token,
                "port": 1,
                "database": "db",
                "role": "role",
                "ready": str(ready),
                "go": str(go),
                "result": str(result),
            }
        ),
        encoding="utf-8",
    )
    # Existing GO from another invocation is deliberately placed at the fresh path.
    go.write_text('{"invocation_token":"stale","pid":1,"go":true}', encoding="utf-8")
    calls: list[str] = []
    child._connect(request, lambda **_kwargs: calls.append("connected"), wait_timeout=0)
    assert calls == []
    payload = __import__("json").loads(result.read_text(encoding="utf-8"))
    assert payload["invocation_token"] == token
    assert payload["ok"] is False


def test_service_stopped_wait_is_bounded_and_required(monkeypatch: pytest.MonkeyPatch) -> None:
    observations = iter((("RUNNING", 1), ("STOPPED", 0)))
    monkeypatch.setattr(probe, "_service_observation", lambda _name: next(observations))
    monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)
    assert probe._wait_service_state("service", "STOPPED", timeout=1) == 0


def test_marked_for_delete_is_not_absent() -> None:
    marked = _completed(1072, stderr="ERROR_SERVICE_MARKED_FOR_DELETE 1072")
    missing = _completed(1060, stderr="OpenService FAILED 1060")
    assert probe._service_is_absent(marked) is False
    assert probe._service_is_absent(missing) is True


def test_owned_cleanup_waits_for_stopped_then_delete_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(probe, "_service_observation", lambda _name: ("RUNNING", 9))

    def sc(action: str, _name: str, **_kwargs):
        events.append(action)
        return _completed()

    monkeypatch.setattr(probe, "_sc", sc)
    monkeypatch.setattr(
        probe,
        "_wait_service_state",
        lambda _name, state, **_kwargs: events.append(f"wait-{state}") or 0,
    )
    monkeypatch.setattr(
        probe,
        "_wait_service_absent",
        lambda _name, **_kwargs: events.append("wait-ABSENT"),
    )
    assert probe._cleanup_service("owned") == []
    assert events == ["stop", "wait-STOPPED", "delete", "wait-ABSENT"]


def test_foreign_service_is_never_created_or_deleted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, ...]] = []

    def sc(*args: str, **_kwargs):
        calls.append(args)
        return _completed(0, "SERVICE_NAME: foreign")

    monkeypatch.setattr(probe, "_sc", sc)
    with pytest.raises(RuntimeError, match="pre-existing"):
        probe._create_service("foreign", "account", tmp_path / "request")
    assert calls == [("query", "foreign")]


def test_services_are_created_once_and_reused_without_intermediate_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created: list[tuple[str, str]] = []
    monkeypatch.setattr(
        probe,
        "_create_service",
        lambda name, account, _request: created.append((name, account)),
    )
    monkeypatch.setattr(probe, "_run", lambda *_a, **_k: _completed())
    owned: set[str] = set()
    requests = probe._prepare_services(tmp_path, owned)
    assert created == [
        (probe.RUNTIME_SERVICE, probe.RUNTIME_ACCOUNT),
        (probe.VERIFIER_SERVICE, probe.VERIFIER_ACCOUNT),
    ]
    assert set(requests) == owned == {probe.RUNTIME_SERVICE, probe.VERIFIER_SERVICE}


def test_successful_invocation_requires_observed_stopped_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe.secrets, "token_hex", lambda _size: "f" * 64)
    monkeypatch.setattr(probe, "_sc", lambda *_a, **_k: _completed())
    states: list[str] = []

    def wait_state(_service: str, expected: str, timeout=probe.SERVICE_TIMEOUT) -> int:
        states.append(expected)
        if expected == "STOPPED":
            raise TimeoutError("not stopped")
        return 77

    monkeypatch.setattr(probe, "_wait_service_state", wait_state)
    monkeypatch.setattr(probe, "_assert_expected_token", lambda *_a: ("sid", "account"))

    def message(path: Path, token: str, *, expected_pid=None, timeout=probe.SERVICE_TIMEOUT):
        if path.name == "ready.json":
            return {"invocation_token": token, "pid": 77, "ready": True}
        return {"invocation_token": token, "pid": 77, "ok": True}

    monkeypatch.setattr(probe, "_wait_invocation_message", message)
    with pytest.raises(probe.Stage8PostgreSQLProbeError, match="not stopped") as failure:
        probe._invoke_service(tmp_path, tmp_path / "request", "service", "account", 1, "role")
    assert failure.value.item == probe.PRINCIPAL_AUTH
    assert states == ["RUNNING", "STOPPED"]


def test_cleanup_primary_stays_primary_and_secondary_is_visible(
    capsys: pytest.CaptureFixture[str],
) -> None:
    primary = RuntimeError("primary")
    probe._finish_cleanup(primary, ["cleanup broke"])
    assert "[STAGE8_CLEANUP] secondary failure: cleanup broke" in capsys.readouterr().err
    assert str(primary) == "primary"


def test_standalone_cleanup_failure_fails_closed() -> None:
    with pytest.raises(probe.Stage8PostgreSQLProbeError) as failure:
        probe._finish_cleanup(None, ["cleanup broke"])
    assert failure.value.label == "STAGE8_CLEANUP"


@pytest.mark.parametrize(
    ("service", "role", "expected", "label"),
    [
        (probe.RUNTIME_SERVICE, probe.RUNTIME_ROLE, True, "WINDOWS_SSPI_RUNTIME_CONNECT"),
        (probe.VERIFIER_SERVICE, probe.VERIFIER_ROLE, True, "WINDOWS_SSPI_VERIFIER_CONNECT"),
        (probe.RUNTIME_SERVICE, probe.VERIFIER_ROLE, False, "WINDOWS_SSPI_CROSS_ROLE_DENIAL"),
        (probe.VERIFIER_SERVICE, probe.RUNTIME_ROLE, False, "WINDOWS_SSPI_CROSS_ROLE_DENIAL"),
    ],
)
def test_matrix_failures_have_specific_labels(
    service: str, role: str, expected: bool, label: str
) -> None:
    assert probe._matrix_label(service, role, expected) == label


def test_svc_stop_signals_cancellation_without_claiming_stopped() -> None:
    from deployment import windows_stage8_postgresql_service as child

    statuses: list[int] = []
    service = type(
        "Service", (), {"ReportServiceStatus": lambda _self, value: statuses.append(value)}
    )()
    stop_requested = threading.Event()
    child._signal_service_stop(service, stop_requested)
    assert stop_requested.is_set()
    assert statuses == [3]  # SERVICE_STOP_PENDING; never fake SERVICE_STOPPED (1).


def test_child_cancellation_before_go_ends_wait_and_never_connects(tmp_path: Path) -> None:
    from deployment import windows_stage8_postgresql_service as child

    token = "1" * 64
    request = tmp_path / "request.json"
    ready, go, result = (tmp_path / name for name in ("ready", "go", "result"))
    request.write_text(
        __import__("json").dumps(
            {
                "invocation_token": token,
                "port": 1,
                "database": "db",
                "role": "role",
                "ready": str(ready),
                "go": str(go),
                "result": str(result),
            }
        ),
        encoding="utf-8",
    )
    cancelled = threading.Event()
    connected: list[bool] = []
    worker = threading.Thread(
        target=child._connect,
        args=(request, lambda **_kwargs: connected.append(True)),
        kwargs={"cancelled": cancelled.is_set, "wait_timeout": 2},
    )
    worker.start()
    deadline = time.monotonic() + 1
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists()
    cancelled.set()
    worker.join(timeout=1)
    assert worker.is_alive() is False
    assert connected == []
    payload = __import__("json").loads(result.read_text(encoding="utf-8"))
    assert payload == {
        "ok": False,
        "cancelled": True,
        "invocation_token": token,
        "pid": payload["pid"],
    }


def test_token_proof_failure_publishes_no_go_and_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe.secrets, "token_hex", lambda _size: "2" * 64)
    monkeypatch.setattr(probe, "_sc", lambda *_a, **_k: _completed())
    monkeypatch.setattr(probe, "_wait_service_state", lambda *_a, **_k: 42)
    monkeypatch.setattr(
        probe,
        "_wait_invocation_message",
        lambda *_a, **_k: {"invocation_token": "2" * 64, "pid": 42, "ready": True},
    )
    monkeypatch.setattr(
        probe,
        "_assert_expected_token",
        lambda *_a: (_ for _ in ()).throw(RuntimeError("bad token")),
    )
    with pytest.raises(probe.Stage8PostgreSQLProbeError) as failure:
        probe._invoke_service(
            tmp_path,
            tmp_path / "request",
            probe.RUNTIME_SERVICE,
            probe.RUNTIME_ACCOUNT,
            1,
            probe.RUNTIME_ROLE,
        )
    assert failure.value.item == probe.PRINCIPAL_AUTH
    assert failure.value.label == "WINDOWS_SSPI_RUNTIME_CONNECT"
    assert not list(tmp_path.glob("invocations/*/go.json"))


def test_token_failure_cleanup_stops_waits_for_exit_then_deletes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    witness = object()
    monkeypatch.setattr(probe, "_service_observation", lambda _name: ("RUNNING", 42))
    monkeypatch.setattr(
        probe, "_open_process_exit_witness", lambda pid: events.append(f"open-{pid}") or witness
    )
    monkeypatch.setattr(
        probe, "_wait_process_exit", lambda value, **_k: events.append("process-exit")
    )
    monkeypatch.setattr(probe, "_close_process_exit_witness", lambda value: events.append("close"))
    monkeypatch.setattr(
        probe,
        "_wait_service_state",
        lambda _name, state, **_k: events.append(f"wait-{state}") or 0,
    )
    monkeypatch.setattr(probe, "_wait_service_absent", lambda _name: events.append("absent"))
    monkeypatch.setattr(
        probe,
        "_sc",
        lambda action, _name, **_k: events.append(action) or _completed(),
    )
    assert probe._cleanup_service(probe.RUNTIME_SERVICE) == []
    assert events == [
        "open-42",
        "stop",
        "wait-STOPPED",
        "process-exit",
        "delete",
        "absent",
        "close",
    ]


def test_alive_helper_or_process_exit_timeout_blocks_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions: list[str] = []
    monkeypatch.setattr(probe, "_service_observation", lambda _name: ("STOPPED", 42))
    monkeypatch.setattr(probe, "_open_process_exit_witness", lambda _pid: object())
    monkeypatch.setattr(
        probe,
        "_wait_process_exit",
        lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError("process still alive")),
    )
    monkeypatch.setattr(probe, "_close_process_exit_witness", lambda _value: None)
    monkeypatch.setattr(
        probe,
        "_sc",
        lambda action, _name, **_k: actions.append(action) or _completed(),
    )
    errors = probe._cleanup_service("owned")
    assert "process still alive" in errors[0]
    assert "delete" not in actions


@pytest.mark.parametrize(
    ("service", "label"),
    [
        (probe.RUNTIME_SERVICE, "WINDOWS_SSPI_RUNTIME_CONNECT"),
        (probe.VERIFIER_SERVICE, "WINDOWS_SSPI_VERIFIER_CONNECT"),
    ],
)
def test_service_start_failures_are_typed_principal_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, service: str, label: str
) -> None:
    monkeypatch.setattr(probe, "_sc", lambda *_a, **_k: _completed(1, stderr="start failed"))
    with pytest.raises(probe.Stage8PostgreSQLProbeError) as failure:
        probe._invoke_service(tmp_path, tmp_path / "request", service, "account", 1, "role")
    assert failure.value.item == probe.PRINCIPAL_AUTH
    assert failure.value.label == label


def test_missing_logged_sspi_identity_is_typed_principal_error(tmp_path: Path) -> None:
    log = tmp_path / "postgres.log"
    log.write_text("no authenticated identity here", encoding="utf-8")
    with pytest.raises(probe.Stage8PostgreSQLProbeError) as failure:
        probe._extract_new_principal(log, 0, "WINDOWS_SSPI_RUNTIME_IDENTITY")
    assert failure.value.item == probe.PRINCIPAL_AUTH
    assert failure.value.label == "WINDOWS_SSPI_RUNTIME_IDENTITY"


def test_pg_ctl_runner_uses_real_files_not_anonymous_pipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    class Process:
        def __init__(self, _command, **kwargs):
            observed.update(kwargs)

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(probe.subprocess, "Popen", Process)
    result = probe._run_pg_ctl(Path("pg_ctl"), tmp_path, tmp_path, "start", ["start"])
    assert result.returncode == 0
    assert observed["stdout"] is not subprocess.PIPE
    assert observed["stderr"] is not subprocess.PIPE
    assert "capture_output" not in observed


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable fixture")
def test_pg_ctl_runner_does_not_wait_for_descendant_inherited_output_handles(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "fake-pg-ctl"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)'])\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    started = time.monotonic()
    result = probe._run_pg_ctl(executable, tmp_path / "data", tmp_path, "start", ["start"])
    assert result.returncode == 0
    assert time.monotonic() - started < 1


@pytest.mark.parametrize("marker", [False, True])
def test_cleanup_stops_live_owned_cluster_even_with_stale_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: bool
) -> None:
    events: list[str] = []
    witness = probe.PostmasterWitness(123, object(), 1.0, Path("postgres.exe"))
    monkeypatch.setattr(probe, "_postmaster_witness", lambda *_a: witness)
    monkeypatch.setattr(probe, "_close_postmaster_witness", lambda *_a: None)
    monkeypatch.setattr(
        probe, "_stop_owned_cluster", lambda *_a, **_k: events.append("stop") or None
    )
    assert (
        probe._stop_if_owned_cluster_running(
            Path("pg_ctl"),
            tmp_path / "cluster",
            tmp_path,
            tmp_path / "postgresql.log",
            cluster_started=marker,
        )
        is None
    )
    assert events == ["stop"]


def test_cleanup_does_not_stop_cluster_qualified_as_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe, "_postmaster_witness", lambda *_a: None)
    monkeypatch.setattr(probe, "_run_pg_ctl", lambda *_a, **_k: pytest.fail("unexpected stop"))
    assert (
        probe._stop_if_owned_cluster_running(
            Path("pg_ctl"),
            tmp_path / "cluster",
            tmp_path,
            tmp_path / "postgresql.log",
            cluster_started=False,
        )
        is None
    )


def test_emergency_cleanup_stale_boolean_stops_before_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "CryptoHunter-Stage8-owned"
    root.mkdir()
    events: list[str] = []
    monkeypatch.setattr(
        probe,
        "_stop_if_owned_cluster_running",
        lambda *_a, **_k: events.append("stop") or None,
    )
    monkeypatch.setattr(probe.shutil, "rmtree", lambda _root: events.append("delete"))
    errors = probe.emergency_cleanup(
        {"root": str(root), "pgbin": str(tmp_path / "bin"), "cluster_started": False},
        tmp_path,
    )
    assert errors == []
    assert events == ["stop", "delete"]


def test_emergency_stop_failure_preserves_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "CryptoHunter-Stage8-owned"
    root.mkdir()
    monkeypatch.setattr(probe, "_stop_if_owned_cluster_running", lambda *_a, **_k: "still running")
    monkeypatch.setattr(probe.shutil, "rmtree", lambda _root: pytest.fail("unsafe deletion"))
    errors = probe.emergency_cleanup(
        {"root": str(root), "pgbin": str(tmp_path / "bin"), "cluster_started": False},
        tmp_path,
    )
    assert "still running" in errors[0]


def test_emergency_cleanup_refuses_unsafe_root(tmp_path: Path) -> None:
    errors = probe.emergency_cleanup(
        {"root": str(tmp_path.parent / "foreign"), "cluster_started": True}, tmp_path
    )
    assert errors == ["refused scratch cleanup outside Stage-8-owned root"]


def test_initial_start_and_final_restart_share_reviewed_runner() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert re.search(r"_run_pg_ctl\([\s\S]{0,150}?root,\s*\"start\"", source)
    assert re.search(r"_run_pg_ctl\([\s\S]{0,150}?root,\s*\"restart\"", source)
    assert source.count("_run_pg_ctl(") >= 5


@pytest.mark.parametrize(("status_code", "expected"), [(0, True), (3, False)])
def test_cluster_running_requires_live_owned_pid_and_exact_pgdata_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    expected: bool,
) -> None:
    data = tmp_path / "cluster"
    data.mkdir()
    witness = probe.PostmasterWitness(123, object(), 1.0, Path("postgres.exe"))
    monkeypatch.setattr(probe, "_postmaster_witness", lambda *_a: witness)
    monkeypatch.setattr(probe, "_close_postmaster_witness", lambda *_a: None)
    calls: list[tuple[Path, list[str]]] = []

    def run_pg_ctl(_pg_ctl, observed_data, _root, _operation, arguments, **_kwargs):
        calls.append((observed_data, arguments))
        return probe.PgCtlResult(status_code, 0.01, False, "", "")

    monkeypatch.setattr(probe, "_run_pg_ctl", run_pg_ctl)
    running, pid = probe._cluster_running(Path("pg_ctl"), data, tmp_path)
    assert running is expected
    assert pid == 123
    assert calls == [(data, ["status"])]


@pytest.mark.parametrize("contents", ["123\n", "bad\nC:\\data\n100\n", "0\nC:\\data\n100\n"])
def test_malformed_or_truncated_postmaster_pid_is_rejected(tmp_path: Path, contents: str) -> None:
    data = tmp_path / "cluster"
    data.mkdir()
    (data / "postmaster.pid").write_text(contents, encoding="utf-8")
    with pytest.raises(probe.Stage8PostgreSQLProbeError) as failure:
        probe._postmaster_pidfile(data)
    assert failure.value.label == "POSTGRESQL_PROCESS_IDENTITY"


def test_postmaster_pid_data_directory_must_match_exact_owned_pgdata(tmp_path: Path) -> None:
    data = tmp_path / "cluster"
    data.mkdir()
    (data / "postmaster.pid").write_text(f"123\n{tmp_path / 'foreign'}\n100\n", encoding="utf-8")
    with pytest.raises(probe.Stage8PostgreSQLProbeError, match="exact owned PGDATA"):
        probe._postmaster_pidfile(data)


@pytest.mark.parametrize(
    ("image", "creation", "matches"),
    [("postgres.exe", 100.0, True), ("foreign.exe", 100.0, False), ("postgres.exe", 110.0, False)],
)
def test_postmaster_identity_requires_exact_image_and_creation_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image: str,
    creation: float,
    matches: bool,
) -> None:
    data = tmp_path / "cluster"
    data.mkdir()
    postgres = tmp_path / "postgres.exe"
    (data / "postmaster.pid").write_text(f"123\n{data}\n100\n", encoding="utf-8")

    class Handle:
        def __init__(self) -> None:
            self.closed = False

        def Close(self) -> None:
            self.closed = True

    handle = Handle()
    snapshot = probe.ProcessIdentitySnapshot(123, handle, creation, tmp_path / image)
    monkeypatch.setattr(probe, "_process_identity_snapshot", lambda _pid: snapshot)
    if matches:
        witness = probe._postmaster_witness(data, postgres)
        assert witness is not None and witness.process is handle
        probe._close_postmaster_witness(witness)
    else:
        with pytest.raises(probe.Stage8PostgreSQLProbeError):
            probe._postmaster_witness(data, postgres)
    assert handle.closed is True


def test_foreign_identity_blocks_pg_ctl_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        probe,
        "_postmaster_witness",
        lambda *_a: (_ for _ in ()).throw(
            probe.Stage8PostgreSQLProbeError(
                "POSTGRESQL_PROCESS_IDENTITY", probe.SUBSTRATE, "executable mismatch"
            )
        ),
    )
    monkeypatch.setattr(probe, "_run_pg_ctl", lambda *_a, **_k: pytest.fail("unsafe pg_ctl"))
    with pytest.raises(probe.Stage8PostgreSQLProbeError, match="executable mismatch"):
        probe._stop_if_owned_cluster_running(
            Path("pg_ctl.exe"),
            tmp_path / "cluster",
            tmp_path,
            tmp_path / "postgresql.log",
            cluster_started=True,
        )


def test_process_state_query_failure_is_typed_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WinError(Exception):
        winerror = 5

    fake_api = type(
        "Api", (), {"OpenProcess": staticmethod(lambda *_a: (_ for _ in ()).throw(WinError()))}
    )
    monkeypatch.setitem(sys.modules, "pywintypes", type("PyWinTypes", (), {"error": WinError}))
    monkeypatch.setitem(sys.modules, "win32api", fake_api)
    monkeypatch.setitem(sys.modules, "win32con", type("Con", (), {"SYNCHRONIZE": 1}))
    monkeypatch.setitem(sys.modules, "win32event", type("Event", (), {})())
    with pytest.raises(probe.Stage8PostgreSQLProbeError) as failure:
        probe._process_exists(123)
    assert failure.value.label == "POSTGRESQL_PROCESS_STATE"


def test_native_image_query_converts_pyhandle_without_taking_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[int] = []

    class Handle:
        def __int__(self) -> int:
            return 456

    class Query:
        argtypes = None
        restype = None

        def __call__(self, native_handle, _flags, buffer, length) -> int:
            observed.append(native_handle.value)
            buffer.value = str(tmp_path / "python.exe")
            length._obj.value = len(buffer.value)
            return 1

    query = Query()
    kernel32 = type("Kernel32", (), {"QueryFullProcessImageNameW": query})()
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_a, **_k: kernel32, raising=False)
    assert probe._query_process_image_path(Handle()) == tmp_path / "python.exe"
    assert observed == [456]
    assert query.argtypes is not None and query.restype is not None


def test_native_image_query_failure_is_typed_and_snapshot_closes_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Handle:
        closed = False

        def __int__(self) -> int:
            return 789

        def Close(self) -> None:
            self.closed = True

    class Query:
        argtypes = None
        restype = None

        def __call__(self, *_args) -> int:
            return 0

    handle = Handle()
    kernel32 = type("Kernel32", (), {"QueryFullProcessImageNameW": Query()})()
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_a, **_k: kernel32, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)

    class WinError(Exception):
        winerror = 87

    monkeypatch.setitem(sys.modules, "pywintypes", type("PyWinTypes", (), {"error": WinError}))
    monkeypatch.setitem(
        sys.modules,
        "win32api",
        type("Api", (), {"OpenProcess": staticmethod(lambda *_a: handle)}),
    )
    monkeypatch.setitem(
        sys.modules,
        "win32con",
        type("Con", (), {"SYNCHRONIZE": 1, "PROCESS_QUERY_LIMITED_INFORMATION": 2}),
    )
    monkeypatch.setitem(
        sys.modules,
        "win32event",
        type(
            "Event",
            (),
            {
                "WAIT_OBJECT_0": 0,
                "WAIT_TIMEOUT": 258,
                "WaitForSingleObject": staticmethod(lambda *_a: 258),
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "win32process",
        type(
            "Process", (), {"GetProcessTimes": staticmethod(lambda *_a: pytest.fail("unexpected"))}
        ),
    )
    with pytest.raises(probe.Stage8PostgreSQLProbeError, match="winerror=5") as failure:
        probe._process_identity_snapshot(123)
    assert failure.value.label == "POSTGRESQL_PROCESS_IDENTITY"
    assert handle.closed is True


def test_stage8_process_liveness_does_not_use_os_kill() -> None:
    tree = ast.parse(Path(probe.__file__).read_text(encoding="utf-8"))
    forbidden = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
        and node.func.attr == "kill"
    ]
    assert forbidden == []


def test_stop_uses_pre_stop_process_witness_and_exact_stopped_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "cluster"
    data.mkdir()
    events: list[str] = []
    handle = object()
    witness = probe.PostmasterWitness(123, handle, 1.0, Path("postgres.exe"))
    monkeypatch.setattr(
        probe, "_postmaster_witness", lambda *_a: events.append("identity") or witness
    )
    monkeypatch.setattr(
        probe, "_wait_process_exit", lambda value, **_k: events.append("exit-confirmed")
    )
    monkeypatch.setattr(probe, "_close_postmaster_witness", lambda value: events.append("close"))
    results = iter(
        (
            probe.PgCtlResult(0, 0.01, False, "", ""),
            probe.PgCtlResult(3, 0.01, False, "", ""),
        )
    )
    monkeypatch.setattr(
        probe,
        "_run_pg_ctl",
        lambda *_a, **_k: events.append("pg_ctl") or next(results),
    )
    assert (
        probe._stop_owned_cluster(Path("pg_ctl"), data, tmp_path, tmp_path / "postgresql.log")
        is None
    )
    assert events == ["identity", "pg_ctl", "pg_ctl", "exit-confirmed", "close"]


def test_stop_rejects_status_that_still_reports_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "cluster"
    data.mkdir()
    witness = probe.PostmasterWitness(123, object(), 1.0, Path("postgres.exe"))
    monkeypatch.setattr(probe, "_postmaster_witness", lambda *_a: witness)
    monkeypatch.setattr(probe, "_close_postmaster_witness", lambda *_a: None)
    results = iter(
        (
            probe.PgCtlResult(0, 0.01, False, "", ""),
            probe.PgCtlResult(0, 0.01, False, "", ""),
        )
    )
    monkeypatch.setattr(probe, "_run_pg_ctl", lambda *_a, **_k: next(results))
    error = probe._stop_owned_cluster(Path("pg_ctl"), data, tmp_path, tmp_path / "postgresql.log")
    assert error is not None
    assert "status_returncode=0" in error


@pytest.mark.skipif(sys.platform != "win32", reason="real Win32 process-liveness contract")
def test_windows_process_alive_check_is_non_destructive() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert probe._process_exists(child.pid) is True
        assert child.poll() is None
        assert probe._process_exists(child.pid) is True
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=10)


@pytest.mark.skipif(sys.platform != "win32", reason="real Win32 process-liveness contract")
def test_windows_exited_process_is_reported_stopped() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=10)
    assert probe._process_exists(child.pid) is False


@pytest.mark.skipif(sys.platform != "win32", reason="real Win32 process-identity contract")
def test_windows_live_foreign_pid_is_rejected_without_mutation(tmp_path: Path) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        snapshot = probe._process_identity_snapshot(child.pid)
        assert snapshot is not None
        creation = snapshot.creation_time
        snapshot.process.Close()
        data = tmp_path / "cluster"
        data.mkdir()
        (data / "postmaster.pid").write_text(f"{child.pid}\n{data}\n{creation}\n", encoding="utf-8")
        with pytest.raises(probe.Stage8PostgreSQLProbeError, match="executable"):
            probe._postmaster_witness(data, tmp_path / "pgbin" / "postgres.exe")
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=10)


@pytest.mark.skipif(sys.platform != "win32", reason="real Win32 process-identity contract")
def test_windows_process_creation_time_is_stable_and_non_destructive() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        first = probe._process_identity_snapshot(child.pid)
        second = probe._process_identity_snapshot(child.pid)
        assert first is not None and second is not None
        try:
            assert first.creation_time == second.creation_time > 0
            assert first.executable.resolve() == second.executable.resolve()
            assert first.executable.resolve() == Path(sys.executable).resolve()
            assert child.poll() is None
        finally:
            first.process.Close()
            second.process.Close()
    finally:
        child.terminate()
        child.wait(timeout=10)
