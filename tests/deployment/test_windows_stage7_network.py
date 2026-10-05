from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from deployment import windows_stage7_network_probe as probe


SOURCE = Path(probe.__file__).read_text(encoding="utf-8")


def _wait_state(path: Path, phase: str) -> dict[str, object]:
    deadline = time.monotonic() + probe.STATE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        state = probe._read_state(path)
        if state is not None and state.get("phase") == phase:
            return state
        time.sleep(0.01)
    raise AssertionError(f"peer did not reach {phase}: {probe._read_state(path)}")


def test_non_windows_cannot_produce_live_pass(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("non-Windows guard is exercised on other matrix runners")
    with pytest.raises(probe.Stage7NetworkProbeError, match="restricted to Windows"):
        probe.run_probe(tmp_path)


def test_probe_uses_production_client_and_real_loopback_socket() -> None:
    assert "from core.network import RateLimitedAsyncClient" in SOURCE
    assert SOURCE.count("client.request(") == 1
    assert "socket.socket(socket.AF_INET, socket.SOCK_STREAM)" in SOURCE
    assert "MockTransport" not in SOURCE
    assert "http://127.0.0.1:" in SOURCE
    assert "SO_LINGER" in SOURCE


def test_atomic_state_update_uses_flushed_temporary_then_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "peer-state.json"
    replaced: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def observe(source: Path, target: Path) -> None:
        assert source != destination
        assert json.loads(source.read_text(encoding="utf-8"))["phase"] == "READY"
        replaced.append((source, target))
        real_replace(source, target)

    monkeypatch.setattr(probe.os, "replace", observe)
    probe._write_state(
        destination, {"phase": "READY", "port": 12345, "connections": 0, "events": ["READY"]}
    )
    assert replaced and replaced[0][1] == destination
    assert probe._read_state(destination)["port"] == 12345  # type: ignore[index]


def test_real_peer_records_dynamic_port_and_exact_order_without_stdout(tmp_path: Path) -> None:
    state_path = tmp_path / "peer-state.json"
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", probe.__name__, "--peer", str(state_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = _wait_state(state_path, "READY")
        port = ready["port"]
        assert isinstance(port, int) and port > 0
        assert ready == {"phase": "READY", "port": port, "connections": 0, "events": ["READY"]}

        first = socket.create_connection(("127.0.0.1", port), timeout=2)
        _wait_state(state_path, "FIRST_CONNECTION_ACCEPTED")
        first.sendall(b"GET /recovery HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        state = _wait_state(state_path, "FIRST_CONNECTION_RESET")
        assert state["connections"] == 1
        assert state["events"] == list(probe.PHASES[:4])
        with pytest.raises((ConnectionResetError, BrokenPipeError, OSError)):
            while first.recv(1024):
                pass
        first.close()

        second = socket.create_connection(("127.0.0.1", port), timeout=2)
        _wait_state(state_path, "SECOND_CONNECTION_ACCEPTED")
        second.sendall(b"GET /recovery HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        response = b""
        while chunk := second.recv(1024):
            response += chunk
        second.close()
        assert b"200 OK" in response and response.endswith(b"RECOVERED")
        process.wait(timeout=3)
        final = _wait_state(state_path, "SUCCESS")
        assert final == {
            "phase": "SUCCESS",
            "port": port,
            "connections": 2,
            "events": list(probe.PHASES),
        }
        assert process.returncode == 0
        assert process.stderr is not None and process.stderr.read() == ""
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


def test_phase_wait_is_bounded_and_does_not_use_stdout(tmp_path: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(1)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="READY"):
            probe._wait_for_phase(tmp_path / "missing.json", process, "READY", timeout=0.05)
        assert time.monotonic() - started < 0.5
    finally:
        process.kill()
        process.communicate(timeout=2)
    assert "readline" not in SOURCE


def test_failure_diagnostic_preserves_phase_exception_and_exited_peer_stderr() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stderr.write('peer exploded')"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    _, stderr = process.communicate(timeout=2)
    detail = probe._diagnostic(
        {
            "phase": "FIRST_CONNECTION_ACCEPTED",
            "connections": 1,
            "events": ["READY", "FIRST_CONNECTION_ACCEPTED"],
        },
        process,
        stderr,
        TimeoutError("client deadline"),
    )
    assert "phase=FIRST_CONNECTION_ACCEPTED" in detail
    assert "connections=1" in detail
    assert "TimeoutError: client deadline" in detail
    assert "peer_alive=false" in detail
    assert "peer_returncode=0" in detail
    assert "peer exploded" in detail


@pytest.mark.skipif(os.name == "nt", reason="POSIX preserves proxy aliases independently")
def test_direct_loopback_proxy_isolation_preserves_posix_aliases_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_PROXY", "internal.example")
    monkeypatch.delenv("no_proxy", raising=False)
    with probe._direct_loopback_environment():
        assert os.environ["NO_PROXY"] == "internal.example,127.0.0.1,localhost"
        assert os.environ["no_proxy"] == "127.0.0.1,localhost"
    assert os.environ["NO_PROXY"] == "internal.example"
    assert "no_proxy" not in os.environ


def _effective_windows_no_proxy() -> str | None:
    matches = [value for name, value in os.environ.items() if name.casefold() == "no_proxy"]
    assert len(matches) <= 1
    return matches[0] if matches else None


@pytest.mark.skipif(os.name != "nt", reason="Windows environment semantics required")
def test_direct_loopback_proxy_isolation_restores_absent_windows_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert _effective_windows_no_proxy() is None

    with probe._direct_loopback_environment():
        assert _effective_windows_no_proxy() == "127.0.0.1,localhost"

    assert _effective_windows_no_proxy() is None


@pytest.mark.skipif(os.name != "nt", reason="Windows environment semantics required")
def test_direct_loopback_proxy_isolation_preserves_and_restores_windows_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "internal.example")
    assert _effective_windows_no_proxy() == "internal.example"

    with probe._direct_loopback_environment():
        assert _effective_windows_no_proxy() == "internal.example,127.0.0.1,localhost"

    assert _effective_windows_no_proxy() == "internal.example"


def test_every_external_wait_and_socket_operation_is_bounded() -> None:
    assert "server.settimeout(PEER_TIMEOUT_SECONDS)" in SOURCE
    assert SOURCE.count("settimeout(HTTP_TIMEOUT_SECONDS)") == 2
    assert "time.monotonic() + timeout" in SOURCE
    assert "process.wait(timeout=PEER_TIMEOUT_SECONDS)" in SOURCE
    assert "process.communicate(timeout=5)" in SOURCE


def test_probe_has_only_owned_scratch_and_subprocess_cleanup() -> None:
    assert "TemporaryDirectory(dir=scratch_parent)" in SOURCE
    assert "process.terminate()" in SOURCE
    assert "process.kill()" in SOURCE
    for forbidden in ("binance", "github.com", "netsh", "shutdown.exe", "MockTransport"):
        assert forbidden not in SOURCE


def test_cleanup_failure_without_primary_fails_closed() -> None:
    with pytest.raises(probe.Stage7NetworkProbeError) as failure:
        probe._report_cleanup_errors(["peer process survived kill deadline"], None)
    assert failure.value.label == "NETWORK_RECOVERY_VERIFY"
    assert str(failure.value) == "[NETWORK_RECOVERY_VERIFY] peer process survived kill deadline"


def test_cleanup_failure_preserves_primary_and_reports_secondary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    primary = probe.Stage7NetworkProbeError(
        "NETWORK_RECOVERY_RECONNECT", "production request did not recover"
    )
    with pytest.raises(probe.Stage7NetworkProbeError) as failure:
        try:
            raise primary
        except BaseException as caught:
            try:
                raise
            finally:
                probe._report_cleanup_errors(["peer process survived kill deadline"], caught)
    assert failure.value is primary
    assert capsys.readouterr().err == (
        "[CLEANUP] secondary failure: peer process survived kill deadline\n"
    )


def test_no_cleanup_error_produces_no_output(capsys: pytest.CaptureFixture[str]) -> None:
    probe._report_cleanup_errors([], RuntimeError("primary"))
    assert capsys.readouterr().err == ""
