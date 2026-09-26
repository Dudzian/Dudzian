from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from deployment import windows_stage7_network_probe as probe


SOURCE = Path(probe.__file__).read_text(encoding="utf-8")


def test_non_windows_cannot_produce_live_pass(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("non-Windows guard is exercised on other matrix runners")
    with pytest.raises(probe.Stage7NetworkProbeError, match="restricted to Windows"):
        probe.run_probe(tmp_path)


def test_probe_uses_production_client_and_real_loopback_socket() -> None:
    assert "from core.network import RateLimitedAsyncClient" in SOURCE
    assert "RateLimitedAsyncClient(" in SOURCE
    assert "socket.socket(socket.AF_INET, socket.SOCK_STREAM)" in SOURCE
    assert '"127.0.0.1"' in SOURCE
    assert "MockTransport" not in SOURCE
    assert "_client.request" not in SOURCE
    assert "http://127.0.0.1:" in SOURCE


def test_failure_reconnect_success_proof_is_ordered_and_exact() -> None:
    tree = ast.parse(SOURCE)
    constants = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]
    assert constants.index("FIRST_FAILURE") < constants.index("RECONNECT")
    assert "SUCCESS" in constants
    assert "RECOVERED" in constants
    assert "response.status_code" in SOURCE
    assert 'response.text != "RECOVERED"' in SOURCE


def test_every_external_wait_and_socket_operation_is_bounded() -> None:
    assert "server.settimeout(PEER_TIMEOUT_SECONDS)" in SOURCE
    assert SOURCE.count("settimeout(HTTP_TIMEOUT_SECONDS)") == 2
    assert "output.get(timeout=MARKER_TIMEOUT_SECONDS)" in SOURCE
    assert "process.wait(timeout=PEER_TIMEOUT_SECONDS)" in SOURCE
    assert "process.wait(timeout=5)" in SOURCE
    assert "time.sleep" not in SOURCE


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
    assert str(failure.value) == ("[NETWORK_RECOVERY_VERIFY] peer process survived kill deadline")


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
    assert failure.value.label == "NETWORK_RECOVERY_RECONNECT"
    assert str(failure.value) == ("[NETWORK_RECOVERY_RECONNECT] production request did not recover")
    assert capsys.readouterr().err == (
        "[CLEANUP] secondary failure: peer process survived kill deadline\n"
    )


def test_no_cleanup_error_produces_no_output(capsys: pytest.CaptureFixture[str]) -> None:
    probe._report_cleanup_errors([], RuntimeError("primary"))
    assert capsys.readouterr().err == ""
