from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVICE = (ROOT / "deployment/windows_test_service.py").read_text(encoding="utf-8")
PROBE = (ROOT / "deployment/windows_scm_probe.ps1").read_text(encoding="utf-8")
SCM = (ROOT / "deployment/windows_stage7_scm.py").read_text(encoding="utf-8")


def test_service_shutdown_is_a_thin_shared_signal_handler() -> None:
    tree = ast.parse(SERVICE)
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    shutdown = ast.unparse(methods["SvcShutdown"])
    stop = ast.unparse(methods["SvcStop"])
    assert "_request_stop('shutdown')" in shutdown
    assert "_request_stop('stop')" in stop
    for forbidden in ("tree.close", "unlink", "subprocess", "sqlite", "wait"):
        assert forbidden not in shutdown.lower()


def test_private_control_is_in_range_and_delegates_exactly() -> None:
    assert "STAGE7_ACCEPTANCE_SHUTDOWN_CONTROL = 200" in SERVICE
    assert "STAGE7_ACCEPTANCE_SHUTDOWN_CONTROL = 200" in SCM
    assert "if control == STAGE7_ACCEPTANCE_SHUTDOWN_CONTROL:" in SERVICE
    assert "self.SvcShutdown()" in SERVICE
    assert "win32service.ControlService(service, STAGE7_ACCEPTANCE_SHUTDOWN_CONTROL)" in SCM


def test_live_probe_queries_acceptance_before_sending_control() -> None:
    query = PROBE.index('"deployment.windows_stage7_scm", "query"')
    accepted = PROBE.index("SERVICE_ACCEPT_SHUTDOWN", query)
    control = PROBE.index('"deployment.windows_stage7_scm", "shutdown"', accepted)
    stopped = PROBE.index('Wait-State "Stopped"', control)
    tree_gone = PROBE.index('Wait-TreeGone $shutdownTree "SAFE_OS_SHUTDOWN"', stopped)
    shutdown_log = PROBE.index('Assert-LogEvent "SERVICE_SHUTDOWN_REQUEST"', tree_gone)
    stop_log = PROBE.index('Assert-LogEvent "SERVICE_STOP"', shutdown_log)
    passed = PROBE.index('$result.WINDOWS_SAFE_OS_SHUTDOWN = "PASS"', stop_log)
    assert query < accepted < control < stopped < tree_gone < shutdown_log < stop_log < passed
    critical = PROBE[control:passed]
    assert "Stop-Service" not in critical
    assert "taskkill" not in critical.lower()
    assert "TerminateProcess" not in critical


def test_stage5_assertions_precede_stage7_and_waits_remain_bounded() -> None:
    stage5_pass = PROBE.index('$result.WINDOWS_PERSISTENT_LOGGING = "PASS"')
    stage7 = PROBE.index('$stage = "SAFE_OS_SHUTDOWN_ACCEPTANCE_QUERY"')
    assert stage5_pass < stage7
    assert "for ($i = 0; $i -lt 60; $i++)" in PROBE
    for label in (
        "SAFE_OS_SHUTDOWN_ACCEPTANCE_QUERY",
        "SAFE_OS_SHUTDOWN_CONTROL",
        "SAFE_OS_SHUTDOWN_STOP",
        "SAFE_OS_SHUTDOWN_PROCESS_TREE",
        "SAFE_OS_SHUTDOWN_LOGGING",
    ):
        assert label in PROBE
