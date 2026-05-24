from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scripts.controlled_paper_runtime_validation as target_module

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/controlled_paper_runtime_validation.py"
SAFE_CONFIG = REPO_ROOT / "config/e2e/demo_paper.yml"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_health_resource_summaries(payload: dict[str, object]) -> None:
    summary = payload["summary"]
    assert isinstance(summary, dict)
    health = summary["health_summary"]
    assert health["enabled"] is True
    assert health["status"] in {"ok", "warning", "unavailable"}
    assert health["long_run_ready"] is False
    assert "duration_guard_below_24h" in health["long_run_blockers"]
    assert "checkpoint_heartbeat_not_enabled" not in health["long_run_blockers"]
    assert isinstance(health["checkpoint_policy"], dict)
    assert health["checkpoint_policy"]["enabled"] is True
    assert health["checkpoint_policy"]["mode"] == "step_boundary"
    assert isinstance(health["heartbeat_policy"], dict)
    assert health["heartbeat_policy"]["enabled"] is True
    assert health["heartbeat_policy"]["mode"] == "step_boundary"
    assert isinstance(health["artifact_policy"], dict)

    resources = summary["process_resource_summary"]
    assert resources["enabled"] is True
    assert isinstance(resources["cpu_process_time_start_seconds"], int | float)
    assert isinstance(resources["cpu_process_time_end_seconds"], int | float)
    assert resources["cpu_process_time_delta_seconds"] >= 0
    assert isinstance(resources["memory_rss_available"], bool)
    if resources["memory_rss_available"]:
        assert isinstance(resources["memory_rss_start_bytes"], int)
        assert isinstance(resources["memory_rss_end_bytes"], int)
        assert isinstance(resources["memory_rss_delta_bytes"], int)
    else:
        assert resources["memory_rss_start_bytes"] is None
        assert resources["memory_rss_end_bytes"] is None
        assert resources["memory_rss_delta_bytes"] is None
    assert isinstance(resources["resource_warnings"], list)

    progress = summary["progress_summary"]
    assert progress["checkpoints_enabled"] is True
    assert progress["checkpoint_count"] >= 2
    assert progress["heartbeat_count"] >= 1
    assert progress["progress_observations_count"] >= progress["checkpoint_count"]
    assert progress["heartbeat_interval_seconds"] is None
    assert progress["heartbeat_mode"] == "step_boundary"
    assert progress["progress_observations_available"] is True
    assert isinstance(progress["checkpoint_labels"], list)

    artifact = summary["artifact_summary"]
    assert isinstance(artifact["artifact_warnings"], list)
    assert artifact["log_size_available"] is False
    assert artifact["log_size_bytes"] is None


def test_controlled_paper_runtime_validation_happy_path_json() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--duration-seconds",
        "5",
        "--max-signals",
        "1",
        "--run-id",
        "test-run-001",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["report_path"] is None
    assert payload["run_id"] == "test-run-001"
    assert isinstance(payload["started_at"], str)
    assert isinstance(payload["ended_at"], str)
    assert isinstance(payload["elapsed_seconds"], int | float)
    assert payload["elapsed_seconds"] >= 0
    assert payload["mode"] == "demo"
    assert len(payload["steps"]) == 3
    assert [step["name"] for step in payload["steps"]] == [
        "preview_plan",
        "mock_runtime_preview",
        "controller_mock_preview",
    ]
    assert all(step["exit_code"] == 0 for step in payload["steps"])
    summary = payload["summary"]
    assert summary["steps_total"] == 3
    assert summary["steps_passed"] == 3
    assert summary["run_id"] == "test-run-001"
    assert summary["started_at"] == payload["started_at"]
    assert summary["ended_at"] == payload["ended_at"]
    assert summary["elapsed_seconds"] == payload["elapsed_seconds"]
    assert summary["bounded_validation_loop"] is True
    assert summary["production_runtime_loop_started"] is False
    assert summary["runtime_loop_started"] is False
    assert summary["shutdown_completed"] is True
    assert isinstance(summary["active_threads_before"], int)
    assert isinstance(summary["active_threads_after_shutdown"], int)
    assert isinstance(summary["active_non_daemon_threads_after_shutdown"], list)
    assert summary["exchange_io"] == "disabled"
    assert summary["api_keys_required"] is False
    assert summary["secrets_read"] is False
    assert summary["keychain_read"] is False
    assert summary["env_values_read"] is False
    assert summary["real_orders_submitted"] is False
    assert summary["order_execution"] == "mocked_or_disabled"
    assert summary["controller_backed_preview"] is True
    assert summary["timeout_triggered"] is False
    assert summary["timeout_step"] is None
    assert "order_events_count" in summary
    assert "simulated_orders_count" in summary
    assert "journal_events_count" in summary
    assert "journal_events_available" in summary
    assert summary["journal_events_count"] is None or isinstance(
        summary["journal_events_count"], int
    )
    assert summary["journal_visibility"] in {"not_available_in_mock_preview", "available"}
    progress = summary["progress_summary"]
    labels = progress.get("checkpoint_labels", [])
    assert "session_started" in labels
    assert "before_step:preview_plan" in labels
    assert "after_step:preview_plan" in labels
    assert "before_step:mock_runtime_preview" in labels
    assert "after_step:mock_runtime_preview" in labels
    assert "before_step:controller_mock_preview" in labels
    assert "after_step:controller_mock_preview" in labels
    assert "session_finished" in labels
    _assert_health_resource_summaries(payload)
    assert payload["issues"] == []
    assert payload["safety_contract_version"] == "controlled_paper_runtime_validation.v1"


def test_controlled_paper_runtime_validation_live_blocked() -> None:
    result = _run(
        "--mode", "live", "--config", str(SAFE_CONFIG), "--run-id", "live-blocked-run", "--json"
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["report_path"] is None
    assert payload["run_id"] == "live-blocked-run"
    assert isinstance(payload["started_at"], str)
    assert isinstance(payload["ended_at"], str)
    assert payload["elapsed_seconds"] >= 0
    assert payload["reason"] == "controlled_paper_runtime_validation_forbids_live_mode"
    assert payload["steps"] == []
    assert payload["child_commands"] == []
    _assert_health_resource_summaries(payload)
    assert "live_mode_not_allowed" in payload["issues"]


def test_controlled_paper_runtime_validation_invalid_duration_low() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--duration-seconds",
        "0",
        "--run-id",
        "invalid-duration-low",
        "--json",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["reason"] == "controlled_paper_runtime_validation_invalid_duration"
    assert payload["run_id"] == "invalid-duration-low"
    assert payload["steps"] == []
    assert payload["child_commands"] == []
    progress = payload["summary"]["progress_summary"]
    assert progress["checkpoint_count"] >= 1
    assert payload["summary"]["health_summary"]["long_run_ready"] is False


def test_controlled_paper_runtime_validation_duration_300_allowed(monkeypatch, capsys) -> None:
    commands: list[list[str]] = []

    def _fake_run(command, **kwargs):  # noqa: ANN001
        commands.append(command)

        class _Result:
            def __init__(self) -> None:
                self.stdout = json.dumps({"status": "ok"})
                self.stderr = ""
                self.returncode = 0

        return _Result()

    monkeypatch.setattr(target_module.subprocess, "run", _fake_run)
    code = target_module.main(
        [
            "--mode",
            "demo",
            "--config",
            str(SAFE_CONFIG),
            "--duration-seconds",
            "300",
            "--max-signals",
            "1",
            "--run-id",
            "duration-300-allowed",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["duration_seconds"] == 300
    assert payload["run_id"] == "duration-300-allowed"
    assert len(commands) == 3


def test_controlled_paper_runtime_validation_duration_3600_allowed(monkeypatch, capsys) -> None:
    commands: list[list[str]] = []

    def _fake_run(command, **kwargs):  # noqa: ANN001
        commands.append(command)

        class _Result:
            def __init__(self) -> None:
                self.stdout = json.dumps({"status": "ok"})
                self.stderr = ""
                self.returncode = 0

        return _Result()

    monkeypatch.setattr(target_module.subprocess, "run", _fake_run)
    code = target_module.main(
        [
            "--mode",
            "demo",
            "--config",
            str(SAFE_CONFIG),
            "--duration-seconds",
            "3600",
            "--run-id",
            "duration-3600-allowed",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["run_id"] == "duration-3600-allowed"
    assert payload["duration_seconds"] == 3600
    assert len(commands) == 3


def test_controlled_paper_runtime_validation_invalid_duration_3601() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--duration-seconds",
        "3601",
        "--run-id",
        "invalid-duration-high",
        "--json",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["reason"] == "controlled_paper_runtime_validation_invalid_duration"
    assert payload["run_id"] == "invalid-duration-high"
    assert payload["steps"] == []
    assert payload["child_commands"] == []


def test_controlled_paper_runtime_validation_invalid_max_signals_low() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--max-signals",
        "0",
        "--run-id",
        "invalid-max-signals-low",
        "--json",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["reason"] == "controlled_paper_runtime_validation_invalid_max_signals"
    assert payload["run_id"] == "invalid-max-signals-low"
    assert payload["steps"] == []
    assert payload["child_commands"] == []


def test_controlled_paper_runtime_validation_invalid_max_signals_high() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--max-signals",
        "999",
        "--run-id",
        "invalid-max-signals-high",
        "--json",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["reason"] == "controlled_paper_runtime_validation_invalid_max_signals"
    assert payload["run_id"] == "invalid-max-signals-high"
    assert payload["steps"] == []
    assert payload["child_commands"] == []


def test_controlled_paper_runtime_validation_happy_path_report_written(tmp_path: Path) -> None:
    report_path = tmp_path / "reports" / "controlled.json"
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--duration-seconds",
        "5",
        "--max-signals",
        "1",
        "--run-id",
        "report-happy",
        "--report-path",
        str(report_path),
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["run_id"] == "report-happy"
    assert payload["report_path"] == str(report_path)
    assert report_path.exists()
    file_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert file_payload == payload
    _assert_health_resource_summaries(payload)
    artifact = payload["summary"]["artifact_summary"]
    assert artifact["report_path"] == str(report_path)
    assert artifact["report_size_available"] is True
    assert isinstance(artifact["report_size_bytes"], int)
    assert artifact["report_size_bytes"] > 0
    assert isinstance(artifact["max_report_size_bytes"], int)
    assert artifact["max_report_size_bytes"] > artifact["report_size_bytes"]


def test_controlled_paper_runtime_validation_live_blocked_report_written(tmp_path: Path) -> None:
    report_path = tmp_path / "reports" / "live_blocked.json"
    result = _run(
        "--mode",
        "live",
        "--config",
        str(SAFE_CONFIG),
        "--run-id",
        "report-live-blocked",
        "--report-path",
        str(report_path),
        "--json",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["reason"] == "controlled_paper_runtime_validation_forbids_live_mode"
    assert payload["steps"] == []
    assert payload["child_commands"] == []
    assert payload["report_path"] == str(report_path)
    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8")) == payload
    _assert_health_resource_summaries(payload)


def test_controlled_paper_runtime_validation_invalid_duration_report_written(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "reports" / "invalid_duration.json"
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--duration-seconds",
        "3601",
        "--run-id",
        "report-invalid-duration",
        "--report-path",
        str(report_path),
        "--json",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["reason"] == "controlled_paper_runtime_validation_invalid_duration"
    assert payload["report_path"] == str(report_path)
    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8")) == payload
    _assert_health_resource_summaries(payload)


def test_controlled_paper_runtime_validation_invalid_duration_86400() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--duration-seconds",
        "86400",
        "--run-id",
        "invalid-duration-86400",
        "--json",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["reason"] == "controlled_paper_runtime_validation_invalid_duration"
    assert payload["steps"] == []
    assert payload["child_commands"] == []


def test_controlled_paper_runtime_validation_invalid_duration_259200() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--duration-seconds",
        "259200",
        "--run-id",
        "invalid-duration-259200",
        "--json",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["reason"] == "controlled_paper_runtime_validation_invalid_duration"
    assert payload["steps"] == []
    assert payload["child_commands"] == []


def test_controlled_paper_runtime_validation_child_timeout_policy_for_3600() -> None:
    assert target_module._child_timeout_seconds("mock_runtime_preview", 3600) == 3630
    assert target_module._child_timeout_seconds("preview_plan", 3600) == 30
    assert target_module._child_timeout_seconds("controller_mock_preview", 3600) == 30


def test_controlled_paper_runtime_validation_no_api_keys_required(monkeypatch) -> None:
    for key in ("API_KEY", "API_SECRET", "BINANCE_API_KEY", "BINANCE_API_SECRET"):
        monkeypatch.delenv(key, raising=False)
    result = _run("--mode", "demo", "--config", str(SAFE_CONFIG), "--json")
    assert result.returncode == 0, result.stderr


def test_controlled_paper_runtime_validation_memory_rss_fallback_without_resource(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(target_module, "_resource", None)
    code = target_module.main(
        ["--mode", "demo", "--config", str(SAFE_CONFIG), "--duration-seconds", "1", "--json"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    resources = payload["summary"]["process_resource_summary"]
    assert resources["memory_rss_available"] is False
    assert resources["memory_rss_start_bytes"] is None
    assert resources["memory_rss_end_bytes"] is None
    assert resources["memory_rss_delta_bytes"] is None
    assert "memory_rss_unavailable" in resources["resource_warnings"]


def test_controlled_paper_runtime_validation_output_cp1252_safe() -> None:
    result = _run("--mode", "demo", "--config", str(SAFE_CONFIG), "--json")
    assert result.returncode == 0, result.stderr
    result.stdout.encode("cp1252")


def test_controlled_paper_runtime_validation_output_cp1252_safe_with_report_path(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "reports" / "cp1252_report.json"
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--report-path",
        str(report_path),
        "--json",
    )
    assert result.returncode == 0, result.stderr
    result.stdout.encode("cp1252")


def test_controlled_paper_runtime_validation_source_safety() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "ccxt",
        "create_order",
        "fetch_balance",
        "fetch_ticker",
        "load_markets",
        "get_secret",
        "keychain",
        "os.environ",
        "getenv",
        "shell=True",
        "TradingController(",
    )
    for token in forbidden:
        assert token not in source
    assert "timeout=" in source
    assert "uuid" in source
    assert "time.perf_counter" in source


def test_controlled_paper_runtime_validation_child_commands_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "scripts/run_local_bot.py" in source
    assert "--preview-plan" in source
    assert "scripts/mock_runtime_preview.py" in source
    assert "scripts/controller_mock_preview.py" in source
    assert source.count("scripts/") == 3


def test_controlled_paper_runtime_validation_timeout_propagation(monkeypatch, capsys) -> None:
    calls = {"count": 0}
    report_path = REPO_ROOT / "tmp" / "timeout-report.json"

    def _fake_run(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise subprocess.TimeoutExpired(
                cmd=kwargs.get("args", args[0]),
                timeout=35,
                output=b"partial stdout \xc5\x82",
                stderr=b"partial stderr \xc5\x82",
            )

        class _Result:
            def __init__(self, stdout: str, returncode: int = 0) -> None:
                self.stdout = stdout
                self.returncode = returncode

        cmd = args[0]
        if "run_local_bot.py" in cmd[1]:
            return _Result(json.dumps({"status": "ok"}), 0)
        return _Result(json.dumps({"status": "ok"}), 0)

    monkeypatch.setattr(target_module.subprocess, "run", _fake_run)
    code = target_module.main(
        [
            "--mode",
            "demo",
            "--config",
            str(SAFE_CONFIG),
            "--duration-seconds",
            "5",
            "--max-signals",
            "1",
            "--run-id",
            "timeout-run-001",
            "--report-path",
            str(report_path),
            "--json",
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["report_path"] == str(report_path)
    assert payload["status"] == "blocked"
    assert payload["run_id"] == "timeout-run-001"
    assert isinstance(payload["started_at"], str)
    assert isinstance(payload["ended_at"], str)
    assert payload["elapsed_seconds"] >= 0
    assert payload["failed_step"] == "mock_runtime_preview"
    failed_payload = payload["steps"][1]["payload"]
    assert failed_payload["reason"] == "controlled_paper_runtime_validation_child_timeout"
    assert isinstance(failed_payload["stdout"], str)
    assert isinstance(failed_payload["stderr"], str)
    json.dumps(payload)
    assert "step_timeout:mock_runtime_preview" in payload["issues"]
    assert "step_failed:mock_runtime_preview" in payload["issues"]
    assert payload["summary"]["shutdown_completed"] is False
    assert payload["summary"]["errors_count"] == 1
    assert payload["summary"]["timeout_triggered"] is True
    assert payload["summary"]["timeout_step"] == "mock_runtime_preview"
    progress = payload["summary"]["progress_summary"]
    assert progress["checkpoint_count"] > 0
    assert payload["safety_contract_version"] == "controlled_paper_runtime_validation.v1"
    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8")) == payload


def test_controlled_paper_runtime_validation_report_write_failure_is_controlled(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    report_path = tmp_path / "report.json"

    def _raise_write_error(payload, path):  # noqa: ANN001
        raise OSError("disk full")

    monkeypatch.setattr(target_module, "_write_report", _raise_write_error)
    code = target_module.main(
        [
            "--mode",
            "demo",
            "--config",
            str(SAFE_CONFIG),
            "--duration-seconds",
            "5",
            "--max-signals",
            "1",
            "--run-id",
            "report-write-failed",
            "--report-path",
            str(report_path),
            "--json",
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["reason"] == "controlled_paper_runtime_validation_report_write_failed"
    assert "report_write_failed" in payload["issues"]
    assert payload["report_path"] == str(report_path)
    assert payload["run_id"] == "report-write-failed"
    assert isinstance(payload["started_at"], str)
    assert isinstance(payload["ended_at"], str)
    assert payload["elapsed_seconds"] >= 0
    if "summary" in payload:
        assert payload["summary"]["run_id"] == "report-write-failed"
    json.dumps(payload)
