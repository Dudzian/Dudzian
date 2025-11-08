from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from tests.test_runtime_pipeline_offline import (
    _LoopbackExchangeState,
    _materialize_loopback_configs,
    loopback_exchange_server as base_loopback_exchange_server,
)


@pytest.fixture()
def loopback_exchange_server(base_loopback_exchange_server: _LoopbackExchangeState) -> _LoopbackExchangeState:
    return base_loopback_exchange_server


def _run_local_bot(runtime_path: Path, entrypoint: str, mode: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    reports_dir = tmp_path / "reports"
    markdown_dir = tmp_path / "markdown"
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = state_dir / "state.json"
    checkpoint_payload = {
        "entrypoint": entrypoint,
        "mode": "demo",
        "config_path": str(runtime_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_path.write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    repo_root = str(Path.cwd())
    env["PYTHONPATH"] = repo_root if not python_path else f"{repo_root}:{python_path}"
    cmd = [
        sys.executable,
        "scripts/run_local_bot.py",
        "--config",
        str(runtime_path),
        "--entrypoint",
        entrypoint,
        "--mode",
        mode,
        "--max-runtime",
        "1.5",
        "--no-ready-stdout",
        "--report-dir",
        str(reports_dir),
        "--report-markdown-dir",
        str(markdown_dir),
        "--state-dir",
        str(state_dir),
    ]
    return subprocess.run(cmd, check=False, capture_output=True, text=True, env=env, cwd=Path.cwd())


def _load_latest_report(reports_dir: Path) -> dict[str, object]:
    if not reports_dir.exists():
        return {}
    candidates = list(reports_dir.glob("run_local_bot_*"))
    if not candidates:
        return {}
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    payload = json.loads(latest.read_text(encoding="utf-8"))
    return payload


def test_run_local_bot_loopback_modes(
    tmp_path: Path,
    loopback_exchange_server: _LoopbackExchangeState,
) -> None:
    core_path, runtime_path = _materialize_loopback_configs(tmp_path, port=loopback_exchange_server.port)

    paper_result = _run_local_bot(runtime_path, "loopback_paper", "paper", tmp_path)
    assert paper_result.returncode == 0, paper_result.stderr

    paper_report = _load_latest_report(tmp_path / "reports")
    assert paper_report.get("status") == "success"
    assert paper_report.get("mode") == "paper"

    live_result = _run_local_bot(runtime_path, "loopback_testnet", "live", tmp_path)
    assert live_result.returncode == 0, live_result.stderr

    live_reports_root = (tmp_path / "reports").resolve()
    live_markdown_root = (tmp_path / "markdown").resolve()

    live_report = _load_latest_report(tmp_path / "reports")
    assert live_report.get("status") == "success"
    assert live_report.get("mode") == "live"

    live_metrics = live_report.get("live_execution_metrics")
    assert live_metrics and live_metrics.get("entries")
    loopback_entries = [entry for entry in live_metrics["entries"] if entry.get("exchange") == "loopback_spot"]
    assert loopback_entries, f"Brak wpisów metryk dla loopback_spot: {live_metrics}"
    aggregated_entry = next((entry for entry in loopback_entries if entry.get("route") is None), None)
    assert aggregated_entry is not None, f"Brak zagregowanego wpisu bez trasy: {loopback_entries}"
    default_route_entry = next((entry for entry in loopback_entries if entry.get("route") == "default"), None)
    aggregated_metrics = aggregated_entry.get("metrics", {})
    if aggregated_metrics.get("orders_total"):
        assert default_route_entry is not None, f"Brak wpisu dla trasy default: {loopback_entries}"
        assert default_route_entry.get("metrics") == aggregated_metrics
    for entry in loopback_entries:
        assert "route" in entry
        metrics = entry.get("metrics", {})
        fill_count = metrics.get("fill_ratio_count")
        assert fill_count is not None and fill_count >= 0
        fill_avg = metrics.get("fill_ratio_avg")
        fill_p95 = metrics.get("fill_ratio_p95")
        fill_min = metrics.get("fill_ratio_min")
        fill_max = metrics.get("fill_ratio_max")
        fill_stddev = metrics.get("fill_ratio_stddev")
        if fill_count:
            assert fill_avg == pytest.approx(1.0)
            assert metrics.get("fill_ratio_sum") == pytest.approx(fill_count * fill_avg)
            assert metrics.get("fill_ratio_p50") == pytest.approx(1.0)
            assert fill_p95 == pytest.approx(1.0)
            assert fill_min == pytest.approx(1.0)
            assert fill_max == pytest.approx(1.0)
            assert fill_stddev == pytest.approx(0.0)
        else:
            assert fill_avg is None
            assert metrics.get("fill_ratio_sum") == pytest.approx(0.0)
            assert metrics.get("fill_ratio_p50") is None
            assert fill_p95 is None
            assert fill_min is None
            assert fill_max is None
            assert fill_stddev is None
        latency_avg = metrics.get("latency_avg")
        latency_p95 = metrics.get("latency_p95")
        latency_p99 = metrics.get("latency_p99")
        latency_count = metrics.get("latency_count")
        latency_sum = metrics.get("latency_sum")
        latency_p50 = metrics.get("latency_p50")
        latency_min = metrics.get("latency_min")
        latency_max = metrics.get("latency_max")
        latency_stddev = metrics.get("latency_stddev")
        if fill_count:
            assert latency_avg is not None and latency_avg >= 0.0
            assert latency_p95 is not None and latency_p95 >= latency_avg
            assert latency_p99 is not None and latency_p99 >= latency_p95
            assert latency_count == fill_count
            assert latency_sum is not None and latency_sum >= 0.0
            assert latency_p50 is not None and 0.0 <= latency_p50 <= latency_p95 <= latency_p99
            assert latency_min is not None and latency_min <= latency_p50
            assert latency_max is not None and latency_max >= latency_p99
            assert latency_max >= latency_min >= 0.0
            assert latency_stddev is not None and latency_stddev >= 0.0
        else:
            assert latency_avg is None
            assert latency_p95 is None
            assert latency_p99 is None
            assert latency_count == 0
            assert latency_sum == pytest.approx(0.0)
            assert latency_p50 is None
            assert latency_min is None
            assert latency_max is None
            assert latency_stddev is None
        assert metrics.get("errors_total") == 0
        orders_total = metrics.get("orders_total")
        orders_success = metrics.get("orders_success_total")
        orders_failed = metrics.get("orders_failed_total")
        orders_routed = metrics.get("orders_routed_total")
        fallback_total = metrics.get("orders_fallback_total")
        attempts_total = metrics.get("orders_attempts_total")
        attempts_success = metrics.get("orders_attempts_success")
        attempts_error = metrics.get("orders_attempts_error")
        attempts_api_error = metrics.get("orders_attempts_api_error")
        attempts_auth_error = metrics.get("orders_attempts_auth_error")
        attempts_exception = metrics.get("orders_attempts_exception")
        attempts_success_rate = metrics.get("orders_attempts_success_rate")
        attempts_error_rate = metrics.get("orders_attempts_error_rate")
        attempts_api_error_rate = metrics.get("orders_attempts_api_error_rate")
        attempts_auth_error_rate = metrics.get("orders_attempts_auth_error_rate")
        attempts_exception_rate = metrics.get("orders_attempts_exception_rate")
        assert orders_total is not None and orders_total >= 0
        assert orders_success is not None and orders_success >= 0
        assert orders_failed is not None and orders_failed >= 0
        assert orders_routed is not None and orders_routed >= 0
        assert fallback_total is not None and fallback_total >= 0
        assert attempts_total is not None and attempts_total >= 0
        assert attempts_success is not None and attempts_success >= 0
        assert attempts_error is not None and attempts_error >= 0
        assert attempts_api_error is not None and attempts_api_error >= 0
        assert attempts_auth_error is not None and attempts_auth_error >= 0
        assert attempts_exception is not None and attempts_exception >= 0
        if fill_count:
            assert orders_success >= 1
            assert orders_failed == 0
            assert orders_total == orders_success
            assert orders_routed >= orders_success
            assert fallback_total == 0 or fallback_total <= orders_success
            assert attempts_total >= attempts_success >= 1
            assert attempts_error == 0
            assert attempts_api_error == 0
            assert attempts_auth_error == 0
            assert attempts_exception == 0
            assert attempts_success_rate is not None and attempts_success_rate == pytest.approx(1.0)
            assert attempts_error_rate is not None and attempts_error_rate == pytest.approx(0.0)
            assert attempts_api_error_rate is not None and attempts_api_error_rate == pytest.approx(0.0)
            assert attempts_auth_error_rate is not None and attempts_auth_error_rate == pytest.approx(0.0)
            assert attempts_exception_rate is not None and attempts_exception_rate == pytest.approx(0.0)
            success_rate = metrics.get("orders_success_rate")
            failure_rate = metrics.get("orders_failure_rate")
            fallback_rate = metrics.get("orders_fallback_rate")
            assert success_rate is not None and success_rate == pytest.approx(1.0)
            assert failure_rate is not None and failure_rate == pytest.approx(0.0)
            assert fallback_rate is not None and fallback_rate >= 0.0
        else:
            assert orders_success == 0
            assert orders_failed == 0
            assert orders_total == 0
            assert fallback_total == 0
            assert attempts_total == 0
            assert attempts_success == 0
            assert attempts_error == 0
            assert attempts_api_error == 0
            assert attempts_auth_error == 0
            assert attempts_exception == 0
            assert attempts_success_rate is None
            assert attempts_error_rate is None
            assert attempts_api_error_rate is None
            assert attempts_auth_error_rate is None
            assert attempts_exception_rate is None
            assert metrics.get("orders_success_rate") is None
            assert metrics.get("orders_failure_rate") is None
            assert metrics.get("orders_fallback_rate") is None

    guardrails = live_report.get("guardrails")
    assert guardrails, f"Brak sekcji guardrails w raporcie: {live_report}"
    guardrail_path = Path(guardrails["report_markdown"]).expanduser()
    assert guardrail_path.exists(), f"Plik guardrails nie istnieje: {guardrail_path}"
    assert guardrail_path.is_relative_to(tmp_path), guardrail_path
    assert guardrail_path.is_relative_to(live_reports_root), guardrail_path

    report_markdown_path = Path(live_report["report_markdown"]).expanduser()
    assert report_markdown_path.exists()
    assert report_markdown_path.is_relative_to(tmp_path)
    assert report_markdown_path.is_relative_to(live_markdown_root)

    report_json_path = Path(live_report["report_json"]).expanduser()
    assert report_json_path.exists()
    assert report_json_path.is_relative_to(tmp_path)
    assert report_json_path.is_relative_to(live_reports_root)

    # podczas obu uruchomień adapter powinien odpytać API o konto i dane
    requested_paths = {entry["path"] for entry in loopback_exchange_server.requests}
    assert "/account" in requested_paths
    assert any(path in requested_paths for path in ("/ohlcv", "/symbols"))
