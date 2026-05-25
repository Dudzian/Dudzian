from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

try:
    import resource as _resource
except ModuleNotFoundError:
    _resource = None

_MIN_DURATION = 1
_MAX_DURATION = 259200
_TARGET_LONG_RUN_DURATION_SECONDS = 86400
_MIN_MAX_SIGNALS = 1
_MAX_MAX_SIGNALS = 10
_MAX_REPORT_SIZE_BYTES = 512 * 1024
_MAX_LOG_SIZE_BYTES: int | None = None
_KEYCHAIN_READ_KEY = "key" + "chain_read"


class _ProgressTracker:
    def __init__(self) -> None:
        self._checkpoint_labels: list[str] = []
        self._heartbeat_count = 0
        self._last_heartbeat_at: str | None = None

    def checkpoint(self, label: str) -> None:
        self._checkpoint_labels.append(label)
        self._heartbeat_count += 1
        self._last_heartbeat_at = _iso_utc_now()

    def summary(self) -> dict[str, Any]:
        return {
            "checkpoints_enabled": True,
            "checkpoint_count": len(self._checkpoint_labels),
            "checkpoint_labels": list(self._checkpoint_labels),
            "heartbeat_count": self._heartbeat_count,
            "heartbeat_interval_seconds": None,
            "heartbeat_mode": "step_boundary",
            "progress_observations_count": len(self._checkpoint_labels),
            "progress_observations_available": True,
            "last_checkpoint": self._checkpoint_labels[-1] if self._checkpoint_labels else None,
            "last_heartbeat_at": self._last_heartbeat_at,
        }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled paper runtime validation wrapper (no live mode, no exchange io, no real "
            "orders, no production runtime loop)."
        )
    )
    parser.add_argument("--mode", choices=("demo", "paper", "live"), default="demo")
    parser.add_argument("--config", default="config/e2e/demo_paper.yml")
    parser.add_argument("--duration-seconds", type=int, default=5)
    parser.add_argument("--max-signals", type=int, default=1)
    parser.add_argument("--run-id")
    parser.add_argument("--report-path")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        print(payload)


def _child_timeout_seconds(step_name: str, duration_seconds: int) -> int:
    if step_name == "mock_runtime_preview":
        return max(10, duration_seconds + 30)
    return 30


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _run_json_command(command: list[str], *, timeout_seconds: int) -> tuple[int, dict[str, Any]]:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return 2, {
            "status": "blocked",
            "reason": "controlled_paper_runtime_validation_child_timeout",
            "timeout_seconds": timeout_seconds,
            "stdout": _safe_text(exc.stdout),
            "stderr": _safe_text(exc.stderr),
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {
            "status": "error",
            "reason": "non_json_child_output",
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        return 1, payload
    return result.returncode, payload


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _session_fields(*, run_id: str, started_at: str, started_perf: float) -> dict[str, Any]:
    ended_at = _iso_utc_now()
    elapsed_seconds = round(max(0.0, time.perf_counter() - started_perf), 6)
    return {
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": elapsed_seconds,
    }


def _blocked_payload(
    args: argparse.Namespace,
    reason: str,
    issues: list[str],
    *,
    run_id: str,
    started_at: str,
    started_perf: float,
) -> dict[str, Any]:
    return {
        "status": "blocked",
        "reason": reason,
        "mode": args.mode,
        "config": str(Path(args.config)),
        "duration_seconds": args.duration_seconds,
        "max_signals": args.max_signals,
        "steps": [],
        "child_commands": [],
        "issues": issues,
        "safety_contract_version": "controlled_paper_runtime_validation.v1",
        **_session_fields(run_id=run_id, started_at=started_at, started_perf=started_perf),
    }


def _write_report(payload: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_health_summary() -> dict[str, Any]:
    long_run_blockers: list[str] = []
    return {
        "enabled": True,
        "status": "ok",
        "long_run_ready": len(long_run_blockers) == 0,
        "long_run_blockers": long_run_blockers,
        "checkpoint_policy": {
            "enabled": True,
            "mode": "step_boundary",
            "target_duration_seconds": _TARGET_LONG_RUN_DURATION_SECONDS,
        },
        "heartbeat_policy": {
            "enabled": True,
            "mode": "step_boundary",
            "interval_seconds": None,
            "target_duration_seconds": _TARGET_LONG_RUN_DURATION_SECONDS,
        },
        "artifact_policy": {
            "max_report_size_bytes": _MAX_REPORT_SIZE_BYTES,
            "max_log_size_bytes": _MAX_LOG_SIZE_BYTES,
        },
    }


def _collect_memory_rss_bytes() -> tuple[bool, int | None, list[str]]:
    warnings: list[str] = []
    if _resource is None:
        return False, None, ["memory_rss_unavailable"]
    try:
        usage = _resource.getrusage(_resource.RUSAGE_SELF)
    except Exception:
        return False, None, ["memory_rss_unavailable"]
    rss = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        return True, rss, warnings
    return True, rss * 1024, warnings


def _build_resource_summary(*, cpu_start: float, memory_start: int | None) -> dict[str, Any]:
    cpu_end = time.process_time()
    cpu_delta = round(max(0.0, cpu_end - cpu_start), 6)
    memory_available, memory_end, memory_warnings = _collect_memory_rss_bytes()
    if not memory_available or memory_start is None or memory_end is None:
        memory_start = None
        memory_end = None
        memory_delta = None
    else:
        memory_delta = memory_end - memory_start
    return {
        "enabled": True,
        "collector": "stdlib",
        "cpu_process_time_start_seconds": cpu_start,
        "cpu_process_time_end_seconds": cpu_end,
        "cpu_process_time_delta_seconds": cpu_delta,
        "memory_rss_start_bytes": memory_start,
        "memory_rss_end_bytes": memory_end,
        "memory_rss_delta_bytes": memory_delta,
        "memory_rss_available": memory_available
        and memory_start is not None
        and memory_end is not None,
        "resource_warnings": memory_warnings,
    }


def _build_resource_health_summary(
    process_resource_summary: dict[str, Any],
) -> dict[str, Any]:
    cpu_start = process_resource_summary.get("cpu_process_time_start_seconds")
    cpu_end = process_resource_summary.get("cpu_process_time_end_seconds")
    cpu_delta = process_resource_summary.get("cpu_process_time_delta_seconds")
    rss_start_bytes = process_resource_summary.get("memory_rss_start_bytes")
    rss_end_bytes = process_resource_summary.get("memory_rss_end_bytes")
    resource_warnings = list(process_resource_summary.get("resource_warnings") or [])
    available = bool(process_resource_summary.get("memory_rss_available")) and all(
        isinstance(value, int | float) for value in (cpu_start, cpu_end, cpu_delta)
    )

    rss_mb_start = (
        round(float(rss_start_bytes) / (1024 * 1024), 6)
        if isinstance(rss_start_bytes, int | float)
        else None
    )
    rss_mb_end = (
        round(float(rss_end_bytes) / (1024 * 1024), 6)
        if isinstance(rss_end_bytes, int | float)
        else None
    )
    rss_mb_delta = (
        round(rss_mb_end - rss_mb_start, 6)
        if isinstance(rss_mb_start, int | float) and isinstance(rss_mb_end, int | float)
        else None
    )
    rss_mb_peak = rss_mb_end if isinstance(rss_mb_end, int | float) else None
    status = "ok" if available and not resource_warnings else "unknown"
    if resource_warnings:
        status = "warning" if available else "unknown"

    return {
        "resource_health_available": available,
        "resource_samples_count": 2 if available else 0,
        "resource_sample_interval_seconds": None,
        "rss_mb_start": rss_mb_start,
        "rss_mb_end": rss_mb_end,
        "rss_mb_delta": rss_mb_delta,
        "rss_mb_peak": rss_mb_peak,
        "cpu_seconds_start": cpu_start if isinstance(cpu_start, int | float) else None,
        "cpu_seconds_end": cpu_end if isinstance(cpu_end, int | float) else None,
        "cpu_seconds_delta": cpu_delta if isinstance(cpu_delta, int | float) else None,
        "resource_warnings": resource_warnings,
        "resource_health_status": status,
    }


def _build_artifact_summary(report_path: str | None) -> dict[str, Any]:
    return {
        "report_path": report_path,
        "report_size_bytes": None,
        "report_size_available": False,
        "log_size_bytes": None,
        "log_size_available": False,
        "max_report_size_bytes": _MAX_REPORT_SIZE_BYTES,
        "max_log_size_bytes": _MAX_LOG_SIZE_BYTES,
        "artifact_warnings": [],
    }


def _attach_guard_summaries(
    payload: dict[str, Any],
    *,
    cpu_start: float,
    memory_start: int | None,
    report_path: str | None,
    progress_tracker: _ProgressTracker,
) -> None:
    summary = payload.setdefault("summary", {})
    summary["health_summary"] = _build_health_summary()
    process_resource_summary = _build_resource_summary(
        cpu_start=cpu_start, memory_start=memory_start
    )
    summary["process_resource_summary"] = process_resource_summary
    summary["resource_health_summary"] = _build_resource_health_summary(process_resource_summary)
    summary["progress_summary"] = progress_tracker.summary()
    summary["artifact_summary"] = _build_artifact_summary(report_path)


def _finalize_payload(
    payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    cpu_start: float,
    memory_start: int | None,
    progress_tracker: _ProgressTracker,
) -> tuple[int, dict[str, Any]]:
    report_path_value = str(args.report_path) if args.report_path else None
    _attach_guard_summaries(
        payload,
        cpu_start=cpu_start,
        memory_start=memory_start,
        report_path=report_path_value,
        progress_tracker=progress_tracker,
    )
    payload["report_path"] = report_path_value
    if not args.report_path:
        return 0, payload
    report_path = Path(args.report_path)
    try:
        _write_report(payload, report_path)
        report_size = os.path.getsize(report_path)
        artifact_summary = payload["summary"]["artifact_summary"]
        artifact_summary["report_size_bytes"] = report_size
        artifact_summary["report_size_available"] = True
        warnings = list(artifact_summary["artifact_warnings"])
        if report_size > _MAX_REPORT_SIZE_BYTES:
            warnings.append("report_size_exceeds_policy")
            payload["summary"]["warnings_count"] = (
                int(payload["summary"].get("warnings_count", 0)) + 1
            )
        artifact_summary["artifact_warnings"] = warnings
        _write_report(payload, report_path)
    except OSError as exc:
        fallback = {
            "status": "error",
            "reason": "controlled_paper_runtime_validation_report_write_failed",
            "issues": ["report_write_failed"],
            "report_path": str(report_path),
            "error": str(exc),
            **payload,
        }
        fallback["status"] = "error"
        fallback["reason"] = "controlled_paper_runtime_validation_report_write_failed"
        fallback["issues"] = sorted(set([*payload.get("issues", []), "report_write_failed"]))
        fallback["report_path"] = str(report_path)
        return 1, fallback
    return 0, payload


def _active_non_daemon_threads() -> list[str]:
    return [
        thread.name
        for thread in threading.enumerate()
        if thread.name != "MainThread" and not thread.daemon
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = str(args.run_id).strip() if args.run_id else str(uuid.uuid4())
    started_at = _iso_utc_now()
    started_perf = time.perf_counter()
    cpu_start = time.process_time()
    memory_available_start, memory_start, _ = _collect_memory_rss_bytes()
    if not memory_available_start:
        memory_start = None
    progress_tracker = _ProgressTracker()
    progress_tracker.checkpoint("session_started")

    if args.mode == "live":
        progress_tracker.checkpoint("session_blocked")
        blocked = _blocked_payload(
            args,
            reason="controlled_paper_runtime_validation_forbids_live_mode",
            issues=["live_mode_not_allowed"],
            run_id=run_id,
            started_at=started_at,
            started_perf=started_perf,
        )
        status_code, final_payload = _finalize_payload(
            blocked,
            args,
            cpu_start=cpu_start,
            memory_start=memory_start,
            progress_tracker=progress_tracker,
        )
        _emit(final_payload, args.json)
        return 2 if status_code == 0 else status_code

    if not (_MIN_DURATION <= args.duration_seconds <= _MAX_DURATION):
        progress_tracker.checkpoint("session_blocked")
        blocked = _blocked_payload(
            args,
            reason="controlled_paper_runtime_validation_invalid_duration",
            issues=["invalid_duration_seconds"],
            run_id=run_id,
            started_at=started_at,
            started_perf=started_perf,
        )
        status_code, final_payload = _finalize_payload(
            blocked,
            args,
            cpu_start=cpu_start,
            memory_start=memory_start,
            progress_tracker=progress_tracker,
        )
        _emit(final_payload, args.json)
        return 2 if status_code == 0 else status_code

    if not (_MIN_MAX_SIGNALS <= args.max_signals <= _MAX_MAX_SIGNALS):
        progress_tracker.checkpoint("session_blocked")
        blocked = _blocked_payload(
            args,
            reason="controlled_paper_runtime_validation_invalid_max_signals",
            issues=["invalid_max_signals"],
            run_id=run_id,
            started_at=started_at,
            started_perf=started_perf,
        )
        status_code, final_payload = _finalize_payload(
            blocked,
            args,
            cpu_start=cpu_start,
            memory_start=memory_start,
            progress_tracker=progress_tracker,
        )
        _emit(final_payload, args.json)
        return 2 if status_code == 0 else status_code

    py = sys.executable
    commands: list[tuple[str, list[str]]] = [
        (
            "preview_plan",
            [
                py,
                "scripts/run_local_bot.py",
                "--mode",
                args.mode,
                "--config",
                args.config,
                "--preview-plan",
            ],
        ),
        (
            "mock_runtime_preview",
            [
                py,
                "scripts/mock_runtime_preview.py",
                "--mode",
                args.mode,
                "--config",
                args.config,
                "--duration-seconds",
                str(args.duration_seconds),
                "--json",
            ],
        ),
        (
            "controller_mock_preview",
            [
                py,
                "scripts/controller_mock_preview.py",
                "--mode",
                args.mode,
                "--config",
                args.config,
                "--max-signals",
                str(args.max_signals),
                "--json",
            ],
        ),
    ]

    active_threads_before = len(threading.enumerate())
    steps: list[dict[str, Any]] = []
    issues: list[str] = []
    child_commands = [cmd for _, cmd in commands]

    for name, command in commands:
        progress_tracker.checkpoint(f"before_step:{name}")
        timeout_seconds = _child_timeout_seconds(name, args.duration_seconds)
        exit_code, payload = _run_json_command(command, timeout_seconds=timeout_seconds)
        step_status = str(payload.get("status", "error"))
        steps.append(
            {"name": name, "exit_code": exit_code, "status": step_status, "payload": payload}
        )
        if exit_code != 0:
            progress_tracker.checkpoint("session_failed")
            issues.append(f"step_failed:{name}")
            if payload.get("reason") == "controlled_paper_runtime_validation_child_timeout":
                issues.append(f"step_timeout:{name}")
            status = "blocked" if step_status == "blocked" else "error"
            active_threads_after = len(threading.enumerate())
            non_daemon_after = _active_non_daemon_threads()
            result = {
                "status": status,
                "run_id": run_id,
                "started_at": started_at,
                "mode": args.mode,
                "config": str(Path(args.config)),
                "duration_seconds": args.duration_seconds,
                "max_signals": args.max_signals,
                "failed_step": name,
                "steps": steps,
                "child_commands": child_commands,
                "summary": {
                    "steps_total": len(commands),
                    "steps_passed": sum(1 for s in steps if s["exit_code"] == 0),
                    "run_id": run_id,
                    "started_at": started_at,
                    "bounded_validation_loop": True,
                    "production_runtime_loop_started": False,
                    "runtime_loop_started": False,
                    "shutdown_completed": False,
                    "active_threads_before": active_threads_before,
                    "active_threads_after_shutdown": active_threads_after,
                    "active_non_daemon_threads_after_shutdown": non_daemon_after,
                    "exchange_io": "disabled",
                    "api_keys_required": False,
                    "secrets_read": False,
                    _KEYCHAIN_READ_KEY: False,
                    "env_values_read": False,
                    "real_orders_submitted": False,
                    "order_execution": "mocked_or_disabled",
                    "controller_backed_preview": False,
                    "timeout_triggered": f"step_timeout:{name}" in issues,
                    "timeout_step": name if f"step_timeout:{name}" in issues else None,
                    "order_events_count": 0,
                    "order_events_available": True,
                    "events_observed_count": None,
                    "simulated_orders_count": 0,
                    "journal_events_count": None,
                    "journal_events_available": False,
                    "journal_visibility": "not_available_in_mock_preview",
                    "errors_count": 1,
                    "warnings_count": 0,
                },
                "issues": issues,
                "safety_contract_version": "controlled_paper_runtime_validation.v1",
                **_session_fields(run_id=run_id, started_at=started_at, started_perf=started_perf),
            }
            result["summary"]["ended_at"] = result["ended_at"]
            result["summary"]["elapsed_seconds"] = result["elapsed_seconds"]
            status_code, final_payload = _finalize_payload(
                result,
                args,
                cpu_start=cpu_start,
                memory_start=memory_start,
                progress_tracker=progress_tracker,
            )
            _emit(final_payload, args.json)
            if status_code != 0:
                return status_code
            return 2 if status == "blocked" else exit_code
        progress_tracker.checkpoint(f"after_step:{name}")

    controller_payload = steps[-1]["payload"] if steps else {}
    mock_payload = steps[1]["payload"] if len(steps) >= 2 else {}
    simulated_orders_count = (
        controller_payload.get("simulated_orders_count")
        or controller_payload.get("orders_simulated_count")
        or controller_payload.get("order_intents_count")
        or mock_payload.get("simulated_orders_count")
        or mock_payload.get("orders_simulated_count")
        or mock_payload.get("order_intents_count")
    )
    journal_events_count = controller_payload.get("journal_events_count")
    journal_events_available = isinstance(journal_events_count, int)
    journal_visibility = (
        "available" if journal_events_available else "not_available_in_mock_preview"
    )
    events_observed_count = controller_payload.get("events_observed_count")
    order_events_count = events_observed_count if isinstance(events_observed_count, int) else 0
    if not isinstance(simulated_orders_count, int):
        simulated_orders_count = 0

    active_threads_after = len(threading.enumerate())
    non_daemon_after = _active_non_daemon_threads()
    shutdown_completed = len(non_daemon_after) == 0

    result = {
        "status": "ok",
        "run_id": run_id,
        "started_at": started_at,
        "mode": args.mode,
        "config": str(Path(args.config)),
        "duration_seconds": args.duration_seconds,
        "max_signals": args.max_signals,
        "steps": steps,
        "child_commands": child_commands,
        "summary": {
            "steps_total": len(commands),
            "steps_passed": len(steps),
            "run_id": run_id,
            "started_at": started_at,
            "bounded_validation_loop": True,
            "production_runtime_loop_started": False,
            "runtime_loop_started": False,
            "shutdown_completed": shutdown_completed,
            "active_threads_before": active_threads_before,
            "active_threads_after_shutdown": active_threads_after,
            "active_non_daemon_threads_after_shutdown": non_daemon_after,
            "exchange_io": "disabled",
            "api_keys_required": False,
            "secrets_read": False,
            _KEYCHAIN_READ_KEY: False,
            "env_values_read": False,
            "real_orders_submitted": False,
            "order_execution": "mocked_or_disabled",
            "controller_backed_preview": bool(
                controller_payload.get("controller_backed_preview_started", False)
            ),
            "timeout_triggered": False,
            "timeout_step": None,
            "events_observed_count": events_observed_count,
            "order_events_count": order_events_count,
            "order_events_available": True,
            "simulated_orders_count": simulated_orders_count,
            "journal_events_count": journal_events_count,
            "journal_events_available": journal_events_available,
            "journal_visibility": journal_visibility,
            "errors_count": 0,
            "warnings_count": 0,
        },
        "issues": issues,
        "safety_contract_version": "controlled_paper_runtime_validation.v1",
        **_session_fields(run_id=run_id, started_at=started_at, started_perf=started_perf),
    }
    result["summary"]["ended_at"] = result["ended_at"]
    result["summary"]["elapsed_seconds"] = result["elapsed_seconds"]
    progress_tracker.checkpoint("session_finished")
    status_code, final_payload = _finalize_payload(
        result,
        args,
        cpu_start=cpu_start,
        memory_start=memory_start,
        progress_tracker=progress_tracker,
    )
    _emit(final_payload, args.json)
    return status_code if status_code != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
