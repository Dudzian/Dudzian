from __future__ import annotations

import subprocess
import sys


COMMANDS: tuple[list[str], ...] = (
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_trading_controller.py",
        "-k",
        "journal or feed or alert or severity or signal_skipped or order_executed or order_partially_executed or opportunity_autonomy_enforcement or performance_guard or risk_rejected or runtime_controls_soft_snapshot or execution_permission",
        "-vv",
        "--maxfail=20",
    ],
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/ui/test_runtime_service_defaults.py",
        "-vv",
    ],
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/ui/test_runtime_service_safety_net.py",
        "-k",
        "feed or transport or grpc or jsonl or demo or fallback or feed_transport or feedTransportSnapshot",
        "-vv",
    ],
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_alerts.py",
        "-k",
        "dedup or severity or alert",
        "-vv",
    ],
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/runtime/test_decision_envelope.py",
        "tests/ui/test_decision_payload_normalizer.py",
        "tests/docs/test_decision_envelope_feasibility_contract.py",
        "-vv",
    ],
)


def _display_command(cmd: list[str]) -> str:
    return " ".join(cmd)


def main() -> int:
    for cmd in COMMANDS:
        print(f"[observability-matrix] RUN: {_display_command(cmd)}", flush=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(
                f"[observability-matrix] FAIL: return code {result.returncode}",
                flush=True,
            )
            return result.returncode
    print("[observability-matrix] OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
