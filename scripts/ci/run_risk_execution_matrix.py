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
        "tests/test_runtime_pipeline.py",
        "-k",
        "risk or performance_guard or ai_failover or failover or without_exchange_credentials",
        "-vv",
    ],
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_trading_controller.py",
        "-k",
        "execution or filled or non_filled or partially or partial or order_executed or order_partially_executed",
        "-vv",
    ],
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_trading_controller.py",
        "-k",
        "duplicate_autonomous_close_replay_suppressed or final_outcome_replay_close_suppressed or close_ranked or no_tracker_close_replay or partial_open_then_close or final_label",
        "-vv",
    ],
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_trading_controller.py",
        "-k",
        "validator or diagnostic_handoff_contract_block or direction_mismatch",
        "-vv",
    ],
)


def _display_command(cmd: list[str]) -> str:
    return " ".join(cmd)


def main() -> int:
    for cmd in COMMANDS:
        print(f"[risk-execution-matrix] RUN: {_display_command(cmd)}", flush=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(
                f"[risk-execution-matrix] FAIL: return code {result.returncode}",
                flush=True,
            )
            return result.returncode
    print("[risk-execution-matrix] OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
