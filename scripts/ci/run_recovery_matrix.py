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
        "restart or after_restart or restore or replay or duplicate",
        "-vv",
    ],
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_trading_controller.py",
        "-k",
        "final_label or outcome_label or open_outcome or tracker or no_tracker",
        "-vv",
    ],
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_trading_controller.py",
        "-k",
        "pending_close or duplicate_autonomous_close_replay_suppressed or final_outcome_replay_close_suppressed or close_ranked or no_tracker_close_replay",
        "-vv",
    ],
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_trading_controller.py",
        "-k",
        "risk_rejected or risk or execution or partial or filled",
        "-vv",
    ],
)


def _display_command(cmd: list[str]) -> str:
    return " ".join(cmd)


def main() -> int:
    for cmd in COMMANDS:
        print(f"[recovery-matrix] RUN: {_display_command(cmd)}", flush=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(
                f"[recovery-matrix] FAIL: return code {result.returncode}",
                flush=True,
            )
            return result.returncode
    print("[recovery-matrix] OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
