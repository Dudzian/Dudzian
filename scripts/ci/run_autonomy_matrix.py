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
        "direction_mismatch",
        "-vv",
    ],
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_trading_controller.py",
        "-k",
        "timestamp_mismatch_validator_does_not_block_legal_close_or_replay_close or open_validator_does_not_block_legal_autonomous_close_or_close_replay or direction_mismatch",
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
        "missing_direction or unknown_direction or provenance or direction_mismatch or rejected_shadow_candidate",
        "-vv",
    ],
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_trading_controller.py",
        "-k",
        "opportunity_autonomy or accepted_autonomous_handoff or shadow_reference or duplicate_open_guard or handoff",
    ],
)


def _display_command(cmd: list[str]) -> str:
    return " ".join(cmd)


def main() -> int:
    for cmd in COMMANDS:
        print(f"[autonomy-matrix] RUN: {_display_command(cmd)}", flush=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(
                f"[autonomy-matrix] FAIL: return code {result.returncode}",
                flush=True,
            )
            return result.returncode
    print("[autonomy-matrix] OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
