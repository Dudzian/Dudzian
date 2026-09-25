"""Ownership-safe native SCM recovery configuration for Windows acceptance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from deployment.windows_service_ownership import SERVICE_PROVEN

SERVICE_NAME = "CryptoHunterBackend"
SERVICE_IDENTITY = r"NT SERVICE\CryptoHunterBackend"
RESET_PERIOD_SECONDS = 86_400
RESTART_DELAYS_MS = (1_000, 5_000, 30_000)


class RecoveryQualificationError(RuntimeError):
    """Recovery may not be changed because current-run ownership was not proven."""


def expected_actions(win32service: Any) -> list[tuple[int, int]]:
    return [
        *([(win32service.SC_ACTION_RESTART, delay) for delay in RESTART_DELAYS_MS]),
        (win32service.SC_ACTION_NONE, 0),
    ]


def validate_record(record: dict[str, Any], *, run_token: str) -> None:
    required = {
        "run_token": run_token,
        "ownership_phase": SERVICE_PROVEN,
        "strict_create_result": "CREATED",
        "service_name": SERVICE_NAME,
        "service_identity": SERVICE_IDENTITY,
    }
    if any(record.get(key) != value for key, value in required.items()):
        raise RecoveryQualificationError("recovery requires exact SERVICE_PROVEN ownership")
    for key in ("service_path_name", "service_host", "service_sid"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise RecoveryQualificationError(f"ownership record lacks {key}")


def _normalize_actions(value: Any) -> list[tuple[int, int]]:
    return [(int(action), int(delay)) for action, delay in value]


def configure_recovery(
    record: dict[str, Any], *, run_token: str, win32service: Any,
    win32security: Any | None = None, mutate: bool = True,
) -> dict[str, Any]:
    """Reobserve exact SCM identity, mutate once, then query and verify the policy."""
    validate_record(record, run_token=run_token)
    manager = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
    service = None
    try:
        desired_access = win32service.SERVICE_QUERY_CONFIG
        if mutate:
            desired_access |= (
                win32service.SERVICE_CHANGE_CONFIG | win32service.SERVICE_START
            )
        service = win32service.OpenService(
            manager, SERVICE_NAME, desired_access,
        )
        config = win32service.QueryServiceConfig(service)
        if config[1] != win32service.SERVICE_AUTO_START:
            raise RecoveryQualificationError("SCM service is not SERVICE_AUTO_START")
        if config[3] != record["service_path_name"] or config[7] != record["service_identity"]:
            raise RecoveryQualificationError("SCM configuration mismatches ownership record")
        if not config[3].startswith(f'"{record["service_host"]}"'):
            raise RecoveryQualificationError("SCM service host mismatches ownership record")
        if win32security is not None:
            sid, _, _ = win32security.LookupAccountName(None, record["service_identity"])
            if win32security.ConvertSidToStringSid(sid) != record["service_sid"]:
                raise RecoveryQualificationError("SCM service SID mismatches ownership record")
        policy = {
            "ResetPeriod": RESET_PERIOD_SECONDS,
            "RebootMsg": "",
            "Command": "",
            "Actions": expected_actions(win32service),
        }
        if mutate:
            win32service.ChangeServiceConfig2(
                service, win32service.SERVICE_CONFIG_FAILURE_ACTIONS, policy,
            )
        observed = win32service.QueryServiceConfig2(
            service, win32service.SERVICE_CONFIG_FAILURE_ACTIONS,
        )
        if (
            int(observed["ResetPeriod"]) != RESET_PERIOD_SECONDS
            or observed.get("RebootMsg", "") not in ("", None)
            or observed.get("Command", "") not in ("", None)
            or _normalize_actions(observed["Actions"]) != expected_actions(win32service)
        ):
            raise RecoveryQualificationError("SCM returned a different failure-actions policy")
        return {"start_type": int(config[1]), "failure_actions": observed}
    finally:
        if service is not None:
            win32service.CloseServiceHandle(service)
        win32service.CloseServiceHandle(manager)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Configure bounded run-owned SCM recovery")
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if os.name != "nt":
        print("Windows SCM is required")
        return 1
    import win32service  # type: ignore[import-not-found]
    import win32security  # type: ignore[import-not-found]

    try:
        record = json.loads(args.record.read_text(encoding="utf-8"))
        result = configure_recovery(
            record, run_token=args.run_token, win32service=win32service,
            win32security=win32security, mutate=not args.verify_only,
        )
    except (OSError, json.JSONDecodeError, RecoveryQualificationError) as exc:
        print(str(exc))
        return 1
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
