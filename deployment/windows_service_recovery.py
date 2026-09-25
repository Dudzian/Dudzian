"""Ownership-safe native SCM recovery configuration for Windows acceptance."""

from __future__ import annotations

import argparse
import json
import ntpath
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


def _service_host(command_line: Any) -> str:
    """Extract the sole executable from an SCM command line, failing closed."""
    if not isinstance(command_line, str) or not command_line.strip():
        raise ValueError("BinaryPathName is empty")
    value = command_line.strip()
    if value.startswith('"'):
        closing_quote = value.find('"', 1)
        if closing_quote < 0 or value[closing_quote + 1:].strip():
            raise ValueError("BinaryPathName is ambiguous or has arguments")
        executable = value[1:closing_quote]
    else:
        # An unquoted command is safe only when the complete value is one path.
        if any(character.isspace() for character in value):
            raise ValueError("unquoted BinaryPathName is ambiguous or has arguments")
        executable = value
    if not executable:
        raise ValueError("BinaryPathName has no executable")
    return executable


def _windows_path_identity(path: str, win32api: Any) -> str:
    return ntpath.normcase(win32api.GetFullPathName(path))


def _qualification_diagnostics(
    record: dict[str, Any], config: Any, *, observed_host: str,
    observed_sid: str,
) -> str:
    return "\n".join((
        f'EXPECTED_SERVICE_PATH_NAME={record["service_path_name"]}',
        f"OBSERVED_SCM_BINARY_PATH_NAME={config[3]}",
        f'EXPECTED_SERVICE_HOST={record["service_host"]}',
        f"OBSERVED_SCM_SERVICE_HOST={observed_host}",
        f'EXPECTED_SERVICE_IDENTITY={record["service_identity"]}',
        f"OBSERVED_SCM_SERVICE_START_NAME={config[7]}",
        f'EXPECTED_SERVICE_SID={record["service_sid"]}',
        f"OBSERVED_SERVICE_SID={observed_sid}",
        f"AUTOSTART_START_TYPE={config[1]}",
    ))


def configure_recovery(
    record: dict[str, Any], *, run_token: str, win32service: Any,
    win32security: Any, win32api: Any, mutate: bool = True,
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
        observed_host = "<unresolved>"
        observed_sid = "<unresolved>"
        diagnostics = lambda: _qualification_diagnostics(  # noqa: E731
            record, config, observed_host=observed_host, observed_sid=observed_sid,
        )
        if config[1] != win32service.SERVICE_AUTO_START:
            raise RecoveryQualificationError(
                f"SCM service is not SERVICE_AUTO_START\n{diagnostics()}"
            )
        try:
            observed_host = _service_host(config[3])
            host_matches = _windows_path_identity(
                observed_host, win32api,
            ) == _windows_path_identity(record["service_host"], win32api)
        except (OSError, TypeError, ValueError) as exc:
            raise RecoveryQualificationError(
                f"SCM BinaryPathName qualification failed: {exc}\n{diagnostics()}"
            ) from exc
        if not host_matches:
            raise RecoveryQualificationError(
                f"SCM service host mismatches ownership record\n{diagnostics()}"
            )
        if not isinstance(config[7], str) or not config[7]:
            raise RecoveryQualificationError(
                f"SCM ServiceStartName is empty\n{diagnostics()}"
            )
        try:
            sid, _, _ = win32security.LookupAccountName(None, config[7])
            observed_sid = win32security.ConvertSidToStringSid(sid)
        except (OSError, TypeError) as exc:
            raise RecoveryQualificationError(
                f"SCM ServiceStartName SID resolution failed: {exc}\n{diagnostics()}"
            ) from exc
        if observed_sid != record["service_sid"]:
            raise RecoveryQualificationError(
                f"SCM service SID mismatches ownership record\n{diagnostics()}"
            )
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
    import win32api  # type: ignore[import-not-found]

    try:
        record = json.loads(args.record.read_text(encoding="utf-8"))
        result = configure_recovery(
            record, run_token=args.run_token, win32service=win32service,
            win32security=win32security, win32api=win32api,
            mutate=not args.verify_only,
        )
    except (OSError, json.JSONDecodeError, RecoveryQualificationError) as exc:
        print(str(exc))
        return 1
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
