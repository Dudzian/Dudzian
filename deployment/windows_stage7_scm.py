"""Small native SCM boundary used only by the reviewed Stage-7 acceptance probe."""

from __future__ import annotations

import argparse
import json
import os

STAGE7_ACCEPTANCE_SHUTDOWN_CONTROL = 200


def _modules():
    if os.name != "nt":
        raise RuntimeError("live SCM qualification is Windows-only")
    import win32service  # type: ignore[import-not-found]

    return win32service


def query_accepted_controls(service_name: str) -> int:
    win32service = _modules()
    manager = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
    service = None
    try:
        service = win32service.OpenService(manager, service_name, win32service.SERVICE_QUERY_STATUS)
        status = win32service.QueryServiceStatusEx(service)
        return int(status["ControlsAccepted"])
    finally:
        if service is not None:
            win32service.CloseServiceHandle(service)
        win32service.CloseServiceHandle(manager)


def send_acceptance_shutdown(service_name: str) -> None:
    win32service = _modules()
    manager = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
    service = None
    try:
        service = win32service.OpenService(
            manager, service_name, win32service.SERVICE_USER_DEFINED_CONTROL
        )
        win32service.ControlService(service, STAGE7_ACCEPTANCE_SHUTDOWN_CONTROL)
    finally:
        if service is not None:
            win32service.CloseServiceHandle(service)
        win32service.CloseServiceHandle(manager)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("query", "shutdown"))
    parser.add_argument("--service", required=True)
    args = parser.parse_args()
    if args.command == "query":
        accepted = query_accepted_controls(args.service)
        win32service = _modules()
        print(
            json.dumps(
                {
                    "controls_accepted": accepted,
                    "accepts_shutdown": bool(accepted & win32service.SERVICE_ACCEPT_SHUTDOWN),
                }
            )
        )
    else:
        send_acceptance_shutdown(args.service)
        print(json.dumps({"control": STAGE7_ACCEPTANCE_SHUTDOWN_CONTROL, "sent": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
