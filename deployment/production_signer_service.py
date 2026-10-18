"""Diagnostic-only entry point for the two installed production signer roles."""

from __future__ import annotations

import argparse
import json
import signal
import threading

from bot_core.production_root_lifecycle import (
    ProductionRootLifecycleError,
    open_installed_production_signer_policy,
)
from bot_core.production_signer_custody import (
    ProductionSignerCustodyError,
    qualify_installed_signer_custody,
)
from bot_core.production_signer_ipc import (
    ProductionSignerIPCError,
    run_diagnostic_server,
)


def _public_custody(role: str, operation: str) -> dict[str, object]:
    if operation not in {"DIAGNOSTIC", "QUALIFY"}:
        raise ProductionSignerCustodyError("DIAGNOSTIC_OPERATION_REQUIRED")
    observation = qualify_installed_signer_custody(role)
    return {"facts": observation.public_facts(), "blockers": list(observation.blockers)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("root_signer", "history_signer"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--serve", action="store_true")
    args = parser.parse_args(argv)
    try:
        policy = open_installed_production_signer_policy(args.role)
        if args.check:
            result = _public_custody(args.role, "QUALIFY")
            print(json.dumps({"status": "BLOCKED", "role": args.role, **result}, sort_keys=True))
            return 2
        stop_event = threading.Event()
        previous = {
            selected: signal.signal(selected, lambda _number, _frame: stop_event.set())
            for selected in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            run_diagnostic_server(
                policy,
                lambda operation: _public_custody(args.role, operation),
                stop_event=stop_event,
            )
        finally:
            for selected, handler in previous.items():
                signal.signal(selected, handler)
        return 0
    except (
        ProductionRootLifecycleError,
        ProductionSignerCustodyError,
        ProductionSignerIPCError,
        OSError,
    ) as exc:
        # Native backend errors are not trusted public diagnostic content.
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "role": args.role,
                    "reason": "SIGNER_QUALIFICATION_UNAVAILABLE",
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
