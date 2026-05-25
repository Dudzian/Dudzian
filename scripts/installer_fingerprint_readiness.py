from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from bot_core.security.hwid import HwIdProvider, HwIdProviderError

SAFETY_CONTRACT_VERSION = "installer_fingerprint_readiness.v1"


@dataclass(frozen=True)
class ReadinessMode:
    value: str


VALID_MODES = {"install", "first-run"}


def _fingerprint_preview(fingerprint: str) -> str:
    compact = fingerprint.strip()
    if len(compact) <= 12:
        return "***masked***"
    return f"{compact[:6]}...{compact[-4:]}"


def build_payload(mode: str) -> dict[str, object]:
    issues: list[str] = []
    status = "ok"
    fingerprint_available = False
    preview: str | None = None

    try:
        fingerprint = HwIdProvider().read()
    except HwIdProviderError:
        status = "warning"
        issues.append("fingerprint_source_unavailable")
    else:
        fingerprint_available = True
        preview = _fingerprint_preview(fingerprint)

    readiness = {
        "enabled": True,
        "local_only": True,
        "installer_safe": True,
        "first_run_safe": True,
        "fingerprint_available": fingerprint_available,
        "fingerprint_value_exposed": False,
        "fingerprint_preview": preview,
        "raw_machine_identifiers_exposed": False,
        "license_activation_performed": False,
        "license_required_for_install": False,
        "secrets_read": False,
        "keychain_read": False,
        "env_values_read": False,
        "api_keys_required": False,
        "exchange_io": "disabled",
        "order_submission": "disabled",
        "runtime_loop_started": False,
        "production_runtime_loop_started": False,
    }

    checks = {
        "mode_supported": mode in VALID_MODES,
        "activation_path": "not_performed",
        "fingerprint_source": "available" if fingerprint_available else "unavailable",
        "exchange_io": "disabled",
        "runtime": "not_started",
    }

    return {
        "status": status,
        "mode": mode,
        "installer_fingerprint_readiness": readiness,
        "checks": checks,
        "issues": issues,
        "safety_contract_version": SAFETY_CONTRACT_VERSION,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Installer fingerprint readiness contract")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="first-run")
    parser.add_argument("--json", action="store_true", help="Emit JSON payload")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = build_payload(args.mode)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
