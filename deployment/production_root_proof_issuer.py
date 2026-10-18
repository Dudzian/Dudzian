"""Qualification entry point for the isolated installed issuer process."""

from __future__ import annotations

import argparse

from bot_core.production_root_lifecycle import open_installed_production_root_lifecycle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    open_installed_production_root_lifecycle().qualify_installed_service()
    print("QUALIFIED_LIFECYCLE_ONLY; PRODUCTION_ROOT_PROOF_EMISSION=BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
