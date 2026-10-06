"""Production key qualification, separate from unavailable request authentication.

The frozen contract requires a signed, retained PDSA challenge and independent
TPM creation/custody attestation. Neither may be replaced with local provider
properties or a request signature. See docs/windows_production_pre_enrollment.md.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, NoReturn

from bot_core.licensing.canonical import canonical_json_bytes
from deployment.platforms.windows import production_trust_package_path, resolve_paths
from deployment.windows_stage9_production_trust import (
    CEREMONY_ID,
    load_production_trust,
    require_verified_production_trust_context,
)

AUTHENTICATION_BLOCKERS = (
    "SIGNED_PRODUCTION_PDSA_CHALLENGE_CONTRACT_UNAVAILABLE",
    "PRODUCTION_TPM_CREATION_CUSTODY_ATTESTATION_UNAVAILABLE",
)


class ProductionPreEnrollmentUnavailable(RuntimeError):
    """A frozen production authentication requirement cannot be established."""


def require_production_request_authentication(context: object) -> NoReturn:
    """Reject synthetic trust and stop the conflicting authentication portion.

    Existing PDSAChallengeV1 is unsigned. Existing TPM verifier proof domains
    contain TEST_ONLY. Do not guess a signed challenge envelope, accept caller
    declarations, or promote either existing component by relabeling it.
    """
    require_verified_production_trust_context(context)
    raise ProductionPreEnrollmentUnavailable(";".join(AUTHENTICATION_BLOCKERS))


def qualify_installed_production_key() -> dict[str, Any]:
    """Resolve the fixed production key using installed, verified trust only.

    This persists a machine identity. It does not construct an authenticated
    request, consume a PDSA challenge, enroll a device, or create LPPI authority.
    No caller-selected trust roots, PDSA keys, key name, or backend are accepted.
    """
    from deployment.windows_cng_pre_enrollment import WindowsCNGPreEnrollmentKey

    paths = resolve_paths()
    trust = require_verified_production_trust_context(
        load_production_trust(production_trust_package_path(CEREMONY_ID))
    )
    with WindowsCNGPreEnrollmentKey.open_or_create(paths.state / "PreEnrollment") as key:
        return {
            "schema_version": "WindowsProductionPreEnrollmentQualificationV1",
            "environment": "PRODUCTION",
            "purpose": "LOCAL_PROVIDER_QUALIFICATION_ONLY",
            "release_policy_digest_sha256": trust.release_payload_digest,
            "release_policy_generation": trust.release_version,
            "key": key.public_evidence,
            "request_authentication": "BLOCKED",
            "authentication_blockers": list(AUTHENTICATION_BLOCKERS),
            "legal_enrollment": "NOT_PERFORMED",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qualify-key",
        action="store_true",
        required=True,
        help="create or reuse the persistent production key; emit public qualification only",
    )
    parser.parse_args(argv)
    try:
        result = qualify_installed_production_key()
    except (OSError, RuntimeError, ValueError):
        # Native failures may include machine-specific paths. No exception or
        # secret-bearing raw input is forwarded to operator evidence/stdout.
        print("PRODUCTION_PRE_ENROLLMENT_QUALIFICATION_FAILED", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
