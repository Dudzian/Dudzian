"""Neutral, immutable contract for canonical Stage-9 pre-ceremony evidence."""

from __future__ import annotations

PRODUCTION_TRUST_CEREMONY_ID = "390299aaa1ea928a6c2bfdd81a4c50cfde82a7744cd054e8628d5937b90f1699"

CLEAN_INSTALL_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "source_revision",
        "ci_provider",
        "ci_run_id",
        "runner_os",
        "runner_arch",
        "probe_id",
        "msi_sha256",
        "manifest_sha256",
        "product_version",
        "install_exit_code",
        "proofs",
        "uninstall_exit_code",
        "post_enrollment_live_qualification",
        "production_trust_ceremony_id",
        "production_trust_package_manifest_sha256",
        "production_trust_artifact_run_id",
    }
)

CLEAN_INSTALL_PRECEREMONY_PROOFS = (
    "files",
    "services",
    "dacl",
    "postgresql",
    "authority_absent",
    "production_enrollment_fail_closed",
    "frozen_production_trust",
    "uninstall",
    "acceptance_cleanup",
)

POST_ENROLLMENT_QUALIFICATION_STATE = "REQUIRED"
