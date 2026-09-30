"""Neutral, immutable contract for canonical Stage-9 pre-ceremony evidence."""

from __future__ import annotations

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
    }
)

CLEAN_INSTALL_PRECEREMONY_PROOFS = (
    "files",
    "services",
    "dacl",
    "postgresql",
    "authority_absent",
    "production_enrollment_fail_closed",
    "uninstall",
    "acceptance_cleanup",
)

POST_ENROLLMENT_QUALIFICATION_STATE = "REQUIRED"
