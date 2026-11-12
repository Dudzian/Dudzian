"""Explicit, non-qualifying Stage 9 checks without signing, builds or VM actions.

This separate validation scope does not replace the repository's full required
quality gates. The selected tests use public verification vectors, data-only
fixtures and mocked privileged boundaries. Omitted operations remain NOT_EXECUTED.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

AUDITED_BASE = "fc15b98298c5430286dd2206ec7eeb8f20e2497b"
COVERAGE_THRESHOLD = 70
SAFE_FILES = (
    "tests/deployment/test_freshness_runtime_bootstrap_security.py",
    "tests/deployment/test_stage9_harness_import_security.py",
    "tests/deployment/test_stage9_harness_runtime_guard.py",
    "tests/deployment/test_stage9_loader_execution.py",
    "tests/deployment/test_stage9_protected_tree.py",
    "tests/deployment/test_stage9_runtime_provenance.py",
    "tests/deployment/test_stage9_revision_contract.py",
    "tests/deployment/test_stage9_inspect_contract.py",
    "tests/deployment/test_stage9_vm_native.py",
    "tests/deployment/test_stage9_vm_performance.py",
    "tests/deployment/test_prepare_stage9_vm_bundle.py",
    "tests/scripts/test_stage9_safe_ci.py",
)
# Every selected node is independent of the signed candidate and approval fixtures.
LEGACY_BOOTSTRAP = (
    "test_bootstrap_refuses_unprotected_system_executable",
    "test_bootstrap_cli_rejects_unrecognized_operation",
    "test_bootstrap_cli_refuses_unprotected_core",
    "test_bootstrap_checks_both_helpers_before_executing_code",
    "test_bootstrap_requires_application_verification_before_success",
    "test_bootstrap_refuses_non_linux",
    "test_bootstrap_refuses_missing_memfd",
)
ENVIRONMENT_NODES = (
    "test_approval_path_is_absolute_and_cannot_traverse_parents",
    "test_approval_is_read_through_pinned_nofollow_descriptors",
    "test_reader_acl_contract_covers_directories_not_existing_final_file",
    "test_directory_acl_refuses_before_reading_approval",
    "test_directory_xattr_denial_refuses_before_reading_approval",
    "test_unprotected_ancestor_or_foreign_approval_refuses",
    "test_approval_size_is_bounded",
    "test_missing_approval_and_symlink_os_errors_are_sanitized",
    "test_network_requires_no_active_links_addresses_or_routes",
    "test_active_or_unavailable_network_fails_closed",
    "test_host_share_filesystems_are_detected_without_reading_their_contents",
    "test_unreadable_or_malformed_mount_table_is_unavailable",
    "test_inspect_on_unsupported_platform_never_issues_linux_commands",
    "test_query_failures_cannot_become_positive_evidence",
    "test_query_uses_fixed_sanitized_subprocess_boundary",
    "test_public_reader_only_reads_bounded_regular_file",
    "test_os_release_is_parsed_as_data_without_shell_expansion",
    "test_incomplete_network_inventory_is_not_disconnected",
    "test_inspect_does_not_publish_arbitrary_probe_contents",
    "test_approval_metadata_drift_during_read_refuses",
)
TRIAL_NODES = (
    "test_snapshot_is_readonly_and_detects_content_mode_link_and_absence",
    "test_snapshot_refuses_special_file_without_blocking",
    "test_preflight_detects_new_sibling_installer_staging",
    "test_staging_inventory_detects_changes_without_hashing_unrelated_data",
    "test_staging_inventory_refuses_symlink_instead_of_parent",
    "test_summary_never_promotes_incomplete_or_blocked",
    "test_summary_rejects_unknown_or_empty_statuses",
    "test_new_report_has_no_native_or_security_success",
    "test_inspect_does_not_load_inputs_or_run_product",
    "test_inspect_refuses_execution_arguments",
    "test_guard_runs_before_inputs_or_any_execution_and_sanitizes",
    "test_preflight_compares_host_even_when_installer_refuses",
    "test_preflight_preserves_exact_refusal_without_promoting",
    "test_install_cannot_follow_failed_preflight",
    "test_install_verifies_published_state_and_identical_reinstall",
    "test_identical_reinstall_detects_writes",
    "test_install_stops_without_reinstall_when_postflight_cannot_verify_activation",
    "test_install_rechecks_idempotency_after_independently_activated_profile",
    "test_unrelated_postflight_refusal_remains_failure",
    "test_native_phases_require_verified_installed_release",
    "test_runtime_verify_remains_readonly_and_uses_current_binding",
    "test_command_has_no_shell_ambient_credentials_or_host_error_output",
    "test_command_timeout_has_fixed_error",
    "test_strict_json_refuses_ambiguous_or_non_object",
    "test_inputs_separate_public_authorities_and_refuse_unknown_fields",
    "test_installer_legacy_success_is_not_v2_success",
    "test_preflight_detects_mutation_even_after_timeout",
    "test_verify_detects_mutation_after_invalid_reply",
    "test_runtime_cli_uses_verified_helper_and_literal_inputs",
    "test_installer_records_only_validated_fields_and_exact_refusal",
    "test_inputs_require_external_distinct_authorities",
    "test_protected_inputs_reject_user_writable_and_relative_paths",
    "test_read_size_limit_and_stability",
    "test_installed_interpreter_is_not_returned_before_system_bootstrap",
    "test_installed_interpreter_binds_revision_and_environment",
    "test_native_worker_cannot_run_in_unverified_interpreter",
    "test_native_worker_uses_signed_interpreter_and_preserves_blocked",
)
WORKFLOW_NODES = (
    "tests/scripts/test_linux_physical_evidence_workflow.py::test_physical_jobs_are_separate_from_pr_and_stage10_and_pin_main",
    "tests/scripts/test_linux_physical_evidence_workflow.py::test_independent_verification_and_gate_receive_raw_bundle_and_trusted_context",
    "tests/scripts/test_linux_physical_evidence_workflow.py::test_new_security_modules_remain_in_type_format_test_and_coverage_gates",
    "tests/scripts/test_platform_core_evidence_workflow.py::test_linux_gate_waits_for_and_consumes_all_current_run_core_artifacts",
    "tests/scripts/test_platform_core_evidence_workflow.py::test_stage10_refuses_before_physical_actions_and_keeps_safe_powershell_regression",
)
NOT_EXECUTED = (
    "Tests generating keys or signatures, including signed candidate fixtures",
    "Tests constructing synthetic VM approvals",
    "Runtime builder, wheelhouse download and native runtime probes",
    "Test executing deliberately substituted bytecode",
    "Root installation, services, PostgreSQL fixtures and physical/VM qualification",
    "Full required quality gates outside this separately selected validation scope",
)


def safe_test_nodes() -> tuple[str, ...]:
    return (
        *SAFE_FILES,
        *("tests/deployment/test_freshness_runtime.py::" + name for name in LEGACY_BOOTSTRAP),
        *("tests/deployment/test_stage9_vm_environment.py::" + name for name in ENVIRONMENT_NODES),
        *("tests/deployment/test_stage9_vm_trial.py::" + name for name in TRIAL_NODES),
        *WORKFLOW_NODES,
    )


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "--no-optional-locks", *arguments], cwd=root, text=True
    ).strip()


def validate_context(root: Path, expected: str, base: str, output: Path) -> dict[str, Any]:
    if (
        sys.platform != "linux"
        or os.geteuid() == 0
        or sys.version_info[:2] not in {(3, 11), (3, 12)}
        or re.fullmatch(r"[0-9a-f]{40}", expected) is None
        or re.fullmatch(r"[0-9a-f]{40}", base) is None
        or base != AUDITED_BASE
        or output.is_relative_to(root)
    ):
        raise ValueError("SAFE_VALIDATION_CONTEXT_REFUSED")
    if _git(root, "rev-parse", "HEAD") != expected:
        raise ValueError("EXPECTED_HEAD_MISMATCH")
    if _git(root, "status", "--porcelain"):
        raise ValueError("CLEAN_CHECKOUT_REQUIRED")
    subprocess.run(["git", "merge-base", "--is-ancestor", base, expected], cwd=root, check=True)
    for node in safe_test_nodes():
        if not (root / node.partition("::")[0]).is_file():
            raise ValueError("REQUIRED_TEST_SOURCE_MISSING")
    return {
        "commit": expected,
        "base_commit": base,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": sys.platform,
        "effective_uid": os.geteuid(),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    }


def changed_paths(root: Path, base: str, head: str) -> tuple[str, ...]:
    return tuple(_git(root, "diff", "--name-only", "--diff-filter=ACMR", base, head).splitlines())


def coverage_sources(root: Path, base: str, head: str) -> tuple[str, ...]:
    changed = changed_paths(root, base, head)
    sources = []
    for name in changed:
        if not name.endswith(".py") or name.startswith(("tests/", "ui/tests/")):
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9/]*\.py", name) is None:
            raise ValueError("COVERAGE_SOURCE_PATH_REFUSED")
        sources.append(name.removesuffix(".py").replace("/", "."))
    if not sources:
        raise ValueError("CHANGED_PYTHON_SOURCE_REQUIRED")
    return tuple(sorted(sources))


def build_steps(
    python: str, root: Path, output: Path, base: str, head: str
) -> list[tuple[str, list[str]]]:
    changed = coverage_sources(root, base, head)
    deployment = [
        name.replace(".", "/") + ".py" for name in changed if name.startswith("deployment.")
    ]
    return [
        (
            "ruff",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--no-cache",
                "bot_core",
                "core",
                "scripts",
                "tests",
                "deploy",
                "stage6_samples",
                "ui",
                *deployment,
            ],
        ),
        ("format-preflight", [python, "-m", "pre_commit", "run", "ruff-format", "--all-files"]),
        ("deployment-format", [python, "-m", "ruff", "format", "--check", *deployment]),
        (
            "secret-scan",
            [python, "scripts/quality/betterleaks_hook.py", *changed_paths(root, base, head)],
        ),
        ("mypy-linux", [python, "-m", "mypy", "--config-file=pyproject.toml"]),
        (
            "mypy-windows",
            [python, "-m", "mypy", "--platform", "win32", "--config-file=pyproject.toml"],
        ),
        (
            "regression",
            [
                python,
                "-m",
                "coverage",
                "run",
                "--branch",
                "--rcfile=.coveragerc",
                "--source=" + ",".join(changed),
                "-m",
                "pytest",
                "--noconftest",
                "-q",
                *safe_test_nodes(),
                "--junitxml=" + str(output / "regression.xml"),
            ],
        ),
        (
            "coverage-xml",
            [
                python,
                "-m",
                "coverage",
                "xml",
                "--rcfile=.coveragerc",
                "-o",
                str(output / "coverage.xml"),
            ],
        ),
        (
            "changed-code-coverage",
            [
                python,
                "-m",
                "diff_cover.diff_cover_tool",
                str(output / "coverage.xml"),
                "--compare-branch=" + base,
                "--fail-under=" + str(COVERAGE_THRESHOLD),
            ],
        ),
    ]


def run_step(
    name: str, argv: list[str], root: Path, output: Path, environment: dict[str, str]
) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    try:
        with (output / (name + ".log")).open("w", encoding="utf-8") as stream:
            code = subprocess.run(
                argv,
                cwd=root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=1200,
                check=False,
            ).returncode
        status = "PASS" if code == 0 else "FAIL"
    except (OSError, subprocess.TimeoutExpired) as exc:
        code = None
        status = "NOT_EXECUTED" if isinstance(exc, OSError) else "FAIL"
        (output / (name + ".error")).write_text(type(exc).__name__ + "\n")
    record = {
        "name": name,
        "argv": argv,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": code,
        "status": status,
    }
    (output / (name + ".result.json")).write_text(json.dumps(record, indent=2) + "\n")
    return record


def junit_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "NOT_EXECUTED", "reason": "JUNIT_MISSING"}
    try:
        cases = list(ET.parse(path).iter("testcase"))
    except ET.ParseError:
        return {"status": "FAIL", "reason": "JUNIT_INVALID"}
    skipped = sum(case.find("skipped") is not None for case in cases)
    failures = sum(
        case.find("failure") is not None or case.find("error") is not None for case in cases
    )
    return {
        "status": "PASS" if cases and not skipped and not failures else "FAIL",
        "cases": len(cases),
        "failed": failures,
        "not_executed": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--base-commit", default=AUDITED_BASE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.is_relative_to(root):
        parser.error("Output must be outside the source checkout")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError:
        print("SAFE_VALIDATION_NEW_OUTPUT_REQUIRED", file=sys.stderr)
        return 2
    report: dict[str, Any] = {
        "status": "BLOCKED",
        "checks": [],
        "s3": "BLOCKED",
        "production": "NO_GO",
        "not_executed": [{"scope": scope, "status": "NOT_EXECUTED"} for scope in NOT_EXECUTED],
    }
    try:
        report["context"] = validate_context(root, args.expected_commit, args.base_commit, output)
        environment = dict(
            os.environ,
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            PYTHONDONTWRITEBYTECODE="1",
            COVERAGE_FILE=str(output / ".coverage"),
            PRE_COMMIT_HOME=str(output / "pre-commit-cache"),
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL="/dev/null",
            STAGE9_SAFE_OUTPUT=str(output),
        )
        for name, arguments in build_steps(
            sys.executable, root, output, args.base_commit, args.expected_commit
        ):
            record = run_step(name, arguments, root, output, environment)
            report["checks"].append(record)
            print(json.dumps(record), flush=True)
        report["junit"] = junit_result(output / "regression.xml")
        report["checkout_unchanged"] = _git(
            root, "rev-parse", "HEAD"
        ) == args.expected_commit and not _git(root, "status", "--porcelain")
        report["status"] = (
            "PASS"
            if (
                all(check["status"] == "PASS" for check in report["checks"])
                and report["junit"]["status"] == "PASS"
                and report["checkout_unchanged"]
            )
            else "FAIL"
        )
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        report["reason"] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 2 if report["status"] == "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
