"""Manual, fail-closed installation of the exact PRE-01 qualified MSI.

No saved report is an execution credential. Both entry points repeat PRE-01B.1
under native custody locks. Execution additionally needs an independent ten-minute
document, a native elevated operator token and digest-specific console consent.
Neither entry point enrolls, starts services, reboots or qualifies Stage 10.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

from bot_core.persistence.fingerprints import canonical_json_sha256
from deployment.windows_stage9_install_preparation import HANDOFF_INPUTS, prepare_installation_plan
from deployment.windows_stage9_operator_approval import (
    APPROVAL_KEYS,
    APPROVAL_SCOPE,
    OperatorApprovalError,
    OperatorAuthenticationUnavailable,
    _utc,
    authenticate_operator,
    validate_operator_approval,
)
from deployment.windows_stage10_artifact_preflight import _document

EXECUTION_SCOPE = ("install-exact-qualified-msi-on-clean-host",)
EXECUTION_KEYS = APPROVAL_KEYS | {
    "plan_sha256",
    "preparation_approval_sha256",
    "policy_sha256",
}
POLICY_KEYS = {
    "schema_version",
    "host_identity",
    "operator_sids",
    "artifact_root",
    "staging_root",
    "executor_root",
    "selection",
}
SELECTION_KEYS = {key for key in HANDOFF_INPUTS if key.startswith("expected_")}
INSTALL_ARGUMENTS = [
    "/i",
    "<protected-materialized-qualified-msi>",
    "/qn",
    "/norestart",
    "REBOOT=ReallySuppress",
    "ALLUSERS=1",
    "MSIINSTALLPERUSER=0",
    "/L*V",
    "<protected-msi-log>",
]


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise OperatorApprovalError(reason)


@contextmanager
def _locked_inputs(policy_path: Path, inputs: dict[str, Any], staging_root: Path) -> Iterator[Any]:
    from deployment.windows_stage9_install_native import locked_installation_inputs

    with locked_installation_inputs(
        artifact_root=Path(inputs["allowed_root"]),
        policy_path=policy_path,
        executor_root=Path(__file__).absolute().parent.parent,
        staging_root=staging_root,
    ) as locked:
        yield locked


def _load_policy(path: Path, inputs: dict[str, Any], staging_root: Path) -> dict[str, Any]:
    # The caller must already hold native locks and independently validated ACLs.
    policy = _document(path)
    _require(set(policy) == POLICY_KEYS, "exact independent execution policy is required")
    _require(
        type(policy["schema_version"]) is int and policy["schema_version"] == 1,
        "unsupported execution policy",
    )
    for key, expected in (
        ("artifact_root", Path(inputs["allowed_root"])),
        ("staging_root", staging_root),
        ("executor_root", Path(__file__).absolute().parent.parent),
    ):
        _require(
            isinstance(policy[key], str)
            and Path(policy[key]).is_absolute()
            and Path(policy[key]) == expected,
            "protected policy path binding differs",
        )
    selection = policy["selection"]
    _require(
        isinstance(selection, dict) and set(selection) == SELECTION_KEYS,
        "independent artifact selection is required",
    )
    _require(
        all(
            type(selection[key]) is type(inputs[key]) and selection[key] == inputs[key]
            for key in SELECTION_KEYS
        ),
        "artifact selection differs from protected policy",
    )
    return policy


def _operator_policy(policy: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(policy[key]) for key in ("host_identity", "operator_sids")}


def _fresh_plan(approval: Any, policy: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    prepared = prepare_installation_plan(
        approval, policy=_operator_policy(policy), handoff_inputs=inputs
    )
    if prepared.get("status") != "PREPARED_REVIEW_ONLY":
        if prepared.get("status") == "FAIL":
            raise OperatorApprovalError("fresh PRE-01 preparation failed")
        raise OperatorAuthenticationUnavailable(
            "fresh handoff, host or operator authority unavailable"
        )
    return {
        "schema_version": 1,
        "scope": list(EXECUTION_SCOPE),
        "qualification": deepcopy(prepared["qualification"]),
        "host_identity": deepcopy(policy["host_identity"]),
        "operator_sid": approval["operator_sid"],
        "preparation_approval_sha256": prepared["approval_sha256"],
        "policy_sha256": canonical_json_sha256(policy),
        "msiexec_arguments": list(INSTALL_ARGUMENTS),
        "base_qualification_required": True,
        "enrollment_authorized": False,
        "restart_authorized": False,
        "stage10_execution_authorized": False,
    }


def validate_execution_authorization(
    document: Any,
    plan: dict[str, Any],
    *,
    policy: dict[str, Any],
    preparation_approval: dict[str, Any],
) -> str:
    """Validate data binding only. Native authentication and console consent follow."""
    _require(
        isinstance(document, dict) and set(document) == EXECUTION_KEYS,
        "separate exact execution authorization is required",
    )
    _require(document["scope"] == list(EXECUTION_SCOPE), "execution scope differs")
    _require(
        document["approval_id"] != preparation_approval["approval_id"],
        "preparation approval cannot authorize execution",
    )
    for key, expected in (
        ("plan_sha256", canonical_json_sha256(plan)),
        ("preparation_approval_sha256", plan["preparation_approval_sha256"]),
        ("policy_sha256", plan["policy_sha256"]),
        ("operator_sid", preparation_approval["operator_sid"]),
    ):
        _require(document[key] == expected, "execution authorization binding differs")
    # Reuse PRE-01B.1's strict host/SID/UUID/run/artifact/trust/expiry validation.
    projection = {key: deepcopy(document[key]) for key in APPROVAL_KEYS}
    projection["scope"] = list(APPROVAL_SCOPE)
    validate_operator_approval(projection, plan["qualification"], policy=_operator_policy(policy))
    issued, expires = _utc(document["issued_at"]), _utc(document["expires_at"])
    _require(
        timedelta(0) < expires - issued <= timedelta(minutes=10),
        "execution authorization exceeds ten minutes",
    )
    digest: str = canonical_json_sha256(document)
    return digest


def _confirm_plan(digest: str) -> bool:
    """Read the exact phrase from a real native console, never a pipe or env flag."""
    if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise OperatorAuthenticationUnavailable("native Windows console required")
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise OperatorAuthenticationUnavailable("native Windows DLL loader unavailable")
    kernel = loader("kernel32.dll", use_last_error=True, winmode=0x00000800)
    kernel.GetStdHandle.argtypes = [ctypes.c_uint32]
    kernel.GetStdHandle.restype = ctypes.c_void_p
    kernel.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.GetConsoleMode.restype = ctypes.c_int
    kernel.ReadConsoleW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    kernel.ReadConsoleW.restype = ctypes.c_int
    kernel.WriteConsoleW.argtypes = list(kernel.ReadConsoleW.argtypes)
    kernel.WriteConsoleW.restype = ctypes.c_int
    source, target = kernel.GetStdHandle(0xFFFFFFF6), kernel.GetStdHandle(0xFFFFFFF5)
    mode, count = ctypes.c_uint32(), ctypes.c_uint32()
    if not all(kernel.GetConsoleMode(handle, ctypes.byref(mode)) for handle in (source, target)):
        raise OperatorAuthenticationUnavailable("interactive native console unavailable")
    prompt = ctypes.create_unicode_buffer(
        f"\r\nType INSTALL {digest} to install this exact MSI:\r\n"
    )
    if not kernel.WriteConsoleW(target, prompt, len(prompt.value), ctypes.byref(count), None):
        raise OperatorAuthenticationUnavailable("console prompt unavailable")
    response = ctypes.create_unicode_buffer(256)
    if not kernel.ReadConsoleW(source, response, 255, ctypes.byref(count), None):
        raise OperatorAuthenticationUnavailable("console consent unavailable")
    return response.value.rstrip("\r\n") == f"INSTALL {digest}"


class InstallerWaitInterrupted(RuntimeError):
    """Operator interrupted the wait; installer subsequently completed under pins."""

    def __init__(self, returncode: int):
        super().__init__("interrupted wait completed without retry or termination")
        self.returncode = returncode


def _run_installer(argv: list[str]) -> int:
    """The only installation process boundary. Never called by qualification or CI."""
    process = subprocess.Popen(
        argv,
        shell=False,
        cwd=str(Path(argv[0]).parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    interrupted = False
    while True:
        try:
            code = process.wait()
        except KeyboardInterrupt:
            # Killing the client does not safely roll back the Installer service.
            # Keep every input pin until completion; never launch another process.
            interrupted = True
            continue
        if interrupted:
            raise InstallerWaitInterrupted(code)
        return code


def _report() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report_type": "windows-stage9-manual-installation",
        "status": "BLOCKED",
        "overall_status": "BLOCKED",
        "checks": [],
        "installation_execution_authorized": False,
        "installer_invoked": False,
        "installation_executed": False,
        "installer_return_code": None,
        "restart_authorized": False,
        "automatic_retry_authorized": False,
        "stage_9": "IN_PROGRESS",
        "legal_production_enrollment": "NOT_PERFORMED",
        "run_stage10": False,
        "stage10_status": "BLOCKED",
        "stage10_qualification": "NOT_RUN",
    }


def _operation(
    approval: Any,
    execution: Any,
    *,
    policy_path: Path,
    handoff_inputs: dict[str, Any],
    staging_root: Path,
    execute: bool,
) -> dict[str, Any]:
    report = _report()
    installation = None
    stack = ExitStack()
    original_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    stage = "protected_inputs"
    try:
        _require(
            isinstance(handoff_inputs, dict) and set(handoff_inputs) == HANDOFF_INPUTS,
            "exact reviewed handoff inputs are required",
        )
        # Caller mutation cannot alter an operation after its review digest is calculated.
        inputs, approval_copy, execution_copy = (
            deepcopy(handoff_inputs),
            deepcopy(approval),
            deepcopy(execution),
        )
        locked = stack.enter_context(_locked_inputs(policy_path, inputs, staging_root))
        policy = _load_policy(policy_path, inputs, staging_root)
        stage = "fresh_preparation"
        plan = _fresh_plan(approval_copy, policy, inputs)
        digest = canonical_json_sha256(plan)
        report.update(
            plan=plan,
            plan_sha256=digest,
            preparation_approval_sha256=plan["preparation_approval_sha256"],
            policy_sha256=plan["policy_sha256"],
        )
        locked.verify_pins()
        if not execute:
            report["status"] = "PREPARED_REVIEW_ONLY"
            return report
        stage = "execution_authorization"
        authorization_digest = validate_execution_authorization(
            execution_copy, plan, policy=policy, preparation_approval=approval_copy
        )
        report["execution_authorization_sha256"] = authorization_digest
        stage = "protected_materialization"
        details = plan["qualification"]
        installation = locked.materialize_msi(
            archive=Path(inputs["archive"]),
            msi_member=details["msi_file"],
            expected_msi_sha256=details["msi_sha256"],
        )
        report.update(
            msi_log=str(installation.log_path),
            report_path=str(installation.report_path),
            intent_journal=str(installation.journal_path),
        )
        stage = "interactive_digest_consent"
        if not _confirm_plan(digest):
            report["status"] = "CANCELLED"
            return report
        stage = "final_authority_and_integrity"
        _require(
            _fresh_plan(approval_copy, policy, inputs) == plan,
            "handoff, host or operator changed after consent",
        )
        _require(
            _load_policy(policy_path, inputs, staging_root) == policy,
            "independent policy changed",
        )
        _require(
            validate_execution_authorization(
                execution_copy, plan, policy=policy, preparation_approval=approval_copy
            )
            == authorization_digest,
            "execution authorization changed or expired",
        )
        session = authenticate_operator(approval_copy, details, policy=_operator_policy(policy))
        _require(
            session["approval_sha256"] == plan["preparation_approval_sha256"],
            "final native session binding differs",
        )
        locked.verify_pins()
        installation.verify_pins()
        expected_argv = [str(installation.msiexec_path), *INSTALL_ARGUMENTS]
        expected_argv[2] = str(installation.msi_path)
        expected_argv[-1] = str(installation.log_path)
        _require(installation.argv == expected_argv, "native installer arguments differ")
        # Write-ahead journal survives interruption; no saved authorization is emitted.
        report.update(status="INSTALLATION_OUTCOME_UNKNOWN", installer_invocation_prepared=True)
        installation.write_report(report)
        # Recheck expiry after the durable journal, immediately before process handoff.
        validate_execution_authorization(
            execution_copy, plan, policy=policy, preparation_approval=approval_copy
        )
        stage = "installer_process"
        report["installer_invoked"] = True
        try:
            code = _run_installer(expected_argv)
        except InstallerWaitInterrupted as exc:
            code = exc.returncode
            report["operator_wait_interrupted"] = True
        _require(type(code) is int, "installer returned no exit status")
        report.update(installer_return_code=code, installation_executed=True)
        report["status"] = {
            0: "INSTALLED",
            3010: "INSTALLED_REBOOT_REQUIRED",
            1641: "UNEXPECTED_REBOOT_INITIATED",
            1602: "CANCELLED",
        }.get(code, "INSTALLER_ERROR")
        if code not in (0, 3010, 1602):
            report["overall_status"] = "FAIL"
        stage = "post_process_integrity"
        locked.verify_pins()
        installation.verify_pins()
        report["checks"].append({"name": stage, "status": "PASS"})
    except KeyboardInterrupt:
        report["status"] = (
            "INTERRUPTED_OUTCOME_UNKNOWN" if report["installer_invoked"] else "CANCELLED"
        )
    except OperatorAuthenticationUnavailable:
        report.update(status="BLOCKED", overall_status="BLOCKED")
        report["checks"].append(
            {
                "name": stage,
                "status": "BLOCKED",
                "reason": "required native or authenticated authority unavailable",
            }
        )
    except (ValueError, TypeError, KeyError, AttributeError):
        report.update(status="FAIL", overall_status="FAIL")
        report["checks"].append(
            {
                "name": stage,
                "status": "FAIL",
                "reason": "authorization, host or integrity check failed",
            }
        )
    except (OSError, RuntimeError, subprocess.SubprocessError):
        report["status"] = (
            "INSTALLATION_OUTCOME_UNKNOWN" if report["installer_invoked"] else "BLOCKED"
        )
        report["checks"].append(
            {
                "name": stage,
                "status": "BLOCKED",
                "reason": "required native operation unavailable; do not retry installation",
            }
        )
    finally:
        if installation is not None:
            try:
                installation.write_report(report)
            except (OSError, RuntimeError, ValueError):
                report.update(status="REPORT_WRITE_FAILED_OUTCOME_UNKNOWN", overall_status="FAIL")
                report["checks"].append({"name": "durable_report", "status": "FAIL"})
        try:
            stack.close()
        except (OSError, RuntimeError, ValueError):
            report.update(status="HANDLE_RELEASE_FAILED", overall_status="FAIL")
            report["checks"].append({"name": "native_handle_release", "status": "FAIL"})
        sys.dont_write_bytecode = original_bytecode_setting
    return report


def review_installation_plan(
    preparation_approval: Any,
    *,
    policy_path: Path,
    handoff_inputs: dict[str, Any],
    staging_root: Path,
) -> dict[str, Any]:
    """Fresh locked review digest; no installer, consent or execution authorization."""
    return _operation(
        preparation_approval,
        None,
        policy_path=policy_path,
        handoff_inputs=handoff_inputs,
        staging_root=staging_root,
        execute=False,
    )


def execute_installation(
    preparation_approval: Any,
    execution_authorization: Any,
    *,
    policy_path: Path,
    handoff_inputs: dict[str, Any],
    staging_root: Path,
) -> dict[str, Any]:
    """Manually execute once, then require independent passive base qualification."""
    return _operation(
        preparation_approval,
        execution_authorization,
        policy_path=policy_path,
        handoff_inputs=handoff_inputs,
        staging_root=staging_root,
        execute=True,
    )
