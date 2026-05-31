from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from preview_artifact_cache import (
    FORBIDDEN_GLOBS,
    FORBIDDEN_SEGMENTS,
    REQUIRED_EVIDENCE,
    _find_main_executable,
    _is_executable_file,
    _main_executable_names,
)

CONTRACT_VERSION = "preview_artifact_launch_plan.v1"
ARTIFACT_POLICY_VERSION = "security_packaging_artifact_policy.v1"
DEFAULT_ROOT = "var/artifacts/exe_preview"
ROOT_OUT_OF_SCOPE = "preview_artifact_root_out_of_scope"
EVIDENCE_MISSING = "preview_artifact_evidence_missing"
ARTIFACT_BINARY_DIR = Path("dist/preview/linux/dudzian-bot-preview")
ALLOWED_ROOT_PREFIXES = (
    Path("dist/preview"),
    Path("var/artifacts"),
    Path("var/tmp"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a safe, plan-only launch manifest for a preview EXE artifact"
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _denied_artifact_patterns() -> list[str]:
    return sorted({*FORBIDDEN_GLOBS, *FORBIDDEN_SEGMENTS})


def _base_payload(root: str) -> dict[str, object]:
    return {
        "status": "blocked",
        "safety_contract_version": CONTRACT_VERSION,
        "root": root,
        "selected_artifact_root": None,
        "selected_executable": None,
        "launch_command_preview": [],
        "launch_mode": "demo",
        "preview_plan": True,
        "live_mode_allowed": False,
        "command_execution_allowed": False,
        "command_executed": False,
        "subprocess_invoked": False,
        "shell_used": False,
        "exchange_io": "disabled",
        "order_submission": "disabled",
        "runtime_loop_started": False,
        "production_runtime_loop_started": False,
        "secrets_read": False,
        "keychain_read": False,
        "env_values_read": False,
        "dot_env_read": False,
        "home_directory_scanned": False,
        "artifact_policy_checked": True,
        "artifact_exclude_policy_version": ARTIFACT_POLICY_VERSION,
        "denied_artifact_patterns": _denied_artifact_patterns(),
        "env_file_bundled": False,
        "local_db_bundled": False,
        "logs_bundled": False,
        "reports_bundled": False,
        "tmp_artifacts_bundled": False,
        "test_secrets_bundled": False,
        "cache_artifacts_bundled": False,
        "local_user_data_bundled": False,
        "keychain_artifacts_bundled": False,
        "issues": [],
        "artifact_found": False,
        "executable_valid": False,
        "evidence_required": True,
        "evidence_present": False,
        "seal_evidence_present": False,
        "hash_evidence_present": False,
        "leak_triage_evidence_present": False,
        "artifact_verified": False,
        "launch_plan_ready": False,
        "candidate_names": list(_main_executable_names()),
    }


def _emit(payload: dict[str, object], as_json: bool) -> int:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True) if as_json else str(payload)
    sys.stdout.buffer.write(text.encode("cp1252", errors="replace") + b"\n")
    return 0 if payload.get("status") == "ok" else 2


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _has_allowed_root_prefix(relative_root: Path) -> bool:
    return any(
        relative_root == prefix or _is_relative_to(relative_root, prefix)
        for prefix in ALLOWED_ROOT_PREFIXES
    )


def _is_root_in_scope(root: Path) -> bool:
    repo_root = _repo_root()
    if root == repo_root:
        return True
    if not _is_relative_to(root, repo_root):
        return False
    relative_root = root.relative_to(repo_root)
    return _has_allowed_root_prefix(relative_root) or (root / ARTIFACT_BINARY_DIR).is_dir()


def _candidate_roots(root: Path) -> list[Path]:
    if (root / ARTIFACT_BINARY_DIR).is_dir():
        return [root]
    if not root.is_dir():
        return []
    children = [child for child in root.iterdir() if child.is_dir()]
    children.sort(key=lambda child: (child.stat().st_mtime, child.name), reverse=True)
    return children


def _select_executable(root: Path) -> tuple[Path | None, Path | None, str]:
    invalid_candidate_seen = False
    for artifact_root in _candidate_roots(root):
        executable = _find_main_executable(artifact_root)
        if executable is None:
            continue
        if _is_executable_file(executable):
            return artifact_root, executable, ""
        invalid_candidate_seen = True
    if invalid_candidate_seen:
        return None, None, "preview_artifact_executable_invalid"
    return None, None, "preview_artifact_missing"


def _evidence_path(artifact_root: Path, name: str) -> Path:
    return artifact_root / "evidence" / name


def _evidence_state(artifact_root: Path) -> dict[str, bool]:
    seal_present = _evidence_path(artifact_root, "preview_artifact_seal.json").is_file()
    hash_present = all(
        _evidence_path(artifact_root, name).is_file()
        for name in ("preview_artifact_hashes.sha256", "main_executable.sha256")
    )
    leak_triage_present = all(
        _evidence_path(artifact_root, name).is_file()
        for name in ("leak_triage_summary.json", "leak_triage_summary.tsv")
    )
    required_present = all(
        _evidence_path(artifact_root, name).is_file() for name in REQUIRED_EVIDENCE
    )
    return {
        "seal_evidence_present": seal_present,
        "hash_evidence_present": hash_present,
        "leak_triage_evidence_present": leak_triage_present,
        "evidence_present": required_present,
    }


def _build_payload(root_value: str) -> dict[str, object]:
    payload = _base_payload(root_value)
    root = Path(root_value).resolve(strict=False)
    if not _is_root_in_scope(root):
        payload["issues"] = [ROOT_OUT_OF_SCOPE]
        return payload

    artifact_root, executable, issue = _select_executable(root)
    if executable is None or artifact_root is None:
        payload["issues"] = [issue]
        return payload

    selected_executable = str(executable)
    evidence_state = _evidence_state(artifact_root)
    payload.update(
        {
            "artifact_found": True,
            "executable_valid": True,
            "selected_artifact_root": str(artifact_root),
            "selected_executable": selected_executable,
            **evidence_state,
        }
    )
    if not evidence_state["evidence_present"]:
        payload["issues"] = [EVIDENCE_MISSING]
        return payload

    payload.update(
        {
            "status": "ok",
            "artifact_verified": True,
            "launch_plan_ready": True,
            "launch_command_preview": [
                selected_executable,
                "--mode",
                "demo",
                "--preview-plan",
            ],
            "issues": [],
        }
    )
    return payload


def main() -> int:
    args = _parse_args()
    payload = _build_payload(str(args.root))
    return _emit(payload, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
