from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from scripts.preview_artifact_cache import (
    DEFAULT_ROOT,
    DEFAULT_TTL_HOURS,
    REQUIRED_EVIDENCE,
    _contains_forbidden_name,
    _is_executable_file,
)

CONTRACT_VERSION = "preview_artifact_recapture_gate.v1"
DECISION_USE_CACHE = "USE_CACHED_ARTIFACT"
DECISION_REBUILD = "CONTROLLED_REBUILD_REQUIRED"
DECISION_BLOCKED = "BLOCKED_CACHE_ERROR"
DIST_DIR = Path("dist/preview/linux/dudzian-bot-preview")
MAIN_EXE = DIST_DIR / "dudzian-bot-preview"
REQUIRED_FILES = [str(DIST_DIR), str(MAIN_EXE), *[f"evidence/{name}" for name in REQUIRED_EVIDENCE]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview artifact recapture gate")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_HOURS)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dry-run-cleanup", action="store_true")
    return parser.parse_args()


def _security_contract() -> dict[str, object]:
    return {
        "local_only": True,
        "network_used": False,
        "subprocess_invoked": False,
        "shell_used": False,
        "secrets_read": False,
        "keychain_read": False,
        "env_values_read": False,
        "dot_env_read": False,
        "home_directory_scanned": False,
        "exchange_io": "disabled",
        "order_submission": "disabled",
        "build_performed": False,
        "pyinstaller_build_performed": False,
        "briefcase_build_performed": False,
        "installer_build_performed": False,
        "signing_performed": False,
        "release_upload_performed": False,
        "promotion_performed": False,
        "runtime_loop_started": False,
        "production_runtime_loop_started": False,
    }


def _base_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "status": "ok",
        "decision": DECISION_REBUILD,
        "safety_contract_version": CONTRACT_VERSION,
        "root": str(args.root),
        "ttl_hours": args.ttl_hours,
        "cleanup_performed": False,
        "cleanup_dry_run": bool(args.dry_run_cleanup),
        "cleanup_summary": {
            "candidates": [],
            "removed": [],
            "removed_count": 0,
            "errors": [],
        },
        "cache_hit": False,
        "cache_complete": False,
        "cache_fresh": False,
        "selected_cache_dir": None,
        "selected_executable": None,
        "selected_evidence_dir": None,
        "required_files": REQUIRED_FILES,
        "missing_files": [],
        "candidates": [],
        "stale_candidates": [],
        "incomplete_candidates": [],
        "errors": [],
        "security": _security_contract(),
    }


def _emit(payload: dict[str, object], as_json: bool) -> int:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True) if as_json else str(payload)
    sys.stdout.buffer.write(text.encode("cp1252", errors="replace") + b"\n")
    return 0 if payload.get("status") == "ok" else 2


def _is_under(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _path_label(root_arg: str, root_resolved: Path, path_resolved: Path) -> str:
    try:
        return str(Path(root_arg) / path_resolved.relative_to(root_resolved))
    except ValueError:
        return str(path_resolved)


def _iter_direct_cache_dirs(root: Path, errors: list[str]) -> list[Path]:
    root_resolved = root.resolve(strict=False)
    cache_dirs: list[Path] = []
    for child in root.iterdir():
        child_resolved = child.resolve(strict=False)
        if child.is_symlink():
            errors.append(f"candidate_symlink_rejected:{child}")
            continue
        if not child.is_dir():
            continue
        if child_resolved.parent != root_resolved or not _is_under(root_resolved, child_resolved):
            errors.append(f"candidate_outside_root:{child}")
            continue
        cache_dirs.append(child)
    return cache_dirs


def _cleanup_expired(
    root: Path, root_arg: str, ttl_hours: float, dry_run: bool
) -> dict[str, object]:
    summary: dict[str, object] = {
        "candidates": [],
        "removed": [],
        "removed_count": 0,
        "errors": [],
    }
    if not root.exists() or not root.is_dir():
        return summary
    root_resolved = root.resolve(strict=False)
    cutoff = time.time() - ttl_hours * 3600
    errors: list[str] = []
    candidates: list[str] = []
    removed: list[str] = []
    for child in sorted(_iter_direct_cache_dirs(root, errors), key=lambda p: p.name):
        child_resolved = child.resolve(strict=False)
        if child.stat().st_mtime >= cutoff:
            continue
        label = _path_label(root_arg, root_resolved, child_resolved)
        candidates.append(label)
        if dry_run:
            continue
        try:
            shutil.rmtree(child_resolved)
            removed.append(label)
        except OSError:
            errors.append(f"remove_failed:{label}")
    summary["candidates"] = candidates
    summary["removed"] = removed
    summary["removed_count"] = len(removed)
    summary["errors"] = errors
    return summary


def _scan_cache_tree(root: Path, cache_dir: Path) -> list[str]:
    diagnostics: list[str] = []
    root_resolved = root.resolve(strict=False)
    for path in cache_dir.rglob("*"):
        rel = path.relative_to(cache_dir).as_posix()
        if _contains_forbidden_name(rel) or _contains_forbidden_name(path.name):
            _append_unique(diagnostics, f"forbidden_artifact:{rel}")
        if path.is_symlink():
            target = path.resolve(strict=False)
            if not _is_under(root_resolved, target):
                _append_unique(diagnostics, f"symlink_outside_root:{rel}")
    return diagnostics


def _manifest_diagnostics(cache_dir: Path) -> list[str]:
    manifest = cache_dir / "cache_manifest.json"
    if not manifest.exists():
        return []
    diagnostics: list[str] = []
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return ["manifest_unreadable"]
    required = data.get("required_files")
    if required is not None:
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            diagnostics.append("manifest_required_files_invalid")
        else:
            missing_from_manifest = sorted(set(REQUIRED_FILES) - set(required))
            for rel in missing_from_manifest:
                diagnostics.append(f"manifest_missing_required:{rel}")
            for rel in required:
                if ".." in Path(rel).parts:
                    diagnostics.append(f"manifest_path_traversal:{rel}")
                    continue
                if not (cache_dir / rel).exists():
                    diagnostics.append(f"manifest_file_missing:{rel}")
    return diagnostics


def _check_complete_secure(root: Path, cache_dir: Path) -> tuple[bool, list[str], list[str]]:
    missing: list[str] = []
    errors: list[str] = []
    root_resolved = root.resolve(strict=False)
    cache_resolved = cache_dir.resolve(strict=False)
    if not _is_under(root_resolved, cache_resolved):
        return False, ["cache_dir_outside_root"], ["cache_dir_outside_root"]
    if cache_dir.is_symlink():
        return False, ["cache_dir_symlink"], ["cache_dir_symlink"]
    dist_dir = cache_dir / DIST_DIR
    exe = cache_dir / MAIN_EXE
    evidence_dir = cache_dir / "evidence"
    for rel_path, required_type in (
        (DIST_DIR, "dir"),
        (MAIN_EXE, "file"),
        (Path("evidence"), "dir"),
    ):
        path = cache_dir / rel_path
        resolved = path.resolve(strict=False)
        if path.is_symlink() or not _is_under(root_resolved, resolved):
            errors.append(f"required_path_outside_root:{rel_path.as_posix()}")
        if required_type == "dir" and not path.is_dir():
            missing.append(rel_path.as_posix())
        if required_type == "file" and not path.is_file():
            missing.append(rel_path.as_posix())
    if exe.is_file() and not _is_executable_file(exe):
        missing.append("executable_not_executable")
    for name in REQUIRED_EVIDENCE:
        rel = Path("evidence") / name
        ev_path = cache_dir / rel
        ev_resolved = ev_path.resolve(strict=False)
        if ev_path.is_symlink() or not _is_under(root_resolved, ev_resolved):
            errors.append(f"required_path_outside_root:{rel.as_posix()}")
        if not ev_path.is_file():
            missing.append(rel.as_posix())
    if dist_dir.exists() or evidence_dir.exists():
        errors.extend(_scan_cache_tree(root, cache_dir))
    errors.extend(_manifest_diagnostics(cache_dir))
    complete = not missing and not errors
    return complete, missing, errors


def _locate_latest(root: Path, root_arg: str, ttl_hours: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "candidates": [],
        "stale_candidates": [],
        "incomplete_candidates": [],
        "errors": [],
        "selected_cache_dir": None,
        "selected_executable": None,
        "selected_evidence_dir": None,
        "missing_files": [],
        "cache_hit": False,
        "cache_complete": False,
        "cache_fresh": False,
    }
    if not root.exists():
        result["errors"].append("root_missing")
        return result
    if not root.is_dir():
        result["errors"].append("root_not_directory")
        return result
    root_resolved = root.resolve(strict=False)
    errors: list[str] = result["errors"]
    candidates = _iter_direct_cache_dirs(root, errors)
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    result["candidates"] = [
        _path_label(root_arg, root_resolved, c.resolve(strict=False)) for c in candidates
    ]
    cutoff = time.time() - ttl_hours * 3600
    for candidate in candidates:
        candidate_resolved = candidate.resolve(strict=False)
        label = _path_label(root_arg, root_resolved, candidate_resolved)
        fresh = candidate.stat().st_mtime >= cutoff
        if not fresh:
            result["stale_candidates"].append(label)
            continue
        complete, missing, diagnostics = _check_complete_secure(root, candidate)
        if complete:
            result.update(
                {
                    "selected_cache_dir": label,
                    "selected_executable": str(Path(label) / MAIN_EXE),
                    "selected_evidence_dir": str(Path(label) / "evidence"),
                    "missing_files": [],
                    "cache_hit": True,
                    "cache_complete": True,
                    "cache_fresh": True,
                }
            )
            return result
        result["selected_cache_dir"] = label
        result["missing_files"] = missing + diagnostics
        for diagnostic in diagnostics:
            _append_unique(errors, diagnostic)
        result["incomplete_candidates"].append(
            {
                "path": label,
                "missing_files": missing,
                "errors": diagnostics,
            }
        )
    return result


def _finalize(payload: dict[str, object]) -> None:
    errors = payload["errors"]
    if not isinstance(errors, list):
        raise TypeError("errors field must be a list")
    blocking_errors = [err for err in errors if err != "root_missing"]
    if blocking_errors:
        payload["status"] = "blocked"
        payload["decision"] = DECISION_BLOCKED
        return
    if payload["cache_hit"] and payload["cache_complete"] and payload["cache_fresh"]:
        payload["status"] = "ok"
        payload["decision"] = DECISION_USE_CACHE
        return
    payload["status"] = "blocked"
    payload["decision"] = DECISION_REBUILD


def main() -> int:
    args = _parse_args()
    payload = _base_payload(args)
    if args.ttl_hours <= 0:
        payload["status"] = "error"
        payload["decision"] = DECISION_BLOCKED
        payload["errors"] = ["invalid_ttl_hours"]
        return _emit(payload, args.json)

    root = Path(args.root)
    cleanup_summary = _cleanup_expired(root, str(args.root), args.ttl_hours, args.dry_run_cleanup)
    payload["cleanup_performed"] = root.exists() and root.is_dir() and not args.dry_run_cleanup
    payload["cleanup_summary"] = cleanup_summary

    locate = _locate_latest(root, str(args.root), args.ttl_hours)
    for key in (
        "candidates",
        "stale_candidates",
        "incomplete_candidates",
        "selected_cache_dir",
        "selected_executable",
        "selected_evidence_dir",
        "missing_files",
        "cache_hit",
        "cache_complete",
        "cache_fresh",
    ):
        payload[key] = locate[key]

    stale_candidates = payload["stale_candidates"]
    cleanup_candidates = cleanup_summary.get("candidates", [])
    if isinstance(stale_candidates, list) and isinstance(cleanup_candidates, list):
        for item in cleanup_candidates:
            if isinstance(item, str):
                _append_unique(stale_candidates, item)

    errors = []
    for source in (cleanup_summary.get("errors", []), locate["errors"]):
        if isinstance(source, list):
            errors.extend(str(item) for item in source)
    payload["errors"] = errors
    _finalize(payload)
    return _emit(payload, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
