from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

CONTRACT_VERSION = "preview_artifact_cache.v1"
DEFAULT_ROOT = "var/artifacts/exe_preview"
DEFAULT_TTL_HOURS = 24
REQUIRED_EVIDENCE = [
    "preview_artifact_seal.json",
    "preview_artifact_hashes.sha256",
    "main_executable.sha256",
    "leak_triage_summary.json",
    "leak_triage_summary.tsv",
]
FORBIDDEN_GLOBS = (
    ".env",
    "*.env",
    "trading.db",
    "*api_key*",
    "*api_secret*",
    "*secret*",
    "*token*",
    "*keychain*",
)
FORBIDDEN_SEGMENTS = ("bot_core/logs", "logs", "reports", "test-results", "var/security")
STAGE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview artifact cache store/locator")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_HOURS)
    parser.add_argument("--stage", default="")
    parser.add_argument("--source", default="dist/preview/linux/dudzian-bot-preview")
    parser.add_argument("--evidence-dir", default="")
    parser.add_argument("--store", action="store_true")
    parser.add_argument("--locate-latest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _emit(payload: dict[str, object], as_json: bool) -> int:
    text = json.dumps(payload, ensure_ascii=True) if as_json else str(payload)
    sys.stdout.buffer.write(text.encode("cp1252", errors="replace") + b"\n")
    return 0 if payload.get("status") == "ok" else 2


def _base_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "status": "ok",
        "safety_contract_version": CONTRACT_VERSION,
        "root": str(args.root),
        "ttl_hours": args.ttl_hours,
        "stage": args.stage,
        "selected_cache_dir": None,
        "cache_hit": False,
        "cache_complete": False,
        "cache_fresh": False,
        "required_files": [
            "dist/preview/linux/dudzian-bot-preview",
            "dist/preview/linux/dudzian-bot-preview/dudzian-bot-preview",
            *[f"evidence/{x}" for x in REQUIRED_EVIDENCE],
        ],
        "missing_files": [],
        "copied_files": [],
        "copied_count": 0,
        "errors": [],
        "forbidden_artifacts": [],
        "candidates": [],
        "stale_candidates": [],
        "incomplete_candidates": [],
        "security": {
            "local_only": True,
            "network_used": False,
            "subprocess_invoked": False,
            "secrets_read": False,
            "keychain_read": False,
            "env_values_read": False,
            "dot_env_read": False,
            "home_directory_scanned": False,
            "exchange_io": "disabled",
            "order_submission": "disabled",
        },
    }


def _inside(root: Path, child: Path) -> bool:
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False


def _is_executable_file(path: Path) -> bool:
    if not path.is_file():
        return False
    mode = path.stat().st_mode
    has_exec_bit = bool(mode & 0o111)
    return has_exec_bit and os.access(path, os.X_OK)


def _check_complete(cache_dir: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    if not (cache_dir / "dist/preview/linux/dudzian-bot-preview").is_dir():
        missing.append("dist/preview/linux/dudzian-bot-preview")
    exe = cache_dir / "dist/preview/linux/dudzian-bot-preview/dudzian-bot-preview"
    if not exe.is_file():
        missing.append("dist/preview/linux/dudzian-bot-preview/dudzian-bot-preview")
    elif not _is_executable_file(exe):
        missing.append("executable_not_executable")
    for name in REQUIRED_EVIDENCE:
        rel = f"evidence/{name}"
        if not (cache_dir / rel).is_file():
            missing.append(rel)
    return (len(missing) == 0), missing


def _contains_forbidden_name(rel: str) -> bool:
    lowered = rel.lower().replace("\\", "/")
    if any(seg in lowered for seg in FORBIDDEN_SEGMENTS):
        return True
    return any(fnmatch.fnmatch(lowered, pat) for pat in FORBIDDEN_GLOBS)


def _scan_source_forbidden(source_dir: Path) -> list[str]:
    forbidden = []
    for path in source_dir.rglob("*"):
        rel = path.relative_to(source_dir).as_posix()
        if _contains_forbidden_name(rel) or _contains_forbidden_name(path.name):
            forbidden.append(rel)
    return sorted(set(forbidden))


def _valid_stage(stage: str) -> bool:
    return (
        bool(stage)
        and ".." not in stage
        and "/" not in stage
        and "\\" not in stage
        and STAGE_RE.fullmatch(stage) is not None
    )


def _do_store(args: argparse.Namespace, payload: dict[str, object]) -> int:
    root = Path(args.root).resolve(strict=False)
    src = Path(args.source).resolve(strict=False)
    evidence = Path(args.evidence_dir).resolve(strict=False)
    if args.ttl_hours <= 0:
        payload["status"] = "error"
        payload["errors"] = ["invalid_ttl_hours"]
        return _emit(payload, args.json)
    if not _valid_stage(args.stage):
        payload["status"] = "error"
        payload["errors"] = ["invalid_stage"]
        return _emit(payload, args.json)
    missing: list[str] = []
    if not src.is_dir():
        missing.append("dist/preview/linux/dudzian-bot-preview")
    main_exe = src / "dudzian-bot-preview"
    if not main_exe.is_file():
        missing.append("dist/preview/linux/dudzian-bot-preview/dudzian-bot-preview")
    elif not _is_executable_file(main_exe):
        missing.append("executable_not_executable")
    forbidden_artifacts = _scan_source_forbidden(src) if src.is_dir() else []
    if forbidden_artifacts:
        payload["status"] = "blocked"
        payload["errors"] = ["forbidden_artifact_detected"]
        payload["forbidden_artifacts"] = forbidden_artifacts
        return _emit(payload, args.json)

    evidence_rel = Path(args.evidence_dir).as_posix().lower()
    if _contains_forbidden_name(evidence_rel) or _contains_forbidden_name(
        Path(args.evidence_dir).name
    ):
        payload["status"] = "blocked"
        payload["errors"] = ["evidence_path_forbidden"]
        return _emit(payload, args.json)

    for n in REQUIRED_EVIDENCE:
        evp = evidence / n
        if not evp.is_file():
            missing.append(f"evidence/{n}")
    payload["missing_files"] = missing
    if missing:
        payload["status"] = "blocked"
        return _emit(payload, args.json)
    run_id = f"{args.stage}_{int(time.time())}"
    target = (root / run_id).resolve(strict=False)
    if target.parent != root or not _inside(root, target):
        payload["status"] = "error"
        payload["errors"] = ["target_outside_root"]
        return _emit(payload, args.json)
    payload["selected_cache_dir"] = str(target)
    copy_list = [
        (src, target / "dist/preview/linux/dudzian-bot-preview"),
        *[(evidence / n, target / "evidence" / n) for n in REQUIRED_EVIDENCE],
    ]
    payload["copied_files"] = [str(dst.relative_to(target)) for _, dst in copy_list]
    if args.dry_run:
        return _emit(payload, args.json)
    (target / "evidence").mkdir(parents=True, exist_ok=True)
    for src_path, dst_path in copy_list:
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
    manifest = {
        "stage": args.stage,
        "created_epoch": int(time.time()),
        "required_files": payload["required_files"],
    }
    (target / "cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    payload["copied_count"] = len(payload["copied_files"])
    payload["cache_complete"], payload["missing_files"] = _check_complete(target)
    payload["cache_hit"] = payload["cache_complete"]
    payload["cache_fresh"] = True
    return _emit(payload, args.json)


def _do_locate(args: argparse.Namespace, payload: dict[str, object]) -> int:
    root = Path(args.root).resolve(strict=False)
    if args.ttl_hours <= 0:
        payload["status"] = "error"
        payload["errors"] = ["invalid_ttl_hours"]
        return _emit(payload, args.json)
    if not root.exists():
        return _emit(payload, args.json)
    cutoff = time.time() - args.ttl_hours * 3600
    candidates = [p for p in root.iterdir() if p.is_dir()]
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    payload["candidates"] = [str(c) for c in candidates]
    for cand in candidates:
        if cand.stat().st_mtime < cutoff:
            payload["stale_candidates"].append(str(cand))
            continue
        complete, missing = _check_complete(cand)
        if complete:
            payload["selected_cache_dir"] = str(cand)
            payload["cache_hit"] = True
            payload["cache_complete"] = True
            payload["cache_fresh"] = True
            payload["missing_files"] = []
            return _emit(payload, args.json)
        payload["selected_cache_dir"] = str(cand)
        payload["missing_files"] = missing
        payload["incomplete_candidates"].append(str(cand))
    payload["status"] = "blocked"
    return _emit(payload, args.json)


def main() -> int:
    args = _parse_args()
    payload = _base_payload(args)
    if args.store and args.locate_latest:
        payload["status"] = "error"
        payload["errors"] = ["store_and_locate_mutually_exclusive"]
        return _emit(payload, args.json)
    if args.store:
        return _do_store(args, payload)
    if args.locate_latest:
        return _do_locate(args, payload)
    return _emit(payload, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
