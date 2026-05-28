from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

CONTRACT_VERSION = "artifact_retention_cleanup.v1"
DEFAULT_ROOT = "var/artifacts/exe_preview"
DEFAULT_TTL_HOURS = 24


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup expired local preview artifact cache")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_HOURS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _emit(payload: dict[str, object], as_json: bool) -> int:
    if as_json:
        text = json.dumps(payload, ensure_ascii=True)
    else:
        text = str(payload)
    sys.stdout.buffer.write(text.encode("cp1252", errors="replace") + b"\n")
    return 0 if payload.get("status") == "ok" else 2


def _is_direct_child(root_resolved: Path, child: Path) -> bool:
    return child.parent == root_resolved


def main() -> int:
    args = _parse_args()
    payload: dict[str, object] = {
        "status": "ok",
        "safety_contract_version": CONTRACT_VERSION,
        "root": str(args.root),
        "ttl_hours": args.ttl_hours,
        "dry_run": bool(args.dry_run),
        "now_epoch": int(time.time()),
        "cutoff_epoch": 0,
        "candidates": [],
        "removed": [],
        "removed_count": 0,
        "errors": [],
        "security": {
            "local_only": True,
            "home_directory_scanned": False,
            "env_values_read": False,
            "dot_env_read": False,
            "secrets_read": False,
            "keychain_read": False,
            "network_used": False,
            "subprocess_invoked": False,
        },
    }

    if args.ttl_hours <= 0:
        payload["status"] = "error"
        payload["errors"] = ["invalid_ttl_hours"]
        return _emit(payload, args.json)

    root = Path(args.root)
    root_resolved = root.resolve(strict=False)
    cutoff_epoch = int(payload["now_epoch"]) - int(args.ttl_hours * 3600)
    payload["cutoff_epoch"] = cutoff_epoch

    if not root.exists():
        return _emit(payload, args.json)

    if not root.is_dir():
        payload["status"] = "error"
        payload["errors"] = ["root_not_directory"]
        return _emit(payload, args.json)

    candidates: list[str] = []
    removed: list[str] = []
    errors: list[str] = []

    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        child_resolved = child.resolve(strict=False)
        if not _is_direct_child(root_resolved, child_resolved):
            continue
        mtime = int(child.stat().st_mtime)
        if mtime >= cutoff_epoch:
            continue
        rel = child_resolved.relative_to(root_resolved)
        candidate = str(Path(args.root) / rel)
        candidates.append(candidate)
        if args.dry_run:
            continue
        try:
            shutil.rmtree(child_resolved)
            removed.append(candidate)
        except OSError:
            errors.append(f"remove_failed:{candidate}")

    payload["candidates"] = candidates
    payload["removed"] = removed
    payload["removed_count"] = len(removed)
    if errors:
        payload["status"] = "error"
        payload["errors"] = errors
    return _emit(payload, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
