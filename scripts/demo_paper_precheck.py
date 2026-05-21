"""Safety precheck dla profilu config/e2e/demo_paper.yml.

Skrypt waliduje wyłącznie statyczne flagi bezpieczeństwa paper/demo i nie
uruchamia runtime, nie łączy się z giełdą oraz nie korzysta z kluczy API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


EXPECTED_RULES: tuple[tuple[str, Any], ...] = (
    ("trading.enable_paper_mode", True),
    ("trading.enable_live_mode", False),
    ("execution.default_mode", "paper"),
    ("execution.force_paper_when_offline", True),
    ("execution.live.enabled", False),
)


def _get_nested(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for segment in dotted_path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Waliduje bezpieczeństwo profilu demo/paper (bez runtime i bez exchange I/O).")
    )
    parser.add_argument(
        "--config",
        default="config/e2e/demo_paper.yml",
        help="Ścieżka do konfiguracji demo/paper overlay.",
    )
    parser.add_argument("--json", action="store_true", help="Zwróć wynik jako JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = Path(args.config)

    if not config_path.exists():
        payload = {
            "status": "error",
            "config": str(config_path),
            "issues": [f"config_not_found:{config_path}"],
        }
        print(json.dumps(payload, ensure_ascii=False) if args.json else payload["issues"][0])
        return 1

    content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        payload = {
            "status": "error",
            "config": str(config_path),
            "issues": ["config_not_mapping"],
        }
        print(json.dumps(payload, ensure_ascii=False) if args.json else "config_not_mapping")
        return 1

    checks: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for path, expected in EXPECTED_RULES:
        observed = _get_nested(content, path)
        ok = observed == expected
        checks[path] = {"expected": expected, "observed": observed, "ok": ok}
        if not ok:
            issues.append(f"unsafe_flag:{path}:expected={expected!r}:observed={observed!r}")

    status = "ok" if not issues else "error"
    response = {"status": status, "config": str(config_path), "checks": checks, "issues": issues}

    if args.json:
        print(json.dumps(response, ensure_ascii=False, sort_keys=True))
    elif status == "ok":
        print(f"OK: {config_path} spełnia wymagania safety dla demo/paper")
    else:
        print("ERROR: wykryto niebezpieczne flagi demo/paper:")
        for issue in issues:
            print(f" - {issue}")

    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
