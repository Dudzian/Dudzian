"""Włącza tryb awaryjny wyłączający scheduler wielostrate-giczny Stage4."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.packaging.build_core_bundle import (  # type: ignore
    _ensure_casefold_safe_tree,
    _ensure_no_symlinks,
    _ensure_windows_safe_tree,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "var/runtime/overrides"
DISABLE_FILENAME = "multi_strategy_disable.json"


def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _calculate_expiration(minutes: int | None) -> str | None:
    if minutes is None:
        return None
    delta = _dt.timedelta(minutes=minutes)
    return (_dt.datetime.utcnow().replace(microsecond=0) + delta).isoformat() + "Z"


def _prepare_output_dir(path: Path) -> Path:
    path = path.expanduser()
    _ensure_no_symlinks(path, label="Katalog overrides")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Katalog overrides musi być katalogiem: {path}")
        _ensure_casefold_safe_tree(path, label="Katalog overrides")
    else:
        path.mkdir(parents=True, exist_ok=True)
    _ensure_windows_safe_tree(path, label="Katalog overrides")
    return path.resolve()


def _build_payload(
    *,
    reason: str,
    requested_by: Optional[str],
    ticket: Optional[str],
    expires_minutes: Optional[int],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "stage4.scheduler.override",
        "schema_version": "1.0",
        "timestamp": _now_iso(),
        "action": "disable_multi_strategy",
        "reason": reason,
    }
    if requested_by:
        payload["requested_by"] = requested_by
    if ticket:
        payload["ticket"] = ticket
    expiration = _calculate_expiration(expires_minutes)
    if expiration:
        payload["expires_at"] = expiration
    return payload


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generuje plik wyłączający scheduler wielostrate-giczny",
    )
    parser.add_argument("--reason", required=True)
    parser.add_argument("--requested-by")
    parser.add_argument("--ticket")
    parser.add_argument(
        "--duration-minutes",
        type=int,
        help="Czas obowiązywania wyłączenia (domyślnie bezterminowo)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Katalog docelowy dla pliku override",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    output_dir = _prepare_output_dir(Path(args.output_dir))
    target = output_dir / DISABLE_FILENAME
    if target.exists():
        raise FileExistsError(f"Plik override już istnieje: {target}")

    payload = _build_payload(
        reason=args.reason.strip(),
        requested_by=(args.requested_by or "").strip() or None,
        ticket=(args.ticket or "").strip() or None,
        expires_minutes=args.duration_minutes,
    )
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if os.name != "nt":
        target.chmod(0o600)

    logging.info("Zapisano override scheduler-a w %s", target)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI entrypoint
    try:
        return run(argv)
    except Exception as exc:  # pragma: no cover - CLI guardrail
        logging.error("Nie udało się wygenerować override scheduler-a: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
