"""Generuje override wyłączający wskazany komponent runtime (Stage4/Stage5)."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import NamedTuple, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.packaging.build_core_bundle import (  # type: ignore
    _ensure_casefold_safe_tree,
    _ensure_no_symlinks,
    _ensure_windows_safe_tree,
)
from scripts._cli_common import now_iso

DEFAULT_OUTPUT_DIR = REPO_ROOT / "var/runtime/overrides"


class ComponentOverrideConfig(NamedTuple):
    action: str
    schema: str
    filename: str


COMPONENTS: dict[str, ComponentOverrideConfig] = {
    "multi_strategy": ComponentOverrideConfig(
        action="disable_multi_strategy",
        schema="stage4.scheduler.override",
        filename="multi_strategy_disable.json",
    ),
    "decision_orchestrator": ComponentOverrideConfig(
        action="disable_decision_orchestrator",
        schema="stage5.decision_orchestrator.override",
        filename="decision_orchestrator_disable.json",
    ),
}

DEFAULT_COMPONENT = "multi_strategy"
DISABLE_FILENAME = COMPONENTS[DEFAULT_COMPONENT].filename


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
    component: ComponentOverrideConfig,
    reason: str,
    requested_by: Optional[str],
    ticket: Optional[str],
    expires_minutes: Optional[int],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": component.schema,
        "schema_version": "1.0",
        "timestamp": now_iso(),
        "action": component.action,
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
        description="Generuje plik override wyłączający wskazany komponent runtime",
    )
    parser.add_argument(
        "--component",
        choices=sorted(COMPONENTS),
        default=DEFAULT_COMPONENT,
        help=(
            "Komponent do wyłączenia (domyślnie multi_strategy – scheduler Stage4). "
            "Użyj decision_orchestrator, aby zatrzymać orchestrator Stage5."
        ),
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


def _execute(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    output_dir = _prepare_output_dir(Path(args.output_dir))
    component_cfg = COMPONENTS[args.component]
    target = output_dir / component_cfg.filename
    if target.exists():
        raise FileExistsError(f"Plik override już istnieje: {target}")

    payload = _build_payload(
        component=component_cfg,
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

    logging.info("Zapisano override %s w %s", args.component, target)
    return 0


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    return _execute(args)


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI entrypoint
    args: argparse.Namespace | None = None
    try:
        args = _parse_args(argv)
        return _execute(args)
    except Exception as exc:  # pragma: no cover - CLI guardrail
        target = "komponentu runtime"
        if args is not None:
            target = f"komponentu {args.component}"
        logging.error("Nie udało się wygenerować override %s: %s", target, exc)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
