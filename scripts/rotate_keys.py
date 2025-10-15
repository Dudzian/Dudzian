"""CLI do planowania i wykonywania rotacji kluczy Stage5."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.config import load_core_config
from bot_core.config.models import KeyRotationConfig, KeyRotationEntryConfig
from bot_core.security import RotationRegistry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "core.yaml"),
        help="Ścieżka do pliku konfiguracji core.yaml",
    )
    parser.add_argument(
        "--registry-path",
        dest="registry_path",
        help="Opcjonalna ścieżka rejestru rotacji (nadpisuje konfigurację)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Katalog docelowy planu (domyślnie z konfiguracji)",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Opcjonalna nazwa bazowa plików (bez rozszerzeń)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Zapisz aktualizacje rotacji dla wpisów wymagających działania",
    )
    parser.add_argument(
        "--interval-override",
        type=float,
        default=None,
        help="Opcjonalne globalne zastąpienie interwału (dni)",
    )
    parser.add_argument(
        "--warn-within-override",
        type=float,
        default=None,
        help="Opcjonalne globalne zastąpienie progu ostrzeżeń (dni)",
    )
    return parser


def _load_rotation_config(config_path: Path) -> KeyRotationConfig:
    core_config = load_core_config(config_path)
    observability = getattr(core_config, "observability", None)
    if observability is None or observability.key_rotation is None:
        raise ValueError("Konfiguracja nie zawiera sekcji observability.key_rotation")
    return observability.key_rotation


def _normalize_path(path: str | None, *, default: str, base: Path) -> Path:
    candidate = Path(path) if path else Path(default)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


def _status_for_entry(
    registry: RotationRegistry,
    entry: KeyRotationEntryConfig,
    *,
    interval: float,
    warn_within: float,
    now: datetime,
) -> dict[str, object]:
    status = registry.status(
        entry.key,
        entry.purpose,
        interval_days=interval,
        now=now,
    )
    state = "ok"
    if status.is_overdue:
        state = "overdue"
    elif status.is_due:
        state = "due"
    elif status.due_in_days <= warn_within:
        state = "warning"
    payload = asdict(status)
    payload["state"] = state
    payload["due_within_days"] = warn_within
    last_rotated = payload.get("last_rotated")
    if isinstance(last_rotated, datetime):
        payload["last_rotated"] = last_rotated.isoformat().replace("+00:00", "Z")
    return payload


def run(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    rotation_config = _load_rotation_config(config_path)
    registry_path = _normalize_path(
        args.registry_path or rotation_config.registry_path,
        default=rotation_config.registry_path,
        base=config_path.parent,
    )

    registry = RotationRegistry(registry_path)
    now = datetime.now(timezone.utc)

    output_root = Path(
        args.output_dir
        if args.output_dir
        else rotation_config.audit_directory
    )
    if not output_root.is_absolute():
        output_root = (config_path.parent / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    interval_override = args.interval_override
    warn_override = args.warn_within_override

    entries = list(rotation_config.entries)
    if not entries:
        raise ValueError("Konfiguracja key_rotation wymaga co najmniej jednego wpisu entries")

    results: list[dict[str, object]] = []
    updated: list[dict[str, object]] = []

    for entry in entries:
        interval_days = interval_override or entry.interval_days or rotation_config.default_interval_days
        warn_days = warn_override or entry.warn_within_days or rotation_config.default_warn_within_days
        status_payload = _status_for_entry(
            registry,
            entry,
            interval=interval_days,
            warn_within=warn_days,
            now=now,
        )
        status_payload["entry"] = {
            "key": entry.key,
            "purpose": entry.purpose,
            "interval_days": interval_days,
            "warn_within_days": warn_days,
        }
        results.append(status_payload)
        if args.execute and status_payload["state"] in {"due", "overdue"}:
            registry.mark_rotated(entry.key, entry.purpose, timestamp=now)
            updated.append(status_payload["entry"])

    if args.basename:
        basename = args.basename
    else:
        basename = f"rotation_plan_{now.strftime('%Y%m%dT%H%M%SZ')}"
    report_path = output_root / f"{basename}.json"
    report = {
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "config_path": str(config_path),
        "registry_path": str(registry_path),
        "execute": bool(args.execute),
        "results": results,
        "updated": updated,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Wypisujemy lokalizację dla runbooka
    print(f"Plan rotacji zapisany w {report_path}")
    if args.execute and updated:
        print("Zaktualizowano wpisy rotacji dla:")
        for entry in updated:
            print(f" - {entry['key']} ({entry['purpose']})")
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(run())
