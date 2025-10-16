#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage5 Key Rotation – merged CLI (HEAD + main)

Subcommands:
  - batch : aktualizuje rejestr rotacji dla wskazanych środowisk i zapisuje
            podpisany raport Stage5 (HMAC Base64)
  - plan  : generuje plan rotacji wg observability.key_rotation i opcjonalnie
            oznacza klucze jako zrotowane (execute)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- kompatybilny import load_core_config (różne gałęzie mają różne miejsca) ---
try:
    from bot_core.config.loader import load_core_config as _load_core_config_loader
except Exception:  # pragma: no cover
    _load_core_config_loader = None
try:
    from bot_core.config import load_core_config as _load_core_config_pkg
except Exception:  # pragma: no cover
    _load_core_config_pkg = None

def load_core_config(path: str | Path):
    if _load_core_config_pkg is not None:
        return _load_core_config_pkg(path)
    if _load_core_config_loader is not None:
        return _load_core_config_loader(path)
    raise ImportError("Nie można załadować load_core_config (sprawdź zależności bot_core).")

# --- RotationRegistry – różne lokalizacje w repo ---
RotationRegistry = None
try:
    from bot_core.security.rotation import RotationRegistry as _RR  # HEAD
    RotationRegistry = _RR
except Exception:
    try:
        from bot_core.security import RotationRegistry as _RR  # main
        RotationRegistry = _RR
    except Exception as exc:  # pragma: no cover
        raise ImportError("Brak RotationRegistry w bot_core.security") from exc

# --- Modele/raport (HEAD) ---
try:
    from bot_core.security.rotation_report import (
        RotationRecord,
        RotationSummary,
        write_rotation_summary,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError("Brak modułu rotation_report (wymagany do 'batch').") from exc

# --- Opcjonalne typy z models (używamy tylko dla hintów/konwencji) ---
try:
    from bot_core.config.models import CoreConfig, EnvironmentConfig  # type: ignore
except Exception:  # pragma: no cover
    CoreConfig = Any  # type: ignore
    EnvironmentConfig = Any  # type: ignore

# =====================================================================
# Wspólne utilsy
# =====================================================================

def _decode_key_b64(value: str | None) -> bytes | None:
    if not value:
        return None
    return base64.b64decode(value, validate=True)

def _resolve_signing_key_b64(
    *, inline: str | None, file_path: str | None, env_name: str | None
) -> bytes | None:
    if inline:
        return _decode_key_b64(inline)
    if file_path:
        return _decode_key_b64(Path(file_path).read_text(encoding="utf-8"))
    if env_name:
        value = os.environ.get(env_name)
        return _decode_key_b64(value) if value else None
    return None

def _parse_datetime(value: str | None) -> datetime:
    if not value:
        now = datetime.now(timezone.utc)
        return now.replace(microsecond=0)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0)

# =====================================================================
# Subcommand: batch  (HEAD – podpisany raport Stage5)
# =====================================================================

def _build_parser_batch(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "batch",
        help="Wsadowa rotacja wg środowisk + podpisany raport Stage5",
        description="Aktualizuje rejestry rotacji (po środowiskach) i zapisuje podpisany raport Stage5.",
    )
    p.add_argument("--config", default="config/core.yaml", help="Ścieżka do CoreConfig")
    p.add_argument("--environment", action="append", dest="environments",
                   help="Nazwa środowiska (można wielokrotnie); domyślnie wszystkie")
    p.add_argument("--interval-days", type=float, default=90.0,
                   help="Interwał rotacji w dniach (domyślnie 90)")
    p.add_argument("--operator", required=True, help="Operator wykonujący rotację")
    p.add_argument("--notes", help="Uwagi do raportu")
    p.add_argument("--artifact-root", default="var/audit/stage5/key_rotation",
                   help="Katalog docelowy raportów")
    p.add_argument("--output", help="Pełna ścieżka pliku raportu (opcjonalnie)")
    p.add_argument("--executed-at", help="Znacznik czasu rotacji (ISO8601); domyślnie teraz")
    p.add_argument("--dry-run", action="store_true", help="Nie zapisuj zmian do rejestru")
    p.add_argument("--signing-key", help="HMAC Base64")
    p.add_argument("--signing-key-file", help="Plik z HMAC (Base64)")
    p.add_argument("--signing-key-env", help="Zmienna środowiskowa z HMAC (Base64)")
    p.add_argument("--signing-key-id", help="Identyfikator klucza podpisującego")
    p.set_defaults(_handler=_handle_batch)
    return p

def _resolve_environments(config: CoreConfig, names: Iterable[str] | None) -> list[EnvironmentConfig]:
    if not names:
        return list(config.environments.values())
    resolved: list[EnvironmentConfig] = []
    for name in names:
        if name not in config.environments:
            raise SystemExit(f"Środowisko '{name}' nie istnieje w konfiguracji")
        resolved.append(config.environments[name])
    return resolved

def _registry_path(environment: EnvironmentConfig) -> Path:
    return Path(environment.data_cache_path) / "security" / "rotation_log.json"

def _resolve_output_path_batch(
    args: argparse.Namespace,
    *,
    executed_at: datetime,
    environments: Sequence[EnvironmentConfig],
) -> Path:
    if args.output:
        return Path(args.output)
    base = Path(args.artifact_root)
    timestamp = executed_at.strftime("%Y%m%dT%H%M%SZ")
    if len(environments) == 1:
        suffix = environments[0].name.replace("/", "-")
        filename = f"key_rotation_{suffix}_{timestamp}.json"
    else:
        filename = f"key_rotation_batch_{timestamp}.json"
    return base / filename

def _handle_batch(args: argparse.Namespace) -> int:
    executed_at = _parse_datetime(args.executed_at)
    try:
        config = load_core_config(args.config)
    except FileNotFoundError:
        print(json.dumps({"error": f"Plik konfiguracji {args.config} nie istnieje"}))
        return 2

    environments = _resolve_environments(config, args.environments)
    if not environments:
        print(json.dumps({"error": "Brak środowisk do rotacji"}))
        return 1

    signing_key = _resolve_signing_key_b64(
        inline=args.signing_key,
        file_path=args.signing_key_file,
        env_name=args.signing_key_env,
    )

    records: list[RotationRecord] = []

    for environment in environments:
        purpose = getattr(environment, "credential_purpose", "trading")
        registry = RotationRegistry(_registry_path(environment))
        status = registry.status(
            environment.keychain_key,
            purpose,
            interval_days=args.interval_days,
            now=executed_at,
        )
        records.append(
            RotationRecord(
                environment=environment.name,
                key=environment.keychain_key,
                purpose=purpose,
                registry_path=_registry_path(environment),
                status_before=status,
                rotated_at=executed_at,
                interval_days=args.interval_days,
                metadata={"exchange": environment.exchange},
            )
        )
        if not args.dry_run:
            registry.mark_rotated(environment.keychain_key, purpose, timestamp=executed_at)

    summary = RotationSummary(
        operator=args.operator,
        executed_at=executed_at,
        records=records,
        notes=args.notes,
        metadata={"dry_run": bool(args.dry_run)},
    )

    output_path = _resolve_output_path_batch(
        args, executed_at=executed_at, environments=environments
    )
    path = write_rotation_summary(
        summary,
        output=output_path,
        signing_key=signing_key,
        signing_key_id=args.signing_key_id,
    )

    print(json.dumps(
        {
            "output": str(path),
            "signed": bool(signing_key),
            "environments": [env.name for env in environments],
            "dry_run": bool(args.dry_run),
        },
        ensure_ascii=False
    ))
    return 0

# =====================================================================
# Subcommand: plan  (main – plan/execute wg observability.key_rotation)
# =====================================================================

def _build_parser_plan(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "plan",
        help="Plan rotacji wg observability.key_rotation (+ opcjonalne execute)",
        description="Generuje plan rotacji i (opcjonalnie) oznacza due/overdue jako zrotowane.",
    )
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "core.yaml"),
                   help="Ścieżka do pliku konfiguracji core.yaml")
    p.add_argument("--registry-path", dest="registry_path",
                   help="Ścieżka rejestru rotacji (nadpisuje konfigurację)")
    p.add_argument("--output-dir", default=None,
                   help="Katalog docelowy planu (domyślnie z konfiguracji)")
    p.add_argument("--basename", default=None,
                   help="Nazwa bazowa pliku raportu (bez rozszerzeń)")
    p.add_argument("--execute", action="store_true",
                   help="Zapisz aktualizacje rotacji dla wpisów wymagających działania")
    p.add_argument("--interval-override", type=float, default=None,
                   help="Globalny override interwału (dni)")
    p.add_argument("--warn-within-override", type=float, default=None,
                   help="Globalny override progu ostrzeżeń (dni)")
    p.set_defaults(_handler=_handle_plan)
    return p

def _load_rotation_config(config_path: Path):
    core_config = load_core_config(config_path)
    observability = getattr(core_config, "observability", None)
    if observability is None or getattr(observability, "key_rotation", None) is None:
        raise ValueError("Konfiguracja nie zawiera sekcji observability.key_rotation")
    return observability.key_rotation

def _normalize_path(path: str | None, *, default: str, base: Path) -> Path:
    candidate = Path(path) if path else Path(default)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate

def _status_for_entry(
    registry: RotationRegistry,
    entry: Any,
    *,
    interval: float,
    warn_within: float,
    now: datetime,
) -> dict[str, object]:
    status = registry.status(entry.key, entry.purpose, interval_days=interval, now=now)
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

def _handle_plan(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    rotation_config = _load_rotation_config(config_path)
    registry_path = _normalize_path(
        args.registry_path or rotation_config.registry_path,
        default=rotation_config.registry_path,
        base=config_path.parent,
    )
    registry = RotationRegistry(registry_path)
    now = datetime.now(timezone.utc)

    output_root = Path(args.output_dir if args.output_dir else rotation_config.audit_directory)
    if not output_root.is_absolute():
        output_root = (config_path.parent / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    interval_override = args.interval_override
    warn_override = args.warn_within_override

    entries = list(rotation_config.entries)
    if not entries:
        raise ValueError("observability.key_rotation.entries jest puste")

    results: list[dict[str, object]] = []
    updated: list[dict[str, object]] = []

    for entry in entries:
        interval_days = interval_override or entry.interval_days or rotation_config.default_interval_days
        warn_days = warn_override or entry.warn_within_days or rotation_config.default_warn_within_days
        status_payload = _status_for_entry(
            registry,
            entry,
            interval=float(interval_days),
            warn_within=float(warn_days),
            now=now,
        )
        status_payload["entry"] = {
            "key": entry.key,
            "purpose": entry.purpose,
            "interval_days": float(interval_days),
            "warn_within_days": float(warn_days),
        }
        results.append(status_payload)
        if args.execute and status_payload["state"] in {"due", "overdue"}:
            registry.mark_rotated(entry.key, entry.purpose, timestamp=now)
            updated.append(status_payload["entry"])

    basename = args.basename or f"rotation_plan_{now.strftime('%Y%m%dT%H%M%SZ')}"
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

    print(f"Plan rotacji zapisany w {report_path}")
    if args.execute and updated:
        print("Zaktualizowano wpisy rotacji dla:")
        for entry in updated:
            print(f" - {entry['key']} ({entry['purpose']})")
    return 0

# =====================================================================
# Entrypoint
# =====================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage5 Key Rotation – merged CLI (batch + plan)"
    )
    sub = parser.add_subparsers(dest="_cmd", metavar="{batch|plan}", required=True)
    _build_parser_batch(sub)
    _build_parser_plan(sub)
    return parser

def run(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args._handler(args)

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
