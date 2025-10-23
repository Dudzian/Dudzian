#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage5 Key Rotation – merged CLI (HEAD + main)

Subcommands:
  - batch  : aktualizuje rejestr rotacji dla wskazanych środowisk i zapisuje
             podpisany raport Stage5 (HMAC Base64)
  - plan   : generuje plan rotacji wg observability.key_rotation i opcjonalnie
             oznacza klucze jako zrotowane (execute)
  - status : szybki podgląd rejestru rotacji dla bundla (np. mTLS `core-oem`)
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

try:  # PyYAML jest zależnością bot_core.config.loader
    import yaml
except Exception:  # pragma: no cover - fallback gdyby brakowało PyYAML
    yaml = None  # type: ignore[assignment]

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


def _maybe_getattr(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _resolve_config_path(value: str | Path | None, *, base: Path) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


def _bundle_key_candidates(bundle: str) -> list[str]:
    normalized = bundle.strip().lower()
    variants: list[str] = []
    for candidate in {
        normalized,
        normalized.replace("-", "_"),
        normalized.replace("_", "-"),
    }:
        if candidate and candidate not in variants:
            variants.append(candidate)
    if normalized.endswith("mtls"):
        suffix_stripped = normalized[: -4].rstrip("-_ ")
        if suffix_stripped and suffix_stripped not in variants:
            variants.append(suffix_stripped)
    else:
        for extra in (f"{normalized}-mtls", f"{normalized}_mtls"):
            if extra and extra not in variants:
                variants.append(extra)
    if "mtls" not in variants:
        variants.append("mtls")
    return variants


def _status_payload(
    registry: RotationRegistry,
    *,
    key: str,
    purpose: str,
    interval_days: float,
    warn_within_days: float,
    now: datetime,
) -> dict[str, object]:
    status = registry.status(key, purpose, interval_days=interval_days, now=now)
    payload = asdict(status)
    payload["key"] = key
    payload["purpose"] = purpose
    payload["state"] = "ok"
    if status.is_overdue:
        payload["state"] = "overdue"
    elif status.is_due:
        payload["state"] = "due"
    elif status.due_in_days <= warn_within_days:
        payload["state"] = "warning"
    payload["due_within_days"] = float(warn_within_days)
    last_rotated = payload.get("last_rotated")
    if isinstance(last_rotated, datetime):
        payload["last_rotated"] = last_rotated.isoformat().replace("+00:00", "Z")
    return payload

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
# Subcommand: status  (Stage5 L2 – szybki podgląd bundla)
# =====================================================================

def _extract_status_defaults(
    config: Any,
    *,
    raw: Mapping[str, Any] | None,
    bundle: str,
) -> tuple[Path | None, float | None, float | None]:
    raw_mapping = raw or {}

    execution = _maybe_getattr(config, "execution")
    if execution is None:
        execution = raw_mapping.get("execution") if isinstance(raw_mapping, Mapping) else None

    mtls = _maybe_getattr(execution, "mtls")
    if mtls is None and isinstance(execution, Mapping):
        mtls = execution.get("mtls")

    rotation_registry = _maybe_getattr(mtls, "rotation_registry") if mtls is not None else None
    if rotation_registry is None and isinstance(mtls, Mapping):
        rotation_registry = mtls.get("rotation_registry")

    observability = _maybe_getattr(config, "observability")
    if observability is None:
        observability = (
            raw_mapping.get("observability")
            if isinstance(raw_mapping, Mapping)
            else None
        )

    key_rotation = _maybe_getattr(observability, "key_rotation")
    if key_rotation is None and isinstance(observability, Mapping):
        key_rotation = observability.get("key_rotation")

    default_interval = (
        _maybe_getattr(key_rotation, "default_interval_days") if key_rotation else None
    )
    if default_interval is None and isinstance(key_rotation, Mapping):
        default_interval = key_rotation.get("default_interval_days")

    default_warn = (
        _maybe_getattr(key_rotation, "default_warn_within_days") if key_rotation else None
    )
    if default_warn is None and isinstance(key_rotation, Mapping):
        default_warn = key_rotation.get("default_warn_within_days")

    entries = _maybe_getattr(key_rotation, "entries") if key_rotation else None
    if entries:
        try:
            iterator = list(entries)
        except TypeError:
            iterator = []
        bundle_candidates = set(_bundle_key_candidates(bundle))
        for entry in iterator:
            entry_key = _maybe_getattr(entry, "key")
            if entry_key and str(entry_key).strip().lower() in bundle_candidates:
                entry_interval = _maybe_getattr(entry, "interval_days")
                entry_warn = _maybe_getattr(entry, "warn_within_days")
                if entry_interval is not None:
                    default_interval = entry_interval
                if entry_warn is not None:
                    default_warn = entry_warn
                break

    resolved_registry = Path(str(rotation_registry)) if rotation_registry else None
    if isinstance(rotation_registry, Path):
        resolved_registry = rotation_registry

    interval_value = float(default_interval) if default_interval is not None else None
    warn_value = float(default_warn) if default_warn is not None else None
    return resolved_registry, interval_value, warn_value


def _build_parser_status(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "status",
        help="Podsumowanie stanu rotacji dla bundla (np. mTLS)",
        description=(
            "Wyświetla czytelny raport rotacji dla wskazanego bundla – domyślnie "
            "korzysta z execution.mtls.rotation_registry i progów observability.key_rotation."
        ),
    )
    p.add_argument(
        "--bundle",
        dest="bundle_opt",
        help="Nazwa bundla do sprawdzenia (np. core-oem)",
    )
    p.add_argument(
        "bundle",
        nargs="?",
        metavar="BUNDLE",
        help="Nazwa bundla do sprawdzenia (np. core-oem)",
    )
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "core.yaml"),
        help="Ścieżka do CoreConfig używana do odczytu ustawień bundla",
    )
    p.add_argument(
        "--registry-path",
        help="Opcjonalna ścieżka rejestru rotacji (domyślnie z konfiguracji execution.mtls)",
    )
    p.add_argument(
        "--interval-days",
        type=float,
        help="Interwał rotacji (domyślnie z konfiguracji lub 90 dni)",
    )
    p.add_argument(
        "--warn-days",
        type=float,
        help="Próg ostrzeżenia (domyślnie z konfiguracji lub 14 dni)",
    )
    p.add_argument(
        "--as-of",
        help="Znacznik czasu referencyjny (ISO8601, domyślnie teraz)",
    )
    p.set_defaults(_handler=_handle_status)
    return p


def _handle_status(args: argparse.Namespace) -> int:
    bundle_flag = getattr(args, "bundle_opt", None)
    bundle_positional = getattr(args, "bundle", None)

    bundle_candidates = [value for value in (bundle_flag, bundle_positional) if value]
    normalized_candidates = [str(value).strip() for value in bundle_candidates if str(value).strip()]

    if bundle_flag and bundle_positional:
        if not normalized_candidates or len(set(normalized_candidates)) > 1:
            print(
                json.dumps(
                    {
                        "error": "Wartości podane przez --bundle i argument pozycyjny muszą być takie same.",
                    },
                    ensure_ascii=False,
                )
            )
            return 2

    bundle = normalized_candidates[0] if normalized_candidates else ""
    if not bundle:
        print(json.dumps({"error": "Parametr --bundle nie może być pusty."}, ensure_ascii=False))
        return 2

    config_path = Path(args.config)
    try:
        config = load_core_config(config_path)
    except FileNotFoundError:
        print(json.dumps({"error": f"Plik konfiguracji {config_path} nie istnieje"}, ensure_ascii=False))
        return 2

    raw_config: Mapping[str, Any] | None = None
    if yaml is not None:
        try:
            raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:  # pragma: no cover - brak dostępu / błędny YAML
            raw_config = None

    registry_path = None
    defaults_registry, default_interval, default_warn = _extract_status_defaults(
        config,
        raw=raw_config,
        bundle=bundle,
    )
    if args.registry_path:
        registry_path = _resolve_config_path(args.registry_path, base=config_path.parent)
    elif defaults_registry is not None:
        registry_path = _resolve_config_path(defaults_registry, base=config_path.parent)

    if registry_path is None:
        print(
            json.dumps(
                {"error": "Nie można wyznaczyć ścieżki rejestru rotacji (użyj --registry-path)."},
                ensure_ascii=False,
            )
        )
        return 2

    interval = float(args.interval_days) if args.interval_days is not None else default_interval or 90.0
    warn_within = float(args.warn_days) if args.warn_days is not None else default_warn or 14.0
    now = _parse_datetime(getattr(args, "as_of", None))

    registry = RotationRegistry(registry_path)
    candidate_keys = _bundle_key_candidates(bundle)
    candidate_set = set(candidate_keys)
    pairs: set[tuple[str, str]] = set()
    for key, purpose, _ in registry.entries():
        normalized_key = str(key).strip().lower()
        if normalized_key in candidate_set:
            pairs.add((normalized_key, str(purpose).strip().lower()))

    matches_found = bool(pairs)
    if not pairs:
        primary_key = candidate_keys[0]
        for purpose in ("ca", "server", "client"):
            pairs.add((primary_key, purpose))

    payloads = []
    for key, purpose in sorted(pairs):
        payloads.append(
            _status_payload(
                registry,
                key=key,
                purpose=purpose,
                interval_days=interval,
                warn_within_days=warn_within,
                now=now,
            )
        )

    summary = {"total": len(payloads), "ok": 0, "warning": 0, "due": 0, "overdue": 0}
    for payload in payloads:
        state = str(payload.get("state", "ok"))
        if state not in summary:
            summary[state] = 0
        summary[state] += 1

    report = {
        "bundle": bundle,
        "checked_at": now.isoformat().replace("+00:00", "Z"),
        "registry_path": str(registry_path),
        "interval_days": float(interval),
        "warn_within_days": float(warn_within),
        "entries_found": matches_found,
        "entries": payloads,
        "summary": summary,
        "matched_keys": sorted({payload["key"] for payload in payloads}),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0

# =====================================================================
# Entrypoint
# =====================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage5 Key Rotation – merged CLI (batch + plan + status)"
    )
    sub = parser.add_subparsers(dest="_cmd", metavar="{batch|plan|status}")
    _build_parser_batch(sub)
    _build_parser_plan(sub)
    _build_parser_status(sub)
    return parser


def _prepare_argv(argv: Sequence[str] | None) -> list[str]:
    source = argv if argv is not None else sys.argv[1:]
    args = list(source)
    if not args or args in [["-h"], ["--help"]]:
        return args
    if args[0] in {"batch", "plan", "status"}:
        return args
    if args[0].startswith("-"):
        normalized: list[str] = []
        saw_status = False
        status_bundle: str | None = None
        idx = 0
        while idx < len(args):
            arg = args[idx]
            if arg == "--status":
                saw_status = True
                next_value = args[idx + 1] if idx + 1 < len(args) else None
                if next_value and not next_value.startswith("-"):
                    status_bundle = next_value
                    idx += 1
                idx += 1
                continue
            if arg.startswith("--status="):
                saw_status = True
                value = arg.split("=", 1)[1]
                if value:
                    status_bundle = value
                idx += 1
                continue
            normalized.append(arg)
            idx += 1
        if saw_status:
            has_bundle_flag = any(
                item == "--bundle" or item.startswith("--bundle=") for item in normalized
            )
            if status_bundle and not has_bundle_flag:
                normalized = ["--bundle", status_bundle, *normalized]
            return ["status", *normalized]
        return ["batch", *args]
    return args


def run(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(_prepare_argv(argv))
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
