#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage5 Training – merged CLI (HEAD + main)

Subcommands:
  - report    : zapisuje podpisany raport szkolenia (pojedynczy JSON)
  - register  : rejestruje warsztaty w JSONL + (opcjonalnie) decision logu
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- konfiguracja logowania przed importami modułów repo ---
logging.getLogger("KryptoLowca.ai_models").disabled = True

# --- HEAD branch API (pojedynczy raport + podpis Base64) ---
from bot_core.compliance.training import TrainingSession, write_training_log

# --- main branch utils (JSONL, decision log, artefakty, HMAC) ---
from bot_core.security.signing import build_hmac_signature
from deploy.packaging.build_core_bundle import (  # type: ignore
    _ensure_casefold_safe_tree,
    _ensure_no_symlinks,
    _ensure_windows_safe_tree,
)

from scripts._cli_common import now_iso

# =====================================================================
# Subcommand: report (HEAD)
# =====================================================================

def _decode_key_b64(value: str | None) -> bytes | None:
    if not value:
        return None
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Niepoprawny klucz HMAC (Base64): {exc}")

def _parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]

def _parse_actions(values: Sequence[str]) -> dict[str, str]:
    actions: dict[str, str] = {}
    for value in values:
        if ":" in value:
            key, _, rest = value.partition(":")
            actions[key.strip()] = rest.strip()
        else:
            idx = len(actions) + 1
            actions[f"action_{idx}"] = value.strip()
    return actions

def _parse_dt_iso(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:  # pragma: no cover
        raise SystemExit(f"Niepoprawna data ISO8601: {value!r} ({exc})")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _resolve_key_report(args: argparse.Namespace) -> bytes | None:
    if args.signing_key:
        return _decode_key_b64(args.signing_key)
    if args.signing_key_file:
        return _decode_key_b64(Path(args.signing_key_file).read_text(encoding="utf-8"))
    if args.signing_key_env:
        value = os.environ.get(args.signing_key_env)
        if value:
            return _decode_key_b64(value)
    return None

def _resolve_output_report(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    base = Path("var/audit/stage5/training")
    sanitized = args.session_id.replace("/", "-")
    filename = f"training_{sanitized}.json"
    return base / filename

def _build_parser_report(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "report",
        help="Zapis podpisanego raportu szkolenia (pojedynczy JSON)",
        description="Zapisuje raport szkoleniowy Stage5 i opcjonalnie podpisuje go HMAC (Base64).",
    )
    p.add_argument("session_id", help="Identyfikator szkolenia (np. data lub numer sesji)")
    p.add_argument("title", help="Temat szkolenia")
    p.add_argument("trainer", help="Prowadzący szkolenie")
    p.add_argument("--occurred-at", default=None, help="Znacznik czasu szkolenia w ISO8601 (domyślnie teraz)")
    p.add_argument("--duration-minutes", type=float, default=90.0, help="Czas trwania (domyślnie 90)")
    p.add_argument("--participants", help="Lista uczestników (CSV)")
    p.add_argument("--topics", help="Lista tematów (CSV)")
    p.add_argument("--materials", help="Materiały referencyjne (CSV)")
    p.add_argument("--compliance-tags", help="Tagi zgodności (CSV, np. stage5,aml)")
    p.add_argument("--summary", required=True, help="Podsumowanie kluczowych wniosków")
    p.add_argument("--action", action="append", default=[], help="Działanie następcze w formacie klucz:opis (wiele razy)")
    p.add_argument("--metadata", help="Dodatkowe metadane w formacie JSON")
    p.add_argument("--output", default=None, help="Ścieżka docelowa raportu (domyślna wg session_id)")
    p.add_argument("--signing-key", help="Klucz HMAC (Base64)")
    p.add_argument("--signing-key-file", help="Plik z kluczem HMAC (Base64)")
    p.add_argument("--signing-key-env", help="Zmienna środowiskowa z kluczem HMAC (Base64)")
    p.add_argument("--signing-key-id", help="Identyfikator klucza podpisującego")
    p.set_defaults(_handler=_handle_report)
    return p

def _handle_report(args: argparse.Namespace) -> int:
    occurred_at = (
        _parse_dt_iso(args.occurred_at)
        if args.occurred_at
        else datetime.now(timezone.utc).replace(microsecond=0)
    )
    participants = _parse_csv_list(args.participants)
    topics = _parse_csv_list(args.topics)
    materials = _parse_csv_list(args.materials)
    tags = _parse_csv_list(args.compliance_tags)
    actions = _parse_actions(args.action)
    metadata = json.loads(args.metadata) if args.metadata else {}

    session = TrainingSession(
        session_id=args.session_id,
        title=args.title,
        trainer=args.trainer,
        participants=participants,
        topics=topics,
        occurred_at=occurred_at,
        duration_minutes=args.duration_minutes,
        summary=args.summary,
        actions=actions,
        materials=materials,
        compliance_tags=tags,
        metadata=metadata,
    )

    signing_key = _resolve_key_report(args)
    output = _resolve_output_report(args)
    path = write_training_log(
        session,
        output=output,
        signing_key=signing_key,
        signing_key_id=args.signing_key_id,
    )
    print(json.dumps({"output": str(path), "signed": bool(signing_key)}, ensure_ascii=False))
    return 0


# =====================================================================
# Subcommand: register (main)
# =====================================================================

DEFAULT_LOG_PATH = REPO_ROOT / "var/audit/training/stage5_training_log.jsonl"

def _parse_args_register(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "register",
        help="Rejestruje warsztaty (JSONL + opcjonalny decision log)",
        description="Rejestracja warsztatów Stage5 – log szkoleniowy JSONL oraz wpis w decision logu (z HMAC).",
    )
    p.add_argument("--log-path", default=str(DEFAULT_LOG_PATH), help="Plik JSONL z logiem szkoleń Stage5")
    p.add_argument("--training-date", help="Data szkolenia (YYYY-MM-DD); domyślnie dzisiejsza data UTC")
    p.add_argument("--start-time", help="Godzina rozpoczęcia (HH:MM, czas lokalny)")
    p.add_argument("--duration-minutes", type=int, default=210, help="Czas trwania w minutach (domyślnie 210)")
    p.add_argument("--session-id", help="Niestandardowy identyfikator sesji")

    p.add_argument("--facilitator", required=True, help="Prowadzący warsztat")
    p.add_argument("--location", required=True, help="Lokalizacja (np. sala, call)")
    p.add_argument("--participant", action="append", dest="participants", default=[], help="Uczestnik (wiele razy)")
    p.add_argument("--topic", action="append", dest="topics", default=[], help="Temat/sekcja (wiele razy)")
    p.add_argument("--material", action="append", dest="materials", default=[], help="Materiał (wiele razy)")
    p.add_argument("--artifact", action="append", dest="artifacts", default=[], help="Ścieżka artefaktu (wiele razy)")
    p.add_argument("--notes", help="Dodatkowe uwagi/wnioski")

    p.add_argument("--log-hmac-key", help="Klucz HMAC (inline) do podpisania wpisu logu szkoleniowego")
    p.add_argument("--log-hmac-key-file", help="Plik z kluczem HMAC do podpisania wpisu logu szkoleniowego")

    p.add_argument("--decision-log-path", help="Ścieżka do decision logu JSONL")
    p.add_argument("--decision-log-hmac-key", help="Klucz HMAC (inline) dla decision logu")
    p.add_argument("--decision-log-hmac-key-file", help="Plik z kluczem HMAC dla decision logu")
    p.add_argument("--decision-log-key-id", help="Id klucza użytego do podpisu wpisu decision logu")
    p.add_argument("--decision-log-category", default="stage5_training", help="Kategoria decision logu")
    p.add_argument("--decision-log-notes", help="Notatka dołączana do wpisu decision logu")
    p.add_argument("--decision-log-allow-unsigned", action="store_true", help="Pozwól na wpis bez HMAC")

    p.add_argument("--print-entry", action="store_true", help="Wypisz wygenerowany wpis na stdout")
    p.set_defaults(_handler=_handle_register)
    return p

def _parse_date(value: str | None) -> str:
    if not value:
        return _dt.datetime.utcnow().date().isoformat()
    try:
        parsed = _dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:  # pragma: no cover
        raise ValueError("Data szkolenia musi mieć format YYYY-MM-DD") from exc
    return parsed.isoformat()

def _parse_time(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        _dt.datetime.strptime(value, "%H:%M")
    except ValueError as exc:
        raise ValueError("Godzina rozpoczęcia musi mieć format HH:MM") from exc
    return value

def _normalize_unique(items: Iterable[str], *, label: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in items:
        item = raw.strip()
        if not item:
            raise ValueError(f"{label} nie może być pusty")
        key = item.casefold()
        if key in seen:
            raise ValueError(f"{label} zawiera duplikat: {item}")
        seen.add(key)
        normalized.append(item)
    return normalized

def _sha256_digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}

def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)

def _prepare_artifacts(paths: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    artifacts: dict[str, Mapping[str, Any]] = {}
    for raw in paths:
        candidate = Path(raw).expanduser()
        _ensure_no_symlinks(candidate, label="Artefakt szkoleniowy")
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise ValueError(f"Artefakt nie jest plikiem: {resolved}")
        _ensure_windows_safe_tree(resolved, label="Artefakt szkoleniowy")
        label = _relative_or_absolute(resolved)
        if label in artifacts:
            raise ValueError(f"Zduplikowany artefakt: {label}")
        artifacts[label] = _sha256_digest(resolved)
    return artifacts

def _load_hmac_key_inline_or_file(inline: str | None, file_path: str | None, *, label: str) -> bytes | None:
    if inline and file_path:
        raise ValueError(f"Podaj klucz {label} inline lub jako plik – nie jednocześnie")
    if inline:
        key = inline.encode("utf-8")
    elif file_path:
        candidate = Path(file_path).expanduser()
        _ensure_no_symlinks(candidate, label=f"Klucz {label}")
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise ValueError(f"Plik klucza {label} nie istnieje: {resolved}")
        if os.name != "nt":
            mode = resolved.stat().st_mode
            if mode & 0o077:
                raise ValueError(f"Plik klucza {label} musi mieć uprawnienia maks. 600")
        key = resolved.read_bytes()
    else:
        return None
    if len(key) < 32:
        raise ValueError(f"Klucz {label} musi mieć co najmniej 32 bajty")
    return key

def _prepare_log_path(path: str) -> Path:
    log_path = Path(path).expanduser()
    _ensure_no_symlinks(log_path, label="Plik logu szkoleniowego")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_casefold_safe_tree(log_path.parent, label="Katalog logu szkoleniowego")
    _ensure_windows_safe_tree(log_path, label="Plik logu szkoleniowego")
    return log_path.resolve()

def _derive_session_id(*, explicit: str | None, training_date: str, facilitator: str, recorded_at: str) -> str:
    if explicit:
        cleaned = explicit.strip()
        if not cleaned:
            raise ValueError("Identyfikator sesji nie może być pusty")
        return cleaned
    digest = hashlib.sha256()
    digest.update(training_date.encode("utf-8"))
    digest.update(facilitator.encode("utf-8"))
    digest.update(recorded_at.encode("utf-8"))
    return f"stage5-training-{digest.hexdigest()[:12]}"

def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

def _append_decision_log(
    *,
    path: str | None,
    payload: Mapping[str, Any],
    key: bytes | None,
    key_id: str | None,
    allow_unsigned: bool,
) -> None:
    if not path:
        if key and not allow_unsigned:
            raise ValueError("Wpis decision logu wymaga --decision-log-path gdy podano klucz HMAC")
        return

    entry = dict(payload)
    if key is not None:
        entry["signature"] = build_hmac_signature(
            payload,
            key=key,
            algorithm="HMAC-SHA256",
            key_id=key_id,
        )
    elif not allow_unsigned:
        raise ValueError(
            "Brak klucza decision logu – podaj --decision-log-hmac-key lub --decision-log-hmac-key-file"
        )

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    _ensure_no_symlinks(target, label="Decision log")
    _ensure_windows_safe_tree(target, label="Decision log")
    _append_jsonl(target, entry)

def _handle_register(args: argparse.Namespace) -> int:
    training_date = _parse_date(args.training_date)
    start_time = _parse_time(args.start_time)
    facilitator = args.facilitator.strip()
    if not facilitator:
        raise ValueError("Prowadzący nie może być pusty")
    location = args.location.strip()
    if not location:
        raise ValueError("Lokalizacja nie może być pusta")

    participants = _normalize_unique(args.participants, label="Lista uczestników")
    if not participants:
        raise ValueError("Podaj co najmniej jednego uczestnika")

    topics = _normalize_unique(args.topics, label="Tematy warsztatu") if args.topics else []
    materials = _normalize_unique(args.materials, label="Materiały warsztatowe") if args.materials else []
    artifacts = _prepare_artifacts(args.artifacts)

    recorded_at = now_iso()
    session_id = _derive_session_id(
        explicit=args.session_id,
        training_date=training_date,
        facilitator=facilitator,
        recorded_at=recorded_at,
    )

    log_key = _load_hmac_key_inline_or_file(args.log_hmac_key, args.log_hmac_key_file, label="logu szkoleniowego")
    decision_key = _load_hmac_key_inline_or_file(
        args.decision_log_hmac_key,
        args.decision_log_hmac_key_file,
        label="decision logu",
    )

    entry_payload: dict[str, Any] = {
        "schema": "stage5.training_log",
        "recorded_at": recorded_at,
        "training_date": training_date,
        "start_time": start_time,
        "duration_minutes": args.duration_minutes,
        "session_id": session_id,
        "facilitator": facilitator,
        "location": location,
        "participants": participants,
        "topics": topics,
        "materials": materials,
        "artifacts": artifacts,
        "notes": args.notes or "",
    }
    entry_payload = {k: v for k, v in entry_payload.items() if v not in (None, [], "")}

    if log_key is not None:
        entry_with_sig = dict(entry_payload)
        entry_with_sig["signature"] = build_hmac_signature(
            entry_payload,
            key=log_key,
            algorithm="HMAC-SHA256",
        )
    else:
        entry_with_sig = entry_payload

    log_path = _prepare_log_path(args.log_path)
    _append_jsonl(log_path, entry_with_sig)

    decision_payload: dict[str, Any] = {
        "schema": "stage5.training_session",
        "recorded_at": recorded_at,
        "training_date": training_date,
        "session_id": session_id,
        "category": args.decision_log_category,
        "facilitator": facilitator,
        "location": location,
        "participants": participants,
        "topics": topics,
        "materials": materials,
        "artifacts": artifacts,
        "duration_minutes": args.duration_minutes,
        "notes": args.decision_log_notes or args.notes or "",
    }
    decision_payload = {k: v for k, v in decision_payload.items() if v not in (None, [], "")}

    _append_decision_log(
        path=args.decision_log_path,
        payload=decision_payload,
        key=decision_key,
        key_id=args.decision_log_key_id,
        allow_unsigned=args.decision_log_allow_unsigned,
    )

    if args.print_entry:
        print(json.dumps(entry_with_sig, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps({"log_path": str(log_path), "session_id": session_id}, ensure_ascii=False))
    return 0


# =====================================================================
# Entrypoint
# =====================================================================

def _build_parser() -> tuple[argparse.ArgumentParser, frozenset[str]]:
    parser = argparse.ArgumentParser(
        description="Stage5 Training – merged CLI (report + register)"
    )
    sub = parser.add_subparsers(dest="_cmd", metavar="{report|register}", required=True)
    _build_parser_report(sub)
    _parse_args_register(sub)
    return parser, frozenset(sub.choices.keys())


HELP_FLAGS = {"-h", "--help"}


def _prepare_argv(argv: Sequence[str] | None, known_commands: Collection[str]) -> list[str]:
    source = sys.argv[1:] if argv is None else argv
    args = list(source)
    if not args:
        return args

    first = args[0]
    if first in known_commands or first in HELP_FLAGS:
        return args

    if first.startswith("-"):
        return ["register", *args]

    return ["report", *args]

def _execute_cli(argv: Sequence[str] | None = None) -> int:
    parser, commands = _build_parser()
    args = _prepare_argv(argv, commands)
    try:
        parsed = parser.parse_args(args)
    except SystemExit as exc:  # pragma: no cover - argument errors map to exit codes
        return int(exc.code)
    return parsed._handler(parsed)

def run(argv: Sequence[str] | None = None) -> int:
    """Programmatic entry point – zachowuje dotychczasowe API."""

    return _execute_cli(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    return _execute_cli(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
