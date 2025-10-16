"""Rejestruje szkolenia Stage5 w decision logu zgodności."""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.compliance.training import TrainingSession, write_training_log


def _decode_key(value: str | None) -> bytes | None:
    if not value:
        return None
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:  # pragma: no cover - walidacja wejścia CLI
        raise SystemExit(f"Niepoprawny klucz HMAC: {exc}")


def _parse_list(value: str | None) -> list[str]:
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


def _parse_datetime(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:  # pragma: no cover - walidacja CLI
        raise SystemExit(f"Niepoprawna data ISO8601: {value!r} ({exc})")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Zapisuje raport szkoleniowy Stage5 do pliku JSON i opcjonalnie podpisuje go HMAC."
        )
    )
    parser.add_argument("session_id", help="Identyfikator szkolenia (np. data lub numer sesji)")
    parser.add_argument("title", help="Temat szkolenia")
    parser.add_argument("trainer", help="Prowadzący szkolenie")
    parser.add_argument(
        "--occurred-at",
        default=None,
        help="Znacznik czasu szkolenia w ISO8601 (domyślnie teraz)",
    )
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=90.0,
        help="Czas trwania szkolenia w minutach (domyślnie 90)",
    )
    parser.add_argument(
        "--participants",
        help="Lista uczestników rozdzielona przecinkami",
    )
    parser.add_argument(
        "--topics",
        help="Lista omawianych tematów rozdzielona przecinkami",
    )
    parser.add_argument(
        "--materials",
        help="Materiał referencyjny (CSV)",
    )
    parser.add_argument(
        "--compliance-tags",
        help="Tagi zgodności (CSV, np. stage5,aml)",
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Podsumowanie kluczowych wniosków",
    )
    parser.add_argument(
        "--action",
        action="append",
        default=[],
        help="Działanie następcze w formacie klucz:opis (można podać wiele razy)",
    )
    parser.add_argument(
        "--metadata",
        help="Dodatkowe metadane w formacie JSON",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Ścieżka docelowa raportu (domyślnie var/audit/stage5/training/<id>.json)",
    )
    parser.add_argument(
        "--signing-key",
        help="Klucz HMAC (Base64)",
    )
    parser.add_argument(
        "--signing-key-file",
        help="Plik z kluczem HMAC (Base64)",
    )
    parser.add_argument(
        "--signing-key-env",
        help="Zmienna środowiskowa zawierająca klucz HMAC (Base64)",
    )
    parser.add_argument(
        "--signing-key-id",
        help="Identyfikator klucza podpisującego",
    )
    return parser


def _resolve_key(args: argparse.Namespace) -> bytes | None:
    if args.signing_key:
        return _decode_key(args.signing_key)
    if args.signing_key_file:
        return _decode_key(Path(args.signing_key_file).read_text(encoding="utf-8"))
    if args.signing_key_env:
        value = os.environ.get(args.signing_key_env)
        if value:
            return _decode_key(value)
    return None


def _resolve_output(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    base = Path("var/audit/stage5/training")
    sanitized = args.session_id.replace("/", "-")
    filename = f"training_{sanitized}.json"
    return base / filename


def run(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    occurred_at = (
        _parse_datetime(args.occurred_at)
        if args.occurred_at
        else datetime.now(timezone.utc).replace(microsecond=0)
    )
    participants = _parse_list(args.participants)
    topics = _parse_list(args.topics)
    materials = _parse_list(args.materials)
    tags = _parse_list(args.compliance_tags)
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

    signing_key = _resolve_key(args)
    output = _resolve_output(args)
    path = write_training_log(
        session,
        output=output,
        signing_key=signing_key,
        signing_key_id=args.signing_key_id,
    )
    print(json.dumps({"output": str(path), "signed": bool(signing_key)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
