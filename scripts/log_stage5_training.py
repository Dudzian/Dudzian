"""Rejestracja warsztatów Stage5 wraz z podpisanym logiem i wpisem decision logu."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.security.signing import build_hmac_signature
from deploy.packaging.build_core_bundle import (  # type: ignore
    _ensure_casefold_safe_tree,
    _ensure_no_symlinks,
    _ensure_windows_safe_tree,
)


DEFAULT_LOG_PATH = REPO_ROOT / "var/audit/training/stage5_training_log.jsonl"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rejestruje warsztaty Stage5 – zapisuje log szkolenia oraz wpis w decision logu."
        )
    )

    parser.add_argument(
        "--log-path",
        default=str(DEFAULT_LOG_PATH),
        help="Ścieżka do pliku JSONL z logiem szkoleń Stage5",
    )
    parser.add_argument(
        "--training-date",
        help="Data szkolenia (YYYY-MM-DD); domyślnie dzisiejsza data w UTC",
    )
    parser.add_argument(
        "--start-time",
        help="Opcjonalna godzina rozpoczęcia szkolenia (HH:MM, czas lokalny)",
    )
    parser.add_argument(
        "--duration-minutes",
        type=int,
        default=210,
        help="Czas trwania szkolenia w minutach (domyślnie 210 = 3,5 h)",
    )
    parser.add_argument("--session-id", help="Niestandardowy identyfikator sesji szkoleniowej")

    parser.add_argument("--facilitator", required=True, help="Prowadzący warsztat")
    parser.add_argument("--location", required=True, help="Lokalizacja (np. sala, call)")
    parser.add_argument(
        "--participant",
        action="append",
        dest="participants",
        default=[],
        help="Uczestnik warsztatu (podaj wiele razy)",
    )
    parser.add_argument(
        "--topic",
        action="append",
        dest="topics",
        default=[],
        help="Dodatkowy temat/sekcja przerobiona podczas warsztatu",
    )
    parser.add_argument(
        "--material",
        action="append",
        dest="materials",
        default=[],
        help="Materiał udostępniony uczestnikom (np. link, dokument)",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        dest="artifacts",
        default=[],
        help="Ścieżka do artefaktu (np. PDF, nagranie) dołączonego do logu",
    )
    parser.add_argument("--notes", help="Dodatkowe uwagi lub wnioski po warsztacie")

    parser.add_argument(
        "--log-hmac-key",
        help="Klucz HMAC (inline) do podpisania wpisu w logu szkoleniowym",
    )
    parser.add_argument(
        "--log-hmac-key-file",
        help="Plik z kluczem HMAC do podpisania wpisu w logu szkoleniowym",
    )

    parser.add_argument("--decision-log-path", help="Ścieżka do decision logu JSONL")
    parser.add_argument(
        "--decision-log-hmac-key",
        help="Klucz HMAC (inline) do podpisania wpisu decision logu",
    )
    parser.add_argument(
        "--decision-log-hmac-key-file",
        help="Plik z kluczem HMAC decision logu",
    )
    parser.add_argument(
        "--decision-log-key-id",
        help="Identyfikator klucza użytego do podpisania wpisu decision logu",
    )
    parser.add_argument(
        "--decision-log-category",
        default="stage5_training",
        help="Kategoria wpisu decision logu (domyślnie stage5_training)",
    )
    parser.add_argument(
        "--decision-log-notes",
        help="Notatka dołączana do wpisu decision logu",
    )
    parser.add_argument(
        "--decision-log-allow-unsigned",
        action="store_true",
        help="Pozwól na wpis decision logu bez podpisu HMAC",
    )

    parser.add_argument(
        "--print-entry",
        action="store_true",
        help="Wyświetl wygenerowany wpis na stdout",
    )

    return parser.parse_args(argv)


def _parse_date(value: str | None) -> str:
    if not value:
        return _dt.datetime.utcnow().date().isoformat()
    try:
        parsed = _dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:  # pragma: no cover - defensive
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


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_hmac_key(inline: str | None, file_path: str | None, *, label: str) -> bytes | None:
    if inline and file_path:
        raise ValueError(
            f"Podaj klucz {label} jako parametr inline lub plik – nie jednocześnie"
        )
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
                raise ValueError(
                    f"Plik klucza {label} musi mieć uprawnienia maks. 600"
                )
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


def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _derive_session_id(
    *,
    explicit: str | None,
    training_date: str,
    facilitator: str,
    recorded_at: str,
) -> str:
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
            return
        if allow_unsigned:
            return
        raise ValueError("Wpis decision logu jest wymagany – podaj --decision-log-path")

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


def run(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

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

    recorded_at = _now_iso()
    session_id = _derive_session_id(
        explicit=args.session_id,
        training_date=training_date,
        facilitator=facilitator,
        recorded_at=recorded_at,
    )

    log_key = _load_hmac_key(args.log_hmac_key, args.log_hmac_key_file, label="logu szkoleniowego")
    decision_key = _load_hmac_key(
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
    # Usuń puste pola, aby podpis obejmował tylko wypełnione informacje
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

    return 0


def main() -> None:  # pragma: no cover - deleguje do run()
    sys.exit(run())


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    main()
