"""Synchronizuje dziennik JSONL smoke testów do magazynu audytowego."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from bot_core.config import load_core_config
from bot_core.reporting.audit import PaperSmokeJsonSynchronizer
from bot_core.security import SecretManager, SecretStorageError
from scripts._cli_common import create_secret_manager

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _SyncResult:
    status: str
    backend: str | None = None
    location: str | None = None
    metadata: Mapping[str, str] | None = None
    record_id: str | None = None
    timestamp: str | None = None
    json_log_path: str | None = None
    error: str | None = None


def _create_secret_manager(args: argparse.Namespace) -> SecretManager:
    return create_secret_manager(
        namespace=args.secret_namespace,
        headless_passphrase=args.headless_passphrase,
        headless_path=args.headless_storage,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronizuje dziennik JSONL smoke testów z magazynem audytowym.",
    )
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do CoreConfig")
    parser.add_argument(
        "--environment",
        required=True,
        help="Nazwa środowiska (zgodnie z sekcją environments w CoreConfig)",
    )
    parser.add_argument(
        "--json-log",
        required=True,
        help="Ścieżka do lokalnego pliku JSONL z raportami smoke testów",
    )
    parser.add_argument(
        "--record-id",
        default=None,
        help="Identyfikator rekordu do użycia przy synchronizacji (domyślnie ostatni wpis)",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Znacznik czasu ISO8601 użyty przy synchronizacji (domyślnie ostatni wpis lub teraz)",
    )
    parser.add_argument("--json", action="store_true", help="Zwróć wynik w formacie JSON")
    parser.add_argument("--dry-run", action="store_true", help="Tylko waliduj parametry, bez synchronizacji")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Poziom logowania",
    )
    parser.add_argument(
        "--secret-namespace",
        default="dudzian.trading",
        help="Namespace używany do odczytu sekretów (keychain / zaszyfrowany plik)",
    )
    parser.add_argument(
        "--headless-passphrase",
        default=None,
        help="Hasło do magazynu sekretów w trybie headless (Linux)",
    )
    parser.add_argument(
        "--headless-storage",
        default=None,
        help="Ścieżka zaszyfrowanego magazynu sekretów dla trybu headless",
    )
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))


def _parse_timestamp_arg(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_last_record(json_log_path: Path) -> Mapping[str, object] | None:
    if not json_log_path.exists():
        return None
    try:
        with json_log_path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except Exception:  # noqa: BLE001
        _LOGGER.exception("Nie udało się wczytać dziennika JSONL: %s", json_log_path)
        return None
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        _LOGGER.warning(
            "Ostatnia linia dziennika JSONL ma nieprawidłowy format JSON (zignorowano)"
        )
        return None


def _serialize_result(result: _SyncResult, *, as_json: bool) -> str:
    if as_json:
        payload: MutableMapping[str, object] = {
            "status": result.status,
        }
        if result.backend:
            payload["backend"] = result.backend
        if result.location:
            payload["location"] = result.location
        if result.metadata is not None:
            payload["metadata"] = dict(result.metadata)
        if result.record_id:
            payload["record_id"] = result.record_id
        if result.timestamp:
            payload["timestamp"] = result.timestamp
        if result.json_log_path:
            payload["json_log_path"] = result.json_log_path
        if result.error:
            payload["error"] = result.error
        return json.dumps(payload, ensure_ascii=False)

    if result.status == "ok":
        lines = [
            "Synchronizacja zakończona pomyślnie:",
            f"  Backend: {result.backend}",
            f"  Lokalizacja: {result.location}",
        ]
        if result.record_id:
            lines.append(f"  Rekord: {result.record_id}")
        if result.timestamp:
            lines.append(f"  Znacznik czasu: {result.timestamp}")
        if result.metadata:
            lines.append("  Metadane:")
            for key, value in sorted(result.metadata.items()):
                lines.append(f"    - {key}: {value}")
        return "\n".join(lines)

    if result.status == "skipped":
        return "Synchronizacja pominięta (dry-run)."

    error_text = result.error or "Nieznany błąd"
    return f"Synchronizacja nie powiodła się: {error_text}"


def _build_success_result(
    sync_result,
    *,
    record_id: str | None,
    timestamp: datetime,
    json_log_path: Path,
) -> _SyncResult:
    return _SyncResult(
        status="ok",
        backend=getattr(sync_result, "backend", None),
        location=getattr(sync_result, "location", None),
        metadata=getattr(sync_result, "metadata", None),
        record_id=record_id,
        timestamp=timestamp.astimezone(timezone.utc).isoformat(),
        json_log_path=str(json_log_path),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    json_log_path = Path(args.json_log)
    if not json_log_path.exists():
        _LOGGER.error("Plik dziennika JSONL nie istnieje: %s", json_log_path)
        result = _SyncResult(
            status="error",
            error="missing_json_log",
            json_log_path=str(json_log_path),
        )
        print(_serialize_result(result, as_json=args.json))
        return 1

    try:
        config = load_core_config(args.config)
    except FileNotFoundError:
        _LOGGER.error("Nie znaleziono pliku konfiguracji: %s", args.config)
        result = _SyncResult(status="error", error="missing_config")
        print(_serialize_result(result, as_json=args.json))
        return 2

    reporting_cfg = getattr(config, "reporting", None)
    json_sync_cfg = PaperSmokeJsonSynchronizer.resolve_config(reporting_cfg)
    if json_sync_cfg is None:
        _LOGGER.error("Konfiguracja reporting.paper_smoke_json_sync nie została zdefiniowana")
        result = _SyncResult(status="error", error="missing_sync_config")
        print(_serialize_result(result, as_json=args.json))
        return 2

    record_data = _load_last_record(json_log_path)
    record_id = args.record_id or (str(record_data.get("record_id")) if record_data else None)

    timestamp = _parse_timestamp_arg(args.timestamp)
    if timestamp is None and isinstance(record_data, Mapping):
        ts_value = record_data.get("timestamp")
        if isinstance(ts_value, str):
            try:
                timestamp = _parse_timestamp_arg(ts_value)
            except ValueError:
                timestamp = None
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    if args.dry_run:
        _LOGGER.info(
            "Dry-run: pominieto synchronizację JSONL (backend=%s, record_id=%s)",
            json_sync_cfg.backend,
            record_id,
        )
        result = _SyncResult(
            status="skipped",
            backend=json_sync_cfg.backend,
            record_id=record_id,
            timestamp=timestamp.isoformat(),
            json_log_path=str(json_log_path),
        )
        print(_serialize_result(result, as_json=args.json))
        return 0

    secret_manager: SecretManager | None = None
    try:
        if json_sync_cfg.backend.lower() == "s3":
            secret_manager = _create_secret_manager(args)
    except SecretStorageError as exc:
        _LOGGER.error("Błąd podczas inicjalizacji SecretManager: %s", exc)
        result = _SyncResult(status="error", error="secret_manager_error")
        print(_serialize_result(result, as_json=args.json))
        return 3

    try:
        synchronizer = PaperSmokeJsonSynchronizer(
            json_sync_cfg,
            secret_manager=secret_manager,
        )
        sync_result = synchronizer.sync(
            json_log_path,
            environment=args.environment,
            record_id=record_id or "",
            timestamp=timestamp,
        )
    except (FileNotFoundError, SecretStorageError) as exc:
        _LOGGER.error("Błąd krytyczny synchronizacji JSONL: %s", exc)
        result = _SyncResult(status="error", error=str(exc))
        print(_serialize_result(result, as_json=args.json))
        return 1
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("Nieoczekiwany błąd synchronizacji JSONL")
        result = _SyncResult(status="error", error=str(exc))
        print(_serialize_result(result, as_json=args.json))
        return 1

    success = _build_success_result(
        sync_result,
        record_id=record_id,
        timestamp=timestamp,
        json_log_path=json_log_path,
    )
    print(_serialize_result(success, as_json=args.json))
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())
