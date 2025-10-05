"""Automatyzuje publikację artefaktów smoke testu paper tradingu."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from bot_core.config import load_core_config
from bot_core.reporting.audit import PaperSmokeJsonSynchronizer
from bot_core.reporting.upload import SmokeArchiveUploader
from bot_core.security import (
    SecretManager,
    SecretStorageError,
    create_default_secret_storage,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _StepOutcome:
    status: str
    backend: str | None = None
    location: str | None = None
    metadata: Mapping[str, str] | None = None
    reason: str | None = None
    record_id: str | None = None
    timestamp: str | None = None


@dataclass(slots=True)
class _PublishResult:
    status: str
    summary_sha256: str | None
    report_dir: str
    json_sync: _StepOutcome
    archive_upload: _StepOutcome


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publikuje dziennik JSONL i archiwum smoke testu do magazynów audytowych.",
    )
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do CoreConfig")
    parser.add_argument(
        "--environment",
        required=True,
        help="Nazwa środowiska (zgodnie z sekcją environments w CoreConfig)",
    )
    parser.add_argument(
        "--report-dir",
        required=True,
        help="Ścieżka katalogu z artefaktami smoke testu (summary.json, ledger.jsonl, itp.)",
    )
    parser.add_argument(
        "--json-log",
        default="docs/audit/paper_trading_log.jsonl",
        help="Lokalny plik JSONL z wpisami smoke testów (domyślnie w repozytorium audytu)",
    )
    parser.add_argument(
        "--record-id",
        default=None,
        help="Identyfikator rekordu JSONL (domyślnie dopasowanie po hash summary)",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help=(
            "Ścieżka istniejącego archiwum ZIP smoke testu; jeśli brak, zostanie wygenerowane z katalogu raportu."
        ),
    )
    parser.add_argument(
        "--skip-json-sync",
        action="store_true",
        help="Pomiń synchronizację dziennika JSONL (np. gdy wykonano ją w innym kroku)",
    )
    parser.add_argument(
        "--skip-archive-upload",
        action="store_true",
        help="Pomiń wysyłkę archiwum smoke testu",
    )
    parser.add_argument("--dry-run", action="store_true", help="Tylko waliduj parametry, bez publikacji")
    parser.add_argument("--json", action="store_true", help="Zwróć wynik w formacie JSON")
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


def _create_secret_manager(args: argparse.Namespace) -> SecretManager:
    storage = create_default_secret_storage(
        namespace=args.secret_namespace,
        headless_passphrase=args.headless_passphrase,
        headless_path=args.headless_storage,
    )
    return SecretManager(storage, namespace=args.secret_namespace)


def _read_summary(report_dir: Path) -> tuple[Mapping[str, Any], str, Path]:
    summary_path = report_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    payload = summary_path.read_bytes()
    summary_sha = hashlib.sha256(payload).hexdigest()
    summary_data = json.loads(payload.decode("utf-8"))
    if not isinstance(summary_data, Mapping):
        raise ValueError("Nieprawidłowy format summary.json (oczekiwano obiektu JSON)")
    return summary_data, summary_sha, summary_path


def _ensure_archive(report_dir: Path, archive_arg: str | None) -> Path | None:
    if archive_arg:
        archive_path = Path(archive_arg)
        return archive_path if archive_path.exists() else None
    default_path = report_dir.with_suffix(".zip")
    if default_path.exists():
        return default_path
    try:
        archive_path_str = shutil.make_archive(str(report_dir), "zip", root_dir=report_dir)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.error("Nie udało się utworzyć archiwum smoke testu: %s", exc)
        return None
    return Path(archive_path_str)


def _load_json_records(json_log_path: Path) -> list[Mapping[str, Any]]:
    if not json_log_path.exists():
        return []
    records: list[Mapping[str, Any]] = []
    try:
        with json_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    _LOGGER.warning(
                        "Pomijam wpis JSONL o niepoprawnym formacie w %s", json_log_path
                    )
                    continue
                if isinstance(record, Mapping):
                    records.append(record)
    except Exception:  # noqa: BLE001
        _LOGGER.exception("Nie udało się odczytać dziennika JSONL: %s", json_log_path)
    return records


def _find_record(
    *,
    records: Sequence[Mapping[str, Any]],
    environment: str,
    record_id: str | None,
    summary_sha256: str | None,
) -> Mapping[str, Any] | None:
    def _matches(record: Mapping[str, Any]) -> bool:
        env_value = str(record.get("environment", "")).lower()
        if env_value and env_value != environment.lower():
            return False
        return True

    if record_id:
        for record in reversed(records):
            if str(record.get("record_id")) == record_id and _matches(record):
                return record

    if summary_sha256:
        for record in reversed(records):
            if str(record.get("summary_sha256")) == summary_sha256 and _matches(record):
                return record

    for record in reversed(records):
        if _matches(record):
            return record
    return None


def _serialize_result(result: _PublishResult, *, as_json: bool) -> str:
    json_payload = {
        "status": result.status,
        "summary_sha256": result.summary_sha256,
        "report_dir": result.report_dir,
        "json_sync": _step_to_dict(result.json_sync),
        "archive_upload": _step_to_dict(result.archive_upload),
    }
    if as_json:
        return json.dumps(json_payload, ensure_ascii=False)

    lines = [
        f"Status publikacji: {result.status}",
        f"Katalog raportu: {result.report_dir}",
        f"SHA-256 summary.json: {result.summary_sha256 or '-'}",
    ]
    lines.extend(_format_step("JSON sync", result.json_sync))
    lines.extend(_format_step("Archiwum", result.archive_upload))
    return "\n".join(lines)


def _step_to_dict(step: _StepOutcome) -> Mapping[str, Any]:
    payload: MutableMapping[str, Any] = {"status": step.status}
    if step.backend:
        payload["backend"] = step.backend
    if step.location:
        payload["location"] = step.location
    if step.metadata is not None:
        payload["metadata"] = dict(step.metadata)
    if step.reason:
        payload["reason"] = step.reason
    if step.record_id:
        payload["record_id"] = step.record_id
    if step.timestamp:
        payload["timestamp"] = step.timestamp
    return payload


def _format_step(title: str, step: _StepOutcome) -> list[str]:
    lines = [f"{title}: {step.status}"]
    if step.reason:
        lines.append(f"  Powód: {step.reason}")
    if step.backend:
        lines.append(f"  Backend: {step.backend}")
    if step.location:
        lines.append(f"  Lokalizacja: {step.location}")
    if step.record_id:
        lines.append(f"  Rekord JSONL: {step.record_id}")
    if step.timestamp:
        lines.append(f"  Znacznik czasu: {step.timestamp}")
    if step.metadata:
        lines.append("  Metadane:")
        for key, value in sorted(step.metadata.items()):
            lines.append(f"    - {key}: {value}")
    return lines


def _build_step(status: str, **kwargs: Any) -> _StepOutcome:
    metadata = kwargs.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        metadata = dict(metadata)
    return _StepOutcome(
        status=status,
        backend=kwargs.get("backend"),
        location=kwargs.get("location"),
        metadata=metadata,
        reason=kwargs.get("reason"),
        record_id=kwargs.get("record_id"),
        timestamp=kwargs.get("timestamp"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    report_dir = Path(args.report_dir).resolve()
    json_log_path = Path(args.json_log).resolve()

    try:
        summary_data, summary_sha, summary_path = _read_summary(report_dir)
    except FileNotFoundError:
        _LOGGER.error("Brak summary.json w katalogu raportu: %s", report_dir)
        result = _PublishResult(
            status="error",
            summary_sha256=None,
            report_dir=str(report_dir),
            json_sync=_build_step("error", reason="missing_summary"),
            archive_upload=_build_step("error", reason="missing_summary"),
        )
        print(_serialize_result(result, as_json=args.json))
        return 1
    except ValueError as exc:
        _LOGGER.error("Nie udało się sparsować summary.json: %s", exc)
        result = _PublishResult(
            status="error",
            summary_sha256=None,
            report_dir=str(report_dir),
            json_sync=_build_step("error", reason="invalid_summary"),
            archive_upload=_build_step("error", reason="invalid_summary"),
        )
        print(_serialize_result(result, as_json=args.json))
        return 1

    try:
        core_config = load_core_config(args.config)
    except FileNotFoundError:
        _LOGGER.error("Nie znaleziono pliku konfiguracyjnego: %s", args.config)
        result = _PublishResult(
            status="error",
            summary_sha256=summary_sha,
            report_dir=str(report_dir),
            json_sync=_build_step("error", reason="missing_config"),
            archive_upload=_build_step("error", reason="missing_config"),
        )
        print(_serialize_result(result, as_json=args.json))
        return 2

    reporting_cfg = getattr(core_config, "reporting", None)
    json_cfg = None
    archive_cfg = None
    if reporting_cfg is not None:
        json_cfg = PaperSmokeJsonSynchronizer.resolve_config(reporting_cfg)
        archive_cfg = SmokeArchiveUploader.resolve_config(reporting_cfg)

    json_result = _build_step("skipped", reason="no_config")
    archive_result = _build_step("skipped", reason="no_config")
    exit_code = 0

    records = _load_json_records(json_log_path)
    record = _find_record(
        records=records,
        environment=args.environment,
        record_id=args.record_id,
        summary_sha256=summary_sha,
    )

    timestamp = datetime.now(timezone.utc)
    record_id_value: str | None = None
    if record is not None:
        record_id_value = str(record.get("record_id"))
        timestamp_value = record.get("timestamp")
        if isinstance(timestamp_value, str):
            try:
                timestamp = datetime.fromisoformat(timestamp_value)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                timestamp = timestamp.astimezone(timezone.utc)
            except ValueError:
                _LOGGER.warning("Nieprawidłowy format znacznika czasu w rekordzie JSONL: %s", timestamp_value)

    window = summary_data.get("window", {}) if isinstance(summary_data.get("window"), Mapping) else {}
    window_payload: MutableMapping[str, str] = {
        "start": str(window.get("start", "")),
        "end": str(window.get("end", "")),
    }

    secret_manager: SecretManager | None = None
    requires_secret = any(
        cfg is not None and str(getattr(cfg, "backend", "")).lower() == "s3"
        for cfg in (json_cfg, archive_cfg)
    )
    if requires_secret:
        try:
            secret_manager = _create_secret_manager(args)
        except SecretStorageError as exc:
            _LOGGER.error("Nie udało się zainicjalizować managera sekretów: %s", exc)
            exit_code = 3

    if json_cfg is None:
        json_result = _build_step("skipped", reason="no_config")
    elif args.skip_json_sync:
        json_result = _build_step("skipped", reason="skipped_by_flag", backend=json_cfg.backend)
    elif args.dry_run:
        json_result = _build_step("skipped", reason="dry_run", backend=json_cfg.backend)
    elif record is None:
        _LOGGER.error(
            "Nie znaleziono rekordu JSONL odpowiadającego smoke testowi (record_id=%s, summary_sha=%s)",
            args.record_id or "-",
            summary_sha,
        )
        json_result = _build_step("error", reason="missing_record", backend=json_cfg.backend)
        exit_code = exit_code or 4
    elif secret_manager is None and json_cfg.backend.lower() == "s3":
        json_result = _build_step("error", reason="secret_manager_unavailable", backend=json_cfg.backend)
        exit_code = exit_code or 5
    else:
        try:
            synchronizer = PaperSmokeJsonSynchronizer(json_cfg, secret_manager=secret_manager)
            sync_result = synchronizer.sync(
                json_log_path,
                environment=args.environment,
                record_id=record_id_value or "",
                timestamp=timestamp,
            )
            json_result = _build_step(
                "ok",
                backend=sync_result.backend,
                location=sync_result.location,
                metadata=sync_result.metadata,
                record_id=record_id_value,
                timestamp=timestamp.isoformat(),
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("Synchronizacja dziennika JSONL nie powiodła się")
            json_result = _build_step("error", reason=str(exc), backend=json_cfg.backend)
            exit_code = exit_code or 6

    archive_path = _ensure_archive(report_dir, args.archive)

    if archive_cfg is None:
        archive_result = _build_step("skipped", reason="no_config")
    elif args.skip_archive_upload:
        archive_result = _build_step("skipped", reason="skipped_by_flag", backend=archive_cfg.backend)
    elif args.dry_run:
        archive_result = _build_step("skipped", reason="dry_run", backend=archive_cfg.backend)
    elif archive_path is None:
        _LOGGER.error("Nie udało się odnaleźć lub utworzyć archiwum smoke testu")
        archive_result = _build_step("error", reason="missing_archive", backend=archive_cfg.backend)
        exit_code = exit_code or 7
    elif secret_manager is None and archive_cfg.backend.lower() == "s3":
        archive_result = _build_step("error", reason="secret_manager_unavailable", backend=archive_cfg.backend)
        exit_code = exit_code or 5
    else:
        try:
            uploader = SmokeArchiveUploader(archive_cfg, secret_manager=secret_manager)
            upload_result = uploader.upload(
                archive_path,
                environment=args.environment,
                summary_sha256=summary_sha,
                window=window_payload,
            )
            archive_result = _build_step(
                "ok",
                backend=upload_result.backend,
                location=upload_result.location,
                metadata=upload_result.metadata,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("Wysyłka archiwum smoke testu nie powiodła się")
            archive_result = _build_step("error", reason=str(exc), backend=archive_cfg.backend)
            exit_code = exit_code or 8

    overall_status = "ok"
    if exit_code != 0:
        overall_status = "error"
    elif json_result.status.startswith("skipped") and archive_result.status.startswith("skipped"):
        overall_status = "skipped"

    result = _PublishResult(
        status=overall_status,
        summary_sha256=summary_sha,
        report_dir=str(report_dir),
        json_sync=json_result,
        archive_upload=archive_result,
    )
    print(_serialize_result(result, as_json=args.json))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
