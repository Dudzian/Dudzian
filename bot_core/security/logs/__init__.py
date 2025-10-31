"""Obsługa lokalnych logów audytowych i ich podpisanego eksportu."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from bot_core.security.fingerprint import decode_secret
from bot_core.security.signing import build_hmac_signature

DEFAULT_AUDIT_PATH = Path("logs/security_admin.log")
DEFAULT_ALERTS_PATH = Path("logs/security_alerts.log")
DEFAULT_LOG_PATHS: tuple[Path, ...] = (
    Path("logs/trading_controller.log"),
    Path("logs/runtime_health.jsonl"),
)
AUDIT_KEY_ENV = "BOT_CORE_SECURITY_AUDIT_KEY"
AUDIT_KEY_ID_ENV = "BOT_CORE_SECURITY_AUDIT_KEY_ID"


@dataclass(slots=True)
class AuditExportResult:
    """Informacje o wygenerowanym pakiecie audytowym."""

    bundle_path: Path
    entry_count: int
    generated_at: datetime
    signature: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_path": str(self.bundle_path),
            "entry_count": self.entry_count,
            "generated_at": self.generated_at.astimezone(timezone.utc).isoformat(),
            "signature": dict(self.signature),
        }


@dataclass(slots=True)
class SecurityBundleExportResult:
    """Informacje o zbiorczym pakiecie logów i alertów."""

    bundle_path: Path
    generated_at: datetime
    audit_entries: int
    alert_entries: int
    files: Mapping[str, Any]
    signature: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "bundle_path": str(self.bundle_path),
            "generated_at": self.generated_at.astimezone(timezone.utc).isoformat(),
            "audit_entries": self.audit_entries,
            "alert_entries": self.alert_entries,
            "signature": dict(self.signature),
        }
        payload["files"] = {name: dict(meta) for name, meta in self.files.items()}
        return payload


def _coerce_path(path: str | os.PathLike[str] | None, *, default: Path) -> Path:
    if path is None:
        return default
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        return candidate
    return candidate


def _load_signing_key(value: str | None) -> tuple[bytes, str | None]:
    raw = value or os.environ.get(AUDIT_KEY_ENV)
    if not raw:
        raise ValueError(
            "Brak klucza HMAC dla eksportu logów bezpieczeństwa (BOT_CORE_SECURITY_AUDIT_KEY)."
        )
    candidate_path = Path(raw)
    if candidate_path.exists() and candidate_path.is_file():
        content = candidate_path.read_text(encoding="utf-8").strip()
    else:
        content = raw.strip()
    key = decode_secret(content)
    key_id = os.environ.get(AUDIT_KEY_ID_ENV)
    return key, key_id


def _read_jsonl_entries(
    path: Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    collected: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                entry = json.loads(text)
            except json.JSONDecodeError:
                entry = {"raw": text}
            collected.append(entry)
    if limit is not None and limit > 0 and len(collected) > limit:
        collected = collected[-limit:]
    return collected


def read_audit_entries(path: str | os.PathLike[str] | None, *, limit: int = 200) -> list[dict[str, Any]]:
    """Zwraca listę wpisów audytowych z pliku JSONL."""

    log_path = _coerce_path(path, default=DEFAULT_AUDIT_PATH)
    if not log_path.exists():
        return []

    return _read_jsonl_entries(log_path, limit=limit if limit > 0 else None)


def read_alert_entries(
    path: str | os.PathLike[str] | None,
    *,
    limit: int | None = 200,
) -> list[dict[str, Any]]:
    """Zwraca wpisy alertów bezpieczeństwa."""

    resolved = _coerce_path(path, default=DEFAULT_ALERTS_PATH)
    return _read_jsonl_entries(resolved, limit=limit)


def _read_text_tail(path: Path, *, max_lines: int | None) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if max_lines is not None and max_lines > 0 and len(lines) > max_lines:
        lines = lines[-max_lines:]
    return lines


def export_signed_audit_log(
    *,
    log_path: str | os.PathLike[str] | None,
    destination_dir: str | os.PathLike[str] | None,
    limit: int | None = None,
    key_source: str | None = None,
    key_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AuditExportResult:
    """Eksportuje log audytowy wraz z podpisem HMAC."""

    entries = read_audit_entries(log_path, limit=limit or 0 if limit is not None else 0)
    generated_at = datetime.now(timezone.utc)
    bundle = {
        "generated_at": generated_at.isoformat(),
        "source": str(_coerce_path(log_path, default=DEFAULT_AUDIT_PATH)),
        "entries": entries,
        "metadata": dict(metadata) if metadata else {},
    }

    key_bytes, env_key_id = _load_signing_key(key_source)
    effective_key_id = key_id or env_key_id
    signature = build_hmac_signature(bundle, key=key_bytes, key_id=effective_key_id)

    output_dir = _coerce_path(destination_dir, default=Path.cwd() / "exports")
    if output_dir.suffix:
        # wskazano konkretny plik docelowy
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_path = output_dir
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"security_audit_{generated_at.strftime('%Y%m%dT%H%M%SZ')}.json"
        output_path = output_dir / filename

    payload = {
        "bundle": bundle,
        "signature": signature,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return AuditExportResult(
        bundle_path=output_path,
        entry_count=len(entries),
        generated_at=generated_at,
        signature=signature,
    )


def export_security_bundle(
    *,
    audit_path: str | os.PathLike[str] | None,
    alerts_path: str | os.PathLike[str] | None,
    destination_dir: str | os.PathLike[str] | None,
    audit_limit: int | None = None,
    alert_limit: int | None = None,
    include_logs: Iterable[str | os.PathLike[str]] | None = None,
    log_tail: int | None = 500,
    key_source: str | None = None,
    key_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SecurityBundleExportResult:
    """Eksportuje zbiorczy pakiet logów i alertów bezpieczeństwa."""

    audit_entries = read_audit_entries(audit_path, limit=audit_limit or 0)
    alert_entries = read_alert_entries(alerts_path, limit=alert_limit)

    extra_logs: MutableMapping[str, Any] = {}
    sources: Iterable[str | os.PathLike[str]]
    if include_logs:
        sources = include_logs
    else:
        sources = DEFAULT_LOG_PATHS
    for raw_path in sources:
        resolved = _coerce_path(str(raw_path), default=Path(str(raw_path)))
        tail = _read_text_tail(resolved, max_lines=log_tail)
        extra_logs[str(resolved)] = {
            "lines": tail,
            "line_count": len(tail),
            "exists": resolved.exists(),
        }

    generated_at = datetime.now(timezone.utc)
    bundle = {
        "generated_at": generated_at.isoformat(),
        "metadata": dict(metadata) if metadata else {},
        "audit": {
            "source": str(_coerce_path(audit_path, default=DEFAULT_AUDIT_PATH)),
            "entries": audit_entries,
        },
        "alerts": {
            "source": str(_coerce_path(alerts_path, default=DEFAULT_ALERTS_PATH)),
            "entries": alert_entries,
        },
        "logs": extra_logs,
    }

    key_bytes, env_key_id = _load_signing_key(key_source)
    effective_key_id = key_id or env_key_id
    signature = build_hmac_signature(bundle, key=key_bytes, key_id=effective_key_id)

    output_dir = _coerce_path(destination_dir, default=Path.cwd() / "exports")
    if output_dir.suffix:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_path = output_dir
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"security_bundle_{generated_at.strftime('%Y%m%dT%H%M%SZ')}.json"
        output_path = output_dir / filename

    payload = {
        "bundle": bundle,
        "signature": signature,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return SecurityBundleExportResult(
        bundle_path=output_path,
        generated_at=generated_at,
        audit_entries=len(audit_entries),
        alert_entries=len(alert_entries),
        files=extra_logs,
        signature=signature,
    )


__all__ = [
    "AuditExportResult",
    "SecurityBundleExportResult",
    "export_signed_audit_log",
    "export_security_bundle",
    "read_alert_entries",
    "read_audit_entries",
]

