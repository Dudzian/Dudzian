"""Moduł zbierający anonimową telemetrię z opcją opt-in."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping

__all__ = [
    "DEFAULT_TELEMETRY_DIR",
    "TelemetryError",
    "TelemetryEvent",
    "TelemetrySettings",
    "AnonymousTelemetryCollector",
]

def _default_telemetry_dir() -> Path:
    base_override = os.environ.get("DUDZIAN_HOME")
    if base_override:
        return (Path(base_override).expanduser() / "telemetry").expanduser()
    return (Path.home() / ".dudzian" / "telemetry").expanduser()


DEFAULT_TELEMETRY_DIR = _default_telemetry_dir()
_SETTINGS_FILE = "settings.json"
_QUEUE_FILE = "queue.jsonl"
_EXPORT_DIR = "exports"
_MAX_QUEUE_EVENTS = 1000


class TelemetryError(RuntimeError):
    """Wyjątek związany z obsługą telemetrii."""


@dataclass(slots=True)
class TelemetrySettings:
    """Trwałe ustawienia zgody na telemetrię."""

    enabled: bool = False
    installation_id: str = ""
    salt: str = ""
    created_at: str = ""
    last_export_at: str | None = None
    pseudonym: str | None = None

    @classmethod
    def default(cls) -> "TelemetrySettings":
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            enabled=False,
            installation_id=secrets.token_hex(16),
            salt=secrets.token_hex(16),
            created_at=now,
            last_export_at=None,
            pseudonym=None,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> "TelemetrySettings":
        if not mapping:
            return cls.default()
        enabled = bool(mapping.get("enabled", False))
        installation_id = str(mapping.get("installation_id", "")) or secrets.token_hex(16)
        salt = str(mapping.get("salt", "")) or secrets.token_hex(16)
        created_at = str(mapping.get("created_at", "")) or datetime.now(timezone.utc).isoformat()
        last_export = mapping.get("last_export_at")
        if isinstance(last_export, str) and last_export.strip():
            last_export_at: str | None = last_export
        else:
            last_export_at = None
        pseudonym = mapping.get("pseudonym")
        if isinstance(pseudonym, str) and pseudonym.strip():
            pseudo_value: str | None = pseudonym
        else:
            pseudo_value = None
        return cls(
            enabled=enabled,
            installation_id=installation_id,
            salt=salt,
            created_at=created_at,
            last_export_at=last_export_at,
            pseudonym=pseudo_value,
        )

    def to_mapping(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "enabled": self.enabled,
            "installation_id": self.installation_id,
            "salt": self.salt,
            "created_at": self.created_at,
        }
        if self.last_export_at:
            payload["last_export_at"] = self.last_export_at
        if self.pseudonym:
            payload["pseudonym"] = self.pseudonym
        return payload


@dataclass(slots=True)
class TelemetryEvent:
    """Pojedyncze zdarzenie telemetrii w kolejce."""

    event_type: str
    created_at: str
    pseudonym: str
    installation_id: str
    payload: Mapping[str, object]

    def to_json(self) -> str:
        return json.dumps(
            {
                "event_type": self.event_type,
                "created_at": self.created_at,
                "pseudonym": self.pseudonym,
                "installation_id": self.installation_id,
                "payload": self._sanitize_payload(self.payload),
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _sanitize_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
        sanitized: dict[str, object] = {}
        for key, value in payload.items():
            key_str = str(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[key_str] = value
            else:
                sanitized[key_str] = str(value)
        return sanitized


class AnonymousTelemetryCollector:
    """Zbiera i buforuje anonimową telemetrię po uzyskaniu zgody."""

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self._dir = Path(storage_dir or DEFAULT_TELEMETRY_DIR).expanduser()
        self._lock = threading.RLock()
        self._settings = self._load_settings()
        self._queue_path = self._dir / _QUEUE_FILE
        self._export_dir = self._dir / _EXPORT_DIR

    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    @property
    def pseudonym(self) -> str | None:
        return self._settings.pseudonym

    @property
    def installation_id(self) -> str:
        return self._settings.installation_id

    @property
    def queue_path(self) -> Path:
        return self._queue_path

    @property
    def last_export_at(self) -> str | None:
        return self._settings.last_export_at

    def queued_events(self) -> int:
        if not self._queue_path.exists():
            return 0
        try:
            with self._queue_path.open("r", encoding="utf-8") as handle:
                return sum(1 for _ in handle)
        except OSError:
            return 0

    def preview_events(self, limit: int = 20) -> list[Mapping[str, object]]:
        if limit <= 0 or not self._queue_path.exists():
            return []
        results: list[Mapping[str, object]] = []
        try:
            with self._queue_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    results.append(payload)
                    if len(results) >= limit:
                        break
        except OSError:
            return []
        return results

    # ------------------------------------------------------------------
    def set_opt_in(self, enabled: bool, fingerprint: str | None = None) -> None:
        with self._lock:
            if enabled:
                self._settings.enabled = True
                self._settings.pseudonym = self._compute_pseudonym(fingerprint)
            else:
                self._settings.enabled = False
            self._persist_settings()

    def refresh_pseudonym(self, fingerprint: str | None) -> str | None:
        with self._lock:
            self._settings.pseudonym = self._compute_pseudonym(fingerprint)
            self._persist_settings()
            return self._settings.pseudonym

    def collect_event(
        self,
        event_type: str,
        payload: Mapping[str, object] | None = None,
        *,
        fingerprint: str | None = None,
        timestamp: datetime | None = None,
    ) -> bool:
        with self._lock:
            if not self._settings.enabled:
                return False
            pseudonym = self._compute_pseudonym(fingerprint)
            if pseudonym:
                self._settings.pseudonym = pseudonym
            event = TelemetryEvent(
                event_type=str(event_type or "unknown"),
                created_at=(timestamp or datetime.now(timezone.utc)).isoformat(),
                pseudonym=self._settings.pseudonym or self._settings.installation_id,
                installation_id=self._settings.installation_id,
                payload=payload or {},
            )
            self._enqueue(event)
            return True

    def export_events(self, destination: str | Path | None = None) -> Path | None:
        with self._lock:
            events = self._drain_queue()
            if not events:
                return None
            export_dir = Path(destination or self._export_dir).expanduser()
            export_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            export_path = export_dir / f"telemetry_{timestamp}.json"
            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "installation_id": self._settings.installation_id,
                "pseudonym": self._settings.pseudonym,
                "events": events,
            }
            export_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self._settings.last_export_at = datetime.now(timezone.utc).isoformat()
            self._persist_settings()
            return export_path

    def clear_queue(self) -> None:
        with self._lock:
            if not self._queue_path.exists():
                return
            try:
                os.remove(self._queue_path)
            except FileNotFoundError:
                return
            except OSError as exc:  # pragma: no cover - środowisko plikowe
                raise TelemetryError(f"Nie można usunąć kolejki telemetrii: {exc}") from exc

    # ------------------------------------------------------------------
    def _load_settings(self) -> TelemetrySettings:
        settings_path = self._dir / _SETTINGS_FILE
        try:
            content = settings_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return TelemetrySettings.default()
        except OSError as exc:  # pragma: no cover - środowisko plikowe
            raise TelemetryError(f"Nie można odczytać ustawień telemetrii: {exc}") from exc
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise TelemetryError(f"Uszkodzony plik ustawień telemetrii: {exc}") from exc
        if not isinstance(payload, MutableMapping):
            raise TelemetryError("Plik ustawień telemetrii musi zawierać obiekt JSON")
        return TelemetrySettings.from_mapping(payload)

    def _persist_settings(self) -> None:
        settings_path = self._dir / _SETTINGS_FILE
        try:
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(self._settings.to_mapping(), indent=2, ensure_ascii=False)
            settings_path.write_text(serialized + "\n", encoding="utf-8")
        except OSError as exc:  # pragma: no cover - środowisko plikowe
            raise TelemetryError(f"Nie można zapisać ustawień telemetrii: {exc}") from exc

    def _compute_pseudonym(self, fingerprint: str | None) -> str:
        base = (fingerprint or self._settings.installation_id).strip() or self._settings.installation_id
        digest = hashlib.sha256(f"{base}|{self._settings.salt}".encode("utf-8")).hexdigest()
        return digest[:32]

    def _enqueue(self, event: TelemetryEvent) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        lines = []
        if self._queue_path.exists():
            try:
                with self._queue_path.open("r", encoding="utf-8") as handle:
                    lines = [line for line in handle if line.strip()]
            except OSError:
                lines = []
        lines.append(event.to_json())
        if len(lines) > _MAX_QUEUE_EVENTS:
            lines = lines[-_MAX_QUEUE_EVENTS:]
        try:
            with self._queue_path.open("w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
        except OSError as exc:  # pragma: no cover - środowisko plikowe
            raise TelemetryError(f"Nie można zapisać kolejki telemetrii: {exc}") from exc

    def _drain_queue(self) -> list[Mapping[str, object]]:
        if not self._queue_path.exists():
            return []
        try:
            content = self._queue_path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - środowisko plikowe
            raise TelemetryError(f"Nie można odczytać kolejki telemetrii: {exc}") from exc
        events: list[Mapping[str, object]] = []
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(payload)
        try:
            os.remove(self._queue_path)
        except FileNotFoundError:
            pass
        except OSError as exc:  # pragma: no cover - środowisko plikowe
            raise TelemetryError(f"Nie można wyczyścić kolejki telemetrii: {exc}") from exc
        return events


__all__ = [
    "DEFAULT_TELEMETRY_DIR",
    "TelemetryError",
    "TelemetryEvent",
    "TelemetrySettings",
    "AnonymousTelemetryCollector",
]
