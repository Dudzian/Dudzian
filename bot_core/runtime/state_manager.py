"""Proste repozytorium checkpointów dla scenariuszy E2E lokalnego runtime."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping


class RuntimeStateError(RuntimeError):
    """Błąd zgłaszany przy niepoprawnych przejściach stanu runtime."""


@dataclass(slots=True)
class RuntimeCheckpoint:
    """Reprezentuje zapisany checkpoint scenariusza uruchomienia."""

    entrypoint: str
    mode: str
    config_path: str
    created_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializuje checkpoint do słownika JSON."""

        payload: dict[str, Any] = {
            "entrypoint": self.entrypoint,
            "mode": self.mode,
            "config_path": self.config_path,
            "created_at": self.created_at.astimezone(timezone.utc).isoformat(),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class RuntimeStateManager:
    """Proste repozytorium checkpointów wykorzystywane przez scenariusze E2E."""

    def __init__(self, root: str | Path = "var/runtime", *, filename: str = "state.json") -> None:
        self._root = Path(root).expanduser()
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / filename
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Zwraca lokalizację pliku checkpointu."""

        return self._path

    def load_checkpoint(self) -> RuntimeCheckpoint | None:
        """Odczytuje ostatni zapisany checkpoint, jeśli istnieje."""

        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        created_at_raw = data.get("created_at")
        try:
            created_at = (
                datetime.fromisoformat(created_at_raw)
                if isinstance(created_at_raw, str)
                else datetime.now(timezone.utc)
            )
        except ValueError:
            created_at = datetime.now(timezone.utc)
        metadata = data.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        return RuntimeCheckpoint(
            entrypoint=str(data.get("entrypoint", "")),
            mode=str(data.get("mode", "")),
            config_path=str(data.get("config_path", "")),
            created_at=created_at.astimezone(timezone.utc),
            metadata=dict(metadata),
        )

    def record_checkpoint(
        self,
        *,
        entrypoint: str,
        mode: str,
        config_path: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> RuntimeCheckpoint:
        """Zapisuje nowy checkpoint i zwraca jego reprezentację."""

        checkpoint = RuntimeCheckpoint(
            entrypoint=str(entrypoint),
            mode=str(mode),
            config_path=str(Path(config_path).expanduser()),
            created_at=datetime.now(timezone.utc),
            metadata=dict(metadata or {}),
        )
        payload = checkpoint.to_dict()
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return checkpoint

    def set_active_model(self, metadata: Mapping[str, Any]) -> None:
        """Aktualizuje metadane checkpointu o identyfikator aktywnego modelu."""

        normalized: MutableMapping[str, Any] = {
            str(key): value for key, value in metadata.items()
        }
        normalized["updated_at"] = datetime.now(timezone.utc).isoformat()

        with self._lock:
            try:
                raw = self._path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}

            meta_section = payload.get("metadata")
            if not isinstance(meta_section, dict):
                meta_section = {}
            meta_section["active_model"] = dict(normalized)
            payload["metadata"] = meta_section

            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def require_checkpoint(
        self,
        *,
        target_mode: str,
        entrypoint: str | None = None,
    ) -> RuntimeCheckpoint:
        """Weryfikuje, że istnieje checkpoint umożliwiający przejście do docelowego trybu."""

        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            raise RuntimeStateError(
                "Brak zapisanego checkpointu fazy demo – uruchom najpierw tryb demo."  # noqa: TRY003
            )
        expected_mode = "demo"
        if checkpoint.mode.lower() != expected_mode:
            raise RuntimeStateError(
                f"Ostatni checkpoint pochodzi z trybu '{checkpoint.mode}', wymagany: '{expected_mode}'."
            )
        if entrypoint and checkpoint.entrypoint and checkpoint.entrypoint != entrypoint:
            raise RuntimeStateError(
                "Checkpoint został utworzony dla innego punktu wejścia – powtórz scenariusz demo."  # noqa: TRY003
            )
        return checkpoint

    def clear(self) -> None:
        """Usuwa zapisany checkpoint."""

        with self._lock:
            try:
                self._path.unlink()
            except FileNotFoundError:
                return


__all__ = [
    "RuntimeCheckpoint",
    "RuntimeStateError",
    "RuntimeStateManager",
]
