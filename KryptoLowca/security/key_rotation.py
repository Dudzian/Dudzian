"""Mechanizm automatycznej rotacji kluczy API."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from KryptoLowca.managers.security_manager import SecurityManager


@dataclass(slots=True)
class RotationStatus:
    last_rotation: Optional[datetime]
    next_due: Optional[datetime]
    rotation_required: bool


class KeyRotationManager:
    """Obsługuje harmonogram rotacji kluczy dla ``SecurityManager``."""

    def __init__(
        self,
        security_manager: SecurityManager,
        *,
        rotation_days: int = 30,
        metadata_file: Optional[str | Path] = None,
    ) -> None:
        self.security_manager = security_manager
        self.rotation_days = rotation_days
        base_file = metadata_file or f"{security_manager.key_file}.rotation.json"
        self.metadata_path = Path(base_file)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------- Metadata helpers ----------------------
    def _load_metadata(self) -> Dict[str, Any]:
        if not self.metadata_path.exists():
            return {}
        try:
            return json.loads(self.metadata_path.read_text())
        except Exception:
            return {}

    def _write_metadata(self, payload: Dict[str, Any]) -> None:
        self.metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    # ---------------------- Public API ----------------------
    def status(self) -> RotationStatus:
        data = self._load_metadata()
        raw = data.get("last_rotation")
        last_rotation = datetime.fromisoformat(raw) if raw else None
        next_due = None
        rotation_required = False
        if last_rotation:
            next_due = last_rotation + timedelta(days=self.rotation_days)
            rotation_required = datetime.now(timezone.utc) >= next_due
        else:
            rotation_required = True
        return RotationStatus(last_rotation, next_due, rotation_required)

    def mark_rotated(self, when: Optional[datetime] = None) -> None:
        when = when or datetime.now(timezone.utc)
        payload = self._load_metadata()
        payload["last_rotation"] = when.isoformat()
        self._write_metadata(payload)

    def rotate_keys(self, password: str, *, new_password: Optional[str] = None) -> Dict[str, Any]:
        """Odczytaj klucze i zapisz ponownie (generując nową sól)."""
        keys = self.security_manager.load_encrypted_keys(password)
        target_password = new_password or password
        self.security_manager.save_encrypted_keys(keys, target_password)
        self.mark_rotated()
        return keys

    def ensure_rotation(self, password: str, *, new_password: Optional[str] = None) -> RotationStatus:
        status = self.status()
        if status.rotation_required:
            self.rotate_keys(password, new_password=new_password)
            status = self.status()
        return status


__all__ = ["KeyRotationManager", "RotationStatus"]
