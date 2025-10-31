"""Prosta warstwa przechowywania kluczy API wykorzystywana w onboardingu."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable

import json


class SecretStoreError(RuntimeError):
    """Wyjątek zgłaszany w przypadku błędów operacji na magazynie sekretów."""


@dataclass(frozen=True)
class ExchangeCredentials:
    """Dane uwierzytelniające pojedynczej giełdy."""

    exchange: str
    api_key: str
    api_secret: str
    api_passphrase: str | None = None

    def normalized_exchange(self) -> str:
        return self.exchange.strip().lower()


class SecretStore:
    """Magazyn sekretów zapisujący dane uwierzytelniające do pliku JSON."""

    def __init__(self, *, storage_path: str | Path | None = None) -> None:
        base_path = Path(storage_path) if storage_path is not None else _default_path()
        self._path = base_path.expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    @property
    def path(self) -> Path:
        return self._path

    def save_exchange_credentials(self, credentials: ExchangeCredentials) -> None:
        """Zapisuje/aktualizuje dane uwierzytelniające giełdy."""

        exchange_id = credentials.normalized_exchange()
        if not exchange_id:
            raise SecretStoreError("Brak identyfikatora giełdy")
        payload = {
            "api_key": credentials.api_key.strip(),
            "api_secret": credentials.api_secret.strip(),
            "api_passphrase": (credentials.api_passphrase or "").strip() or None,
        }
        if not payload["api_key"] or not payload["api_secret"]:
            raise SecretStoreError("Klucz API i sekret są wymagane")

        with self._lock:
            data = self._read_all()
            data[exchange_id] = payload
            self._write_all(data)

    def list_exchanges(self) -> Iterable[str]:
        with self._lock:
            return tuple(self._read_all().keys())

    def _read_all(self) -> Dict[str, Dict[str, str | None]]:
        if not self._path.exists():
            return {}
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - błędy systemowe
            raise SecretStoreError(f"Nie można odczytać magazynu sekretów: {exc}") from exc
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SecretStoreError("Magazyn sekretów jest uszkodzony") from exc
        if not isinstance(data, dict):
            raise SecretStoreError("Nieprawidłowy format magazynu sekretów")
        result: Dict[str, Dict[str, str | None]] = {}
        for key, value in data.items():
            if isinstance(key, str) and isinstance(value, dict):
                result[key] = {
                    "api_key": str(value.get("api_key", "")),
                    "api_secret": str(value.get("api_secret", "")),
                    "api_passphrase": value.get("api_passphrase"),
                }
        return result

    def _write_all(self, payload: Dict[str, Dict[str, str | None]]) -> None:
        serialized = json.dumps(payload, indent=2, ensure_ascii=False)
        try:
            self._path.write_text(serialized, encoding="utf-8")
        except OSError as exc:
            raise SecretStoreError(f"Nie można zapisać magazynu sekretów: {exc}") from exc


def _default_path() -> Path:
    config_dir = Path.home() / ".kryptolowca"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "api_credentials.json"


__all__ = ["ExchangeCredentials", "SecretStore", "SecretStoreError"]
