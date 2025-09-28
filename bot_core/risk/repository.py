"""Implementacje repozytoriów stanu ryzyka."""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Mapping, MutableMapping

from bot_core.risk.base import RiskRepository


def _sanitize_profile_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return sanitized or "profile"


def _normalize_state(state: Mapping[str, object]) -> Mapping[str, object]:
    def _convert(value: object) -> object:
        if isinstance(value, Mapping):
            return {str(k): _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(item) for item in value]
        return value

    return {str(key): _convert(value) for key, value in state.items()}


class FileRiskRepository(RiskRepository):
    """Przechowuje stan profili ryzyka w plikach JSON z atomowym zapisem."""

    def __init__(
        self,
        directory: str | Path,
        *,
        encoding: str = "utf-8",
        fsync: bool = False,
    ) -> None:
        self._base_path = Path(directory)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._encoding = encoding
        self._fsync = fsync
        self._lock = threading.Lock()

    def load(self, profile: str) -> Mapping[str, object] | None:
        path = self._path_for(profile)
        try:
            with self._lock:
                with path.open("r", encoding=self._encoding) as handle:
                    data = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            return None

        if isinstance(data, Mapping):
            return {str(key): value for key, value in data.items()}
        return None

    def store(self, profile: str, state: Mapping[str, object]) -> None:
        path = self._path_for(profile)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        payload: MutableMapping[str, object] = dict(_normalize_state(state))
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tmp_path.open("w", encoding=self._encoding) as handle:
                handle.write(serialized)
                handle.write("\n")
                handle.flush()
                if self._fsync:
                    os.fsync(handle.fileno())
            os.replace(tmp_path, path)

    def _path_for(self, profile: str) -> Path:
        filename = f"{_sanitize_profile_name(profile)}.json"
        return self._base_path / filename


__all__ = ["FileRiskRepository"]
