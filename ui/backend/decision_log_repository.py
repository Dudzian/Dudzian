"""Repozytorium odpowiedzialne za odczyt wpisów decision log (JSONL)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path

from .decision_payload_normalizer import DecisionRecord

_LOGGER = logging.getLogger(__name__)


class DecisionLogRepository:
    """Udostępnia iterację i ładowanie wpisów decision logu z pliku JSONL."""

    def iter_jsonl_entries(self, log_path: Path) -> Iterator[DecisionRecord]:
        """Iteruje po poprawnych wpisach mapowanych z pliku JSONL."""

        try:
            with log_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    payload = line.strip()
                    if not payload:
                        continue
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        _LOGGER.warning(
                            "Pominięto uszkodzony wpis decision logu %s",
                            log_path,
                            exc_info=True,
                        )
                        continue
                    if isinstance(data, Mapping):
                        yield data
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise RuntimeError(
                f"Nie udało się odczytać decision logu '{log_path}': {exc}"
            ) from exc

    def load_jsonl_entries(self, log_path: Path, limit: int) -> list[DecisionRecord]:
        entries = list(self.iter_jsonl_entries(log_path))
        if limit > 0:
            entries = entries[-limit:]
        return entries
