"""Komponent obsługi akcji operatorskich runtime UI."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import logging

from .qml_bridge import to_plain_dict, to_plain_value


class OperatorActionService:
    """Normalizuje i rejestruje akcje operatorskie runtime."""

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _unwrap_operator_entry_mapping(payload: Mapping[str, object]) -> Mapping[str, object]:
        payload_dict = dict(payload)
        nested_record = payload_dict.get("record")
        if isinstance(nested_record, Mapping):
            return dict(nested_record)
        return payload_dict

    def normalize_operator_entry(self, entry: object | None) -> Mapping[str, object]:
        if entry is None:
            return {}
        if isinstance(entry, Mapping):
            return self._unwrap_operator_entry_mapping(entry)
        plain = to_plain_value(entry)
        if isinstance(plain, Mapping):
            return self._unwrap_operator_entry_mapping(plain)
        variant = None
        if hasattr(entry, "toVariant"):
            try:
                variant = entry.toVariant()
            except Exception:
                variant = None
        if variant is None and hasattr(entry, "toPyObject"):
            try:
                variant = entry.toPyObject()
            except Exception:
                variant = None
        if variant is not None:
            variant_plain = to_plain_value(variant)
            if isinstance(variant_plain, Mapping):
                return self._unwrap_operator_entry_mapping(variant_plain)
        return {}

    @staticmethod
    def normalize_operator_action(action: object) -> str:
        raw = str(action or "").strip()
        mapping = {
            "requestFreeze": "freeze",
            "requestUnfreeze": "unfreeze",
            "requestUnblock": "unblock",
            "freeze": "freeze",
            "unfreeze": "unfreeze",
            "unblock": "unblock",
        }
        return mapping.get(raw, raw)

    def record_action(self, action: object, entry: object | None) -> dict[str, object]:
        try:
            plain_action = to_plain_value(action)
        except Exception:
            plain_action = ""
        if isinstance(plain_action, str):
            action_str = plain_action.strip()
        else:
            action_str = str(plain_action or "")

        normalized_action = self.normalize_operator_action(action_str)
        try:
            normalized_entry = self.normalize_operator_entry(entry)
            sanitized = to_plain_dict(normalized_entry)
        except Exception:
            sanitized = {}

        timestamp_value: object | None = None
        for key in ("timestamp", "time", "ts"):
            if key in sanitized and sanitized[key] is not None:
                timestamp_value = sanitized[key]
                break
        if timestamp_value is None:
            timestamp_value = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

        payload = {
            "action": normalized_action,
            "timestamp": str(timestamp_value),
            "entry": sanitized,
        }

        if sanitized:
            reference = sanitized.get("event") or sanitized.get("timestamp") or sanitized.get("id")
        else:
            reference = None

        if reference:
            self._logger.info("Operator action '%s' triggered for %s", normalized_action, reference)
        else:
            self._logger.info("Operator action '%s' triggered", normalized_action)

        return payload
