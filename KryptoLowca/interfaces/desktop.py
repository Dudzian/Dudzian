"""Szkielet interfejsu desktopowego (.exe) – do rozbudowy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from KryptoLowca.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class DesktopMenuEntry:
    label: str
    action: Callable[[], None]


class DesktopInterface:
    def __init__(self) -> None:
        self._entries: Dict[str, DesktopMenuEntry] = {}

    def register_entry(self, name: str, entry: DesktopMenuEntry) -> None:
        logger.debug("Rejestruję akcję GUI %s", name)
        self._entries[name] = entry

    def trigger(self, name: str) -> None:
        entry = self._entries.get(name)
        if not entry:
            raise KeyError(f"Brak akcji GUI '{name}'")
        logger.info("Uruchamiam akcję GUI %s", name)
        entry.action()

    def available_entries(self) -> Dict[str, DesktopMenuEntry]:
        return dict(self._entries)


__all__ = ["DesktopInterface", "DesktopMenuEntry"]
