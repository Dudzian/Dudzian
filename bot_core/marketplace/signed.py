"""Obsługa lokalnego marketplace'u podpisanych presetów."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

from bot_core.security.hwid import HwIdProvider

from .presets import PresetRepository

if TYPE_CHECKING:
    from bot_core.strategies.catalog import StrategyCatalog, StrategyPresetDescriptor


@dataclass(slots=True)
class MarketplaceSyncResult:
    """Raport z synchronizacji marketplace'u podpisanych presetów."""

    descriptors: tuple["StrategyPresetDescriptor", ...]
    installed: tuple[str, ...] = field(default_factory=tuple)
    skipped: tuple[str, ...] = field(default_factory=tuple)
    issues: Mapping[str, Sequence[str]] = field(default_factory=dict)


class SignedPresetMarketplace:
    """Ładuje i waliduje presety Marketplace wymagając podpisu cyfrowego."""

    def __init__(
        self,
        root: str | Path,
        *,
        signing_keys: Mapping[str, bytes | str],
    ) -> None:
        self._repository = PresetRepository(root)
        self._signing_keys = dict(signing_keys)

    def sync(
        self,
        catalog: "StrategyCatalog",
        *,
        hwid_provider: HwIdProvider | None = None,
    ) -> MarketplaceSyncResult:
        """Wczytuje presety i rejestruje je w katalogu strategi."""

        documents = self._repository.load_all(signing_keys=self._signing_keys)
        installed: list[str] = []
        skipped: list[str] = []
        issues: dict[str, list[str]] = {}
        descriptors: list[StrategyPresetDescriptor] = []

        for document in documents:
            preset_id = document.preset_id or document.name or "unknown"
            if not document.verification.verified:
                skipped.append(preset_id)
                issues[preset_id] = list(document.verification.issues or ("signature-invalid",))
                continue

            try:
                descriptor = catalog.register_signed_preset(
                    document,
                    hwid_provider=hwid_provider,
                )
            except Exception as exc:  # pragma: no cover - defensywne logowanie
                skipped.append(preset_id)
                payload = list(document.verification.issues or ())
                payload.append(str(exc))
                issues[preset_id] = payload
                continue

            descriptors.append(descriptor)
            installed.append(descriptor.preset_id)

        normalized_issues = {key: tuple(value) for key, value in issues.items()}
        return MarketplaceSyncResult(
            descriptors=tuple(descriptors),
            installed=tuple(installed),
            skipped=tuple(skipped),
            issues=normalized_issues,
        )


__all__ = ["SignedPresetMarketplace", "MarketplaceSyncResult"]

