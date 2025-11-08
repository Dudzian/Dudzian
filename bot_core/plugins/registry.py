"""Rejestr pluginów strategii wraz z procesem review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

from .manifest import PluginSignature, SignedStrategyPlugin, StrategyPluginManifest
from .review import PluginReviewBoard, PluginReviewResult, ReviewStatus


class PluginRegistryError(RuntimeError):
    """Wyjątek zgłaszany przy błędach rejestracji pluginu."""


@dataclass(slots=True)
class RegisteredPlugin:
    """Wpis katalogu pluginów."""

    manifest: StrategyPluginManifest
    signature: PluginSignature
    review: PluginReviewResult

    def to_dict(self) -> Mapping[str, object]:
        return {
            "manifest": dict(self.manifest.to_dict()),
            "signature": dict(self.signature.to_dict()),
            "review": self.review.to_dict(),
        }


class StrategyPluginRegistry:
    """Rejestr pluginów dostępnych dla platformy."""

    def __init__(self, review_board: PluginReviewBoard) -> None:
        self._review_board = review_board
        self._registry: MutableMapping[str, RegisteredPlugin] = {}

    def register(self, package: SignedStrategyPlugin) -> RegisteredPlugin:
        review = self._review_board.evaluate(package)
        if review.status == ReviewStatus.REJECTED:
            issues = "; ".join(finding.message for finding in review.findings if finding.severity == "error")
            raise PluginRegistryError(f"Plugin '{package.manifest.identifier}' odrzucony: {issues}")

        entry = RegisteredPlugin(
            manifest=package.manifest,
            signature=package.signature,
            review=review,
        )
        self._registry[package.manifest.identifier] = entry
        return entry

    def get(self, identifier: str) -> RegisteredPlugin:
        try:
            return self._registry[identifier]
        except KeyError as exc:
            raise KeyError(f"Plugin '{identifier}' nie jest zarejestrowany") from exc

    def list(self) -> Sequence[RegisteredPlugin]:
        return tuple(self._registry.values())


__all__ = ["StrategyPluginRegistry", "PluginRegistryError", "RegisteredPlugin"]

