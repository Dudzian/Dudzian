"""Ładowanie pluginów strategii do katalogu handlowego."""

from __future__ import annotations

from importlib import import_module
from types import MappingProxyType
from typing import Mapping, MutableMapping, Sequence

from bot_core.trading.strategies.plugins import StrategyCatalog, StrategyPlugin

from .manifest import SignedStrategyPlugin
from .registry import PluginRegistryError, RegisteredPlugin, StrategyPluginRegistry
from .review import ReviewStatus


class PluginLoadError(RuntimeError):
    """Wyjątek zgłaszany przy problemach ładowania pluginu."""


def _normalize_entrypoints(metadata: Mapping[str, object]) -> Mapping[str, str]:
    """Ekstrahuje definicje entrypointów strategii z metadanych manifestu."""

    if not metadata:
        return MappingProxyType({})

    entry_section = metadata.get("entry_points")
    if isinstance(entry_section, Mapping):
        strategies = entry_section.get("strategies")
    else:
        strategies = None

    if not isinstance(strategies, Mapping):
        return MappingProxyType({})

    normalized: MutableMapping[str, str] = {}
    for name, target in strategies.items():
        if not isinstance(name, str) or not isinstance(target, str):
            continue
        strategy_name = name.strip()
        import_target = target.strip()
        if not strategy_name or not import_target:
            continue
        normalized[strategy_name] = import_target
    return MappingProxyType(dict(normalized))


def _load_symbol(target: str) -> StrategyPlugin | type[StrategyPlugin]:
    module_name, _, symbol_name = target.partition(":")
    if not module_name or not symbol_name:
        raise PluginLoadError(f"Nieprawidłowy entrypoint pluginu: '{target}'")

    module = import_module(module_name)
    if not hasattr(module, symbol_name):
        raise PluginLoadError(
            f"Entrypoint '{target}' nie istnieje (brak symbolu '{symbol_name}')"
        )
    return getattr(module, symbol_name)


class StrategyPluginLoader:
    """Odpowiada za rejestrację zaakceptowanych pluginów w katalogu strategii."""

    def __init__(
        self,
        *,
        catalog: StrategyCatalog | None = None,
        registry: StrategyPluginRegistry | None = None,
    ) -> None:
        self._catalog = catalog or StrategyCatalog()
        self._registry = registry

    @property
    def catalog(self) -> StrategyCatalog:
        return self._catalog

    def install(
        self,
        package: SignedStrategyPlugin,
        *,
        strict_review: bool = True,
    ) -> Sequence[str]:
        """Rejestruje plugin w katalogu strategii.

        Zwraca nazwy strategii dostępne po instalacji.
        """

        registered: RegisteredPlugin | None = None
        manifest = package.manifest

        if self._registry is not None:
            try:
                registered = self._registry.register(package)
            except PluginRegistryError as exc:  # pragma: no cover - propagacja błędów
                raise PluginLoadError(str(exc)) from exc
            manifest = registered.manifest
            if strict_review and registered.review.status is not ReviewStatus.ACCEPTED:
                raise PluginLoadError(
                    "Plugin wymaga dodatkowych zmian (status review: %s)"
                    % registered.review.status.value
                )

        entrypoints = _normalize_entrypoints(manifest.metadata)
        if not entrypoints:
            raise PluginLoadError("Manifest nie definiuje entrypointów strategii")

        missing = [name for name in manifest.strategies if name not in entrypoints]
        if missing:
            raise PluginLoadError(
                "Brak entrypointów dla strategii: %s" % ", ".join(sorted(missing))
            )

        installed: list[str] = []
        for strategy_name in manifest.strategies:
            target = entrypoints[strategy_name]
            symbol = _load_symbol(target)

            if isinstance(symbol, type) and issubclass(symbol, StrategyPlugin):
                plugin_cls = symbol
                if plugin_cls.name != strategy_name:
                    raise PluginLoadError(
                        "Nazwa pluginu '%s' nie zgadza się z manifestem (oczekiwano '%s')"
                        % (plugin_cls.name, strategy_name)
                    )
                self._catalog.register(plugin_cls)
            elif callable(symbol):
                plugin = symbol()
                if not isinstance(plugin, StrategyPlugin):
                    raise PluginLoadError(
                        f"Entrypoint '{target}' nie tworzy instancji StrategyPlugin"
                    )
                if plugin.name != strategy_name:
                    raise PluginLoadError(
                        "Nazwa pluginu '%s' nie zgadza się z manifestem (oczekiwano '%s')"
                        % (plugin.name, strategy_name)
                    )
                self._catalog.register(plugin)
            else:  # pragma: no cover - defensywnie
                raise PluginLoadError(
                    f"Entrypoint '{target}' nie zwraca pluginu strategii"
                )

            installed.append(strategy_name)

        return tuple(installed)


__all__ = ["StrategyPluginLoader", "PluginLoadError"]

