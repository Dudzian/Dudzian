"""Zachowana dla kompatybilności warstwa przekazująca API katalogu strategii."""

from bot_core.strategies.public import StrategyDescriptor, list_available_strategies

__all__ = ["StrategyDescriptor", "list_available_strategies"]
