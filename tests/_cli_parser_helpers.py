"""Wspólne narzędzia do introspekcji parserów CLI w testach."""
from __future__ import annotations

from typing import Callable, Iterable

import argparse


def _collect_option_strings(parser: argparse.ArgumentParser) -> set[str]:
    """Zbiera wszystkie przełączniki z parsera oraz jego subparserów."""

    option_strings: set[str] = set()
    actions: Iterable[argparse.Action] = getattr(parser, "_actions", ())
    for action in actions:
        option_strings.update(getattr(action, "option_strings", []) or [])
        if isinstance(action, argparse._SubParsersAction):  # pragma: no branch - małe drzewo
            for sub_parser in action.choices.values():
                option_strings.update(_collect_option_strings(sub_parser))
    return option_strings


def parser_supports(parser_factory: Callable[[], argparse.ArgumentParser], *flags: str) -> bool:
    """Sprawdza, czy parser CLI udostępnia wszystkie podane przełączniki."""

    parser = parser_factory()
    option_strings = _collect_option_strings(parser)
    return all(flag in option_strings for flag in flags)


__all__ = ["parser_supports"]
