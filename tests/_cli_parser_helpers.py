"""Wspólne narzędzia do introspekcji parserów CLI w testach."""
from __future__ import annotations

from typing import Callable, Iterable

import argparse


def parser_supports(parser_factory: Callable[[], argparse.ArgumentParser], *flags: str) -> bool:
    """Sprawdza, czy parser CLI udostępnia wszystkie podane przełączniki."""

    parser = parser_factory()
    actions: Iterable[argparse.Action] = getattr(parser, "_actions", ())
    option_strings: set[str] = set()
    for action in actions:
        option_strings.update(getattr(action, "option_strings", []) or [])
    return all(flag in option_strings for flag in flags)


__all__ = ["parser_supports"]
