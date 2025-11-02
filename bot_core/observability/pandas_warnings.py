"""Narzędzia do przechwytywania ostrzeżeń pandas i raportowania ich do obserwowalności."""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Tuple

import pandas as pd

from bot_core.observability.metrics import MetricsRegistry, observe_pandas_warning

_SETTING_WITH_COPY_WARNING = getattr(pd.errors, "SettingWithCopyWarning", None)

MONITORED_PANDAS_WARNING_CATEGORIES: Tuple[type[Warning], ...] = tuple(
    warning
    for warning in (
        pd.errors.PerformanceWarning,
        _SETTING_WITH_COPY_WARNING,
    )
    if isinstance(warning, type) and issubclass(warning, Warning)
)


def is_relevant_pandas_warning(
    warning: warnings.WarningMessage,
    monitored_categories: Tuple[type[Warning], ...] = MONITORED_PANDAS_WARNING_CATEGORIES,
) -> bool:
    """Sprawdź, czy ostrzeżenie powinno być raportowane."""
    category = getattr(warning, "category", None)
    if category is None:
        return False

    if monitored_categories and any(
        issubclass(category, monitored) for monitored in monitored_categories
    ):
        return True

    module_name = getattr(category, "__module__", "")
    if module_name.startswith("pandas"):
        return True

    if issubclass(category, FutureWarning):
        message_text = str(getattr(warning, "message", "")).lower()
        if "pandas" in message_text:
            return True

    return False


@contextmanager
def capture_pandas_warnings(
    logger: logging.Logger,
    *,
    component: str,
    monitored_categories: Tuple[type[Warning], ...] = MONITORED_PANDAS_WARNING_CATEGORIES,
    registry: MetricsRegistry | None = None,
) -> Iterator[None]:
    """Rejestruj ostrzeżenia pandas w logach i metrykach w zadanym komponencie."""

    with warnings.catch_warnings(record=True) as caught:
        for category in monitored_categories:
            warnings.simplefilter("always", category=category)
        warnings.simplefilter("always", category=FutureWarning)
        yield

    if not caught:
        return

    for warning in caught:
        if not is_relevant_pandas_warning(warning, monitored_categories):
            continue

        message_text = str(getattr(warning, "message", "")).strip()
        filename = getattr(warning, "filename", "") or "<unknown>"
        location = f"{Path(filename).name}:{getattr(warning, 'lineno', '?')}"

        logger.warning(
            "Pandas warning captured in %s (%s): %s",
            component,
            location,
            message_text,
        )

        observe_pandas_warning(
            component=component,
            category=getattr(warning, "category", Warning).__name__,
            message=message_text,
            registry=registry,
        )


__all__ = [
    "MONITORED_PANDAS_WARNING_CATEGORIES",
    "capture_pandas_warnings",
    "is_relevant_pandas_warning",
]
