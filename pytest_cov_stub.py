"""Prosty plugin pytest umożliwiający korzystanie z opcji --cov bez pytest-cov."""
from __future__ import annotations

import importlib.util
import warnings
from typing import Any


def _pytest_cov_available() -> bool:
    return importlib.util.find_spec("pytest_cov") is not None


def pytest_addoption(parser: Any) -> None:  # pragma: no cover - hook wywoływany przez pytest
    if _pytest_cov_available():
        return
    group = parser.getgroup("cov_stub")
    group.addoption(
        "--cov",
        action="append",
        default=[],
        metavar="MODULE",
        help="(stub) opcja dostępna dla kompatybilności, brak realnego pokrycia bez pytest-cov.",
    )
    group.addoption(
        "--cov-report",
        action="append",
        default=[],
        metavar="TYPE",
        help="(stub) raport pokrycia pomijany gdy pytest-cov nie jest zainstalowany.",
    )
    group.addoption(
        "--cov-config",
        action="store",
        default=None,
        metavar="PATH",
        help="(stub) ignorowane bez pytest-cov.",
    )
    group.addoption(
        "--cov-append",
        action="store_true",
        default=False,
        help="(stub) ignorowane bez pytest-cov.",
    )
    group.addoption(
        "--cov-branch",
        action="store_true",
        default=False,
        help="(stub) ignorowane bez pytest-cov.",
    )
    group.addoption(
        "--cov-fail-under",
        action="store",
        default=None,
        type=float,
        metavar="MIN",
        help="(stub) ignorowane bez pytest-cov.",
    )
    group.addoption(
        "--no-cov",
        action="store_true",
        default=False,
        help="(stub) ignorowane bez pytest-cov.",
    )


def pytest_configure(config: Any) -> None:  # pragma: no cover - hook wywoływany przez pytest
    if _pytest_cov_available():
        return
    if getattr(config.option, "cov", None) or getattr(config.option, "cov_report", None):
        warnings.warn(
            "pytest-cov nie jest zainstalowany; opcje pokrycia zostają zignorowane.",
            RuntimeWarning,
            stacklevel=0,
        )
