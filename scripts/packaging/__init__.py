"""Pakowanie artefaktów desktopowych dla powłoki Qt."""

__all__ = [
    "configure_and_build",
    "archive_bundle",
]

from .qt_bundle import configure_and_build, archive_bundle  # noqa: F401
