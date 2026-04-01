"""Compat helper for TOML parser used across deploy.packaging."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - compatibility for local py<3.11 test env
    import tomli as tomllib

__all__ = ["tomllib"]
