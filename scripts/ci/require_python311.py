#!/usr/bin/env python
"""Fail-closed Python runtime guard for desktop lock generation."""

from __future__ import annotations

import sys
from typing import Sequence


def require_python311(version_info: Sequence[int] = sys.version_info) -> None:
    if tuple(version_info[:2]) != (3, 11):
        raise SystemExit(
            f"desktop lock generation requires Python 3.11; got {version_info[0]}.{version_info[1]}"
        )


if __name__ == "__main__":
    require_python311()
