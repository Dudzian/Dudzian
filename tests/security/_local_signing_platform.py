"""Platform boundaries shared by production-local signing custody tests."""

from __future__ import annotations

import os

import pytest


requires_posix_custody_locking = pytest.mark.skipif(
    os.name != "posix",
    reason="production-local signing custody currently requires POSIX cross-process locking",
)
