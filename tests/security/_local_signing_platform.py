"""Platform boundaries shared by production-local signing custody tests."""

from __future__ import annotations

import os

import pytest


requires_native_custody_locking = pytest.mark.skipif(
    os.name not in {"posix", "nt"},
    reason="production-local signing custody requires POSIX or Windows native process locking",
)
