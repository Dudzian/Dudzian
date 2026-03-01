from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def enable_security_heuristics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Security tests should exercise anti-tamper heuristics, not skip them.

    This restores default behavior (no DUDZIAN_SECURITY_SKIP) for tests/security/.
    """
    # Override global skip from tests/conftest.py for this package.
    monkeypatch.delenv("DUDZIAN_SECURITY_SKIP", raising=False)
