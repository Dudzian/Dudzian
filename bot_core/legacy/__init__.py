"""Kompatybilne shimy zapewniające dostęp do legacy komponentów GUI."""

from __future__ import annotations

from .security_manager import SecurityError, SecurityManager

__all__ = ["SecurityManager", "SecurityError"]

