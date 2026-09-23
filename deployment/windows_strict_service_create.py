"""Create-only boundary for the reviewed Windows SCM acceptance harness."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

ERROR_SERVICE_EXISTS = 1073


class StrictCreateResult(StrEnum):
    CREATED = "CREATED"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    FAILED = "FAILED"


class StrictCreateFailure(RuntimeError):
    def __init__(self, result: StrictCreateResult, detail: str) -> None:
        super().__init__(detail)
        self.result = result
        self.detail = detail

    def __str__(self) -> str:
        return f"{self.result}: {self.detail}"


def strict_create_service(create_service: Callable[[], None]) -> StrictCreateResult:
    """Call exactly one create-only primitive; never update an existing service."""
    try:
        create_service()
    except Exception as exc:
        winerror = getattr(exc, "winerror", None)
        if winerror == ERROR_SERVICE_EXISTS:
            raise StrictCreateFailure(
                StrictCreateResult.ALREADY_EXISTS,
                "canonical service already exists; it remains foreign and untouched",
            ) from exc
        raise StrictCreateFailure(
            StrictCreateResult.FAILED,
            "create-only service installation failed",
        ) from exc
    return StrictCreateResult.CREATED
