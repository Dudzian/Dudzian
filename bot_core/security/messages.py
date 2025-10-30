"""Wspólne struktury opisujące rezultaty walidacji bezpieczeństwa."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Severity = Literal["error", "warning"]


@dataclass(slots=True)
class ValidationMessage:
    """Opis pojedynczego scenariusza walidacyjnego prezentowany w UI."""

    code: str
    message: str
    severity: Severity
    hint: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        payload: dict[str, str | None] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.hint:
            payload["hint"] = self.hint
        return payload


def make_error(code: str, message: str, *, hint: str | None = None) -> ValidationMessage:
    """Buduje komunikat o błędzie."""

    return ValidationMessage(code=code, message=message, severity="error", hint=hint)


def make_warning(code: str, message: str, *, hint: str | None = None) -> ValidationMessage:
    """Buduje komunikat ostrzegawczy."""

    return ValidationMessage(code=code, message=message, severity="warning", hint=hint)


__all__ = ["ValidationMessage", "make_error", "make_warning"]
