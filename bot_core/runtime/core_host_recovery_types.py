"""Exact process-local result types for the CoreHost recovery boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StartupSubsystemRecoveryClassification(str, Enum):
    EMPTY_UNINITIALIZED = "EMPTY_UNINITIALIZED"
    INITIALIZED_DURABLE_RECOVERY_RESOLVED = "INITIALIZED_DURABLE_RECOVERY_RESOLVED"


@dataclass(frozen=True, slots=True)
class StartupSubsystemRecoveryResult:
    classification: StartupSubsystemRecoveryClassification


class CoreHostRecoveryClassification(str, Enum):
    EMPTY_UNINITIALIZED = "EMPTY_UNINITIALIZED"
    INITIALIZED_RECOVERY_COMPLETE = "INITIALIZED_RECOVERY_COMPLETE"


@dataclass(frozen=True, slots=True)
class CoreHostRecoveryResult:
    classification: CoreHostRecoveryClassification


class CoreHostStartupDisposition(str, Enum):
    SETUP_REQUIRED = "SETUP_REQUIRED"
    PROCEED_TO_LATER_STARTUP_GATES = "PROCEED_TO_LATER_STARTUP_GATES"


__all__ = [
    "CoreHostRecoveryClassification",
    "CoreHostRecoveryResult",
    "CoreHostStartupDisposition",
    "StartupSubsystemRecoveryClassification",
    "StartupSubsystemRecoveryResult",
]
