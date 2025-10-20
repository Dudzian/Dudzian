"""Helpery rotacji sekretów zgodne z infrastrukturą ``bot_core``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.security import RotationRegistry, RotationStatus, SecretManager, SecretStorageError

__all__ = ["KeyRotationManager", "RotationState"]


@dataclass(slots=True)
class RotationState:
    """Podsumowanie aktualnego stanu rotacji dla danego wpisu."""

    status: RotationStatus
    was_rotated: bool


class KeyRotationManager:
    """Wysokopoziomowy pomocnik rotacji kluczy API i sekretów.

    Menedżer korzysta z :class:`bot_core.security.SecretManager` do odczytu i zapisu
    poświadczeń oraz z :class:`bot_core.security.RotationRegistry` do śledzenia dat
    ostatniej rotacji.  Dzięki temu produkcyjne środowisko ``KryptoLowca`` może
    współdzielić infrastrukturę z modułami ``bot_core`` bez utrzymywania
    przestarzałych managerów.
    """

    def __init__(
        self,
        secret_manager: SecretManager,
        *,
        registry: RotationRegistry | None = None,
        registry_path: str | Path | None = None,
        default_purpose: str = "trading",
        default_interval_days: float = 90.0,
    ) -> None:
        if registry is None:
            path = Path(registry_path) if registry_path is not None else Path("var/security/rotation.json")
            registry = RotationRegistry(path)
        self._secret_manager = secret_manager
        self._registry = registry
        self._purpose = default_purpose
        self._default_interval = float(default_interval_days)

    # ------------------------------------------------------------------
    # Sekcje publiczne – status rotacji
    # ------------------------------------------------------------------
    def status(
        self,
        keychain_key: str,
        *,
        purpose: str | None = None,
        interval_days: float | None = None,
        now: datetime | None = None,
    ) -> RotationStatus:
        """Zwraca status rotacji dla wskazanego klucza."""

        return self._registry.status(
            keychain_key,
            purpose or self._purpose,
            interval_days=interval_days or self._default_interval,
            now=now,
        )

    def mark_rotated(
        self,
        keychain_key: str,
        *,
        purpose: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Aktualizuje rejestr rotacji bez modyfikowania sekretów."""

        self._registry.mark_rotated(keychain_key, purpose or self._purpose, timestamp=timestamp)

    # ------------------------------------------------------------------
    # Rotacja poświadczeń giełdowych
    # ------------------------------------------------------------------
    def rotate_exchange_credentials(
        self,
        keychain_key: str,
        *,
        expected_environment: Environment,
        rotation_callback: Callable[[ExchangeCredentials], ExchangeCredentials],
        purpose: str | None = None,
        required_permissions: Sequence[str] | None = None,
        forbidden_permissions: Sequence[str] | None = None,
    ) -> ExchangeCredentials:
        """Rotuje poświadczenia giełdowe i aktualizuje rejestr rotacji."""

        current = self._secret_manager.load_exchange_credentials(
            keychain_key,
            expected_environment=expected_environment,
            purpose=purpose or self._purpose,
            required_permissions=required_permissions,
            forbidden_permissions=forbidden_permissions,
        )

        updated = rotation_callback(current)
        if not isinstance(updated, ExchangeCredentials):
            raise TypeError("rotation_callback musi zwracać ExchangeCredentials")

        self._secret_manager.store_exchange_credentials(
            keychain_key,
            updated,
            purpose=purpose or self._purpose,
        )
        self._registry.mark_rotated(keychain_key, purpose or self._purpose)
        return updated

    def ensure_exchange_rotation(
        self,
        keychain_key: str,
        *,
        expected_environment: Environment,
        rotation_callback: Callable[[ExchangeCredentials], ExchangeCredentials],
        purpose: str | None = None,
        interval_days: float | None = None,
        required_permissions: Sequence[str] | None = None,
        forbidden_permissions: Sequence[str] | None = None,
        now: datetime | None = None,
    ) -> RotationState:
        """Wymusza rotację poświadczeń, gdy wpis jest zaległy."""

        status = self.status(
            keychain_key,
            purpose=purpose,
            interval_days=interval_days,
            now=now,
        )
        rotated = False
        if status.is_due or status.is_overdue:
            self.rotate_exchange_credentials(
                keychain_key,
                expected_environment=expected_environment,
                rotation_callback=rotation_callback,
                purpose=purpose,
                required_permissions=required_permissions,
                forbidden_permissions=forbidden_permissions,
            )
            status = self.status(
                keychain_key,
                purpose=purpose,
                interval_days=interval_days,
                now=now,
            )
            rotated = True
        return RotationState(status=status, was_rotated=rotated)

    # ------------------------------------------------------------------
    # Rotacja wartości tekstowych (tokeny alertów, webhooks itp.)
    # ------------------------------------------------------------------
    def rotate_secret_value(
        self,
        keychain_key: str,
        *,
        rotation_callback: Callable[[str], str],
        purpose: str = "generic",
    ) -> str:
        """Aktualizuje arbitralny sekret i zapisuje datę rotacji."""

        current = self._secret_manager.load_secret_value(keychain_key, purpose=purpose)
        updated = rotation_callback(current)
        if not isinstance(updated, str) or not updated:
            raise ValueError("rotation_callback musi zwrócić niepusty łańcuch znaków")
        self._secret_manager.store_secret_value(keychain_key, updated, purpose=purpose)
        self._registry.mark_rotated(keychain_key, purpose)
        return updated

    def ensure_secret_rotation(
        self,
        keychain_key: str,
        *,
        rotation_callback: Callable[[str], str],
        purpose: str = "generic",
        interval_days: float | None = None,
        now: datetime | None = None,
    ) -> RotationState:
        """Zapewnia rotację sekretu tekstowego, jeśli wpis jest przeterminowany."""

        status = self.status(
            keychain_key,
            purpose=purpose,
            interval_days=interval_days,
            now=now,
        )
        rotated = False
        if status.is_due or status.is_overdue:
            self.rotate_secret_value(
                keychain_key,
                rotation_callback=rotation_callback,
                purpose=purpose,
            )
            status = self.status(
                keychain_key,
                purpose=purpose,
                interval_days=interval_days,
                now=now,
            )
            rotated = True
        return RotationState(status=status, was_rotated=rotated)

    # ------------------------------------------------------------------
    # Narzędzia diagnostyczne
    # ------------------------------------------------------------------
    def assert_secret_exists(self, keychain_key: str, *, purpose: str | None = None) -> None:
        """Upewnia się, że wpis istnieje w magazynie sekretów."""

        try:
            self._secret_manager.load_secret_value(keychain_key, purpose=purpose or "generic")
        except SecretStorageError as exc:  # pragma: no cover - ścieżka wyjątkowa
            raise SecretStorageError(
                f"Sekret '{keychain_key}' o celu '{purpose or 'generic'}' nie istnieje"
            ) from exc
