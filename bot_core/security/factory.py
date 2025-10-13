"""Fabryki magazynów sekretów zależne od systemu operacyjnego."""
from __future__ import annotations

import os
import platform
from pathlib import Path

from bot_core.security.base import SecretStorage, SecretStorageError
from bot_core.security.file_storage import EncryptedFileSecretStorage
from bot_core.security.fingerprint import HardwareFingerprintService, build_key_provider
from bot_core.security.keyring_storage import KeyringSecretStorage

_GUI_ENV_VARS = ("DISPLAY", "WAYLAND_DISPLAY", "DBUS_SESSION_BUS_ADDRESS")


def _has_graphical_session() -> bool:
    return any(os.environ.get(var) for var in _GUI_ENV_VARS)


def create_default_secret_storage(
    *,
    namespace: str = "dudzian.trading",
    headless_passphrase: str | None = None,
    headless_path: str | os.PathLike[str] | None = None,
) -> SecretStorage:
    """Dobiera magazyn sekretów odpowiedni dla danego systemu.

    - Windows/macOS/Linux z aktywnym środowiskiem graficznym → ``KeyringSecretStorage``
    - Linux headless → ``EncryptedFileSecretStorage`` (wymaga hasła użytkownika)
    """

    system = platform.system().lower()
    if system in {"windows", "darwin"}:
        return KeyringSecretStorage(service_name=namespace)

    if system == "linux" and not _has_graphical_session():
        if headless_passphrase is None:
            raise SecretStorageError(
                "W środowisku headless Linux wymagane jest hasło do szyfrowania magazynu sekretów."
            )
        path = Path(headless_path or Path.home() / ".dudzian" / "secrets.age")
        return EncryptedFileSecretStorage(path, headless_passphrase)

    # Domyślnie traktujemy pozostałe systemy jak środowisko desktopowe
    return KeyringSecretStorage(service_name=namespace)


def create_default_fingerprint_service(
    keys: dict[str, bytes | str],
    *,
    rotation_log: str | os.PathLike[str] = "var/licenses/fingerprint_rotation.json",
    purpose: str = "hardware-fingerprint",
    interval_days: float = 90.0,
    cpu_probe=None,
    tpm_probe=None,
    mac_probe=None,
    dongle_probe=None,
    clock=None,
) -> HardwareFingerprintService:
    """Buduje domyślną usługę fingerprintu sprzętowego."""

    provider = build_key_provider(
        keys,
        rotation_log,
        purpose=purpose,
        interval_days=interval_days,
    )
    return HardwareFingerprintService(
        provider,
        cpu_probe=cpu_probe,
        tpm_probe=tpm_probe,
        mac_probe=mac_probe,
        dongle_probe=dongle_probe,
        clock=clock,
    )


__all__ = ["create_default_secret_storage", "create_default_fingerprint_service"]
