"""Fabryki magazynów sekretów zależne od systemu operacyjnego."""
from __future__ import annotations

import os
import platform
from pathlib import Path

from bot_core.security.base import SecretStorage, SecretStorageError
from bot_core.security.file_storage import EncryptedFileSecretStorage
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


__all__ = ["create_default_secret_storage"]
