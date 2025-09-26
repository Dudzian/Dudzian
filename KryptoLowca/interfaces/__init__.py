"""Warstwa interfejsów użytkownika (API i aplikacja desktopowa)."""

from .api import TradingAPI
from .desktop import DesktopInterface

__all__ = ["TradingAPI", "DesktopInterface"]
