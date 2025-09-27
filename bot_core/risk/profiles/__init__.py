"""Profile ryzyka."""

from bot_core.risk.profiles.aggressive import AggressiveProfile
from bot_core.risk.profiles.balanced import BalancedProfile
from bot_core.risk.profiles.conservative import ConservativeProfile
from bot_core.risk.profiles.manual import ManualProfile

__all__ = [
    "AggressiveProfile",
    "BalancedProfile",
    "ConservativeProfile",
    "ManualProfile",
]
