"""Loader/wspólny katalog profili ryzyka dla runtime i symulacji."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence, Type

from bot_core.config.models import RiskProfileConfig
from bot_core.risk.base import RiskProfile
from bot_core.risk.profiles.aggressive import AggressiveProfile
from bot_core.risk.profiles.balanced import BalancedProfile
from bot_core.risk.profiles.conservative import ConservativeProfile
from bot_core.risk.profiles.manual import ManualProfile


DEFAULT_PROFILE_NAMES: tuple[str, ...] = (
    "conservative",
    "balanced",
    "aggressive",
    "manual",
)


@dataclass(slots=True)
class RiskProfileLoader:
    """Buduje profile ryzyka niezależnie od kontekstu (runtime/backtest)."""

    profile_classes: Mapping[str, Type[RiskProfile]] = field(
        default_factory=lambda: {
            "conservative": ConservativeProfile,
            "balanced": BalancedProfile,
            "aggressive": AggressiveProfile,
        }
    )
    manual_profile_name: str = "manual"

    def available_profiles(self) -> Sequence[str]:
        builtin: Iterable[str] = self.profile_classes.keys()
        ordered = [*builtin]
        if self.manual_profile_name not in ordered:
            ordered.append(self.manual_profile_name)
        return tuple(ordered)

    def build(
        self, profile_name: str, *, manual_overrides: Mapping[str, object] | None = None
    ) -> RiskProfile:
        normalized = profile_name.strip().lower()
        if normalized == self.manual_profile_name:
            return self._build_manual(manual_overrides)

        try:
            profile_cls = self.profile_classes[normalized]
        except KeyError as exc:
            raise KeyError(f"Unsupported risk profile: {profile_name}") from exc
        return profile_cls()  # type: ignore[call-arg]

    def build_from_config(self, config: RiskProfileConfig) -> RiskProfile:
        profile_key = config.name.lower()
        profile_cls = self.profile_classes.get(profile_key)
        if profile_cls is not None:
            return profile_cls()

        return ManualProfile(
            name=config.name,
            max_positions=config.max_open_positions,
            max_leverage=config.max_leverage,
            drawdown_limit=config.hard_drawdown_pct,
            daily_loss_limit=config.max_daily_loss_pct,
            max_position_pct=config.max_position_pct,
            target_volatility=config.target_volatility,
            stop_loss_atr_multiple=config.stop_loss_atr_multiple,
        )

    def _build_manual(self, manual_overrides: Mapping[str, object] | None) -> ManualProfile:
        if manual_overrides is None:
            raise ValueError("Manual profile requires overrides with explicit limits")
        required = {
            "max_positions",
            "max_leverage",
            "drawdown_limit",
            "daily_loss_limit",
            "max_position_pct",
            "target_volatility",
            "stop_loss_atr_multiple",
        }
        missing = [key for key in required if key not in manual_overrides]
        if missing:
            raise ValueError(f"Missing manual profile overrides: {', '.join(missing)}")

        return ManualProfile(
            name=str(manual_overrides.get("name", self.manual_profile_name)),
            max_positions=int(manual_overrides["max_positions"]),
            max_leverage=float(manual_overrides["max_leverage"]),
            drawdown_limit=float(manual_overrides["drawdown_limit"]),
            daily_loss_limit=float(manual_overrides["daily_loss_limit"]),
            max_position_pct=float(manual_overrides["max_position_pct"]),
            target_volatility=float(manual_overrides["target_volatility"]),
            stop_loss_atr_multiple=float(manual_overrides["stop_loss_atr_multiple"]),
        )


__all__ = [
    "DEFAULT_PROFILE_NAMES",
    "RiskProfileLoader",
]
