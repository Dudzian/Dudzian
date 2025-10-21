from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence

from bot_core.data.base import CacheStorage, DataSource, OHLCVRequest, OHLCVResponse
from bot_core.config.models import (
    ControllerRuntimeConfig,
    CoreConfig,
    EnvironmentConfig,
    InstrumentUniverseConfig,
    RiskProfileConfig,
)
from bot_core.exchanges.base import Environment


class InMemoryStorage(CacheStorage):
    """Prosta implementacja pamięci podręcznej do testów Daily Trend."""

    def __init__(self) -> None:
        self._store: dict[str, Mapping[str, Sequence[Sequence[float]]]] = {}
        self._metadata: dict[str, str] = {}

    def read(self, key: str) -> Mapping[str, Sequence[Sequence[float]]]:
        if key not in self._store:
            raise KeyError(key)
        return self._store[key]

    def write(self, key: str, payload: Mapping[str, Sequence[Sequence[float]]]) -> None:
        self._store[key] = payload

    def metadata(self) -> MutableMapping[str, str]:
        return self._metadata

    def latest_timestamp(self, key: str) -> float | None:
        rows = self._store.get(key, {}).get("rows")
        if not rows:
            return None
        return float(rows[-1][0])


@dataclass(slots=True)
class FixtureSource(DataSource):
    """Dostarczanie świec OHLCV z predefiniowanych danych."""

    rows: Sequence[Sequence[float]]

    def fetch_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        filtered = [row for row in self.rows if request.start <= float(row[0]) <= request.end]
        limit = request.limit or len(filtered)
        return OHLCVResponse(
            columns=("open_time", "open", "high", "low", "close", "volume"),
            rows=filtered[:limit],
        )

    def warm_cache(self, symbols: Iterable[str], intervals: Iterable[str]) -> None:  # pragma: no cover
        del symbols, intervals


def build_core_config(
    runtime: ControllerRuntimeConfig, environment_name: str, risk_profile: str
) -> CoreConfig:
    """Tworzy minimalną konfigurację CoreConfig dla testów strategii Daily Trend."""

    return CoreConfig(
        environments={
            environment_name: EnvironmentConfig(
                name=environment_name,
                exchange="paper",
                environment=Environment.PAPER,
                keychain_key="paper",
                data_cache_path="./var/data",
                risk_profile=risk_profile,
                alert_channels=(),
            )
        },
        risk_profiles={
            risk_profile: RiskProfileConfig(
                name=risk_profile,
                max_daily_loss_pct=1.0,
                max_position_pct=1.0,
                target_volatility=0.0,
                max_leverage=10.0,
                stop_loss_atr_multiple=2.0,
                max_open_positions=10,
                hard_drawdown_pct=1.0,
            )
        },
        instrument_universes={
            "default": InstrumentUniverseConfig(
                name="default",
                description="",
                instruments=(),
            )
        },
        strategies={},
        reporting={},
        sms_providers={},
        telegram_channels={},
        email_channels={},
        signal_channels={},
        whatsapp_channels={},
        messenger_channels={},
        runtime_controllers={"daily_trend": runtime},
    )


__all__ = [
    "FixtureSource",
    "InMemoryStorage",
    "build_core_config",
]
