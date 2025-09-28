"""Monitor luk danych OHLCV z integracją alertów."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Mapping, MutableMapping, Sequence

from bot_core.alerts import AlertMessage, AlertRouter

_MILLISECONDS_IN_MINUTE = 60_000


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _interval_to_minutes(interval: str) -> int:
    mapping = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "6h": 360,
        "8h": 480,
        "12h": 720,
        "1d": 1440,
        "3d": 4320,
        "1w": 10_080,
        "1M": 43_200,
    }
    try:
        return mapping[interval]
    except KeyError as exc:  # pragma: no cover - walidacja configu na starcie
        raise ValueError(f"Nieobsługiwany interwał: {interval}") from exc


@dataclass(slots=True)
class GapAlertPolicy:
    """Parametry eskalacji luk danych."""

    warning_gap_minutes: Mapping[str, int]
    incident_threshold_count: int = 5
    incident_window_minutes: int = 10
    sms_escalation_minutes: int = 15

    def warning_threshold_minutes(self, interval: str) -> int:
        minutes = self.warning_gap_minutes.get(interval)
        if minutes is not None:
            return max(1, int(minutes))
        # domyślnie przyjmujemy dwukrotność interwału jako bezpieczne okno
        return max(1, _interval_to_minutes(interval) * 2)


@dataclass(slots=True)
class _GapState:
    warnings: list[datetime] = field(default_factory=list)
    incident_open: bool = False
    incident_open_at: datetime | None = None
    sms_escalated: bool = False

    def register_warning(self, timestamp: datetime, *, window: timedelta) -> int:
        self.warnings.append(timestamp)
        cutoff = timestamp - window
        self.warnings = [entry for entry in self.warnings if entry >= cutoff]
        return len(self.warnings)

    def reset(self) -> None:
        self.warnings.clear()
        self.incident_open = False
        self.incident_open_at = None
        self.sms_escalated = False


@dataclass(slots=True)
class DataGapIncidentTracker:
    """Pilnuje luk w danych OHLCV i wysyła alerty zgodnie z polityką eskalacji."""

    router: AlertRouter
    metadata_provider: Callable[[], MutableMapping[str, str]]
    policy: GapAlertPolicy
    environment_name: str
    exchange: str
    clock: Callable[[], datetime] = _utc_now

    _states: dict[tuple[str, str], _GapState] = field(default_factory=dict, init=False, repr=False)

    def handle_summaries(
        self,
        *,
        interval: str,
        summaries: Sequence[object],
        as_of_ms: int,
    ) -> None:
        if not summaries:
            return
        metadata = self.metadata_provider()
        for summary in summaries:
            symbol = getattr(summary, "symbol", None)
            if not symbol:
                continue
            state = self._states.setdefault((symbol, interval), _GapState())
            last_ts_key = f"last_timestamp::{symbol}::{interval}"
            row_count_key = f"row_count::{symbol}::{interval}"
            last_ts_raw = metadata.get(last_ts_key)
            row_count_raw = metadata.get(row_count_key)

            if last_ts_raw is None:
                # brak danych – traktujemy jak incydent krytyczny
                self._emit_alert(
                    severity="critical",
                    title=f"Brak danych OHLCV {symbol} {interval}",
                    body=(
                        "Manifest nie posiada wpisu last_timestamp – należy zweryfikować pipeline backfillu."
                    ),
                    context={
                        "environment": self.environment_name,
                        "exchange": self.exchange,
                        "symbol": symbol,
                        "interval": interval,
                        "row_count": str(row_count_raw or "0"),
                    },
                )
                continue

            try:
                last_ts_ms = int(float(last_ts_raw))
            except (TypeError, ValueError):
                self._emit_alert(
                    severity="critical",
                    title=f"Uszkodzona metadana OHLCV {symbol} {interval}",
                    body="Wartość last_timestamp nie jest liczbą – konieczna ręczna interwencja.",
                    context={
                        "environment": self.environment_name,
                        "exchange": self.exchange,
                        "symbol": symbol,
                        "interval": interval,
                        "raw_value": str(last_ts_raw),
                    },
                )
                continue

            gap_ms = max(0, as_of_ms - last_ts_ms)
            gap_minutes = gap_ms / _MILLISECONDS_IN_MINUTE
            warning_threshold = self.policy.warning_threshold_minutes(interval)

            now = self.clock()
            if gap_minutes < warning_threshold:
                if state.incident_open:
                    duration = (
                        (now - state.incident_open_at).total_seconds() / 60
                        if state.incident_open_at
                        else 0
                    )
                    self._emit_alert(
                        severity="info",
                        title=f"Incydent zamknięty – luka danych {symbol} {interval}",
                        body=(
                            "Dane OHLCV zostały uzupełnione. Zamykam incydent i resetuję licznik ostrzeżeń."
                        ),
                        context={
                            "environment": self.environment_name,
                            "exchange": self.exchange,
                            "symbol": symbol,
                            "interval": interval,
                            "incident_minutes": f"{duration:.1f}",
                            "gap_minutes": f"{gap_minutes:.1f}",
                            "row_count": str(row_count_raw or "0"),
                        },
                    )
                state.reset()
                continue

            window = timedelta(minutes=max(1, self.policy.incident_window_minutes))
            warn_count = state.register_warning(now, window=window)

            context = {
                "environment": self.environment_name,
                "exchange": self.exchange,
                "symbol": symbol,
                "interval": interval,
                "gap_minutes": f"{gap_minutes:.1f}",
                "row_count": str(row_count_raw or "0"),
                "last_timestamp": datetime.fromtimestamp(last_ts_ms / 1000, tz=timezone.utc).isoformat(),
            }

            if not state.incident_open and warn_count >= self.policy.incident_threshold_count:
                state.incident_open = True
                state.incident_open_at = now
                state.sms_escalated = False
                self._emit_alert(
                    severity="critical",
                    title=f"INCIDENT – luka danych {symbol} {interval}",
                    body=(
                        "Wykryto powtarzające się luki w danych OHLCV. "
                        "Incydent został otwarty i wymaga ręcznej analizy."
                    ),
                    context={
                        **context,
                        "warnings_in_window": str(warn_count),
                        "window_minutes": str(self.policy.incident_window_minutes),
                    },
                )
                continue

            if state.incident_open:
                assert state.incident_open_at is not None
                elapsed = (now - state.incident_open_at).total_seconds() / 60
                if (
                    not state.sms_escalated
                    and elapsed >= max(1, self.policy.sms_escalation_minutes)
                ):
                    state.sms_escalated = True
                    self._emit_alert(
                        severity="critical",
                        title=f"Eskalacja SMS – luka danych {symbol} {interval}",
                        body=(
                            "Incydent trwa dłużej niż zakładany próg eskalacji. "
                            "Wysyłam powiadomienie SMS zgodnie z polityką."
                        ),
                        context={**context, "incident_minutes": f"{elapsed:.1f}"},
                    )
                continue

            # Ostrzeżenie Telegram – pojedynczy alert o dłuższej luce
            self._emit_alert(
                severity="warning",
                title=f"Luka danych {symbol} {interval}",
                body=(
                    "Brak świec OHLCV od ponad wyznaczonego progu. Monitoruję dalsze próby synchronizacji."
                ),
                context={**context, "warnings_in_window": str(warn_count)},
            )

    def _emit_alert(
        self,
        *,
        severity: str,
        title: str,
        body: str,
        context: Mapping[str, str],
    ) -> None:
        message = AlertMessage(
            category="data.ohlcv",
            title=title,
            body=body,
            severity=severity,
            context=dict(context),
        )
        self.router.dispatch(message)


__all__ = ["GapAlertPolicy", "DataGapIncidentTracker"]

