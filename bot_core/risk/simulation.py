"""Symulacje profili ryzyka, stres testy paper→live oraz scenariusze Parquet."""
from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median, pstdev
from typing import Iterable, Mapping, MutableMapping, Sequence, TYPE_CHECKING

# --- Importy opcjonalne ------------------------------------------------------
try:  # pyarrow może nie być dostępny
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pa = None  # type: ignore
    pq = None  # type: ignore

# konfiguracja i profile (różne gałęzie)
try:
    from bot_core.config import load_core_config
    from bot_core.risk.factory import build_risk_profile_from_config
except Exception:  # pragma: no cover
    load_core_config = None  # type: ignore[misc,assignment]
    build_risk_profile_from_config = None  # type: ignore[misc,assignment]

# gotowe profile (gałąź "main")
try:  # pragma: no cover
    from bot_core.risk.profiles import (
        AggressiveProfile,
        BalancedProfile,
        ConservativeProfile,
        ManualProfile,
    )
except Exception:  # pragma: no cover
    AggressiveProfile = BalancedProfile = ConservativeProfile = ManualProfile = None  # type: ignore

# API exchanges (gałąź "main")
try:  # pragma: no cover
    from bot_core.exchanges.base import AccountSnapshot, OrderRequest
except Exception:  # pragma: no cover
    AccountSnapshot = OrderRequest = None  # type: ignore

# API ryzyka (gałąź "main")
try:  # pragma: no cover
    from bot_core.risk.base import RiskEngine, RiskProfile
except Exception:  # pragma: no cover
    RiskEngine = RiskProfile = None  # type: ignore

if TYPE_CHECKING:  # tylko dla typowania w IDE
    from bot_core.risk.base import RiskProfile as _RiskProfileT  # noqa: F401

_LOGGER = logging.getLogger(__name__)

# --- Stałe -------------------------------------------------------------------
DEFAULT_PROFILES: Sequence[str] = ("conservative", "balanced", "aggressive", "manual")
_BASE_EQUITY_DEFAULT = 100_000.0

# PDF
_PDF_LINE_HEIGHT = 14
_PDF_PAGE_WIDTH = 612
_PDF_PAGE_HEIGHT = 792

# smoke scenariusze (gałąź "main")
_SMOKE_BASE_TIMESTAMP = datetime(2024, 1, 1, 12, 0, 0)
DEFAULT_SMOKE_SCENARIOS: tuple[Mapping[str, object], ...] = (
    {
        "profile": "conservative",
        "timestamp": _SMOKE_BASE_TIMESTAMP.isoformat(),
        "symbol": "BTCUSDT",
        "side": "buy",
        "price": 10000.0,
        "quantity": 0.01,
        "total_equity": 10000.0,
        "available_margin": 8000.0,
        "maintenance_margin": 100.0,
        "atr": 150.0,
        "stop_price": 9850.0,
        "position_value": 100.0,
        "pnl": 25.0,
    },
    {
        "profile": "conservative",
        "timestamp": (_SMOKE_BASE_TIMESTAMP + timedelta(minutes=1)).isoformat(),
        "symbol": "BTCUSDT",
        "side": "buy",
        "price": 10000.0,
        "quantity": 0.1,
        "total_equity": 10000.0,
        "available_margin": 8000.0,
        "maintenance_margin": 100.0,
        "atr": 150.0,
        "stop_price": 9850.0,
        "position_value": 1000.0,
        "pnl": -10.0,
    },
    {
        "profile": "balanced",
        "timestamp": (_SMOKE_BASE_TIMESTAMP + timedelta(minutes=2)).isoformat(),
        "symbol": "ETHUSDT",
        "side": "buy",
        "price": 100.0,
        "quantity": 1.0,
        "total_equity": 20000.0,
        "available_margin": 15000.0,
        "maintenance_margin": 100.0,
        "atr": 5.0,
        "stop_price": 98.0,
        "position_value": 100.0,
        "pnl": 0.0,
    },
    {
        "profile": "aggressive",
        "timestamp": (_SMOKE_BASE_TIMESTAMP + timedelta(minutes=3)).isoformat(),
        "symbol": "SOLUSDT",
        "side": "buy",
        "price": 150.0,
        "quantity": 0.5,
        "total_equity": 5000.0,
        "available_margin": 4000.0,
        "maintenance_margin": 100.0,
        "atr": 8.0,
        "stop_price": 134.0,
        "position_value": 75.0,
        "pnl": 12.0,
    },
    {
        "profile": "manual",
        "timestamp": (_SMOKE_BASE_TIMESTAMP + timedelta(minutes=4)).isoformat(),
        "symbol": "ADAUSDT",
        "side": "buy",
        "price": 50.0,
        "quantity": 2.0,
        "total_equity": 10000.0,
        "available_margin": 9000.0,
        "maintenance_margin": 150.0,
        "atr": 4.0,
        "stop_price": 45.0,
        "position_value": 100.0,
        "pnl": 5.0,
    },
)

# --- Modele danych -----------------------------------------------------------
@dataclass(slots=True)
class Candle:
    """Pojedyncza świeca OHLCV wykorzystywana w symulacji."""
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class StressTestResult:
    """Wynik pojedynczego stres testu."""
    name: str
    status: str
    metrics: MutableMapping[str, float | str] = field(default_factory=dict)
    notes: str | None = None

    def to_mapping(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "metrics": dict(self.metrics),
        }
        if self.notes:
            payload["notes"] = self.notes
        return payload

    def is_failure(self) -> bool:
        return self.status.lower() not in {"ok", "passed", "success"}


@dataclass(slots=True)
class ProfileSimulationResult:
    """Metryki wynikowe dla profilu ryzyka (symulacja OHLCV)."""
    profile: str
    base_equity: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    worst_daily_loss_pct: float
    realized_volatility: float
    breaches: Sequence[str] = field(default_factory=tuple)
    stress_tests: Sequence[StressTestResult] = field(default_factory=tuple)
    sample_size: int = 0

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "profile": self.profile,
            "base_equity": self.base_equity,
            "final_equity": self.final_equity,
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "worst_daily_loss_pct": self.worst_daily_loss_pct,
            "realized_volatility": self.realized_volatility,
            "breaches": list(self.breaches),
            "stress_tests": [result.to_mapping() for result in self.stress_tests],
            "sample_size": self.sample_size,
        }

    def has_failures(self) -> bool:
        return bool(self.breaches) or any(result.is_failure() for result in self.stress_tests)


@dataclass(slots=True)
class RiskScenarioResult:
    """Zbiorczy wynik symulacji scenariuszy (Parquet/engine)."""
    profile: str
    total_orders: int
    accepted_orders: int
    rejected_orders: int
    force_liquidation: bool
    rejection_reasons: Sequence[str]
    decisions: Sequence[Mapping[str, object]]
    final_state: Mapping[str, object] | None

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "profile": self.profile,
            "total_orders": self.total_orders,
            "accepted_orders": self.accepted_orders,
            "rejected_orders": self.rejected_orders,
            "force_liquidation": self.force_liquidation,
            "rejection_reasons": list(self.rejection_reasons),
            "decisions": list(self.decisions),
            "final_state": dict(self.final_state) if self.final_state is not None else None,
        }


@dataclass(slots=True)
class RiskSimulationReport:
    """Zbiorczy raport z symulacji profili ryzyka (OHLCV)."""
    generated_at: str
    base_equity: float
    profiles: Sequence[ProfileSimulationResult]
    synthetic_data: bool = False

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "generated_at": self.generated_at,
            "base_equity": self.base_equity,
            "synthetic_data": self.synthetic_data,
            "profiles": [p.to_mapping() for p in self.profiles],
            "breach_count": sum(len(p.breaches) for p in self.profiles),
            "stress_failures": sum(sum(1 for r in p.stress_tests if r.is_failure()) for p in self.profiles),
        }

    def has_failures(self) -> bool:
        return any(p.has_failures() for p in self.profiles)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_mapping(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

    def write_pdf(self, path: Path) -> None:
        lines: list[str] = [
            f"Risk Simulation Report — {self.generated_at}",
            f"Base equity: {self.base_equity:,.2f} USD",
            f"Synthetic dataset fallback: {'yes' if self.synthetic_data else 'no'}",
        ]
        for profile in self.profiles:
            lines.append("")
            lines.append(f"Profile: {profile.profile}")
            lines.append(
                f"  Final equity: {profile.final_equity:,.2f} USD (return {profile.total_return_pct * 100:.2f}%)"
            )
            lines.append(
                f"  Max drawdown: {profile.max_drawdown_pct * 100:.2f}% — worst daily loss {profile.worst_daily_loss_pct * 100:.2f}%"
            )
            lines.append(f"  Realized volatility: {profile.realized_volatility * 100:.2f}%")
            lines.append(f"  Breaches: {', '.join(profile.breaches) if profile.breaches else 'none'}")
            for stress in profile.stress_tests:
                status = stress.status.upper()
                severity = stress.metrics.get("severity")
                if severity:
                    lines.append(f"  Stress {stress.name}: {status} (severity {severity})")
                else:
                    lines.append(f"  Stress {stress.name}: {status}")
                if stress.notes:
                    lines.append(f"    Notes: {stress.notes}")
        _write_simple_pdf(path, lines)


@dataclass(slots=True)
class RiskSimulationSuite:
    """Raport zbiorczy dla scenariuszy Parquet/engine."""
    scenarios: Sequence[RiskScenarioResult]
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "scenarios": [s.to_mapping() for s in self.scenarios],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_mapping(), indent=indent, sort_keys=True)

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=indent), encoding="utf-8")
        return target

    def render_pdf(self, path: str | Path) -> Path:
        lines: list[str] = [
            "Paper Labs – Risk Simulation Report",
            f"Generated at: {self.generated_at.isoformat()}",
            "",
        ]
        for scenario in self.scenarios:
            lines.extend(
                [
                    f"Profile: {scenario.profile}",
                    f"Total orders: {scenario.total_orders}",
                    f"Accepted: {scenario.accepted_orders}",
                    f"Rejected: {scenario.rejected_orders}",
                    f"Forced liquidation: {'yes' if scenario.force_liquidation else 'no'}",
                ]
            )
            if scenario.rejection_reasons:
                lines.append("Rejection reasons:")
                for reason in scenario.rejection_reasons:
                    lines.append(f" • {reason}")
            lines.append("")
        _write_simple_pdf(path, lines)
        return Path(path)


@dataclass(slots=True)
class SimulationSettings:
    """Parametry wejściowe symulacji OHLCV."""
    base_equity: float = _BASE_EQUITY_DEFAULT
    max_bars: int | None = 720


@dataclass(slots=True)
class SimulationOrder:
    """Pojedynczy rekord wejściowy używany podczas symulacji (Parquet/engine)."""
    profile: str
    timestamp: datetime
    symbol: str
    side: str
    price: float
    quantity: float
    total_equity: float
    available_margin: float
    maintenance_margin: float
    atr: float | None = None
    stop_price: float | None = None
    position_value: float | None = None
    pnl: float | None = None

    def to_order_request(self):
        if OrderRequest is None:
            raise RuntimeError("OrderRequest is not available in this build.")
        return OrderRequest(
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            order_type="limit",
            price=self.price,
            stop_price=self.stop_price,
            atr=self.atr,
            metadata={"source": "risk_simulation"},
        )

    def to_account_snapshot(self):
        if AccountSnapshot is None:
            raise RuntimeError("AccountSnapshot is not available in this build.")
        balances: MutableMapping[str, float] = {"USD": float(self.total_equity)}
        return AccountSnapshot(
            balances=balances,
            total_equity=float(self.total_equity),
            available_margin=float(self.available_margin),
            maintenance_margin=float(self.maintenance_margin),
        )

    def to_fill_arguments(self) -> Mapping[str, object]:
        position_value = (
            float(self.position_value)
            if self.position_value is not None
            else abs(float(self.quantity)) * float(self.price)
        )
        return {
            "symbol": self.symbol,
            "side": self.side,
            "position_value": position_value,
            "pnl": float(self.pnl or 0.0),
            "timestamp": self.timestamp,
        }

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "profile": self.profile,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "price": float(self.price),
            "quantity": float(self.quantity),
            "total_equity": float(self.total_equity),
            "available_margin": float(self.available_margin),
            "maintenance_margin": float(self.maintenance_margin),
            "atr": None if self.atr is None else float(self.atr),
            "stop_price": None if self.stop_price is None else float(self.stop_price),
            "position_value": None if self.position_value is None else float(self.position_value),
            "pnl": None if self.pnl is None else float(self.pnl),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "SimulationOrder":
        timestamp_raw = data.get("timestamp")
        timestamp = _parse_timestamp(timestamp_raw)
        return cls(
            profile=str(data["profile"]).lower(),
            timestamp=timestamp,
            symbol=str(data["symbol"]),
            side=str(data["side"]),
            price=float(data["price"]),
            quantity=float(data["quantity"]),
            total_equity=float(data["total_equity"]),
            available_margin=float(data["available_margin"]),
            maintenance_margin=float(data["maintenance_margin"]),
            atr=_coerce_optional_float(data.get("atr")),
            stop_price=_coerce_optional_float(data.get("stop_price")),
            position_value=_coerce_optional_float(data.get("position_value")),
            pnl=_coerce_optional_float(data.get("pnl")),
        )


# --- Funkcje pomocnicze (czas/konwersje) ------------------------------------
def _parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.utcfromtimestamp(float(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("timestamp field cannot be empty")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    raise TypeError(f"Unsupported timestamp type: {type(value)!r}")


def _coerce_optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --- Helpery do pobierania wartości profilu (zgodność nazw/metod) -----------
def _profile_float(profile: object, names: Sequence[str], default: float = 0.0) -> float:
    """Spróbuj kolejno metod/atrybutów – zwróć pierwszą poprawną wartość float."""
    for name in names:
        if hasattr(profile, name):
            obj = getattr(profile, name)
            try:
                return float(obj()) if callable(obj) else float(obj)
            except Exception:
                continue
    return float(default)


def _drawdown_limit(profile: RiskProfile) -> float:
    return _profile_float(profile, ("drawdown_limit", "drawdown_limit_pct"), 0.0)


def _daily_loss_limit(profile: RiskProfile) -> float:
    return _profile_float(profile, ("daily_loss_limit", "max_daily_loss_limit"), 0.0)


def _target_volatility(profile: RiskProfile) -> float:
    return _profile_float(profile, ("target_volatility", "target_volatility_pct"), 0.0)


def _stop_loss_atr_multiple(profile: RiskProfile) -> float:
    return _profile_float(profile, ("stop_loss_atr_multiple", "stop_loss_atr_mult"), 1.0)


def _max_leverage(profile: RiskProfile) -> float:
    return _profile_float(profile, ("max_leverage", "leverage_limit"), 1.0)


def _max_position_exposure(profile: RiskProfile) -> float:
    return _profile_float(profile, ("max_position_exposure", "max_position_pct"), 0.0)


# --- Loader świec Parquet (OHLCV) -------------------------------------------
class MarketDatasetLoader:
    """Loader świec OHLCV z plików Parquet (year=*/month=*/data.parquet)."""

    def __init__(self, root: Path, *, namespace: str) -> None:
        self._root = Path(root)
        self._namespace = namespace

    def load(self, symbol: str, interval: str) -> Sequence[Candle]:
        if pq is None:
            raise RuntimeError("pyarrow nie jest dostępny — nie można wczytać danych Parquet")
        partition = self._root / self._namespace / symbol / interval
        if not partition.exists():
            raise FileNotFoundError(partition)
        candles: list[Candle] = []
        for year_dir in sorted(partition.glob("year=*")):
            for month_dir in sorted(year_dir.glob("month=*")):
                file_path = month_dir / "data.parquet"
                if not file_path.exists():
                    continue
                table = pq.read_table(file_path)
                if table.num_rows == 0:
                    continue
                data = table.to_pydict()
                open_times = data.get("open_time", [])
                highs = data.get("high", [])
                lows = data.get("low", [])
                opens = data.get("open", [])
                closes = data.get("close", [])
                volumes = data.get("volume", [])
                for idx in range(len(open_times)):
                    candles.append(
                        Candle(
                            timestamp_ms=int(open_times[idx]),
                            open=float(opens[idx]),
                            high=float(highs[idx]),
                            low=float(lows[idx]),
                            close=float(closes[idx]),
                            volume=float(volumes[idx]),
                        )
                    )
        candles.sort(key=lambda c: c.timestamp_ms)
        if not candles:
            raise FileNotFoundError(partition)
        return candles


# --- Generator syntetycznych świec ------------------------------------------
def _generate_synthetic_candles(*, bars: int, seed: int = 42, interval_seconds: int = 3600) -> Sequence[Candle]:
    rng = random.Random(seed)
    candles: list[Candle] = []
    timestamp = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    price = 21_000.0
    for index in range(bars):
        drift = math.sin(index / 12.0) * 0.004
        shock = (rng.random() - 0.5) * 0.01
        change = drift + shock
        open_price = price
        close_price = max(1.0, open_price * (1 + change))
        high = max(open_price, close_price) * (1 + abs(shock) * 0.3)
        low = min(open_price, close_price) * (1 - abs(shock) * 0.3)
        volume = 1_000 + rng.random() * 600
        candles.append(
            Candle(
                timestamp_ms=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=volume,
            )
        )
        price = close_price
        timestamp += interval_seconds * 1000
    return candles


# --- Metryki z serii ---------------------------------------------------------
def _compute_returns(candles: Sequence[Candle]) -> list[float]:
    returns: list[float] = []
    for idx in range(1, len(candles)):
        prev = candles[idx - 1]
        current = candles[idx]
        if prev.close <= 0:
            continue
        returns.append((current.close - prev.close) / prev.close)
    return returns


def _compute_daily_losses(candles: Sequence[Candle], pnl_series: Sequence[float], base_equity: float) -> float:
    worst_loss = 0.0
    equity = base_equity
    by_day: dict[str, float] = {}
    for candle, pnl in zip(candles[1:], pnl_series):
        day = datetime.fromtimestamp(candle.timestamp_ms / 1000.0, tz=timezone.utc).date().isoformat()
        by_day.setdefault(day, equity)
        equity += pnl
        change = equity - by_day[day]
        if by_day[day] > 0 and change < 0:
            loss_pct = abs(change) / by_day[day]
            worst_loss = max(worst_loss, loss_pct)
    return worst_loss


def _compute_max_drawdown(pnl_series: Sequence[float], base_equity: float) -> float:
    equity = base_equity
    peak = base_equity
    max_drawdown = 0.0
    for pnl in pnl_series:
        equity += pnl
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def _realized_volatility(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    return math.sqrt(pstdev(returns)) * math.sqrt(len(returns)) if len(returns) > 1 else abs(returns[0])


# --- Stres testy -------------------------------------------------------------
def _run_flash_crash_test(profile: RiskProfile, candles: Sequence[Candle], base_equity: float) -> StressTestResult:
    max_drop = 0.0
    for idx in range(1, len(candles)):
        prev_close = candles[idx - 1].close
        current_low = candles[idx].low
        if prev_close <= 0:
            continue
        drop = (current_low - prev_close) / prev_close
        max_drop = min(max_drop, drop)
    severity = abs(max_drop)
    limit = _drawdown_limit(profile)
    metrics: dict[str, float] = {
        "max_low_gap_pct": severity,
        "severity": severity / max(limit, 1e-6),
    }
    allowed = severity <= limit
    status = "passed" if allowed else "failed"
    notes = None if allowed else f"Observed crash {severity * 100:.2f}% breaches drawdown limit {limit * 100:.2f}%"
    return StressTestResult(name="flash_crash", status=status, metrics=metrics, notes=notes)


def _run_liquidity_dryout_test(profile: RiskProfile, candles: Sequence[Candle]) -> StressTestResult:
    volumes = [candle.volume for candle in candles]
    if not volumes:
        return StressTestResult(
            name="dry_liquidity",
            status="failed",
            metrics={"severity": float("inf")},
            notes="Brak danych wolumenu dla symulacji",
        )
    median_volume = median(volumes)
    threshold = median_volume * 0.2
    low_volume_ratio = sum(1 for v in volumes if v <= threshold) / len(volumes)
    allowed_ratio = max(0.05, 0.35 - _target_volatility(profile))
    metrics: dict[str, float] = {
        "low_volume_ratio": low_volume_ratio,
        "allowed_ratio": allowed_ratio,
        "severity": low_volume_ratio / max(allowed_ratio, 1e-6),
    }
    status = "passed" if low_volume_ratio <= allowed_ratio else "failed"
    notes = None if status == "passed" else f"{low_volume_ratio * 100:.2f}% słabych wolumenów przekracza próg {allowed_ratio * 100:.2f}%"
    return StressTestResult(name="dry_liquidity", status=status, metrics=metrics, notes=notes)


def _run_latency_spike_test(profile: RiskProfile, candles: Sequence[Candle]) -> StressTestResult:
    spreads = []
    for candle in candles:
        if candle.open <= 0:
            continue
        spreads.append((candle.high - candle.low) / candle.open)
    if not spreads:
        return StressTestResult(
            name="latency_spike",
            status="failed",
            metrics={"severity": float("inf")},
            notes="Brak danych spreadu dla symulacji",
        )
    max_spread = max(spreads)
    threshold = 0.01 + _stop_loss_atr_multiple(profile) * 0.015
    metrics: dict[str, float] = {
        "max_spread_pct": max_spread,
        "threshold": threshold,
        "severity": max_spread / max(threshold, 1e-6),
    }
    status = "passed" if max_spread <= threshold else "failed"
    notes = None if status == "passed" else f"Spread {max_spread * 100:.2f}% przekracza próg {threshold * 100:.2f}% – ryzyko poślizgu"
    return StressTestResult(name="latency_spike", status=status, metrics=metrics, notes=notes)


# --- Runner OHLCV ------------------------------------------------------------
class RiskSimulationRunner:
    """Uruchamia symulacje profili ryzyka dla danych rynkowych (OHLCV)."""

    def __init__(
        self,
        *,
        profiles: Sequence[RiskProfile],
        candles_by_symbol: Mapping[str, Sequence[Candle]],
        settings: SimulationSettings | None = None,
    ) -> None:
        self._profiles = list(profiles)
        self._candles_by_symbol = candles_by_symbol
        self._settings = settings or SimulationSettings()

    def run(self) -> RiskSimulationReport:
        base_equity = self._settings.base_equity
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        synthetic = all(len(candles) == 0 for candles in self._candles_by_symbol.values())
        profile_results: list[ProfileSimulationResult] = []
        for profile in self._profiles:
            profile_results.append(self._run_for_profile(profile=profile, base_equity=base_equity))
        return RiskSimulationReport(
            generated_at=now,
            base_equity=base_equity,
            profiles=profile_results,
            synthetic_data=synthetic,
        )

    def _run_for_profile(self, *, profile: RiskProfile, base_equity: float) -> ProfileSimulationResult:
        total_returns: list[float] = []
        pnl_series: list[float] = []
        sample_size = 0
        leverage_limit = max(_max_leverage(profile), 1.0)
        position_pct = max(_max_position_exposure(profile), 0.0)
        for candles in self._candles_by_symbol.values():
            if not candles:
                continue
            max_bars = self._settings.max_bars
            subset = candles[-max_bars:] if max_bars else candles
            returns = _compute_returns(subset)
            if not returns:
                continue
            notional = base_equity * position_pct
            notional = min(notional, base_equity * leverage_limit)
            pnl = [notional * r for r in returns]
            pnl_series.extend(pnl)
            total_returns.extend(returns)
            sample_size += len(returns)
        if not pnl_series:
            stress_tests = (
                StressTestResult(name="flash_crash", status="failed", metrics={"severity": float("inf")}, notes="Brak danych do symulacji"),
                StressTestResult(name="dry_liquidity", status="failed", metrics={"severity": float("inf")}, notes="Brak danych do symulacji"),
                StressTestResult(name="latency_spike", status="failed", metrics={"severity": float("inf")}, notes="Brak danych do symulacji"),
            )
            return ProfileSimulationResult(
                profile=profile.name,
                base_equity=base_equity,
                final_equity=base_equity,
                total_return_pct=0.0,
                max_drawdown_pct=0.0,
                worst_daily_loss_pct=0.0,
                realized_volatility=0.0,
                breaches=("data_unavailable",),
                stress_tests=stress_tests,
                sample_size=0,
            )
        total_pnl = sum(pnl_series)
        final_equity = base_equity + total_pnl
        max_drawdown = _compute_max_drawdown(pnl_series, base_equity)
        worst_daily_loss = _compute_daily_losses(next(iter(self._candles_by_symbol.values())), pnl_series, base_equity)
        realized_vol = _realized_volatility(total_returns)
        breaches = []
        if max_drawdown > _drawdown_limit(profile):
            breaches.append("drawdown_limit")
        if worst_daily_loss > _daily_loss_limit(profile):
            breaches.append("daily_loss_limit")
        if realized_vol > _target_volatility(profile) * 1.6:
            breaches.append("volatility_target")
        stress_results = [
            _run_flash_crash_test(profile, candles, base_equity) for candles in self._candles_by_symbol.values() if candles
        ] or [StressTestResult(name="flash_crash", status="failed", metrics={"severity": float("inf")}, notes="Brak danych do symulacji")]
        liquidity_results = [
            _run_liquidity_dryout_test(profile, candles) for candles in self._candles_by_symbol.values() if candles
        ]
        latency_results = [
            _run_latency_spike_test(profile, candles) for candles in self._candles_by_symbol.values() if candles
        ]
        stress_tests = tuple(stress_results + liquidity_results + latency_results)
        return ProfileSimulationResult(
            profile=profile.name,
            base_equity=base_equity,
            final_equity=final_equity,
            total_return_pct=(final_equity / base_equity) - 1.0,
            max_drawdown_pct=max_drawdown,
            worst_daily_loss_pct=worst_daily_loss,
            realized_volatility=realized_vol,
            breaches=tuple(breaches),
            stress_tests=stress_tests,
            sample_size=sample_size,
        )


# --- Wczytywanie profili z konfiguracji --------------------------------------
def load_profiles_from_config(path: str | Path) -> Sequence[RiskProfile]:
    """Wczytuje i instancjuje profile ryzyka na podstawie konfiguracji."""
    if load_core_config is None or build_risk_profile_from_config is None:
        raise RuntimeError("Config-based profiles are not available in this build.")
    config = load_core_config(path)
    profiles: list[RiskProfile] = []
    for profile_config in config.risk_profiles.values():
        profiles.append(build_risk_profile_from_config(profile_config))
    if not profiles:
        raise RuntimeError("Brak profili ryzyka w konfiguracji")
    return profiles


def run_simulations_from_config(
    *,
    config_path: str | Path,
    dataset_root: Path | None,
    namespace: str,
    symbols: Sequence[str],
    interval: str,
    settings: SimulationSettings | None = None,
    synthetic_fallback: bool = False,
) -> RiskSimulationReport:
    """Uruchamia symulacje OHLCV na podstawie konfiguracji."""
    profiles = load_profiles_from_config(config_path)
    if dataset_root is None:
        dataset_root = Path("data/ohlcv")
    candles: dict[str, Sequence[Candle]] = {}
    loader = MarketDatasetLoader(dataset_root, namespace=namespace)
    for symbol in symbols:
        try:
            candles[symbol] = loader.load(symbol, interval)
        except FileNotFoundError:
            if not synthetic_fallback:
                raise
            _LOGGER.warning("Brak danych Parquet dla %s – generuję syntetyczne świece", symbol)
            candles[symbol] = _generate_synthetic_candles(bars=settings.max_bars if settings else 720)
    if synthetic_fallback and any(not c for c in candles.values()):
        for symbol, existing in list(candles.items()):
            if existing:
                continue
            candles[symbol] = _generate_synthetic_candles(
                bars=settings.max_bars if settings else 720,
                seed=hash(symbol) % 10_000,
            )
    runner = RiskSimulationRunner(profiles=profiles, candles_by_symbol=candles, settings=settings)
    report = runner.run()
    if synthetic_fallback:
        report.synthetic_data = True
    return report


# --- Scenariusze Parquet/engine (gałąź "main") -------------------------------
def write_default_smoke_scenarios(path: str | Path) -> Path:
    """Zapisuje domyślne scenariusze smoke testu do pliku Parquet."""
    if pa is None or pq is None:
        raise RuntimeError("pyarrow nie jest dostępny — nie można zapisać pliku Parquet")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in DEFAULT_SMOKE_SCENARIOS])
    pq.write_table(table, target)
    return target


def load_orders_from_parquet(path: str | Path) -> Sequence[SimulationOrder]:
    if pq is None:
        raise RuntimeError("pyarrow nie jest dostępny — nie można wczytać pliku Parquet")
    table = pq.read_table(path)
    records = table.to_pylist()
    orders = [SimulationOrder.from_mapping(record) for record in records]
    orders.sort(key=lambda order: order.timestamp)
    return tuple(orders)


def build_profile(profile_name: str, *, manual_overrides: Mapping[str, object] | None = None) -> RiskProfile:
    """Buduje profil ryzyka z wbudowanych klas (aggressive/balanced/conservative/manual)."""
    if RiskProfile is None:
        raise RuntimeError("RiskProfile base class is not available in this build.")
    normalized = profile_name.strip().lower()
    if normalized == "manual":
        if ManualProfile is None:
            raise RuntimeError("ManualProfile is not available in this build.")
        if not manual_overrides:
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
            name=str(manual_overrides.get("name", "manual")),
            max_positions=int(manual_overrides["max_positions"]),
            max_leverage=float(manual_overrides["max_leverage"]),
            drawdown_limit=float(manual_overrides["drawdown_limit"]),
            daily_loss_limit=float(manual_overrides["daily_loss_limit"]),
            max_position_pct=float(manual_overrides["max_position_pct"]),
            target_volatility=float(manual_overrides["target_volatility"]),
            stop_loss_atr_multiple=float(manual_overrides["stop_loss_atr_multiple"]),
        )
    mapping = {
        "conservative": ConservativeProfile,
        "balanced": BalancedProfile,
        "aggressive": AggressiveProfile,
    }
    try:
        factory = mapping[normalized]
    except KeyError as exc:  # pragma: no cover
        raise KeyError(f"Unsupported risk profile: {profile_name}") from exc
    if factory is None:  # pragma: no cover
        raise RuntimeError(f"Profile '{profile_name}' is not available in this build.")
    return factory()  # type: ignore[call-arg]


def run_profile_scenario(
    engine: RiskEngine,
    profile: RiskProfile,
    orders: Iterable[SimulationOrder],
) -> RiskScenarioResult:
    if RiskEngine is None:
        raise RuntimeError("RiskEngine is not available in this build.")
    engine.register_profile(profile)
    accepted = 0
    rejected = 0
    rejection_reasons: list[str] = []
    decisions: list[Mapping[str, object]] = []
    for order in orders:
        if order.profile != profile.name:
            continue
        snapshot = order.to_account_snapshot()
        request = order.to_order_request()
        result = engine.apply_pre_trade_checks(request, account=snapshot, profile_name=profile.name)
        decision_payload: MutableMapping[str, object] = {
            "order": order.to_mapping(),
            "allowed": result.allowed,
            "reason": result.reason,
        }
        if getattr(result, "adjustments", None) is not None:
            decision_payload["adjustments"] = dict(result.adjustments)
        decisions.append(decision_payload)
        if result.allowed:
            accepted += 1
            fill_args = order.to_fill_arguments()
            engine.on_fill(
                profile_name=profile.name,
                symbol=str(fill_args["symbol"]),
                side=str(fill_args["side"]),
                position_value=float(fill_args["position_value"]),
                pnl=float(fill_args["pnl"]),
                timestamp=fill_args["timestamp"],
            )
        else:
            rejected += 1
            if result.reason:
                rejection_reasons.append(result.reason)
    state = engine.snapshot_state(profile.name)
    try:
        force_liquidation = engine.should_liquidate(profile_name=profile.name)
    except KeyError:  # pragma: no cover
        force_liquidation = False
    return RiskScenarioResult(
        profile=profile.name,
        total_orders=accepted + rejected,
        accepted_orders=accepted,
        rejected_orders=rejected,
        force_liquidation=force_liquidation,
        rejection_reasons=tuple(rejection_reasons),
        decisions=tuple(decisions),
        final_state=state,
    )


# --- PDF utils ---------------------------------------------------------------
def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_simple_pdf(lines: Sequence[str]) -> bytes:
    content_lines = ["BT", "/F1 12 Tf", f"72 {_PDF_PAGE_HEIGHT - 72} Td"]
    for index, line in enumerate(lines):
        escaped = _escape_pdf_text(line)
        if index == 0:
            content_lines.append(f"({escaped}) Tj")
        else:
            content_lines.append(f"0 -{_PDF_LINE_HEIGHT} Td ({escaped}) Tj")
    content_lines.append("ET")
    content_stream = "\n".join(content_lines).encode("utf-8")
    stream = b"<< /Length %d >>\nstream\n" % len(content_stream) + content_stream + b"\nendstream"
    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    page_dict = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        % (_PDF_PAGE_WIDTH, _PDF_PAGE_HEIGHT)
    )
    objects.append(page_dict)
    objects.append(stream)
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    output = bytearray()
    output.extend(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("utf-8"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("utf-8"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        output.extend(f"{offset:010} 00000 n \n".encode("utf-8"))
    output.extend(b"trailer\n")
    output.extend(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("utf-8"))
    output.extend(b"startxref\n")
    output.extend(str(xref_offset).encode("utf-8"))
    output.extend(b"\n%%EOF\n")
    return bytes(output)


def _write_simple_pdf(path: str | Path, lines: Sequence[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(_build_simple_pdf(lines))


# --- Public API --------------------------------------------------------------
__all__ = [
    # OHLCV branch
    "Candle",
    "RiskSimulationRunner",
    "RiskSimulationReport",
    "ProfileSimulationResult",
    "StressTestResult",
    "SimulationSettings",
    "MarketDatasetLoader",
    "load_profiles_from_config",
    "run_simulations_from_config",
    # Parquet/engine branch
    "DEFAULT_PROFILES",
    "SimulationOrder",
    "RiskScenarioResult",
    "RiskSimulationSuite",
    "DEFAULT_SMOKE_SCENARIOS",
    "write_default_smoke_scenarios",
    "build_profile",
    "load_orders_from_parquet",
    "run_profile_scenario",
]
