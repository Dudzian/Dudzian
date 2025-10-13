"""Symulacje profili ryzyka oraz stres testy paper→live."""
from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, pstdev
from typing import Iterable, Mapping, MutableMapping, Sequence

from bot_core.config import load_core_config
from bot_core.risk.base import RiskProfile
from bot_core.risk.factory import build_risk_profile_from_config

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - środowiska bez pyarrow zgłoszą błąd dopiero przy próbie odczytu
    pq = None  # type: ignore


_LOGGER = logging.getLogger(__name__)

_BASE_EQUITY_DEFAULT = 100_000.0
_PDF_LINE_HEIGHT = 14
_PDF_PAGE_WIDTH = 612
_PDF_PAGE_HEIGHT = 792


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
    """Metryki wynikowe dla profilu ryzyka."""

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
        if self.breaches:
            return True
        return any(result.is_failure() for result in self.stress_tests)


@dataclass(slots=True)
class RiskSimulationReport:
    """Zbiorczy raport z symulacji profili ryzyka."""

    generated_at: str
    base_equity: float
    profiles: Sequence[ProfileSimulationResult]
    synthetic_data: bool = False

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "generated_at": self.generated_at,
            "base_equity": self.base_equity,
            "synthetic_data": self.synthetic_data,
            "profiles": [profile.to_mapping() for profile in self.profiles],
            "breach_count": sum(len(profile.breaches) for profile in self.profiles),
            "stress_failures": sum(
                sum(1 for result in profile.stress_tests if result.is_failure())
                for profile in self.profiles
            ),
        }

    def has_failures(self) -> bool:
        return any(profile.has_failures() for profile in self.profiles)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_mapping(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

    def write_pdf(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
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
            if profile.breaches:
                lines.append(f"  Breaches: {', '.join(profile.breaches)}")
            else:
                lines.append("  Breaches: none")
            for stress in profile.stress_tests:
                status = stress.status.upper()
                severity = stress.metrics.get("severity")
                if severity:
                    lines.append(f"  Stress {stress.name}: {status} (severity {severity})")
                else:
                    lines.append(f"  Stress {stress.name}: {status}")
                if stress.notes:
                    lines.append(f"    Notes: {stress.notes}")
        pdf_bytes = _build_simple_pdf(lines)
        path.write_bytes(pdf_bytes)


@dataclass(slots=True)
class SimulationSettings:
    """Parametry wejściowe symulacji."""

    base_equity: float = _BASE_EQUITY_DEFAULT
    max_bars: int | None = 720


class MarketDatasetLoader:
    """Loader świec OHLCV z plików Parquet."""

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
        candles.sort(key=lambda candle: candle.timestamp_ms)
        if not candles:
            raise FileNotFoundError(partition)
        return candles


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
    metrics: dict[str, float] = {
        "max_low_gap_pct": severity,
        "severity": severity / max(profile.drawdown_limit(), 1e-6),
    }
    allowed = severity <= profile.drawdown_limit()
    status = "passed" if allowed else "failed"
    notes = None
    if not allowed:
        notes = (
            f"Observed crash {severity * 100:.2f}% breaches drawdown limit {profile.drawdown_limit() * 100:.2f}%"
        )
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
    low_volume_ratio = sum(1 for volume in volumes if volume <= threshold) / len(volumes)
    allowed_ratio = max(0.05, 0.35 - profile.target_volatility())
    metrics: dict[str, float] = {
        "low_volume_ratio": low_volume_ratio,
        "allowed_ratio": allowed_ratio,
        "severity": low_volume_ratio / max(allowed_ratio, 1e-6),
    }
    status = "passed" if low_volume_ratio <= allowed_ratio else "failed"
    notes = None
    if status == "failed":
        notes = (
            f"{low_volume_ratio * 100:.2f}% słabych wolumenów przekracza próg {allowed_ratio * 100:.2f}%"
        )
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
    threshold = 0.01 + profile.stop_loss_atr_multiple() * 0.015
    metrics: dict[str, float] = {
        "max_spread_pct": max_spread,
        "threshold": threshold,
        "severity": max_spread / max(threshold, 1e-6),
    }
    status = "passed" if max_spread <= threshold else "failed"
    notes = None
    if status == "failed":
        notes = (
            f"Spread {max_spread * 100:.2f}% przekracza próg {threshold * 100:.2f}% – ryzyko poślizgu"
        )
    return StressTestResult(name="latency_spike", status=status, metrics=metrics, notes=notes)


class RiskSimulationRunner:
    """Uruchamia symulacje profili ryzyka dla danych rynkowych."""

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
            profile_results.append(
                self._run_for_profile(profile=profile, base_equity=base_equity)
            )
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
        leverage_limit = max(profile.max_leverage(), 1.0)
        position_pct = max(profile.max_position_exposure(), 0.0)
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
                StressTestResult(
                    name="flash_crash",
                    status="failed",
                    metrics={"severity": float("inf")},
                    notes="Brak danych do symulacji",
                ),
                StressTestResult(
                    name="dry_liquidity",
                    status="failed",
                    metrics={"severity": float("inf")},
                    notes="Brak danych do symulacji",
                ),
                StressTestResult(
                    name="latency_spike",
                    status="failed",
                    metrics={"severity": float("inf")},
                    notes="Brak danych do symulacji",
                ),
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
        worst_daily_loss = _compute_daily_losses(
            next(iter(self._candles_by_symbol.values())), pnl_series, base_equity
        )
        realized_vol = _realized_volatility(total_returns)
        breaches = []
        if max_drawdown > profile.drawdown_limit():
            breaches.append("drawdown_limit")
        if worst_daily_loss > profile.daily_loss_limit():
            breaches.append("daily_loss_limit")
        if realized_vol > profile.target_volatility() * 1.6:
            breaches.append("volatility_target")
        stress_results = [
            _run_flash_crash_test(profile, candles, base_equity)
            for candles in self._candles_by_symbol.values()
            if candles
        ]
        if not stress_results:
            stress_results = [
                StressTestResult(
                    name="flash_crash",
                    status="failed",
                    metrics={"severity": float("inf")},
                    notes="Brak danych do symulacji",
                )
            ]
        liquidity_results = [
            _run_liquidity_dryout_test(profile, candles)
            for candles in self._candles_by_symbol.values()
            if candles
        ]
        latency_results = [
            _run_latency_spike_test(profile, candles)
            for candles in self._candles_by_symbol.values()
            if candles
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


def load_profiles_from_config(path: str | Path) -> Sequence[RiskProfile]:
    """Wczytuje i instancjuje profile ryzyka na podstawie konfiguracji."""

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
    """Pomocnicza funkcja uruchamiająca symulacje na podstawie konfiguracji."""

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
            _LOGGER.warning(
                "Brak danych Parquet dla %s – generuję syntetyczne świece", symbol
            )
            candles[symbol] = _generate_synthetic_candles(bars=settings.max_bars if settings else 720)
    if synthetic_fallback and any(not c for c in candles.values()):
        for symbol, existing in list(candles.items()):
            if existing:
                continue
            candles[symbol] = _generate_synthetic_candles(
                bars=settings.max_bars if settings else 720,
                seed=hash(symbol) % 10_000,
            )
    runner = RiskSimulationRunner(
        profiles=profiles,
        candles_by_symbol=candles,
        settings=settings,
    )
    report = runner.run()
    if synthetic_fallback:
        report.synthetic_data = True
    return report


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


__all__ = [
    "Candle",
    "RiskSimulationRunner",
    "RiskSimulationReport",
    "ProfileSimulationResult",
    "StressTestResult",
    "SimulationSettings",
    "MarketDatasetLoader",
    "load_profiles_from_config",
    "run_simulations_from_config",
]
