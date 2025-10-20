from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.config import load_core_config
from bot_core.market_intel import MarketIntelSnapshot
from bot_core.risk.stress_lab_calibration import (
    StressLabCalibrator,
    StressLabCalibrationReport,
    StressLabCalibrationSegment,
    write_calibration_csv,
    write_calibration_json,
    write_calibration_signature,
    build_volume_segments,
)
from bot_core.risk.simulation import RiskSimulationReport


def _load_market_intel(path: Path) -> dict[str, MarketIntelSnapshot]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # noqa: BLE001 - komunikat CLI
        raise SystemExit(f"Nie znaleziono pliku Market Intel: {path}") from exc
    except json.JSONDecodeError as exc:  # noqa: BLE001 - komunikat CLI
        raise SystemExit(f"Plik Market Intel ma niepoprawny format JSON: {exc}") from exc

    snapshots_payload = payload.get("snapshots")
    if not isinstance(snapshots_payload, dict):
        raise SystemExit("Plik Market Intel nie zawiera sekcji 'snapshots'")

    snapshots: dict[str, MarketIntelSnapshot] = {}
    for symbol, data in snapshots_payload.items():
        if not isinstance(data, dict):
            print(f"Pomijam {symbol}: brak poprawnej struktury snapshotu", file=sys.stderr)
            continue
        snapshots[symbol] = MarketIntelSnapshot(
            symbol=symbol,
            interval=str(data.get("interval", "unknown")),
            start=None,
            end=None,
            bar_count=int(data.get("bar_count", 0)),
            price_change_pct=data.get("price_change_pct"),
            volatility_pct=data.get("volatility_pct"),
            max_drawdown_pct=data.get("max_drawdown_pct"),
            average_volume=data.get("average_volume"),
            liquidity_usd=data.get("liquidity_usd"),
            momentum_score=data.get("momentum_score"),
            metadata=data.get("metadata", {}),
        )
    if not snapshots:
        raise SystemExit("Brak snapshotów w pliku Market Intel")
    return snapshots


def _load_segments(
    *,
    market_snapshots: Mapping[str, MarketIntelSnapshot],
    segments_path: Path | None,
    config_path: Path | None,
    governor_name: str | None,
    volume_buckets: int | None,
    volume_min_symbols: int,
    volume_name_prefix: str,
    volume_risk_budget_prefix: str | None,
) -> list[StressLabCalibrationSegment]:
    segments: list[StressLabCalibrationSegment] = []
    if segments_path is not None:
        try:
            raw = json.loads(segments_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:  # noqa: BLE001 - jasny komunikat
            raise SystemExit(f"Nie znaleziono pliku segmentów: {segments_path}") from exc
        except json.JSONDecodeError as exc:  # noqa: BLE001
            raise SystemExit(f"Plik segmentów ma niepoprawny JSON: {exc}") from exc
        if not isinstance(raw, list):
            raise SystemExit("Plik segmentów powinien zawierać listę definicji")
        for entry in raw:
            if not isinstance(entry, dict):
                print("Pomijam segment bez struktury słownikowej", file=sys.stderr)
                continue
            name = entry.get("name")
            if not isinstance(name, str):
                print("Pomijam segment bez nazwy", file=sys.stderr)
                continue
            segments.append(
                StressLabCalibrationSegment(
                    name=name,
                    symbols=tuple(entry.get("symbols", ())),
                    tags=tuple(entry.get("tags", ())),
                    risk_budgets=tuple(entry.get("risk_budgets", ())),
                )
            )
    elif volume_buckets:
        try:
            segments = list(
                build_volume_segments(
                    market_snapshots,
                    buckets=volume_buckets,
                    min_symbols_per_bucket=volume_min_symbols,
                    name_prefix=volume_name_prefix,
                    risk_budget_prefix=volume_risk_budget_prefix,
                )
            )
        except ValueError as exc:  # noqa: BLE001 - komunikat CLI
            raise SystemExit(f"Nie udało się zbudować segmentów wolumenowych: {exc}") from exc
    elif config_path is not None and governor_name:
        config = load_core_config(str(config_path))
        governor = config.portfolio_governors.get(governor_name)
        if governor is None:
            raise SystemExit(f"PortfolioGovernor '{governor_name}' nie istnieje w konfiguracji")

        segments_by_tag: dict[str, set[str]] = {}
        for asset in governor.assets:
            segment_tags = [tag.split(":", 1)[1] for tag in asset.tags if tag.startswith("segment:")]
            if not segment_tags:
                continue
            for tag in segment_tags:
                segments_by_tag.setdefault(tag, set()).add(asset.symbol)

        if segments_by_tag:
            for name, symbols in sorted(segments_by_tag.items()):
                segments.append(
                    StressLabCalibrationSegment(name=name, symbols=tuple(sorted(symbols)))
                )

    if not segments:
        segments.append(
            StressLabCalibrationSegment(
                name="default", symbols=tuple(sorted(market_snapshots.keys()))
            )
        )
    return segments


def _read_hmac_key(path: Path) -> bytes:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:  # noqa: BLE001
        raise SystemExit(f"Nie znaleziono pliku klucza HMAC: {path}") from exc
    if not raw:
        raise SystemExit(f"Plik klucza HMAC jest pusty: {path}")
    return raw


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kalibruje progi Stress Lab Stage6 na podstawie Market Intel i raportów ryzyka"
    )
    parser.add_argument("--market-intel", required=True, help="Ścieżka do pliku JSON Market Intel")
    parser.add_argument("--risk-report", help="Opcjonalny raport risk_simulation_report.json dla kalibracji latencji")
    parser.add_argument("--segments", help="Plik JSON z definicjami segmentów kalibracyjnych")
    parser.add_argument("--config", help="Ścieżka do config/core.yaml w celu automatycznych segmentów")
    parser.add_argument("--governor", help="Nazwa PortfolioGovernora z configu dla segmentów")
    parser.add_argument(
        "--volume-buckets",
        type=int,
        help="Liczba segmentów budowanych automatycznie na podstawie płynności",
    )
    parser.add_argument(
        "--volume-min-symbols",
        type=int,
        default=3,
        help="Minimalna liczba symboli w automatycznym segmencie (domyślnie: 3)",
    )
    parser.add_argument(
        "--volume-name-prefix",
        default="volume",
        help="Prefiks nazw segmentów wolumenowych (domyślnie: volume)",
    )
    parser.add_argument(
        "--volume-risk-budget-prefix",
        help="Opcjonalny prefiks budżetu ryzyka przypisywanego segmentom wolumenowym",
    )
    parser.add_argument("--output-json", required=True, help="Ścieżka raportu JSON z progami")
    parser.add_argument("--output-csv", help="Ścieżka raportu CSV z segmentami")
    parser.add_argument("--signing-key", help="Ścieżka do klucza HMAC dla podpisu raportu")
    parser.add_argument("--signing-key-id", help="Opcjonalny identyfikator klucza HMAC")
    parser.add_argument("--signature-path", help="Ścieżka pliku podpisu (domyślnie raport.json.sig)")
    return parser


def _load_risk_report(path: Path | None) -> RiskSimulationReport | None:
    if path is None:
        return None
    try:
        return RiskSimulationReport.from_json(path)
    except (FileNotFoundError, ValueError) as exc:  # noqa: PERF203 - komunikat CLI
        print(f"Błąd ładowania raportu ryzyka: {exc}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    market_path = Path(args.market_intel).expanduser().resolve()
    snapshots = _load_market_intel(market_path)

    segments_path = Path(args.segments).expanduser().resolve() if args.segments else None
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    segments = _load_segments(
        market_snapshots=snapshots,
        segments_path=segments_path,
        config_path=config_path,
        governor_name=args.governor,
        volume_buckets=args.volume_buckets,
        volume_min_symbols=args.volume_min_symbols,
        volume_name_prefix=args.volume_name_prefix,
        volume_risk_budget_prefix=args.volume_risk_budget_prefix,
    )

    risk_report = _load_risk_report(Path(args.risk_report).expanduser().resolve() if args.risk_report else None)

    calibrator = StressLabCalibrator()
    report: StressLabCalibrationReport = calibrator.calibrate(
        market_snapshots=snapshots,
        segments=segments,
        risk_report=risk_report,
    )

    output_json = Path(args.output_json).expanduser()
    payload = write_calibration_json(report, output_json)

    if args.output_csv:
        write_calibration_csv(report, Path(args.output_csv).expanduser())

    if args.signing_key:
        key_path = Path(args.signing_key).expanduser().resolve()
        key = _read_hmac_key(key_path)
        signature_path = (
            Path(args.signature_path).expanduser()
            if args.signature_path
            else output_json.with_suffix(output_json.suffix + ".sig")
        )
        write_calibration_signature(
            payload,
            signature_path,
            key=key,
            key_id=args.signing_key_id,
            target=output_json.name,
        )

    print(
        "Zapisano kalibrację Stress Lab do "
        f"{output_json} (segmenty: {len(report.liquidity_segments)},"
        f" latencja: {'tak' if report.latency_warning_threshold_ms else 'nie'})."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
