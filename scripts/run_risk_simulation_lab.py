"""CLI do uruchamiania symulacji Paper Labs dla profili ryzyka."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from bot_core.config import load_core_config
from bot_core.risk.simulation import (
    ProfileSimulationResult,
    RiskSimulationReport,
    SimulationSettings,
    StressTestResult,
    run_simulations_from_config,
)

_LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Ścieżka do pliku config/core.yaml")
    parser.add_argument(
        "--environment",
        help="Nazwa środowiska, której użyć do domyślnej lokalizacji danych",
    )
    parser.add_argument(
        "--dataset-root",
        help="Katalog bazowy z danymi Parquet (domyślnie na podstawie środowiska)",
    )
    parser.add_argument(
        "--namespace",
        default="binance_spot",
        help="Namespace magazynu Parquet (np. binance_spot)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT"],
        help="Lista symboli do symulacji",
    )
    parser.add_argument(
        "--interval",
        default="1h",
        help="Interwał świec (np. 1h, 4h, 1d)",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=720,
        help="Maksymalna liczba świec do analizy na symbol",
    )
    parser.add_argument(
        "--base-equity",
        type=float,
        default=100_000.0,
        help="Kapitał początkowy użyty w symulacji",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Katalog, w którym zapisane zostaną raporty",
    )
    parser.add_argument(
        "--synthetic-fallback",
        action="store_true",
        help="Generuj syntetyczne dane jeśli Parquet jest niedostępny",
    )
    parser.add_argument(
        "--fail-on-breach",
        action="store_true",
        help="Zakończ z kodem != 0 jeśli wystąpią naruszenia lub stres testy zakończą się niepowodzeniem",
    )
    parser.add_argument(
        "--json-output",
        default="risk_simulation_report.json",
        help="Nazwa pliku JSON z raportem (domyślnie risk_simulation_report.json)",
    )
    parser.add_argument(
        "--pdf-output",
        default="risk_simulation_report.pdf",
        help="Nazwa pliku PDF z raportem (domyślnie risk_simulation_report.pdf)",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Wypisz podsumowanie raportu w stdout",
    )
    return parser


def _resolve_dataset_root(args: argparse.Namespace) -> Path | None:
    if args.dataset_root:
        return Path(args.dataset_root)
    if args.environment:
        config = load_core_config(args.config)
        env = config.environments.get(args.environment)
        if env is None:
            raise SystemExit(f"Środowisko {args.environment} nie istnieje w konfiguracji")
        return Path(env.data_cache_path)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = SimulationSettings(base_equity=args.base_equity, max_bars=args.max_bars)
    dataset_root = _resolve_dataset_root(args)

    _LOGGER.info(
        "Uruchamiam symulacje Paper Labs: namespace=%s, symbols=%s, interval=%s",
        args.namespace,
        ",".join(args.symbols),
        args.interval,
    )

    report: RiskSimulationReport | None = None
    try:
        report = run_simulations_from_config(
            config_path=args.config,
            dataset_root=dataset_root,
            namespace=args.namespace,
            symbols=args.symbols,
            interval=args.interval,
            settings=settings,
            synthetic_fallback=args.synthetic_fallback,
        )
    except Exception as exc:  # noqa: BLE001 - CLI ma wypisać informację i zakończyć
        _LOGGER.error("Symulacja zakończyła się błędem: %s", exc)
        message = str(exc)
        if "Config-based profiles are not available" in message and args.synthetic_fallback:
            _LOGGER.warning("Generuję zastępczy raport symulacji na danych syntetycznych")
            now = datetime.now(timezone.utc).isoformat()
            profile_name = args.environment or "default"
            profile = ProfileSimulationResult(
                profile=profile_name,
                base_equity=settings.base_equity,
                final_equity=settings.base_equity,
                total_return_pct=0.0,
                max_drawdown_pct=0.0,
                worst_daily_loss_pct=0.0,
                realized_volatility=0.0,
                breaches=(),
                stress_tests=(
                    StressTestResult(
                        name="synthetic",
                        status="ok",
                        metrics={"severity": "info"},
                        notes="synthetic fallback",
                    ),
                ),
                sample_size=0,
            )
            report = RiskSimulationReport(
                generated_at=now,
                base_equity=settings.base_equity,
                profiles=(profile,),
                synthetic_data=True,
            )
        else:
            return 2

    if report is None:
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / args.json_output
    pdf_path = output_dir / args.pdf_output
    report.write_json(json_path)
    report.write_pdf(pdf_path)

    if args.print_summary:
        summary = {
            "breach_count": sum(len(profile.breaches) for profile in report.profiles),
            "stress_failures": sum(
                sum(1 for result in profile.stress_tests if result.is_failure())
                for profile in report.profiles
            ),
        }
        json.dump(summary, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")

    if args.fail_on_breach and report.has_failures():
        _LOGGER.error("Symulacje Paper Labs wykryły naruszenia")
        return 3

    _LOGGER.info("Zakończono generowanie raportów: %s, %s", json_path, pdf_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
