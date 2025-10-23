"""CLI do uruchamiania symulacji Paper Labs dla profili ryzyka."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping, Sequence

from bot_core.config import load_core_config
from bot_core.risk.simulation import (
    ProfileSimulationResult,
    RiskSimulationReport,
    SimulationSettings,
    StressTestResult,
    run_simulations_from_config,
)

try:  # pragma: no cover - PyYAML może nie być zainstalowany
    import yaml
except Exception:  # noqa: BLE001 - opcjonalna zależność
    yaml = None  # type: ignore[assignment]

_LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = REPO_ROOT / "config/core.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/paper_labs"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Ścieżka do pliku config/core.yaml",
    )
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
        default=str(DEFAULT_OUTPUT_DIR),
        help="Katalog, w którym zapisane zostaną raporty",
    )
    parser.add_argument(
        "--synthetic-fallback",
        action="store_true",
        help="Generuj syntetyczne dane jeśli Parquet jest niedostępny",
    )
    parser.add_argument(
        "--profile",
        nargs="+",
        help="Nazwy profili ryzyka do uruchomienia (np. conservative). Użyj 'all', aby przetworzyć wszystkie profili z konfiguracji.",
    )
    parser.add_argument(
        "--scenario",
        nargs="+",
        help="Nazwy scenariuszy stress testów (flash_crash, dry_liquidity, latency_spike). Domyślnie wszystkie.",
    )
    parser.add_argument(
        "--include-tco",
        action="store_true",
        help="Dołącz sekcję TCO na podstawie decision_engine.tco i dostępnych raportów kosztowych.",
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


def _resolve_dataset_root(args: argparse.Namespace, *, config: Mapping[str, object] | object) -> Path | None:
    if args.dataset_root:
        return Path(args.dataset_root)
    if args.environment:
        environments = getattr(config, "environments", None)
        if environments is None or not isinstance(environments, Mapping):
            raise SystemExit("Konfiguracja nie zawiera sekcji environments")
        env = environments.get(args.environment)
        if env is None:
            raise SystemExit(f"Środowisko {args.environment} nie istnieje w konfiguracji")
        data_path = getattr(env, "data_cache_path", None)
        if not data_path:
            raise SystemExit(f"Środowisko {args.environment} nie definiuje data_cache_path")
        return Path(data_path)
    return None


def _collect_tco_summary(
    config: object,
    *,
    config_path: Path,
) -> Mapping[str, object]:
    summary: dict[str, object] = {"reports": []}
    decision_engine = getattr(config, "decision_engine", None)
    tco_config = getattr(decision_engine, "tco", None) if decision_engine is not None else None
    report_paths: Sequence[str] = ()
    if tco_config is not None:
        report_paths = tuple(getattr(tco_config, "report_paths", ())) or tuple(getattr(tco_config, "reports", ()))
    if not report_paths and yaml is not None:
        try:
            raw_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 - fallback jest opcjonalny
            raw_payload = {}
        if isinstance(raw_payload, Mapping):
            decision_raw = raw_payload.get("decision_engine")
            if isinstance(decision_raw, Mapping):
                tco_raw = decision_raw.get("tco")
                if isinstance(tco_raw, Mapping):
                    reports_raw = tco_raw.get("reports")
                    if isinstance(reports_raw, Sequence):
                        report_paths = tuple(str(path) for path in reports_raw if str(path).strip())
    if not report_paths:
        summary["status"] = "missing_reports"
        return summary

    base_dir = config_path.parent
    reports_summary: list[Mapping[str, object]] = []
    overall_status = "ok"
    for raw_path in report_paths:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        entry: dict[str, object] = {"path": str(candidate)}
        if not candidate.exists():
            entry["status"] = "missing"
            overall_status = "partial"
            reports_summary.append(entry)
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - raport ma być informacyjny
            entry["status"] = "error"
            entry["summary"] = {"error": str(exc)}
            overall_status = "partial"
            reports_summary.append(entry)
            continue
        entry["status"] = "ok"
        details: dict[str, object] = {}
        if isinstance(payload, Mapping):
            generated_at = payload.get("generated_at")
            if generated_at:
                details["generated_at"] = str(generated_at)
            total = payload.get("total")
            if isinstance(total, Mapping):
                for key in ("cost_bps", "monthly_total", "annual_total"):
                    value = total.get(key)
                    if value is not None:
                        details[key] = value
            metadata = payload.get("metadata")
            if isinstance(metadata, Mapping):
                details["metadata"] = dict(metadata)
        entry["summary"] = details
        reports_summary.append(entry)
    summary["reports"] = reports_summary
    summary["status"] = overall_status
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config).expanduser().resolve()
    core_config = load_core_config(str(config_path))

    settings = SimulationSettings(base_equity=args.base_equity, max_bars=args.max_bars)
    dataset_root = _resolve_dataset_root(args, config=core_config)

    _LOGGER.info(
        "Uruchamiam symulacje Paper Labs: namespace=%s, symbols=%s, interval=%s",
        args.namespace,
        ",".join(args.symbols),
        args.interval,
    )

    report: RiskSimulationReport | None = None
    try:
        report = run_simulations_from_config(
            config_path=str(config_path),
            dataset_root=dataset_root,
            namespace=args.namespace,
            symbols=args.symbols,
            interval=args.interval,
            settings=settings,
            synthetic_fallback=args.synthetic_fallback,
            profile_names=args.profile,
            stress_scenarios=args.scenario,
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

    if args.include_tco:
        tco_summary = _collect_tco_summary(core_config, config_path=config_path)
        report.tco_summary = tco_summary

    output_dir = Path(args.output_dir).expanduser()
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
