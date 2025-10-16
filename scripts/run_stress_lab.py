from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bot_core.config import load_core_config
from bot_core.risk.simulation import RiskSimulationReport
from bot_core.risk.stress_lab import (
    StressLabEvaluator,
    write_overrides_csv,
    write_report_csv,
    write_report_json,
    write_report_signature,
)


def _read_hmac_key(path: Path) -> bytes:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:  # noqa: BLE001 - komunikat użytkownika
        raise SystemExit(f"Nie znaleziono pliku z kluczem HMAC: {path}") from exc
    if not raw:
        raise SystemExit(f"Plik klucza HMAC jest pusty: {path}")
    return raw


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Uruchamia Stress Lab Stage6 na podstawie raportu Paper Labs")
    parser.add_argument("--risk-report", required=True, help="Ścieżka do risk_simulation_report.json")
    parser.add_argument("--config", help="Ścieżka do config/core.yaml dla mapowania tagów portfela")
    parser.add_argument("--governor", help="Nazwa PortfolioGovernora z konfiguracji")
    parser.add_argument("--output-json", required=True, help="Ścieżka do raportu JSON Stress Lab")
    parser.add_argument("--output-csv", help="Ścieżka do raportu CSV z insightami")
    parser.add_argument("--overrides-csv", help="Ścieżka do CSV z rekomendacjami override")
    parser.add_argument("--signing-key", help="Ścieżka do klucza HMAC dla podpisu raportu")
    parser.add_argument("--signing-key-id", help="Opcjonalny identyfikator klucza HMAC")
    parser.add_argument(
        "--signature-path",
        help="Ścieżka do pliku podpisu (domyślnie obok raportu JSON z rozszerzeniem .sig)",
    )
    return parser


def _load_portfolio_assets(config_path: Path, governor_name: str | None):
    if governor_name is None:
        return None

    config = load_core_config(str(config_path))
    governor = config.portfolio_governors.get(governor_name)
    if governor is None:
        raise SystemExit(f"PortfolioGovernor '{governor_name}' nie istnieje w konfiguracji")
    return {asset.symbol: asset for asset in governor.assets}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    risk_report_path = Path(args.risk_report).expanduser().resolve()
    try:
        risk_report = RiskSimulationReport.from_json(risk_report_path)
    except (FileNotFoundError, ValueError) as exc:  # noqa: PERF203 - komunikat CLI
        print(f"Błąd ładowania raportu symulacji: {exc}", file=sys.stderr)
        return 2

    assets = None
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        try:
            assets = _load_portfolio_assets(config_path, args.governor)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - jasna informacja dla operatora
            print(f"Błąd ładowania konfiguracji portfela: {exc}", file=sys.stderr)
            return 2

    evaluator = StressLabEvaluator()
    report = evaluator.evaluate(risk_report, portfolio=assets)

    output_json = Path(args.output_json).expanduser()
    payload = write_report_json(report, output_json)

    if args.output_csv:
        write_report_csv(report, Path(args.output_csv).expanduser())
    if args.overrides_csv:
        write_overrides_csv(report, Path(args.overrides_csv).expanduser())

    if args.signing_key:
        key_path = Path(args.signing_key).expanduser().resolve()
        key = _read_hmac_key(key_path)
        signature_path = (
            Path(args.signature_path).expanduser()
            if args.signature_path
            else output_json.with_suffix(output_json.suffix + ".sig")
        )
        write_report_signature(
            payload,
            signature_path,
            key=key,
            key_id=args.signing_key_id,
            target=output_json.name,
        )

    print(
        f"Raport Stress Lab zapisany do {output_json} (insightów: {len(report.insights)}, overridów: {len(report.overrides)})."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
