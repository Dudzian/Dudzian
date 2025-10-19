#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage6 Stress Lab – merged CLI (HEAD + main)

Subcommands:
  - evaluate : ocenia raport Paper Labs (RiskSimulationReport) i generuje raport/CSV/podpis
  - run      : uruchamia Stress Lab z configu i generuje podpisany raport
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

# --- repo path bootstrap ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.config import load_core_config

# --- HEAD pathway imports (evaluate) ---
from bot_core.risk.simulation import RiskSimulationReport
from bot_core.risk.stress_lab import (
    StressLabEvaluator,
    write_overrides_csv,
    write_report_csv,
    write_report_json,
    write_report_signature,
)

# --- main pathway import (run) ---
from bot_core.risk.stress_lab import StressLab  # type: ignore

_LOGGER = logging.getLogger(__name__)


# =====================================================================
# evaluate (HEAD) helpers
# =====================================================================

def _read_hmac_key(path: Path) -> bytes:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:  # jasny komunikat CLI
        raise SystemExit(f"Nie znaleziono pliku z kluczem HMAC: {path}") from exc
    if not raw:
        raise SystemExit(f"Plik klucza HMAC jest pusty: {path}")
    return raw


def _load_portfolio_assets(config_path: Path, governor_name: str | None):
    if governor_name is None:
        return None
    config = load_core_config(str(config_path))
    governor = config.portfolio_governors.get(governor_name)
    if governor is None:
        raise SystemExit(f"PortfolioGovernor '{governor_name}' nie istnieje w konfiguracji")
    return {asset.symbol: asset for asset in governor.assets}


# =====================================================================
# run (main) helpers
# =====================================================================

def _resolve_signing_key_from_config_or_args(
    args: argparse.Namespace, config
) -> tuple[Optional[bytes], Optional[str]]:
    # prefer CLI overrides, fallback to config.stress_lab.{signing_key_path|signing_key_env|signing_key_id}
    stress_cfg = getattr(config, "stress_lab", None)
    key_path = args.signing_key_path or (stress_cfg.signing_key_path if stress_cfg else None)
    key_env = args.signing_key_env or (stress_cfg.signing_key_env if stress_cfg else None)
    key_id = args.signing_key_id or (stress_cfg.signing_key_id if stress_cfg else None)

    key_bytes: Optional[bytes] = None
    if key_path:
        key_bytes = Path(key_path).read_bytes()
    elif key_env:
        value = os.environ.get(key_env)
        if not value:
            raise ValueError(f"Zmienna środowiskowa {key_env} jest pusta")
        key_bytes = value.encode("utf-8")
    return key_bytes, key_id


# =====================================================================
# parsers
# =====================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage6 Stress Lab – merged CLI")
    sub = parser.add_subparsers(dest="_cmd", metavar="{evaluate|run}", required=True)

    # evaluate (HEAD)
    pe = sub.add_parser(
        "evaluate",
        help="Oceń raport Paper Labs (RiskSimulationReport) i zbuduj raport/CSV/podpis",
        description="Uruchamia Stress Lab Stage6 na podstawie risk_simulation_report.json",
    )
    pe.add_argument("--risk-report", required=True, help="Ścieżka do risk_simulation_report.json")
    pe.add_argument("--config", help="Ścieżka do config/core.yaml dla mapowania tagów portfela")
    pe.add_argument("--governor", help="Nazwa PortfolioGovernora z konfiguracji")
    pe.add_argument("--output-json", required=True, help="Ścieżka do raportu JSON Stress Lab")
    pe.add_argument("--output-csv", help="Ścieżka do raportu CSV z insightami")
    pe.add_argument("--overrides-csv", help="Ścieżka do CSV z rekomendacjami override")
    pe.add_argument("--signing-key", help="Ścieżka do klucza HMAC (podpis raportu)")
    pe.add_argument("--signing-key-id", help="Opcjonalny identyfikator klucza HMAC")
    pe.add_argument(
        "--signature-path",
        help="Ścieżka do pliku podpisu (domyślnie obok raportu JSON z rozszerzeniem .sig)",
    )
    pe.set_defaults(_handler=_handle_evaluate)

    # run (main)
    pr = sub.add_parser(
        "run",
        help="Uruchom Stress Lab z core.yaml i zapisz podpisany raport",
        description="Uruchamia Stage6 Stress Lab wg sekcji 'stress_lab' w core.yaml.",
    )
    pr.add_argument(
        "--config",
        default="config/core.yaml",
        help="Ścieżka do pliku konfiguracji core (domyślnie config/core.yaml)",
    )
    pr.add_argument(
        "--output",
        help="Ścieżka pliku raportu JSON (domyślnie na podstawie konfiguracji)",
    )
    pr.add_argument("--signing-key-path", help="Plik z kluczem HMAC do podpisania raportu")
    pr.add_argument("--signing-key-env", help="Nazwa zmiennej środowiskowej z kluczem HMAC")
    pr.add_argument("--signing-key-id", help="Identyfikator klucza umieszczany w podpisie")
    pr.add_argument(
        "--fail-on-breach",
        action="store_true",
        help="Zakończ z kodem != 0 jeśli scenariusze zakończą się niepowodzeniem",
    )
    pr.add_argument("--log-level", default="INFO", help="Poziom logowania (domyślnie INFO)")
    pr.set_defaults(_handler=_handle_run)

    return parser


# =====================================================================
# handlers
# =====================================================================

def _handle_evaluate(args: argparse.Namespace) -> int:
    risk_report_path = Path(args.risk_report).expanduser().resolve()
    try:
        risk_report = RiskSimulationReport.from_json(risk_report_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Błąd ładowania raportu symulacji: {exc}", file=sys.stderr)
        return 2

    assets = None
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        try:
            assets = _load_portfolio_assets(config_path, args.governor)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"Błąd ładowania konfiguracji portfela: {exc}", file=sys.stderr)
            return 2

    evaluator = StressLabEvaluator()
    report = evaluator.evaluate(risk_report, portfolio=assets)

    output_json = Path(args.output_json).expanduser()
    payload = write_report_json(report, output_json)

    if args.output_csv:
        write_report_csv(report, Path(args.output_csv).expanduser())
    overrides_csv = getattr(args, "overrides_csv", None)
    if overrides_csv:
        write_overrides_csv(report, Path(overrides_csv).expanduser())

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
        f"Raport Stress Lab zapisany do {output_json} "
        f"(insightów: {len(report.insights)}, overridów: {len(report.overrides)})."
    )
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")

    config = load_core_config(args.config)
    if config.stress_lab is None or not config.stress_lab.enabled:
        _LOGGER.warning("Stress Lab jest wyłączony w konfiguracji")
        return 0

    stress_config = config.stress_lab
    output_path = Path(
        args.output if args.output else Path(stress_config.report_directory) / "stress_lab_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lab = StressLab(stress_config)
    report = lab.run()
    report.write_json(output_path)
    _LOGGER.info("Raport Stress Lab zapisany do %s", output_path)

    key_bytes, key_id = _resolve_signing_key_from_config_or_args(args, config)
    if key_bytes:
        signature_path = output_path.with_suffix(output_path.suffix + ".sig")
        report.write_signature(signature_path, key=key_bytes, key_id=key_id)
        _LOGGER.info("Podpis HMAC zapisany do %s", signature_path)

    if (args.fail_on_breach or stress_config.require_success) and report.has_failures():
        _LOGGER.error("Stress Lab wykrył naruszenia progów")
        return 3

    return 0


# =====================================================================

def _inject_default_command(argv: Sequence[str] | None) -> list[str]:
    args = list(argv or ())
    if args and args[0] in {"run", "evaluate"}:
        return args
    if any(item.startswith("--definitions") or item == "--definitions" for item in args):
        return ["evaluate", *args]
    return ["run", *args]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    effective_argv = _inject_default_command(argv)
    args = parser.parse_args(effective_argv)
    return args._handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
