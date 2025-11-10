"""CLI uruchamiające symulator portfolio_stress Stage6."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import yaml

# --- repo path bootstrap -----------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.config import load_core_config
from bot_core.config import loader as _config_loader
from bot_core.config.models import PortfolioStressConfig, PortfolioStressScenarioConfig
from bot_core.risk.portfolio_stress import (
    PortfolioStressBaseline,
    PortfolioStressReport,
    load_portfolio_stress_baseline,
    run_portfolio_stress,
)

_LOGGER = logging.getLogger("portfolio_stress.cli")




def _resolve_signing_key(
    args: argparse.Namespace, config: PortfolioStressConfig
) -> tuple[bytes | None, str | None]:
    key_path = args.signing_key_path or getattr(config, "signing_key_path", None)
    key_env = args.signing_key_env or getattr(config, "signing_key_env", None)
    key_id = args.signing_key_id or getattr(config, "signing_key_id", None)

    key_bytes: bytes | None = None
    if key_path:
        key_bytes = Path(key_path).expanduser().read_bytes()
    elif key_env:
        value = os.environ.get(key_env)
        if not value:
            raise SystemExit(f"Zmienna środowiskowa {key_env} jest pusta")
        key_bytes = value.encode("utf-8")
    return key_bytes, key_id


def _select_baseline_path(
    args_baseline: str | None, config: PortfolioStressConfig
) -> Path:
    if args_baseline:
        candidate = Path(args_baseline).expanduser()
        if not candidate.exists():
            raise SystemExit(f"Plik baseline {candidate} nie istnieje")
        return candidate

    for source in config.baseline_sources:
        candidate = Path(source.path).expanduser()
        if candidate.exists():
            return candidate
        if source.required:
            raise SystemExit(
                f"Wymagane źródło baseline '{source.name}' ({candidate}) jest niedostępne"
            )
    raise SystemExit("Brak skonfigurowanego baseline dla portfolio_stress")


def _filter_scenarios(
    scenarios: Sequence[PortfolioStressScenarioConfig],
    requested: Sequence[str] | None,
) -> tuple[PortfolioStressScenarioConfig, ...]:
    if not requested:
        return tuple(scenarios)
    wanted = {name.strip().lower() for name in requested if name.strip()}
    filtered = tuple(s for s in scenarios if s.name.lower() in wanted)
    if not filtered:
        raise SystemExit(
            "Żaden z żądanych scenariuszy nie istnieje: " + ", ".join(sorted(wanted))
        )
    return filtered


def _apply_config_portfolio_id(
    baseline: PortfolioStressBaseline, config: PortfolioStressConfig
) -> PortfolioStressBaseline:
    portfolio_id = getattr(config, "portfolio_id", None)
    if portfolio_id and portfolio_id != baseline.portfolio_id:
        return replace(baseline, portfolio_id=portfolio_id)
    return baseline


def _write_outputs(
    report: PortfolioStressReport,
    *,
    output_json: Path,
    output_csv: Path | None,
    pretty: bool,
    signature_path: Path | None,
    signing_key: bytes | None,
    signing_key_id: str | None,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(output_json, pretty=pretty)
    if output_csv is not None:
        report.write_csv(output_csv)
    if signing_key:
        sig_path = signature_path or output_json.with_suffix(output_json.suffix + ".sig")
        report.write_signature(sig_path, key=signing_key, key_id=signing_key_id)
        _LOGGER.info("Podpis HMAC zapisany do %s", sig_path)


def _load_portfolio_stress_config(config_path: Path) -> PortfolioStressConfig | None:
    config_path = config_path.expanduser()
    try:
        config = load_core_config(config_path)
    except Exception as exc:  # pragma: no cover - fallback dla brakujących modułów
        _LOGGER.warning(
            "Nie udało się załadować pełnej konfiguracji (%s). Używam fallbacku portfolio_stress.",
            exc,
        )
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        try:
            base_dir = config_path.resolve(strict=False).parent
        except Exception:  # pragma: no cover
            base_dir = config_path.parent
        return _config_loader._load_portfolio_stress_config(raw, base_dir=base_dir)
    return getattr(config, "portfolio_stress", None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Uruchamia portfolio_stress Stage6 z config/core.yaml.",
    )
    parser.add_argument(
        "--config",
        default="config/core.yaml",
        help="Ścieżka do config/core.yaml (domyślnie config/core.yaml)",
    )
    parser.add_argument("--baseline", help="Opcjonalna ścieżka do pliku baseline (JSON)")
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        help="Uruchom tylko wybrane scenariusze (można podać wielokrotnie)",
    )
    parser.add_argument("--output-json", required=True, help="Ścieżka do raportu JSON")
    parser.add_argument("--output-csv", help="Opcjonalna ścieżka do CSV z wynikami")
    parser.add_argument(
        "--signature-path",
        help="Ścieżka do pliku podpisu HMAC (domyślnie <output-json>.sig)",
    )
    parser.add_argument(
        "--signing-key-path",
        help="Ścieżka do klucza HMAC (domyślnie z configu portfolio_stress)",
    )
    parser.add_argument(
        "--signing-key-env",
        help="Zmienna środowiskowa z kluczem HMAC (domyślnie z configu portfolio_stress)",
    )
    parser.add_argument("--signing-key-id", help="Identyfikator klucza HMAC")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Poziom logowania (domyślnie INFO)",
    )
    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="Zapisz JSON bez wcięć (kompaktowy)",
    )
    return parser


def _summarize(report: PortfolioStressReport) -> str:
    scenario_metrics = ", ".join(
        f"{scenario.scenario.name}: {scenario.drawdown_pct:.2%} dd"
        for scenario in report.scenarios
    )
    summary = report.summary
    tag_fragment = ""
    tag_aggregates = summary.get("tag_aggregates")
    if isinstance(tag_aggregates, Sequence) and tag_aggregates:
        top_tag = max(
            (
                item
                for item in tag_aggregates
                if isinstance(item, Mapping)
                and "tag" in item
                and isinstance(item.get("max_drawdown_pct"), (int, float))
            ),
            key=lambda item: item["max_drawdown_pct"],
            default=None,
        )
        if top_tag:
            tag_fragment = (
                f", najcięższa kategoria: {top_tag['tag']} "
                f"({top_tag['max_drawdown_pct']:.2%} dd)"
            )
    var_text = summary.get("var_95_return_pct")
    cvar_text = summary.get("cvar_95_return_pct")
    var_fragment = (
        f", VaR95: {var_text:.2%}"
        if isinstance(var_text, (int, float))
        else ""
    )
    cvar_fragment = (
        f", CVaR95: {cvar_text:.2%}"
        if isinstance(cvar_text, (int, float))
        else ""
    )
    max_drawdown = max(
        (scenario.drawdown_pct for scenario in report.scenarios),
        default=0.0,
    )
    return (
        f"Portfolio stress report for {report.baseline.portfolio_id} "
        f"({len(report.scenarios)} scenariuszy, max drawdown: "
        f"{max_drawdown:.2%}{var_fragment}{cvar_fragment}{tag_fragment}). "
        f"Scenariusze: {scenario_metrics}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s"
    )

    stress_cfg = _load_portfolio_stress_config(Path(args.config))
    if stress_cfg is None or not stress_cfg.enabled:
        _LOGGER.warning("portfolio_stress jest wyłączony w konfiguracji")
        return 0

    baseline_path = _select_baseline_path(args.baseline, stress_cfg)
    baseline = load_portfolio_stress_baseline(baseline_path)
    baseline = _apply_config_portfolio_id(baseline, stress_cfg)

    scenarios = _filter_scenarios(stress_cfg.scenarios, args.scenarios)
    if not scenarios:
        _LOGGER.warning("Brak scenariuszy do uruchomienia")
        return 0

    report = run_portfolio_stress(
        baseline,
        scenarios,
        report_metadata=stress_cfg.metadata,
    )

    output_json = Path(args.output_json).expanduser()
    output_csv = Path(args.output_csv).expanduser() if args.output_csv else None
    signature_path = Path(args.signature_path).expanduser() if args.signature_path else None

    key_bytes, key_id = _resolve_signing_key(args, stress_cfg)
    _write_outputs(
        report,
        output_json=output_json,
        output_csv=output_csv,
        pretty=not args.no_pretty,
        signature_path=signature_path,
        signing_key=key_bytes,
        signing_key_id=key_id,
    )
    _LOGGER.info("Raport portfolio_stress zapisany do %s", output_json)
    _LOGGER.info(_summarize(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
