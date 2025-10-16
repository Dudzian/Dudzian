"""Uruchamia cykl hypercare PortfolioGovernora Stage6."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.config import load_core_config  # noqa: E402 - import po modyfikacji sys.path
from bot_core.portfolio import (  # noqa: E402
    PortfolioCycleConfig,
    PortfolioCycleInputs,
    PortfolioCycleOutputConfig,
    PortfolioDecisionLog,
    PortfolioGovernor,
    PortfolioHypercareCycle,
    resolve_decision_log_config,
)


def _default_summary_path(governor: str) -> Path:
    return Path("var/audit/portfolio") / f"portfolio_cycle_{governor}.json"


def _default_decision_log_path(governor: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("var/audit/decision_log") / f"portfolio_decision_{governor}_{timestamp}.jsonl"


def _minutes_to_timedelta(value: float | None) -> timedelta | None:
    if value in (None, 0):
        return None
    return timedelta(minutes=float(value))


def _parse_key_value(items: Sequence[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not items:
        return result
    for item in items:
        if "=" not in item:
            raise ValueError("Wpisz metadane w formacie klucz=wartość")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Klucz metadanych nie może być pusty")
        result[key] = value.strip()
    return result


def _load_signing_key(value: str | None, path: str | None, env: str | None) -> bytes | None:
    provided = [item for item in (value, path, env) if item]
    if len(provided) > 1:
        raise ValueError("Klucz HMAC podaj jako wartość, plik lub zmienną środowiskową – wybierz jedną opcję")
    if value:
        return value.encode("utf-8")
    if env:
        env_value = os.environ.get(env)
        if not env_value:
            raise ValueError(f"Zmienna środowiskowa {env} nie zawiera klucza HMAC")
        return env_value.encode("utf-8")
    if path:
        file_path = Path(path).expanduser()
        if not file_path.is_file():
            raise ValueError(f"Plik z kluczem HMAC nie istnieje: {file_path}")
        return file_path.read_bytes().strip()
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wykonuje cykl hypercare PortfolioGovernora Stage6")
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do pliku core.yaml")
    parser.add_argument("--environment", required=True, help="Środowisko z pliku konfiguracyjnego")
    parser.add_argument("--governor", required=True, help="Nazwa PortfolioGovernora w konfiguracji")
    parser.add_argument("--allocations", required=True, help="Plik JSON/YAML z aktualnymi alokacjami")
    parser.add_argument("--portfolio-value", type=float, required=True, help="Wartość portfela w USD")
    parser.add_argument("--market-intel", required=True, help="Raport Market Intel (JSON)")
    parser.add_argument("--slo-report", help="Raport SLO (JSON)")
    parser.add_argument("--stress-report", help="Raport Stress Lab (JSON)")
    parser.add_argument("--fallback-dir", action="append", help="Katalog fallback dla raportów SLO/Stress")
    parser.add_argument("--market-intel-required", action="append", help="Symbol wymagany w raporcie Market Intel")
    parser.add_argument("--market-intel-max-age", type=float, help="Maksymalny wiek Market Intel (min)")
    parser.add_argument("--slo-max-age", type=float, help="Maksymalny wiek raportu SLO (min)")
    parser.add_argument("--stress-max-age", type=float, help="Maksymalny wiek raportu Stress Lab (min)")
    parser.add_argument("--summary", help="Ścieżka raportu podsumowania (JSON)")
    parser.add_argument("--summary-signature", help="Ścieżka podpisu HMAC dla podsumowania")
    parser.add_argument("--summary-csv", help="Opcjonalny CSV z korektami alokacji")
    parser.add_argument("--summary-pretty", action="store_true", help="Formatuj JSON z wcięciami")
    parser.add_argument("--metadata", action="append", help="Metadane raportu w formacie klucz=wartość")
    parser.add_argument("--log-context", action="append", help="Dodatkowy kontekst logowania klucz=wartość")
    parser.add_argument("--signing-key", help="Klucz HMAC dla raportu podsumowania")
    parser.add_argument("--signing-key-env", help="Zmienna środowiskowa z kluczem HMAC")
    parser.add_argument("--signing-key-path", help="Plik z kluczem HMAC")
    parser.add_argument("--signing-key-id", help="Identyfikator klucza HMAC")
    parser.add_argument("--decision-log", help="Ścieżka pliku JSONL decision logu")
    parser.add_argument("--skip-decision-log", action="store_true", help="Pomiń zapis do decision logu")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        core_config = load_core_config(args.config)
        if args.environment not in core_config.environments:
            raise ValueError(f"Środowisko {args.environment} nie istnieje w konfiguracji")
        if args.governor not in core_config.portfolio_governors:
            raise ValueError(f"PortfolioGovernor {args.governor} nie istnieje w konfiguracji")

        governor_cfg = core_config.portfolio_governors[args.governor]
        metadata = {"environment": args.environment, "governor": args.governor}
        metadata.update(_parse_key_value(args.metadata))

        log_context = {"environment": args.environment, "governor": args.governor}
        log_context.update(_parse_key_value(args.log_context))

        signing_key = _load_signing_key(args.signing_key, args.signing_key_path, args.signing_key_env)

        fallback_dirs = tuple(Path(item).expanduser() for item in args.fallback_dir or [])
        inputs = PortfolioCycleInputs(
            allocations_path=Path(args.allocations).expanduser(),
            market_intel_path=Path(args.market_intel).expanduser(),
            portfolio_value=float(args.portfolio_value),
            slo_report_path=Path(args.slo_report).expanduser() if args.slo_report else None,
            stress_report_path=Path(args.stress_report).expanduser() if args.stress_report else None,
            fallback_directories=fallback_dirs,
            market_intel_required_symbols=tuple(args.market_intel_required) if args.market_intel_required else None,
            market_intel_max_age=_minutes_to_timedelta(args.market_intel_max_age),
            slo_max_age=_minutes_to_timedelta(args.slo_max_age),
            stress_max_age=_minutes_to_timedelta(args.stress_max_age),
        )

        summary_path = Path(args.summary).expanduser() if args.summary else _default_summary_path(args.governor)
        output = PortfolioCycleOutputConfig(
            summary_path=summary_path,
            signature_path=Path(args.summary_signature).expanduser() if args.summary_signature else None,
            csv_path=Path(args.summary_csv).expanduser() if args.summary_csv else None,
            pretty_json=bool(args.summary_pretty),
        )

        cycle_config = PortfolioCycleConfig(
            inputs=inputs,
            output=output,
            signing_key=signing_key,
            signing_key_id=args.signing_key_id,
            metadata=metadata,
            log_context=log_context,
        )

        decision_log_path: Path | None = None
        log_kwargs: dict[str, Any] = {}
        if not args.skip_decision_log:
            configured_path, log_kwargs = resolve_decision_log_config(core_config)
            decision_log_path = (
                Path(args.decision_log).expanduser()
                if args.decision_log
                else configured_path
            )
            if decision_log_path is None:
                decision_log_path = _default_decision_log_path(args.governor)
            decision_log_path.parent.mkdir(parents=True, exist_ok=True)
            decision_log = PortfolioDecisionLog(
                jsonl_path=decision_log_path,
                **log_kwargs,
            )
        else:
            decision_log = None

        governor = PortfolioGovernor(governor_cfg, decision_log=decision_log)
        cycle = PortfolioHypercareCycle(governor, cycle_config)
        result = cycle.run()

        print(f"Wygenerowano podsumowanie cyklu portfelowego w {result.summary_path}")
        if result.signature_path:
            key_id: str | None = args.signing_key_id
            if not key_id and not args.skip_decision_log:
                candidate = log_kwargs.get("signing_key_id")
                if isinstance(candidate, bytes):
                    key_id = candidate.decode("utf-8")
                elif isinstance(candidate, str):
                    key_id = candidate
            info = key_id or "brak-id"
            print(f"Podpis HMAC zapisany w {result.signature_path} (key_id={info})")
        if result.csv_path:
            print(f"Raport CSV korekt zapisano w {result.csv_path}")
        if decision_log_path is not None:
            print(f"Decision log: {decision_log_path}")
    except Exception as exc:  # pragma: no cover - obsługa błędów CLI
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(run())
