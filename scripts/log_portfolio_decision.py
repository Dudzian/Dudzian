"""Loguje decyzję PortfolioGovernora i zapisuje podpisany wpis JSONL."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from bot_core.config import load_core_config
from bot_core.observability import SLOStatus
from bot_core.portfolio import (
    PortfolioDecisionLog,
    PortfolioGovernor,
    load_allocations_file,
    load_json_or_yaml,
    parse_market_intel_payload,
    parse_slo_status_payload,
    parse_stress_overrides_payload,
    resolve_decision_log_config,
)
from bot_core.risk import StressOverrideRecommendation


def _default_output(governor: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("var/audit/decision_log") / f"portfolio_decision_{governor}_{timestamp}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Loguje decyzję PortfolioGovernora Stage6")
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do pliku core.yaml")
    parser.add_argument("--environment", required=True, help="Nazwa środowiska w core.yaml")
    parser.add_argument("--governor", required=True, help="Nazwa PortfolioGovernora")
    parser.add_argument("--allocations", required=True, help="Plik JSON/YAML z alokacjami")
    parser.add_argument("--portfolio-value", type=float, required=True, help="Wartość portfela w USD")
    parser.add_argument("--market-intel", required=True, help="Raport Market Intel w formacie JSON")
    parser.add_argument("--slo-status", help="Raport SLO (opcjonalnie) JSON/YAML")
    parser.add_argument(
        "--stress-overrides",
        help="Raport Stress Lab (JSON) z sekcją overrides do zastosowania w decyzji",
    )
    parser.add_argument("--output", help="Ścieżka pliku JSONL z decyzją portfelową")
    parser.add_argument("--signing-key", help="Klucz HMAC podany bezpośrednio w CLI")
    parser.add_argument("--signing-key-env", help="Nazwa zmiennej z kluczem HMAC")
    parser.add_argument("--signing-key-path", help="Plik z kluczem HMAC")
    parser.add_argument("--signing-key-id", help="Identyfikator klucza HMAC")
    args = parser.parse_args()

    core_config = load_core_config(args.config)
    if args.environment not in core_config.environments:
        raise SystemExit(f"Środowisko {args.environment} nie istnieje w konfiguracji")
    if args.governor not in core_config.portfolio_governors:
        raise SystemExit(f"PortfolioGovernor {args.governor} nie istnieje w konfiguracji")

    governor_cfg = core_config.portfolio_governors[args.governor]
    allocations = load_allocations_file(Path(args.allocations))
    market_payload = load_json_or_yaml(Path(args.market_intel))
    if not isinstance(market_payload, Mapping):
        raise SystemExit("Raport Market Intel musi być strukturą mapy")
    market_data = parse_market_intel_payload(market_payload)

    slo_statuses: dict[str, SLOStatus] | None = None
    if args.slo_status:
        slo_payload = load_json_or_yaml(Path(args.slo_status))
        if isinstance(slo_payload, Mapping):
            slo_statuses = parse_slo_status_payload(slo_payload)
        else:
            slo_statuses = {}

    stress_overrides: list[StressOverrideRecommendation] | None = None
    if args.stress_overrides:
        stress_payload = load_json_or_yaml(Path(args.stress_overrides))
        stress_overrides = list(parse_stress_overrides_payload(stress_payload))

    configured_path, log_kwargs = resolve_decision_log_config(core_config)
    output_path = Path(args.output) if args.output else configured_path or _default_output(args.governor)

    key = None
    key_id = None
    if args.signing_key:
        key = args.signing_key.encode("utf-8")
    elif args.signing_key_env:
        env_value = os.environ.get(args.signing_key_env)
        key = env_value.encode("utf-8") if env_value else None
    elif args.signing_key_path:
        path = Path(args.signing_key_path)
        key = path.read_bytes().strip() if path.exists() else None
    else:
        key = log_kwargs.pop("signing_key", None)
        key_id = log_kwargs.pop("signing_key_id", None)

    if args.signing_key_id:
        key_id = args.signing_key_id

    signing_key = key if key is not None else log_kwargs.pop("signing_key", None)
    signing_key_id = key_id if key_id is not None else log_kwargs.pop("signing_key_id", None)

    decision_log = PortfolioDecisionLog(
        jsonl_path=output_path,
        signing_key=signing_key,
        signing_key_id=signing_key_id,
        **log_kwargs,
    )
    governor = PortfolioGovernor(governor_cfg, decision_log=decision_log)

    metadata = {
        "environment": args.environment,
        "governor": args.governor,
        "inputs": {
            "allocations": str(Path(args.allocations)),
            "market_intel": str(Path(args.market_intel)),
            "slo_status": str(Path(args.slo_status)) if args.slo_status else None,
            "stress_overrides": str(Path(args.stress_overrides)) if args.stress_overrides else None,
        },
        "portfolio_value": float(args.portfolio_value),
    }

    governor.evaluate(
        portfolio_value=float(args.portfolio_value),
        allocations=allocations,
        market_data=market_data,
        stress_overrides=stress_overrides,
        slo_statuses=slo_statuses,
        log_context=metadata,
    )

    tail = decision_log.tail(limit=1)
    signature_info = "bez podpisu HMAC"
    if tail and isinstance(tail[-1], Mapping) and "signature" in tail[-1]:
        signature_info = f"z podpisem HMAC ({tail[-1]['signature'].get('key_id', 'brak-id')})"
    print(f"Zapisano decyzję portfelową do {output_path} {signature_info}")


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    main()
