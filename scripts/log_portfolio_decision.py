"""Loguje decyzję PortfolioGovernora i zapisuje podpisany wpis JSONL."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.config import load_core_config
from bot_core.observability import SLOStatus
from bot_core.portfolio import (
    AssetPortfolioGovernorConfig,
    PortfolioAssetConfig,
    PortfolioDecisionLog,
    PortfolioDriftTolerance,
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


def _fallback_governor_config(
    raw_config: Mapping[str, object],
    name: str,
) -> AssetPortfolioGovernorConfig | None:
    if not isinstance(raw_config, Mapping):
        return None
    assets_cfg: list[PortfolioAssetConfig] = []
    for entry in raw_config.get("assets", []) or []:  # type: ignore[assignment]
        if not isinstance(entry, Mapping):
            continue
        symbol = entry.get("symbol")
        if not symbol:
            continue
        assets_cfg.append(
            PortfolioAssetConfig(
                symbol=str(symbol),
                target_weight=float(entry.get("target_weight", 0.0)),
                min_weight=float(entry.get("min_weight", 0.0)),
                max_weight=float(entry.get("max_weight", 1.0)),
            )
        )
    if not assets_cfg:
        return None

    drift_entry = raw_config.get("drift_tolerance") or raw_config.get("drift") or {}
    drift = PortfolioDriftTolerance()
    if isinstance(drift_entry, Mapping):
        drift = PortfolioDriftTolerance(
            absolute=float(drift_entry.get("absolute", drift_entry.get("abs", drift.absolute))),
            relative=float(drift_entry.get("relative", drift_entry.get("rel", drift.relative))),
        )

    return AssetPortfolioGovernorConfig(
        name=str(raw_config.get("name", name)),
        portfolio_id=str(raw_config.get("portfolio_id", name)),
        drift_tolerance=drift,
        rebalance_cooldown_seconds=int(raw_config.get("rebalance_cooldown_seconds", 900)),
        min_rebalance_value=float(raw_config.get("min_rebalance_value", 0.0)),
        min_rebalance_weight=float(raw_config.get("min_rebalance_weight", 0.0)),
        assets=tuple(assets_cfg),
    )


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
    governor_cfg = core_config.portfolio_governors.get(args.governor)
    if governor_cfg is None:
        raw_core = load_json_or_yaml(Path(args.config))
        if isinstance(raw_core, Mapping):
            raw_governor = (raw_core.get("portfolio_governors") or {}).get(args.governor)  # type: ignore[index]
            governor_cfg = _fallback_governor_config(raw_governor or {}, args.governor) if raw_governor else None
    if governor_cfg is None:
        raise SystemExit(f"PortfolioGovernor {args.governor} nie istnieje w konfiguracji")
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
