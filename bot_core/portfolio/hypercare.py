"""Cykl hypercare dla PortfolioGovernora Stage6."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from bot_core.market_intel import MarketIntelSnapshot
from bot_core.observability.slo import SLOStatus
from bot_core.portfolio.governor import PortfolioDecision, PortfolioGovernor
from bot_core.portfolio.io import load_allocations_file, load_market_intel_report
from bot_core.risk import StressOverrideRecommendation
from bot_core.runtime.portfolio_inputs import (
    build_portfolio_stress_provider,
    build_slo_status_provider,
    build_stress_override_provider,
)
from bot_core.security.signing import build_hmac_signature


@dataclass(slots=True)
class PortfolioCycleInputs:
    """Ścieżki wejściowe i parametry ewaluacji governora."""

    allocations_path: Path
    market_intel_path: Path
    portfolio_value: float
    slo_report_path: Path | None = None
    stress_report_path: Path | None = None
    portfolio_stress_report_path: Path | None = None
    fallback_directories: Sequence[Path] = ()
    market_intel_required_symbols: Sequence[str] | None = None
    market_intel_max_age: timedelta | None = None
    slo_max_age: timedelta | None = None
    stress_max_age: timedelta | None = None
    portfolio_stress_max_age: timedelta | None = None


@dataclass(slots=True)
class PortfolioCycleOutputConfig:
    """Konfiguracja artefaktów wynikowych cyklu."""

    summary_path: Path
    signature_path: Path | None = None
    csv_path: Path | None = None
    pretty_json: bool = True


@dataclass(slots=True)
class PortfolioCycleConfig:
    """Pełna konfiguracja cyklu hypercare portfela."""

    inputs: PortfolioCycleInputs
    output: PortfolioCycleOutputConfig
    signing_key: bytes | None = None
    signing_key_id: str | None = None
    metadata: Mapping[str, Any] | None = None
    log_context: Mapping[str, object] | None = None


@dataclass(slots=True)
class PortfolioCycleResult:
    """Artefakty zwracane po wykonaniu cyklu."""

    decision: PortfolioDecision
    summary_path: Path
    signature_path: Path | None
    csv_path: Path | None
    market_intel_metadata: Mapping[str, Any]
    slo_statuses: Mapping[str, SLOStatus]
    stress_overrides: Sequence[StressOverrideRecommendation]
    portfolio_stress: Mapping[str, Any]


class PortfolioHypercareCycle:
    """Automatyzuje pojedynczy przebieg ewaluacji PortfolioGovernora."""

    def __init__(self, governor: PortfolioGovernor, config: PortfolioCycleConfig) -> None:
        self._governor = governor
        self._config = config

    def run(self) -> PortfolioCycleResult:
        inputs = self._config.inputs

        allocations = load_allocations_file(inputs.allocations_path.expanduser())
        market_data, market_meta = load_market_intel_report(inputs.market_intel_path.expanduser())
        self._validate_market_intel(inputs, market_data, market_meta)

        slo_statuses = self._load_slo_statuses()
        stress_overrides = self._load_stress_overrides()
        portfolio_stress = self._load_portfolio_stress_summary()

        log_context = dict(self._config.log_context or {})
        log_context.setdefault("source", "portfolio_hypercare_cycle")
        log_context.setdefault(
            "inputs",
            {
                "allocations": str(inputs.allocations_path),
                "market_intel": str(inputs.market_intel_path),
                "slo_report": str(inputs.slo_report_path) if inputs.slo_report_path else None,
                "stress_report": str(inputs.stress_report_path) if inputs.stress_report_path else None,
                "portfolio_stress": str(inputs.portfolio_stress_report_path)
                if inputs.portfolio_stress_report_path
                else None,
            },
        )
        log_context.setdefault("portfolio_value", float(inputs.portfolio_value))

        decision = self._governor.evaluate(
            portfolio_value=float(inputs.portfolio_value),
            allocations=allocations,
            market_data=market_data,
            stress_overrides=stress_overrides,
            slo_statuses=slo_statuses or None,
            log_context=log_context,
        )

        summary = self._build_summary(
            decision,
            market_meta,
            slo_statuses,
            stress_overrides,
            portfolio_stress,
        )
        summary_path = self._write_summary(summary)
        csv_path = self._write_csv(decision)
        signature_path = self._write_signature(summary)

        return PortfolioCycleResult(
            decision=decision,
            summary_path=summary_path,
            signature_path=signature_path,
            csv_path=csv_path,
            market_intel_metadata=market_meta,
            slo_statuses=slo_statuses,
            stress_overrides=stress_overrides,
            portfolio_stress=portfolio_stress,
        )

    def _validate_market_intel(
        self,
        inputs: PortfolioCycleInputs,
        market_data: Mapping[str, MarketIntelSnapshot],
        market_meta: Mapping[str, Any],
    ) -> None:
        required = inputs.market_intel_required_symbols
        if not required:
            governor_config = getattr(self._governor, "_config", None)
            if governor_config is not None:
                required = [asset.symbol for asset in getattr(governor_config, "assets", ())]
        missing = [symbol for symbol in required or () if symbol not in market_data]
        if missing:
            raise ValueError(
                "Brak metryk Market Intel dla: " + ", ".join(sorted(missing))
            )
        max_age = inputs.market_intel_max_age
        generated_at = market_meta.get("generated_at")
        if max_age and isinstance(generated_at, datetime):
            now = datetime.now(timezone.utc)
            if now - generated_at > max_age:
                raise ValueError(
                    "Raport Market Intel jest przestarzały (" "wiek {} min)".format(
                        round((now - generated_at).total_seconds() / 60.0, 1)
                    )
                )

    def _load_slo_statuses(self) -> dict[str, SLOStatus]:
        path = self._config.inputs.slo_report_path
        if not path:
            return {}
        provider = build_slo_status_provider(
            path,
            fallback_directories=self._config.inputs.fallback_directories,
            max_age=self._config.inputs.slo_max_age,
        )
        return dict(provider())

    def _load_stress_overrides(self) -> tuple[StressOverrideRecommendation, ...]:
        path = self._config.inputs.stress_report_path
        if not path:
            return ()
        provider = build_stress_override_provider(
            path,
            fallback_directories=self._config.inputs.fallback_directories,
            max_age=self._config.inputs.stress_max_age,
        )
        return tuple(provider())

    def _load_portfolio_stress_summary(self) -> Mapping[str, Any]:
        path = self._config.inputs.portfolio_stress_report_path
        if not path:
            return {}
        provider = build_portfolio_stress_provider(
            path,
            fallback_directories=self._config.inputs.fallback_directories,
            max_age=self._config.inputs.portfolio_stress_max_age,
        )
        return dict(provider())

    def _build_summary(
        self,
        decision: PortfolioDecision,
        market_meta: Mapping[str, Any],
        slo_statuses: Mapping[str, SLOStatus],
        stress_overrides: Sequence[StressOverrideRecommendation],
        portfolio_stress: Mapping[str, Any],
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        summary: dict[str, Any] = {
            "generated_at": now.isoformat(),
            "portfolio_id": decision.portfolio_id,
            "portfolio_value": decision.portfolio_value,
            "rebalance_required": decision.rebalance_required,
            "adjustment_count": len(decision.adjustments),
            "advisory_count": len(decision.advisories),
            "adjustments": [adjustment.to_dict() for adjustment in decision.adjustments],
            "advisories": [advisory.to_dict() for advisory in decision.advisories],
            "inputs": {
                "allocations": str(self._config.inputs.allocations_path),
                "market_intel": str(self._config.inputs.market_intel_path),
                "slo_report": str(self._config.inputs.slo_report_path)
                if self._config.inputs.slo_report_path
                else None,
                "stress_report": str(self._config.inputs.stress_report_path)
                if self._config.inputs.stress_report_path
                else None,
            },
        }
        if self._config.metadata:
            summary["metadata"] = dict(self._config.metadata)
        if market_meta:
            meta_payload: dict[str, Any] = {}
            for key, value in market_meta.items():
                if isinstance(value, datetime):
                    meta_payload[key] = value.isoformat()
                else:
                    meta_payload[key] = value
            summary["market_intel"] = meta_payload
        if slo_statuses:
            summary["slo_statuses"] = {
                name: {
                    "status": status.status,
                    "severity": status.severity,
                    "value": status.value,
                    "target": status.target,
                }
                for name, status in slo_statuses.items()
            }
        if stress_overrides:
            summary["stress_overrides"] = [
                {
                    "reason": override.reason,
                    "severity": override.severity,
                    "symbol": override.symbol,
                    "risk_budget": override.risk_budget,
                    "force_rebalance": override.force_rebalance,
                }
                for override in stress_overrides
            ]
        if portfolio_stress:
            summary["portfolio_stress"] = portfolio_stress
        return summary

    def _write_summary(self, payload: Mapping[str, Any]) -> Path:
        path = self._config.output.summary_path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            if self._config.output.pretty_json:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            else:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        return path

    def _write_csv(self, decision: PortfolioDecision) -> Path | None:
        if not self._config.output.csv_path:
            return None
        path = self._config.output.csv_path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "symbol",
            "current_weight",
            "proposed_weight",
            "reason",
            "severity",
            "target_weight",
            "tolerance",
            "value_delta",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for adjustment in decision.adjustments:
                metadata = dict(getattr(adjustment, "metadata", {}) or {})
                writer.writerow(
                    {
                        "symbol": adjustment.symbol,
                        "current_weight": adjustment.current_weight,
                        "proposed_weight": adjustment.proposed_weight,
                        "reason": adjustment.reason,
                        "severity": adjustment.severity,
                        "target_weight": metadata.get("target_weight"),
                        "tolerance": metadata.get("tolerance"),
                        "value_delta": metadata.get("value_delta"),
                    }
                )
        return path

    def _write_signature(self, payload: Mapping[str, Any]) -> Path | None:
        if not self._config.signing_key:
            return None
        path = (
            self._config.output.signature_path.expanduser()
            if self._config.output.signature_path
            else self._config.output.summary_path.with_suffix(".sig")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        signature = build_hmac_signature(
            payload,
            key=self._config.signing_key,
            key_id=self._config.signing_key_id,
        )
        with path.open("w", encoding="utf-8") as handle:
            json.dump(signature, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        return path


__all__ = [
    "PortfolioCycleInputs",
    "PortfolioCycleOutputConfig",
    "PortfolioCycleConfig",
    "PortfolioCycleResult",
    "PortfolioHypercareCycle",
]
