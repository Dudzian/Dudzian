"""Runtime closed-loop utilities for Opportunity AI shadow/outcome lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from .opportunity_evaluation import (
    OpportunityPromotionGateConfig,
    OpportunityPromotionReport,
    OpportunityTemporalEvaluation,
    OpportunityTemporalEvaluator,
)
from .repository import ModelRepository
from .trading_engine import OpportunitySnapshot
from .trading_opportunity_shadow import (
    OpportunityOutcomeLabel,
    OpportunityShadowRecord,
    OpportunityShadowRepository,
)

_REQUIRED_FEATURE_CONTEXT_KEYS: tuple[str, ...] = (
    "signal_strength",
    "momentum_5m",
    "volatility_30m",
    "spread_bps",
    "fee_bps",
    "slippage_bps",
    "liquidity_score",
    "risk_penalty_bps",
)
_FINAL_LABEL_PREFIXES: tuple[str, ...] = ("final", "partial")


@dataclass(slots=True, frozen=True)
class OpportunityActivationReadiness:
    activation_ready: bool
    recommendation: str
    reasons: tuple[str, ...]
    alias_targets: Mapping[str, str | None]


@dataclass(slots=True, frozen=True)
class OpportunityPersistedPromotionReadinessReport:
    champion_version: str
    challenger_version: str
    champion_shadow_evaluation: OpportunityTemporalEvaluation | None
    challenger_shadow_evaluation: OpportunityTemporalEvaluation | None
    promotion_report: OpportunityPromotionReport | None
    activation_readiness: OpportunityActivationReadiness
    degraded_reasons: tuple[str, ...]
    matched_outcomes: Mapping[str, int]
    evaluation_window_start: datetime | None
    evaluation_window_end: datetime | None


class OpportunityLifecycleService:
    """Bridge runtime persisted shadow/outcome data with evaluator/promotion readiness."""

    def __init__(self, evaluator: OpportunityTemporalEvaluator | None = None) -> None:
        self._evaluator = evaluator or OpportunityTemporalEvaluator()

    def attach_outcome_label(
        self,
        repository: OpportunityShadowRepository,
        label: OpportunityOutcomeLabel,
    ) -> tuple[bool, str]:
        _, missing = repository.append_outcome_labels_for_existing_records([label])
        if missing:
            return False, f"missing_shadow_record:{missing[0]}"
        return True, "attached"

    def build_persisted_promotion_readiness(
        self,
        *,
        model_repository: ModelRepository,
        shadow_repository: OpportunityShadowRepository,
        champion_version: str,
        challenger_version: str,
        gate_config: OpportunityPromotionGateConfig | None = None,
    ) -> OpportunityPersistedPromotionReadinessReport:
        reasons: list[str] = []
        labels = shadow_repository.load_outcome_labels()
        quality_counts: dict[str, int] = {}
        for label in labels:
            quality = str(label.label_quality or "unknown")
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        eligible_labels = [
            label
            for label in labels
            if str(label.label_quality or "").strip().lower().startswith(_FINAL_LABEL_PREFIXES)
        ]
        if labels and not eligible_labels:
            reasons.append("proxy_only_outcomes_excluded_from_governance")
        elif labels and len(eligible_labels) < len(labels):
            reasons.append(
                "non_final_outcomes_excluded:"
                + ",".join(f"{key}:{quality_counts[key]}" for key in sorted(quality_counts))
            )
        champion_records = shadow_repository.load_shadow_records_for_model(champion_version)
        challenger_records = shadow_repository.load_shadow_records_for_model(challenger_version)

        champion_shadow = self._safe_shadow_eval(champion_records, eligible_labels, reasons)
        challenger_shadow = self._safe_shadow_eval(challenger_records, eligible_labels, reasons)

        promotion_report: OpportunityPromotionReport | None = None
        window_start: datetime | None = None
        window_end: datetime | None = None

        if champion_shadow is not None and challenger_shadow is not None:
            champion_artifact = model_repository.load_model(champion_version)
            challenger_artifact = model_repository.load_model(challenger_version)
            reference_samples, reference_missing_context = self._samples_from_persisted_data(
                champion_records,
                eligible_labels,
            )
            evaluation_samples, evaluation_missing_context = self._samples_from_persisted_data(
                challenger_records,
                eligible_labels,
            )
            missing_context_total = reference_missing_context + evaluation_missing_context
            if missing_context_total:
                reasons.append(
                    f"incomplete_feature_context_for_snapshot_reconstruction:{missing_context_total}"
                )
            if reference_samples and evaluation_samples:
                promotion_report = self._evaluator.build_promotion_report(
                    champion_artifact=champion_artifact,
                    challenger_artifact=challenger_artifact,
                    evaluation_samples=evaluation_samples,
                    reference_samples=reference_samples,
                    gate_config=gate_config,
                    champion_shadow=champion_shadow,
                    challenger_shadow=challenger_shadow,
                )
                ordered = sorted(
                    [*reference_samples, *evaluation_samples],
                    key=lambda row: row.as_of.timestamp()
                    if row.as_of is not None
                    else float("-inf"),
                )
                if ordered:
                    window_start = ordered[0].as_of
                    window_end = ordered[-1].as_of
            else:
                reasons.append("insufficient_persisted_samples_for_promotion_report")
        else:
            reasons.append("insufficient_shadow_outcome_pairs_for_promotion_report")

        alias_targets = {
            "latest": model_repository.get_alias_target("latest"),
            "champion": model_repository.get_alias_target("champion"),
            "challenger": model_repository.get_alias_target("challenger"),
        }
        readiness = self._build_activation_readiness(
            promotion_report=promotion_report,
            alias_targets=alias_targets,
            degraded_reasons=tuple(reasons),
            challenger_version=challenger_version,
        )

        return OpportunityPersistedPromotionReadinessReport(
            champion_version=champion_version,
            challenger_version=challenger_version,
            champion_shadow_evaluation=champion_shadow,
            challenger_shadow_evaluation=challenger_shadow,
            promotion_report=promotion_report,
            activation_readiness=readiness,
            degraded_reasons=tuple(reasons),
            matched_outcomes={
                "champion": champion_shadow.matched_outcomes if champion_shadow is not None else 0,
                "challenger": challenger_shadow.matched_outcomes
                if challenger_shadow is not None
                else 0,
            },
            evaluation_window_start=window_start,
            evaluation_window_end=window_end,
        )

    def _safe_shadow_eval(
        self,
        records: Sequence[OpportunityShadowRecord],
        labels: Sequence[OpportunityOutcomeLabel],
        reasons: list[str],
    ) -> OpportunityTemporalEvaluation | None:
        if not records:
            reasons.append("missing_shadow_records")
            return None
        try:
            return self._evaluator.evaluate_from_shadow_labels(records, labels)
        except ValueError as exc:
            reasons.append(str(exc))
            return None

    @staticmethod
    def _samples_from_persisted_data(
        records: Sequence[OpportunityShadowRecord],
        labels: Sequence[OpportunityOutcomeLabel],
    ) -> tuple[list[OpportunitySnapshot], int]:
        labels_by_key = {str(label.correlation_key): label for label in labels}
        samples: list[OpportunitySnapshot] = []
        missing_context_count = 0
        for record in records:
            label = labels_by_key.get(str(record.record_key))
            if label is None:
                continue
            candidate_metadata_obj = record.snapshot.get("candidate_metadata", {})
            candidate_metadata = (
                candidate_metadata_obj if isinstance(candidate_metadata_obj, Mapping) else {}
            )
            if any(key not in candidate_metadata for key in _REQUIRED_FEATURE_CONTEXT_KEYS):
                missing_context_count += 1
            samples.append(
                OpportunitySnapshot(
                    symbol=record.symbol,
                    signal_strength=float(candidate_metadata.get("signal_strength", 0.0)),
                    momentum_5m=float(candidate_metadata.get("momentum_5m", 0.0)),
                    volatility_30m=float(candidate_metadata.get("volatility_30m", 0.0)),
                    spread_bps=float(candidate_metadata.get("spread_bps", 0.0)),
                    fee_bps=float(candidate_metadata.get("fee_bps", 0.0)),
                    slippage_bps=float(candidate_metadata.get("slippage_bps", 0.0)),
                    liquidity_score=float(candidate_metadata.get("liquidity_score", 0.0)),
                    risk_penalty_bps=float(candidate_metadata.get("risk_penalty_bps", 0.0)),
                    realized_return_bps=float(label.realized_return_bps),
                    as_of=record.decision_timestamp,
                )
            )
        return samples, missing_context_count

    @staticmethod
    def _build_activation_readiness(
        *,
        promotion_report: OpportunityPromotionReport | None,
        alias_targets: Mapping[str, str | None],
        degraded_reasons: tuple[str, ...],
        challenger_version: str,
    ) -> OpportunityActivationReadiness:
        reasons: list[str] = list(degraded_reasons)
        if promotion_report is None:
            reasons.append("promotion_report_unavailable")
        elif not promotion_report.promotion_recommended:
            reasons.append("promotion_gate_failed")
        if reasons:
            return OpportunityActivationReadiness(
                activation_ready=False,
                recommendation="hold_current_aliases",
                reasons=tuple(reasons),
                alias_targets=alias_targets,
            )
        return OpportunityActivationReadiness(
            activation_ready=True,
            recommendation=f"manual_alias_switch_to:{challenger_version}",
            reasons=(),
            alias_targets=alias_targets,
        )


__all__ = [
    "OpportunityActivationReadiness",
    "OpportunityLifecycleService",
    "OpportunityPersistedPromotionReadinessReport",
]
