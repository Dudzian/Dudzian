"""Runtime closed-loop utilities for Opportunity AI shadow/outcome lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
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
_FINAL_LABEL_PREFIXES: tuple[str, ...] = ("final",)
_PARTIAL_LABEL_PREFIXES: tuple[str, ...] = ("partial",)
_NON_FINAL_EXCLUDED_PREFIX = "non_final_outcomes_excluded:"
_MIXED_FINAL_PARTIAL_PREFIX = "mixed_final_partial_outcomes_degraded:"


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


class OpportunityAutonomyMode(str, Enum):
    DENIED = "denied"
    SHADOW_ONLY = "shadow_only"
    PAPER_AUTONOMOUS = "paper_autonomous"
    LIVE_ASSISTED = "live_assisted"
    LIVE_AUTONOMOUS = "live_autonomous"


@dataclass(slots=True, frozen=True)
class OpportunityAutonomyGateConfig:
    min_final_outcomes_for_live_autonomy: int = 8
    min_total_matched_outcomes: int = 6
    min_observed_outcomes_for_paper_autonomy: int = 2
    allow_partial_only_for_shadow_or_paper: bool = False
    require_promotion_pass_for_live: bool = True
    require_activation_ready_for_live: bool = True
    max_allowed_degraded_reasons_for_live_assisted: int = 2
    blocking_degraded_reasons: tuple[str, ...] = (
        "proxy_only_outcomes_excluded_from_governance",
        "promotion_gate_failed",
    )
    allowlisted_degraded_reasons: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class OpportunityAutonomyDecision:
    mode: OpportunityAutonomyMode
    primary_reason: str
    reasons: tuple[str, ...]
    blocking_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    evidence_summary: Mapping[str, int | bool | str]

    @property
    def autonomous_execution_allowed(self) -> bool:
        return self.mode in {
            OpportunityAutonomyMode.PAPER_AUTONOMOUS,
            OpportunityAutonomyMode.LIVE_ASSISTED,
            OpportunityAutonomyMode.LIVE_AUTONOMOUS,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "autonomous_execution_allowed": self.autonomous_execution_allowed,
            "primary_reason": self.primary_reason,
            "reasons": list(self.reasons),
            "blocking_reasons": list(self.blocking_reasons),
            "warnings": list(self.warnings),
            "evidence_summary": dict(self.evidence_summary),
        }


@dataclass(slots=True, frozen=True)
class OpportunityAutonomyDowngradeConfig:
    downgrade_on_blocking_reason: bool = True
    blocking_reason_target_mode: OpportunityAutonomyMode = OpportunityAutonomyMode.DENIED
    downgrade_live_on_promotion_fail: bool = True
    downgrade_live_on_activation_not_ready: bool = True
    max_non_blocking_degradations_for_live_autonomous: int = 0
    max_non_blocking_degradations_for_live_assisted: int = 1
    allow_paper_on_partial_only: bool = False
    minimum_final_outcomes_to_keep_live: int = 8
    minimum_total_outcomes_to_keep_paper: int = 2
    minimum_total_outcomes_to_keep_shadow: int = 1
    proxy_only_target_mode: OpportunityAutonomyMode = OpportunityAutonomyMode.SHADOW_ONLY
    partial_only_target_mode: OpportunityAutonomyMode = OpportunityAutonomyMode.SHADOW_ONLY
    fail_closed_mode: OpportunityAutonomyMode = OpportunityAutonomyMode.DENIED
    blocking_reason_prefixes: tuple[str, ...] = (
        "proxy_only_outcomes_excluded_from_governance",
        "promotion_gate_failed",
    )


@dataclass(slots=True, frozen=True)
class OpportunityAutonomyDowngradeDecision:
    requested_mode: OpportunityAutonomyMode
    effective_mode: OpportunityAutonomyMode
    downgraded: bool
    primary_reason: str
    reasons: tuple[str, ...]
    blocking_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    downgrade_step_count: int
    downgrade_source: str
    evidence_summary: Mapping[str, int | bool | str]

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_mode": self.requested_mode.value,
            "effective_mode": self.effective_mode.value,
            "downgraded": self.downgraded,
            "primary_reason": self.primary_reason,
            "reasons": list(self.reasons),
            "blocking_reasons": list(self.blocking_reasons),
            "warnings": list(self.warnings),
            "downgrade_step_count": self.downgrade_step_count,
            "downgrade_source": self.downgrade_source,
            "evidence_summary": dict(self.evidence_summary),
        }


_AUTONOMY_MODE_ORDER: dict[OpportunityAutonomyMode, int] = {
    OpportunityAutonomyMode.DENIED: 0,
    OpportunityAutonomyMode.SHADOW_ONLY: 1,
    OpportunityAutonomyMode.PAPER_AUTONOMOUS: 2,
    OpportunityAutonomyMode.LIVE_ASSISTED: 3,
    OpportunityAutonomyMode.LIVE_AUTONOMOUS: 4,
}


def _downgrade_clamp_mode(
    *,
    requested_mode: OpportunityAutonomyMode,
    current_mode: OpportunityAutonomyMode,
    candidate_mode: OpportunityAutonomyMode,
) -> OpportunityAutonomyMode:
    """Enforce downgrade-only invariant for autonomy mode transitions."""
    max_allowed_mode = min(requested_mode, current_mode, key=_AUTONOMY_MODE_ORDER.get)
    return min(max_allowed_mode, candidate_mode, key=_AUTONOMY_MODE_ORDER.get)


@dataclass(slots=True, frozen=True)
class OpportunityExecutionPermission:
    environment: str
    autonomy_mode: OpportunityAutonomyMode
    autonomous_execution_allowed: bool
    assisted_override_required: bool
    assisted_override_used: bool
    primary_reason: str
    denial_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "environment": self.environment,
            "autonomy_mode": self.autonomy_mode.value,
            "autonomous_execution_allowed": self.autonomous_execution_allowed,
            "assisted_override_required": self.assisted_override_required,
            "assisted_override_used": self.assisted_override_used,
            "primary_reason": self.primary_reason,
            "denial_reason": self.denial_reason,
        }


def evaluate_autonomy_downgrade(
    *,
    requested_mode: OpportunityAutonomyMode,
    readiness_report: OpportunityPersistedPromotionReadinessReport,
    config: OpportunityAutonomyDowngradeConfig | None = None,
) -> OpportunityAutonomyDowngradeDecision:
    policy = config or OpportunityAutonomyDowngradeConfig()
    degraded_reasons = tuple(readiness_report.degraded_reasons)
    quality_counts = OpportunityLifecycleService._extract_quality_counts(degraded_reasons)
    matched = tuple(int(value) for value in readiness_report.matched_outcomes.values())
    paired_matched_outcomes = min(matched) if matched else 0
    final_outcomes = max(quality_counts.get("final", 0), paired_matched_outcomes)
    partial_outcomes = sum(
        value for key, value in quality_counts.items() if key.startswith(_PARTIAL_LABEL_PREFIXES)
    )
    observed_outcomes = sum(quality_counts.values()) if quality_counts else paired_matched_outcomes
    has_partial_only = "partial_only_outcomes_excluded_from_governance" in degraded_reasons
    has_proxy_only = "proxy_only_outcomes_excluded_from_governance" in degraded_reasons
    promotion_passed = (
        readiness_report.promotion_report is not None
        and readiness_report.promotion_report.promotion_recommended
    )
    activation_ready = readiness_report.activation_readiness.activation_ready
    blocking_reasons = tuple(
        reason
        for reason in degraded_reasons
        if any(reason.startswith(prefix) for prefix in policy.blocking_reason_prefixes)
    )
    non_blocking_degraded_reasons = tuple(
        reason for reason in degraded_reasons if reason not in set(blocking_reasons)
    )

    evidence_summary: dict[str, int | bool | str] = {
        "paired_matched_outcomes": paired_matched_outcomes,
        "observed_outcomes": observed_outcomes,
        "final_outcomes": final_outcomes,
        "partial_outcomes": partial_outcomes,
        "promotion_passed": promotion_passed,
        "activation_ready": activation_ready,
        "has_proxy_only_evidence": has_proxy_only,
        "has_partial_only_evidence": has_partial_only,
        "blocking_reason_count": len(blocking_reasons),
        "non_blocking_degradation_count": len(non_blocking_degraded_reasons),
    }
    reasons: list[str] = []
    warnings: list[str] = list(degraded_reasons)

    try:
        target_mode = requested_mode
        if policy.downgrade_on_blocking_reason and blocking_reasons:
            target_mode = _downgrade_clamp_mode(
                requested_mode=requested_mode,
                current_mode=target_mode,
                candidate_mode=policy.blocking_reason_target_mode,
            )
            reasons.append("blocking_degradation_present")
        elif has_proxy_only:
            target_mode = _downgrade_clamp_mode(
                requested_mode=requested_mode,
                current_mode=target_mode,
                candidate_mode=policy.proxy_only_target_mode,
            )
            reasons.append("proxy_only_evidence_requires_safe_mode")
        elif has_partial_only and not policy.allow_paper_on_partial_only:
            target_mode = _downgrade_clamp_mode(
                requested_mode=requested_mode,
                current_mode=target_mode,
                candidate_mode=policy.partial_only_target_mode,
            )
            reasons.append("partial_only_evidence_requires_safe_mode")

        if requested_mode is OpportunityAutonomyMode.LIVE_AUTONOMOUS:
            if policy.downgrade_live_on_promotion_fail and not promotion_passed:
                target_mode = _downgrade_clamp_mode(
                    requested_mode=requested_mode,
                    current_mode=target_mode,
                    candidate_mode=OpportunityAutonomyMode.LIVE_ASSISTED,
                )
                reasons.append("promotion_not_ready_for_live_autonomous")
            if policy.downgrade_live_on_activation_not_ready and not activation_ready:
                target_mode = _downgrade_clamp_mode(
                    requested_mode=requested_mode,
                    current_mode=target_mode,
                    candidate_mode=OpportunityAutonomyMode.LIVE_ASSISTED,
                )
                reasons.append("activation_not_ready_for_live_autonomous")
            if final_outcomes < policy.minimum_final_outcomes_to_keep_live:
                target_mode = _downgrade_clamp_mode(
                    requested_mode=requested_mode,
                    current_mode=target_mode,
                    candidate_mode=OpportunityAutonomyMode.LIVE_ASSISTED,
                )
                reasons.append("insufficient_final_outcomes_for_live_autonomous")
            if (
                len(non_blocking_degraded_reasons)
                > policy.max_non_blocking_degradations_for_live_autonomous
            ):
                target_mode = _downgrade_clamp_mode(
                    requested_mode=requested_mode,
                    current_mode=target_mode,
                    candidate_mode=OpportunityAutonomyMode.LIVE_ASSISTED,
                )
                reasons.append("too_many_non_blocking_degradations_for_live_autonomous")

        if requested_mode in {
            OpportunityAutonomyMode.LIVE_AUTONOMOUS,
            OpportunityAutonomyMode.LIVE_ASSISTED,
        }:
            if (
                len(non_blocking_degraded_reasons)
                > policy.max_non_blocking_degradations_for_live_assisted
            ):
                target_mode = _downgrade_clamp_mode(
                    requested_mode=requested_mode,
                    current_mode=target_mode,
                    candidate_mode=OpportunityAutonomyMode.PAPER_AUTONOMOUS,
                )
                reasons.append("too_many_non_blocking_degradations_for_live_assisted")
            if observed_outcomes < policy.minimum_total_outcomes_to_keep_paper:
                target_mode = _downgrade_clamp_mode(
                    requested_mode=requested_mode,
                    current_mode=target_mode,
                    candidate_mode=OpportunityAutonomyMode.SHADOW_ONLY,
                )
                reasons.append("insufficient_outcomes_for_paper_autonomy")

        if requested_mode is OpportunityAutonomyMode.PAPER_AUTONOMOUS:
            if observed_outcomes < policy.minimum_total_outcomes_to_keep_paper:
                target_mode = _downgrade_clamp_mode(
                    requested_mode=requested_mode,
                    current_mode=target_mode,
                    candidate_mode=OpportunityAutonomyMode.SHADOW_ONLY,
                )
                reasons.append("insufficient_outcomes_for_paper_autonomy")

        if target_mode is OpportunityAutonomyMode.SHADOW_ONLY:
            if observed_outcomes < policy.minimum_total_outcomes_to_keep_shadow:
                target_mode = _downgrade_clamp_mode(
                    requested_mode=requested_mode,
                    current_mode=target_mode,
                    candidate_mode=OpportunityAutonomyMode.DENIED,
                )
                reasons.append("insufficient_outcomes_for_shadow_mode")
        elif target_mode is OpportunityAutonomyMode.PAPER_AUTONOMOUS:
            if observed_outcomes < policy.minimum_total_outcomes_to_keep_paper:
                target_mode = _downgrade_clamp_mode(
                    requested_mode=requested_mode,
                    current_mode=target_mode,
                    candidate_mode=OpportunityAutonomyMode.SHADOW_ONLY,
                )
                reasons.append("paper_autonomy_requires_more_outcomes")

        downgraded = _AUTONOMY_MODE_ORDER[target_mode] < _AUTONOMY_MODE_ORDER[requested_mode]
        if not reasons:
            reasons.append("downgrade_not_required")
        primary_reason = reasons[0]
        step_count = max(0, _AUTONOMY_MODE_ORDER[requested_mode] - _AUTONOMY_MODE_ORDER[target_mode])
        return OpportunityAutonomyDowngradeDecision(
            requested_mode=requested_mode,
            effective_mode=target_mode,
            downgraded=downgraded,
            primary_reason=primary_reason,
            reasons=tuple(reasons),
            blocking_reasons=blocking_reasons,
            warnings=tuple(warnings),
            downgrade_step_count=step_count,
            downgrade_source="opportunity_autonomy_downgrade_policy_v1",
            evidence_summary=evidence_summary,
        )
    except Exception as exc:  # noqa: BLE001
        return OpportunityAutonomyDowngradeDecision(
            requested_mode=requested_mode,
            effective_mode=_downgrade_clamp_mode(
                requested_mode=requested_mode,
                current_mode=requested_mode,
                candidate_mode=policy.fail_closed_mode,
            ),
            downgraded=True,
            primary_reason="downgrade_evaluation_failed",
            reasons=("downgrade_evaluation_failed",),
            blocking_reasons=blocking_reasons,
            warnings=(*warnings, f"downgrade_evaluation_error:{exc}"),
            downgrade_step_count=max(
                0,
                _AUTONOMY_MODE_ORDER[requested_mode]
                - _AUTONOMY_MODE_ORDER[
                    _downgrade_clamp_mode(
                        requested_mode=requested_mode,
                        current_mode=requested_mode,
                        candidate_mode=policy.fail_closed_mode,
                    )
                ],
            ),
            downgrade_source="opportunity_autonomy_downgrade_policy_v1",
            evidence_summary=evidence_summary,
        )


def evaluate_opportunity_execution_permission(
    *,
    decision: OpportunityAutonomyDecision,
    environment: str,
    assisted_approval: bool = False,
) -> OpportunityExecutionPermission:
    normalized_environment = str(environment or "").strip().lower()
    is_live_environment = normalized_environment in {"live", "prod", "production"}
    mode = decision.mode
    assisted_override_required = (
        mode is OpportunityAutonomyMode.LIVE_ASSISTED and is_live_environment
    )
    assisted_override_used = assisted_override_required and bool(assisted_approval)
    denial_reason: str | None = None
    allowed = False

    if mode is OpportunityAutonomyMode.DENIED:
        denial_reason = "autonomy_mode_denied"
    elif mode is OpportunityAutonomyMode.SHADOW_ONLY:
        denial_reason = "autonomy_mode_shadow_only_blocks_order_execution"
    elif mode is OpportunityAutonomyMode.PAPER_AUTONOMOUS:
        if is_live_environment:
            denial_reason = "paper_autonomy_blocks_live_environment"
        else:
            allowed = True
    elif mode is OpportunityAutonomyMode.LIVE_ASSISTED:
        if is_live_environment:
            if assisted_override_used:
                allowed = True
            else:
                denial_reason = "live_assisted_requires_explicit_approval"
        else:
            allowed = True
    elif mode is OpportunityAutonomyMode.LIVE_AUTONOMOUS:
        allowed = True
    else:  # pragma: no cover - defensywny fallback na nieznany enum
        denial_reason = "unsupported_autonomy_mode"

    return OpportunityExecutionPermission(
        environment=normalized_environment,
        autonomy_mode=mode,
        autonomous_execution_allowed=allowed,
        assisted_override_required=assisted_override_required,
        assisted_override_used=assisted_override_used,
        primary_reason=decision.primary_reason,
        denial_reason=denial_reason,
    )


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
        final_labels = [
            label
            for label in labels
            if str(label.label_quality or "").strip().lower().startswith(_FINAL_LABEL_PREFIXES)
        ]
        partial_labels = [
            label
            for label in labels
            if str(label.label_quality or "").strip().lower().startswith(_PARTIAL_LABEL_PREFIXES)
        ]
        eligible_labels = [label for label in final_labels]
        if labels and not final_labels and partial_labels:
            reasons.append("partial_only_outcomes_excluded_from_governance")
        elif labels and not eligible_labels:
            reasons.append("proxy_only_outcomes_excluded_from_governance")
        elif final_labels and partial_labels:
            reasons.append(
                "mixed_final_partial_outcomes_degraded:"
                + f"final:{len(final_labels)},partial:{len(partial_labels)}"
            )
        if labels and len(eligible_labels) < len(labels):
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

    def evaluate_autonomy_gate(
        self,
        *,
        readiness_report: OpportunityPersistedPromotionReadinessReport,
        gate_config: OpportunityAutonomyGateConfig | None = None,
    ) -> OpportunityAutonomyDecision:
        config = gate_config or OpportunityAutonomyGateConfig()
        degraded_reasons = tuple(readiness_report.degraded_reasons)
        matched = tuple(int(value) for value in readiness_report.matched_outcomes.values())
        paired_matched_outcomes = min(matched) if matched else 0
        quality_counts = self._extract_quality_counts(degraded_reasons)
        final_outcomes = max(quality_counts.get("final", 0), paired_matched_outcomes)
        partial_outcomes = sum(
            value
            for key, value in quality_counts.items()
            if key.startswith(_PARTIAL_LABEL_PREFIXES)
        )
        non_final_non_partial_outcomes = max(
            0,
            sum(quality_counts.values()) - quality_counts.get("final", 0) - partial_outcomes,
        )
        observed_outcomes = (
            final_outcomes + partial_outcomes + non_final_non_partial_outcomes
            if quality_counts
            else paired_matched_outcomes
        )
        has_partial_only = "partial_only_outcomes_excluded_from_governance" in degraded_reasons
        has_proxy_only = "proxy_only_outcomes_excluded_from_governance" in degraded_reasons
        mixed_final_partial = any(
            reason.startswith(_MIXED_FINAL_PARTIAL_PREFIX) for reason in degraded_reasons
        )
        promotion_passed = (
            readiness_report.promotion_report is not None
            and readiness_report.promotion_report.promotion_recommended
        )
        activation_ready = readiness_report.activation_readiness.activation_ready

        blocking_reasons = self._collect_blocking_reasons(
            degraded_reasons=degraded_reasons,
            config=config,
            has_proxy_only=has_proxy_only,
        )
        non_blocking_degradations = tuple(
            reason for reason in degraded_reasons if reason not in set(blocking_reasons)
        )
        evidence_summary: dict[str, int | bool | str] = {
            "paired_matched_outcomes": paired_matched_outcomes,
            "observed_outcomes": observed_outcomes,
            "final_outcomes": final_outcomes,
            "partial_outcomes": partial_outcomes,
            "non_final_non_partial_outcomes": non_final_non_partial_outcomes,
            "promotion_passed": promotion_passed,
            "activation_ready": activation_ready,
            "has_proxy_only_evidence": has_proxy_only,
            "has_partial_only_evidence": has_partial_only,
            "has_mixed_final_partial_evidence": mixed_final_partial,
        }

        if blocking_reasons:
            reasons = ("blocking_degradation_present", *blocking_reasons)
            return OpportunityAutonomyDecision(
                mode=OpportunityAutonomyMode.DENIED,
                primary_reason="blocking_degradation_present",
                reasons=reasons,
                blocking_reasons=blocking_reasons,
                warnings=degraded_reasons,
                evidence_summary=evidence_summary,
            )
        if observed_outcomes == 0:
            return OpportunityAutonomyDecision(
                mode=OpportunityAutonomyMode.DENIED,
                primary_reason="missing_outcome_evidence",
                reasons=("missing_outcome_evidence",),
                blocking_reasons=(),
                warnings=degraded_reasons,
                evidence_summary=evidence_summary,
            )
        if has_proxy_only:
            return OpportunityAutonomyDecision(
                mode=OpportunityAutonomyMode.DENIED,
                primary_reason="proxy_only_evidence",
                reasons=("proxy_only_evidence",),
                blocking_reasons=(),
                warnings=degraded_reasons,
                evidence_summary=evidence_summary,
            )

        live_ready = (
            paired_matched_outcomes >= config.min_total_matched_outcomes
            and final_outcomes >= config.min_final_outcomes_for_live_autonomy
            and (not config.require_promotion_pass_for_live or promotion_passed)
            and (not config.require_activation_ready_for_live or activation_ready)
        )
        if live_ready:
            if non_blocking_degradations:
                if (
                    len(non_blocking_degradations)
                    <= config.max_allowed_degraded_reasons_for_live_assisted
                ):
                    return OpportunityAutonomyDecision(
                        mode=OpportunityAutonomyMode.LIVE_ASSISTED,
                        primary_reason="live_allowed_with_degradations",
                        reasons=("live_allowed_with_degradations",),
                        blocking_reasons=(),
                        warnings=degraded_reasons,
                        evidence_summary=evidence_summary,
                    )
                return OpportunityAutonomyDecision(
                    mode=OpportunityAutonomyMode.PAPER_AUTONOMOUS,
                    primary_reason="live_downgraded_due_to_degradations",
                    reasons=("live_downgraded_due_to_degradations",),
                    blocking_reasons=(),
                    warnings=degraded_reasons,
                    evidence_summary=evidence_summary,
                )
            return OpportunityAutonomyDecision(
                mode=OpportunityAutonomyMode.LIVE_AUTONOMOUS,
                primary_reason="live_autonomy_gate_passed",
                reasons=("live_autonomy_gate_passed",),
                blocking_reasons=(),
                warnings=(),
                evidence_summary=evidence_summary,
            )

        allow_paper_with_partial_only = (
            has_partial_only and config.allow_partial_only_for_shadow_or_paper
        )
        paper_evidence_ready = observed_outcomes >= config.min_observed_outcomes_for_paper_autonomy
        if paper_evidence_ready and (not has_partial_only or allow_paper_with_partial_only):
            return OpportunityAutonomyDecision(
                mode=OpportunityAutonomyMode.PAPER_AUTONOMOUS,
                primary_reason="sufficient_evidence_for_paper_only",
                reasons=("sufficient_evidence_for_paper_only",),
                blocking_reasons=(),
                warnings=degraded_reasons,
                evidence_summary=evidence_summary,
            )

        return OpportunityAutonomyDecision(
            mode=OpportunityAutonomyMode.SHADOW_ONLY,
            primary_reason="insufficient_evidence_for_paper_or_live",
            reasons=("insufficient_evidence_for_paper_or_live",),
            blocking_reasons=(),
            warnings=degraded_reasons,
            evidence_summary=evidence_summary,
        )

    @staticmethod
    def _extract_quality_counts(degraded_reasons: Sequence[str]) -> dict[str, int]:
        for reason in degraded_reasons:
            if not reason.startswith(_NON_FINAL_EXCLUDED_PREFIX):
                continue
            counts_raw = reason.split(_NON_FINAL_EXCLUDED_PREFIX, 1)[1]
            counts: dict[str, int] = {}
            for item in counts_raw.split(","):
                key, _, value = item.partition(":")
                key = key.strip()
                if not key:
                    continue
                try:
                    counts[key] = int(value.strip())
                except ValueError:
                    continue
            return counts
        return {}

    @staticmethod
    def _collect_blocking_reasons(
        *,
        degraded_reasons: Sequence[str],
        config: OpportunityAutonomyGateConfig,
        has_proxy_only: bool,
    ) -> tuple[str, ...]:
        blocking_prefixes = tuple(config.blocking_degraded_reasons)
        allowlisted = set(config.allowlisted_degraded_reasons)
        blocked = [
            reason
            for reason in degraded_reasons
            if reason not in allowlisted
            and any(reason.startswith(prefix) for prefix in blocking_prefixes)
        ]
        if has_proxy_only and "proxy_only_outcomes_excluded_from_governance" not in blocked:
            blocked.append("proxy_only_outcomes_excluded_from_governance")
        return tuple(blocked)

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
    "OpportunityAutonomyDowngradeConfig",
    "OpportunityAutonomyDowngradeDecision",
    "OpportunityAutonomyDecision",
    "OpportunityAutonomyGateConfig",
    "OpportunityAutonomyMode",
    "OpportunityExecutionPermission",
    "evaluate_autonomy_downgrade",
    "evaluate_opportunity_execution_permission",
    "OpportunityActivationReadiness",
    "OpportunityLifecycleService",
    "OpportunityPersistedPromotionReadinessReport",
]
