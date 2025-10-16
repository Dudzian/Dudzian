from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bot_core.observability.hypercare import ObservabilityCycleConfig, ObservabilityCycleResult, SLOOutputConfig
from bot_core.portfolio import PortfolioDecision
from bot_core.portfolio.hypercare import (
    PortfolioCycleConfig,
    PortfolioCycleInputs,
    PortfolioCycleOutputConfig,
    PortfolioCycleResult,
)
from bot_core.resilience.hypercare import AuditConfig, BundleConfig, FailoverConfig, ResilienceCycleConfig
from bot_core.runtime.stage6_hypercare import Stage6HypercareConfig, Stage6HypercareCycle
from bot_core.security.signing import verify_hmac_signature


class _StubObservabilityCycle:
    def __init__(self, config: ObservabilityCycleConfig) -> None:
        self.config = config

    def run(self) -> ObservabilityCycleResult:
        base = self.config.slo.json_path.parent
        base.mkdir(parents=True, exist_ok=True)
        slo_report = base / "slo_report.json"
        slo_report.write_text("{}", encoding="utf-8")
        signature = base / "slo_report.sig"
        signature.write_text("{}", encoding="utf-8")
        overrides = base / "overrides.json"
        overrides.write_text("{}", encoding="utf-8")
        return ObservabilityCycleResult(
            slo_report_path=slo_report,
            slo_signature_path=signature,
            slo_csv_path=None,
            overrides_path=overrides,
            overrides_signature_path=None,
            dashboard_annotations_path=None,
            dashboard_signature_path=None,
            bundle_path=None,
            bundle_manifest_path=None,
            bundle_signature_path=None,
            bundle_verification=None,
        )


def test_stage6_hypercare_builds_signed_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    config = Stage6HypercareConfig(
        output_path=summary_path,
        signing_key=b"stage6-key",
        signing_key_id="stage6",
        metadata={"run_id": "abc123"},
        observability=ObservabilityCycleConfig(
            definitions_path=tmp_path / "defs.yaml",
            metrics_path=tmp_path / "metrics.json",
            slo=SLOOutputConfig(json_path=tmp_path / "observability" / "slo.json"),
        ),
    )

    cycle = Stage6HypercareCycle(
        config,
        observability_factory=lambda cfg: _StubObservabilityCycle(cfg),
    )

    result = cycle.run()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "ok"
    assert payload["metadata"]["run_id"] == "abc123"
    assert payload["components"]["observability"]["status"] == "ok"
    assert result.observability is not None

    signature_path = summary_path.with_suffix(summary_path.suffix + ".sig")
    signature = json.loads(signature_path.read_text(encoding="utf-8"))
    assert verify_hmac_signature(payload, signature, key=b"stage6-key")


class _StubPortfolioCycle:
    def __init__(self, governor: object, config: PortfolioCycleConfig) -> None:
        self.governor = governor
        self.config = config

    def run(self) -> PortfolioCycleResult:
        decision = PortfolioDecision(
            timestamp=datetime.now(timezone.utc),
            portfolio_id="demo",
            portfolio_value=100_000.0,
            adjustments=(),
            advisories=(),
            rebalance_required=True,
        )
        summary = self.config.output.summary_path
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text("{}", encoding="utf-8")
        return PortfolioCycleResult(
            decision=decision,
            summary_path=summary,
            signature_path=None,
            csv_path=None,
            market_intel_metadata={},
            slo_statuses={},
            stress_overrides=(),
        )


def test_stage6_hypercare_collects_component_failures(tmp_path: Path) -> None:
    observability = None
    resilience = ResilienceCycleConfig(
        bundle=BundleConfig(
            source=tmp_path / "bundle_src",
            output_dir=tmp_path / "bundle_out",
        ),
        audit=AuditConfig(json_path=tmp_path / "audit.json"),
        failover=FailoverConfig(
            plan_path=tmp_path / "plan.yaml",
            json_path=tmp_path / "failover.json",
        ),
    )
    portfolio_inputs = PortfolioCycleInputs(
        allocations_path=tmp_path / "allocations.json",
        market_intel_path=tmp_path / "intel.json",
        portfolio_value=10_000.0,
    )
    portfolio_output = PortfolioCycleOutputConfig(summary_path=tmp_path / "portfolio.json")
    portfolio = PortfolioCycleConfig(inputs=portfolio_inputs, output=portfolio_output)

    summary_path = tmp_path / "stage6_summary.json"
    config = Stage6HypercareConfig(
        output_path=summary_path,
        observability=observability,
        resilience=resilience,
        portfolio=portfolio,
    )

    def failing_resilience_factory(_: ResilienceCycleConfig) -> ResilienceCycleConfig:  # type: ignore[return-value]
        raise RuntimeError("audit failure")

    cycle = Stage6HypercareCycle(
        config,
        portfolio_governor=object(),
        resilience_factory=failing_resilience_factory,
        portfolio_factory=lambda gov, cfg: _StubPortfolioCycle(gov, cfg),
    )

    result = cycle.run()
    payload = result.payload

    assert payload["overall_status"] == "fail"
    assert payload["components"]["resilience"]["status"] == "fail"
    assert any("audit failure" in issue for issue in payload["issues"])
    assert payload["components"]["portfolio"]["status"] == "warn"
    assert any("rebalance" in warning.lower() for warning in payload["warnings"])
