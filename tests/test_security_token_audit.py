from __future__ import annotations

from pathlib import Path

from bot_core.config.models import (
    CoreConfig,
    MetricsServiceConfig,
    MultiStrategySchedulerConfig,
    RiskServiceConfig,
    ServiceTokenConfig,
    StrategyScheduleConfig,
)
from bot_core.security.token_audit import audit_service_tokens


def _base_core_config(**overrides):
    base = dict(
        environments={},
        risk_profiles={},
        instrument_universes={},
        strategies={},
    )
    base.update(overrides)
    return CoreConfig(**base)


def test_audit_service_tokens_success_with_rbac():
    core_config = _base_core_config(
        metrics_service=MetricsServiceConfig(
            enabled=True,
            rbac_tokens=(
                ServiceTokenConfig(
                    token_id="metrics-reader",
                    token_value="secret",
                    scopes=("metrics.read",),
                ),
            ),
        ),
        risk_service=RiskServiceConfig(
            enabled=True,
            rbac_tokens=(
                ServiceTokenConfig(
                    token_id="risk-reader",
                    token_value="secret",
                    scopes=("risk.read",),
                ),
            ),
        ),
    )

    report = audit_service_tokens(core_config)
    payload = report.as_dict()

    assert not payload["errors"]
    assert not payload["warnings"]
    metrics = next(service for service in payload["services"] if service["service"] == "metrics_service")
    assert metrics["coverage"]["metrics.read"] == ["metrics-reader"]
    risk = next(service for service in payload["services"] if service["service"] == "risk_service")
    assert risk["coverage"]["risk.read"] == ["risk-reader"]


def test_audit_warns_when_env_missing_and_shared_secret_only():
    token_config = ServiceTokenConfig(
        token_id="metrics-env",
        token_env="MISSING_TOKEN",
        scopes=("metrics.read",),
    )
    core_config = _base_core_config(
        metrics_service=MetricsServiceConfig(
            enabled=True,
            auth_token="static",
            rbac_tokens=(token_config,),
        ),
        risk_service=RiskServiceConfig(
            enabled=True,
            auth_token="static",
            rbac_tokens=(),
        ),
    )

    report = audit_service_tokens(core_config, env={})
    payload = report.as_dict()

    assert payload["warnings"]
    metrics = next(service for service in payload["services"] if service["service"] == "metrics_service")
    assert any("statycznego" in finding["message"] for finding in metrics["findings"])
    assert any("nie jest ustawiona" in finding["message"] for finding in metrics["findings"])
    risk = next(service for service in payload["services"] if service["service"] == "risk_service")
    assert any(finding["level"] == "warning" for finding in risk["findings"])


def test_audit_reports_missing_metrics_auth_token_env() -> None:
    core_config = _base_core_config(
        metrics_service=MetricsServiceConfig(
            enabled=True,
            auth_token_env="METRICS_SERVICE_AUTH_TOKEN",
            rbac_tokens=(),
        ),
        risk_service=RiskServiceConfig(enabled=False),
    )

    report = audit_service_tokens(core_config, env={})
    payload = report.as_dict()

    metrics = next(service for service in payload["services"] if service["service"] == "metrics_service")
    assert metrics["configured"] is False
    assert any(
        finding["level"] == "error"
        and "auth_token_env" in (finding.get("message") or "")
        for finding in metrics["findings"]
    )


def test_audit_accepts_metrics_auth_token_env_present() -> None:
    core_config = _base_core_config(
        metrics_service=MetricsServiceConfig(
            enabled=True,
            auth_token_env="METRICS_SERVICE_AUTH_TOKEN",
            rbac_tokens=(),
        ),
        risk_service=RiskServiceConfig(enabled=False),
    )

    report = audit_service_tokens(
        core_config,
        env={"METRICS_SERVICE_AUTH_TOKEN": "secret"},
    )
    payload = report.as_dict()

    metrics = next(service for service in payload["services"] if service["service"] == "metrics_service")
    assert metrics["configured"] is True
    assert not any(
        "auth_token_env" in (finding.get("message") or "") for finding in metrics["findings"]
    )


def test_audit_reports_over_permissive_auth_token_file(tmp_path: Path) -> None:
    token_file = tmp_path / "metrics.token"
    token_file.write_text("secret", encoding="utf-8")
    token_file.chmod(0o644)

    core_config = _base_core_config(
        metrics_service=MetricsServiceConfig(
            enabled=True,
            auth_token_file=str(token_file),
            rbac_tokens=(),
        ),
        risk_service=RiskServiceConfig(enabled=False),
    )

    report = audit_service_tokens(core_config, env={})
    payload = report.as_dict()

    metrics = next(service for service in payload["services"] if service["service"] == "metrics_service")
    assert metrics["configured"] is True
    assert any(
        finding["level"] == "warning"
        and "zbyt szerokie uprawnienia" in (finding.get("message") or "")
        for finding in metrics["findings"]
    )
    details = next(
        finding["details"]
        for finding in metrics["findings"]
        if "zbyt szerokie uprawnienia" in (finding.get("message") or "")
    )
    assert details
    assert details.get("mode") == "0o644"
    assert details.get("expected_max_mode") == "0o600"


def test_audit_reports_error_when_scope_missing():
    core_config = _base_core_config(
        metrics_service=MetricsServiceConfig(
            enabled=True,
            rbac_tokens=(
                ServiceTokenConfig(
                    token_id="metrics-writer",
                    token_value="secret",
                    scopes=("metrics.write",),
                ),
            ),
        ),
        risk_service=RiskServiceConfig(
            enabled=True,
            rbac_tokens=(),
        ),
    )

    report = audit_service_tokens(core_config)
    payload = report.as_dict()

    assert payload["errors"]
    metrics = next(service for service in payload["services"] if service["service"] == "metrics_service")
    assert any(
        finding["level"] == "error" and finding["details"]["scope"] == "metrics.read"
        for finding in metrics["findings"]
    )
    risk = next(service for service in payload["services"] if service["service"] == "risk_service")
    assert any(finding["level"] == "error" for finding in risk["findings"])


def test_audit_scheduler_tokens_and_overrides():
    scheduler = MultiStrategySchedulerConfig(
        name="core_multi",
        telemetry_namespace="runtime.multi.core",
        schedules=(
            StrategyScheduleConfig(
                name="mean_reversion_intraday",
                strategy="core_mean_reversion",
                cadence_seconds=60,
                max_drift_seconds=10,
                warmup_bars=20,
                risk_profile="balanced",
            ),
        ),
        rbac_tokens=(
            ServiceTokenConfig(
                token_id="scheduler-writer",
                token_value="secret",
                scopes=("runtime.schedule.write", "runtime.schedule.read"),
            ),
        ),
    )
    core_config = _base_core_config(
        metrics_service=MetricsServiceConfig(
            enabled=True,
            rbac_tokens=(
                ServiceTokenConfig(
                    token_id="metrics-reader",
                    token_value="secret",
                    scopes=("metrics.read",),
                ),
            ),
        ),
        risk_service=RiskServiceConfig(
            enabled=True,
            rbac_tokens=(
                ServiceTokenConfig(
                    token_id="risk-reader",
                    token_value="secret",
                    scopes=("risk.read",),
                ),
            ),
        ),
        multi_strategy_schedulers={"core_multi": scheduler},
    )

    report = audit_service_tokens(
        core_config,
        scheduler_required_scopes={
            "*": ("runtime.schedule.read", "runtime.schedule.write"),
            "core_multi": ("runtime.schedule.write",),
        },
    )
    payload = report.as_dict()

    scheduler_report = next(
        service
        for service in payload["services"]
        if service["service"] == "multi_strategy_scheduler:core_multi"
    )
    assert all(finding["level"] != "error" for finding in scheduler_report["findings"])
    assert "runtime.schedule.write" in scheduler_report["coverage"]
    assert scheduler_report["coverage"]["runtime.schedule.write"] == ["scheduler-writer"]


def test_audit_scheduler_reports_missing_tokens():
    scheduler = MultiStrategySchedulerConfig(
        name="paper_scheduler",
        telemetry_namespace="runtime.multi.paper",
        schedules=(
            StrategyScheduleConfig(
                name="vol_target_daily",
                strategy="core_volatility_target",
                cadence_seconds=300,
                max_drift_seconds=30,
                warmup_bars=15,
                risk_profile="conservative",
            ),
        ),
        rbac_tokens=(),
    )
    core_config = _base_core_config(
        metrics_service=MetricsServiceConfig(enabled=False),
        risk_service=RiskServiceConfig(enabled=False),
        multi_strategy_schedulers={"paper_scheduler": scheduler},
    )

    report = audit_service_tokens(core_config)
    payload = report.as_dict()

    scheduler_report = next(
        service
        for service in payload["services"]
        if service["service"] == "multi_strategy_scheduler:paper_scheduler"
    )
    assert any(finding["level"] == "error" for finding in scheduler_report["findings"])
