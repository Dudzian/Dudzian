from __future__ import annotations

from bot_core.config.models import (
    CoreConfig,
    MetricsServiceConfig,
    RiskServiceConfig,
    ServiceTokenConfig,
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


def test_audit_warns_when_env_missing_and_legacy_only():
    token_config = ServiceTokenConfig(
        token_id="metrics-env",
        token_env="MISSING_TOKEN",
        scopes=("metrics.read",),
    )
    core_config = _base_core_config(
        metrics_service=MetricsServiceConfig(
            enabled=True,
            auth_token="legacy",
            rbac_tokens=(token_config,),
        ),
        risk_service=RiskServiceConfig(
            enabled=True,
            auth_token="legacy",
            rbac_tokens=(),
        ),
    )

    report = audit_service_tokens(core_config, env={})
    payload = report.as_dict()

    assert payload["warnings"]
    metrics = next(service for service in payload["services"] if service["service"] == "metrics_service")
    assert any("legacy" in finding["message"] for finding in metrics["findings"])
    assert any("nie jest ustawiona" in finding["message"] for finding in metrics["findings"])
    risk = next(service for service in payload["services"] if service["service"] == "risk_service")
    assert any(finding["level"] == "warning" for finding in risk["findings"])


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
