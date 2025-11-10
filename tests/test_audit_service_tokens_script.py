from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


from scripts import audit_service_tokens as audit_service_tokens_script


def _write_core_yaml(path: Path, *, with_rbac: bool) -> Path:
    if with_rbac:
        content = """
        environments: {}
        risk_profiles: {}
        metrics_service:
          enabled: true
          rbac_tokens:
            - token_id: metrics-reader
              token_value: secret
              scopes: [metrics.read]
        risk_service:
          enabled: true
          rbac_tokens:
            - token_id: risk-reader
              token_value: secret
              scopes: [risk.read]
        """
    else:
        content = """
        environments: {}
        risk_profiles: {}
        metrics_service:
          enabled: true
          auth_token: static-token
        risk_service:
          enabled: true
          auth_token: static-token
        """
    path.write_text(content, encoding="utf-8")
    return path


def _stub_core_config(with_rbac: bool) -> SimpleNamespace:
    if with_rbac:
        metrics_tokens = (
            SimpleNamespace(token_id="metrics-reader", token_value="secret", token_env=None, token_hash=None, scopes=("metrics.read",)),
        )
        risk_tokens = (
            SimpleNamespace(token_id="risk-reader", token_value="secret", token_env=None, token_hash=None, scopes=("risk.read",)),
        )
        metrics = SimpleNamespace(enabled=True, auth_token=None, rbac_tokens=metrics_tokens)
        risk = SimpleNamespace(enabled=True, auth_token=None, rbac_tokens=risk_tokens)
    else:
        metrics = SimpleNamespace(enabled=True, auth_token="static-token", rbac_tokens=())
        risk = SimpleNamespace(enabled=True, auth_token="static-token", rbac_tokens=())
    return SimpleNamespace(metrics_service=metrics, risk_service=risk)


def test_audit_service_tokens_script_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _write_core_yaml(tmp_path / "core.yaml", with_rbac=True)
    stub = _stub_core_config(with_rbac=True)
    monkeypatch.setattr(audit_service_tokens_script, "load_core_config", lambda path: stub)

    output_path = tmp_path / "report.json"
    exit_code = audit_service_tokens_script.main(
        [
            "--config",
            str(config_path),
            "--json-output",
            str(output_path),
            "--pretty",
            "--fail-on-warning",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert not payload["errors"]
    assert not payload["warnings"]
    assert any(service["service"] == "metrics_service" for service in payload["services"])


def test_audit_service_tokens_script_warns_on_shared_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = _write_core_yaml(tmp_path / "core.yaml", with_rbac=False)
    stub = _stub_core_config(with_rbac=False)
    monkeypatch.setattr(audit_service_tokens_script, "load_core_config", lambda path: stub)

    monkeypatch.setenv("BOT_CORE_TOKEN_AUDIT_CONFIG", str(config_path))
    monkeypatch.setenv("BOT_CORE_TOKEN_AUDIT_PRINT", "true")
    monkeypatch.setenv("BOT_CORE_TOKEN_AUDIT_PRETTY", "true")
    monkeypatch.setenv("BOT_CORE_TOKEN_AUDIT_FAIL_ON_WARNING", "true")

    exit_code = audit_service_tokens_script.main([])

    assert exit_code == 1
    stdout = capsys.readouterr().out
    assert "statycznego" in stdout.lower()
