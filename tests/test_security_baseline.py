import os
from pathlib import Path
from types import SimpleNamespace

import os
import pytest


from bot_core.security.baseline import generate_security_baseline_report
from tests.test_audit_tls_assets_script import _CERT, _KEY


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _tls_config(
    *,
    enabled: bool,
    cert: Path | None = None,
    key: Path | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=enabled,
        certificate_path=str(cert) if enabled and cert else None,
        private_key_path=str(key) if enabled and key else None,
        client_ca_path=None,
        require_client_auth=False,
        private_key_password_env=None,
        pinned_fingerprints=(),
    )


def test_security_baseline_reports_errors_when_tokens_missing() -> None:
    metrics_tls = _tls_config(enabled=False)
    metrics_service = SimpleNamespace(
        enabled=True,
        auth_token="",
        tls=metrics_tls,
        rbac_tokens=(),
    )
    risk_service = SimpleNamespace(
        enabled=False,
        auth_token=None,
        tls=_tls_config(enabled=False),
        rbac_tokens=(),
    )
    scheduler = SimpleNamespace(
        name="core_multi",
        rbac_tokens=(),
    )
    core_config = SimpleNamespace(
        metrics_service=metrics_service,
        risk_service=risk_service,
        multi_strategy_schedulers={"core_multi": scheduler},
    )

    report = generate_security_baseline_report(core_config, env={})

    assert report.status == "error"
    assert report.errors
    assert any("RBAC" in message or "token" in message.lower() for message in report.errors)


def test_security_baseline_reports_ok_for_hardened_config(tmp_path: Path) -> None:
    cert_path = _write(tmp_path / "cert.pem", _CERT)
    key_path = _write(tmp_path / "key.pem", _KEY)
    os.chmod(cert_path, 0o600)
    os.chmod(key_path, 0o600)

    metrics_service = SimpleNamespace(
        enabled=True,
        auth_token=None,
        rbac_tokens=(
            SimpleNamespace(
                token_id="metrics-reader",
                token_value="secret",
                token_env=None,
                token_hash=None,
                scopes=("metrics.read",),
            ),
        ),
        tls=_tls_config(enabled=True, cert=cert_path, key=key_path),
    )
    risk_service = SimpleNamespace(
        enabled=True,
        auth_token=None,
        rbac_tokens=(
            SimpleNamespace(
                token_id="risk-reader",
                token_value="secret",
                token_env=None,
                token_hash=None,
                scopes=("risk.read",),
            ),
        ),
        tls=_tls_config(enabled=True, cert=cert_path, key=key_path),
    )
    scheduler = SimpleNamespace(
        name="core_multi",
        rbac_tokens=(
            SimpleNamespace(
                token_id="scheduler-writer",
                token_value="secret",
                token_env=None,
                token_hash=None,
                scopes=("runtime.schedule.read", "runtime.schedule.write"),
            ),
        ),
    )
    core_config = SimpleNamespace(
        metrics_service=metrics_service,
        risk_service=risk_service,
        multi_strategy_schedulers={"core_multi": scheduler},
    )

    report = generate_security_baseline_report(
        core_config,
        env={},
        scheduler_required_scopes={"*": ("runtime.schedule.read", "runtime.schedule.write")},
    )

    assert report.status == "ok"
    assert not report.errors
    assert not report.warnings

