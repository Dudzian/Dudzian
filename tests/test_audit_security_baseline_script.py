import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import audit_security_baseline as audit_security_baseline_script
from tests.test_audit_tls_assets_script import _CERT, _KEY


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _stub_config_error() -> SimpleNamespace:
    metrics_tls = SimpleNamespace(
        enabled=False,
        certificate_path=None,
        private_key_path=None,
        client_ca_path=None,
        require_client_auth=False,
        private_key_password_env=None,
        pinned_fingerprints=(),
    )
    metrics_service = SimpleNamespace(
        enabled=True,
        auth_token="",
        tls=metrics_tls,
        rbac_tokens=(),
    )
    risk_service = SimpleNamespace(
        enabled=False,
        auth_token=None,
        tls=SimpleNamespace(
            enabled=False,
            certificate_path=None,
            private_key_path=None,
            client_ca_path=None,
            require_client_auth=False,
            private_key_password_env=None,
            pinned_fingerprints=(),
        ),
        rbac_tokens=(),
    )
    scheduler = SimpleNamespace(
        name="core_multi",
        rbac_tokens=(),
    )
    return SimpleNamespace(
        metrics_service=metrics_service,
        risk_service=risk_service,
        multi_strategy_schedulers={"core_multi": scheduler},
    )


def _stub_config_secure(cert: Path, key: Path) -> SimpleNamespace:
    def tls_config() -> SimpleNamespace:
        return SimpleNamespace(
            enabled=True,
            certificate_path=str(cert),
            private_key_path=str(key),
            client_ca_path=None,
            require_client_auth=False,
            private_key_password_env=None,
            pinned_fingerprints=(),
        )

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
        tls=tls_config(),
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
        tls=tls_config(),
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
    return SimpleNamespace(
        metrics_service=metrics_service,
        risk_service=risk_service,
        multi_strategy_schedulers={"core_multi": scheduler},
    )


def test_audit_security_baseline_script_detects_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text("runtime: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        audit_security_baseline_script,
        "load_core_config",
        lambda path: _stub_config_error(),
    )

    output_path = tmp_path / "report.json"
    exit_code = audit_security_baseline_script.main(
        [
            "--config",
            str(config_path),
            "--json-output",
            str(output_path),
            "--pretty",
            "--fail-on-warning",
            "--fail-on-error",
        ]
    )

    assert exit_code == 2
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["errors"]


def test_audit_security_baseline_script_env_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cert_path = _write(tmp_path / "cert.pem", _CERT)
    key_path = _write(tmp_path / "key.pem", _KEY)
    os.chmod(cert_path, 0o600)
    os.chmod(key_path, 0o600)
    config_path = tmp_path / "core.yaml"
    config_path.write_text("runtime: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        audit_security_baseline_script,
        "load_core_config",
        lambda path: _stub_config_secure(cert_path, key_path),
    )

    output_path = tmp_path / "baseline.json"
    monkeypatch.setenv("BOT_CORE_SECURITY_BASELINE_CONFIG", str(config_path))
    monkeypatch.setenv("BOT_CORE_SECURITY_BASELINE_JSON_OUTPUT", str(output_path))
    monkeypatch.setenv("BOT_CORE_SECURITY_BASELINE_PRINT", "true")
    monkeypatch.setenv("BOT_CORE_SECURITY_BASELINE_PRETTY", "true")
    monkeypatch.setenv("BOT_CORE_SECURITY_BASELINE_METRICS_SCOPES", "metrics.read")
    monkeypatch.setenv("BOT_CORE_SECURITY_BASELINE_RISK_SCOPES", "risk.read")
    monkeypatch.setenv(
        "BOT_CORE_SECURITY_BASELINE_SCHEDULER_SCOPES",
        "core_multi:runtime.schedule.write,runtime.schedule.read",
    )

    exit_code = audit_security_baseline_script.main([])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert '"status": "ok"' in stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert not payload["warnings"]
    assert not payload["errors"]


def test_audit_security_baseline_script_generates_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cert_path = _write(tmp_path / "cert.pem", _CERT)
    key_path = _write(tmp_path / "key.pem", _KEY)
    os.chmod(cert_path, 0o600)
    os.chmod(key_path, 0o600)

    config_path = tmp_path / "core.yaml"
    config_path.write_text("runtime: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        audit_security_baseline_script,
        "load_core_config",
        lambda path: _stub_config_secure(cert_path, key_path),
    )

    signing_key_file = tmp_path / "hmac.key"
    signing_key_file.write_text("super-secret", encoding="utf-8")

    output_path = tmp_path / "report.json"
    exit_code = audit_security_baseline_script.main(
        [
            "--config",
            str(config_path),
            "--json-output",
            str(output_path),
            "--summary-hmac-key-file",
            str(signing_key_file),
            "--summary-hmac-key-id",
            "baseline-key",
            "--require-summary-signature",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    signature = payload.get("summary_signature")
    assert isinstance(signature, dict)
    assert signature.get("value")
    assert signature.get("algorithm") == "HMAC-SHA256"
    assert signature.get("key_id") == "baseline-key"


def test_audit_security_baseline_script_requires_signature_without_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cert_path = _write(tmp_path / "cert.pem", _CERT)
    key_path = _write(tmp_path / "key.pem", _KEY)
    os.chmod(cert_path, 0o600)
    os.chmod(key_path, 0o600)

    config_path = tmp_path / "core.yaml"
    config_path.write_text("runtime: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        audit_security_baseline_script,
        "load_core_config",
        lambda path: _stub_config_secure(cert_path, key_path),
    )

    with pytest.raises(SystemExit) as excinfo:
        audit_security_baseline_script.main(
            ["--config", str(config_path), "--require-summary-signature"]
        )

    assert excinfo.value.code == 2

