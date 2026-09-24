import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot_core.runtime import file_metadata
from scripts import audit_tls_assets as audit_tls_assets_script


class _WindowsOsProxy:
    def __init__(self, real_os: object) -> None:
        self._real_os = real_os
        self.name = "nt"

    def __getattr__(self, item: str):
        return getattr(self._real_os, item)


def _healthy_certificate_pair() -> tuple[str, str]:
    """Create TLS material whose health does not depend on a checked-in expiry date."""
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    now = datetime.now(timezone.utc)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test.local")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=400))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_pem = certificate.public_bytes(serialization.Encoding.PEM).decode("ascii")
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("ascii")
    return cert_pem, key_pem


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _stub_config(cert: Path, key: Path, *, with_warning: bool) -> SimpleNamespace:
    metrics_tls = SimpleNamespace(
        enabled=True,
        certificate_path=str(cert),
        private_key_path=str(key),
        client_ca_path=None,
        require_client_auth=False,
        private_key_password_env=None,
        pinned_fingerprints=("sha256:deadbeef",) if with_warning else (),
    )
    risk_tls = SimpleNamespace(
        enabled=False,
        certificate_path=None,
        private_key_path=None,
        client_ca_path=None,
        require_client_auth=False,
        private_key_password_env=None,
        pinned_fingerprints=(),
    )
    metrics_service = SimpleNamespace(
        enabled=True, auth_token="" if with_warning else "secret", tls=metrics_tls
    )
    risk_service = SimpleNamespace(enabled=False, auth_token="token", tls=risk_tls)
    return SimpleNamespace(metrics_service=metrics_service, risk_service=risk_service)


def test_audit_tls_assets_script_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cert_pem, key_pem = _healthy_certificate_pair()
    cert_path = _write(tmp_path / "cert.pem", cert_pem)
    key_path = _write(tmp_path / "key.pem", key_pem)
    config_path = tmp_path / "core.yaml"
    config_path.write_text("runtime: {}\n", encoding="utf-8")

    stub = _stub_config(cert_path, key_path, with_warning=True)
    monkeypatch.setattr(audit_tls_assets_script, "load_core_config", lambda path: stub)

    output_path = tmp_path / "report.json"
    exit_code = audit_tls_assets_script.main(
        [
            "--config",
            str(config_path),
            "--json-output",
            str(output_path),
            "--pretty",
            "--fail-on-warning",
        ]
    )

    assert exit_code == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "metrics_service" in payload["services"]
    assert payload["errors"]


def test_audit_tls_assets_script_env_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cert_pem, key_pem = _healthy_certificate_pair()
    cert_path = _write(tmp_path / "cert.pem", cert_pem)
    key_path = _write(tmp_path / "key.pem", key_pem)
    config_path = tmp_path / "core.yaml"
    config_path.write_text("runtime: {}\n", encoding="utf-8")

    stub = _stub_config(cert_path, key_path, with_warning=False)
    monkeypatch.setattr(audit_tls_assets_script, "load_core_config", lambda path: stub)

    output_path = tmp_path / "report.json"
    monkeypatch.setenv("BOT_CORE_TLS_AUDIT_CONFIG", str(config_path))
    monkeypatch.setenv("BOT_CORE_TLS_AUDIT_JSON_OUTPUT", str(output_path))
    monkeypatch.setenv("BOT_CORE_TLS_AUDIT_PRINT", "true")
    monkeypatch.setenv("BOT_CORE_TLS_AUDIT_PRETTY", "true")

    exit_code = audit_tls_assets_script.main([])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "metrics_service" in stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert not payload["errors"]


def test_audit_tls_assets_accepts_env_auth_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cert_pem, key_pem = _healthy_certificate_pair()
    cert_path = _write(tmp_path / "cert.pem", cert_pem)
    key_path = _write(tmp_path / "key.pem", key_pem)
    os.chmod(cert_path, 0o600)
    os.chmod(key_path, 0o600)

    config_path = tmp_path / "core.yaml"
    config_path.write_text("runtime: {}\n", encoding="utf-8")

    metrics_tls = SimpleNamespace(
        enabled=True,
        certificate_path=str(cert_path),
        private_key_path=str(key_path),
        client_ca_path=None,
        require_client_auth=False,
        private_key_password_env=None,
        pinned_fingerprints=(),
    )

    metrics_service = SimpleNamespace(
        enabled=True,
        auth_token=None,
        auth_token_env="METRICS_AUTH_TOKEN",
        auth_token_file=None,
        rbac_tokens=(),
        tls=metrics_tls,
    )
    risk_service = SimpleNamespace(enabled=False, tls=SimpleNamespace(enabled=False))

    monkeypatch.setattr(
        audit_tls_assets_script,
        "load_core_config",
        lambda path: SimpleNamespace(metrics_service=metrics_service, risk_service=risk_service),
    )

    monkeypatch.setenv("METRICS_AUTH_TOKEN", "from-env")
    output_path = tmp_path / "report.json"

    exit_code = audit_tls_assets_script.main(
        [
            "--config",
            str(config_path),
            "--json-output",
            str(output_path),
            "--fail-on-warning",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    metrics = payload["services"]["metrics_service"]
    assert metrics["auth_token_configured"] is True
    assert not metrics["warnings"]


def test_audit_tls_assets_windows_skips_permission_warnings_but_keeps_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit_tls_assets_script, "os", _WindowsOsProxy(os))
    monkeypatch.setattr(file_metadata, "os", _WindowsOsProxy(os))
    os.chmod(tmp_path, 0o777)

    cert_pem, key_pem = _healthy_certificate_pair()
    cert_path = _write(tmp_path / "cert.pem", cert_pem)
    key_path = _write(tmp_path / "key.pem", key_pem)
    os.chmod(cert_path, 0o600)
    os.chmod(key_path, 0o600)

    config_path = tmp_path / "core.yaml"
    config_path.write_text("runtime: {}\n", encoding="utf-8")

    metrics_tls = SimpleNamespace(
        enabled=True,
        certificate_path=str(cert_path),
        private_key_path=str(key_path),
        client_ca_path=None,
        require_client_auth=False,
        private_key_password_env=None,
        pinned_fingerprints=(),
    )

    metrics_service = SimpleNamespace(
        enabled=True,
        auth_token=None,
        auth_token_env=None,
        auth_token_file=None,
        rbac_tokens=(),
        tls=metrics_tls,
    )
    risk_service = SimpleNamespace(enabled=False, tls=SimpleNamespace(enabled=False))

    monkeypatch.setattr(
        audit_tls_assets_script,
        "load_core_config",
        lambda path: SimpleNamespace(metrics_service=metrics_service, risk_service=risk_service),
    )

    output_path = tmp_path / "report.json"

    exit_code = audit_tls_assets_script.main(
        [
            "--config",
            str(config_path),
            "--json-output",
            str(output_path),
            "--fail-on-warning",
        ]
    )

    assert exit_code == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    metrics = payload["services"]["metrics_service"]
    assert metrics["auth_token_configured"] is False
    assert metrics["warnings"] == ["MetricsService ma włączone API bez tokenu autoryzacyjnego"]
    assert metrics["tls"]["warnings"] == []
