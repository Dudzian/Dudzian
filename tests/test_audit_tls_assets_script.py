import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


from scripts import audit_tls_assets as audit_tls_assets_script


_CERT = """-----BEGIN CERTIFICATE-----
MIIDCzCCAfOgAwIBAgIUD1ssHMmVtzTxIQqrLaC/gQ5pvWkwDQYJKoZIhvcNAQEL
BQAwFTETMBEGA1UEAwwKdGVzdC5sb2NhbDAeFw0yNTEwMTIxNDE0NDVaFw0yNjEw
MTIxNDE0NDVaMBUxEzARBgNVBAMMCnRlc3QubG9jYWwwggEiMA0GCSqGSIb3DQEB
AQUAA4IBDwAwggEKAoIBAQDs3HnkDaloIH5Ifr3wA3xYYYWJdUVQ7hiDho8pO1Q2
kIwGVThRRAF+J/nLC6K3O7b66mjfNiivjnWKMrHF7+Rq3HkoqD2RGDa5rJG5Nd5r
Udj8MJhMI8t7TwSc1wK6kea4+aa/PeGBSpK3Aqtu2+J6nwYMOjtuKZu8lC5IH3KU
ywclIaobw8C9+LgZW2dXHD2vbexIUkBGPKhXXC0CTxE/owFgQ/GimdJ5g28yiPCN
5NqA5zvFF7wji7zRE5gKwKOx7lYr+wO0ed2mOJs16OmW21g2ztSecxw5m8+7YiSY
7R/w3kHbxJrkmi9Jcs+fl3aE2jWMHIi+MwhvtzMBIa2PAgMBAAGjUzBRMB0GA1Ud
DgQWBBQcALwC+lC4jeEneWiSeQVlL31SmDAfBgNVHSMEGDAWgBQcALwC+lC4jeEn
eWiSeQVlL31SmDAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBCwUAA4IBAQDT
VsP9bnAY/rPo7N1DGLHeJKJbITw8BH1TztqyysxIy0X4qWsVJJysUflaaLNXMSXM
X5ALREOUFjpwg/1YGX6MRYU5YgwYIMrQ4fhDhlyf3o7mcfcKBekS4WVrAhTpVdAP
FlVU80guFlDYmBymaurjTsFmdK+5PJiJGuE9qJUETL4tp1C/s9EGBWncpFO9a3zf
q/3eaVtHsI9dR2U9NYtCfwnTVMqAjyDL4B+Sgti509+fdCTpwynoMgAZVBRnnz24
FX6px8Myi0dPLn+ofeJZBE2EzuvYSUlJmWuu0WJFiz2WaRpPzvnRure/Yelu9CJT
kCpwO4mSUi3mqgeFcKKB
-----END CERTIFICATE-----
"""

_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDs3HnkDaloIH5I
fr3wA3xYYYWJdUVQ7hiDho8pO1Q2kIwGVThRRAF+J/nLC6K3O7b66mjfNiivjnWK
MrHF7+Rq3HkoqD2RGDa5rJG5Nd5rUdj8MJhMI8t7TwSc1wK6kea4+aa/PeGBSpK3
Aqtu2+J6nwYMOjtuKZu8lC5IH3KUywclIaobw8C9+LgZW2dXHD2vbexIUkBGPKhX
XC0CTxE/owFgQ/GimdJ5g28yiPCN5NqA5zvFF7wji7zRE5gKwKOx7lYr+wO0ed2m
OJs16OmW21g2ztSecxw5m8+7YiSY7R/w3kHbxJrkmi9Jcs+fl3aE2jWMHIi+Mwhv
tzMBIa2PAgMBAAECggEAF12UhKO4X3Y9Huep0wB1Br7wDmRMJzSlpGvkuXuJziwq
NAG6IYIk544H/Tizn4G7hjsTh1lvYAocnDpuAQmuLcB8Dz+xexu5Yk2cvnmK7GlR
j2c3zuMFEq/z04j+UutLqFmwUlNHaJqzqGwR/0ifqdsAHLqt3Cssmsi/XAwGIJlK
DsXYd0O2NuORzoMEcHaHDshupwZd9MqclrcRmL4WA1onNYFd5cbzwKb3/5HbbxJ9
YfShqr8wio5JGhDvXtOdPaAAm13s7Hjp7FuPtWWi/hz49nBjCLbS4ds8dkCVk4fj
9+ja7ahBvfPNTN2Av6DtLbC/LonB0iL3FFCk4MqJeQKBgQD6zkE7MNJKXOnXAAAt
dXfrIyhy7nCEMiwRFruZAW0ipRK6r4XU9T7XyINBG3KpJNgsWBpnCcumZ2iiqt/o
vUVzLn5Tt1DbxYZqLMkWQOMhEVsauTHgDGkFuo/WPAtApUROUagyJfbw3pUlKxcY
BEa8STs2uH8zn378rgs4b4/CjQKBgQDxxEoSIrrCQ69hoWHC/VyXFFmIa3ZivUiR
Lli8Pgi/YsN+VsDoUrjOHrKfKQfTWhWmqGI/oXwvUz/S+l2aAiqvN473RkZHW7uQ
F8Sm7weW4C5tlfOyCsdhqmbWylX8iZkUJ3lsd7TmUp61bpdd6ZLS6uJRGYHWXxkN
K8PthhH3iwKBgQCt+OxetproYlMChEmbPuSUAqtILgV2bacLo401sTuW0JKRVLes
5QFWhQwm2XdLxPb+q46E8tKE9y4pyAXRV5kZMKMIRxyblgSLGc3S1ee3RGbBvrzO
AU3IX4Tuwm+7w+gBu7rELnmA06T5R06Zpj261cFxT8FMedKIS+IUn65E8QKBgFLH
GXFiML2o/RiZb+aaZyRXVFxwJuWh14HV843oU4hr4XGVdJFXGW2BdzkljEdiNb2N
M66DtQhjIZw0Gu5LRaAejrW9evydvPeWG7/oYZnYAi2FRR41sJSRCosKViyUVDRh
W6K2zdp6eVq3ld4SxjGvOHP6HsluYB5xWLEv0WEHAoGBAM5mjddMrxDaGhAutKMr
f6ErbXvn90V+eMVmA+blazYBjwUaanEcFkwPIeZHLBw3AU2OC6evn2aTDE/FeRH8
qGmvs8oPXofZ+FvUxcVxpC6/brSzmZoMhtRoWzuejFqpy7UUxA3oP4WAvzaqQBGY
E3wAGFVUsR1oryyt2K5QVCjW
-----END PRIVATE KEY-----
"""


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
    metrics_service = SimpleNamespace(enabled=True, auth_token="" if with_warning else "secret", tls=metrics_tls)
    risk_service = SimpleNamespace(enabled=False, auth_token="token", tls=risk_tls)
    return SimpleNamespace(metrics_service=metrics_service, risk_service=risk_service)


def test_audit_tls_assets_script_json_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cert_path = _write(tmp_path / "cert.pem", _CERT)
    key_path = _write(tmp_path / "key.pem", _KEY)
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


def test_audit_tls_assets_script_env_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    cert_path = _write(tmp_path / "cert.pem", _CERT)
    key_path = _write(tmp_path / "key.pem", _KEY)
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
