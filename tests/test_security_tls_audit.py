from pathlib import Path
from types import SimpleNamespace

from bot_core.security.tls_audit import audit_tls_assets, audit_tls_entry, verify_certificate_key_pair


_CERT_PEM = """-----BEGIN CERTIFICATE-----
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

_MATCHING_KEY_PEM = """-----BEGIN PRIVATE KEY-----
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

_MISMATCH_KEY_PEM = """-----BEGIN PRIVATE KEY-----
MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCoTFYSrtuGtDlf
V6zjE1YQZz8rDoW58bid8CcMXAA/6sXvCDYFepgjJpD1dDaXLIORNaoteLJLDwd/
GbWhK589n/+KzLfnrS+vC+Y9/Zwr9mCJbQ6liPYleqidG98cY50nrru8IiLORWDD
WPLMcNJbGuTkg+JUkmVhaLSmWF1uGLN+5IFJ+NXo3fUW5B1swcGYFlta0KelpNaa
tyMKQZZ7wO4QXp8H/ajBbXpWnE/nJAYOqdBS03+AVpy2Qr0HnCw8NxSur9pqLY4E
DepPgowBaMuVv8XgG8+XYJm6y9HHr1sDd7u1Nq+EnqDfuwZ7HRe5sNQXCW8MwOp+
kWqVgrGVAgMBAAECggEATxe89b/OdIJbWibancb3EfNruOD00Lu8XyE/QKw2A9Pi
XKE3viBswkw8INaSVz54wHP/e6o25FZ2V/GtrcZR6oS4dDMclJkMCVBmzqhSzkhV
+w/RK9NvlpKMDnXMR0u7TixslxBV2im5vWSeipzVBzLe8lPWuJcqZPpvt6NcmUHo
dCVW+fqrZeTmYAMi4oCtCr9GqznYjK0Lpp1RaeGTEo9nzdULaR/9KcqjsMSUXrmY
5JgGgA0iFLcA9EYCMC6C6kpLuimW5FYO+pNNhRPPGWA9Lti5UMjLAG9bgr4cnsLK
PXaCYTlL95LGSq4JvMPwsytmSUnzDPejp+sEy8gXawKBgQDPssDA31r8tDpO7MRs
fvmICQ0760EyQ17bpYZ6cKdfHfMfnEqaJU6MBurhkLA9EsajmYgHJQbCyX10Pso9
3fcTlMCGsvFYT8F5iNkEjGg7PAKIEmjgufmOoip20+uKLCm2kFi4km/BY10EQ5sk
znmFV61A6BMLu+cqnNpXYqOvCwKBgQDPb+nTVN8knCXn1CRc/JBvfZpktN0/INZ2
Rlnjn/C1oVVhuXiAuvEfULs6wkViCNd9C6Sle69BCAu3rVX1D2YPAorUx5jp07sE
2HrFIpLl15QqT5YuE6sOP+tXxLSF9J8SiJWYHGJBfHF8HXuci+wRXJb01mXkdAIN
c4lLvMYF3wKBgFkKiRgmqRstKNItLwhUZyWqu8G0WX7y4vfHPp+/LAHbFR+4IUN0
OvhM/uU04llMc1wvteFaPkvDlcUAJjPftMzwOJmGnXD+wDMaN+97QjQixfMP8WZm
VFaRryLCN3hE9p0NxPtbzA1cS8RIN3rQCcjgjaYF2CRvqera08AiyYmBAoGAUlRt
roXB5sretItbP1iyjr2AOLYcFcEXvWugo5pINB5rP9UYAaewqagmF2Uhmo490JB9
cXyMizgBRo5STmglLpHovhjWFQAG+x5cY7+cJAMS+FQMHA+MVaSC6JvWtk/njriM
/wlM6gbVF9ivxes275EbDOPHHwv4AJS5ikjLI2sCgYArrbobSUf8aDrs86EgS4Jb
Nd1bjYx8SWC+uRyZNfsE/TllEtyrP8yWMP6Tq9S+uMXrHfcE1Rj7bj9fJYxeOzQe
geENbbsvj9oU2pNPKWlwk861WMZcppkpmVHrgIIukGC+DSTWFntBGNyDRBdgE+gg
ymp4BN4Riifev8GdFf+lMg==
-----END PRIVATE KEY-----
"""


def _write_material(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_verify_certificate_key_pair_success(tmp_path: Path) -> None:
    cert_path = _write_material(tmp_path / "cert.pem", _CERT_PEM)
    key_path = _write_material(tmp_path / "key.pem", _MATCHING_KEY_PEM)

    ok, message = verify_certificate_key_pair(cert_path, key_path)

    assert ok is True
    assert message is None


def test_verify_certificate_key_pair_failure(tmp_path: Path) -> None:
    cert_path = _write_material(tmp_path / "cert.pem", _CERT_PEM)
    key_path = _write_material(tmp_path / "key.pem", _MISMATCH_KEY_PEM)

    ok, message = verify_certificate_key_pair(cert_path, key_path)

    assert ok is False
    assert message is not None
    assert "Klucz prywatny" in message


def test_audit_tls_entry_reports_pins(tmp_path: Path) -> None:
    cert_path = _write_material(tmp_path / "cert.pem", _CERT_PEM)
    key_path = _write_material(tmp_path / "key.pem", _MATCHING_KEY_PEM)
    config = SimpleNamespace(
        enabled=True,
        certificate_path=str(cert_path),
        private_key_path=str(key_path),
        client_ca_path=None,
        require_client_auth=False,
        private_key_password_env=None,
        pinned_fingerprints=("sha256:7d820a06ec208056b274865f8a430b761172fef4c3e2fa9c26b9de010e692efb",),
    )

    report = audit_tls_entry(config, role_prefix="test_tls", env={})

    assert report["enabled"] is True
    assert report["key_matches_certificate"] is True
    assert report["pinned_fingerprint_match"] is True
    assert not report["errors"]


def test_audit_tls_entry_detects_pin_mismatch(tmp_path: Path) -> None:
    cert_path = _write_material(tmp_path / "cert.pem", _CERT_PEM)
    key_path = _write_material(tmp_path / "key.pem", _MATCHING_KEY_PEM)
    config = SimpleNamespace(
        enabled=True,
        certificate_path=str(cert_path),
        private_key_path=str(key_path),
        client_ca_path=None,
        require_client_auth=False,
        private_key_password_env=None,
        pinned_fingerprints=("sha256:deadbeef",),
    )

    report = audit_tls_entry(config, role_prefix="test_tls", env={})

    assert report["pinned_fingerprint_match"] is False
    assert report["errors"]


def test_audit_tls_assets_aggregates_services(tmp_path: Path) -> None:
    cert_path = _write_material(tmp_path / "cert.pem", _CERT_PEM)
    key_path = _write_material(tmp_path / "key.pem", _MATCHING_KEY_PEM)

    metrics_tls = SimpleNamespace(
        enabled=True,
        certificate_path=str(cert_path),
        private_key_path=str(key_path),
        client_ca_path=None,
        require_client_auth=False,
        private_key_password_env="METRICS_TLS_KEY",
        pinned_fingerprints=("sha256:deadbeef",),
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
    core_config = SimpleNamespace(
        metrics_service=SimpleNamespace(
            enabled=True,
            auth_token="",
            tls=metrics_tls,
        ),
        risk_service=SimpleNamespace(
            enabled=True,
            auth_token=None,
            tls=risk_tls,
        ),
    )

    report = audit_tls_assets(core_config)

    assert "metrics_service" in report["services"]
    metrics_report = report["services"]["metrics_service"]
    assert metrics_report["warnings"]
    assert report["warnings"]
    assert report["errors"]  # brak dopasowania fingerprintu
