from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.security.certificates import certificate_reference_metadata


_SAMPLE_CERT = """-----BEGIN CERTIFICATE-----
MIIDCzCCAfOgAwIBAgIUXg2adJP0b1IzyHUIygGCz5e4CZswDQYJKoZIhvcNAQEL
BQAwFTETMBEGA1UEAwwKdGVzdC5sb2NhbDAeFw0yNTEwMTIxMzUzMThaFw0yNjEw
MTIxMzUzMThaMBUxEzARBgNVBAMMCnRlc3QubG9jYWwwggEiMA0GCSqGSIb3DQEB
AQUAA4IBDwAwggEKAoIBAQCoTFYSrtuGtDlfV6zjE1YQZz8rDoW58bid8CcMXAA/
6sXvCDYFepgjJpD1dDaXLIORNaoteLJLDwd/GbWhK589n/+KzLfnrS+vC+Y9/Zwr
9mCJbQ6liPYleqidG98cY50nrru8IiLORWDDWPLMcNJbGuTkg+JUkmVhaLSmWF1u
GLN+5IFJ+NXo3fUW5B1swcGYFlta0KelpNaatyMKQZZ7wO4QXp8H/ajBbXpWnE/n
JAYOqdBS03+AVpy2Qr0HnCw8NxSur9pqLY4EDepPgowBaMuVv8XgG8+XYJm6y9HH
r1sDd7u1Nq+EnqDfuwZ7HRe5sNQXCW8MwOp+kWqVgrGVAgMBAAGjUzBRMB0GA1Ud
DgQWBBQgFv08KgCA0frYHzc8GrR6RFyi+jAfBgNVHSMEGDAWgBQgFv08KgCA0frY
Hzc8GrR6RFyi+jAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBCwUAA4IBAQAP
LvBg5dsZoS2IZXZMv89QTJsI2pndypdL8iQND7StJstgp57f18rOIqn6M3Hw+RwN
NnFs84Jm9qrPRto8jd/sxFFEKYwraLOYadTdIorRYFNdoSYA8dQwh4vUbWLGL5pV
EYrSDEItLWPznRLYx2O5NIo1sM6/LWDDSYbwE9EUCxrCFzF9TQT8lVLQDE2KTCbB
EiSKc+jb0Y2FlBCTDbfpK3CrvS2nhP+Vh+M9iyacJNHEhbiEkam0wje5/6gDsBoi
ouji8dqM2xUMITUW+BWHn/eDOqLss5ZlRbGTNm0Hq3CP7uf0u15r5QxBJjzZYddr
WTcLAxCHylb4RYSCze+i
-----END CERTIFICATE-----
"""


@pytest.mark.parametrize("role", ["risk_tls_certificate", "tls_cert"])
def test_certificate_reference_metadata_parses_fingerprint(tmp_path: Path, role: str) -> None:
    cert_path = tmp_path / "risk_cert.pem"
    cert_path.write_text(_SAMPLE_CERT, encoding="utf-8")

    metadata = certificate_reference_metadata(cert_path, role=role)

    assert metadata["exists"] is True
    warnings = metadata.get("security_warnings")
    assert isinstance(warnings, list)
    assert any("Plik jest dostępny do odczytu" in warning for warning in warnings)
    certificates = metadata.get("certificates")
    assert isinstance(certificates, list)
    assert len(certificates) == 1
    entry = certificates[0]
    assert entry["fingerprint_sha256"] == "a49ede6616a62c76e2affdf692aa103cfeb89ddbd1f0f03b13a8a3166aa63079"
    assert entry["subject"]["commonName"] == "test.local"
    assert "not_after" in entry


def test_certificate_reference_metadata_warns_on_invalid_file(tmp_path: Path) -> None:
    cert_path = tmp_path / "broken.pem"
    cert_path.write_text("not-a-certificate", encoding="utf-8")

    metadata = certificate_reference_metadata(cert_path)

    assert metadata["exists"] is True
    warnings = metadata.get("security_warnings")
    assert isinstance(warnings, list)
    assert any("formacie PEM" in warning for warning in warnings)
