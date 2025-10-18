import json
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from bot_core.config.models import LicenseValidationConfig
from bot_core.security.license import (
    LicenseValidationError,
    validate_license,
    validate_license_from_config,
    load_hmac_keys_file,
)
from bot_core.security.signing import build_hmac_signature


LICENSE_KEY = bytes.fromhex("6b" * 32)
FINGERPRINT_KEY = bytes.fromhex("5a" * 32)
REVOCATION_KEY = bytes.fromhex("4c" * 32)


def _write_keys(path: Path, *, key_id: str, secret: bytes) -> None:
    payload = {"keys": {key_id: f"hex:{secret.hex()}"}}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _fingerprint_payload(value: str) -> dict[str, object]:
    collected = datetime(2024, 1, 1, 12, 0, 0).isoformat() + "Z"
    return {
        "version": 1,
        "collected_at": collected,
        "components": {},
        "component_digests": {},
        "fingerprint": {"algorithm": "sha256", "value": value},
    }


def _signed_fingerprint(value: str, *, key_id: str = "fp-1") -> tuple[dict[str, object], dict[str, str]]:
    payload = _fingerprint_payload(value)
    signature = build_hmac_signature(payload, key=FINGERPRINT_KEY, algorithm="HMAC-SHA384", key_id=key_id)
    return payload, signature


def _signed_license(
    path: Path,
    *,
    fingerprint_value: str = "abc123",
    issued_at: str | None = None,
    expires_at: str | None = None,
    issuer: str = "qa",
    profile: str = "paper",
    license_id: str = "lic-valid",
) -> tuple[Path, Path, dict[str, bytes], dict[str, bytes]]:
    fingerprint_payload, fingerprint_signature = _signed_fingerprint(fingerprint_value)
    issued = issued_at or "2024-01-01T00:00:00Z"
    expires = expires_at or "2040-01-01T00:00:00Z"
    license_payload = {
        "schema": "core.oem.license",
        "schema_version": "1.0",
        "issued_at": issued,
        "expires_at": expires,
        "issuer": issuer,
        "profile": profile,
        "license_id": license_id,
        "fingerprint": fingerprint_payload["fingerprint"],
        "fingerprint_payload": fingerprint_payload,
        "fingerprint_signature": fingerprint_signature,
    }
    license_signature = build_hmac_signature(
        license_payload,
        key=LICENSE_KEY,
        algorithm="HMAC-SHA384",
        key_id="lic-1",
    )
    license_document = {"payload": license_payload, "signature": license_signature}
    license_path = path / "license.json"
    license_path.write_text(json.dumps(license_document, ensure_ascii=False), encoding="utf-8")

    fingerprint_path = path / "fingerprint.json"
    fingerprint_document = {"payload": fingerprint_payload, "signature": fingerprint_signature}
    fingerprint_path.write_text(json.dumps(fingerprint_document, ensure_ascii=False), encoding="utf-8")

    license_keys_path = path / "license_keys.json"
    fingerprint_keys_path = path / "fingerprint_keys.json"
    _write_keys(license_keys_path, key_id="lic-1", secret=LICENSE_KEY)
    _write_keys(fingerprint_keys_path, key_id="fp-1", secret=FINGERPRINT_KEY)

    return (
        license_path,
        fingerprint_path,
        load_hmac_keys_file(license_keys_path),
        load_hmac_keys_file(fingerprint_keys_path),
    )


def _write_revocation_list(
    path: Path,
    *,
    revoked: Sequence[object] = (),
    generated_at: str | None = None,
    sign: bool = False,
    key_id: str = "rev-1",
) -> Path:
    entries: list[dict[str, object]] = []
    for value in revoked:
        if isinstance(value, Mapping):
            entry = {**value}
        else:
            entry = {"license_id": value}
        entries.append(entry)
    payload: dict[str, object] = {
        "revoked": entries,
    }
    if generated_at:
        payload["generated_at"] = generated_at
    revocation_document: dict[str, object] | dict[str, object]
    if sign:
        signature = build_hmac_signature(
            payload,
            key=REVOCATION_KEY,
            algorithm="HMAC-SHA384",
            key_id=key_id,
        )
        revocation_document = {"payload": payload, "signature": signature}
    else:
        revocation_document = payload
    revocation_path = path / "revocations.json"
    revocation_path.write_text(
        json.dumps(revocation_document, ensure_ascii=False),
        encoding="utf-8",
    )
    return revocation_path


def _write_revocation_keys(path: Path, *, key_id: str = "rev-1") -> Path:
    payload = {"keys": {key_id: f"hex:{REVOCATION_KEY.hex()}"}}
    output = path / "revocation_keys.json"
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return output


def test_validate_license_accepts_signed_document(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(tmp_path)

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
    )

    assert result.is_valid
    assert result.fingerprint == "abc123"
    assert result.issued_at == "2024-01-01T00:00:00Z"
    assert result.expires_at == "2040-01-01T00:00:00Z"
    assert result.profile == "paper"
    assert result.issuer == "qa"
    assert result.schema == "core.oem.license"
    assert result.schema_version == "1.0"
    assert result.license_id == "lic-valid"
    assert result.revocation_status == "skipped"
    assert result.revocation_checked is False
    assert result.revocation_reason is None
    assert result.revocation_revoked_at is None
    assert result.revocation_list_path is None


def test_validate_license_detects_invalid_signature(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(tmp_path)
    document = json.loads(license_path.read_text(encoding="utf-8"))
    document["signature"]["value"] = "corrupted"
    license_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
    )

    assert result.status == "invalid"
    assert result.errors
    assert result.schema == "core.oem.license"


def test_validate_license_checks_revocation_list(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(tmp_path)
    revocation_path = _write_revocation_list(
        tmp_path,
        revoked=[
            {
                "license_id": "lic-valid",
                "reason": "Fingerprint mismatch",
                "revoked_at": "2024-05-31T10:00:00Z",
            }
        ],
        generated_at="2024-06-01T00:00:00Z",
    )

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        revocation_list_path=revocation_path,
        revocation_required=True,
        revocation_list_max_age_hours=48,
        current_time=datetime(2024, 6, 2, tzinfo=timezone.utc),
    )

    assert result.status == "invalid"
    assert any("liście odwołań" in err for err in result.errors)
    assert result.revocation_status == "revoked"
    assert result.revocation_checked is True
    assert result.revocation_reason == "Fingerprint mismatch"
    assert result.revocation_revoked_at == "2024-05-31T10:00:00+00:00"


def test_validate_license_reports_invalid_revoked_timestamp(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(tmp_path)
    revocation_path = _write_revocation_list(
        tmp_path,
        revoked=[
            {
                "license_id": "lic-valid",
                "reason": "Key compromised",
                "revoked_at": "31-05-2024",
            }
        ],
        generated_at="2024-06-01T00:00:00Z",
    )

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        revocation_list_path=revocation_path,
        revocation_required=True,
        revocation_list_max_age_hours=48,
        current_time=datetime(2024, 6, 2, tzinfo=timezone.utc),
    )

    assert result.status == "invalid"
    assert any("revoked_at" in warn for warn in result.warnings)
    assert result.revocation_status == "revoked"
    assert result.revocation_reason == "Key compromised"
    assert result.revocation_revoked_at == "31-05-2024"


def test_validate_license_accepts_clear_revocation_list(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(
        tmp_path,
        license_id="lic-active",
    )
    revocation_path = _write_revocation_list(
        tmp_path,
        revoked=["lic-other"],
        generated_at="2024-06-01T12:00:00Z",
    )

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        revocation_list_path=revocation_path,
        revocation_required=True,
        revocation_list_max_age_hours=72,
        current_time=datetime(2024, 6, 2, tzinfo=timezone.utc),
    )

    assert result.is_valid
    assert result.revocation_status == "clear"
    assert result.revocation_generated_at == "2024-06-01T12:00:00+00:00"
    assert result.revocation_checked is True
    assert result.revocation_reason is None
    assert result.revocation_revoked_at is None


def test_validate_license_verifies_signed_revocation_list(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(
        tmp_path,
        license_id="lic-signed",
    )
    revocation_path = _write_revocation_list(
        tmp_path,
        revoked=["lic-other"],
        generated_at="2024-06-01T12:00:00Z",
        sign=True,
        key_id="rev-main",
    )
    revocation_keys_path = _write_revocation_keys(tmp_path, key_id="rev-main")
    revocation_keys = load_hmac_keys_file(revocation_keys_path)

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        revocation_list_path=revocation_path,
        revocation_keys=revocation_keys,
        revocation_signature_required=True,
        revocation_list_max_age_hours=48,
        current_time=datetime(2024, 6, 2, tzinfo=timezone.utc),
    )

    assert result.is_valid
    assert result.revocation_status == "clear"
    assert result.revocation_signature_key == "rev-main"
    assert result.revocation_reason is None
    assert result.revocation_revoked_at is None


def test_validate_license_requires_revocation_when_flag_set(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(tmp_path)

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        revocation_required=True,
    )

    assert result.status == "invalid"
    assert any("wymaga listy odwołań" in err for err in result.errors)
    assert result.revocation_status == "missing"
    assert result.revocation_checked is False
    assert result.revocation_reason is None
    assert result.revocation_revoked_at is None


def test_validate_license_requires_signed_revocation_when_flag_set(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(
        tmp_path,
        license_id="lic-unsigned",
    )
    revocation_path = _write_revocation_list(
        tmp_path,
        revoked=[],
        generated_at="2024-06-01T00:00:00Z",
    )

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        revocation_list_path=revocation_path,
        revocation_required=True,
        revocation_signature_required=True,
        revocation_list_max_age_hours=24,
        current_time=datetime(2024, 6, 1, 6, tzinfo=timezone.utc),
    )

    assert result.status == "invalid"
    assert any("podpisanej listy odwołań" in err for err in result.errors)


def test_validate_license_requires_revocation_keys_for_signature(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(
        tmp_path,
        license_id="lic-unsigned",
    )
    revocation_path = _write_revocation_list(
        tmp_path,
        revoked=[],
        generated_at="2024-06-01T00:00:00Z",
        sign=True,
        key_id="rev-main",
    )

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        revocation_list_path=revocation_path,
        revocation_signature_required=True,
        revocation_list_max_age_hours=24,
        current_time=datetime(2024, 6, 1, 6, tzinfo=timezone.utc),
    )

    assert result.status == "invalid"
    assert any("brak dostępnych kluczy" in err.lower() for err in result.errors)


def test_validate_license_detects_stale_revocation_list(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(tmp_path)
    revocation_path = _write_revocation_list(
        tmp_path,
        revoked=[],
        generated_at="2024-06-01T00:00:00Z",
    )

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        revocation_list_path=revocation_path,
        revocation_required=True,
        revocation_list_max_age_hours=2,
        current_time=datetime(2024, 6, 1, 5, tzinfo=timezone.utc),
    )

    assert result.status == "invalid"
    assert any("starsza" in err for err in result.errors)
    assert result.revocation_status == "stale"
    assert result.revocation_reason is None
    assert result.revocation_revoked_at is None


def test_validate_license_warns_about_missing_revocation_timestamp(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(tmp_path)
    revocation_path = _write_revocation_list(tmp_path, revoked=["other"])

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        revocation_list_path=revocation_path,
        revocation_list_max_age_hours=12,
        current_time=datetime(2024, 6, 1, 12, tzinfo=timezone.utc),
    )

    assert result.status == "ok"
    assert any("generated_at" in warn for warn in result.warnings)
    assert result.revocation_status in {"unknown", "clear"}
    assert result.revocation_reason is None
    assert result.revocation_revoked_at is None
def test_validate_license_detects_mismatched_fingerprint_file(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(tmp_path)
    other_payload, other_signature = _signed_fingerprint("other")
    fingerprint_document = {"payload": other_payload, "signature": other_signature}
    fingerprint_path.write_text(json.dumps(fingerprint_document, ensure_ascii=False), encoding="utf-8")

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
    )

    assert result.status == "invalid"
    assert any("nie zgadza" in msg.lower() for msg in result.errors)
    assert result.schema == "core.oem.license"


def test_validate_license_reports_expired_license(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(
        tmp_path,
        issued_at="2019-01-01T00:00:00Z",
        expires_at="2020-01-01T00:00:00Z",
    )

    now = datetime(2021, 1, 1, tzinfo=timezone.utc)
    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        current_time=now,
    )

    assert result.status == "invalid"
    assert any("wygas" in msg.lower() for msg in result.errors)
    assert result.schema == "core.oem.license"


def test_validate_license_warns_on_upcoming_expiry(tmp_path: Path) -> None:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    expires = (now + timedelta(days=10)).isoformat().replace("+00:00", "Z")
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(
        tmp_path,
        expires_at=expires,
    )

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        current_time=now,
    )

    assert result.is_valid
    assert any("30 dni" in msg for msg in result.warnings)
    assert result.schema == "core.oem.license"


def test_validate_license_warns_on_future_issue_date(tmp_path: Path) -> None:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    future_issue = (now + timedelta(days=1)).isoformat().replace("+00:00", "Z")
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(
        tmp_path,
        issued_at=future_issue,
        expires_at="2040-01-01T00:00:00Z",
    )

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        current_time=now,
    )

    assert result.is_valid
    assert any("w przyszłości" in msg.lower() for msg in result.warnings)
    assert result.schema == "core.oem.license"


def test_validate_license_rejects_disallowed_profile(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(
        tmp_path,
        profile="paper",
    )

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        allowed_profiles=("ops",),
    )

    assert result.status == "invalid"
    assert any("Profil licencji" in msg for msg in result.errors)


def test_validate_license_rejects_disallowed_issuer(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(
        tmp_path,
        issuer="qa",
    )

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        allowed_issuers=("security",),
    )

    assert result.status == "invalid"
    assert any("Wystawca licencji" in msg for msg in result.errors)


def test_validate_license_rejects_unexpected_schema(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(tmp_path)
    document = json.loads(license_path.read_text(encoding="utf-8"))
    document["payload"]["schema"] = "unexpected.schema"
    license_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
    )

    assert result.status == "invalid"
    assert any("nieoczekiwany" in msg for msg in result.errors)


def test_validate_license_rejects_unknown_schema_version(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(tmp_path)
    document = json.loads(license_path.read_text(encoding="utf-8"))
    document["payload"]["schema_version"] = "9.9"
    license_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
    )

    assert result.status == "invalid"
    assert any("nieobsługiwan" in msg for msg in result.errors)


def test_validate_license_rejects_excessive_validity(tmp_path: Path) -> None:
    license_path, fingerprint_path, license_keys, fingerprint_keys = _signed_license(
        tmp_path,
        issued_at="2024-01-01T00:00:00Z",
        expires_at="2027-01-01T00:00:00Z",
    )

    result = validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        max_validity_days=365.0,
    )

    assert result.status == "invalid"
    assert any("Okres ważności" in msg for msg in result.errors)


def test_validate_license_from_config_raises_on_failure(tmp_path: Path) -> None:
    license_path, fingerprint_path, *_ = _signed_license(tmp_path)
    config = LicenseValidationConfig(
        license_path=str(license_path),
        fingerprint_path=str(fingerprint_path),
        license_keys_path=str(tmp_path / "missing_license_keys.json"),
        fingerprint_keys_path=str(tmp_path / "missing_fingerprint_keys.json"),
    )

    with pytest.raises(LicenseValidationError):
        validate_license_from_config(config)


def test_validate_license_from_config_passes(tmp_path: Path) -> None:
    license_path, fingerprint_path, _, _ = _signed_license(tmp_path)
    config = LicenseValidationConfig(
        license_path=str(license_path),
        fingerprint_path=str(fingerprint_path),
        license_keys_path=str(tmp_path / "license_keys.json"),
        fingerprint_keys_path=str(tmp_path / "fingerprint_keys.json"),
        allowed_profiles=("paper",),
        allowed_issuers=("qa",),
        max_validity_days=6000.0,
    )
    # klucze generowane w helperze _signed_license
    result = validate_license_from_config(config)
    assert result.is_valid


def test_validate_license_from_config_rejects_disallowed_profile(tmp_path: Path) -> None:
    license_path, fingerprint_path, _, _ = _signed_license(tmp_path)
    config = LicenseValidationConfig(
        license_path=str(license_path),
        fingerprint_path=str(fingerprint_path),
        license_keys_path=str(tmp_path / "license_keys.json"),
        fingerprint_keys_path=str(tmp_path / "fingerprint_keys.json"),
        allowed_profiles=("ops",),
    )

    with pytest.raises(LicenseValidationError) as excinfo:
        validate_license_from_config(config)

    assert excinfo.value.result is not None
    assert any("Profil licencji" in msg for msg in excinfo.value.result.errors)


def test_validate_license_from_config_rejects_schema_version(tmp_path: Path) -> None:
    license_path, fingerprint_path, _, _ = _signed_license(tmp_path)
    config = LicenseValidationConfig(
        license_path=str(license_path),
        fingerprint_path=str(fingerprint_path),
        license_keys_path=str(tmp_path / "license_keys.json"),
        fingerprint_keys_path=str(tmp_path / "fingerprint_keys.json"),
        allowed_schema_versions=("9.9",),
    )

    with pytest.raises(LicenseValidationError) as excinfo:
        validate_license_from_config(config)

    assert excinfo.value.result is not None
    assert any("nieobsługiwan" in msg for msg in excinfo.value.result.errors)


def test_validate_license_from_config_checks_revocation(tmp_path: Path) -> None:
    license_path, fingerprint_path, _, _ = _signed_license(tmp_path)
    revocation_path = _write_revocation_list(
        tmp_path,
        revoked=["lic-valid"],
        generated_at="2024-07-01T00:00:00Z",
        sign=True,
        key_id="rev-main",
    )
    revocation_keys_path = _write_revocation_keys(tmp_path, key_id="rev-main")
    config = LicenseValidationConfig(
        license_path=str(license_path),
        fingerprint_path=str(fingerprint_path),
        license_keys_path=str(tmp_path / "license_keys.json"),
        fingerprint_keys_path=str(tmp_path / "fingerprint_keys.json"),
        revocation_list_path=str(revocation_path),
        revocation_required=True,
        revocation_list_max_age_hours=None,
        revocation_keys_path=str(revocation_keys_path),
        revocation_signature_required=True,
    )

    with pytest.raises(LicenseValidationError) as excinfo:
        validate_license_from_config(config)

    assert excinfo.value.result is not None
    assert any("liście odwołań" in msg for msg in excinfo.value.result.errors)


def test_validate_license_from_config_requires_keys_for_signed_revocations(tmp_path: Path) -> None:
    license_path, fingerprint_path, _, _ = _signed_license(tmp_path)
    revocation_path = _write_revocation_list(
        tmp_path,
        revoked=[],
        generated_at="2024-07-01T00:00:00Z",
        sign=True,
    )
    config = LicenseValidationConfig(
        license_path=str(license_path),
        fingerprint_path=str(fingerprint_path),
        license_keys_path=str(tmp_path / "license_keys.json"),
        fingerprint_keys_path=str(tmp_path / "fingerprint_keys.json"),
        revocation_list_path=str(revocation_path),
        revocation_required=True,
        revocation_list_max_age_hours=None,
        revocation_signature_required=True,
    )

    with pytest.raises(LicenseValidationError) as excinfo:
        validate_license_from_config(config)

    assert "revocation_keys_path" in str(excinfo.value)


def test_validate_license_from_config_accepts_signed_revocations(tmp_path: Path) -> None:
    license_path, fingerprint_path, _, _ = _signed_license(
        tmp_path,
        license_id="lic-signed",
    )
    revocation_path = _write_revocation_list(
        tmp_path,
        revoked=["other"],
        generated_at="2024-07-01T00:00:00Z",
        sign=True,
        key_id="rev-main",
    )
    revocation_keys_path = _write_revocation_keys(tmp_path, key_id="rev-main")
    config = LicenseValidationConfig(
        license_path=str(license_path),
        fingerprint_path=str(fingerprint_path),
        license_keys_path=str(tmp_path / "license_keys.json"),
        fingerprint_keys_path=str(tmp_path / "fingerprint_keys.json"),
        revocation_list_path=str(revocation_path),
        revocation_required=True,
        revocation_list_max_age_hours=None,
        revocation_keys_path=str(revocation_keys_path),
        revocation_signature_required=True,
    )

    result = validate_license_from_config(config)

    assert result.is_valid
    assert result.revocation_signature_key == "rev-main"
    assert result.revocation_status == "clear"
    assert result.revocation_reason is None
