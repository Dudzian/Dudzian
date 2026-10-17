"""TEST_ONLY native boundaries; exact transport, quorum and LPPI crypto are real."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot_core.licensing import enrollment_file_io as file_io
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.enrollment_file_io import read_artifact_file, write_artifact_file
from bot_core.licensing.enrollment_transport import ArtifactEnvelope
from deployment import (
    production_enrollment_offline as pdsa_offline,
    windows_enrollment_offline as offline,
    windows_production_lppi_authority as lifecycle,
    windows_production_pre_enrollment as installed,
)
from deployment.windows_cng_pre_enrollment import WindowsCNGPreEnrollmentKey
from tests.licensing import (
    test_lppi_authority_lifecycle as lppi_tests,
    test_pdsa_enrollment_authorization as authorization_tests,
)

NOW = lppi_tests.NOW
challenge_harness = lppi_tests.challenge_harness
integration = lppi_tests.integration
package = lppi_tests.package
lppi = lppi_tests.lppi
issuance = authorization_tests.issuance


@pytest.fixture
def exchange(lppi, monkeypatch, tmp_path):
    item = lppi.item
    state = tmp_path / "TEST_ONLY_OFFLINE_STATE"
    state.mkdir()
    monkeypatch.setattr(offline, "resolve_paths", lambda: SimpleNamespace(state=state))
    monkeypatch.setattr(offline, "_trust", lambda: item.authority.context)
    monkeypatch.setattr(offline, "_utc_now", lambda: NOW)
    # The reused fixture owns the genuine verifier-issued NCrypt handle. Keep
    # it alive across simulated process restarts; no production injection API.
    for method in ("open_or_create", "open_existing"):
        monkeypatch.setattr(WindowsCNGPreEnrollmentKey, method, lambda path: nullcontext(item.key))
    calls = []

    def producer(**arguments):
        calls.append(arguments)
        assert arguments["ak_key_name"] == "Stage9.Retained.AK"
        return SimpleNamespace(
            request_raw=item.arguments["request_raw"],
            signature=item.arguments["signature"],
            custody_evidence_raw=item.arguments["custody_evidence_raw"],
        )

    monkeypatch.setattr(offline, "prepare_installed_production_request", producer)
    device = offline.export_device_request(
        activation_request_raw=item.arguments["activation_request_raw"],
        tpm_request_raw=item.arguments["tpm_request_raw"],
        ak_key_name="Stage9.Retained.AK",
        output=tmp_path / "device.json",
    )
    challenge = ArtifactEnvelope.create(
        "TPM_CHALLENGE",
        device.artifacts
        | {
            "pdsa_challenge": item.pdsa_raw,
            "endorsement": item.endorsement_raw,
            "tpm_challenge": item.arguments["tpm_challenge_raw"],
        },
        parent_id=device.request_id,
    )
    challenge_file = tmp_path / "challenge.json"
    write_artifact_file(challenge_file, challenge.canonical_bytes)
    return SimpleNamespace(**locals())


def _prepare(exchange):
    offline.import_tpm_challenge(exchange.challenge_file)
    return offline.prepare_pre_enrollment(
        device_request_id=exchange.device.request_id,
        tpm_response_raw=exchange.item.arguments["tpm_response_raw"],
        output=exchange.tmp_path / "pre-enrollment.json",
    )


def _package(exchange, prepared, *, raw=None, parent=None):
    envelope = ArtifactEnvelope.create(
        "PDSA_PACKAGE",
        {"package": exchange.lppi.package.raw if raw is None else raw},
        parent_id=prepared.request_id if parent is None else parent,
    )
    path = exchange.tmp_path / (envelope.request_id + ".json")
    write_artifact_file(path, envelope.canonical_bytes)
    return envelope, path


def test_exact_offline_retries_restart_and_live_package_acceptance(exchange):
    prepared = _prepare(exchange)
    # A lost output and a fresh invocation recover the exact retained nonce,
    # signature and custody rather than generating a second request.
    (exchange.tmp_path / "pre-enrollment.json").unlink()
    assert _prepare(exchange).canonical_bytes == prepared.canonical_bytes
    assert len(exchange.calls) == 1
    envelope, source = _package(exchange, prepared)
    assert (
        offline.import_pdsa_package(source, device_request_id=exchange.device.request_id)
        == envelope
    )
    assert (
        offline.import_pdsa_package(source, device_request_id=exchange.device.request_id)
        == envelope
    )
    assert (
        read_artifact_file(exchange.state / "OfflineEnrollment" / "accepted-package.json")
        == envelope.canonical_bytes
    )
    # Real LPPI acceptance performed fresh target PoP and AK custody, but it
    # never called the successor lifecycle or established membership.
    assert not exchange.lppi.successor.calls
    assert not lifecycle._state_path().exists()


def test_challenge_from_other_session_is_rejected(exchange):
    wrong = json.loads(exchange.challenge.canonical_bytes)
    wrong["parent_id"] = "ff" * 32
    path = exchange.tmp_path / "wrong-parent.json"
    write_artifact_file(path, canonical_json_bytes(wrong))
    with pytest.raises((FileNotFoundError, ValueError)):
        offline.import_tpm_challenge(path)
    assert not exchange.calls


def test_expired_tpm_challenge_never_prepares_request(exchange, monkeypatch):
    monkeypatch.setattr(offline, "_utc_now", lambda: NOW + timedelta(days=8))
    with pytest.raises(ValueError, match="EXPIRY"):
        offline.import_tpm_challenge(exchange.challenge_file)
    assert not exchange.calls


def test_changed_local_identity_is_rejected_before_import(exchange):
    path = exchange.state / "OfflineEnrollment" / exchange.device.request_id / "local-identity.json"
    value = json.loads(path.read_bytes())
    value["pre_enrollment_key_unique_name"] = "TEST_ONLY_other_device"
    # Simulate retained-state corruption; the transport never overwrites it.
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError, match="IDENTITY_CHANGED"):
        offline.import_tpm_challenge(exchange.challenge_file)


def test_live_other_tpm_is_rejected_before_import(exchange):
    exchange.lppi.pre_tbs.ak = b"TEST_ONLY_other_AK"
    with pytest.raises((ValueError, RuntimeError)):
        offline.import_tpm_challenge(exchange.challenge_file)


def test_package_wrong_parent_and_bad_quorum_signature_fail_closed(exchange):
    prepared = _prepare(exchange)
    _envelope, wrong_parent = _package(exchange, prepared, parent="dd" * 32)
    with pytest.raises(ValueError, match="PARENT"):
        offline.import_pdsa_package(wrong_parent, device_request_id=exchange.device.request_id)
    package = json.loads(exchange.lppi.package.raw)
    package["signatures"][0]["signature_hex"] = "00" * 64
    _envelope, bad_signature = _package(exchange, prepared, raw=canonical_json_bytes(package))
    with pytest.raises(RuntimeError, match="INVALID_PDSA_SIGNATURE"):
        offline.import_pdsa_package(bad_signature, device_request_id=exchange.device.request_id)
    assert not (exchange.state / "OfflineEnrollment" / "accepted-package.json").exists()
    assert not exchange.lppi.successor.calls


def test_completed_package_prevents_pre_enrollment_reactivation(exchange):
    prepared = _prepare(exchange)
    _envelope, source = _package(exchange, prepared)
    offline.import_pdsa_package(source, device_request_id=exchange.device.request_id)
    with pytest.raises(ValueError, match="ALREADY_ACCEPTED"):
        _prepare(exchange)


def test_successor_lifecycle_lock_excludes_offline_import(exchange):
    with lifecycle._locked_state():
        with pytest.raises(RuntimeError, match="LPPI_AUTHORITY_LIFECYCLE_BUSY"):
            offline.import_tpm_challenge(exchange.challenge_file)
    assert not exchange.calls
    assert not lifecycle._state_path().exists()


def test_native_package_acceptance_holds_shared_successor_lifecycle_lock(exchange, monkeypatch):
    prepared = _prepare(exchange)
    _envelope, source = _package(exchange, prepared)
    accept = offline.accept_production_lppi_package
    checked = []

    def concurrent_accept(*arguments, **keywords):
        # A separate lifecycle invocation cannot advance after the offline
        # absence check and before completion of native package acceptance.
        with pytest.raises(RuntimeError, match="LPPI_AUTHORITY_LIFECYCLE_BUSY"):
            with lifecycle._locked_state():
                pytest.fail("successor lifecycle raced with offline acceptance")
        checked.append(True)
        return accept(*arguments, **keywords)

    monkeypatch.setattr(offline, "accept_production_lppi_package", concurrent_accept)
    offline.import_pdsa_package(source, device_request_id=exchange.device.request_id)
    assert checked == [True]
    assert not lifecycle._state_path().exists()
    assert not exchange.lppi.successor.calls


def test_both_adapters_complete_actual_producer_issuer_and_live_lppi(
    exchange, issuance, monkeypatch
):
    """Only installed trust/configuration and native TPM/signers are TEST_ONLY.

    Both command adapters, the actual Windows request producer, independent
    issuer verification/signing/commit, and live LPPI acceptance execute here.
    """
    item = exchange.item
    assert issuance is item
    monkeypatch.setattr(
        offline,
        "prepare_installed_production_request",
        installed.prepare_installed_production_request,
    )
    monkeypatch.setattr(installed, "resolve_paths", lambda: SimpleNamespace(state=exchange.state))
    monkeypatch.setattr(
        installed, "production_trust_package_path", lambda ceremony: exchange.tmp_path / ceremony
    )
    monkeypatch.setattr(installed, "load_production_trust", lambda path: item.authority.context)
    pdsa_path = exchange.tmp_path / "public-pdsa-challenge.json"
    endorsement_path = exchange.tmp_path / "public-endorsement.json"
    write_artifact_file(pdsa_path, item.pdsa_raw)
    write_artifact_file(endorsement_path, item.endorsement_raw)
    challenge = pdsa_offline.issue_offline_tpm_challenge(
        exchange.tmp_path / "device.json",
        pdsa_challenge_path=pdsa_path,
        endorsement_path=endorsement_path,
        output_path=exchange.challenge_file,
    )
    assert challenge.canonical_bytes == exchange.challenge.canonical_bytes
    prepared = _prepare(exchange)
    assert not exchange.calls  # the actual request producer ran instead
    assert prepared.artifacts["request"] != item.request.canonical_bytes
    signed = pdsa_offline.authorize_offline_pre_enrollment(
        exchange.tmp_path / "pre-enrollment.json",
        output_path=exchange.tmp_path / "signed-package.json",
    )
    assert len(item.sign_calls) == 1
    accepted = offline.import_pdsa_package(
        exchange.tmp_path / "signed-package.json", device_request_id=exchange.device.request_id
    )
    assert accepted.canonical_bytes == signed.canonical_bytes
    assert not exchange.lppi.successor.calls


def test_windows_session_lock_uses_real_pywin32_argument_shapes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pywin32 wrappers omit dwReserved from the Win32 API signatures."""
    from bot_core import windows_custody_filesystem as filesystem

    calls: list[object] = []

    class Handle:
        def Close(self) -> None:
            calls.append("close")

    handle = Handle()

    def lock_file_ex(
        opened: Handle, flags: int, size_low: int, size_high: int, overlapped: object
    ) -> None:
        assert opened is handle
        assert (flags, size_low, size_high) == (0x2, 1, 0)
        calls.append("locked")

    def unlock_file_ex(
        opened: Handle, size_low: int, size_high: int, overlapped: object
    ) -> None:
        assert opened is handle
        assert (size_low, size_high) == (1, 0)
        calls.append("unlocked")

    files = SimpleNamespace(
        CreateFile=lambda *args: handle,
        LockFileEx=lock_file_ex,
        UnlockFileEx=unlock_file_ex,
    )

    @contextmanager
    def pinned_directory(path: Path):
        assert path == tmp_path
        yield (files, None, None), frozenset()

    monkeypatch.setattr(filesystem, "pinned_directory", pinned_directory)
    monkeypatch.setattr(
        filesystem, "_qualify", lambda *args, **kwargs: (0, None, None, None, None, 0, 0)
    )
    monkeypatch.setitem(sys.modules, "pywintypes", SimpleNamespace(OVERLAPPED=object))

    with file_io._windows_session_lock(tmp_path):
        assert calls == ["locked"]
    assert calls == ["locked", "unlocked", "close"]


def test_cli_failure_does_not_expose_artifacts_or_exception(monkeypatch, capsys):
    def fail(*arguments, **keywords):
        raise RuntimeError("TEST_ONLY secret-looking private_blob")

    monkeypatch.setattr(offline, "import_tpm_challenge", fail)
    assert offline.main(["import-challenge", "--input", "untrusted.json"]) == 1
    result = capsys.readouterr()
    assert result.out == ""
    assert result.err == "WINDOWS_OFFLINE_ENROLLMENT_FAILED\n"
