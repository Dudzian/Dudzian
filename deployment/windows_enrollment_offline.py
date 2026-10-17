"""Installed Windows offline transport; importing bytes never creates membership.

The append-only journal retains exact artifacts and local key identity. Production
TPM response creation is a separate, qualified hardware operation: the historical
TEST_ONLY activation bridge is deliberately not called here.
"""

from __future__ import annotations

import argparse
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, cast

from bot_core.licensing.activation_request import ActivationRequestV1
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.licensing.device_enrollment import TPMPublicProjectionV1
from bot_core.licensing.enrollment_file_io import read_artifact_file, write_artifact_file
from bot_core.licensing.enrollment_transport import ArtifactEnvelope
from bot_core.licensing.lppi_package_acceptance import accept_production_lppi_package
from bot_core.licensing.pdsa_enrollment_challenge import verify_signed_production_pdsa_challenge
from bot_core.licensing.pre_enrollment import PreEnrollmentRequestV1, public_key_fingerprint
from bot_core.licensing.production_tpm_custody import (
    K_PSA_PROFILE,
    PROJECTION_PROFILE,
    PROJECTION_SOURCE_PROFILE,
    _activation_and_request,
    parse_production_ecc_public,
    verify_production_tpm_endorsement,
)
from bot_core.licensing.tpm_attestation import (
    TPMEnrollmentChallengeResponseV1,
    TPMEnrollmentChallengeV1,
    TPMEnrollmentRequestV1,
)
from deployment import windows_cng_custody_bridge as bridge
from deployment.platforms.windows import production_trust_package_path, resolve_paths
from deployment.windows_cng_pre_enrollment import KEY_NAME, WindowsCNGPreEnrollmentKey
from deployment.windows_production_pre_enrollment import prepare_installed_production_request
from deployment.windows_stage9_production_trust import (
    CEREMONY_ID,
    load_production_trust,
    require_current_production_trust_context,
)

_ID = re.compile(r"[0-9a-f]{64}\Z")
_AK = re.compile(r"[A-Za-z0-9._\\:-]{1,256}\Z")
_LOCAL_FIELDS = {
    "schema_version",
    "device_request_id",
    "ak_key_name",
    "pre_enrollment_key_unique_name",
    "pre_enrollment_public_key_fingerprint_sha256",
    "release_policy_digest_sha256",
    "release_policy_generation",
}


class WindowsOfflineEnrollmentError(ValueError):
    """An artifact, local identity or enrollment stage does not match."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _trust():
    return require_current_production_trust_context(
        load_production_trust(production_trust_package_path(CEREMONY_ID))
    )


def _root() -> Path:
    return cast(Path, resolve_paths().state / "OfflineEnrollment")


def _session(device_request_id: object) -> Path:
    if type(device_request_id) is not str or _ID.fullmatch(device_request_id) is None:
        raise WindowsOfflineEnrollmentError("INVALID_OFFLINE_DEVICE_REQUEST_ID")
    return _root() / device_request_id


@contextmanager
def _locked(directory: Path, *, create: bool = False) -> Iterator[None]:
    # Reuse the qualified filesystem boundary; its lock is separate from the
    # fixed CNG identity lock and never authorizes a native operation itself.
    from bot_core.licensing.enrollment_file_io import (
        artifact_directory_lock,
        ensure_artifact_directory,
    )

    if create:
        ensure_artifact_directory(directory)
    with artifact_directory_lock(directory):
        yield


def _read(path: Path) -> ArtifactEnvelope:
    return ArtifactEnvelope.from_canonical_bytes(read_artifact_file(path))


def _stage(directory: Path, name: str, kind: str) -> ArtifactEnvelope:
    result = _read(directory / (name + ".json"))
    if result.kind != kind:
        raise WindowsOfflineEnrollmentError("OFFLINE_STAGE_TYPE_MISMATCH")
    return result


def _retain(directory: Path, name: str, envelope: ArtifactEnvelope) -> None:
    # The deterministic stage slot is the commit point. Repeated identical
    # imports restore the content-addressed audit copy after interrupted writes.
    write_artifact_file(directory / (name + ".json"), envelope.canonical_bytes)
    write_artifact_file(directory / (envelope.request_id + ".json"), envelope.canonical_bytes)


@contextmanager
def _unenrolled_state() -> Iterator[None]:
    # Hold the successor lifecycle's own lock across validation and native
    # acceptance. The offline journal lock alone cannot exclude that lifecycle.
    # Creating its empty directory/lock does not create an authority identity.
    from deployment.windows_production_lppi_authority import _locked_state

    with _locked_state() as path:
        try:
            read_artifact_file(path)
        except FileNotFoundError:
            pass
        else:
            raise WindowsOfflineEnrollmentError(
                "EXISTING_LPPI_AUTHORITY_REQUIRES_SEPARATE_LIFECYCLE"
            )
        yield


def _assert_not_completed() -> None:
    try:
        read_artifact_file(_root() / "accepted-package.json")
    except FileNotFoundError:
        return
    raise WindowsOfflineEnrollmentError("OFFLINE_ENROLLMENT_ALREADY_ACCEPTED")


def _projection(artifacts: dict[str, bytes], trust: Any) -> TPMPublicProjectionV1:
    activation = ActivationRequestV1.from_mapping(parse_canonical(artifacts["activation_request"]))
    request = TPMEnrollmentRequestV1.from_canonical_bytes(artifacts["tpm_request"])
    projection = TPMPublicProjectionV1.verify(request.document["public_projection"])
    reproduced = TPMEnrollmentRequestV1.create(
        activation_request=activation,
        public_projection=projection,
        client_nonce=bytes.fromhex(request.document["client_nonce_hex"]),
    )
    public = projection.document
    if (
        reproduced.canonical_bytes != request.canonical_bytes
        or activation.document["release"]
        != {
            "release_policy_digest": trust.release_payload_digest,
            "release_policy_version": trust.release_version,
        }
        or public["evidence_profile"] != PROJECTION_PROFILE
        or public["source"] != {"substrate": "Windows-TBS", "profile": PROJECTION_SOURCE_PROFILE}
        or public["k_psa"]["algorithm_profile"] != K_PSA_PROFILE
    ):
        raise WindowsOfflineEnrollmentError("OFFLINE_LOCAL_DEVICE_OR_RELEASE_MISMATCH")
    for role in ("ek", "ak", "k_psa"):
        parse_production_ecc_public(bytes.fromhex(public[role]["public_area"]["hex"]), role=role)
    return projection


def _require_local_ak(
    key: WindowsCNGPreEnrollmentKey, projection: TPMPublicProjectionV1, name: str
):
    if type(name) is not str or _AK.fullmatch(name) is None or name == KEY_NAME:
        raise WindowsOfflineEnrollmentError("INVALID_RETAINED_AK_KEY_NAME")
    native = key._native
    context = bridge._platform_handle(native, key._provider, provider=True)
    tbs = bridge._load_tbs_native()
    if type(tbs) is not bridge._TBSCustodyAPI:
        raise WindowsOfflineEnrollmentError("EXACT_NATIVE_TBS_BOUNDARY_REQUIRED")
    tbs.require_tpm20()
    handle = bridge._open_retained_ak(native, key._provider, name)
    try:
        bridge._require_ak_profile(native, handle, name)
        raw, actual_name, _qualified_name = bridge._read_public(
            tbs, context, bridge._platform_handle(native, handle, provider=False)
        )
        expected = projection.document["ak"]
        if (
            raw.hex() != expected["public_area"]["hex"]
            or actual_name.hex() != expected["name"]
            or parse_production_ecc_public(raw, role="ak").name != actual_name
        ):
            raise WindowsOfflineEnrollmentError("OFFLINE_LIVE_TARGET_AK_MISMATCH")
    finally:
        native.free(handle)


def _identity(key: WindowsCNGPreEnrollmentKey, *, device_id: str, ak_name: str, trust: Any):
    evidence = key.public_evidence
    return {
        "schema_version": "WindowsOfflineEnrollmentLocalIdentityV1",
        "device_request_id": device_id,
        "ak_key_name": ak_name,
        "pre_enrollment_key_unique_name": evidence["cng_key_unique_name"],
        "pre_enrollment_public_key_fingerprint_sha256": public_key_fingerprint(
            key.public_key_bytes
        ),
        "release_policy_digest_sha256": trust.release_payload_digest,
        "release_policy_generation": trust.release_version,
    }


def _local(directory: Path, device: ArtifactEnvelope, key: WindowsCNGPreEnrollmentKey, trust: Any):
    identity = parse_canonical(read_artifact_file(directory / "local-identity.json", maximum=4096))
    if (
        type(identity) is not dict
        or set(identity) != _LOCAL_FIELDS
        or type(identity["release_policy_generation"]) is not int
        or device.request_id != directory.name
    ):
        raise WindowsOfflineEnrollmentError("INVALID_OFFLINE_LOCAL_IDENTITY")
    expected = _identity(
        key, device_id=device.request_id, ak_name=identity["ak_key_name"], trust=trust
    )
    if identity != expected:
        raise WindowsOfflineEnrollmentError("OFFLINE_LOCAL_IDENTITY_CHANGED")
    _require_local_ak(key, _projection(device.artifacts, trust), identity["ak_key_name"])
    return identity


def _validate_challenge(envelope: ArtifactEnvelope, device: ArtifactEnvelope, trust: Any) -> None:
    if envelope.kind != "TPM_CHALLENGE" or envelope.parent_id != device.request_id:
        raise WindowsOfflineEnrollmentError("OFFLINE_CHALLENGE_PARENT_MISMATCH")
    artifacts = envelope.artifacts
    if any(artifacts[name] != raw for name, raw in device.artifacts.items()):
        raise WindowsOfflineEnrollmentError("OFFLINE_CHALLENGE_DEVICE_MISMATCH")
    pdsa = verify_signed_production_pdsa_challenge(artifacts["pdsa_challenge"], trust)
    endorsement = verify_production_tpm_endorsement(artifacts["endorsement"], context=trust)
    _activation, request, projection = _activation_and_request(
        artifacts["activation_request"], artifacts["tpm_request"], trust, endorsement
    )
    challenge = TPMEnrollmentChallengeV1.from_canonical_bytes(artifacts["tpm_challenge"]).document
    expiry = datetime.strptime(challenge["expires_at_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    if (
        challenge["request_id"] != request.document["request_id"]
        or challenge["public_projection_id"] != projection.evidence_reference
        or not _utc_now() < expiry
        or challenge["expires_at_utc"]
        > min(
            pdsa.document["payload"]["expires_at_utc"],
            endorsement.document["payload"]["expires_at_utc"],
        )
    ):
        raise WindowsOfflineEnrollmentError("OFFLINE_TPM_CHALLENGE_BINDING_OR_EXPIRY_MISMATCH")


def export_device_request(
    *, activation_request_raw: bytes, tpm_request_raw: bytes, ak_key_name: str, output: Path
) -> ArtifactEnvelope:
    envelope = ArtifactEnvelope.create(
        "DEVICE_REQUEST",
        {"activation_request": activation_request_raw, "tpm_request": tpm_request_raw},
    )
    directory = _session(envelope.request_id)
    with _locked(_root(), create=True), _unenrolled_state(), _locked(directory, create=True):
        _assert_not_completed()
        trust = _trust()
        projection = _projection(envelope.artifacts, trust)
        with WindowsCNGPreEnrollmentKey.open_or_create(
            resolve_paths().state / "PreEnrollment"
        ) as key:
            _require_local_ak(key, projection, ak_key_name)
            identity = _identity(
                key, device_id=envelope.request_id, ak_name=ak_key_name, trust=trust
            )
            write_artifact_file(directory / "local-identity.json", canonical_json_bytes(identity))
            _retain(directory, "device-request", envelope)
        write_artifact_file(output, envelope.canonical_bytes)
    return envelope


def import_tpm_challenge(source: Path) -> ArtifactEnvelope:
    envelope = _read(source)
    if envelope.kind != "TPM_CHALLENGE":
        raise WindowsOfflineEnrollmentError("OFFLINE_STAGE_TYPE_MISMATCH")
    directory = _session(envelope.parent_id)
    with _locked(_root()), _unenrolled_state(), _locked(directory):
        _assert_not_completed()
        device = _stage(directory, "device-request", "DEVICE_REQUEST")
        trust = _trust()
        with WindowsCNGPreEnrollmentKey.open_existing(
            resolve_paths().state / "PreEnrollment"
        ) as key:
            _local(directory, device, key, trust)
            _validate_challenge(envelope, device, trust)
            _retain(directory, "tpm-challenge", envelope)
    return envelope


def prepare_pre_enrollment(
    *, device_request_id: str, tpm_response_raw: bytes, output: Path
) -> ArtifactEnvelope:
    directory = _session(device_request_id)
    with _locked(_root()), _unenrolled_state(), _locked(directory):
        _assert_not_completed()
        device = _stage(directory, "device-request", "DEVICE_REQUEST")
        challenge = _stage(directory, "tpm-challenge", "TPM_CHALLENGE")
        trust = _trust()
        with WindowsCNGPreEnrollmentKey.open_existing(
            resolve_paths().state / "PreEnrollment"
        ) as key:
            identity = _local(directory, device, key, trust)
        _validate_challenge(challenge, device, trust)
        response = TPMEnrollmentChallengeResponseV1.from_canonical_bytes(tpm_response_raw).document
        tpm_challenge = TPMEnrollmentChallengeV1.from_canonical_bytes(
            challenge.artifacts["tpm_challenge"]
        ).document
        if any(
            response[field] != tpm_challenge[field]
            for field in ("request_id", "challenge_id", "issuer_nonce_hex")
        ):
            raise WindowsOfflineEnrollmentError("OFFLINE_TPM_RESPONSE_BINDING_MISMATCH")
        try:
            envelope = _stage(directory, "pre-enrollment", "PRE_ENROLLMENT")
        except FileNotFoundError:
            artifacts = challenge.artifacts
            prepared = prepare_installed_production_request(
                challenge_raw=artifacts["pdsa_challenge"],
                activation_request_raw=artifacts["activation_request"],
                tpm_request_raw=artifacts["tpm_request"],
                tpm_challenge_raw=artifacts["tpm_challenge"],
                tpm_response_raw=tpm_response_raw,
                ak_key_name=identity["ak_key_name"],
            )
            envelope = ArtifactEnvelope.create(
                "PRE_ENROLLMENT",
                artifacts
                | {
                    "tpm_response": tpm_response_raw,
                    "request": prepared.request_raw,
                    "request_signature": prepared.signature,
                    "custody_evidence": prepared.custody_evidence_raw,
                },
                parent_id=challenge.request_id,
            )
        if (
            envelope.parent_id != challenge.request_id
            or envelope.artifacts["tpm_response"] != tpm_response_raw
        ):
            raise WindowsOfflineEnrollmentError("OFFLINE_PRE_ENROLLMENT_RETRY_CONFLICT")
        if any(envelope.artifacts[name] != raw for name, raw in challenge.artifacts.items()):
            raise WindowsOfflineEnrollmentError("OFFLINE_PRE_ENROLLMENT_RETRY_CONFLICT")
        request = PreEnrollmentRequestV1.from_canonical_bytes(envelope.artifacts["request"])
        request.require_production_trust_binding(trust)
        request.verify_signature(envelope.artifacts["request_signature"])
        if (
            request.document["pre_enrollment_public_key_fingerprint_sha256"]
            != identity["pre_enrollment_public_key_fingerprint_sha256"]
        ):
            raise WindowsOfflineEnrollmentError("OFFLINE_LOCAL_IDENTITY_CHANGED")
        _retain(directory, "pre-enrollment", envelope)
        write_artifact_file(output, envelope.canonical_bytes)
    return envelope


def import_pdsa_package(source: Path, *, device_request_id: str) -> ArtifactEnvelope:
    envelope = _read(source)
    directory = _session(device_request_id)
    with _locked(_root()), _unenrolled_state(), _locked(directory):
        device = _stage(directory, "device-request", "DEVICE_REQUEST")
        challenge = _stage(directory, "tpm-challenge", "TPM_CHALLENGE")
        prepared = _stage(directory, "pre-enrollment", "PRE_ENROLLMENT")
        if envelope.kind != "PDSA_PACKAGE" or envelope.parent_id != prepared.request_id:
            raise WindowsOfflineEnrollmentError("OFFLINE_PACKAGE_PARENT_MISMATCH")
        trust = _trust()
        _validate_challenge(challenge, device, trust)
        if prepared.parent_id != challenge.request_id or any(
            prepared.artifacts[name] != raw for name, raw in challenge.artifacts.items()
        ):
            raise WindowsOfflineEnrollmentError("OFFLINE_PRE_ENROLLMENT_PARENT_MISMATCH")
        try:
            completed = read_artifact_file(_root() / "accepted-package.json")
        except FileNotFoundError:
            completed = None
        if completed is not None and completed != envelope.canonical_bytes:
            raise WindowsOfflineEnrollmentError("OFFLINE_ENROLLMENT_ALREADY_ACCEPTED")
        artifacts = prepared.artifacts
        with WindowsCNGPreEnrollmentKey.open_existing(
            resolve_paths().state / "PreEnrollment"
        ) as key:
            identity = _local(directory, device, key, trust)
            accept_production_lppi_package(
                envelope.artifacts["package"],
                context=trust,
                key=key,
                request_raw=artifacts["request"],
                request_signature=artifacts["request_signature"],
                challenge_raw=artifacts["pdsa_challenge"],
                activation_request_raw=artifacts["activation_request"],
                tpm_request_raw=artifacts["tpm_request"],
                tpm_challenge_raw=artifacts["tpm_challenge"],
                tpm_response_raw=artifacts["tpm_response"],
                endorsement_raw=artifacts["endorsement"],
                custody_evidence_raw=artifacts["custody_evidence"],
                ak_key_name=identity["ak_key_name"],
            )
            # This is transport acceptance only. No successor creation or
            # membership API is called and no verifier capability is persisted.
            write_artifact_file(_root() / "accepted-package.json", envelope.canonical_bytes)
            _retain(directory, "pdsa-package", envelope)
    return envelope


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start")
    start.add_argument("--activation-request", type=Path, required=True)
    start.add_argument("--tpm-request", type=Path, required=True)
    start.add_argument("--ak-key-name", required=True)
    start.add_argument("--output", type=Path, required=True)
    challenge = commands.add_parser("import-challenge")
    challenge.add_argument("--input", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--device-request-id", required=True)
    prepare.add_argument("--tpm-response", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    package = commands.add_parser("import-package")
    package.add_argument("--device-request-id", required=True)
    package.add_argument("--input", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "start":
            envelope = export_device_request(
                activation_request_raw=read_artifact_file(arguments.activation_request),
                tpm_request_raw=read_artifact_file(arguments.tpm_request),
                ak_key_name=arguments.ak_key_name,
                output=arguments.output,
            )
        elif arguments.command == "import-challenge":
            envelope = import_tpm_challenge(arguments.input)
        elif arguments.command == "prepare":
            envelope = prepare_pre_enrollment(
                device_request_id=arguments.device_request_id,
                tpm_response_raw=read_artifact_file(arguments.tpm_response),
                output=arguments.output,
            )
        else:
            envelope = import_pdsa_package(
                arguments.input, device_request_id=arguments.device_request_id
            )
    except (OSError, RuntimeError, ValueError, TypeError):
        print("WINDOWS_OFFLINE_ENROLLMENT_FAILED", file=sys.stderr)
        return 1
    print(envelope.request_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
