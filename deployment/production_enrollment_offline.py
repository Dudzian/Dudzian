"""File delivery to the fixed, independently deployed PDSA issuer.

Transport parsing and preview confer no authority. Consequential commands open
only the installed issuer, retain exact inputs before invoking its existing
verifiers, and retain exact replies. This module never signs a challenge or an
EK endorsement and never accepts a signer, trust root, endpoint or state path.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.enrollment_file_io import (
    artifact_directory_lock,
    ensure_artifact_directory,
    read_artifact_file,
    write_artifact_file,
)
from bot_core.licensing.enrollment_transport import ArtifactEnvelope
from bot_core.licensing.pdsa_enrollment_authorization import (
    issue_production_pdsa_enrollment_authorization,
    retry_production_pdsa_enrollment_authorization,
)
from bot_core.licensing.production_pre_enrollment import authenticate_production_pre_enrollment
from bot_core.licensing.production_tpm_custody import (
    _activation_and_request,
    verify_production_tpm_endorsement,
)
from deployment.production_enrollment_issuer import (
    ProductionEnrollmentIssuerContext,
    open_installed_production_enrollment_issuer,
    require_production_enrollment_issuer,
)

JOURNAL_SCHEMA = "CryptoHunter.PDSAOfflineExchangeJournal.v1"
AUDIT_SCHEMA = "CryptoHunter.PDSAOfflineExchangeAudit.v1"


class ProductionOfflineExchangeError(ValueError):
    """A transport correlation or installed journal check failed closed."""


def preview_offline_artifact(path: Path) -> dict[str, object]:
    """Strict syntax preview; no issuer bootstrap, signing, or filesystem writes."""
    envelope = ArtifactEnvelope.from_canonical_bytes(read_artifact_file(path))
    return {
        "kind": envelope.kind,
        "request_id": envelope.request_id,
        "parent_id": envelope.parent_id,
        "artifacts": {
            name: {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in sorted(envelope.artifacts.items())
        },
        "authorization": "NOT_ESTABLISHED",
    }


@contextmanager
def _journal(issuer: ProductionEnrollmentIssuerContext) -> Iterator[Path]:
    """Serialize exact recipes and replies under fixed protected issuer state."""
    require_production_enrollment_issuer(issuer)
    path = issuer.pdsa_store.path.parent / "offline"
    ensure_artifact_directory(path)
    with artifact_directory_lock(path):
        require_production_enrollment_issuer(issuer)
        yield path
        require_production_enrollment_issuer(issuer)


def _existing(path: Path) -> bytes | None:
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    raw: bytes = read_artifact_file(path)
    return raw


def _retain(path: Path, raw: bytes) -> None:
    retained = _existing(path)
    if retained is not None:
        if retained != raw:
            raise ProductionOfflineExchangeError("OFFLINE_RETAINED_INPUT_CONFLICT")
        return
    write_artifact_file(path, raw)


def _audit(journal: Path, *, operation: str, request_id: str, result: str) -> None:
    raw = canonical_json_bytes(
        {
            "schema_version": AUDIT_SCHEMA,
            "operation": operation,
            "request_id": request_id,
            "result": result,
            "recorded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "legal_enrollment": "NOT_PERFORMED",
        }
    )
    write_artifact_file(journal / f"audit-{uuid.uuid4().hex}.json", raw)


def _require_kind(envelope: ArtifactEnvelope, expected: str) -> None:
    if envelope.kind != expected:
        raise ProductionOfflineExchangeError("OFFLINE_ARTIFACT_STAGE_MISMATCH")


def issue_offline_tpm_challenge(
    request_path: Path,
    *,
    pdsa_challenge_path: Path,
    endorsement_path: Path,
    output_path: Path,
) -> ArtifactEnvelope:
    """Verify a device request and return the existing production TPM challenge.

    The PDSA challenge must already be live in this issuer's canonical store;
    an imported valid signature alone cannot establish its retained authority.
    """
    request_raw = read_artifact_file(request_path)
    request = ArtifactEnvelope.from_canonical_bytes(request_raw)
    _require_kind(request, "DEVICE_REQUEST")
    pdsa_raw = read_artifact_file(pdsa_challenge_path, maximum=16_384)
    endorsement_raw = read_artifact_file(endorsement_path, maximum=131_072)
    issuer = open_installed_production_enrollment_issuer()
    try:
        with _journal(issuer) as journal:
            try:
                selected = canonical_json_bytes(
                    {
                        "schema_version": JOURNAL_SCHEMA,
                        "request_id": request.request_id,
                        "pdsa_challenge_sha256": hashlib.sha256(pdsa_raw).hexdigest(),
                        "endorsement_sha256": hashlib.sha256(endorsement_raw).hexdigest(),
                    }
                )
                pdsa = issuer.pdsa_store.verify_issued(pdsa_raw, issuer.trust)
                endorsement = verify_production_tpm_endorsement(
                    endorsement_raw, context=issuer.trust
                )
                artifacts = request.artifacts
                _activation_and_request(
                    artifacts["activation_request"],
                    artifacts["tpm_request"],
                    issuer.trust,
                    endorsement,
                )
                # Authenticate public inputs before retaining a recipe. Retain
                # before issue so a crash cannot select different freshness.
                _retain(journal / f"{request.request_id}.device.json", request_raw)
                _retain(journal / f"{request.request_id}.selection.json", selected)
                challenge = issuer.tpm_store.issue(
                    artifacts["activation_request"],
                    artifacts["tpm_request"],
                    pdsa_challenge=pdsa,
                    pdsa_store=issuer.pdsa_store,
                    endorsement=endorsement,
                    context=issuer.trust,
                )
                response = ArtifactEnvelope.create(
                    "TPM_CHALLENGE",
                    artifacts
                    | {
                        "pdsa_challenge": pdsa_raw,
                        "endorsement": endorsement_raw,
                        "tpm_challenge": challenge.canonical_bytes,
                    },
                    parent_id=request.request_id,
                )
                _retain(journal / f"{request.request_id}.tpm.json", response.canonical_bytes)
                _audit(
                    journal, operation="ISSUE_TPM", request_id=request.request_id, result="RETAINED"
                )
                write_artifact_file(output_path, response.canonical_bytes)
                return response
            except (ValueError, OSError):
                _audit(
                    journal, operation="ISSUE_TPM", request_id=request.request_id, result="REJECTED"
                )
                raise
    finally:
        issuer.close()


def authorize_offline_pre_enrollment(request_path: Path, *, output_path: Path) -> ArtifactEnvelope:
    """Authenticate through existing PDSA and publish only committed signed bytes."""
    request_raw = read_artifact_file(request_path)
    request = ArtifactEnvelope.from_canonical_bytes(request_raw)
    _require_kind(request, "PRE_ENROLLMENT")
    artifacts = request.artifacts
    issuer = open_installed_production_enrollment_issuer()
    try:
        with _journal(issuer) as journal:
            try:
                # Reconstruct the exact earlier envelope instead of trusting an
                # unauthenticated parent identifier carried by removable media.
                challenge = ArtifactEnvelope.create(
                    "TPM_CHALLENGE",
                    {
                        name: artifacts[name]
                        for name in (
                            "activation_request",
                            "tpm_request",
                            "pdsa_challenge",
                            "endorsement",
                            "tpm_challenge",
                        )
                    },
                    parent_id=ArtifactEnvelope.create(
                        "DEVICE_REQUEST",
                        {name: artifacts[name] for name in ("activation_request", "tpm_request")},
                    ).request_id,
                )
                if request.parent_id != challenge.request_id:
                    raise ProductionOfflineExchangeError("OFFLINE_PARENT_BINDING_MISMATCH")
                retained_challenge = _existing(journal / f"{challenge.parent_id}.tpm.json")
                if retained_challenge != challenge.canonical_bytes:
                    raise ProductionOfflineExchangeError("OFFLINE_RETAINED_TPM_CHALLENGE_REQUIRED")
                request_digest = hashlib.sha256(artifacts["request"]).hexdigest()
                selection_path = journal / f"{request_digest}.authorization-input.json"
                selected_input = _existing(selection_path)
                if selected_input is not None and selected_input != request_raw:
                    raise ProductionOfflineExchangeError("OFFLINE_RETAINED_INPUT_CONFLICT")
                package = retry_production_pdsa_enrollment_authorization(
                    issuer,
                    request_raw=artifacts["request"],
                    challenge_raw=artifacts["pdsa_challenge"],
                )
                if package is not None:
                    if selected_input != request_raw:
                        raise ProductionOfflineExchangeError("OFFLINE_RETAINED_INPUT_REQUIRED")
                else:
                    accepted = authenticate_production_pre_enrollment(
                        artifacts["request"],
                        artifacts["request_signature"],
                        challenge_raw=artifacts["pdsa_challenge"],
                        challenge_store=issuer.pdsa_store,
                        activation_request_raw=artifacts["activation_request"],
                        tpm_request_raw=artifacts["tpm_request"],
                        tpm_challenge_raw=artifacts["tpm_challenge"],
                        tpm_response_raw=artifacts["tpm_response"],
                        pending=issuer.tpm_store,
                        endorsement_raw=artifacts["endorsement"],
                        custody_evidence_raw=artifacts["custody_evidence"],
                        context=issuer.trust,
                    )
                    # Freeze every public proof before issuing. Terminal retries
                    # must not accept altered signatures, response or custody
                    # bytes merely because the signed request is unchanged.
                    _retain(selection_path, request_raw)
                    _retain(journal / f"{request.request_id}.pre-enrollment.json", request_raw)
                    package = issue_production_pdsa_enrollment_authorization(accepted)
                response = ArtifactEnvelope.create(
                    "PDSA_PACKAGE", {"package": package}, parent_id=request.request_id
                )
                _retain(journal / f"{request.request_id}.package.json", response.canonical_bytes)
                _audit(
                    journal,
                    operation="AUTHORIZE_PRE_ENROLLMENT",
                    request_id=request.request_id,
                    result="COMMITTED_PACKAGE_RETAINED",
                )
                write_artifact_file(output_path, response.canonical_bytes)
                return response
            except (ValueError, OSError):
                _audit(
                    journal,
                    operation="AUTHORIZE_PRE_ENROLLMENT",
                    request_id=request.request_id,
                    result="REJECTED",
                )
                raise
    finally:
        issuer.close()


def export_retained_pdsa_challenge(challenge_id: str, *, output_path: Path) -> None:
    """Export an already-issued live signed challenge, without generating one."""
    issuer = open_installed_production_enrollment_issuer()
    try:
        with _journal(issuer) as journal:
            with issuer.pdsa_store._connect() as database:
                row = database.execute(
                    "SELECT challenge_raw FROM pdsa_challenges WHERE challenge_id=?",
                    (challenge_id,),
                ).fetchone()
            if row is None:
                raise ProductionOfflineExchangeError("OFFLINE_RETAINED_PDSA_CHALLENGE_REQUIRED")
            raw = bytes(row[0])
            issuer.pdsa_store.verify_issued(raw, issuer.trust)
            _audit(
                journal,
                operation="EXPORT_PDSA_CHALLENGE",
                request_id=hashlib.sha256(raw).hexdigest(),
                result="EXISTING_SIGNED_CHALLENGE_EXPORTED",
            )
            write_artifact_file(output_path, raw, maximum=16_384)
    finally:
        issuer.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("preview", help="syntax preview; establishes no authority")
    preview.add_argument("file", type=Path)
    export = commands.add_parser("export-challenge", help="export an existing retained challenge")
    export.add_argument("--challenge-id", required=True)
    export.add_argument("--output", type=Path, required=True)
    issue = commands.add_parser("issue-tpm", help="production-authorized TPM challenge issuance")
    issue.add_argument("file", type=Path)
    issue.add_argument("--pdsa-challenge", type=Path, required=True)
    issue.add_argument("--endorsement", type=Path, required=True)
    issue.add_argument("--output", type=Path, required=True)
    authorize = commands.add_parser("authorize", help="production-authorized PDSA package issuance")
    authorize.add_argument("file", type=Path)
    authorize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "preview":
            print(canonical_json_bytes(preview_offline_artifact(args.file)).decode("utf-8"))
        elif args.command == "export-challenge":
            export_retained_pdsa_challenge(args.challenge_id, output_path=args.output)
        elif args.command == "issue-tpm":
            issue_offline_tpm_challenge(
                args.file,
                pdsa_challenge_path=args.pdsa_challenge,
                endorsement_path=args.endorsement,
                output_path=args.output,
            )
        else:
            authorize_offline_pre_enrollment(args.file, output_path=args.output)
    except (ValueError, OSError) as exc:
        print(f"OFFLINE_EXCHANGE_REJECTED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
