"""Nonproduction offline APT signature boundary; no authority is enrolled here.

Successful cryptography and caller-policy matching remain separate from external
acceptance. There is deliberately no caller-controlled switch for approval.
V4a stays unchanged. This API reuses its parsers against the exact immutable
Release bytes passed to gpgv, and checks complete inert package inventories.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from deployment import stage9_observer_apt_v4a as v4a
from deployment.stage9_observer_contract import ContractRefusal
from deployment.stage9_observer_gpgv_v4b import (
    GpgvEvidenceV4b,
    GpgvLimitsV4b,
    GpgvRefusalV4b,
    GpgvToolV4b,
    verify_detached_v4b,
)
from deployment.stage9_observer_offline import Limits, _Budget, _InputReader, _validate_limits
from deployment.stage9_observer_package_v3 import _parse, _verify_archive
from deployment.stage9_observer_ubuntu import _pread

_DEFAULT_LIMITS = Limits()
_DEFAULT_GPGV_LIMITS = GpgvLimitsV4b()


@dataclass(frozen=True, slots=True)
class ExternalAuthorityReferenceV4b:
    """Unverified reference supplied by the caller, never an authority decision."""

    authority_id: str
    decision_reference: str
    delivery_mechanism: str


@dataclass(frozen=True, slots=True)
class SignerRuleV4b:
    signing_fingerprint: str
    primary_fingerprint: str
    valid_from: int
    valid_until: int
    revoked: bool = False


@dataclass(frozen=True, slots=True)
class TimeRulesV4b:
    """Local comparison rules; supplied time and references are unauthenticated."""

    evaluation_time: int
    maximum_signature_age_seconds: int
    maximum_future_skew_seconds: int
    freshness_authority_reference: str | None = None
    checkpoint_reference: str | None = None
    rollback_policy_reference: str | None = None


@dataclass(frozen=True, slots=True)
class TrustContextV4b:
    context_id: str
    context_version: int
    allowed_signers: tuple[SignerRuleV4b, ...]
    approved_algorithms: tuple[tuple[int, int], ...]
    subkey_policy: Literal["EXACT_SIGNING_PRIMARY_PAIRS"]
    repository_id: str
    snapshot_id: str
    release_sha256: str
    keyring_sha256: str
    valid_from: int
    valid_until: int
    revoked: bool
    time_rules: TimeRulesV4b
    external_authority: ExternalAuthorityReferenceV4b | None
    schema: Literal["stage9-apt-trust-context-v1"] = field(
        default="stage9-apt-trust-context-v1", init=False
    )


@dataclass(frozen=True, slots=True)
class SignatureDeclarationV4b:
    """Frozen local files and labels. These claims confer no external trust."""

    signature_path: str
    signature_sha256: str
    signature_size: int
    keyring_path: str
    keyring_sha256: str
    keyring_size: int
    repository_id: str
    snapshot_id: str


@dataclass(frozen=True, slots=True)
class AptResultV4b:
    cryptographic_signature: Literal["VALID", "UNVERIFIED"]
    signer_policy: Literal["MATCH", "UNVERIFIED"]
    reason_codes: tuple[str, ...]
    evidence: GpgvEvidenceV4b | None = None
    v4a: v4a.AptResultV4a | None = None
    release_sha256: str | None = None
    profile: Literal["stage9-offline-apt-signature-v4b"] = field(
        default="stage9-offline-apt-signature-v4b", init=False
    )
    external_authority_acceptance: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    source_authenticity: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    freshness: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    checkpoint: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    rollback: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    tool_acceptance: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    system_isolation: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    acceptance: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    stage_9: Literal["IN_PROGRESS"] = field(default="IN_PROGRESS", init=False)
    qualification: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    stage_3b: Literal["NOT AUTHORIZED"] = field(default="NOT AUTHORIZED", init=False)
    STAGE_3A_COMPLETE: Literal["NO"] = field(default="NO", init=False)
    PRODUCTION_QUALIFIED: Literal["NO"] = field(default="NO", init=False)
    production_status: Literal["PRODUCTION_NO_GO"] = field(default="PRODUCTION_NO_GO", init=False)
    refused: Literal[True] = field(default=True, init=False)


class _Refusal(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require(condition: bool, code: str = "TRUST_CONTEXT_REJECTED") -> None:
    if not condition:
        raise _Refusal(code)


def _label(value: str) -> None:
    _require(
        type(value) is str
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+#=-]{0,255}", value) is not None
    )


def _timestamp(value: int) -> None:
    _require(type(value) is int and 0 <= value <= 253402300799)


def _context(
    context: TrustContextV4b | None,
    selection: SignatureDeclarationV4b,
    declaration: v4a.AptDeclarationV4a,
) -> TrustContextV4b:
    _require(type(context) is TrustContextV4b, "TRUST_CONTEXT_MISSING")
    assert context is not None
    _label(context.context_id)
    _require(type(context.context_version) is int and 1 <= context.context_version <= 2147483647)
    _require(context.subkey_policy == "EXACT_SIGNING_PRIMARY_PAIRS")
    for value in (
        context.repository_id,
        context.snapshot_id,
        selection.repository_id,
        selection.snapshot_id,
    ):
        _label(value)
    _require(
        (context.repository_id, context.snapshot_id)
        == (selection.repository_id, selection.snapshot_id)
    )
    v4a._digest(context.release_sha256)
    v4a._digest(context.keyring_sha256)
    _require(context.release_sha256 == declaration.release_sha256)
    _require(context.keyring_sha256 == selection.keyring_sha256)
    _timestamp(context.valid_from)
    _timestamp(context.valid_until)
    _require(context.valid_from < context.valid_until)
    _require(type(context.revoked) is bool and not context.revoked, "TRUST_CONTEXT_REVOKED")
    _require(type(context.time_rules) is TimeRulesV4b)
    rules = context.time_rules
    _timestamp(rules.evaluation_time)
    _require(
        context.valid_from <= rules.evaluation_time < context.valid_until, "CONTEXT_TIME_REJECTED"
    )
    for seconds in (rules.maximum_signature_age_seconds, rules.maximum_future_skew_seconds):
        _require(type(seconds) is int and 0 <= seconds <= 31536000)
    for reference in (
        rules.freshness_authority_reference,
        rules.checkpoint_reference,
        rules.rollback_policy_reference,
    ):
        if reference is not None:
            _label(reference)
    if context.external_authority is not None:
        _require(type(context.external_authority) is ExternalAuthorityReferenceV4b)
        for value in (
            context.external_authority.authority_id,
            context.external_authority.decision_reference,
            context.external_authority.delivery_mechanism,
        ):
            _label(value)
    _require(type(context.allowed_signers) is tuple and 1 <= len(context.allowed_signers) <= 32)
    pairs: set[tuple[str, str]] = set()
    for signer in context.allowed_signers:
        _require(type(signer) is SignerRuleV4b)
        for fingerprint in (signer.signing_fingerprint, signer.primary_fingerprint):
            _require(
                type(fingerprint) is str and re.fullmatch(r"[0-9A-F]{40}", fingerprint) is not None
            )
            _require(fingerprint != "0" * 40)
        pair = (signer.signing_fingerprint, signer.primary_fingerprint)
        _require(pair not in pairs)
        pairs.add(pair)
        _timestamp(signer.valid_from)
        _timestamp(signer.valid_until)
        _require(signer.valid_from < signer.valid_until and type(signer.revoked) is bool)
    _require(
        type(context.approved_algorithms) is tuple and 1 <= len(context.approved_algorithms) <= 8
    )
    algorithms: set[tuple[int, int]] = set()
    for algorithm in context.approved_algorithms:
        _require(type(algorithm) is tuple and len(algorithm) == 2)
        _require(all(type(item) is int for item in algorithm))
        _require(algorithm in {(1, 8), (1, 10), (22, 8), (22, 10)})
        _require(algorithm not in algorithms)
        algorithms.add(algorithm)
    return context


def _signer(context: TrustContextV4b, evidence: GpgvEvidenceV4b) -> None:
    _require(
        (evidence.public_key_algorithm, evidence.hash_algorithm) in context.approved_algorithms,
        "SIGNATURE_ALGORITHM_REJECTED",
    )
    matches = [
        signer
        for signer in context.allowed_signers
        if (signer.signing_fingerprint, signer.primary_fingerprint)
        == (evidence.signing_fingerprint, evidence.primary_fingerprint)
    ]
    _require(len(matches) == 1, "SIGNER_REJECTED")
    signer = matches[0]
    _require(not signer.revoked, "SIGNER_REVOKED")
    timestamp = evidence.signature_timestamp
    now = context.time_rules.evaluation_time
    _require(
        evidence.signature_expiry == 0
        or timestamp < evidence.signature_expiry
        and now < evidence.signature_expiry,
        "SIGNATURE_EXPIRED",
    )
    _require(context.valid_from <= timestamp < context.valid_until, "SIGNATURE_TIME_REJECTED")
    _require(
        signer.valid_from <= timestamp < signer.valid_until
        and signer.valid_from <= now < signer.valid_until,
        "SIGNER_TIME_REJECTED",
    )
    _require(
        now - context.time_rules.maximum_signature_age_seconds
        <= timestamp
        <= now + context.time_rules.maximum_future_skew_seconds,
        "SIGNATURE_TIME_REJECTED",
    )


def _selection(
    selection: SignatureDeclarationV4b, declaration: v4a.AptDeclarationV4a, limits: Limits
) -> None:
    _require(type(selection) is SignatureDeclarationV4b, "SIGNATURE_DECLARATION_REJECTED")
    for path, size, digest, maximum in (
        (selection.signature_path, selection.signature_size, selection.signature_sha256, 65536),
        (selection.keyring_path, selection.keyring_size, selection.keyring_sha256, 4 * 1024 * 1024),
    ):
        v4a._path(path, limits)
        v4a._digest(digest)
        _require(
            type(size) is int and 0 < size <= min(maximum, limits.max_file_bytes), "LIMIT_EXCEEDED"
        )
    _require(
        len(
            {
                selection.signature_path,
                selection.keyring_path,
                declaration.release_path,
                declaration.packages_path,
            }
        )
        == 4,
        "FILE_POLICY_REJECTED",
    )


def _frozen_archive(reader: _InputReader, path: str, size: int, digest: str) -> int:
    """Hash and seal one bounded DEB copy; parse that same immutable descriptor."""
    if sys.platform != "linux":
        raise _Refusal("BYTE_BINDING_UNPROVEN")
    import fcntl

    budget = reader.budget
    budget.reserve(size + budget.limits.chunk_bytes)
    frozen = os.memfd_create("stage9-v4b-inert-deb", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        with reader.open_regular(path) as source:
            _require(os.fstat(source).st_size == size, "ARTIFACT_MISMATCH")
            hasher = hashlib.sha256()
            position = 0
            while position < size:
                budget.check()
                block = _pread(source, min(budget.limits.chunk_bytes, size - position), position)
                _require(bool(block), "PROVENANCE_INCOMPLETE")
                budget.read(len(block))
                hasher.update(block)
                _require(os.write(frozen, block) == len(block), "PROVENANCE_INCOMPLETE")
                position += len(block)
            _require(hasher.hexdigest() == digest, "ARTIFACT_MISMATCH")
        # Linux UAPI: seal/write/grow/shrink=0x0f, ADD_SEALS=1033, GET_SEALS=1034.
        # Some managed Python builds omit names; successful kernel readback is
        # mandatory. Unsupported sealing refuses, with no mutable fallback.
        seals = 0x0F
        fcntl.fcntl(frozen, 1033, seals)
        _require(fcntl.fcntl(frozen, 1034) == seals, "BYTE_BINDING_UNPROVEN")
        return frozen
    except BaseException:
        os.close(frozen)
        raise


def verify_apt_signature_v4b(
    input_directory: str | Path,
    *,
    declaration: v4a.AptDeclarationV4a,
    signature: SignatureDeclarationV4b,
    inventory_raw: bytes,
    context: TrustContextV4b | None,
    tool: GpgvToolV4b,
    limits: Limits = _DEFAULT_LIMITS,
    gpgv_limits: GpgvLimitsV4b = _DEFAULT_GPGV_LIMITS,
) -> AptResultV4b:
    """Complete local evidence or no evidence. External qualification stays blocked.

    References and labels are compared as caller claims, never resolved online.
    Original paths are monitored; immutable Release bytes and sealed DEB copies
    prevent a signature/check-chain byte swap within this invocation. This is
    not independent acceptance of the tool, kernel, loader or runtime isolation.
    """
    reader: _InputReader | None = None
    budget: _Budget | None = None
    evidence: GpgvEvidenceV4b | None = None
    chain: v4a.AptResultV4a | None = None
    reasons: set[str] = set()
    try:
        _validate_limits(limits)
        _require(sys.platform == "linux" and os.geteuid() != 0, "FILE_POLICY_REJECTED")
        budget = _Budget(limits)
        relative = v4a._selection(declaration, limits)
        _selection(signature, declaration, limits)
        checked_context = _context(context, signature, declaration)
        packages = _parse(inventory_raw, limits, budget)
        budget.reserve(limits.chunk_bytes * 8 + 1048576)
        expected = v4a._tree(
            (
                declaration.release_path,
                declaration.packages_path,
                signature.signature_path,
                signature.keyring_path,
                *(package.filename for package in packages),
            ),
            limits,
            budget,
        )
        reader = _InputReader(input_directory, budget)
        reader.baseline = reader.scan(expected)
        release = v4a._read_metadata(
            reader, declaration.release_path, declaration.release_size, declaration.release_sha256
        )
        signature_bytes = v4a._read_metadata(
            reader, signature.signature_path, signature.signature_size, signature.signature_sha256
        )
        keyring_bytes = v4a._read_metadata(
            reader, signature.keyring_path, signature.keyring_size, signature.keyring_sha256
        )
        evidence = verify_detached_v4b(
            release=release,
            signature=signature_bytes,
            keyring=keyring_bytes,
            tool=tool,
            limits=gpgv_limits,
        )
        budget.check()
        _signer(checked_context, evidence)
        v4a._release(release, relative, declaration, limits, budget)
        index = v4a._read_metadata(
            reader,
            declaration.packages_path,
            declaration.packages_size,
            declaration.packages_sha256,
        )
        decoded = v4a._decode_index(index, declaration, limits, budget)
        v4a._packages(decoded, packages, limits, budget)
        del release, signature_bytes, keyring_bytes, index, decoded
        staged = []
        for package in packages:
            frozen = _frozen_archive(reader, package.filename, package.size, package.sha256)
            try:
                staged.append(_verify_archive(frozen, package, limits, budget))
            finally:
                os.close(frozen)
        reader.finish(expected)
        budget.check()
        chain = v4a.AptResultV4a("MATCH", (), tuple(staged))
    except (ContractRefusal, GpgvRefusalV4b, _Refusal) as exc:
        reasons.add(exc.code)
    except (
        EOFError,
        zlib.error,
        UnicodeError,
        ValueError,
        OverflowError,
        RecursionError,
        TypeError,
        AttributeError,
    ):
        reasons.add("FORMAT_REJECTED")
    except MemoryError:
        reasons.add("LIMIT_EXCEEDED")
    except OSError:
        reasons.add("PROVENANCE_INCOMPLETE")
    finally:
        if reader is not None:
            try:
                reader.close()
            except OSError:
                reasons.add("PROVENANCE_INCOMPLETE")
        if budget is not None:
            try:
                budget.check()
            except ContractRefusal as exc:
                reasons.add(exc.code)
    if reasons or chain is None or evidence is None:
        return AptResultV4b(
            "UNVERIFIED", "UNVERIFIED", tuple(sorted(reasons or {"BYTE_BINDING_UNPROVEN"}))
        )
    return AptResultV4b(
        "VALID",
        "MATCH",
        (
            "EXTERNAL_AUTHORITY_UNVERIFIED",
            "FRESHNESS_BLOCKED",
            "CHECKPOINT_BLOCKED",
            "ROLLBACK_BLOCKED",
            "TOOL_ACCEPTANCE_UNVERIFIED",
            "SYSTEM_ISOLATION_UNVERIFIED",
        ),
        evidence,
        chain,
        declaration.release_sha256,
    )
