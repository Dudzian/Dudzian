"""Complete synthetic V4b boundaries, atomic refusals and trust separation."""

from __future__ import annotations

import errno
import hashlib
import os
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterator

import pytest

from deployment import stage9_observer_apt_v4b as apt
from deployment.stage9_observer_contract import ContractRefusal
from deployment.stage9_observer_gpgv_v4b import (
    GpgvEvidenceV4b,
    GpgvRefusalV4b,
    GpgvToolV4b,
)
from deployment.stage9_observer_offline import Limits
from tests.deployment.test_stage9_observer_apt_v4a import AptCase, _build
from tests.deployment.test_stage9_observer_apt_v4b_crypto import (
    TestSignerV4b,
    local_gpgv_tool,
    runtime_test_signer,
)
from tests.deployment.test_stage9_observer_package_v3 import build_package_case

fcntl = pytest.importorskip("fcntl", reason="V4b sealed descriptors require Linux fcntl")
pytestmark = [
    pytest.mark.security,
    pytest.mark.unit,
    pytest.mark.skipif(
        sys.platform != "linux", reason="V4b descriptor boundary supports Linux only"
    ),
]

_TIMESTAMP = 1794657600
_SIGNING = "A" * 40
_PRIMARY = "B" * 40
_EVIDENCE = GpgvEvidenceV4b(_SIGNING, _PRIMARY, 22, 8, _TIMESTAMP, 0)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass
class BoundaryCase:
    chain: AptCase
    signature: apt.SignatureDeclarationV4b
    context: apt.TrustContextV4b
    declaration: Any

    def verify(self, **kwargs: Any) -> apt.AptResultV4b:
        selected = {
            "declaration": self.declaration,
            "signature": self.signature,
            "context": self.context,
            "inventory_raw": self.chain.inventory_raw,
            "tool": GpgvToolV4b("/test-only/gpgv", "1" * 64),
        }
        selected.update(kwargs)
        return apt.verify_apt_signature_v4b(self.chain.directory, **selected)


def _boundary(directory: Path, **kwargs: Any) -> BoundaryCase:
    chain = _build(directory, **kwargs)
    signature_bytes = b"SYNTHETIC TEST SIGNATURE; mocked crypto in policy unit tests\n"
    keyring_bytes = b"SYNTHETIC TEST KEYRING; never a production root\n"
    (directory / "Release.gpg").write_bytes(signature_bytes)
    (directory / "test-keyring.gpg").write_bytes(keyring_bytes)
    signature = apt.SignatureDeclarationV4b(
        "Release.gpg",
        _sha(signature_bytes),
        len(signature_bytes),
        "test-keyring.gpg",
        _sha(keyring_bytes),
        len(keyring_bytes),
        "TEST-repository",
        "TEST-snapshot",
    )
    context = apt.TrustContextV4b(
        "TEST-context",
        1,
        (apt.SignerRuleV4b(_SIGNING, _PRIMARY, _TIMESTAMP - 3600, _TIMESTAMP + 3600),),
        ((22, 8),),
        "EXACT_SIGNING_PRIMARY_PAIRS",
        signature.repository_id,
        signature.snapshot_id,
        chain.declaration.release_sha256,
        signature.keyring_sha256,
        _TIMESTAMP - 3600,
        _TIMESTAMP + 3600,
        False,
        apt.TimeRulesV4b(_TIMESTAMP, 60, 0),
        None,
    )
    return BoundaryCase(chain, signature, context, chain.declaration)


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BoundaryCase:
    if sys.platform != "linux":
        pytest.skip("V4b descriptor boundary supports Linux only")
    monkeypatch.setattr(apt, "verify_detached_v4b", lambda **kwargs: _EVIDENCE)
    return _boundary(tmp_path / "apt")


def _blocked(result: apt.AptResultV4b) -> None:
    assert result.external_authority_acceptance == "UNVERIFIED"
    assert result.source_authenticity == "UNVERIFIED"
    assert result.freshness == result.checkpoint == result.rollback == "BLOCKED"
    assert result.tool_acceptance == result.system_isolation == "UNVERIFIED"
    assert result.acceptance == "BLOCKED"
    assert result.stage_9 == "IN_PROGRESS"
    assert result.qualification == "BLOCKED"
    assert result.stage_3b == "NOT AUTHORIZED"
    assert result.STAGE_3A_COMPLETE == result.PRODUCTION_QUALIFIED == "NO"
    assert result.production_status == "PRODUCTION_NO_GO"
    assert result.refused is True


def _refused(result: apt.AptResultV4b, code: str | None = None) -> None:
    assert result.cryptographic_signature == "UNVERIFIED"
    assert result.signer_policy == "UNVERIFIED"
    assert result.evidence is None
    assert result.v4a is None
    assert result.release_sha256 is None
    assert result.reason_codes
    if code is not None:
        assert code in result.reason_codes
    _blocked(result)


def test_complete_local_evidence_cannot_approve_any_external_boundary(case: BoundaryCase) -> None:
    result = case.verify()
    assert result.cryptographic_signature == "VALID", result.reason_codes
    assert result.signer_policy == "MATCH"
    assert result.evidence == _EVIDENCE
    assert result.release_sha256 == case.declaration.release_sha256
    assert result.v4a is not None
    assert result.v4a.consistency == "MATCH"
    assert len(result.v4a.packages) == 1
    assert result.v4a.signature_verification == "NOT_PERFORMED"
    assert result.v4a.source_authenticity == result.v4a.acceptance == "UNVERIFIED"
    _blocked(result)


@pytest.mark.parametrize("index_format", ["Packages", "Packages.gz"])
@pytest.mark.parametrize("variant", ["raw", "gz", "zst"])
def test_complete_chain_reuses_existing_supported_v4a_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index_format: str, variant: str
) -> None:
    monkeypatch.setattr(apt, "verify_detached_v4b", lambda **kwargs: _EVIDENCE)
    case = _boundary(tmp_path / "apt", index_format=index_format, variant=variant)
    result = case.verify()
    assert result.cryptographic_signature == "VALID", result.reason_codes
    _blocked(result)


def test_missing_context_refuses_before_crypto(
    case: BoundaryCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(**kwargs: Any) -> Any:
        pytest.fail("No crypto execution without a structurally valid context")

    monkeypatch.setattr(apt, "verify_detached_v4b", unexpected)
    _refused(case.verify(context=None), "TRUST_CONTEXT_MISSING")


def test_caller_references_cannot_accept_their_own_origin(case: BoundaryCase) -> None:
    context = replace(
        case.context,
        external_authority=apt.ExternalAuthorityReferenceV4b(
            "TEST-only-unverified-claim", "TEST-decision", "TEST-caller-document"
        ),
        time_rules=replace(
            case.context.time_rules,
            freshness_authority_reference="TEST-caller-freshness",
            checkpoint_reference="TEST-caller-checkpoint",
            rollback_policy_reference="TEST-caller-rollback",
        ),
    )
    result = case.verify(context=context)
    assert result.cryptographic_signature == "VALID", result.reason_codes
    _blocked(result)
    assert all(
        not item.init
        for item in fields(result)
        if item.name
        in {
            "external_authority_acceptance",
            "acceptance",
            "source_authenticity",
            "freshness",
            "checkpoint",
            "rollback",
            "tool_acceptance",
            "system_isolation",
        }
    )


@pytest.mark.parametrize("name", ["test-keyring.gpg", "Release.gpg", "Release", "Packages"])
def test_missing_local_file_discards_the_whole_result(case: BoundaryCase, name: str) -> None:
    (case.chain.directory / name).unlink()
    _refused(case.verify())


@pytest.mark.parametrize("name", ["test-keyring.gpg", "Release.gpg", "Release", "Packages"])
def test_frozen_material_digest_is_required(case: BoundaryCase, name: str) -> None:
    path = case.chain.directory / name
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    _refused(case.verify(), "ARTIFACT_MISMATCH")


@pytest.mark.parametrize(
    "field", ["repository_id", "snapshot_id", "release_sha256", "keyring_sha256"]
)
def test_context_must_match_the_exact_selection(case: BoundaryCase, field: str) -> None:
    replacement = "1" * 64 if field.endswith("sha256") else "different-TEST-claim"
    _refused(case.verify(context=replace(case.context, **{field: replacement})))


@pytest.mark.parametrize(
    "kind",
    [
        "unknown-signer",
        "wrong-subkey",
        "wrong-primary",
        "algorithm",
        "revoked",
        "expired",
        "not-yet-valid",
    ],
)
def test_signer_policy_rejects_unapproved_pairs_and_rules(case: BoundaryCase, kind: str) -> None:
    signer = case.context.allowed_signers[0]
    changes: dict[str, Any] = {}
    expected = "SIGNER_REJECTED"
    if kind == "unknown-signer":
        signer = replace(signer, signing_fingerprint="C" * 40, primary_fingerprint="D" * 40)
    elif kind == "wrong-subkey":
        signer = replace(signer, signing_fingerprint="C" * 40)
    elif kind == "wrong-primary":
        signer = replace(signer, primary_fingerprint="C" * 40)
    elif kind == "algorithm":
        changes["approved_algorithms"] = ((1, 8),)
        expected = "SIGNATURE_ALGORITHM_REJECTED"
    elif kind == "revoked":
        signer = replace(signer, revoked=True)
        expected = "SIGNER_REVOKED"
    elif kind == "expired":
        signer = replace(signer, valid_until=_TIMESTAMP)
        expected = "SIGNER_TIME_REJECTED"
    else:
        signer = replace(signer, valid_from=_TIMESTAMP + 1)
        expected = "SIGNER_TIME_REJECTED"
    changes["allowed_signers"] = (signer,)
    _refused(case.verify(context=replace(case.context, **changes)), expected)


@pytest.mark.parametrize(
    "changes",
    [
        {"revoked": True},
        {"valid_until": _TIMESTAMP},
        {"valid_from": _TIMESTAMP + 1},
        {"context_version": 0},
        {"context_version": True},
        {"context_id": "../invalid TEST"},
        {"subkey_policy": "ALLOW_ANY_SUBKEY"},
        {"allowed_signers": ()},
        {"allowed_signers": []},
        {"allowed_signers": (None,)},
        {"approved_algorithms": ()},
        {"approved_algorithms": ((22, 2),)},
        {"approved_algorithms": ((True, 8),)},
        {"approved_algorithms": ((22, 8), (22, 8))},
        {"approved_algorithms": ([22, 8],)},
        {"time_rules": None},
        {"external_authority": "caller-approved"},
    ],
)
def test_malformed_or_unusable_context_fails_closed(
    case: BoundaryCase, changes: dict[str, Any]
) -> None:
    _refused(case.verify(context=replace(case.context, **changes)))


@pytest.mark.parametrize(
    "changes",
    [
        {"evaluation_time": True},
        {"maximum_signature_age_seconds": -1},
        {"maximum_signature_age_seconds": float("nan")},
        {"maximum_future_skew_seconds": float("inf")},
        {"freshness_authority_reference": "bad reference with spaces"},
    ],
)
def test_invalid_time_claims_fail_closed(case: BoundaryCase, changes: dict[str, Any]) -> None:
    context = replace(case.context, time_rules=replace(case.context.time_rules, **changes))
    _refused(case.verify(context=context))


@pytest.mark.parametrize(
    "timestamp", [_TIMESTAMP - 61, _TIMESTAMP + 1, _TIMESTAMP - 4000, _TIMESTAMP + 4000]
)
def test_signature_time_enforces_all_local_rules(
    case: BoundaryCase, monkeypatch: pytest.MonkeyPatch, timestamp: int
) -> None:
    monkeypatch.setattr(
        apt,
        "verify_detached_v4b",
        lambda **kwargs: replace(_EVIDENCE, signature_timestamp=timestamp),
    )
    _refused(case.verify(), "SIGNATURE_TIME_REJECTED")


def test_signature_expiry_cannot_be_ignored_by_policy_time(
    case: BoundaryCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        apt, "verify_detached_v4b", lambda **kwargs: replace(_EVIDENCE, signature_expiry=_TIMESTAMP)
    )
    _refused(case.verify(), "SIGNATURE_EXPIRED")


@pytest.mark.parametrize(
    "code",
    [
        "GPGV_STATUS_REJECTED",
        "GPGV_PROCESS_FAILED",
        "GPGV_TIMEOUT",
        "GPGV_OUTPUT_LIMIT",
        "GPGV_INPUT_CHANGED",
        "GPGV_UNSUPPORTED_ALGORITHM",
    ],
)
def test_crypto_refusal_never_leaks_a_partial_chain(
    case: BoundaryCase, monkeypatch: pytest.MonkeyPatch, code: str
) -> None:
    def refuse(**kwargs: Any) -> Any:
        raise GpgvRefusalV4b(code)

    monkeypatch.setattr(apt, "verify_detached_v4b", refuse)
    _refused(case.verify(), code)


@pytest.mark.parametrize("target", ["Release", "Packages", "Release.gpg", "test-keyring.gpg"])
def test_material_change_during_crypto_refuses_even_after_valid_crypto(
    case: BoundaryCase, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    path = case.chain.directory / target

    def change(**kwargs: Any) -> GpgvEvidenceV4b:
        raw = path.read_bytes()
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        return _EVIDENCE

    monkeypatch.setattr(apt, "verify_detached_v4b", change)
    _refused(case.verify(), "FILE_CHANGED")


def test_release_parser_and_crypto_receive_identical_immutable_object(
    case: BoundaryCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[bytes] = []
    original_release = apt.v4a._release

    def crypto(**kwargs: Any) -> GpgvEvidenceV4b:
        observed.append(kwargs["release"])
        return _EVIDENCE

    def parse_release(raw: bytes, *args: Any) -> None:
        assert raw is observed[0]
        assert type(raw) is bytes
        original_release(raw, *args)

    monkeypatch.setattr(apt, "verify_detached_v4b", crypto)
    monkeypatch.setattr(apt.v4a, "_release", parse_release)
    assert case.verify().cryptographic_signature == "VALID"


def test_deb_archive_parser_uses_sealed_bytes_that_were_hashed(
    case: BoundaryCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = apt._verify_archive
    observed: list[int] = []

    def sealed_archive(fd: int, package: Any, *args: Any) -> Any:
        required = 0x0008 | 0x0004 | 0x0002 | 0x0001
        assert fcntl.fcntl(fd, getattr(fcntl, "F_GET_SEALS", 1034)) == required
        raw = os.pread(fd, package.size, 0)
        assert _sha(raw) == package.sha256
        with pytest.raises(OSError) as error:
            os.pwrite(fd, b"x", 0)
        assert error.value.errno == errno.EPERM
        observed.append(fd)
        return original(fd, package, *args)

    monkeypatch.setattr(apt, "_verify_archive", sealed_archive)
    assert case.verify().cryptographic_signature == "VALID"
    assert len(observed) == 1


def test_bad_second_deb_discards_first_package_measurements(
    case: BoundaryCase,
) -> None:
    second = build_package_case(case.chain.directory, package="libpython3.12-minimal")
    case.chain.package_case.declaration["packages"].append(second.package)
    case.chain.refresh()
    case.declaration = case.chain.declaration
    case.context = replace(case.context, release_sha256=case.declaration.release_sha256)
    second.path.write_bytes(second.path.read_bytes()[:-1] + b"x")
    _refused(case.verify(), "ARTIFACT_MISMATCH")


@pytest.mark.parametrize("phase", ["finish", "close", "deadline"])
def test_late_failure_discards_all_evidence(
    case: BoundaryCase, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    original_close = apt._InputReader.close
    if phase == "finish":

        def finish(self: Any, expected: Any) -> None:
            raise ContractRefusal("FILE_CHANGED")

        monkeypatch.setattr(apt._InputReader, "finish", finish)
    else:

        def close(self: Any) -> None:
            original_close(self)
            if phase == "deadline":
                self.budget.started -= 61
            else:
                raise OSError("synthetic cleanup failure")

        monkeypatch.setattr(apt._InputReader, "close", close)
    _refused(case.verify())


@pytest.mark.parametrize(
    "limits",
    [
        Limits(max_total_bytes=1),
        Limits(working_memory_bytes=1048576),
        Limits(timeout_s=float("nan")),
    ],
)
def test_resource_limits_never_publish_partial_success(case: BoundaryCase, limits: Limits) -> None:
    _refused(case.verify(limits=limits), "LIMIT_EXCEEDED")


@pytest.mark.parametrize(
    "error,code",
    [
        (ValueError("synthetic native decode failure"), "FORMAT_REJECTED"),
        (MemoryError("synthetic allocator failure"), "LIMIT_EXCEEDED"),
        (OSError("synthetic kernel failure"), "PROVENANCE_INCOMPLETE"),
    ],
)
def test_unexpected_native_failure_never_publishes_local_evidence(
    case: BoundaryCase,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    code: str,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(apt, "_verify_archive", fail)
    _refused(case.verify(), code)


@pytest.mark.parametrize("failure", ["unsupported", "readback"])
def test_missing_kernel_seal_enforcement_has_no_mutable_fallback(
    case: BoundaryCase,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    original = fcntl.fcntl

    def seal(fd: int, operation: int, *args: Any) -> Any:
        if operation == 1033 and failure == "unsupported":
            raise OSError(errno.EINVAL, "synthetic kernel without seal support")
        if operation == 1034 and failure == "readback":
            return 0
        return original(fd, operation, *args)

    monkeypatch.setattr(fcntl, "fcntl", seal)
    _refused(case.verify())


def test_truncated_read_during_deb_copy_discards_all_evidence(
    case: BoundaryCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(apt, "_pread", lambda *args: b"")
    _refused(case.verify(), "PROVENANCE_INCOMPLETE")


@pytest.fixture(scope="module")
def real_signer(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestSignerV4b]:
    local_gpgv_tool()
    with runtime_test_signer(tmp_path_factory.mktemp("TEST-v4b-signer")) as signer:
        yield signer


def _signed_boundary(directory: Path, signer: TestSignerV4b) -> BoundaryCase:
    case = _boundary(directory)
    signature_bytes = signer.sign(case.chain.release_path.read_bytes())
    (directory / case.signature.signature_path).write_bytes(signature_bytes)
    (directory / case.signature.keyring_path).write_bytes(signer.keyring)
    case.signature = replace(
        case.signature,
        signature_size=len(signature_bytes),
        signature_sha256=_sha(signature_bytes),
        keyring_size=len(signer.keyring),
        keyring_sha256=_sha(signer.keyring),
    )
    timestamp = signer.signature_timestamp
    case.context = replace(
        case.context,
        allowed_signers=(
            apt.SignerRuleV4b(
                signer.signing_fingerprint,
                signer.primary_fingerprint,
                timestamp - 3600,
                timestamp + 3600,
            ),
        ),
        keyring_sha256=case.signature.keyring_sha256,
        valid_from=timestamp - 3600,
        valid_until=timestamp + 3600,
        time_rules=apt.TimeRulesV4b(timestamp, 60, 0),
    )
    return case


def test_real_signed_complete_apt_chain_is_local_evidence_only(
    tmp_path: Path,
    real_signer: TestSignerV4b,
) -> None:
    case = _signed_boundary(tmp_path / "apt", real_signer)
    result = case.verify(tool=local_gpgv_tool())
    assert result.cryptographic_signature == "VALID", result.reason_codes
    assert result.signer_policy == "MATCH"
    assert result.evidence is not None
    assert result.evidence.signing_fingerprint == real_signer.signing_fingerprint
    assert result.v4a is not None and result.v4a.consistency == "MATCH"
    assert result.v4a.source_authenticity == "UNVERIFIED"
    assert result.v4a.signature_verification == "NOT_PERFORMED"
    _blocked(result)


@pytest.mark.parametrize(
    "mutation",
    [
        "changed-release",
        "invalid-signature",
        "wrong-subkey",
        "unapproved-signer",
        "revoked-policy",
        "changed-index",
        "rehashed-invalid-inner-payload",
    ],
)
def test_real_crypto_and_complete_chain_refusals_are_atomic(
    tmp_path: Path,
    real_signer: TestSignerV4b,
    mutation: str,
) -> None:
    case = _signed_boundary(tmp_path / "apt", real_signer)
    if mutation == "changed-release":
        case.chain.release_path.write_bytes(
            b"Origin: SYNTHETIC TEST CHANGE\n" + case.chain.release_path.read_bytes()
        )
        case.declaration = case.chain.declaration
        case.context = replace(case.context, release_sha256=case.declaration.release_sha256)
    elif mutation == "invalid-signature":
        signature = (case.chain.directory / case.signature.signature_path).read_bytes()
        signature = signature.replace(
            b"-----BEGIN PGP SIGNATURE-----", b"-----BEGIN PGP SOMETHING-----"
        )
        (case.chain.directory / case.signature.signature_path).write_bytes(signature)
        case.signature = replace(
            case.signature, signature_sha256=_sha(signature), signature_size=len(signature)
        )
    elif mutation == "wrong-subkey":
        signature = real_signer.sign(
            case.chain.release_path.read_bytes(),
            signing_fingerprint=real_signer.wrong_signing_fingerprint,
        )
        (case.chain.directory / case.signature.signature_path).write_bytes(signature)
        case.signature = replace(
            case.signature, signature_sha256=_sha(signature), signature_size=len(signature)
        )
    elif mutation == "unapproved-signer":
        case.context = replace(
            case.context,
            allowed_signers=(
                replace(
                    case.context.allowed_signers[0],
                    signing_fingerprint="C" * 40,
                    primary_fingerprint="D" * 40,
                ),
            ),
        )
    elif mutation == "revoked-policy":
        case.context = replace(
            case.context, allowed_signers=(replace(case.context.allowed_signers[0], revoked=True),)
        )
    elif mutation == "changed-index":
        case.chain.index_path.write_bytes(
            case.chain.index_path.read_bytes().replace(
                b"Architecture: amd64", b"Architecture: arm64"
            )
        )
        case.declaration = case.chain.declaration
    else:
        package_case = case.chain.package_case
        archive = package_case.path.read_bytes()
        changed = archive.replace(b"inert fixture bytes\n", b"wrong fixture bytes\n")
        assert changed != archive
        package_case.rebind_archive(changed)
        case.chain.refresh()
        case.declaration = case.chain.declaration
        case.context = replace(case.context, release_sha256=case.declaration.release_sha256)
        signature = real_signer.sign(case.chain.release_path.read_bytes())
        (case.chain.directory / case.signature.signature_path).write_bytes(signature)
        case.signature = replace(
            case.signature, signature_sha256=_sha(signature), signature_size=len(signature)
        )
    _refused(case.verify(tool=local_gpgv_tool()))
