"""Synthetic V4c signatures exercise contracts without creating production trust."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from deployment import (
    stage9_observer_apt_v4b as v4b,
    stage9_observer_apt_v4c as apt,
    stage9_observer_trust_v4c as trust,
)
from deployment.canonical_json import canonical_json_bytes
from deployment.stage9_observer_apt_v4a import AptResultV4a
from deployment.stage9_observer_apt_v4b import (
    AptResultV4b,
    SignerRuleV4b,
    TimeRulesV4b,
    TrustContextV4b,
)
from deployment.stage9_observer_checkpoint_v4c import SnapshotStateV4c
from deployment.stage9_observer_gpgv_v4b import GpgvEvidenceV4b
from tests.deployment.test_stage9_observer_apt_v4b_crypto import (
    TestSignerV4b,
    local_gpgv_tool,
    runtime_test_signer,
)

pytestmark = [pytest.mark.security, pytest.mark.unit]
_TIME = 1794657600
_SIGNING = "A" * 40
_PRIMARY = "B" * 40
_RELEASE = (
    b"Origin: SYNTHETIC TEST ONLY\n"
    b"Label: TEST\nSuite: TEST\nCodename: TEST\n"
    b"Date: Sat, 14 Nov 2026 11:59:00 UTC\n"
    b"Valid-Until: Sat, 14 Nov 2026 13:00:00 UTC\n"
    b"Architectures: amd64\nComponents: main\nSHA256:\n"
    + b" "
    + b"1" * 64
    + b" 1 main/binary-amd64/Packages\n"
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass
class SyntheticCase:
    context: TrustContextV4b
    result: AptResultV4b
    release: bytes
    state: SnapshotStateV4c
    payloads: dict[str, dict[str, Any]]
    authorities: dict[str, Any]
    keys: dict[str, tuple[Ed25519PrivateKey, ...]]
    previous_checkpoint_raw: bytes = b""

    def raw(self, role: str, payload: dict[str, Any] | None = None) -> bytes:
        payload = self.payloads[role] if payload is None else payload
        message = (
            trust.DOMAINS[role].encode("ascii")
            + b"\x00"
            + hashlib.sha256(canonical_json_bytes(payload)).digest()
        )
        signatures = [
            {"signer_id": key.signer_id, "signature_hex": private.sign(message).hex()}
            for key, private in zip(self.authorities[role].keys, self.keys[role], strict=True)
        ]
        return canonical_json_bytes({"payload": payload, "signatures": signatures})

    def verify(self, **overrides: Any) -> Any:
        selected = {
            "v4b_result": self.result,
            "context": self.context,
            "release_raw": self.release,
            "acceptance_raw": self.raw("acceptance"),
            "checkpoint_raw": self.raw("checkpoint"),
            "previous_checkpoint_raw": self.previous_checkpoint_raw,
            "time_evidence_raw": self.raw("time"),
            "acceptance_authority": self.authorities["acceptance"],
            "checkpoint_authority": self.authorities["checkpoint"],
            "time_authority": self.authorities["time"],
            "previous_state": self.state,
            "observed_checkpoint_before": self.state.checkpoint_sha256,
            "observed_checkpoint_after": self.state.checkpoint_sha256,
        }
        selected.update(overrides)
        return apt.verify_apt_acceptance_v4c(**selected)

    def rebind_prior_checkpoint(self) -> None:
        payload = copy.deepcopy(self.payloads["checkpoint"])
        payload.update(
            checkpoint_id=self.state.checkpoint_id,
            sequence=self.state.sequence,
            repository_id=self.state.repository_id,
            checkpoint_source_id=self.state.checkpoint_source_id,
            context_sha256=self.state.context_sha256,
            snapshot_id="TEST-previous-snapshot",
            snapshot_sha256=self.state.snapshot_sha256,
            previous_checkpoint_id=None,
            previous_snapshot_sha256=None,
            previous_context_sha256=None,
            permitted_context_successors=list(self.state.permitted_context_successors),
            revoked=self.state.revoked,
        )
        self.previous_checkpoint_raw = self.raw("checkpoint", payload)
        self.state = replace(self.state, checkpoint_sha256=_sha(self.previous_checkpoint_raw))

    def rebind_checkpoint(self) -> None:
        self.payloads["acceptance"]["checkpoint_sha256"] = _sha(self.raw("checkpoint"))

    def rebind_context(self) -> None:
        digest = apt.context_sha256_v4c(self.context)
        for payload in self.payloads.values():
            payload["context_sha256"] = digest
        self.rebind_checkpoint()

    def replace_release(self, release: bytes) -> None:
        self.release = release
        digest = _sha(release)
        self.context = replace(self.context, release_sha256=digest)
        self.result = replace(self.result, release_sha256=digest)
        for payload in self.payloads.values():
            payload["snapshot_sha256"] = digest
        self.rebind_context()
        self.state = replace(
            self.state, permitted_context_successors=(apt.context_sha256_v4c(self.context),)
        )
        self.rebind_prior_checkpoint()


def _authority(role: str, timestamp: int = _TIME) -> tuple[Any, tuple[Ed25519PrivateKey, ...]]:
    private = (Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate())
    keys = tuple(
        trust.TestAuthorityKeyV4c(
            f"TEST-{role}-{index}",
            key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex(),
            timestamp - 3600,
            timestamp + 3600,
        )
        for index, key in enumerate(private)
    )
    return (
        trust.TestAuthorityV4c(
            f"TEST-{role}-authority", 1, keys, 2, timestamp - 3600, timestamp + 3600
        ),
        private,
    )


@pytest.fixture
def case() -> SyntheticCase:
    digest = _sha(_RELEASE)
    context = TrustContextV4b(
        "TEST-context",
        1,
        (SignerRuleV4b(_SIGNING, _PRIMARY, _TIME - 3600, _TIME + 3600),),
        ((22, 8),),
        "EXACT_SIGNING_PRIMARY_PAIRS",
        "TEST-repository",
        "TEST-snapshot",
        digest,
        "2" * 64,
        _TIME - 3600,
        _TIME + 3600,
        False,
        TimeRulesV4b(_TIME, 300, 0),
        None,
    )
    result = AptResultV4b(
        "VALID",
        "MATCH",
        (),
        GpgvEvidenceV4b(_SIGNING, _PRIMARY, 22, 8, _TIME - 30, 0),
        AptResultV4a("MATCH", ()),
        digest,
    )
    return _v4c_case(context, result, _RELEASE)


def _v4c_case(context: TrustContextV4b, result: AptResultV4b, release: bytes) -> SyntheticCase:
    """Sign TEST V4c materials around an unchanged supplied V4b result."""
    digest = _sha(release)
    timestamp = context.time_rules.evaluation_time
    context_digest = apt.context_sha256_v4c(context)
    state = SnapshotStateV4c(
        context.repository_id,
        "TEST-checkpoint-source",
        "TEST-checkpoint-previous",
        4,
        "3" * 64,
        context_digest,
        "4" * 64,
    )
    authorities = {}
    keys = {}
    payloads = {}
    for role, schema in (
        ("acceptance", "stage9-apt-context-acceptance-v1"),
        ("checkpoint", "stage9-apt-checkpoint-v1"),
        ("time", "stage9-apt-time-evidence-v1"),
    ):
        authorities[role], keys[role] = _authority(role, timestamp)
        payloads[role] = {
            "schema": schema,
            "authority_id": authorities[role].authority_id,
            "policy_version": 1,
            "algorithm": "ed25519-sha256",
            "domain": schema,
            "valid_from": timestamp - 300,
            "valid_until": timestamp + 300,
            "repository_id": context.repository_id,
            "snapshot_id": context.snapshot_id,
            "context_sha256": context_digest,
            "snapshot_sha256": digest,
            "checkpoint_id": "TEST-checkpoint-next",
        }
    payloads["checkpoint"].update(
        checkpoint_source_id="TEST-checkpoint-source",
        sequence=5,
        successor_policy="EXACT_NEXT_SEQUENCE",
        rotation_policy="EXPLICIT_CONTEXT_LINK",
        revoked=False,
        previous_checkpoint_id=state.checkpoint_id,
        previous_snapshot_sha256=state.snapshot_sha256,
        previous_context_sha256=state.context_sha256,
        permitted_context_successors=[],
    )
    payloads["acceptance"].update(
        keyring_sha256=context.keyring_sha256,
        allowed_signers=sorted(
            [
                {
                    "signing_fingerprint": signer.signing_fingerprint,
                    "primary_fingerprint": signer.primary_fingerprint,
                }
                for signer in context.allowed_signers
            ],
            key=lambda signer: (signer["signing_fingerprint"], signer["primary_fingerprint"]),
        ),
        sequence=5,
        issued_at=timestamp - 60,
        checkpoint_sha256="0" * 64,
        rotation_policy="EXPLICIT_CONTEXT_LINK",
        revocation_policy="DENY_REVOKED",
        previous_context_sha256=state.context_sha256,
    )
    payloads["time"].update(
        proof_id="TEST-time-proof",
        time_source_id="TEST-time-source",
        evaluation_time=timestamp,
    )
    value = SyntheticCase(context, result, release, state, payloads, authorities, keys)
    value.rebind_checkpoint()
    value.rebind_prior_checkpoint()
    return value


def _blocked(result: Any) -> None:
    assert result.refused is True
    assert result.acceptance == "BLOCKED"
    assert result.external_authority_acceptance == result.authenticated_time == "UNVERIFIED"
    assert result.durable_anti_rollback == result.runtime_qualification == "BLOCKED"
    assert result.freshness == "BLOCKED"
    assert result.stage_9 == "IN_PROGRESS"
    assert result.qualification == "BLOCKED"
    assert result.stage_3b == "NOT AUTHORIZED"
    assert result.production == "NO-GO"
    assert result.production_status == "PRODUCTION_NO_GO"


def _refused(result: Any) -> None:
    assert result.technical_validation == "REFUSED"
    assert result.proposed_state is None
    assert result.reason_codes
    _blocked(result)


def test_complete_genuine_test_signatures_only_validate_technical_contract(
    case: SyntheticCase,
) -> None:
    before = copy.deepcopy(case.result)
    result = case.verify()
    assert result.technical_validation == "VALID", result.reason_codes
    assert result.proposed_state is not None
    assert result.proposed_state.sequence == 5
    assert result.proposed_state.snapshot_sha256 == _sha(case.release)
    assert case.state.sequence == 4
    assert case.result == before
    _blocked(result)


@pytest.mark.parametrize("role", ["acceptance", "checkpoint", "time"])
def test_self_signed_documents_cannot_supply_their_own_authority(
    case: SyntheticCase, role: str
) -> None:
    _refused(case.verify(**{f"{role}_authority": None}))


def test_fully_self_signed_bundle_without_independent_anchor_is_refused(
    case: SyntheticCase,
) -> None:
    _refused(case.verify(acceptance_authority=None, checkpoint_authority=None, time_authority=None))


@pytest.mark.parametrize("role", ["acceptance", "checkpoint", "time"])
@pytest.mark.parametrize(
    "field", ["context_sha256", "repository_id", "snapshot_id", "snapshot_sha256", "checkpoint_id"]
)
def test_each_signed_record_is_bound_to_exact_selection(
    case: SyntheticCase, role: str, field: str
) -> None:
    case.payloads[role][field] = "9" * 64 if field.endswith("sha256") else "TEST-substitution"
    if role == "checkpoint":
        case.rebind_checkpoint()
    _refused(case.verify())


def test_acceptance_is_bound_to_approved_keyring(case: SyntheticCase) -> None:
    case.payloads["acceptance"]["keyring_sha256"] = "9" * 64
    _refused(case.verify())


def test_changed_context_is_not_accepted_by_stale_signature(case: SyntheticCase) -> None:
    _refused(case.verify(context=replace(case.context, context_version=2)))


@pytest.mark.parametrize("role", ["acceptance", "checkpoint", "time"])
@pytest.mark.parametrize(
    "mutation",
    ["unknown-signer", "substituted-key", "duplicate-signature", "invalid-signature", "extra-key"],
)
def test_signature_inventory_and_key_substitution_fail_closed(
    case: SyntheticCase, role: str, mutation: str
) -> None:
    raw_name = {
        "acceptance": "acceptance_raw",
        "checkpoint": "checkpoint_raw",
        "time": "time_evidence_raw",
    }[role]
    envelope = json.loads(case.raw(role))
    if mutation == "unknown-signer":
        envelope["signatures"][0]["signer_id"] = "TEST-unknown-reviewer"
    elif mutation == "substituted-key":
        anchor = case.authorities[role]
        substituted = replace(
            anchor.keys[0],
            public_key_hex=Ed25519PrivateKey.generate()
            .public_key()
            .public_bytes(Encoding.Raw, PublicFormat.Raw)
            .hex(),
        )
        case.authorities[role] = replace(anchor, keys=(substituted, anchor.keys[1]))
        _refused(case.verify(**{raw_name: canonical_json_bytes(envelope)}))
        return
    elif mutation == "duplicate-signature":
        envelope["signatures"][1] = envelope["signatures"][0]
    elif mutation == "invalid-signature":
        signature = envelope["signatures"][0]["signature_hex"]
        envelope["signatures"][0]["signature_hex"] = (
            "01" if signature[:2] != "01" else "02"
        ) + signature[2:]
    else:
        envelope["signatures"][0]["public_key_hex"] = "1" * 64
    _refused(case.verify(**{raw_name: canonical_json_bytes(envelope)}))


@pytest.mark.parametrize("role", ["acceptance", "checkpoint", "time"])
@pytest.mark.parametrize("threshold", [0, 3, True, -1, 1.5])
def test_invalid_authority_threshold_never_yields_partial_success(
    case: SyntheticCase, role: str, threshold: Any
) -> None:
    case.authorities[role] = replace(case.authorities[role], threshold=threshold)
    _refused(case.verify())


@pytest.mark.parametrize("role", ["acceptance", "checkpoint", "time"])
@pytest.mark.parametrize(
    "mutation",
    [
        "revoked-authority",
        "revoked-key",
        "expired-authority",
        "future-authority",
        "expired-key",
        "same-key-two-identities",
    ],
)
def test_authority_policy_validity_revocation_and_distinct_keys(
    case: SyntheticCase, role: str, mutation: str
) -> None:
    anchor = case.authorities[role]
    if mutation == "revoked-authority":
        anchor = replace(anchor, revoked=True)
    elif mutation == "revoked-key":
        anchor = replace(anchor, keys=(replace(anchor.keys[0], revoked=True), anchor.keys[1]))
    elif mutation == "expired-authority":
        anchor = replace(anchor, valid_until=_TIME)
    elif mutation == "future-authority":
        anchor = replace(anchor, valid_from=_TIME + 1)
    elif mutation == "expired-key":
        anchor = replace(anchor, keys=(replace(anchor.keys[0], valid_until=_TIME), anchor.keys[1]))
    else:
        anchor = replace(
            anchor,
            keys=(
                anchor.keys[0],
                replace(anchor.keys[1], public_key_hex=anchor.keys[0].public_key_hex),
            ),
        )
        case.keys[role] = (case.keys[role][0], case.keys[role][0])
    case.authorities[role] = anchor
    _refused(case.verify())


@pytest.mark.parametrize("role", ["acceptance", "checkpoint", "time"])
@pytest.mark.parametrize(
    "mutation",
    [
        "expired",
        "future",
        "wrong-policy-version",
        "wrong-algorithm",
        "wrong-domain",
        "wrong-schema",
        "unknown-field",
        "missing-field",
    ],
)
def test_signed_contract_policy_and_shape_are_strict(
    case: SyntheticCase, role: str, mutation: str
) -> None:
    payload = case.payloads[role]
    if mutation == "expired":
        payload["valid_until"] = _TIME
    elif mutation == "future":
        payload["valid_from"] = _TIME + 1
    elif mutation == "wrong-policy-version":
        payload["policy_version"] = 2
    elif mutation == "wrong-algorithm":
        payload["algorithm"] = "ed25519"
    elif mutation == "wrong-domain":
        payload["domain"] = "stage9-other-subsystem"
    elif mutation == "wrong-schema":
        payload["schema"] = "stage9-acceptance-v1"
    elif mutation == "unknown-field":
        payload["caller_claims_verified"] = True
    else:
        del payload["authority_id"]
    if role == "checkpoint":
        case.rebind_checkpoint()
    _refused(case.verify())


@pytest.mark.parametrize(
    "argument",
    [
        "acceptance_raw",
        "checkpoint_raw",
        "time_evidence_raw",
        "previous_state",
        "observed_checkpoint_before",
        "observed_checkpoint_after",
    ],
)
def test_missing_materials_refuse_without_state_proposal(
    case: SyntheticCase, argument: str
) -> None:
    _refused(case.verify(**{argument: None}))


@pytest.mark.parametrize("argument", ["acceptance_raw", "checkpoint_raw", "time_evidence_raw"])
@pytest.mark.parametrize(
    "raw", [b"{", b"[]", b"null", b"{}", b" " * 65537, b"[[[[[[[[[[[[[[[[[0]]]]]]]]]]]]]]]]]"]
)
def test_corrupt_or_excessive_records_refuse_atomically(
    case: SyntheticCase, argument: str, raw: bytes
) -> None:
    _refused(case.verify(**{argument: raw}))


def test_changed_checkpoint_between_observations_refuses(case: SyntheticCase) -> None:
    _refused(case.verify(observed_checkpoint_after="9" * 64))


def test_unchanged_wrong_checkpoint_observations_still_refuse(case: SyntheticCase) -> None:
    _refused(case.verify(observed_checkpoint_before="9" * 64, observed_checkpoint_after="9" * 64))


@pytest.mark.parametrize("sequence", [3, 4, 6])
def test_rollback_equivocation_or_missing_successor_are_refused(
    case: SyntheticCase, sequence: int
) -> None:
    case.payloads["acceptance"]["sequence"] = sequence
    case.payloads["checkpoint"]["sequence"] = sequence
    case.rebind_checkpoint()
    result = case.verify()
    _refused(result)
    expected = {
        3: "SNAPSHOT_ROLLBACK",
        4: "SNAPSHOT_EQUIVOCATION",
        6: "SNAPSHOT_SUCCESSOR_SEQUENCE_REJECTED",
    }
    assert expected[sequence] in result.reason_codes


@pytest.mark.parametrize(
    "field", ["previous_checkpoint_id", "previous_snapshot_sha256", "previous_context_sha256"]
)
def test_signed_successor_requires_exact_predecessor(case: SyntheticCase, field: str) -> None:
    case.payloads["checkpoint"][field] = (
        "9" * 64 if field.endswith("sha256") else "TEST-wrong-predecessor"
    )
    case.rebind_checkpoint()
    _refused(case.verify())


def test_revoked_checkpoint_refuses(case: SyntheticCase) -> None:
    case.payloads["checkpoint"]["revoked"] = True
    case.rebind_checkpoint()
    _refused(case.verify())


def test_revoked_prior_state_refuses(case: SyntheticCase) -> None:
    _refused(case.verify(previous_state=replace(case.state, revoked=True)))


@pytest.mark.parametrize("field", ["repository_id", "checkpoint_source_id"])
def test_new_repository_or_checkpoint_source_cannot_inherit_old_state(
    case: SyntheticCase, field: str
) -> None:
    _refused(case.verify(previous_state=replace(case.state, **{field: "TEST-other"})))


@pytest.mark.parametrize(
    "replacement", [b"Fri, 13 Nov 2026 13:00:00 UTC", b"Sat, 14 Nov 2026 12:00:00 UTC"]
)
def test_expired_snapshot_is_not_fresh_with_valid_signatures(
    case: SyntheticCase, replacement: bytes
) -> None:
    release = case.release.replace(b"Sat, 14 Nov 2026 13:00:00 UTC", replacement)
    if replacement.startswith(b"Fri"):
        release = release.replace(
            b"Sat, 14 Nov 2026 11:59:00 UTC", b"Fri, 13 Nov 2026 11:59:00 UTC"
        )
    case.replace_release(release)
    result = case.verify()
    _refused(result)
    assert "SNAPSHOT_EXPIRED" in result.reason_codes


def test_future_snapshot_is_not_fresh_with_valid_signatures(case: SyntheticCase) -> None:
    case.replace_release(
        case.release.replace(b"Sat, 14 Nov 2026 11:59:00 UTC", b"Sat, 14 Nov 2026 12:01:00 UTC")
    )
    result = case.verify()
    _refused(result)
    assert "SNAPSHOT_FROM_FUTURE" in result.reason_codes


@pytest.mark.parametrize("field", [b"Date", b"Valid-Until"])
def test_release_requires_real_bound_dates(case: SyntheticCase, field: bytes) -> None:
    lines = [
        line for line in case.release.splitlines(keepends=True) if not line.startswith(field + b":")
    ]
    case.replace_release(b"".join(lines))
    _refused(case.verify())


@pytest.mark.parametrize(
    "line",
    [
        b"Date: Sat, 14 Nov 2026 11:59:00 UTC\n",
        b"date: Sat, 14 Nov 2026 11:59:00 UTC\n",
        b"Valid-Until: Sat, 14 Nov 2026 13:00:00 UTC\n",
    ],
)
def test_duplicate_release_time_fields_refuse(case: SyntheticCase, line: bytes) -> None:
    case.replace_release(line + case.release)
    _refused(case.verify())


def test_release_substitution_cannot_reuse_v4b_evidence(case: SyntheticCase) -> None:
    _refused(case.verify(release_raw=case.release + b"\n"))


@pytest.mark.parametrize(
    "mutation",
    [
        "crypto",
        "policy",
        "missing-evidence",
        "missing-chain",
        "wrong-chain",
        "wrong-hash",
        "unknown-signer",
        "future-signature",
        "expired-signature",
        "wrong-algorithm",
    ],
)
def test_v4b_evidence_is_revalidated_and_never_upgraded(case: SyntheticCase, mutation: str) -> None:
    result = case.result
    assert result.evidence is not None
    if mutation == "crypto":
        result = replace(result, cryptographic_signature="UNVERIFIED")
    elif mutation == "policy":
        result = replace(result, signer_policy="UNVERIFIED")
    elif mutation == "missing-evidence":
        result = replace(result, evidence=None)
    elif mutation == "missing-chain":
        result = replace(result, v4a=None)
    elif mutation == "wrong-chain":
        result = replace(result, v4a=AptResultV4a("MISMATCH", ("TEST",)))
    elif mutation == "wrong-hash":
        result = replace(result, release_sha256="9" * 64)
    elif mutation == "unknown-signer":
        result = replace(result, evidence=replace(result.evidence, signing_fingerprint="C" * 40))
    elif mutation == "future-signature":
        result = replace(result, evidence=replace(result.evidence, signature_timestamp=_TIME + 1))
    elif mutation == "expired-signature":
        result = replace(result, evidence=replace(result.evidence, signature_expiry=_TIME))
    else:
        result = replace(result, evidence=replace(result.evidence, hash_algorithm=2))
    before = copy.deepcopy(result)
    _refused(case.verify(v4b_result=result))
    assert result == before


@pytest.mark.parametrize("exception", [InterruptedError, KeyboardInterrupt])
def test_interrupted_operation_does_not_propose_state(
    case: SyntheticCase, monkeypatch: pytest.MonkeyPatch, exception: type[BaseException]
) -> None:
    def interrupted(*args: Any, **kwargs: Any) -> None:
        raise exception()

    monkeypatch.setattr(apt, "verify_record_v4c", interrupted)
    result = case.verify()
    _refused(result)
    assert "OPERATION_INTERRUPTED" in result.reason_codes
    assert case.state.sequence == 4


def test_no_local_clock_is_required_for_genuine_test_validation(
    case: SyntheticCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    def prohibited() -> float:
        pytest.fail("The local wall clock must never authenticate evaluation time")

    monkeypatch.setattr(time, "time", prohibited)
    result = case.verify()
    assert result.technical_validation == "VALID", result.reason_codes
    _blocked(result)


def test_technical_proposal_can_be_replayed_without_writing_state(case: SyntheticCase) -> None:
    first = case.verify()
    assert first.technical_validation == "VALID", first.reason_codes
    assert first.proposed_state is not None
    state = first.proposed_state
    replayed = case.verify(
        previous_checkpoint_raw=case.raw("checkpoint"),
        previous_state=state,
        observed_checkpoint_before=state.checkpoint_sha256,
        observed_checkpoint_after=state.checkpoint_sha256,
    )
    assert replayed.technical_validation == "VALID", replayed.reason_codes
    assert replayed.proposed_state == state
    assert case.state.sequence == 4
    _blocked(replayed)


def test_explicit_predecessor_authorization_allows_technical_context_rotation(
    case: SyntheticCase,
) -> None:
    old_digest = case.state.context_sha256
    case.context = replace(case.context, context_version=2)
    case.rebind_context()
    new_digest = apt.context_sha256_v4c(case.context)
    assert old_digest != new_digest
    case.state = replace(case.state, permitted_context_successors=(new_digest,))
    case.rebind_prior_checkpoint()
    result = case.verify()
    assert result.technical_validation == "VALID", result.reason_codes
    assert result.proposed_state is not None
    assert result.proposed_state.context_sha256 == new_digest
    assert result.proposed_state.permitted_context_successors == ()
    _blocked(result)


def test_signed_new_context_cannot_authorize_its_own_rotation(case: SyntheticCase) -> None:
    case.context = replace(case.context, context_version=2)
    case.rebind_context()
    result = case.verify()
    _refused(result)
    assert "CONTEXT_ROTATION_REJECTED" in result.reason_codes


def test_v4b_local_evaluation_time_is_not_reused_as_independent_time(case: SyntheticCase) -> None:
    case.context = replace(
        case.context, time_rules=replace(case.context.time_rules, evaluation_time=_TIME - 600)
    )
    case.rebind_context()
    case.state = replace(
        case.state, permitted_context_successors=(apt.context_sha256_v4c(case.context),)
    )
    case.rebind_prior_checkpoint()
    result = case.verify()
    assert result.technical_validation == "VALID", result.reason_codes
    _blocked(result)


def test_acceptance_issued_after_signed_evaluation_time_is_refused(case: SyntheticCase) -> None:
    case.payloads["acceptance"]["issued_at"] = _TIME + 1
    result = case.verify()
    _refused(result)
    assert "V4C_SIGNING_TIME_REJECTED" in result.reason_codes


@pytest.mark.parametrize("mutation", ["missing-one", "none", "extra-valid-unknown"])
def test_threshold_cannot_be_satisfied_by_partial_or_unknown_signatures(
    case: SyntheticCase, mutation: str
) -> None:
    envelope = json.loads(case.raw("acceptance"))
    if mutation == "missing-one":
        envelope["signatures"].pop()
    elif mutation == "none":
        envelope["signatures"] = []
    else:
        payload = envelope["payload"]
        message = trust.signing_message_v4c("acceptance", payload)
        envelope["signatures"].append(
            {
                "signer_id": "TEST-zz-unknown",
                "signature_hex": Ed25519PrivateKey.generate().sign(message).hex(),
            }
        )
    _refused(case.verify(acceptance_raw=canonical_json_bytes(envelope)))


@pytest.mark.parametrize("role", ["acceptance", "checkpoint", "time"])
def test_valid_threshold_does_not_ignore_an_additional_invalid_signature(
    case: SyntheticCase, role: str
) -> None:
    anchor = case.authorities[role]
    extra_key = Ed25519PrivateKey.generate()
    additional = trust.TestAuthorityKeyV4c(
        "TEST-zz-additional",
        extra_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex(),
        _TIME - 3600,
        _TIME + 3600,
    )
    envelope = json.loads(case.raw(role))
    envelope["signatures"].append({"signer_id": additional.signer_id, "signature_hex": "0" * 128})
    case.authorities[role] = replace(anchor, keys=anchor.keys + (additional,))
    case.keys[role] = case.keys[role] + (extra_key,)
    argument = {
        "acceptance": "acceptance_raw",
        "checkpoint": "checkpoint_raw",
        "time": "time_evidence_raw",
    }[role]
    _refused(case.verify(**{argument: canonical_json_bytes(envelope)}))


@pytest.mark.parametrize("role", ["acceptance", "checkpoint", "time"])
@pytest.mark.parametrize(
    "mutation", ["whitespace", "duplicate-field", "float", "nonfinite", "boolean-integer"]
)
def test_wire_profile_rejects_ambiguous_json_even_with_genuine_signatures(
    case: SyntheticCase, role: str, mutation: str
) -> None:
    raw = case.raw(role)
    if mutation == "whitespace":
        raw = b" " + raw
    elif mutation == "duplicate-field":
        raw = raw.replace(b'"policy_version":1', b'"policy_version":1,"policy_version":1')
    elif mutation == "float":
        raw = raw.replace(b'"policy_version":1', b'"policy_version":1.0')
    elif mutation == "nonfinite":
        raw = raw.replace(b'"policy_version":1', b'"policy_version":NaN')
    else:
        raw = raw.replace(b'"policy_version":1', b'"policy_version":true')
    argument = {
        "acceptance": "acceptance_raw",
        "checkpoint": "checkpoint_raw",
        "time": "time_evidence_raw",
    }[role]
    _refused(case.verify(**{argument: raw}))


@pytest.mark.parametrize("role", ["acceptance", "checkpoint", "time"])
def test_signature_from_other_stage9_domain_does_not_verify(case: SyntheticCase, role: str) -> None:
    envelope = json.loads(case.raw(role))
    foreign_message = (
        b"stage9-independent-acceptance-v1\x00"
        + hashlib.sha256(canonical_json_bytes(envelope["payload"])).digest()
    )
    for signature, private in zip(envelope["signatures"], case.keys[role], strict=True):
        signature["signature_hex"] = private.sign(foreign_message).hex()
    argument = {
        "acceptance": "acceptance_raw",
        "checkpoint": "checkpoint_raw",
        "time": "time_evidence_raw",
    }[role]
    result = case.verify(**{argument: canonical_json_bytes(envelope)})
    _refused(result)
    assert "V4C_SIGNATURE_INVALID" in result.reason_codes


@pytest.mark.parametrize(
    "replacement",
    [
        b"Sat, 14 Nov 2026 11:59:00 +0100",
        b"Fri, 14 Nov 2026 11:59:00 UTC",
        b"Sat, 32 Nov 2026 11:59:00 UTC",
        b"2026-11-14T11:59:00Z",
    ],
)
def test_release_time_representation_is_bounded_and_unambiguous(
    case: SyntheticCase, replacement: bytes
) -> None:
    case.replace_release(case.release.replace(b"Sat, 14 Nov 2026 11:59:00 UTC", replacement))
    _refused(case.verify())


@pytest.mark.parametrize("raw", [b"X" * (1048576 + 1), b"Date: \xff\n", b"\n" * 4097])
def test_release_resource_and_encoding_limits_refuse(case: SyntheticCase, raw: bytes) -> None:
    _refused(case.verify(release_raw=raw))


def test_failed_late_transition_discards_all_other_positive_evidence(case: SyntheticCase) -> None:
    before = copy.deepcopy(case.state)
    result = case.verify(observed_checkpoint_after="9" * 64)
    _refused(result)
    assert "CHECKPOINT_CHANGED" in result.reason_codes
    assert case.state == before
    assert all(
        value in {"UNVERIFIED", "BLOCKED"}
        for value in (
            result.external_authority_acceptance,
            result.authenticated_time,
            result.durable_anti_rollback,
            result.runtime_qualification,
        )
    )


@pytest.mark.parametrize(
    "field", ["cryptographic_signature", "signer_policy", "release_sha256", "evidence", "v4a"]
)
def test_v4b_object_cannot_be_implicitly_repaired_by_v4c(case: SyntheticCase, field: str) -> None:
    value = None if field in {"evidence", "v4a", "release_sha256"} else "UNVERIFIED"
    result = replace(case.result, **{field: value})
    before = copy.deepcopy(result)
    _refused(case.verify(v4b_result=result))
    assert result == before
    assert result.external_authority_acceptance == "UNVERIFIED"
    assert result.acceptance == "BLOCKED"


@pytest.mark.parametrize("raw", [None, b"{", b"{}", b" " * 65537])
def test_prior_checkpoint_requires_a_valid_bounded_signed_record(
    case: SyntheticCase, raw: bytes | None
) -> None:
    _refused(case.verify(previous_checkpoint_raw=raw))


@pytest.mark.parametrize(
    "field,value",
    [
        ("repository_id", "TEST-foreign-repository"),
        ("checkpoint_source_id", "TEST-foreign-source"),
        ("checkpoint_id", "TEST-forged-id"),
        ("sequence", 3),
        ("snapshot_sha256", "9" * 64),
        ("context_sha256", "9" * 64),
        ("checkpoint_sha256", "9" * 64),
        ("permitted_context_successors", ("9" * 64,)),
        ("revoked", True),
    ],
)
def test_caller_cannot_mutate_prior_state_while_retaining_signed_checkpoint(
    case: SyntheticCase, field: str, value: Any
) -> None:
    mutated = replace(case.state, **{field: value})
    _refused(case.verify(previous_state=mutated))


@pytest.mark.parametrize(
    "mutation", ["signature", "foreign-authority", "context", "snapshot", "rotations"]
)
def test_prior_checkpoint_signatures_and_full_state_projection_are_verified(
    case: SyntheticCase, mutation: str
) -> None:
    envelope = json.loads(case.previous_checkpoint_raw)
    if mutation == "signature":
        envelope["signatures"][0]["signature_hex"] = "0" * 128
        raw = canonical_json_bytes(envelope)
    else:
        payload = envelope["payload"]
        if mutation == "foreign-authority":
            payload["authority_id"] = "TEST-foreign-authority"
        elif mutation == "context":
            payload["context_sha256"] = "9" * 64
        elif mutation == "snapshot":
            payload["snapshot_sha256"] = "9" * 64
        else:
            payload["permitted_context_successors"] = ["9" * 64]
        raw = case.raw("checkpoint", payload)
    _refused(case.verify(previous_checkpoint_raw=raw))


def test_recomputed_prior_checkpoint_hash_cannot_hide_projection_substitution(
    case: SyntheticCase,
) -> None:
    envelope = json.loads(case.previous_checkpoint_raw)
    envelope["payload"]["permitted_context_successors"] = ["9" * 64]
    substituted = case.raw("checkpoint", envelope["payload"])
    digest = _sha(substituted)
    _refused(
        case.verify(
            previous_checkpoint_raw=substituted,
            previous_state=replace(case.state, checkpoint_sha256=digest),
            observed_checkpoint_before=digest,
            observed_checkpoint_after=digest,
        )
    )


@pytest.fixture(scope="module")
def real_v4c_signer(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestSignerV4b]:
    # Tool selection skips unsupported platforms/missing gpgv; it never downloads.
    try:
        local_gpgv_tool()
    except FileNotFoundError:
        pytest.skip("Full V4b -> V4c integration requires an existing local TEST gpgv")
    with runtime_test_signer(tmp_path_factory.mktemp("TEST-v4c-signer")) as signer:
        yield signer


@pytest.fixture
def real_v4b_v4c_case(tmp_path: Path, real_v4c_signer: TestSignerV4b) -> tuple[SyntheticCase, Any]:
    # Import only after local_gpgv_tool's Linux check: this helper's module uses
    # importorskip(fcntl), which must not skip the portable V4c contract tests.
    from tests.deployment.test_stage9_observer_apt_v4b import _boundary

    boundary = _boundary(tmp_path / "apt")
    release = (
        b"Date: Thu, 01 Jan 2026 00:00:00 UTC\n"
        b"Valid-Until: Thu, 01 Jan 2026 01:00:00 UTC\n" + boundary.chain.release_path.read_bytes()
    )
    boundary.chain.set_release(release)
    boundary.declaration = boundary.chain.declaration
    signature = real_v4c_signer.sign(release)
    (boundary.chain.directory / boundary.signature.signature_path).write_bytes(signature)
    (boundary.chain.directory / boundary.signature.keyring_path).write_bytes(
        real_v4c_signer.keyring
    )
    boundary.signature = replace(
        boundary.signature,
        signature_sha256=_sha(signature),
        signature_size=len(signature),
        keyring_sha256=_sha(real_v4c_signer.keyring),
        keyring_size=len(real_v4c_signer.keyring),
    )
    timestamp = real_v4c_signer.signature_timestamp
    boundary.context = replace(
        boundary.context,
        release_sha256=_sha(release),
        keyring_sha256=boundary.signature.keyring_sha256,
        allowed_signers=(
            SignerRuleV4b(
                real_v4c_signer.signing_fingerprint,
                real_v4c_signer.primary_fingerprint,
                timestamp - 3600,
                timestamp + 3600,
            ),
        ),
        valid_from=timestamp - 3600,
        valid_until=timestamp + 3600,
        # gpg's faked-system-time can advance while signing; keep evaluation
        # separately fixed and bounded rather than assuming the same second.
        time_rules=TimeRulesV4b(timestamp + 30, 60, 0),
    )
    result = v4b.verify_apt_signature_v4b(
        boundary.chain.directory,
        declaration=boundary.declaration,
        signature=boundary.signature,
        context=boundary.context,
        inventory_raw=boundary.chain.inventory_raw,
        tool=local_gpgv_tool(),
    )
    assert result.cryptographic_signature == "VALID", result.reason_codes
    assert result.signer_policy == "MATCH"
    assert result.evidence is not None
    assert timestamp <= result.evidence.signature_timestamp <= timestamp + 30
    assert result.evidence.signing_fingerprint == real_v4c_signer.signing_fingerprint
    assert result.release_sha256 == _sha(release)
    assert result.v4a is not None and result.v4a.consistency == "MATCH"
    assert len(result.v4a.packages) == 1
    assert result.v4a.packages[0].archive_sha256 == boundary.chain.package_case.package["sha256"]
    case = _v4c_case(boundary.context, result, release)
    assert case.result is result
    return case, boundary


def test_real_v4b_to_v4c_proposes_only_immutable_technical_state(
    real_v4b_v4c_case: tuple[SyntheticCase, Any],
) -> None:
    """Actual gpgv and both public verifiers run in-process, without crypto mocks.

    The returned V4b object demonstrates composition; it is not independently
    authenticated execution evidence or a production authority token.
    """
    case, _ = real_v4b_v4c_case
    before_result, before_state = copy.deepcopy(case.result), case.state
    result = case.verify()
    assert result.technical_validation == "VALID", result.reason_codes
    proposal = result.proposed_state
    assert proposal is not None
    assert proposal.sequence == before_state.sequence + 1
    assert proposal.snapshot_sha256 == _sha(case.release)
    assert proposal.context_sha256 == apt.context_sha256_v4c(case.context)
    assert proposal.checkpoint_sha256 == _sha(case.raw("checkpoint"))
    with pytest.raises(FrozenInstanceError):
        proposal.sequence = 99
    assert case.state == before_state
    assert case.result == before_result
    assert case.result.external_authority_acceptance == "UNVERIFIED"
    assert case.result.source_authenticity == "UNVERIFIED"
    assert case.result.tool_acceptance == case.result.system_isolation == "UNVERIFIED"
    assert case.result.freshness == case.result.checkpoint == case.result.rollback == "BLOCKED"
    assert case.result.acceptance == case.result.qualification == "BLOCKED"
    assert case.result.stage_9 == "IN_PROGRESS"
    assert case.result.stage_3b == "NOT AUTHORIZED"
    assert case.result.STAGE_3A_COMPLETE == case.result.PRODUCTION_QUALIFIED == "NO"
    assert case.result.production_status == "PRODUCTION_NO_GO"
    assert case.result.refused is True
    _blocked(result)
    assert result.STAGE_3A_COMPLETE == result.PRODUCTION_QUALIFIED == "NO"


@pytest.mark.parametrize(
    "mutation, expected_reason",
    [
        ("changed-release-after-signing", "SNAPSHOT_BINDING_REJECTED"),
        ("invalid-apt-signature", "SNAPSHOT_BINDING_REJECTED"),
        ("disallowed-apt-signer", "SNAPSHOT_BINDING_REJECTED"),
        ("substituted-v4b-context", "CONTEXT_BINDING_REJECTED"),
        ("invalid-acceptance-signature", "V4C_SIGNATURE_INVALID"),
        ("invalid-checkpoint-signature", "V4C_SIGNATURE_INVALID"),
        ("invalid-previous-checkpoint-signature", "V4C_SIGNATURE_INVALID"),
        ("rollback-sequence", "SNAPSHOT_ROLLBACK"),
        ("mismatched-predecessor-hash", "SNAPSHOT_SUCCESSOR_PROOF_REJECTED"),
        ("missing-time-evidence", "TIME_EVIDENCE_MISSING"),
        ("mismatched-time-evidence", "SNAPSHOT_MISMATCH"),
    ],
)
def test_real_v4b_to_v4c_refuses_without_partial_proposal(
    real_v4b_v4c_case: tuple[SyntheticCase, Any], mutation: str, expected_reason: str
) -> None:
    case, boundary = real_v4b_v4c_case
    before_state = case.state
    overrides = {}
    if mutation in {
        "changed-release-after-signing",
        "invalid-apt-signature",
        "disallowed-apt-signer",
    }:
        if mutation == "changed-release-after-signing":
            original_signature = (boundary.chain.directory / "Release.gpg").read_bytes()
            changed = b"Origin: SYNTHETIC TEST CHANGE\n" + case.release
            boundary.chain.set_release(changed)
            boundary.declaration = boundary.chain.declaration
            boundary.context = replace(boundary.context, release_sha256=_sha(changed))
            assert (boundary.chain.directory / "Release.gpg").read_bytes() == original_signature
        elif mutation == "invalid-apt-signature":
            signature_path = boundary.chain.directory / "Release.gpg"
            armored = signature_path.read_bytes()
            encoded = b"".join(
                line
                for line in armored.split(b"\n\n", 1)[1].splitlines()
                if line and not line.startswith((b"=", b"-"))
            )
            packet = base64.b64decode(encoded, validate=True)
            signature = packet[:-1] + bytes([packet[-1] ^ 1])
            signature_path.write_bytes(signature)
            boundary.signature = replace(
                boundary.signature, signature_sha256=_sha(signature), signature_size=len(signature)
            )
        else:
            boundary.context = replace(
                boundary.context,
                allowed_signers=(
                    replace(boundary.context.allowed_signers[0], signing_fingerprint="C" * 40),
                ),
            )
        # Propagate the actual failed public V4b result, never a fabricated or
        # manually "repaired" AptResultV4b, across the V4c boundary.
        case.result = v4b.verify_apt_signature_v4b(
            boundary.chain.directory,
            declaration=boundary.declaration,
            signature=boundary.signature,
            context=boundary.context,
            inventory_raw=boundary.chain.inventory_raw,
            tool=local_gpgv_tool(),
        )
        overrides["context"] = boundary.context
        overrides["release_raw"] = boundary.chain.release_path.read_bytes()
        assert case.result.cryptographic_signature == "UNVERIFIED"
        assert case.result.signer_policy == "UNVERIFIED"
        assert case.result.evidence is None and case.result.v4a is None
        if mutation == "disallowed-apt-signer":
            assert "SIGNER_REJECTED" in case.result.reason_codes
        else:
            assert case.result.reason_codes == ("GPGV_PROCESS_FAILED",)
    elif mutation == "substituted-v4b-context":
        overrides["context"] = replace(
            case.context, context_version=case.context.context_version + 1
        )
    elif mutation.startswith("invalid-"):
        role = "acceptance" if mutation == "invalid-acceptance-signature" else "checkpoint"
        raw = case.previous_checkpoint_raw if "previous" in mutation else case.raw(role)
        envelope = json.loads(raw)
        envelope["signatures"][0]["signature_hex"] = "0" * 128
        target = "previous_checkpoint_raw" if "previous" in mutation else f"{role}_raw"
        overrides[target] = canonical_json_bytes(envelope)
    elif mutation == "rollback-sequence":
        case.payloads["acceptance"]["sequence"] = case.payloads["checkpoint"]["sequence"] = 3
        case.rebind_checkpoint()
    elif mutation == "mismatched-predecessor-hash":
        case.payloads["checkpoint"]["previous_snapshot_sha256"] = "9" * 64
        case.rebind_checkpoint()
    elif mutation == "missing-time-evidence":
        overrides["time_evidence_raw"] = None
    else:
        case.payloads["time"]["snapshot_id"] = "TEST-different-snapshot"
    before_result = copy.deepcopy(case.result)
    result = case.verify(**overrides)
    _refused(result)
    assert expected_reason in result.reason_codes
    assert case.result == before_result
    assert case.state == before_state
