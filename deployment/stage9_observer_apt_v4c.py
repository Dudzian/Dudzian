"""Nonproduction APT acceptance composition; technical evidence never authorizes.

All roots, source observations and signed time here are explicitly TEST inputs.
There is no production acceptance adapter or durable writer. V4b is read only.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Literal

from deployment import stage9_observer_apt_v4a as v4a, stage9_observer_apt_v4b as v4b
from deployment.canonical_json import canonical_json_bytes
from deployment.stage9_observer_checkpoint_v4c import (
    SnapshotCandidateV4c,
    SnapshotStateV4c,
    propose_transition_v4c,
)
from deployment.stage9_observer_gpgv_v4b import GpgvEvidenceV4b
from deployment.stage9_observer_trust_v4c import (
    TestAuthorityV4c,
    V4cRefusal,
    parse_acceptance_v4c,
    parse_checkpoint_v4c,
    parse_time_evidence_v4c,
    verify_record_v4c,
)

_MAX_RELEASE_BYTES = 1048576
_MAX_TIMESTAMP = 253402300799
_BLOCKERS = (
    "EXTERNAL_AUTHORITY_UNVERIFIED",
    "AUTHENTICATED_TIME_UNVERIFIED",
    "DURABLE_ANTI_ROLLBACK_BLOCKED",
    "RUNTIME_QUALIFICATION_BLOCKED",
)


@dataclass(frozen=True, slots=True)
class AptResultV4c:
    technical_validation: Literal["VALID", "REFUSED"]
    reason_codes: tuple[str, ...]
    proposed_state: SnapshotStateV4c | None = None
    profile: Literal["stage9-apt-acceptance-foundation-v4c"] = field(
        default="stage9-apt-acceptance-foundation-v4c", init=False
    )
    external_authority_acceptance: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    authenticated_time: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    durable_anti_rollback: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    runtime_qualification: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    freshness: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    acceptance: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    stage_9: Literal["IN_PROGRESS"] = field(default="IN_PROGRESS", init=False)
    qualification: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    stage_3b: Literal["NOT AUTHORIZED"] = field(default="NOT AUTHORIZED", init=False)
    production: Literal["NO-GO"] = field(default="NO-GO", init=False)
    production_status: Literal["PRODUCTION_NO_GO"] = field(default="PRODUCTION_NO_GO", init=False)
    STAGE_3A_COMPLETE: Literal["NO"] = field(default="NO", init=False)
    PRODUCTION_QUALIFIED: Literal["NO"] = field(default="NO", init=False)
    refused: Literal[True] = field(default=True, init=False)


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise V4cRefusal(code)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_value(value: Any) -> Any:
    if type(value) is dict:
        return {key: _json_value(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_json_value(item) for item in value]
    return value


def context_sha256_v4c(context: v4b.TrustContextV4b) -> str:
    """Hash every V4b context field, preserving array order; tuples become lists.

    V4b's existing technical context validator is reused with inert declarations.
    Neither its references nor its caller evaluation_time gain authority here.
    """
    _require(type(context) is v4b.TrustContextV4b, "CONTEXT_REJECTED")
    _require(context.schema == "stage9-apt-trust-context-v1", "CONTEXT_REJECTED")
    declaration = v4a.AptDeclarationV4a(
        "Release", context.release_sha256, 1, "Packages", "0" * 64, 1, "Packages"
    )
    selection = v4b.SignatureDeclarationV4b(
        "Release.gpg",
        "0" * 64,
        1,
        "keyring.gpg",
        context.keyring_sha256,
        1,
        context.repository_id,
        context.snapshot_id,
    )
    try:
        v4b._context(context, selection, declaration)
    except v4b._Refusal as exc:
        raise V4cRefusal(exc.code) from None
    raw = canonical_json_bytes(_json_value(asdict(context)))
    _require(len(raw) <= 65536, "LIMIT_EXCEEDED")
    return _sha(raw)


def _release_times(raw: bytes) -> tuple[int, int]:
    _require(type(raw) is bytes and 0 < len(raw) <= _MAX_RELEASE_BYTES, "RELEASE_REJECTED")
    _require(raw.count(b"\n") <= 4096, "LIMIT_EXCEEDED")
    _require(raw.endswith(b"\n") and b"\r" not in raw, "RELEASE_REJECTED")
    fields: dict[str, str] = {}
    previous = ""
    ended = False
    for line in raw.decode("ascii", errors="strict").splitlines():
        _require(len(line) <= 4096, "LIMIT_EXCEEDED")
        if not line:
            ended = True
            continue
        _require(not ended, "RELEASE_REJECTED")
        if line.startswith(" "):
            _require(previous == "sha256", "RELEASE_REJECTED")
            continue
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9-]*):(?: (.*))?", line)
        _require(match is not None, "RELEASE_REJECTED")
        assert match is not None
        name, value = match.groups()
        previous = name.lower()
        _require(previous not in fields, "RELEASE_REJECTED")
        fields[previous] = value or ""
    parsed = []
    for name in ("date", "valid-until"):
        text = fields.get(name, "")
        _require(
            re.fullmatch(
                r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), [0-9]{2} "
                r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
                r"[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} (?:UTC|GMT|\+0000)",
                text,
            )
            is not None,
            "RELEASE_TIME_REJECTED",
        )
        instant = parsedate_to_datetime(text)
        _require(instant.utcoffset() == timezone.utc.utcoffset(None), "RELEASE_TIME_REJECTED")
        _require(
            text[:3] == ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[instant.weekday()],
            "RELEASE_TIME_REJECTED",
        )
        timestamp = int(instant.timestamp())
        _require(0 <= timestamp <= _MAX_TIMESTAMP, "RELEASE_TIME_REJECTED")
        parsed.append(timestamp)
    _require(parsed[0] < parsed[1], "RELEASE_TIME_REJECTED")
    return parsed[0], parsed[1]


def _v4b_binding(result: v4b.AptResultV4b, context: v4b.TrustContextV4b, now: int) -> None:
    _require(
        type(result) is v4b.AptResultV4b
        and result.cryptographic_signature == "VALID"
        and result.signer_policy == "MATCH"
        and type(result.evidence) is GpgvEvidenceV4b
        and type(result.v4a) is v4a.AptResultV4a
        and result.v4a.consistency == "MATCH",
        "V4B_EVIDENCE_REJECTED",
    )
    assert result.evidence is not None
    for value in (result.evidence.signature_timestamp, result.evidence.signature_expiry):
        _require(type(value) is int and 0 <= value <= _MAX_TIMESTAMP, "V4B_EVIDENCE_REJECTED")
    _require(
        type(result.evidence.public_key_algorithm) is int
        and type(result.evidence.hash_algorithm) is int,
        "V4B_EVIDENCE_REJECTED",
    )
    _require(context.valid_from <= now < context.valid_until, "CONTEXT_TIME_REJECTED")
    evaluated = replace(context, time_rules=replace(context.time_rules, evaluation_time=now))
    try:
        v4b._signer(evaluated, result.evidence)
    except v4b._Refusal as exc:
        raise V4cRefusal(exc.code) from None


def verify_apt_acceptance_v4c(
    *,
    v4b_result: v4b.AptResultV4b,
    context: v4b.TrustContextV4b,
    release_raw: bytes,
    acceptance_raw: bytes | None,
    checkpoint_raw: bytes | None,
    time_evidence_raw: bytes | None,
    acceptance_authority: TestAuthorityV4c | None = None,
    checkpoint_authority: TestAuthorityV4c | None = None,
    time_authority: TestAuthorityV4c | None = None,
    previous_state: SnapshotStateV4c | None = None,
    previous_checkpoint_raw: bytes | None = None,
    observed_checkpoint_before: str | None = None,
    observed_checkpoint_after: str | None = None,
) -> AptResultV4c:
    """Atomically validate supplied TEST materials and propose, never apply, state.

    Before/after values are captured SHA-256 observations of the *previous*
    checkpoint; the incoming signed checkpoint describes the candidate. This
    interface does not itself read or authenticate storage or a clock.
    """
    try:
        context_digest = context_sha256_v4c(context)
        release_date, valid_until = _release_times(release_raw)
        snapshot_digest = _sha(release_raw)
        _require(type(v4b_result) is v4b.AptResultV4b, "V4B_EVIDENCE_REJECTED")
        _require(
            snapshot_digest == context.release_sha256 == v4b_result.release_sha256,
            "SNAPSHOT_BINDING_REJECTED",
        )
        _require(acceptance_raw is not None, "ACCEPTANCE_MISSING")
        _require(checkpoint_raw is not None, "CHECKPOINT_MISSING")
        _require(time_evidence_raw is not None, "TIME_EVIDENCE_MISSING")
        assert (
            acceptance_raw is not None
            and checkpoint_raw is not None
            and time_evidence_raw is not None
        )
        acceptance = parse_acceptance_v4c(acceptance_raw)
        checkpoint = parse_checkpoint_v4c(checkpoint_raw)
        time_proof = parse_time_evidence_v4c(time_evidence_raw)
        ap, cp, tp = acceptance.payload, checkpoint.payload, time_proof.payload
        now = tp["evaluation_time"]
        verify_record_v4c(time_proof, time_authority, now)
        verify_record_v4c(checkpoint, checkpoint_authority, now)
        verify_record_v4c(acceptance, acceptance_authority, now)
        _require(previous_checkpoint_raw is not None, "PREVIOUS_CHECKPOINT_MISSING")
        assert previous_checkpoint_raw is not None
        predecessor = parse_checkpoint_v4c(previous_checkpoint_raw)
        verify_record_v4c(predecessor, checkpoint_authority, now)
        _require(type(previous_state) is SnapshotStateV4c, "CHECKPOINT_STATE_UNKNOWN")
        assert previous_state is not None
        pp = predecessor.payload
        expected_state = SnapshotStateV4c(
            repository_id=pp["repository_id"],
            checkpoint_source_id=pp["checkpoint_source_id"],
            checkpoint_id=pp["checkpoint_id"],
            sequence=pp["sequence"],
            snapshot_sha256=pp["snapshot_sha256"],
            context_sha256=pp["context_sha256"],
            checkpoint_sha256=_sha(previous_checkpoint_raw),
            permitted_context_successors=tuple(pp["permitted_context_successors"]),
            revoked=pp["revoked"],
        )
        _require(previous_state == expected_state, "PREVIOUS_CHECKPOINT_BINDING_REJECTED")
        for payload in (ap, cp, tp):
            _require(payload["repository_id"] == context.repository_id, "REPOSITORY_MISMATCH")
            _require(payload["snapshot_id"] == context.snapshot_id, "SNAPSHOT_MISMATCH")
            _require(payload["snapshot_sha256"] == snapshot_digest, "SNAPSHOT_BINDING_REJECTED")
            _require(payload["context_sha256"] == context_digest, "CONTEXT_BINDING_REJECTED")
            _require(payload["checkpoint_id"] == cp["checkpoint_id"], "CHECKPOINT_BINDING_REJECTED")
        _require(ap["checkpoint_sha256"] == _sha(checkpoint_raw), "CHECKPOINT_BINDING_REJECTED")
        _require(ap["sequence"] == cp["sequence"], "SEQUENCE_MISMATCH")
        _require(ap["keyring_sha256"] == context.keyring_sha256, "KEYRING_BINDING_REJECTED")
        _require(
            ap["previous_context_sha256"] == cp["previous_context_sha256"],
            "PREDECESSOR_BINDING_REJECTED",
        )
        signers = sorted(
            [
                {
                    "signing_fingerprint": item.signing_fingerprint,
                    "primary_fingerprint": item.primary_fingerprint,
                }
                for item in context.allowed_signers
            ],
            key=lambda item: (item["signing_fingerprint"], item["primary_fingerprint"]),
        )
        _require(ap["allowed_signers"] == signers, "SIGNER_BINDING_REJECTED")
        _v4b_binding(v4b_result, context, now)
        _require(
            release_date <= now + context.time_rules.maximum_future_skew_seconds,
            "SNAPSHOT_FROM_FUTURE",
        )
        _require(now < valid_until, "SNAPSHOT_EXPIRED")
        candidate = SnapshotCandidateV4c(
            repository_id=cp["repository_id"],
            checkpoint_source_id=cp["checkpoint_source_id"],
            checkpoint_id=cp["checkpoint_id"],
            sequence=cp["sequence"],
            snapshot_sha256=cp["snapshot_sha256"],
            context_sha256=cp["context_sha256"],
            checkpoint_sha256=_sha(checkpoint_raw),
            previous_checkpoint_id=cp["previous_checkpoint_id"],
            previous_snapshot_sha256=cp["previous_snapshot_sha256"],
            previous_context_sha256=cp["previous_context_sha256"],
            permitted_context_successors=tuple(cp["permitted_context_successors"]),
            revoked=cp["revoked"],
        )
        transition = propose_transition_v4c(
            previous_state,
            candidate,
            observed_checkpoint_before=observed_checkpoint_before,
            observed_checkpoint_after=observed_checkpoint_after,
        )
        _require(
            transition.technical_transition == "VALID",
            transition.reason_codes[0] if transition.reason_codes else "TRANSITION_REJECTED",
        )
        _require(transition.proposed_state is not None, "TRANSITION_REJECTED")
        return AptResultV4c("VALID", _BLOCKERS, transition.proposed_state)
    except V4cRefusal as exc:
        reason = exc.code
    except (InterruptedError, KeyboardInterrupt):
        reason = "OPERATION_INTERRUPTED"
    except MemoryError:
        reason = "LIMIT_EXCEEDED"
    except (
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        UnicodeError,
        OverflowError,
        RecursionError,
    ):
        reason = "FORMAT_REJECTED"
    except OSError:
        reason = "EVIDENCE_UNAVAILABLE"
    return AptResultV4c("REFUSED", (reason,))
