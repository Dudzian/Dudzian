"""Pure nonproduction APT checkpoint transitions; never a durable trust store.

Inputs are caller-supplied observations. Cryptographic binding and external
source acceptance belong to the surrounding V4c boundary. A valid transition
only proposes immutable data; it cannot enroll a source, authenticate an input,
write a file, or authorize installation. The caller must independently verify
both the predecessor and candidate signed checkpoints before using this model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

_MAX_SEQUENCE = 9_007_199_254_740_991
_MAX_ROTATIONS = 32
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+#=-]{0,255}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class SnapshotStateV4c:
    """A captured predecessor; presence alone conveys no source acceptance."""

    repository_id: str
    checkpoint_source_id: str
    checkpoint_id: str
    sequence: int
    snapshot_sha256: str
    context_sha256: str
    checkpoint_sha256: str
    permitted_context_successors: tuple[str, ...] = ()
    revoked: bool = False
    successor_policy: Literal["EXACT_NEXT_SEQUENCE"] = field(
        default="EXACT_NEXT_SEQUENCE", init=False
    )
    rotation_policy: Literal["EXPLICIT_CONTEXT_LINK"] = field(
        default="EXPLICIT_CONTEXT_LINK", init=False
    )


@dataclass(frozen=True, slots=True)
class SnapshotCandidateV4c:
    """Signed candidate fields after separate envelope verification.

    Predecessor references must all be present for a successor. A missing
    predecessor is permitted only for exact idempotent replay of a fully bound
    known checkpoint; it never permits bootstrap from an unknown state.
    """

    repository_id: str
    checkpoint_source_id: str
    checkpoint_id: str
    sequence: int
    snapshot_sha256: str
    context_sha256: str
    checkpoint_sha256: str
    previous_checkpoint_id: str | None
    previous_snapshot_sha256: str | None
    previous_context_sha256: str | None
    permitted_context_successors: tuple[str, ...] = ()
    revoked: bool = False
    successor_policy: Literal["EXACT_NEXT_SEQUENCE"] = field(
        default="EXACT_NEXT_SEQUENCE", init=False
    )
    rotation_policy: Literal["EXPLICIT_CONTEXT_LINK"] = field(
        default="EXPLICIT_CONTEXT_LINK", init=False
    )


@dataclass(frozen=True, slots=True)
class TransitionResultV4c:
    """Atomic technical result; no source or persistence authorization."""

    technical_transition: Literal["VALID", "REFUSED"]
    reason_codes: tuple[str, ...]
    proposed_state: SnapshotStateV4c | None = None
    source_acceptance: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)
    durable_state: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    authorization: Literal["BLOCKED"] = field(default="BLOCKED", init=False)


class _TransitionRefusal(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise _TransitionRefusal(code)


def _label(value: object, code: str) -> None:
    _require(type(value) is str and _LABEL.fullmatch(value) is not None, code)


def _digest(value: object, code: str) -> None:
    _require(
        type(value) is str and _DIGEST.fullmatch(value) is not None and value != "0" * 64,
        code,
    )


def _record(value: SnapshotStateV4c | SnapshotCandidateV4c, code: str) -> None:
    for label in (value.repository_id, value.checkpoint_source_id, value.checkpoint_id):
        _label(label, code)
    _require(type(value.sequence) is int and 0 <= value.sequence <= _MAX_SEQUENCE, code)
    for digest in (value.snapshot_sha256, value.context_sha256, value.checkpoint_sha256):
        _digest(digest, code)
    _require(type(value.revoked) is bool, code)
    _require(value.successor_policy == "EXACT_NEXT_SEQUENCE", code)
    _require(value.rotation_policy == "EXPLICIT_CONTEXT_LINK", code)
    successors = value.permitted_context_successors
    _require(type(successors) is tuple and len(successors) <= _MAX_ROTATIONS, code)
    for digest in successors:
        _digest(digest, code)
    _require(tuple(sorted(set(successors))) == successors, code)
    _require(value.context_sha256 not in successors, code)


def _candidate_references(value: SnapshotCandidateV4c) -> None:
    code = "SNAPSHOT_CANDIDATE_REJECTED"
    references = (
        value.previous_checkpoint_id,
        value.previous_snapshot_sha256,
        value.previous_context_sha256,
    )
    if all(reference is None for reference in references):
        return
    _require(all(reference is not None for reference in references), code)
    _label(value.previous_checkpoint_id, code)
    _digest(value.previous_snapshot_sha256, code)
    _digest(value.previous_context_sha256, code)


def _propose(
    state: SnapshotStateV4c | None,
    candidate: SnapshotCandidateV4c,
    before: str | None,
    after: str | None,
) -> SnapshotStateV4c:
    _require(state is not None, "CHECKPOINT_STATE_UNKNOWN")
    _require(type(state) is SnapshotStateV4c, "CHECKPOINT_STATE_REJECTED")
    assert state is not None
    _record(state, "CHECKPOINT_STATE_REJECTED")
    _require(type(candidate) is SnapshotCandidateV4c, "SNAPSHOT_CANDIDATE_REJECTED")
    _record(candidate, "SNAPSHOT_CANDIDATE_REJECTED")
    _candidate_references(candidate)
    _require(before is not None and after is not None, "CHECKPOINT_OBSERVATION_MISSING")
    _digest(before, "CHECKPOINT_OBSERVATION_REJECTED")
    _digest(after, "CHECKPOINT_OBSERVATION_REJECTED")
    _require(before == after == state.checkpoint_sha256, "CHECKPOINT_CHANGED")
    _require(not state.revoked and not candidate.revoked, "CHECKPOINT_TRUST_REVOKED")
    _require(candidate.repository_id == state.repository_id, "CHECKPOINT_REPOSITORY_MISMATCH")
    _require(
        candidate.checkpoint_source_id == state.checkpoint_source_id,
        "CHECKPOINT_SOURCE_MISMATCH",
    )
    _require(candidate.sequence >= state.sequence, "SNAPSHOT_ROLLBACK")
    if candidate.sequence == state.sequence:
        _require(candidate.snapshot_sha256 == state.snapshot_sha256, "SNAPSHOT_EQUIVOCATION")
        _require(
            candidate.context_sha256 == state.context_sha256
            and candidate.checkpoint_id == state.checkpoint_id
            and candidate.checkpoint_sha256 == state.checkpoint_sha256
            and candidate.permitted_context_successors == state.permitted_context_successors,
            "CHECKPOINT_EQUIVOCATION",
        )
        return state
    _require(candidate.sequence == state.sequence + 1, "SNAPSHOT_SUCCESSOR_SEQUENCE_REJECTED")
    _require(
        candidate.previous_checkpoint_id == state.checkpoint_id
        and candidate.previous_snapshot_sha256 == state.snapshot_sha256
        and candidate.previous_context_sha256 == state.context_sha256,
        "SNAPSHOT_SUCCESSOR_PROOF_REJECTED",
    )
    _require(
        candidate.checkpoint_id != state.checkpoint_id
        and candidate.checkpoint_sha256 != state.checkpoint_sha256,
        "CHECKPOINT_SUCCESSOR_IDENTITY_REJECTED",
    )
    _require(
        candidate.context_sha256 == state.context_sha256
        or candidate.context_sha256 in state.permitted_context_successors,
        "CONTEXT_ROTATION_REJECTED",
    )
    return SnapshotStateV4c(
        repository_id=candidate.repository_id,
        checkpoint_source_id=candidate.checkpoint_source_id,
        checkpoint_id=candidate.checkpoint_id,
        sequence=candidate.sequence,
        snapshot_sha256=candidate.snapshot_sha256,
        context_sha256=candidate.context_sha256,
        checkpoint_sha256=candidate.checkpoint_sha256,
        permitted_context_successors=candidate.permitted_context_successors,
        revoked=False,
    )


def propose_transition_v4c(
    state: SnapshotStateV4c | None,
    candidate: SnapshotCandidateV4c,
    *,
    observed_checkpoint_before: str | None = None,
    observed_checkpoint_after: str | None = None,
) -> TransitionResultV4c:
    """Propose an exact successor without mutating or persisting accepted state.

    Observation digests are compared with the captured predecessor digest.
    Equal arbitrary tokens never establish authenticity. Signature verification
    and independent source enrollment remain separate obligations. No genesis,
    source replacement, automatic context rotation, skipped sequence or write is
    performed. Refusal/interruption returns no partial candidate state.
    """

    try:
        proposed = _propose(state, candidate, observed_checkpoint_before, observed_checkpoint_after)
    except _TransitionRefusal as exc:
        return TransitionResultV4c("REFUSED", (exc.code,))
    except (InterruptedError, KeyboardInterrupt):
        return TransitionResultV4c("REFUSED", ("TRANSITION_INTERRUPTED",))
    return TransitionResultV4c("VALID", (), proposed)
