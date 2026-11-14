"""Nonproduction antirollback semantics and atomic refusal of malformed state."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, replace
from typing import Any

import pytest

from deployment import stage9_observer_checkpoint_v4c as checkpoint

pytestmark = [pytest.mark.security, pytest.mark.unit]


@pytest.fixture
def state() -> checkpoint.SnapshotStateV4c:
    return checkpoint.SnapshotStateV4c(
        repository_id="TEST-repository",
        checkpoint_source_id="TEST-checkpoint-source",
        checkpoint_id="TEST-checkpoint-7",
        sequence=7,
        snapshot_sha256="1" * 64,
        context_sha256="2" * 64,
        checkpoint_sha256="3" * 64,
    )


@pytest.fixture
def candidate(state: checkpoint.SnapshotStateV4c) -> checkpoint.SnapshotCandidateV4c:
    return checkpoint.SnapshotCandidateV4c(
        repository_id=state.repository_id,
        checkpoint_source_id=state.checkpoint_source_id,
        checkpoint_id="TEST-checkpoint-8",
        sequence=8,
        snapshot_sha256="4" * 64,
        context_sha256=state.context_sha256,
        checkpoint_sha256="5" * 64,
        previous_checkpoint_id=state.checkpoint_id,
        previous_snapshot_sha256=state.snapshot_sha256,
        previous_context_sha256=state.context_sha256,
    )


def _propose(
    state: checkpoint.SnapshotStateV4c,
    candidate: checkpoint.SnapshotCandidateV4c,
    **kwargs: Any,
) -> checkpoint.TransitionResultV4c:
    observations = {
        "observed_checkpoint_before": state.checkpoint_sha256,
        "observed_checkpoint_after": state.checkpoint_sha256,
    }
    observations.update(kwargs)
    return checkpoint.propose_transition_v4c(state, candidate, **observations)


def _refused(result: checkpoint.TransitionResultV4c, code: str) -> None:
    assert result.technical_transition == "REFUSED"
    assert result.reason_codes == (code,)
    assert result.proposed_state is None
    assert result.source_acceptance == "UNVERIFIED"
    assert result.durable_state == "BLOCKED"
    assert result.authorization == "BLOCKED"


def test_exact_successor_only_proposes_immutable_unauthorized_state(state, candidate):
    original = asdict(state)
    result = _propose(state, candidate)
    assert result.technical_transition == "VALID"
    assert result.reason_codes == ()
    assert result.proposed_state is not None
    assert result.proposed_state.sequence == 8
    assert result.proposed_state.snapshot_sha256 == candidate.snapshot_sha256
    assert result.proposed_state.checkpoint_sha256 == candidate.checkpoint_sha256
    assert result.source_acceptance == "UNVERIFIED"
    assert result.durable_state == result.authorization == "BLOCKED"
    assert asdict(state) == original
    with pytest.raises(FrozenInstanceError):
        result.proposed_state.sequence = 9
    with pytest.raises(FrozenInstanceError):
        result.technical_transition = "REFUSED"


def test_exact_known_checkpoint_is_idempotent(state, candidate):
    replay = replace(
        candidate,
        checkpoint_id=state.checkpoint_id,
        sequence=state.sequence,
        snapshot_sha256=state.snapshot_sha256,
        checkpoint_sha256=state.checkpoint_sha256,
        previous_checkpoint_id=None,
        previous_snapshot_sha256=None,
        previous_context_sha256=None,
    )
    result = _propose(state, replay)
    assert result.technical_transition == "VALID"
    assert result.proposed_state is state
    assert result.source_acceptance == "UNVERIFIED"


def test_unknown_state_never_bootstraps(candidate):
    result = checkpoint.propose_transition_v4c(None, candidate)
    _refused(result, "CHECKPOINT_STATE_UNKNOWN")


@pytest.mark.parametrize(
    ("field", "replacement", "code"),
    [
        ("sequence", 6, "SNAPSHOT_ROLLBACK"),
        ("sequence", 7, "SNAPSHOT_EQUIVOCATION"),
        ("sequence", 9, "SNAPSHOT_SUCCESSOR_SEQUENCE_REJECTED"),
        ("repository_id", "TEST-other-repository", "CHECKPOINT_REPOSITORY_MISMATCH"),
        ("checkpoint_source_id", "TEST-other-source", "CHECKPOINT_SOURCE_MISMATCH"),
        ("previous_checkpoint_id", "TEST-other-checkpoint", "SNAPSHOT_SUCCESSOR_PROOF_REJECTED"),
        ("previous_snapshot_sha256", "9" * 64, "SNAPSHOT_SUCCESSOR_PROOF_REJECTED"),
        ("previous_context_sha256", "9" * 64, "SNAPSHOT_SUCCESSOR_PROOF_REJECTED"),
        ("checkpoint_id", "TEST-checkpoint-7", "CHECKPOINT_SUCCESSOR_IDENTITY_REJECTED"),
        ("checkpoint_sha256", "3" * 64, "CHECKPOINT_SUCCESSOR_IDENTITY_REJECTED"),
        ("context_sha256", "9" * 64, "CONTEXT_ROTATION_REJECTED"),
        ("revoked", True, "CHECKPOINT_TRUST_REVOKED"),
    ],
)
def test_refusals_are_atomic_and_preserve_predecessor(state, candidate, field, replacement, code):
    original = asdict(state)
    _refused(_propose(state, replace(candidate, **{field: replacement})), code)
    assert asdict(state) == original


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("repository_id", ""),
        ("repository_id", "x" * 257),
        ("repository_id", "TEST repo"),
        ("repository_id", 1),
        ("checkpoint_source_id", "TEST-source\n"),
        ("checkpoint_id", None),
        ("sequence", True),
        ("sequence", -1),
        ("sequence", 9_007_199_254_740_992),
        ("sequence", 7.0),
        ("sequence", float("nan")),
        ("sequence", float("inf")),
        ("snapshot_sha256", "A" * 64),
        ("snapshot_sha256", "0" * 64),
        ("snapshot_sha256", "a" * 63),
        ("context_sha256", None),
        ("checkpoint_sha256", b"3" * 64),
        ("permitted_context_successors", []),
        ("permitted_context_successors", ("a" * 64,) * 33),
        ("permitted_context_successors", ("a" * 64, "a" * 64)),
        ("permitted_context_successors", ("b" * 64, "a" * 64)),
        ("permitted_context_successors", ("2" * 64,)),
        ("permitted_context_successors", ("BAD",)),
        ("revoked", 0),
    ],
)
@pytest.mark.parametrize("record_kind", ["state", "candidate"])
def test_exact_types_limits_and_closed_rules(state, candidate, record_kind, field, replacement):
    if record_kind == "state":
        bad_state = replace(state, **{field: replacement})
        result = _propose(bad_state, candidate)
        code = "CHECKPOINT_STATE_REJECTED"
    else:
        result = _propose(state, replace(candidate, **{field: replacement}))
        code = "SNAPSHOT_CANDIDATE_REJECTED"
    _refused(result, code)


@pytest.mark.parametrize(
    "references",
    [
        {"previous_checkpoint_id": None},
        {"previous_snapshot_sha256": None},
        {"previous_context_sha256": None},
        {"previous_checkpoint_id": ""},
        {"previous_snapshot_sha256": "A" * 64},
        {"previous_context_sha256": 1},
    ],
)
def test_malformed_or_partial_predecessor_references_refuse(state, candidate, references):
    _refused(
        _propose(state, replace(candidate, **references)),
        "SNAPSHOT_CANDIDATE_REJECTED",
    )


def test_missing_all_predecessor_proofs_never_allows_new_sequence(state, candidate):
    missing = replace(
        candidate,
        previous_checkpoint_id=None,
        previous_snapshot_sha256=None,
        previous_context_sha256=None,
    )
    _refused(_propose(state, missing), "SNAPSHOT_SUCCESSOR_PROOF_REJECTED")


@pytest.mark.parametrize(
    "field",
    ["context_sha256", "checkpoint_sha256", "checkpoint_id", "permitted_context_successors"],
)
def test_same_sequence_cannot_rotate_context_or_rebind_checkpoint(state, candidate, field):
    replay = replace(
        candidate,
        sequence=state.sequence,
        snapshot_sha256=state.snapshot_sha256,
        checkpoint_id=state.checkpoint_id,
        checkpoint_sha256=state.checkpoint_sha256,
    )
    replacement: Any = ("a" * 64,) if field == "permitted_context_successors" else "a" * 64
    _refused(
        _propose(state, replace(replay, **{field: replacement})),
        "CHECKPOINT_EQUIVOCATION",
    )


@pytest.mark.parametrize("side", ["observed_checkpoint_before", "observed_checkpoint_after"])
@pytest.mark.parametrize("value", [None, "x", "6" * 64])
def test_checkpoint_change_missing_or_corruption_refuses(state, candidate, side, value):
    expected = {
        None: "CHECKPOINT_OBSERVATION_MISSING",
        "x": "CHECKPOINT_OBSERVATION_REJECTED",
        "6" * 64: "CHECKPOINT_CHANGED",
    }[value]
    _refused(_propose(state, candidate, **{side: value}), expected)


def test_equal_arbitrary_checkpoint_tokens_do_not_establish_acceptance(state, candidate):
    _refused(
        _propose(
            state,
            candidate,
            observed_checkpoint_before="9" * 64,
            observed_checkpoint_after="9" * 64,
        ),
        "CHECKPOINT_CHANGED",
    )


def test_checkpoint_reads_are_required(state, candidate):
    _refused(
        checkpoint.propose_transition_v4c(state, candidate),
        "CHECKPOINT_OBSERVATION_MISSING",
    )


def test_revoked_predecessor_refuses_every_transition(state, candidate):
    _refused(_propose(replace(state, revoked=True), candidate), "CHECKPOINT_TRUST_REVOKED")


def test_explicit_rotation_is_technical_only_and_consumes_old_grant(state, candidate):
    old = replace(state, permitted_context_successors=("9" * 64,))
    new = replace(candidate, context_sha256="9" * 64)
    result = _propose(old, new)
    assert result.technical_transition == "VALID"
    assert result.proposed_state is not None
    assert result.proposed_state.context_sha256 == "9" * 64
    assert result.proposed_state.permitted_context_successors == ()
    assert old.permitted_context_successors == ("9" * 64,)
    assert result.source_acceptance == "UNVERIFIED"


def test_next_state_rotation_rules_come_only_from_candidate(state, candidate):
    old = replace(state, permitted_context_successors=("9" * 64,))
    new = replace(candidate, permitted_context_successors=("a" * 64,))
    result = _propose(old, new)
    assert result.proposed_state is not None
    assert result.proposed_state.permitted_context_successors == ("a" * 64,)


@pytest.mark.parametrize("interrupt", [InterruptedError, KeyboardInterrupt])
def test_interrupted_operation_returns_no_partial_proposal(
    state, candidate, monkeypatch, interrupt
):
    original = asdict(state)

    def interrupted(*args):
        raise interrupt()

    monkeypatch.setattr(checkpoint, "_record", interrupted)
    _refused(_propose(state, candidate), "TRANSITION_INTERRUPTED")
    assert asdict(state) == original


def test_authorization_and_transition_rules_cannot_be_constructor_switches():
    with pytest.raises(TypeError):
        checkpoint.TransitionResultV4c("VALID", (), authorization="VERIFIED")
    with pytest.raises(TypeError):
        checkpoint.SnapshotStateV4c(
            "TEST-repo",
            "TEST-source",
            "TEST-cp",
            0,
            "1" * 64,
            "2" * 64,
            "3" * 64,
            successor_policy="ALLOW_ANY_SEQUENCE",
        )


def test_same_sequence_cannot_inherit_another_repository_or_source(state, candidate):
    replay = replace(
        candidate,
        sequence=state.sequence,
        snapshot_sha256=state.snapshot_sha256,
        checkpoint_id=state.checkpoint_id,
        checkpoint_sha256=state.checkpoint_sha256,
    )
    _refused(
        _propose(state, replace(replay, repository_id="TEST-other-repository")),
        "CHECKPOINT_REPOSITORY_MISMATCH",
    )
    _refused(
        _propose(state, replace(replay, checkpoint_source_id="TEST-other-source")),
        "CHECKPOINT_SOURCE_MISMATCH",
    )
