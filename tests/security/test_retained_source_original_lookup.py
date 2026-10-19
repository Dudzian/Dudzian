"""S9-RPI-14: fake retained source event lookup; all outcomes stay NO-GO."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace

import pytest

from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.test_only_retained_source_original import (
    SyntheticRetainedSourcePin,
    TestOnlyRetainedSourceError,
    inspect_synthetic_retained_source_original,
)
from tests.architecture.test_stage9_atomic_root_proof_commit_boundary import (
    _repository_dependencies,
)
from tests.security.test_authenticated_custody_mutation_source import (
    source_case,
)
from tests.security.test_independent_custody_completion_qualification import (
    completion as completion,
)


class _SyntheticReader:
    def __init__(self, event: bytes):
        self.original = event
        self.lookups = 0
        self.history: list[tuple[str, str, str]] = []
        self.swap_on_second: bytes | None = None

    def lookup_original_event(
        self, *, installation_id: str, operation_id: str, execution_id: str
    ) -> bytes:
        self.lookups += 1
        self.history.append((installation_id, operation_id, execution_id))
        if self.lookups == 2 and self.swap_on_second is not None:
            self.original = self.swap_on_second
        return self.original


def _test_pin(case) -> SyntheticRetainedSourcePin:
    payload = case.event.payload
    pos = payload["source_log_position"]
    return SyntheticRetainedSourcePin(
        installation_id=payload["installation"]["installation_id"],
        operation_id=payload["close"]["operation_id"],
        execution_id=payload["execution_id"],
        namespace=pos["namespace"],
        sequence=pos["sequence"],
        previous_entry_sha256=pos["previous_entry_sha256"],
        retained_envelope_sha256=hashlib.sha256(case.envelope).hexdigest(),
    )


def _inspect(case, reader, pin=None, *, claimed=None, policy_loader=None):
    return inspect_synthetic_retained_source_original(
        pin=_test_pin(case) if pin is None else pin,
        claimed_original=case.envelope if claimed is None else claimed,
        reader=reader,
        policy_loader=(lambda: case.policy) if policy_loader is None else policy_loader,
    )


def test_exact_original_retrieval_rechecked_before_and_after_consumer(source_case):
    case = source_case
    reader = _SyntheticReader(case.envelope)
    with _inspect(case, reader) as evidence:
        assert evidence.status == "NOT_QUALIFIED"
        assert evidence.production_authorization == "BLOCKED"
        assert evidence.provenance_qualified is False
        assert evidence.durability_attested is False
        assert evidence.original_envelope == case.envelope
        assert evidence.original_sha256 == hashlib.sha256(case.envelope).hexdigest()
        assert reader.lookups == 2
    assert reader.lookups == 3
    assert reader.history == [
        (
            case.payload["installation"]["installation_id"],
            case.payload["close"]["operation_id"],
            case.payload["execution_id"],
        )
    ] * 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("operation_id", "unknown"),
        ("execution_id", "other-execution"),
        ("installation_id", "other-installation"),
        ("namespace", "other-source-log"),
        ("sequence", 2),
        ("previous_entry_sha256", "f" * 64),
        ("retained_envelope_sha256", "f" * 64),
    ],
)
def test_pin_substitution_is_denied(source_case, field, value):
    pin = replace(_test_pin(source_case), **{field: value})
    with pytest.raises(TestOnlyRetainedSourceError, match="lookup denied"):
        with _inspect(source_case, _SyntheticReader(source_case.envelope), pin):
            pytest.fail("modified pin cannot authorize original")


def test_caller_supplied_envelope_cannot_replace_retained_original(source_case):
    with pytest.raises(TestOnlyRetainedSourceError, match="lookup denied"):
        with _inspect(
            source_case, _SyntheticReader(source_case.envelope), claimed=b"{}"
        ):
            pytest.fail("caller original differs from retained bytes")


def test_source_signature_is_verified_against_independent_policy(source_case):
    envelope = json.loads(source_case.envelope)
    envelope["signature_b64"] = base64.b64encode(bytes(64)).decode("ascii")
    altered = canonical_json_bytes(envelope)
    pin = replace(
        _test_pin(source_case), retained_envelope_sha256=hashlib.sha256(altered).hexdigest()
    )
    with pytest.raises(TestOnlyRetainedSourceError, match="lookup denied"):
        with _inspect(
            source_case, _SyntheticReader(altered), pin, claimed=altered
        ):
            pytest.fail("matching retention hash is not signature verification")


def test_source_original_drift_before_yield_is_denied(source_case):
    reader = _SyntheticReader(source_case.envelope)
    reader.swap_on_second = b"{}"
    with pytest.raises(TestOnlyRetainedSourceError, match="lookup denied"):
        with _inspect(source_case, reader):
            pytest.fail("second lookup must detect retention replacement")
    assert reader.lookups == 2


def test_source_original_drift_during_consumer_is_denied(source_case):
    reader = _SyntheticReader(source_case.envelope)
    with pytest.raises(TestOnlyRetainedSourceError, match="changed during consumer"):
        with _inspect(source_case, reader) as evidence:
            assert evidence.provenance_qualified is False
            reader.original = b"{}"
    assert reader.lookups == 3


def test_independent_policy_revoke_during_consumer_is_denied(source_case):
    reader = _SyntheticReader(source_case.envelope)
    policy = source_case.policy
    revoked = False

    def load():
        return None if revoked else policy

    with pytest.raises(TestOnlyRetainedSourceError, match="changed during consumer"):
        with _inspect(source_case, reader, policy_loader=load):
            revoked = True


def test_missing_retained_original_or_invalid_reader_output_fail_closed(source_case):
    for missing in (None, "", b"", b"invalid", bytearray(source_case.envelope)):
        reader = _SyntheticReader(source_case.envelope)
        reader.original = missing
        with pytest.raises(TestOnlyRetainedSourceError, match="lookup denied"):
            with _inspect(source_case, reader):
                pytest.fail("unavailable or malformed retained source original")


def test_production_verifier_import_graph_excludes_test_only_retention_harness():
    pending = ["bot_core.independent_cca_decision"]
    visited = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        assert module != "bot_core.test_only_retained_source_original"
        pending.extend(_repository_dependencies(module) - visited)
