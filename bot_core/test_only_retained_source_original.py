"""S9-RPI-14: repeated exact source-original lookup, qualification-only.

This is a TEST-only contract harness. A caller-provided reader and expectation
are never installation provenance, evidence of fsync, or a production authority.
"""

from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator

from bot_core import authenticated_custody_mutation_source as source
from bot_core.independent_cca_decision import RetainedSourceEventReader

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")


class TestOnlyRetainedSourceError(RuntimeError):
    """Retained original, its exact binding or independent policy was unavailable."""


@dataclass(frozen=True, slots=True)
class SyntheticRetainedSourcePin:
    """External *synthetic* expectations; no self-discovery or TOFU is allowed."""

    installation_id: str
    operation_id: str
    execution_id: str
    namespace: str
    sequence: int
    previous_entry_sha256: str
    retained_envelope_sha256: str

    def __post_init__(self) -> None:
        for value in (
            self.installation_id,
            self.operation_id,
            self.execution_id,
            self.namespace,
        ):
            if type(value) is not str or _ID.fullmatch(value) is None:
                raise ValueError("exact bounded synthetic source identity required")
        if type(self.sequence) is not int or self.sequence <= 0:
            raise ValueError("positive source log position required")
        for digest in (self.previous_entry_sha256, self.retained_envelope_sha256):
            if type(digest) is not str or _SHA256.fullmatch(digest) is None:
                raise ValueError("lowercase source log SHA256 required")
        if self.sequence == 1 and self.previous_entry_sha256 != "0" * 64:
            raise ValueError("genesis source log must have zero previous digest")
        if self.sequence > 1 and self.previous_entry_sha256 == "0" * 64:
            raise ValueError("non-genesis source log requires previous digest")


@dataclass(frozen=True, slots=True)
class SyntheticRetainedSourceEvidence:
    """Never a qualified source, close reader, durable log or CCA decision."""

    original_envelope: bytes = field(repr=False)
    original_sha256: str
    policy_sha256: str
    status: str = field(default="NOT_QUALIFIED", init=False)
    production_authorization: str = field(default="BLOCKED", init=False)
    provenance_qualified: bool = field(default=False, init=False)
    durability_attested: bool = field(default=False, init=False)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise TestOnlyRetainedSourceError(message)


def _read_original(
    reader: RetainedSourceEventReader,
    pin: SyntheticRetainedSourcePin,
    claimed: bytes,
) -> bytes:
    retained = reader.lookup_original_event(
        installation_id=pin.installation_id,
        operation_id=pin.operation_id,
        execution_id=pin.execution_id,
    )
    _require(
        type(retained) is bytes and 0 < len(retained) <= source.MAX_ENVELOPE_BYTES,
        "bounded exact retained original required",
    )
    _require(
        hashlib.sha256(retained).hexdigest() == pin.retained_envelope_sha256
        and retained == claimed,
        "retained source original differs from independent expectation",
    )
    return retained


def _load_policy(
    loader: Callable[[], source.ExplicitSourceVerificationPolicy],
) -> source.ExplicitSourceVerificationPolicy:
    policy = loader()
    _require(
        type(policy) is source.ExplicitSourceVerificationPolicy,
        "exact independent source policy required",
    )
    return policy


def _check_binding(
    event: source.CustodyMutationExecutionV1,
    policy: source.ExplicitSourceVerificationPolicy,
    pin: SyntheticRetainedSourcePin,
) -> None:
    payload = event.payload
    position = payload["source_log_position"]
    expectation = {
        "namespace": pin.namespace,
        "sequence": pin.sequence,
        "previous_entry_sha256": pin.previous_entry_sha256,
        "entry_id": pin.execution_id,
    }
    _require(
        payload["installation"]["installation_id"] == pin.installation_id
        and payload["close"]["operation_id"] == pin.operation_id
        and payload["execution_id"] == pin.execution_id
        and position == expectation
        and policy.document["source_log_position"] == expectation
        and policy.document["execution_id"] == pin.execution_id,
        "source original log position or identity differs from external pin",
    )


@contextmanager
def inspect_synthetic_retained_source_original(
    *,
    pin: SyntheticRetainedSourcePin,
    claimed_original: bytes,
    reader: RetainedSourceEventReader,
    policy_loader: Callable[[], source.ExplicitSourceVerificationPolicy],
) -> Iterator[SyntheticRetainedSourceEvidence]:
    """Compare and re-read an original; never hold a real source custody lease.

    The caller must independently provide reader, policy and immutable pin.
    There is no trusted installed source, log floor, OS peer or durable receipt.
    """
    if type(pin) is not SyntheticRetainedSourcePin:
        raise TestOnlyRetainedSourceError("synthetic external source pin required")
    if (
        type(claimed_original) is not bytes
        or not 0 < len(claimed_original) <= source.MAX_ENVELOPE_BYTES
    ):
        raise TestOnlyRetainedSourceError("bounded source original bytes required")
    try:
        before = _load_policy(policy_loader)
        original = _read_original(reader, pin, claimed_original)
        event = source.parse_custody_mutation_execution(original)
        _check_binding(event, before, pin)
        verification = source.verify_custody_mutation_execution(
            event, policy_loader=policy_loader
        )
        _require(
            verification.status == "NOT_QUALIFIED"
            and verification.production_emission == "BLOCKED",
            "source signature match must not qualify execution",
        )
        _require(
            _load_policy(policy_loader).canonical_bytes == before.canonical_bytes,
            "independent source policy changed during verification",
        )
        _read_original(reader, pin, original)
    except Exception as exc:
        raise TestOnlyRetainedSourceError("retained source lookup denied") from exc

    yield SyntheticRetainedSourceEvidence(
        original_envelope=original,
        original_sha256=pin.retained_envelope_sha256,
        policy_sha256=hashlib.sha256(before.canonical_bytes).hexdigest(),
    )
    try:
        _read_original(reader, pin, original)
        _require(
            _load_policy(policy_loader).canonical_bytes == before.canonical_bytes,
            "independent source policy changed while held",
        )
    except Exception as exc:
        raise TestOnlyRetainedSourceError("retained source changed during consumer use") from exc
