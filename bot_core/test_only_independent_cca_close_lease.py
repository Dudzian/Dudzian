"""S9-RPI-12: TEST-only transaction-scoped CCA close read experiment.

Never a production reader, CCA lease, enrollment, qualification or admission
capability. The expected original is supplied by the isolated test harness.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import psycopg
from psycopg import sql
from psycopg.pq import TransactionStatus

from bot_core.postgresql_independent_cca_reader_inspection import (
    IndependentCCAReaderCatalogError,
    IndependentCCAReaderCatalogPin,
    inspect_independent_cca_reader_catalog,
)
from bot_core.test_only_cca_catalog_ddl_gate import (
    TestOnlyCCACatalogGateError,
    acquire_synthetic_reader_catalog_gate,
)

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_XID = re.compile(r"[1-9][0-9]*\Z")
_REQUEST_KEYS = frozenset(
    {
        "operation_id",
        "semantic_role",
        "identity",
        "generation",
        "predecessor_state",
        "target_state",
    }
)
_RESULT_KEYS = frozenset(
    {
        "operation_id",
        "semantic_role",
        "request_text",
        "request_sha256",
        "close_xid",
        "observation_xid",
        "phase",
    }
)


class TestOnlyCloseLeaseError(RuntimeError):
    """An isolated read/compare/transaction invariant did not hold."""


@dataclass(frozen=True, slots=True)
class SyntheticCloseExpectation:
    """Out-of-band TEST input, not a trusted production source envelope."""

    operation_id: str
    semantic_role: str
    identity: str
    generation: int
    predecessor_state: str
    target_state: str
    request_bytes: bytes = field(repr=False)
    request_sha256: str
    close_xid: str
    source_observation_xid: str

    def __post_init__(self) -> None:
        if type(self.operation_id) is not str or _ID.fullmatch(self.operation_id) is None:
            raise ValueError("exact synthetic operation ID required")
        if self.semantic_role != "ROOT_PROOF_SIGNING":
            raise ValueError("exact synthetic semantic role required")
        if type(self.identity) is not str or not self.identity or len(self.identity) > 512:
            raise ValueError("exact synthetic credential identity required")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("positive synthetic generation required")
        if self.predecessor_state != "ACTIVE" or self.target_state != "VERIFY_ONLY":
            raise ValueError("closed synthetic transition required")
        if type(self.request_bytes) is not bytes or not 0 < len(self.request_bytes) <= 16384:
            raise ValueError("bounded retained PostgreSQL UTF8 required")
        if type(self.request_sha256) is not str or _SHA256.fullmatch(self.request_sha256) is None:
            raise ValueError("lowercase retained SHA-256 required")
        for xid in (self.close_xid, self.source_observation_xid):
            if type(xid) is not str or _XID.fullmatch(xid) is None:
                raise ValueError("canonical separate source/close XIDs required")
        if self.close_xid == self.source_observation_xid:
            raise ValueError("source observation must not alias close commit")


@dataclass(frozen=True, slots=True)
class SyntheticHeldCloseEvidence:
    """Never authorizes signing, completion, issuance or a production decision."""

    request_bytes: bytes = field(repr=False)
    request_sha256: str
    close_xid: str
    observation_xid: str
    status: str = field(default="NOT_QUALIFIED", init=False)
    production_authorization: str = field(default="BLOCKED", init=False)
    close_lease_qualified: bool = field(default=False, init=False)


def _require(predicate: bool, message: str) -> None:
    if not predicate:
        raise TestOnlyCloseLeaseError(message)


def _compare_original(response: Any, expected: SyntheticCloseExpectation) -> str:
    _require(
        type(response) is dict and set(response) == _RESULT_KEYS,
        "CCA reader returned a changed or incomplete response",
    )
    for key in _RESULT_KEYS:
        _require(type(response[key]) is str, "CCA read result has non-string original fields")
    _require(
        response["operation_id"] == expected.operation_id
        and response["semantic_role"] == expected.semantic_role
        and response["phase"] == "PENDING_CLOSED",
        "CCA close identity or phase changed",
    )
    _require(
        response["request_sha256"] == expected.request_sha256
        and response["close_xid"] == expected.close_xid,
        "CCA close digest or transaction does not match source",
    )
    observed_xid = response["observation_xid"]
    _require(
        _XID.fullmatch(observed_xid) is not None
        and observed_xid not in {expected.close_xid, expected.source_observation_xid},
        "CCA observation must have its own independent transaction XID",
    )
    request_bytes = response["request_text"].encode("utf-8")
    _require(
        request_bytes == expected.request_bytes
        and hashlib.sha256(request_bytes).hexdigest() == expected.request_sha256,
        "original retained request bytes do not match authenticated source expectation",
    )
    try:
        document = json.loads(request_bytes.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise TestOnlyCloseLeaseError("invalid original PostgreSQL JSONB text") from exc
    _require(
        type(document) is dict
        and set(document) == _REQUEST_KEYS
        and type(document["generation"]) is int
        and document["generation"] > 0
        and document
        == {
            "operation_id": expected.operation_id,
            "semantic_role": expected.semantic_role,
            "identity": expected.identity,
            "generation": expected.generation,
            "predecessor_state": expected.predecessor_state,
            "target_state": expected.target_state,
        },
        "close request differs from expected identity/generation/state",
    )
    return observed_xid


@contextmanager
def hold_synthetic_committed_close(
    conn: psycopg.Connection[Any],
    *,
    catalog_pin: IndependentCCAReaderCatalogPin,
    expected: SyntheticCloseExpectation,
) -> Iterator[SyntheticHeldCloseEvidence]:
    """Inspect then lock/read/compare in ONE caller-owned transaction.

    The caller keeps its transaction open around this context manager.
    The returned bytes are never a CCA authority or production credential.
    A future production implementation needs independently installed pins,
    SQL catalogue migration and protection against concurrent catalog DDL.
    """
    if (
        type(catalog_pin) is not IndependentCCAReaderCatalogPin
        or type(expected) is not SyntheticCloseExpectation
    ):
        raise TestOnlyCloseLeaseError("exact synthetic test inputs required")
    try:
        # Advisory locks are cooperative: this serializes only reviewed test DDL.
        acquire_synthetic_reader_catalog_gate(conn, catalog_pin)
        inspect_independent_cca_reader_catalog(conn, catalog_pin)
        rows = conn.execute(
            sql.SQL("SELECT {}.read_exact_close(%s)").format(sql.Identifier(catalog_pin.schema)),
            (expected.operation_id,),
        ).fetchall()
        _require(len(rows) == 1, "exactly one committed close result required")
        observation_xid = _compare_original(rows[0][0], expected)
        # Detect a non-cooperating DDL writer that drifted after the first
        # inspection. This does NOT prevent a fast change-and-restore (ABA).
        inspect_independent_cca_reader_catalog(conn, catalog_pin)
        live_xid = conn.execute("SELECT pg_catalog.pg_current_xact_id()::text").fetchone()[0]
        _require(live_xid == observation_xid, "read escaped the held CCA transaction")
    except (
        psycopg.Error,
        IndependentCCAReaderCatalogError,
        TestOnlyCCACatalogGateError,
    ) as exc:
        raise TestOnlyCloseLeaseError("CCA test-only close read denied") from exc

    yield SyntheticHeldCloseEvidence(
        request_bytes=expected.request_bytes,
        request_sha256=expected.request_sha256,
        close_xid=expected.close_xid,
        observation_xid=observation_xid,
    )
    try:
        _require(
            conn.info.transaction_status is TransactionStatus.INTRANS,
            "CCA caller transaction ended before close comparison completed",
        )
        live_xid = conn.execute("SELECT pg_catalog.pg_current_xact_id()::text").fetchone()[0]
        _require(
            live_xid == observation_xid,
            "CCA caller transaction changed during synthetic evidence hold",
        )
        # Detect committed drift during the consumer callback before allowing
        # a successful exit. The result itself is still strictly TEST_ONLY.
        inspect_independent_cca_reader_catalog(conn, catalog_pin)
    except (psycopg.Error, IndependentCCAReaderCatalogError) as exc:
        raise TestOnlyCloseLeaseError("CCA close transaction or catalog drifted") from exc
