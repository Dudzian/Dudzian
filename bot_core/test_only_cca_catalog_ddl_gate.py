"""S9-RPI-13: advisory catalog DDL serialization experiment, TEST-only.

Only *cooperating* reviewed DDL writers acquire the exclusive gate. Ordinary
PostgreSQL ALTER/GRANT statements do not automatically respect advisory locks.
Neither a successful gate nor a catalog check is a production security proof.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import psycopg
from psycopg.pq import TransactionStatus

from bot_core.postgresql_independent_cca_reader_inspection import (
    IndependentCCAReaderCatalogPin,
)

_DOMAIN = b"CryptoHunter/TestOnlyCCAReaderDDL/v1\0"


class TestOnlyCCACatalogGateError(RuntimeError):
    """A synthetic CCA catalog DDL gate could not be held in this transaction."""

    __test__ = False  # Exception contract, not a pytest test class.


def _keys(pin: IndependentCCAReaderCatalogPin) -> tuple[int, int]:
    if type(pin) is not IndependentCCAReaderCatalogPin:
        raise TestOnlyCCACatalogGateError("exact pinned test-only CCA catalog required")
    physical = pin.physical_identity
    payload = (
        f"{physical.system_identifier}:{physical.database_oid}:"
        f"{physical.instance_id}:{pin.schema}:{pin.function_oid}"
    ).encode("utf-8")
    return struct.unpack("!ii", hashlib.sha256(_DOMAIN + payload).digest()[:8])


def _require_transaction(conn: psycopg.Connection[Any]) -> None:
    if (
        conn.info.server_version // 10000 != 16
        or conn.info.transaction_status is not TransactionStatus.INTRANS
    ):
        raise TestOnlyCCACatalogGateError("PostgreSQL 16 active caller transaction required")


def acquire_synthetic_reader_catalog_gate(
    conn: psycopg.Connection[Any], pin: IndependentCCAReaderCatalogPin
) -> None:
    """Hold shared test-only DDL gate until caller transaction COMMIT/ROLLBACK."""
    _require_transaction(conn)
    try:
        conn.execute(
            "SELECT pg_catalog.pg_advisory_xact_lock_shared(%s::integer,%s::integer)",
            _keys(pin),
        )
    except psycopg.Error as exc:
        raise TestOnlyCCACatalogGateError("shared synthetic CCA catalog gate unavailable") from exc


def acquire_synthetic_reviewed_ddl_gate(
    conn: psycopg.Connection[Any], pin: IndependentCCAReaderCatalogPin
) -> None:
    """Exclusive cooperating migration gate, never an enforcement of arbitrary DDL."""
    _require_transaction(conn)
    try:
        conn.execute(
            "SELECT pg_catalog.pg_advisory_xact_lock(%s::integer,%s::integer)",
            _keys(pin),
        )
    except psycopg.Error as exc:
        raise TestOnlyCCACatalogGateError(
            "exclusive reviewed synthetic DDL gate unavailable"
        ) from exc


__all__ = [
    "TestOnlyCCACatalogGateError",
    "acquire_synthetic_reader_catalog_gate",
    "acquire_synthetic_reviewed_ddl_gate",
]
