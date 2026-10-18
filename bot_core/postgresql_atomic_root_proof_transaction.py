"""Connection-owned final transaction discipline for Root Proof persistence.

This observation is reusable by independently qualified lifecycle authorities.
It grants no writer rights, accepts no callback, and never begins, commits or
retries a transaction. Production write composition remains separately gated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.pq import TransactionStatus

_OBSERVED_ORIGIN = object()


class AtomicRootProofTransactionError(RuntimeError):
    """The same direct-login backend no longer owns the final transaction."""


@dataclass(frozen=True, slots=True)
class _TransactionSnapshot:
    backend_pid: int
    transaction_id: str
    database_oid: int
    database_name: str
    login_oid: int
    login: str


@dataclass(frozen=True, slots=True, eq=False)
class AtomicRootProofTransactionOwner:
    """Observed transaction continuity, never a production authorization."""

    _connection: psycopg.Connection[Any] = field(repr=False)
    _snapshot: _TransactionSnapshot = field(repr=False)
    _origin: object = field(repr=False, default=None)

    def __repr__(self) -> str:
        return "AtomicRootProofTransactionOwner(connection=<redacted>, authorization=False)"


def _observe(connection: psycopg.Connection[Any], expected_login: str) -> _TransactionSnapshot:
    if (
        type(connection) is not psycopg.Connection
        or connection.closed
        or connection.info.transaction_status is not TransactionStatus.INTRANS
        or type(expected_login) is not str
        or not expected_login
    ):
        raise AtomicRootProofTransactionError("OPEN_FINAL_TRANSACTION_REQUIRED")
    try:
        row = connection.execute(
            "SELECT pg_catalog.pg_backend_pid(),CASE WHEN "
            "pg_catalog.current_setting('transaction_isolation')='serializable' AND "
            "pg_catalog.current_setting('transaction_read_only')='off' "
            "THEN pg_catalog.pg_current_xact_id()::text END,"
            "d.oid,d.datname,r.oid,session_user::text,current_user::text,"
            "pg_catalog.current_setting('transaction_isolation'),"
            "pg_catalog.current_setting('transaction_read_only') "
            "FROM pg_catalog.pg_database d,pg_catalog.pg_roles r "
            "WHERE d.datname=pg_catalog.current_database() AND r.rolname=session_user"
        ).fetchone()
        if (
            row is None
            or len(row) != 9
            or any(type(row[index]) is not int or row[index] < 1 for index in (0, 2, 4))
            or type(row[1]) is not str
            or not row[1].isascii()
            or not row[1].isdecimal()
            or int(row[1]) < 1
            or type(row[3]) is not str
            or not row[3]
            or row[5] != expected_login
            or row[6] != expected_login
            or row[7] != "serializable"
            or row[8] != "off"
            or row[0] != connection.info.backend_pid
        ):
            raise AtomicRootProofTransactionError("DIRECT_LOGIN_WRITABLE_SERIALIZABLE_REQUIRED")
        return _TransactionSnapshot(row[0], row[1], row[2], row[3], row[4], row[5])
    except AtomicRootProofTransactionError:
        raise
    except Exception:
        raise AtomicRootProofTransactionError("FINAL_TRANSACTION_OBSERVATION_UNAVAILABLE") from None


def capture_atomic_root_proof_transaction_owner(
    connection: psycopg.Connection[Any], *, expected_login: str
) -> AtomicRootProofTransactionOwner:
    """Pin a live final transaction after its independent authority qualification.

    The physical deployment, catalogs, ACLs and lifecycle eligibility must still
    be qualified by their owners. This token establishes only continuity of the
    supplied connection, direct login, database namespace and PostgreSQL xid.
    It does not attest row-lock retention across caller-owned savepoint rollback;
    the final write coordinator must own all statements in that transaction.
    """
    return AtomicRootProofTransactionOwner(
        connection, _observe(connection, expected_login), _OBSERVED_ORIGIN
    )


def validate_atomic_root_proof_transaction_owner(
    connection: psycopg.Connection[Any], owner: AtomicRootProofTransactionOwner
) -> None:
    """Refuse connection replacement, role switching and an intervening COMMIT.

    Call immediately before the single final COMMIT. No local observation can
    resolve an unknown outcome after that COMMIT was sent.
    """
    if (
        type(owner) is not AtomicRootProofTransactionOwner
        or owner._origin is not _OBSERVED_ORIGIN
        or connection is not owner._connection
    ):
        raise AtomicRootProofTransactionError("EXACT_OBSERVED_TRANSACTION_OWNER_REQUIRED")
    if _observe(connection, owner._snapshot.login) != owner._snapshot:
        raise AtomicRootProofTransactionError("FINAL_TRANSACTION_OWNER_CHANGED")
