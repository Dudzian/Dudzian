"""S9-RPI-11: fail-closed catalog inspection of a TEST-only CCA SQL reader.

No installation, positive admission, provisioning, SQL migration, or production
reader capability is implemented here. A trusted external pin is required;
the caller must not derive it from the database under inspection.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.pq import TransactionStatus

from bot_core.postgresql_atomic_root_proof_qualification import (
    PostgreSQLAtomicRootProofPhysicalIdentity,
    read_atomic_root_proof_physical_identity,
)

_NAME = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class IndependentCCAReaderCatalogError(RuntimeError):
    """The tested SQL catalog differs from the expected installation pin."""


@dataclass(frozen=True, slots=True)
class IndependentCCAReaderCatalogPin:
    """A supplied expectation, never self-pinned from the live reader.

    This value alone does not attest trusted provisioning or authorize use.
    """

    schema: str
    owner_role: str
    reader_role: str
    source_role: str
    owner_oid: int
    reader_oid: int
    source_oid: int
    function_oid: int
    function_definition_sha256: str
    physical_identity: PostgreSQLAtomicRootProofPhysicalIdentity

    def __post_init__(self) -> None:
        names = (self.schema, self.owner_role, self.reader_role, self.source_role)
        if any(type(name) is not str or _NAME.fullmatch(name) is None for name in names):
            raise ValueError("exact reviewed CCA identifiers required")
        if len({self.owner_role, self.reader_role, self.source_role}) != 3:
            raise ValueError("independent CCA and source principals required")
        oids = (self.owner_oid, self.reader_oid, self.source_oid, self.function_oid)
        if any(type(value) is not int or value <= 0 for value in oids):
            raise ValueError("positive pinned PostgreSQL OIDs required")
        if len(set(oids[:3])) != 3:
            raise ValueError("distinct principal OIDs required")
        if (
            type(self.function_definition_sha256) is not str
            or _DIGEST.fullmatch(self.function_definition_sha256) is None
        ):
            raise ValueError("approved SQL function source digest required")
        if type(self.physical_identity) is not PostgreSQLAtomicRootProofPhysicalIdentity:
            raise TypeError("external physical installation pin required")


@dataclass(frozen=True, slots=True)
class IndependentCCAReaderCatalogInspection:
    """Non-authorizing inspection result; cannot become CCA proof."""

    function_oid: int
    catalog_sha256: str
    status: str = field(default="NOT_QUALIFIED", init=False)
    production_authorization: str = field(default="BLOCKED", init=False)
    reader_installed: bool = field(default=False, init=False)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentCCAReaderCatalogError(message)


def _check_roles(conn: psycopg.Connection[Any], pin: IndependentCCAReaderCatalogPin) -> None:
    expected = {
        pin.owner_role: (pin.owner_oid, False),
        pin.reader_role: (pin.reader_oid, True),
        pin.source_role: (pin.source_oid, True),
    }
    for role, (oid, can_login) in expected.items():
        row = conn.execute(
            "SELECT oid,rolcanlogin,rolsuper,rolcreaterole,rolcreatedb,"
            "rolreplication,rolbypassrls FROM pg_catalog.pg_roles WHERE rolname=%s",
            (role,),
        ).fetchone()
        _require(
            row == (oid, can_login, False, False, False, False, False),
            "CCA principal OID or effective role attributes changed",
        )
    ids = [pin.owner_oid, pin.reader_oid, pin.source_oid]
    _require(
        conn.execute(
            "SELECT 1 FROM pg_catalog.pg_auth_members "
            "WHERE roleid=ANY(%s) OR member=ANY(%s) LIMIT 1",
            (ids, ids),
        ).fetchone()
        is None,
        "CCA role membership or SET ROLE path forbidden",
    )


def _check_catalog(conn: psycopg.Connection[Any], pin: IndependentCCAReaderCatalogPin) -> None:
    schema = conn.execute(
        "SELECT oid,nspowner FROM pg_catalog.pg_namespace WHERE nspname=%s",
        (pin.schema,),
    ).fetchone()
    _require(schema is not None and schema[1] == pin.owner_oid, "trusted schema owner differs")

    functions = conn.execute(
        "SELECT p.oid,p.proname,p.proowner,p.prosecdef,p.proconfig,"
        "pg_catalog.pg_get_functiondef(p.oid),p.proacl,p.pronargs,"
        "p.proargtypes::text,p.prorettype "
        "FROM pg_catalog.pg_proc p "
        "WHERE p.pronamespace=%s ORDER BY p.oid",
        (schema[0],),
    ).fetchall()
    _require(len(functions) == 1, "unexpected SQL function or overload in CCA catalog")
    fn = functions[0]
    _require(
        fn[0] == pin.function_oid
        and fn[1] == "read_exact_close"
        and fn[2] == pin.owner_oid
        and fn[3] is True
        and fn[7] == 1
        and fn[8] == "25"
        and fn[9] == 3802,
        "CCA SQL function identity or signature differs",
    )
    _require(
        fn[4] == [f"search_path=pg_catalog, {pin.schema}, pg_temp"],
        "SECURITY DEFINER function-local search_path differs",
    )
    observed_digest = hashlib.sha256(fn[5].encode("utf-8")).hexdigest()
    _require(
        observed_digest == pin.function_definition_sha256,
        "CCA SQL function body differs from reviewed external pin",
    )

    # PostgreSQL default PUBLIC EXECUTE cannot be trusted; inspect ACL facts.
    _require(fn[6] is not None, "explicit function ACL required")
    grants = conn.execute(
        "SELECT grantee,privilege_type,is_grantable "
        "FROM pg_catalog.aclexplode(%s::aclitem[])",
        (fn[6],),
    ).fetchall()
    _require(
        bool(grants)
        and all(
            grantee in {pin.owner_oid, pin.reader_oid}
            and grantee != 0
            and privilege == "EXECUTE"
            and (grantee == pin.owner_oid or not grantable)
            for grantee, privilege, grantable in grants
        )
        and any(grantee == pin.reader_oid for grantee, _, _ in grants),
        "CCA function has unexpected PUBLIC, inherited or privileged EXECUTE",
    )
    _require(
        conn.execute(
            "SELECT pg_catalog.has_schema_privilege(%s,%s,'USAGE'),"
            "pg_catalog.has_schema_privilege(%s,%s,'CREATE')",
            (pin.reader_role, schema[0], pin.reader_role, schema[0]),
        ).fetchone()
        == (True, False),
        "CCA reader schema privileges changed",
    )
    _require(
        conn.execute(
            "SELECT pg_catalog.has_database_privilege("
            "%s,pg_catalog.current_database(),'CREATE,TEMPORARY')",
            (pin.reader_role,),
        ).fetchone()
        == (False,),
        "CCA reader may create persistent or temporary objects",
    )
    _require(
        conn.execute(
            "SELECT 1 FROM pg_catalog.pg_class c "
            "WHERE c.relnamespace=%s AND c.relkind IN ('r','p','v','m','f','S') "
            "AND pg_catalog.has_table_privilege("
            "%s,c.oid,'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') "
            "LIMIT 1",
            (schema[0], pin.reader_role),
        ).fetchone()
        is None,
        "CCA reader has direct relation privileges",
    )


def inspect_independent_cca_reader_catalog(
    conn: psycopg.Connection[Any], pin: IndependentCCAReaderCatalogPin
) -> IndependentCCAReaderCatalogInspection:
    """Inspect expected catalog facts, never issue a positive production decision.

    Must run with a direct CCA login in an already-active READ COMMITTED /
    READ WRITE transaction. This is not a held lease, lock, or CCA validator.
    """
    if type(pin) is not IndependentCCAReaderCatalogPin:
        raise IndependentCCAReaderCatalogError("exact externally provisioned pin required")
    try:
        _require(
            conn.info.server_version // 10000 == 16
            and conn.info.transaction_status is TransactionStatus.INTRANS,
            "live PostgreSQL 16 transaction required",
        )
        transaction = conn.execute(
            "SELECT pg_catalog.current_setting('transaction_isolation'),"
            "pg_catalog.current_setting('transaction_read_only'),"
            "session_user,current_user"
        ).fetchone()
        _require(
            transaction == ("read committed", "off", pin.reader_role, pin.reader_role),
            "independent direct-login READ COMMITTED / READ WRITE required",
        )
        _require(
            read_atomic_root_proof_physical_identity(conn, pin.physical_identity.instance_id)
            == pin.physical_identity,
            "CCA physical database or service pin mismatch",
        )
        _check_roles(conn, pin)
        _check_catalog(conn, pin)
        return IndependentCCAReaderCatalogInspection(
            function_oid=pin.function_oid,
            catalog_sha256=pin.function_definition_sha256,
        )
    except psycopg.Error as exc:
        raise IndependentCCAReaderCatalogError("CCA catalog inspection unavailable") from exc
