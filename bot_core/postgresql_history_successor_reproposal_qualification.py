"""Explicit deployment and restart qualification for successor reproposal V1.

Migration is a bootstrap operation. Runtime pins cannot manufacture this
deployment, and restart reconstruction reviews compiled source and exact ACLs
before accepting the immutable, physically bound catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from bot_core.postgresql_atomic_root_proof_migration import (
    _application_bytes_fingerprint,
    _component_roles,
    _verify_database_manifest,
    _verify_original_sources,
    _verify_principal_boundary,
)
from bot_core.postgresql_atomic_root_proof_qualification import (
    _ADMINISTRATIVE_ORIGIN,
    AtomicRootProofQualificationError,
    PostgreSQLAtomicRootProofPhysicalIdentity,
    PostgreSQLAtomicRootProofTestDeployment,
    _catalog_fingerprint,
    _physical_identity,
    _qualify_roles,
    qualify_atomic_root_proof_authority,
    validate_atomic_root_proof_deployment,
)
from bot_core.postgresql_atomic_root_proof_schema import (
    FUNC_ADMIN as V1_ADMIN,
    FUNC_RUNTIME as V1_RUNTIME,
    PostgreSQLAtomicRootProofProvisioning,
    reviewed_function_inventory,
)
from bot_core.postgresql_custody_lifecycle_qualification import (
    _LIFECYCLE_ADMINISTRATIVE_ORIGIN,
    PostgreSQLCustodyLifecycleTestDeployment,
    _bootstrap_principal,
    _lifecycle_records_fingerprint,
    _postqualify,
    _read_secret,
    _verify_v2_functions,
    qualify_custody_lifecycle_authority,
    validate_custody_lifecycle_deployment,
)
from bot_core.postgresql_custody_lifecycle_schema import (
    FUNC_ADMIN as LIFECYCLE_ADMIN,
    reviewed_lifecycle_enrollment_function_sources,
    reviewed_lifecycle_function_sources,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_history_successor_reproposal_schema import (
    ADMIN_FUNCTIONS,
    BOUNDARY,
    CLOSED_FUNCTIONS,
    MIGRATION_ID,
    RUNTIME_FUNCTIONS,
    install_history_successor_reproposal_v1,
    reviewed_reproposal_inventory,
    source_hashes,
)
from bot_core.postgresql_root_proof_preparation_readonly import (
    PostgreSQLRootProofPreparationV2Deployment,
    _fingerprint as _preparation_fingerprint,
    _verify_preparation_sources,
    validate_root_proof_preparation_v2_deployment,
)

ATOMIC_ISSUER_READY = "NOT_READY"
PRODUCTION_LIFECYCLE_ISOLATION = "BLOCKED"
_ORIGIN = object()
_LEGACY_CLOSED = {"provision_fence(text,text,bigint)", "transition_fence(text,bigint,text,bigint)"}


class HistorySuccessorReproposalQualificationError(AtomicRootProofQualificationError):
    """Durable reviewed proposal authority or direct principal differs."""


@dataclass(frozen=True, slots=True)
class PostgreSQLHistorySuccessorReproposalV1Deployment:
    lifecycle_deployment: PostgreSQLCustodyLifecycleTestDeployment
    preparation_deployment: PostgreSQLRootProofPreparationV2Deployment | None
    predecessor_catalog_sha256: str
    _origin: object = field(repr=False, compare=False, default=None)

    @property
    def atomic_deployment(self) -> PostgreSQLAtomicRootProofTestDeployment:
        return self.lifecycle_deployment.atomic_deployment

    def __repr__(self) -> str:
        return (
            "PostgreSQLHistorySuccessorReproposalV1Deployment("
            f"schema={self.atomic_deployment.provisioning.schema!r}, "
            "boundary='ISOLATED_TEST_ONLY', ATOMIC_ISSUER_READY='NOT_READY')"
        )


def validate_history_successor_reproposal_v1_deployment(deployment: object) -> None:
    if (
        type(deployment) is not PostgreSQLHistorySuccessorReproposalV1Deployment
        or deployment._origin is not _ORIGIN
    ):
        raise HistorySuccessorReproposalQualificationError(
            "explicit reviewed reproposal deployment required"
        )
    validate_custody_lifecycle_deployment(deployment.lifecycle_deployment)
    if (
        type(deployment.predecessor_catalog_sha256) is not str
        or len(deployment.predecessor_catalog_sha256) != 64
        or any(c not in "0123456789abcdef" for c in deployment.predecessor_catalog_sha256)
    ):
        raise HistorySuccessorReproposalQualificationError(
            "immutable predecessor catalog pin required"
        )
    preparation = deployment.preparation_deployment
    if preparation is not None:
        validate_root_proof_preparation_v2_deployment(preparation)
        if (
            preparation.provisioning != deployment.atomic_deployment.provisioning
            or preparation.identity != deployment.atomic_deployment.identity
            or preparation.role_oids != deployment.atomic_deployment.role_oids
        ):
            raise HistorySuccessorReproposalQualificationError(
                "same physically bound preparation authority required"
            )


def _verify_composed_sources(
    conn: psycopg.Connection[Any],
    config: PostgreSQLAtomicRootProofProvisioning,
) -> None:
    """Every predecessor function is reviewed, with only explicit replacements."""
    inventory = reviewed_function_inventory(conn, config)
    for signature, source in (
        reviewed_lifecycle_function_sources(config)
        | reviewed_lifecycle_enrollment_function_sources(config)
    ).items():
        inventory[(config.schema, signature)] = (config.schema_owner_role, source, "", "")
    additions = reviewed_reproposal_inventory(conn, config)
    inventory.update(additions)
    expected_functions = _verify_original_sources(conn, config)
    expected_functions[config.schema] = set()
    for (schema, signature), (owner, source, _, _) in inventory.items():
        expected_functions.setdefault(schema, set()).add(signature)
        row = conn.execute(
            "SELECT p.proowner,r.rolname,p.prosrc,p.prosecdef,p.proconfig,l.lanname "
            "FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles r ON r.oid=p.proowner "
            "JOIN pg_catalog.pg_language l ON l.oid=p.prolang "
            "WHERE p.oid=pg_catalog.to_regprocedure(%s)",
            (schema + "." + signature,),
        ).fetchone()
        if row is None or row[1:] != (owner, source, True, ["search_path=pg_catalog"], "plpgsql"):
            raise HistorySuccessorReproposalQualificationError(
                "exact compiled predecessor/proposal function source differs"
            )
        callers = []
        if schema != config.schema and not signature.endswith("constraint()"):
            callers.append(config.schema_owner_role)
        elif schema == config.schema:
            if signature in (V1_RUNTIME - CLOSED_FUNCTIONS) | RUNTIME_FUNCTIONS:
                callers.append(config.issuer_writer_role)
            if (
                signature
                in (V1_ADMIN - CLOSED_FUNCTIONS - _LEGACY_CLOSED)
                | LIFECYCLE_ADMIN
                | ADMIN_FUNCTIONS
            ):
                callers.append(config.test_admin_role)
            if signature == "qualify_catalog()":
                callers.append(config.test_admin_role)
            if signature == "qualify_lifecycle_enrollment_v3()":
                callers.extend((config.test_admin_role, config.issuer_writer_role))
            if signature.startswith(("assert_bound_link(", "assert_history_link(")):
                component = (
                    config.entitlement_schema
                    if signature.startswith("assert_bound_link(")
                    else config.history_schema
                )
                component_owner = conn.execute(
                    sql.SQL("SELECT schema_owner_role FROM {}.metadata WHERE singleton").format(
                        sql.Identifier(component)
                    )
                ).fetchone()
                if component_owner is None:
                    raise HistorySuccessorReproposalQualificationError("component owner missing")
                callers.append(component_owner[0])
        expected = {(row[0], "EXECUTE", False)}
        for caller in callers:
            role = conn.execute(
                "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s", (caller,)
            ).fetchone()
            if role is None:
                raise HistorySuccessorReproposalQualificationError("required direct role missing")
            expected.add((role[0], "EXECUTE", False))
        actual = set(
            conn.execute(
                "SELECT a.grantee,a.privilege_type,a.is_grantable FROM pg_catalog.pg_proc p "
                "CROSS JOIN LATERAL pg_catalog.aclexplode(p.proacl) a WHERE p.oid=pg_catalog.to_regprocedure(%s)",
                (schema + "." + signature,),
            ).fetchall()
        )
        if actual != expected:
            raise HistorySuccessorReproposalQualificationError(
                "exact composed function ACL differs"
            )
    for schema, signatures in expected_functions.items():
        rows = conn.execute(
            "SELECT p.proname || '(' || pg_catalog.replace(pg_catalog.oidvectortypes(p.proargtypes),', ', ',') || ')' FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname=%s",
            (schema,),
        ).fetchall()
        if {row[0] for row in rows} != signatures:
            raise HistorySuccessorReproposalQualificationError(
                "exact composed function inventory differs"
            )
    archive = conn.execute(
        sql.SQL(
            "SELECT migration_id,schema_version,boundary,predecessor_catalog_sha256,predecessor_snapshot_sha256=pg_catalog.sha256(pg_catalog.convert_to(predecessor_snapshot::text,'UTF8')),compiled_source_sha256 FROM {}.history_reproposal_contract WHERE singleton"
        ).format(sql.Identifier(config.schema))
    ).fetchall()
    if (
        len(archive) != 1
        or archive[0][:3] != (MIGRATION_ID, 1, BOUNDARY)
        or archive[0][4:] != (True, source_hashes(additions))
    ):
        raise HistorySuccessorReproposalQualificationError(
            "immutable predecessor/source archive differs"
        )
    # Verify that replacement archives contain the original V1 source, rather
    # than accepting a self-consistent hash of an arbitrary archived function.
    predecessor = conn.execute(
        sql.SQL(
            "SELECT predecessor_snapshot->'functions' FROM {}.history_reproposal_contract WHERE singleton"
        ).format(sql.Identifier(config.schema))
    ).fetchone()
    old = reviewed_function_inventory(conn, config)
    if predecessor is None or type(predecessor[0]) is not list:
        raise HistorySuccessorReproposalQualificationError("predecessor source inventory missing")
    # Original standalone component APIs are deliberately retained with their
    # already reviewed, narrowed ACLs. They are not additional coordinator
    # helpers, so verify their complete archived pg_proc facts separately.
    archived_by_oid = {row["oid"]: row for row in predecessor[0]}
    for schema, signature in inventory:
        if (schema, signature) in additions:
            continue
        current = conn.execute(
            "SELECT pg_catalog.to_jsonb(p) FROM pg_catalog.pg_proc p WHERE p.oid=pg_catalog.to_regprocedure(%s)",
            (schema + "." + signature,),
        ).fetchone()
        if current is None or current[0].get("oid") not in archived_by_oid:
            raise HistorySuccessorReproposalQualificationError(
                "original function OID archive differs"
            )
        observed = dict(current[0])
        original = dict(archived_by_oid[observed["oid"]])
        if schema == config.schema and signature in CLOSED_FUNCTIONS:
            # The only permitted old function catalog change closes EXECUTE.
            observed.pop("proacl")
            original.pop("proacl")
        if observed != original:
            raise HistorySuccessorReproposalQualificationError(
                "original function catalog archive differs"
            )
    for schema, signatures in expected_functions.items():
        for signature in signatures:
            if (schema, signature) in inventory:
                continue
            current = conn.execute(
                "SELECT pg_catalog.to_jsonb(p) FROM pg_catalog.pg_proc p WHERE p.oid=pg_catalog.to_regprocedure(%s)",
                (schema + "." + signature,),
            ).fetchone()
            if (
                current is None
                or current[0].get("oid") not in archived_by_oid
                or current[0] != archived_by_oid[current[0]["oid"]]
            ):
                raise HistorySuccessorReproposalQualificationError(
                    "retained standalone source or ACL archive differs"
                )
    for signature in ("validate_decision()", "approve_commit(jsonb)", "commit_root_proof(bytea)"):
        name = signature.split("(", 1)[0]
        archived = [row for row in predecessor[0] if row.get("proname") == name]
        if len(archived) != 1 or archived[0].get("prosrc") != old[(config.schema, signature)][1]:
            raise HistorySuccessorReproposalQualificationError("original V1 source archive differs")


def _verify_proposal_relations(
    conn: psycopg.Connection[Any],
    config: PostgreSQLAtomicRootProofProvisioning,
) -> None:
    """Reconstruction does not bless a freshly pinned, altered proposal table."""
    owner = conn.execute(
        "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s", (config.schema_owner_role,)
    ).fetchone()
    if owner is None:
        raise HistorySuccessorReproposalQualificationError("proposal owner missing")
    decision_trigger = conn.execute(
        "SELECT t.tgtype,t.tgenabled,t.tgdeferrable,t.tginitdeferred,t.tgisinternal,"
        "t.tgfoid=pg_catalog.to_regprocedure(%s),t.tgattr::text,t.tgqual::text,t.tgnargs,t.tgargs,"
        "t.tgoldtable,t.tgnewtable,t.tgconstraint<>0 FROM pg_catalog.pg_trigger t "
        "WHERE t.tgrelid=pg_catalog.to_regclass(%s) AND t.tgname='atomic_decision_link'",
        (config.schema + ".validate_history_proposal_decision()", config.schema + ".decisions"),
    ).fetchall()
    if decision_trigger != [(5, "A", True, True, False, True, "", None, 0, b"", None, None, True)]:
        raise HistorySuccessorReproposalQualificationError(
            "exact deferred proposal decision trigger required"
        )
    archived = conn.execute(
        sql.SQL(
            "SELECT predecessor_snapshot FROM {}.history_reproposal_contract WHERE singleton"
        ).format(sql.Identifier(config.schema))
    ).fetchone()
    if archived is None or type(archived[0]) is not dict:
        raise HistorySuccessorReproposalQualificationError(
            "original catalog relation archive required"
        )
    expected_added_relations = {
        "history_reproposal_contract",
        "history_reproposal_contract_pkey",
        "history_proposals",
        "history_proposals_pkey",
        "history_proposals_attempt_key_revision_candidate_id_key",
        "history_proposals_attempt_key_candidate_id_key",
        "history_proposal_current",
        "history_proposal_current_pkey",
    }
    namespace_oid = conn.execute(
        "SELECT oid FROM pg_catalog.pg_namespace WHERE nspname=%s", (config.schema,)
    ).fetchone()
    if namespace_oid is None:
        raise HistorySuccessorReproposalQualificationError("proposal namespace missing")
    predecessor_relations = {
        row[1] for row in archived[0]["relations"] if row[2] == str(namespace_oid[0])
    }
    current_relations = {
        row[0]
        for row in conn.execute(
            "SELECT c.relname FROM pg_catalog.pg_class c WHERE c.relnamespace=%s",
            (namespace_oid[0],),
        ).fetchall()
    }
    if current_relations != predecessor_relations | expected_added_relations:
        raise HistorySuccessorReproposalQualificationError(
            "exact composed proposal relation inventory required"
        )
    predecessor_types = {
        row["typname"]
        for row in archived[0]["types"]
        if row["typnamespace"] == str(namespace_oid[0])
    }
    expected_added_types = {
        name
        for table in (
            "history_reproposal_contract",
            "history_proposals",
            "history_proposal_current",
        )
        for name in (table, "_" + table)
    }
    current_types = {
        row[0]
        for row in conn.execute(
            "SELECT typname FROM pg_catalog.pg_type WHERE typnamespace=%s", (namespace_oid[0],)
        ).fetchall()
    }
    if current_types != predecessor_types | expected_added_types:
        raise HistorySuccessorReproposalQualificationError(
            "exact composed proposal type inventory required"
        )
    expected_columns = {
        "history_reproposal_contract": [
            ("singleton", "boolean"),
            ("migration_id", "text"),
            ("schema_version", "integer"),
            ("boundary", "text"),
            ("predecessor_catalog_sha256", "text"),
            ("predecessor_snapshot", "jsonb"),
            ("predecessor_snapshot_sha256", "bytea"),
            ("compiled_source_sha256", "jsonb"),
        ],
        "history_proposals": [
            ("attempt_key", "bytea"),
            ("revision", "bigint"),
            ("candidate_id", "text"),
            ("predecessor_candidate_id", "text"),
            ("package", "jsonb"),
        ],
        "history_proposal_current": [
            ("attempt_key", "bytea"),
            ("revision", "bigint"),
            ("candidate_id", "text"),
        ],
    }
    expected_constraints = {
        "history_reproposal_contract": [
            ("c", False, False, "CHECK (schema_version = 1)"),
            ("c", False, False, "CHECK (singleton)"),
            ("p", False, False, "PRIMARY KEY (singleton)"),
        ],
        "history_proposals": [
            ("c", False, False, "CHECK (candidate_id ~ '^hp1_[0-9a-f]{64}$'::text)"),
            ("c", False, False, "CHECK (revision >= 1)"),
            (
                "c",
                False,
                False,
                "CHECK (revision = 1 AND predecessor_candidate_id IS NULL OR revision > 1 AND predecessor_candidate_id IS NOT NULL)",
            ),
            (
                "f",
                False,
                False,
                f"FOREIGN KEY (attempt_key) REFERENCES {config.schema}.approved_commits(attempt_key)",
            ),
            (
                "f",
                False,
                False,
                f"FOREIGN KEY (attempt_key, predecessor_candidate_id) REFERENCES {config.schema}.history_proposals(attempt_key, candidate_id)",
            ),
            ("p", False, False, "PRIMARY KEY (attempt_key, revision)"),
            ("u", False, False, "UNIQUE (attempt_key, candidate_id)"),
            ("u", False, False, "UNIQUE (attempt_key, revision, candidate_id)"),
        ],
        "history_proposal_current": [
            (
                "f",
                True,
                True,
                f"FOREIGN KEY (attempt_key, revision, candidate_id) REFERENCES {config.schema}.history_proposals(attempt_key, revision, candidate_id) DEFERRABLE INITIALLY DEFERRED",
            ),
            ("p", False, False, "PRIMARY KEY (attempt_key)"),
        ],
    }
    for table, columns in expected_columns.items():
        qualified = config.schema + "." + table
        relation = conn.execute(
            "SELECT relkind,relpersistence,relowner,relrowsecurity,relforcerowsecurity,reloptions FROM pg_catalog.pg_class WHERE oid=pg_catalog.to_regclass(%s)",
            (qualified,),
        ).fetchone()
        if relation != ("r", "p", owner[0], False, False, None):
            raise HistorySuccessorReproposalQualificationError(
                "exact protected proposal relation required"
            )
        actual = conn.execute(
            "SELECT a.attname,pg_catalog.format_type(a.atttypid,a.atttypmod),a.attnotnull,a.attidentity,a.attgenerated,a.attisdropped,a.attacl::text,pg_catalog.pg_get_expr(d.adbin,d.adrelid) "
            "FROM pg_catalog.pg_attribute a LEFT JOIN pg_catalog.pg_attrdef d ON (d.adrelid,d.adnum)=(a.attrelid,a.attnum) "
            "WHERE a.attrelid=pg_catalog.to_regclass(%s) AND a.attnum>0 ORDER BY a.attnum",
            (qualified,),
        ).fetchall()
        expected = [
            (
                name,
                kind,
                name != "predecessor_candidate_id",
                "",
                "",
                False,
                None,
                "true" if name == "singleton" else None,
            )
            for name, kind in columns
        ]
        if actual != expected:
            raise HistorySuccessorReproposalQualificationError(
                "exact proposal columns/defaults/ACL required"
            )
        constraints = conn.execute(
            "SELECT contype,condeferrable,condeferred,pg_catalog.pg_get_constraintdef(oid,true) FROM pg_catalog.pg_constraint WHERE conrelid=pg_catalog.to_regclass(%s)",
            (qualified,),
        ).fetchall()
        if sorted(constraints) != sorted(expected_constraints[table]):
            raise HistorySuccessorReproposalQualificationError(
                "exact proposal constraints required"
            )
        if (
            conn.execute(
                "SELECT 1 FROM pg_catalog.pg_constraint WHERE conrelid=pg_catalog.to_regclass(%s) AND NOT convalidated LIMIT 1",
                (qualified,),
            ).fetchone()
            is not None
        ):
            raise HistorySuccessorReproposalQualificationError(
                "validated proposal constraints required"
            )
        acl = set(
            conn.execute(
                "SELECT a.grantor,a.grantee,a.privilege_type,a.is_grantable FROM pg_catalog.pg_class c CROSS JOIN LATERAL pg_catalog.aclexplode(c.relacl) a WHERE c.oid=pg_catalog.to_regclass(%s)",
                (qualified,),
            ).fetchall()
        )
        if acl != {
            (owner[0], owner[0], privilege, False)
            for privilege in (
                "SELECT",
                "INSERT",
                "UPDATE",
                "DELETE",
                "TRUNCATE",
                "REFERENCES",
                "TRIGGER",
            )
        }:
            raise HistorySuccessorReproposalQualificationError(
                "owner-only proposal table ACL required"
            )
        triggers = conn.execute(
            "SELECT t.tgname,t.tgtype,t.tgenabled,p.proname,t.tgattr::text,t.tgqual::text,t.tgnargs,t.tgargs,t.tgdeferrable,t.tginitdeferred FROM pg_catalog.pg_trigger t JOIN pg_catalog.pg_proc p ON p.oid=t.tgfoid WHERE t.tgrelid=pg_catalog.to_regclass(%s) AND NOT t.tgisinternal ORDER BY t.tgname",
            (qualified,),
        ).fetchall()
        expected_triggers = [
            (
                "reproposal_immutable_row",
                27,
                "A",
                "guard_history_proposal_selector"
                if table == "history_proposal_current"
                else "reject_mutation",
                "",
                None,
                0,
                b"",
                False,
                False,
            ),
            (
                "reproposal_immutable_truncate",
                34,
                "A",
                "reject_mutation",
                "",
                None,
                0,
                b"",
                False,
                False,
            ),
        ]
        if triggers != expected_triggers:
            raise HistorySuccessorReproposalQualificationError(
                "exact ALWAYS proposal guards required"
            )
        for catalog, column in (("pg_rewrite", "ev_class"), ("pg_policy", "polrelid")):
            if (
                conn.execute(
                    sql.SQL(
                        "SELECT 1 FROM pg_catalog.{} WHERE {}=pg_catalog.to_regclass(%s) LIMIT 1"
                    ).format(sql.Identifier(catalog), sql.Identifier(column)),
                    (qualified,),
                ).fetchone()
                is not None
            ):
                raise HistorySuccessorReproposalQualificationError("proposal rule/policy forbidden")
        # Every index is the exact plain unique constraint index. No predicate,
        # expression or invalid index may alter protected identity selection.
        indexes = conn.execute(
            "SELECT i.indisunique,i.indisvalid,i.indisready,i.indislive,i.indisexclusion,i.indnullsnotdistinct,i.indexprs::text,i.indpred::text,pg_catalog.pg_get_indexdef(i.indexrelid) FROM pg_catalog.pg_index i WHERE i.indrelid=pg_catalog.to_regclass(%s)",
            (qualified,),
        ).fetchall()
        expected_index_count = sum(
            kind in {"p", "u"} for kind, _, _, _ in expected_constraints[table]
        )
        if len(indexes) != expected_index_count or any(
            row[:8] != (True, True, True, True, False, False, None, None) for row in indexes
        ):
            raise HistorySuccessorReproposalQualificationError(
                "exact proposal identity indexes required"
            )


def qualify_history_successor_reproposal_v1(
    conn: psycopg.Connection[Any],
    deployment: PostgreSQLHistorySuccessorReproposalV1Deployment,
    *,
    expected_role: str = "issuer_writer",
) -> None:
    validate_history_successor_reproposal_v1_deployment(deployment)
    if expected_role not in {"issuer_writer", "admin"}:
        raise HistorySuccessorReproposalQualificationError("explicit proposal principal required")
    try:
        qualify_custody_lifecycle_authority(
            conn, deployment.lifecycle_deployment, expected_role=expected_role
        )
        if conn.execute(
            sql.SQL("SELECT {}.qualify_history_reproposal_v1()").format(
                sql.Identifier(deployment.atomic_deployment.provisioning.schema)
            )
        ).fetchone() != (True,):
            raise HistorySuccessorReproposalQualificationError(
                "proposal contract qualification refused"
            )
        if deployment.preparation_deployment is not None:
            if (
                _preparation_fingerprint(conn, deployment.preparation_deployment)
                != deployment.preparation_deployment.catalog_sha256
            ):
                raise HistorySuccessorReproposalQualificationError(
                    "preparation composed catalog differs"
                )
    except HistorySuccessorReproposalQualificationError:
        raise
    except (psycopg.Error, AtomicRootProofQualificationError, ValueError, TypeError) as exc:
        raise HistorySuccessorReproposalQualificationError(
            "live successor reproposal qualification unavailable"
        ) from exc


def _postqualify_reproposal(deployment: PostgreSQLHistorySuccessorReproposalV1Deployment) -> None:
    for connection, role in (
        (deployment.atomic_deployment.writer_connection, "issuer_writer"),
        (deployment.lifecycle_deployment.admin_connection, "admin"),
    ):
        with psycopg.connect(connection.dsn, autocommit=True) as conn, conn.transaction():
            qualify_history_successor_reproposal_v1(conn, deployment, expected_role=role)


def migrate_history_successor_reproposal_v1(
    bootstrap: PostgreSQLConnectionConfig,
    lifecycle_deployment: PostgreSQLCustodyLifecycleTestDeployment,
    *,
    preparation_deployment: PostgreSQLRootProofPreparationV2Deployment | None = None,
) -> PostgreSQLHistorySuccessorReproposalV1Deployment:
    """Install only on a freshly qualified lifecycle SQL3 predecessor."""
    validate_custody_lifecycle_deployment(lifecycle_deployment)
    if type(bootstrap) is not PostgreSQLConnectionConfig:
        raise TypeError("separate exact bootstrap connection required")
    atomic = lifecycle_deployment.atomic_deployment
    config = atomic.provisioning
    _postqualify(lifecycle_deployment)
    if preparation_deployment is not None:
        validate_root_proof_preparation_v2_deployment(preparation_deployment)
        if (
            preparation_deployment.provisioning != config
            or preparation_deployment.identity != atomic.identity
        ):
            raise HistorySuccessorReproposalQualificationError(
                "same preparation predecessor required"
            )
    try:
        with psycopg.connect(bootstrap.dsn, autocommit=True) as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            conn.execute("SET LOCAL search_path=pg_catalog")
            _bootstrap_principal(conn, config)
            if _physical_identity(conn, config.instance_id) != atomic.identity:
                raise HistorySuccessorReproposalQualificationError(
                    "migration physical authority differs"
                )
            conn.execute(
                sql.SQL("LOCK TABLE {}.decisions IN ACCESS EXCLUSIVE MODE").format(
                    sql.Identifier(config.schema)
                )
            )
            conn.execute(
                sql.SQL("LOCK TABLE {}.fences IN ACCESS EXCLUSIVE MODE").format(
                    sql.Identifier(config.schema)
                )
            )
            _verify_v2_functions(conn, config)
            _verify_database_manifest(conn, config)
            if _catalog_fingerprint(conn, config, atomic.role_oids) != atomic.catalog_sha256:
                raise HistorySuccessorReproposalQualificationError(
                    "exact predecessor catalog differs"
                )
            if preparation_deployment is not None:
                _verify_preparation_sources(conn, preparation_deployment)
                if (
                    _preparation_fingerprint(conn, preparation_deployment)
                    != preparation_deployment.catalog_sha256
                ):
                    raise HistorySuccessorReproposalQualificationError(
                        "preparation predecessor catalog differs"
                    )
            application_before = _application_bytes_fingerprint(conn, config)
            lifecycle_before = _lifecycle_records_fingerprint(conn, config)
            install_history_successor_reproposal_v1(conn, config, atomic.catalog_sha256)
            _verify_composed_sources(conn, config)
            _verify_proposal_relations(conn, config)
            _verify_database_manifest(conn, config)
            _verify_principal_boundary(conn, config)
            if (
                _application_bytes_fingerprint(conn, config) != application_before
                or _lifecycle_records_fingerprint(conn, config) != lifecycle_before
            ):
                raise HistorySuccessorReproposalQualificationError(
                    "migration changed protected original bytes"
                )
            revised_atomic = replace(
                atomic, catalog_sha256=_catalog_fingerprint(conn, config, atomic.role_oids)
            )
            _qualify_roles(conn, revised_atomic)
            revised_lifecycle = replace(
                lifecycle_deployment,
                atomic_deployment=revised_atomic,
                catalog_sha256=revised_atomic.catalog_sha256,
            )
            revised_preparation = preparation_deployment
            if revised_preparation is not None:
                revised_preparation = replace(
                    revised_preparation,
                    catalog_sha256=_preparation_fingerprint(conn, revised_preparation),
                )
            deployment = PostgreSQLHistorySuccessorReproposalV1Deployment(
                revised_lifecycle, revised_preparation, atomic.catalog_sha256, _ORIGIN
            )
        _postqualify_reproposal(deployment)
        return deployment
    except HistorySuccessorReproposalQualificationError:
        raise
    except (
        psycopg.Error,
        AtomicRootProofQualificationError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
    ) as exc:
        raise HistorySuccessorReproposalQualificationError(
            "explicit successor reproposal migration refused"
        ) from exc


def requalify_existing_history_successor_reproposal_v1(
    bootstrap: PostgreSQLConnectionConfig,
    atomic_config: PostgreSQLAtomicRootProofProvisioning,
    writer_connection: PostgreSQLConnectionConfig,
    admin_connection: PostgreSQLConnectionConfig,
    *,
    expected_identity: PostgreSQLAtomicRootProofPhysicalIdentity,
    preparation_deployment: PostgreSQLRootProofPreparationV2Deployment | None = None,
) -> PostgreSQLHistorySuccessorReproposalV1Deployment:
    """Read-only reconstruction; never repairs or refreshes a catalog object."""
    if (
        type(atomic_config) is not PostgreSQLAtomicRootProofProvisioning
        or type(expected_identity) is not PostgreSQLAtomicRootProofPhysicalIdentity
        or any(
            type(c) is not PostgreSQLConnectionConfig
            for c in (bootstrap, writer_connection, admin_connection)
        )
    ):
        raise TypeError("exact bootstrap, physical identity and direct credentials required")
    config = atomic_config
    if (
        conninfo_to_dict(writer_connection.dsn).get("user") != config.issuer_writer_role
        or conninfo_to_dict(admin_connection.dsn).get("user") != config.test_admin_role
    ):
        raise HistorySuccessorReproposalQualificationError(
            "exact direct reconstruction credentials required"
        )
    try:
        with psycopg.connect(bootstrap.dsn, autocommit=True) as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE READ ONLY")
            conn.execute("SET LOCAL search_path=pg_catalog")
            _bootstrap_principal(conn, config)
            if _physical_identity(conn, config.instance_id) != expected_identity:
                raise HistorySuccessorReproposalQualificationError(
                    "reconstruction physical authority differs"
                )
            _verify_composed_sources(conn, config)
            _verify_proposal_relations(conn, config)
            _verify_database_manifest(conn, config)
            _verify_principal_boundary(conn, config)
            roles, owners = _component_roles(conn, config)
            atomic = PostgreSQLAtomicRootProofTestDeployment(
                writer_connection,
                config,
                expected_identity,
                _catalog_fingerprint(conn, config, roles),
                roles,
                owners,
                _ADMINISTRATIVE_ORIGIN,
            )
            _qualify_roles(conn, atomic)
            lifecycle = PostgreSQLCustodyLifecycleTestDeployment(
                atomic,
                admin_connection,
                _read_secret(conn, config),
                atomic.catalog_sha256,
                _LIFECYCLE_ADMINISTRATIVE_ORIGIN,
                3,
            )
            predecessor = conn.execute(
                sql.SQL(
                    "SELECT predecessor_catalog_sha256 FROM {}.history_reproposal_contract WHERE singleton"
                ).format(sql.Identifier(config.schema))
            ).fetchone()
            if predecessor is None:
                raise HistorySuccessorReproposalQualificationError(
                    "immutable predecessor pin missing"
                )
            preparation = preparation_deployment
            if preparation is not None:
                validate_root_proof_preparation_v2_deployment(preparation)
                if (
                    preparation.provisioning != config
                    or preparation.identity != expected_identity
                    or preparation.role_oids != roles
                ):
                    raise HistorySuccessorReproposalQualificationError(
                        "preparation authority differs"
                    )
                _verify_preparation_sources(conn, preparation)
                if _preparation_fingerprint(conn, preparation) != preparation.catalog_sha256:
                    raise HistorySuccessorReproposalQualificationError(
                        "preparation restart catalog differs"
                    )
            deployment = PostgreSQLHistorySuccessorReproposalV1Deployment(
                lifecycle, preparation, predecessor[0], _ORIGIN
            )
        _postqualify_reproposal(deployment)
        return deployment
    except HistorySuccessorReproposalQualificationError:
        raise
    except (
        psycopg.Error,
        AtomicRootProofQualificationError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
    ) as exc:
        raise HistorySuccessorReproposalQualificationError(
            "existing successor reproposal reconstruction refused"
        ) from exc


__all__ = [
    "ATOMIC_ISSUER_READY",
    "PRODUCTION_LIFECYCLE_ISOLATION",
    "MIGRATION_ID",
    "HistorySuccessorReproposalQualificationError",
    "PostgreSQLHistorySuccessorReproposalV1Deployment",
    "validate_history_successor_reproposal_v1_deployment",
    "qualify_history_successor_reproposal_v1",
    "migrate_history_successor_reproposal_v1",
    "requalify_existing_history_successor_reproposal_v1",
]
