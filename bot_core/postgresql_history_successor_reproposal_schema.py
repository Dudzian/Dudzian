"""Explicit immutable history-successor proposal SQL overlay.

The original approval and commit builders remain unchanged. This version uses
their reviewed validations and write set, with an append-only proposal journal
and a shared row-lock/SSI boundary around selection and commit.
"""

from __future__ import annotations

import hashlib
from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from bot_core.authenticated_issuer_history import ATTESTATION_DOMAIN, RECORD_DOMAIN
from bot_core.postgresql_atomic_root_proof_schema import (
    PostgreSQLAtomicRootProofProvisioning,
    _function_sources,
    _guard,
    _q,
    reviewed_function_inventory,
)
from bot_core.postgresql_custody_lifecycle_schema import (
    reviewed_lifecycle_enrollment_function_sources,
)

MIGRATION_ID = "DURABLE_HISTORY_SUCCESSOR_REPROPOSAL_V1"
SCHEMA_VERSION = 1
BOUNDARY = "ISOLATED_TEST_ONLY"
PROPOSAL_DOMAIN = "CryptoHunter/HistorySuccessorProposal/v1"
_EVENT_DOMAIN = b"CryptoHunter/M0.5/IssuerHistoryEvent/v1\0"
RUNTIME_FUNCTIONS = frozenset(
    {
        "commit_history_proposal(bytea,text)",
        "read_history_proposal(bytea,text)",
        "qualify_history_reproposal_v1()",
    }
)
ADMIN_FUNCTIONS = frozenset(
    {
        "approve_history_proposal(jsonb,bigint,text)",
        "read_history_proposal(bytea,text)",
        "qualify_history_reproposal_v1()",
    }
)
CLOSED_FUNCTIONS = frozenset({"approve_commit(jsonb)", "commit_root_proof(bytea)"})
FUNCTION_DECLARATIONS = {
    "history_proposal_id(jsonb,bigint,text)": (
        "p_package jsonb,p_revision bigint,p_predecessor text",
        "text",
    ),
    "approve_history_proposal(jsonb,bigint,text)": (
        "p_package jsonb,p_revision bigint,p_predecessor text",
        "jsonb",
    ),
    "read_history_proposal(bytea,text)": ("p_attempt bytea,p_candidate text", "jsonb"),
    "commit_history_proposal(bytea,text)": ("p_attempt bytea,p_candidate text", "jsonb"),
    "qualify_history_reproposal_v1()": ("", "boolean"),
    "validate_history_proposal_decision()": ("", "trigger"),
    "validate_history_proposal_base(bytea,jsonb)": ("p_attempt bytea,p jsonb", "void"),
    "validate_history_proposal_package(jsonb)": ("p_package jsonb", "void"),
    "validate_history_proposal_enrollment(jsonb)": ("p jsonb", "void"),
    "guard_history_proposal_selector()": ("", "trigger"),
}


def _admin_alias(source: str, config: PostgreSQLAtomicRootProofProvisioning) -> str:
    """Narrow additional helpers to the same owner and explicit test verifier."""
    source = source.replace(
        f"session_user <> '{config.issuer_writer_role}'",
        f"session_user NOT IN ('{config.issuer_writer_role}','{config.test_admin_role}')",
    ).replace(
        f"session_user::text IS DISTINCT FROM '{config.issuer_writer_role}'",
        f"session_user::text NOT IN ('{config.issuer_writer_role}','{config.test_admin_role}')",
    )
    for name in (
        "atomic_read_intent",
        "atomic_read_retention",
        "atomic_intent_member",
        "atomic_retention_member",
        "atomic_history_member",
        "atomic_lock_credential",
    ):
        source = source.replace(name + "(", "reproposal_" + name + "(")
    return source


def reviewed_reproposal_function_sources(
    config: PostgreSQLAtomicRootProofProvisioning,
) -> dict[str, str]:
    """Reviewed composition; no source read from the database is executed."""
    s, e, h = map(_q, (config.schema, config.entitlement_schema, config.history_schema))
    original = _function_sources(config)
    writer = _guard(config, config.schema_owner_role)
    admin = _guard(config, config.schema_owner_role, admin=True)
    both = writer.replace(
        f"session_user <> '{config.issuer_writer_role}'",
        f"session_user NOT IN ('{config.issuer_writer_role}','{config.test_admin_role}')",
    )
    writable = """
 IF pg_catalog.current_setting('transaction_isolation')<>'serializable'
 OR pg_catalog.current_setting('transaction_read_only')<>'off' THEN
 RAISE EXCEPTION 'writable SERIALIZABLE proposal transaction required' USING ERRCODE='25001'; END IF;
"""
    # This no-op UPDATE is intentional. A serializable transaction that waited
    # for a previous commit cannot continue with an earlier negative snapshot.
    select_current = f"""
 SELECT * INTO selected FROM {s}.history_proposal_current WHERE attempt_key=p_attempt FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'history proposal missing' USING ERRCODE='42501'; END IF;
 UPDATE {s}.history_proposal_current SET revision=revision WHERE attempt_key=p_attempt;
"""
    commit = original["commit_root_proof(bytea)"]
    commit = commit.replace(
        "DECLARE p jsonb;", f"DECLARE selected {s}.history_proposal_current%ROWTYPE; p jsonb;", 1
    )
    old_select = f" SELECT package INTO p FROM {s}.approved_commits WHERE attempt_key=p_attempt;\n IF NOT FOUND THEN RAISE EXCEPTION 'qualified test verifier approval required' USING ERRCODE='42501'; END IF;"
    new_select = (
        select_current
        + f"""
 SELECT package INTO STRICT p FROM {s}.history_proposals
 WHERE attempt_key=p_attempt AND candidate_id=p_candidate;
 IF selected.candidate_id IS DISTINCT FROM p_candidate THEN
 RAISE EXCEPTION 'superseded history candidate' USING ERRCODE='23514'; END IF;
"""
    )
    if old_select not in commit:
        raise ValueError("reviewed V1 commit selection changed")
    commit = commit.replace(old_select, new_select, 1)
    # Positive exact recovery is unchanged, and precedes all lifecycle checks.
    validate = original["validate_decision()"].replace(
        f"SELECT package INTO p FROM {s}.approved_commits WHERE attempt_key=NEW.attempt_key;",
        f"SELECT z.package INTO p FROM {s}.history_proposals z JOIN {s}.history_proposal_current c "
        "USING(attempt_key,revision,candidate_id) WHERE z.attempt_key=NEW.attempt_key;",
        1,
    )
    declarations = original["commit_root_proof(bytea)"].split("BEGIN\n", 1)[0]
    declarations = declarations.replace("DECLARE p jsonb; existing", "DECLARE existing", 1)
    prebind = (
        original["commit_root_proof(bytea)"]
        .split(" scope:=", 1)[1]
        .split(f" PERFORM {e}.atomic_bind(", 1)[0]
    )
    base = _admin_alias(
        declarations + "BEGIN\n " + admin + "\n scope:=" + prebind + "\nEND", config
    )
    enrollment = reviewed_lifecycle_enrollment_function_sources(config)[
        "guard_lifecycle_decision_enrollment_v3()"
    ]
    enrollment = (
        enrollment.replace("NEW.package", "p")
        .replace("NEW.reserved_envelope", "pg_catalog.decode(p->>'reserved_envelope_hex','hex')")
        .replace("NEW.canonical_head", "pg_catalog.decode(p->>'history_head_hex','hex')")
    )
    enrollment = _admin_alias(enrollment.replace("RETURN NEW;", "RETURN;"), config)
    immutable_fields = "ARRAY['expected_history_digest','history_record','history_record_hex','history_head','history_head_hex','history_signature_hex']::text[]"
    package_validation = original["approve_commit(jsonb)"].split(" attempt :=", 1)[0] + "\nEND"
    return {
        "history_proposal_id(jsonb,bigint,text)": f"""
BEGIN
 RETURN 'hp1_' || pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to('{PROPOSAL_DOMAIN}','UTF8')
 || pg_catalog.convert_to(p_package::text,'UTF8') || pg_catalog.int8send(p_revision)
 || pg_catalog.convert_to(COALESCE(p_predecessor,''),'UTF8')),'hex');
END
""",
        "qualify_history_reproposal_v1()": f"""
BEGIN
 {both}
 PERFORM {s}.assert_lifecycle_catalog();
 IF NOT EXISTS(SELECT 1 FROM {s}.history_reproposal_contract WHERE singleton
 AND migration_id='{MIGRATION_ID}' AND schema_version=1 AND boundary='{BOUNDARY}'
 AND predecessor_snapshot_sha256=pg_catalog.sha256(pg_catalog.convert_to(predecessor_snapshot::text,'UTF8')))
 THEN RAISE EXCEPTION 'immutable history reproposal contract missing' USING ERRCODE='42501'; END IF;
 RETURN true;
END
""",
        "read_history_proposal(bytea,text)": f"""
DECLARE result jsonb;
BEGIN
 {both}
 PERFORM {s}.qualify_history_reproposal_v1();
 SELECT pg_catalog.jsonb_build_object('candidate_id',z.candidate_id,'revision',z.revision,
 'predecessor_candidate_id',z.predecessor_candidate_id,'package',z.package) INTO result
 FROM {s}.history_proposals z JOIN {s}.history_proposal_current c USING(attempt_key)
 WHERE z.attempt_key=p_attempt AND z.candidate_id=COALESCE(p_candidate,c.candidate_id);
 RETURN result;
END
""",
        "approve_history_proposal(jsonb,bigint,text)": f"""
DECLARE p_attempt bytea; candidate text; selected {s}.history_proposal_current%ROWTYPE;
 prior {s}.history_proposals%ROWTYPE; base jsonb; retained jsonb; result jsonb; st jsonb;
BEGIN
 {admin}
 {writable}
 PERFORM {s}.qualify_history_reproposal_v1();
 IF p_revision IS NULL OR p_revision<1 OR p_revision>9007199254740991
 OR (p_revision=1 AND p_predecessor IS NOT NULL)
 OR (p_revision>1 AND (p_predecessor IS NULL OR p_predecessor !~ '^hp1_[0-9a-f]{{64}}$'))
 OR p_package IS NULL OR pg_catalog.jsonb_typeof(p_package)<>'object'
 OR pg_catalog.jsonb_typeof(p_package->'attempt_hex') IS DISTINCT FROM 'string'
 OR p_package->>'attempt_hex' !~ '^([0-9a-f][0-9a-f])+$'
 THEN RAISE EXCEPTION 'exact versioned history proposal required' USING ERRCODE='22023'; END IF;
 p_attempt:=pg_catalog.decode(p_package->>'attempt_hex','hex');
 PERFORM {s}.validate_history_proposal_package(p_package);
 candidate:={s}.history_proposal_id(p_package,p_revision,p_predecessor);
 -- An absent attempt is serialized by a real inserted selector, not a
 -- read-only NOT_FOUND observation. The FK is deferred until journal append.
 INSERT INTO {s}.history_proposal_current VALUES(p_attempt,p_revision,candidate) ON CONFLICT DO NOTHING;
 {select_current}
 SELECT * INTO prior FROM {s}.history_proposals WHERE attempt_key=p_attempt AND revision=p_revision;
 IF FOUND THEN
 IF prior.candidate_id IS DISTINCT FROM candidate OR prior.package IS DISTINCT FROM p_package
 OR prior.predecessor_candidate_id IS DISTINCT FROM p_predecessor THEN
 RAISE EXCEPTION 'proposal revision identity conflict' USING ERRCODE='23505'; END IF;
 RETURN pg_catalog.jsonb_build_object('candidate_id',prior.candidate_id,'revision',prior.revision,
 'predecessor_candidate_id',prior.predecessor_candidate_id,'package',prior.package);
 END IF;
 SELECT package INTO retained FROM {s}.decisions WHERE attempt_key=p_attempt;
 IF FOUND THEN RAISE EXCEPTION 'committed attempt cannot be superseded' USING ERRCODE='23514'; END IF;
 SELECT package INTO base FROM {s}.approved_commits WHERE attempt_key=p_attempt;
 IF NOT FOUND THEN
 IF p_revision<>1 THEN RAISE EXCEPTION 'initial proposal revision required' USING ERRCODE='23514'; END IF;
 PERFORM {s}.approve_commit(p_package);
 base:=p_package;
 ELSE
 IF p_revision<>selected.revision+1 OR p_predecessor IS DISTINCT FROM selected.candidate_id THEN
 RAISE EXCEPTION 'current exact predecessor required' USING ERRCODE='23514'; END IF;
 IF (p_package - {immutable_fields}) IS DISTINCT FROM (base - {immutable_fields})
 OR ((p_package->'history_record') - ARRAY['sequence','predecessor_authenticated_digest','authenticated_digest']::text[])
 IS DISTINCT FROM ((base->'history_record') - ARRAY['sequence','predecessor_authenticated_digest','authenticated_digest']::text[])
 OR p_package->'history_head'->'canonical_credential_hex' IS DISTINCT FROM base->'history_head'->'canonical_credential_hex'
 THEN RAISE EXCEPTION 'reproposal must preserve original proof and event bytes' USING ERRCODE='23514'; END IF;
 END IF;
 PERFORM {s}.validate_history_proposal_base(p_attempt,p_package);
 PERFORM {s}.validate_history_proposal_enrollment(p_package);
 PERFORM {e}.reproposal_validate_unbound(p_package);
 SELECT package INTO retained FROM {s}.history_proposals WHERE attempt_key=p_attempt AND candidate_id=selected.candidate_id;
 PERFORM {h}.reproposal_validate_successor(p_package,retained);
 INSERT INTO {s}.history_proposals VALUES(p_attempt,p_revision,candidate,p_predecessor,p_package);
 UPDATE {s}.history_proposal_current SET revision=p_revision,candidate_id=candidate WHERE attempt_key=p_attempt;
 RETURN pg_catalog.jsonb_build_object('candidate_id',candidate,'revision',p_revision,
 'predecessor_candidate_id',p_predecessor,'package',p_package);
END
""",
        "commit_history_proposal(bytea,text)": commit,
        "validate_history_proposal_decision()": validate,
        "validate_history_proposal_base(bytea,jsonb)": base,
        "validate_history_proposal_package(jsonb)": package_validation,
        "validate_history_proposal_enrollment(jsonb)": enrollment,
        "guard_history_proposal_selector()": f"""
BEGIN
 IF session_user::text NOT IN ('{config.issuer_writer_role}','{config.test_admin_role}')
 OR current_user::text IS DISTINCT FROM '{config.schema_owner_role}'
 OR TG_OP='DELETE' THEN RAISE EXCEPTION 'protected proposal selector' USING ERRCODE='42501'; END IF;
 IF TG_OP='UPDATE' AND (NEW.attempt_key IS DISTINCT FROM OLD.attempt_key OR NEW.revision<OLD.revision)
 THEN RAISE EXCEPTION 'proposal selector cannot regress' USING ERRCODE='23514'; END IF;
 RETURN NEW;
END
""",
    }


def reviewed_reproposal_inventory(
    conn: psycopg.Connection[Any],
    config: PostgreSQLAtomicRootProofProvisioning,
) -> dict[tuple[str, str], tuple[str, str, str, str]]:
    """Additional fixed owner helpers allow the explicit verifier preflight."""
    inventory = reviewed_function_inventory(conn, config)
    result = {}
    alias_names = {
        "atomic_read_intent",
        "atomic_read_retention",
        "atomic_intent_member",
        "atomic_retention_member",
        "atomic_history_member",
        "atomic_lock_credential",
    }
    for (schema, signature), (owner, source, declaration, returns) in inventory.items():
        name = signature.split("(", 1)[0]
        if schema != config.schema and name in alias_names:
            result[(schema, "reproposal_" + signature)] = (
                owner,
                _admin_alias(source, config),
                declaration,
                returns,
            )
    e, h, s = map(_q, (config.entitlement_schema, config.history_schema, config.schema))
    owner = next(
        value[0] for key, value in inventory.items() if key[0] == config.entitlement_schema
    )
    both = _admin_alias(_guard(config, owner), config)
    result[(config.entitlement_schema, "reproposal_validate_unbound(jsonb)")] = (
        owner,
        f"""
DECLARE actual bigint; old_state jsonb;
BEGIN
 {both}
 PERFORM {e}.validate_state((p->'bound_state')::json);
 SELECT current_revision INTO actual FROM {e}.lineages WHERE lookup_handle=p->>'lookup_handle'
 AND environment=p->>'environment' AND trust_domain=p->>'trust_domain' FOR UPDATE;
 IF actual IS DISTINCT FROM (p->>'expected_revision')::bigint THEN
 RAISE EXCEPTION 'original entitlement revision differs' USING ERRCODE='23514'; END IF;
 SELECT state INTO old_state FROM {e}.history WHERE lookup_handle=p->>'lookup_handle'
 AND environment=p->>'environment' AND trust_domain=p->>'trust_domain' AND revision=actual;
 PERFORM {e}.validate_state(old_state::json);
 IF old_state->>'lifecycle' IS DISTINCT FROM 'ACTIVE' OR old_state#>>'{{binding,kind}}' IS DISTINCT FROM 'UNBOUND'
 OR p->'bound_state'->>'lifecycle' IS DISTINCT FROM 'ACTIVE'
 OR p->'bound_state'#>>'{{binding,kind}}' IS DISTINCT FROM 'BOUND'
 OR p->'bound_state'->'subject' IS DISTINCT FROM old_state->'subject'
 OR p->'bound_state'->'identity' IS DISTINCT FROM old_state->'identity'
 OR p->'bound_state'->'provenance' IS DISTINCT FROM old_state->'provenance'
 OR (p->'bound_state'->>'authoritative_state_revision')::bigint IS DISTINCT FROM actual+1
 OR (p->'bound_state'->>'predecessor_revision')::bigint IS DISTINCT FROM actual
 OR EXISTS(SELECT 1 FROM {e}.history WHERE lookup_handle=p->>'lookup_handle'
 AND environment=p->>'environment' AND trust_domain=p->>'trust_domain' AND state#>>'{{binding,kind}}'='BOUND')
 THEN RAISE EXCEPTION 'exact ACTIVE UNBOUND predecessor required' USING ERRCODE='23514'; END IF;
END
""",
        "p jsonb",
        "void",
    )
    owner = next(value[0] for key, value in inventory.items() if key[0] == config.history_schema)
    both = _admin_alias(_guard(config, owner), config)
    result[(config.history_schema, "reproposal_validate_successor(jsonb,jsonb)")] = (
        owner,
        f"""
DECLARE st {h}.streams%ROWTYPE; stream bytea; record_bytes bytea; body bytea; event bytea; payload bytea;
 head_bytes bytea; head_body bytea; physical jsonb; sequence bigint; digest text;
BEGIN
 {both}
 stream:=pg_catalog.decode(p->>'stream_hex','hex');
 SELECT * INTO st FROM {h}.streams WHERE stream_identity=stream FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'history stream missing' USING ERRCODE='23514'; END IF;
 PERFORM {h}.verify_stream(stream);
 IF st.head_sequence>0 AND NOT EXISTS(SELECT 1 FROM {h}.attestations a WHERE a.stream_identity=stream
 AND a.sequence=st.head_sequence AND a.record_digest=st.head_digest
 AND {s}.assert_history_link(stream,a.sequence,NULL,a.canonical_head,a.signature)) THEN
 RAISE EXCEPTION 'authoritative retained signed predecessor required' USING ERRCODE='23514'; END IF;
 IF prior IS NOT NULL AND (prior->>'expected_history_digest' IS NOT DISTINCT FROM st.head_digest
 OR (prior->'history_record'->>'sequence')::bigint>st.head_sequence) THEN
 RAISE EXCEPTION 'current candidate is not a stale successor' USING ERRCODE='23514'; END IF;
 IF prior IS NOT NULL AND (prior->'history_record'->>'predecessor_authenticated_digest'
 IS DISTINCT FROM prior->>'expected_history_digest'
 OR CASE WHEN (prior->'history_record'->>'sequence')::bigint=1
 THEN prior->>'expected_history_digest' IS DISTINCT FROM 'NO_PREDECESSOR'
 ELSE NOT EXISTS(SELECT 1 FROM {h}.records old WHERE old.stream_identity=stream
 AND old.sequence=(prior->'history_record'->>'sequence')::bigint-1
 AND old.authenticated_digest=prior->>'expected_history_digest') END) THEN
 RAISE EXCEPTION 'previous candidate predecessor is not retained ancestry' USING ERRCODE='23514'; END IF;
 IF p->>'expected_history_digest' IS DISTINCT FROM st.head_digest
 OR (p->'history_record'->>'sequence')::bigint IS DISTINCT FROM st.head_sequence+1
 OR p->'history_record'->>'predecessor_authenticated_digest' IS DISTINCT FROM st.head_digest
 THEN RAISE EXCEPTION 'authoritative current history successor required' USING ERRCODE='40001'; END IF;
 physical:=p->'history_record';
 IF (SELECT pg_catalog.array_agg(k ORDER BY k) FROM pg_catalog.jsonb_object_keys(physical) k)
 IS DISTINCT FROM ARRAY['authenticated_digest','canonical_event_identity_hex','canonical_payload_hex','event_digest','predecessor_authenticated_digest','sequence']::text[]
 THEN RAISE EXCEPTION 'exact history record fields required' USING ERRCODE='22023'; END IF;
 event:=pg_catalog.decode(physical->>'canonical_event_identity_hex','hex');
 payload:=pg_catalog.decode(physical->>'canonical_payload_hex','hex');
 PERFORM {h}.validate_canonical_json(event); PERFORM {h}.validate_canonical_json(payload);
 IF EXISTS(SELECT 1 FROM {h}.records WHERE stream_identity=stream AND event_identity=event) THEN
 RAISE EXCEPTION 'original history event already retained' USING ERRCODE='23514'; END IF;
 body:=pg_catalog.convert_to('{{"canonical_event_payload":','UTF8') || payload
 || pg_catalog.convert_to(',"event_digest":','UTF8') || pg_catalog.convert_to(pg_catalog.to_json(physical->>'event_digest')::text,'UTF8')
 || pg_catalog.convert_to(',"event_identity":','UTF8') || event
 || pg_catalog.convert_to(',"predecessor_authenticated_digest":','UTF8') || pg_catalog.convert_to(pg_catalog.to_json(st.head_digest)::text,'UTF8')
 || pg_catalog.convert_to(',"sequence":' || (st.head_sequence+1)::text || ',"stream":','UTF8') || stream || pg_catalog.convert_to('}}','UTF8');
 record_bytes:=pg_catalog.decode(p->>'history_record_hex','hex');
 digest:='sha256:' || pg_catalog.encode(pg_catalog.sha256(pg_catalog.decode('{RECORD_DOMAIN.hex()}','hex') || body),'hex');
 IF record_bytes IS DISTINCT FROM body OR physical->>'authenticated_digest' IS DISTINCT FROM digest
 OR physical->>'event_digest' IS DISTINCT FROM 'sha256:' || pg_catalog.encode(pg_catalog.sha256(
 pg_catalog.decode('{_EVENT_DOMAIN.hex()}','hex') || payload),'hex') THEN
 RAISE EXCEPTION 'exact canonical successor record required' USING ERRCODE='23514'; END IF;
 physical:=p->'history_head';
 IF (SELECT pg_catalog.array_agg(k ORDER BY k) FROM pg_catalog.jsonb_object_keys(physical) k)
 IS DISTINCT FROM ARRAY['canonical_credential_hex','record_digest','sequence']::text[]
 OR physical->>'record_digest' IS DISTINCT FROM digest
 OR (physical->>'sequence')::bigint IS DISTINCT FROM st.head_sequence+1
 THEN RAISE EXCEPTION 'successor head identity differs' USING ERRCODE='23514'; END IF;
 head_bytes:=pg_catalog.decode(p->>'history_head_hex','hex');
 IF pg_catalog.substring(head_bytes,1,{len(ATTESTATION_DOMAIN)}) IS DISTINCT FROM pg_catalog.decode('{ATTESTATION_DOMAIN.hex()}','hex') THEN
 RAISE EXCEPTION 'canonical signed successor head required' USING ERRCODE='23514'; END IF;
 head_body:=pg_catalog.substring(head_bytes,{len(ATTESTATION_DOMAIN) + 1});
 PERFORM {h}.validate_canonical_json(head_body);
 PERFORM {h}.canonical_object_members(head_body,ARRAY['record_digest','sequence','signing_credential_identity','stream']);
 IF {h}.canonical_object_member(head_body,'stream') IS DISTINCT FROM stream
 OR {h}.canonical_object_member(head_body,'sequence') IS DISTINCT FROM pg_catalog.convert_to((st.head_sequence+1)::text,'UTF8')
 OR {h}.canonical_object_member(head_body,'record_digest') IS DISTINCT FROM pg_catalog.convert_to(pg_catalog.to_json(digest)::text,'UTF8')
 OR {h}.canonical_object_member(head_body,'signing_credential_identity') IS DISTINCT FROM pg_catalog.decode(physical->>'canonical_credential_hex','hex')
 OR pg_catalog.octet_length(pg_catalog.decode(p->>'history_signature_hex','hex'))<>64 THEN
 RAISE EXCEPTION 'successor signed head relation differs' USING ERRCODE='23514'; END IF;
END
""",
        "p jsonb,prior jsonb",
        "void",
    )
    for signature, source in reviewed_reproposal_function_sources(config).items():
        declaration, returns = FUNCTION_DECLARATIONS[signature]
        result[(config.schema, signature)] = (
            config.schema_owner_role,
            source,
            declaration,
            returns,
        )
    return result


def source_hashes(inventory: dict[tuple[str, str], tuple[str, str, str, str]]) -> dict[str, str]:
    return {
        schema + "." + signature: hashlib.sha256(source.encode()).hexdigest()
        for (schema, signature), (_, source, _, _) in sorted(inventory.items())
    }


def install_history_successor_reproposal_v1(
    conn: psycopg.Connection[Any],
    config: PostgreSQLAtomicRootProofProvisioning,
    predecessor_catalog_sha256: str,
) -> None:
    """One explicit bootstrap transaction; approval and decision bytes survive."""
    if (
        type(config) is not PostgreSQLAtomicRootProofProvisioning
        or conn.info.transaction_status != psycopg.pq.TransactionStatus.INTRANS
    ):
        raise ValueError("explicit qualified history reproposal migration required")
    inventory = reviewed_reproposal_inventory(conn, config)
    s = sql.Identifier(config.schema)
    conn.execute(
        sql.SQL("""
CREATE TABLE {s}.history_reproposal_contract(singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton),
 migration_id text NOT NULL,schema_version integer NOT NULL CHECK(schema_version=1),boundary text NOT NULL,
 predecessor_catalog_sha256 text NOT NULL,predecessor_snapshot jsonb NOT NULL,
 predecessor_snapshot_sha256 bytea NOT NULL,compiled_source_sha256 jsonb NOT NULL);
CREATE TABLE {s}.history_proposals(attempt_key bytea NOT NULL,revision bigint NOT NULL CHECK(revision>=1),
 candidate_id text NOT NULL CHECK(candidate_id ~ '^hp1_[0-9a-f]{{64}}$'),predecessor_candidate_id text,
 package jsonb NOT NULL,PRIMARY KEY(attempt_key,revision),UNIQUE(attempt_key,revision,candidate_id),
 UNIQUE(attempt_key,candidate_id),FOREIGN KEY(attempt_key) REFERENCES {s}.approved_commits(attempt_key),
 FOREIGN KEY(attempt_key,predecessor_candidate_id) REFERENCES {s}.history_proposals(attempt_key,candidate_id),
 CHECK((revision=1 AND predecessor_candidate_id IS NULL) OR (revision>1 AND predecessor_candidate_id IS NOT NULL)));
CREATE TABLE {s}.history_proposal_current(attempt_key bytea PRIMARY KEY,revision bigint NOT NULL,candidate_id text NOT NULL,
 FOREIGN KEY(attempt_key,revision,candidate_id) REFERENCES {s}.history_proposals(attempt_key,revision,candidate_id)
 DEFERRABLE INITIALLY DEFERRED);
""").format(s=s)
    )
    conn.execute(
        sql.SQL(
            "INSERT INTO {}.history_reproposal_contract SELECT true,%s,1,%s,%s,snapshot,pg_catalog.sha256(pg_catalog.convert_to(snapshot::text,'UTF8')),%s FROM {}.catalog_manifest WHERE singleton"
        ).format(s, s),
        (MIGRATION_ID, BOUNDARY, predecessor_catalog_sha256, Jsonb(source_hashes(inventory))),
    )
    for (schema, signature), (owner, source, declaration, returns) in inventory.items():
        conn.execute(
            sql.SQL(
                "CREATE FUNCTION {}.{}({}) RETURNS {} LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS {}"
            ).format(
                sql.Identifier(schema),
                sql.Identifier(signature.split("(", 1)[0]),
                sql.SQL(declaration),
                sql.SQL(returns),
                sql.Literal(source),
            )
        )
        conn.execute(
            sql.SQL("ALTER FUNCTION {}.{} OWNER TO {}").format(
                sql.Identifier(schema), sql.SQL(signature), sql.Identifier(owner)
            )
        )
        conn.execute(
            sql.SQL("REVOKE ALL ON FUNCTION {}.{} FROM PUBLIC,{},{}").format(
                sql.Identifier(schema),
                sql.SQL(signature),
                sql.Identifier(config.issuer_writer_role),
                sql.Identifier(config.test_admin_role),
            )
        )
        callers = []
        if schema != config.schema:
            callers.append(config.schema_owner_role)
        if schema == config.schema and signature in RUNTIME_FUNCTIONS:
            callers.append(config.issuer_writer_role)
        if schema == config.schema and signature in ADMIN_FUNCTIONS:
            callers.append(config.test_admin_role)
        for caller in callers:
            conn.execute(
                sql.SQL("GRANT EXECUTE ON FUNCTION {}.{} TO {}").format(
                    sql.Identifier(schema), sql.SQL(signature), sql.Identifier(caller)
                )
            )
    for signature in CLOSED_FUNCTIONS:
        conn.execute(
            sql.SQL("REVOKE ALL ON FUNCTION {}.{} FROM PUBLIC,{},{}").format(
                s,
                sql.SQL(signature),
                sql.Identifier(config.issuer_writer_role),
                sql.Identifier(config.test_admin_role),
            )
        )
    # Preserve the original V1 validator source and OID. Only this explicitly
    # reviewed deferred trigger switches to the new proposal-aware validator.
    conn.execute(sql.SQL("DROP TRIGGER atomic_decision_link ON {}.decisions").format(s))
    conn.execute(
        sql.SQL(
            "CREATE CONSTRAINT TRIGGER atomic_decision_link AFTER INSERT ON {}.decisions DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION {}.validate_history_proposal_decision()"
        ).format(s, s)
    )
    conn.execute(
        sql.SQL("ALTER TABLE {}.decisions ENABLE ALWAYS TRIGGER atomic_decision_link").format(s)
    )
    for table in ("history_reproposal_contract", "history_proposals", "history_proposal_current"):
        conn.execute(
            sql.SQL("ALTER TABLE {}.{} OWNER TO {}").format(
                s, sql.Identifier(table), sql.Identifier(config.schema_owner_role)
            )
        )
        conn.execute(
            sql.SQL("REVOKE ALL ON {}.{} FROM PUBLIC,{},{}").format(
                s,
                sql.Identifier(table),
                sql.Identifier(config.issuer_writer_role),
                sql.Identifier(config.test_admin_role),
            )
        )
        for name, event, level in (
            ("reproposal_immutable_row", "UPDATE OR DELETE", "ROW"),
            ("reproposal_immutable_truncate", "TRUNCATE", "STATEMENT"),
        ):
            function = (
                "guard_history_proposal_selector"
                if table == "history_proposal_current" and level == "ROW"
                else "reject_mutation"
            )
            conn.execute(
                sql.SQL(
                    "CREATE TRIGGER {} BEFORE {} ON {}.{} FOR EACH {} EXECUTE FUNCTION {}.{}()"
                ).format(
                    sql.Identifier(name),
                    sql.SQL(event),
                    s,
                    sql.Identifier(table),
                    sql.SQL(level),
                    s,
                    sql.Identifier(function),
                )
            )
            conn.execute(
                sql.SQL("ALTER TABLE {}.{} ENABLE ALWAYS TRIGGER {}").format(
                    s, sql.Identifier(table), sql.Identifier(name)
                )
            )
    # Finish relation DDL before seeding a nonempty legacy approval journal.
    # The selector's deferred FK must remain deferred until native COMMIT;
    # its pending events would prevent subsequent ALTER TABLE statements.
    conn.execute(
        sql.SQL(
            "INSERT INTO {s}.history_proposals SELECT attempt_key,1,{s}.history_proposal_id(package,1,NULL),NULL,package FROM {s}.approved_commits; INSERT INTO {s}.history_proposal_current SELECT attempt_key,revision,candidate_id FROM {s}.history_proposals"
        ).format(s=s)
    )
    conn.execute(sql.SQL("ALTER TABLE {}.catalog_manifest DISABLE TRIGGER immutable_row").format(s))
    conn.execute("SAVEPOINT history_reproposal_manifest_capture")
    conn.execute(
        sql.SQL("ALTER TABLE {}.catalog_manifest ENABLE ALWAYS TRIGGER immutable_row").format(s)
    )
    snapshot = conn.execute(sql.SQL("SELECT {}.catalog_snapshot()").format(s)).fetchone()
    if snapshot is None:
        raise ValueError("reviewed reproposal manifest unavailable")
    conn.execute("ROLLBACK TO SAVEPOINT history_reproposal_manifest_capture")
    conn.execute("RELEASE SAVEPOINT history_reproposal_manifest_capture")
    conn.execute(
        sql.SQL("UPDATE {}.catalog_manifest SET snapshot=%s WHERE singleton").format(s),
        (Jsonb(snapshot[0]),),
    )
    conn.execute(
        sql.SQL("ALTER TABLE {}.catalog_manifest ENABLE ALWAYS TRIGGER immutable_row").format(s)
    )


__all__ = [
    "MIGRATION_ID",
    "SCHEMA_VERSION",
    "BOUNDARY",
    "install_history_successor_reproposal_v1",
    "reviewed_reproposal_function_sources",
    "reviewed_reproposal_inventory",
]
