# Stage 9 production-local durable issuer history

This change implements a separate PostgreSQL persistence component for the
existing `IssuerAuthenticatedHistory` port. It preserves the executable value
model in `bot_core.authenticated_issuer_history` and keeps
`ReferenceAuthenticatedHistory` as the in-memory semantic oracle. Durable
history is one production component; it does not complete or activate Semantic
RootProofIssuer.

## Source discovery and contract reuse

The implementation was reviewed against approved main
`570c5b8c3a6687c0bdb93f2af11bcdf840f83649`, these concrete modules, and the
existing executable contract:

- `bot_core/authenticated_issuer_history.py`
- `bot_core/root_proof_issuer_substrate.py`
- `bot_core/postgresql_entitlement_registry.py`
- `bot_core/postgresql_root_proof_issuance_authority.py`
- `bot_core/root_proof_issuer_runtime.py`
- [Authenticated history and checkpoint contract](authenticated_issuer_history_checkpoint_contract.md)
- [M0.5 independent issuer contract](cryptohunter_product_architecture/m05_account_genesis_independent_root_proof_issuer_contract.json)
- [M0.5 implementation readiness contract](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_implementation_readiness_contract.json)
- [M0.5 production substrate selection](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.json)
- [M0.5 physical persistence and crash atomicity](cryptohunter_product_architecture/m05_account_genesis_physical_persistence_crash_atomicity_contract.json)
- `deployment/stage9_current_status.json`

The stream identity is exactly `(stream_id, issuer_authority_identity,
security_profile, environment, trust_domain, product_scope, security_epoch)`.
All fields are bound into canonical records; the epoch is a positive integer.
Changing any dimension selects a different stream, not an alternate view of
existing records. Event identity is exactly `(logical_operation_id,
issuance_attempt_id, root_proof_id)`.

Genesis has sequence 1 and predecessor `NO_PREDECESSOR`. A successor has the
next contiguous sequence and the preceding authenticated record digest. The
adapter reuses `build_record`, `HistoryRecord`, `HistoryStreamIdentity`,
`HistoryEventIdentity`, canonical JSON, and the existing domain-separated
digests. JSON spelling, key order, or database JSON normalization cannot define
a second record format. Full authoritative reads must validate every retained
record and predecessor, the retained event index, and the current-head relation;
a broken prefix is corruption, not absence.

The oracle's `append` returns `(original_record, True)` for an exact event and
payload retry, even when its original expected predecessor is now stale. The
same event identity with another payload is a conflict. A new event requires
the exact current predecessor and returns `(new_record, False)`. PostgreSQL
provides the cross-process transaction and predecessor CAS; a Python lock is
not the storage authority. A lost response requires an exact replay or
authoritative reread. Connection failure or an ambiguous commit never implies
that the event was absent.

V2 `AttestedHistoryHead` binds full stream identity, sequence, record digest,
canonical attestation bytes, signature, and complete namespaced
`CredentialRoleIdentity`. Authentication reuses `attest_head` and `verify_head`
through the separate `HistoryAttestationSigningProvider`. Its public-key
resolution, key-material identity, history-specific semantic role, and
lifecycle rules remain mandatory. No private key belongs in PostgreSQL or
database configuration.

The SQL head-retention function validates canonical bytes, signer identity
shape, and the exact retained-record binding. Cryptographic signature and
lifecycle verification require the trusted signing provider through the adapter;
raw database bytes alone are never authenticated-head authority.

The existing entitlement registry remains its own PostgreSQL authority. The
new history component has no registry BIND function and performs no writes to
entitlement or CHA state. The existing issuance-authority composition and
semantic preflight remain separate. Preflight returns only
`VERIFIED_NOT_AUTHORIZED_TO_ISSUE`; that reporting DTO is not accepted as
history append authorization or promoted into a Root Proof issuance decision.
An opaque retained history payload or a valid history attestation establishes
history integrity within the selected authority. It does not establish a legal
Root Proof, entitlement decision, legal enrollment, or production provisioning.

## Adapter operations

`PostgreSQLAuthenticatedIssuerHistory(connection, *, schema, stream)` takes the
existing redacted `PostgreSQLConnectionConfig` and an exact
`HistoryStreamIdentity`. Construction and each authority operation qualify the
installed schema and the directly authenticated runtime principal. Provider
identity/capability evidence uses the existing substrate types and
`ISSUER_AUTHENTICATED_HISTORY` role; it is not an override that qualifies an
in-memory provider or the complete issuer composition.

`append(expected_digest=..., event_identity=..., payload=...)` preserves the
oracle's tuple result. `append_exact_successor(expected_head, record)` is the
substrate-port facade and accepts only exact `HistoryRecord` values, with
`None` for a genesis predecessor. It validates the exact candidate relation
before executing the database append. `current_head` returns the current
`HistoryRecord`; an unsigned record is not an attestation.
`current_record`, `record_at`, `retained_history`, and `verify` read one
complete verified transaction snapshot, including physical stream/event/head
envelope consistency. Returned values are detached from the durable authority.

Signed-head creation remains outside PostgreSQL. Obtain a head through the
existing `attest_head(record, signing_provider)` and persist it with
`retain_attested_head(head, verification_authority)`. The adapter verifies its
exact retained record and trusted signature before retention; replay returns
the original head and another head at that sequence conflicts. Use
`retained_attested_head(verification_authority=...)` for the exact current
head, or an explicit sequence for a retained historical head. Reads reverify
the public signing authority. Missing retained attestation is distinct from an
empty history; it grants no issuance or checkpoint authorization.

`HistoryStorageUnavailable` means the database could not supply authoritative
history. `HistoryOperationIndeterminate` means a database write did not produce
an unambiguous response; the caller must resolve it with an exact event/head
retry or authoritative reread. Both are fail-closed contract errors. The
adapter does not blindly retry a write after connection loss or infer absence
from an error. An explicit reviewed SQL corruption rejection is a determinate
`HistoryContractError`; an explicit stale-CAS rejection is likewise a conflict,
not a claim that a different write won. Corrupt canonical bytes, changed
digests, sequence gaps, wrong predecessors, duplicate/forked identities, scope
disagreement, and conflicting replay produce `HistoryContractError`.

## Offline installation and access control

The separate schema installer is
`bot_core.postgresql_issuer_history_schema`, schema identity
`cryptohunter.issuer_authenticated_history.postgresql`, version 1, storage
family `POSTGRESQL_ISSUER_AUTHENTICATED_HISTORY_V1`. PostgreSQL 16 or later,
`fsync=on`, and `synchronous_commit=on` are required. Qualification checks these
settings inside every authority operation; disabling them makes the provider
unavailable.

Installation is a privileged, offline operator action. Runtime construction
never creates roles, schema, or streams. Use a dedicated new schema and three
distinct lowercase PostgreSQL role names. `provision_postgresql_issuer_history`
creates a `NOLOGIN` schema owner and separate `LOGIN` runtime and administrator
roles, all `NOINHERIT`, without role memberships or superuser, database-create,
role-create, replication, or RLS-bypass privileges. The bootstrap principal
needs the privileges to create these objects and establish ownership. It is
not a runtime principal. Names that already exist cause installation to fail
and the transaction to roll back; the installer never overwrites an existing
authority.

The installed tables are `metadata`, `streams`, `records`, and `attestations`.
Stream and event identity components and canonical record/head bytes are
retained as `bytea`; JSON-escaped U+0000 and large exact payload integers are
not coerced into PostgreSQL text or bounded numeric values. One atomic append
locks the stream row, validates the complete retained chain, resolves exact
event replay, compares the predecessor, inserts the immutable successor, and
advances the head. Sequence and full event-identity uniqueness constraints
support the database CAS. Different authorized streams use different rows.

The administrator may execute `provision_stream`; the runtime may execute
`read_stream`, `append_record`, and `retain_head`. Both may read qualification
metadata. Neither receives direct SELECT or DML on streams, records, or
attestations. PUBLIC receives no schema/table/function authority. Reviewed
`SECURITY DEFINER` functions pin `search_path=pg_catalog` and check the direct
session principal. Using `SET ROLE`, an owner/admin connection, an unrelated
role, or a membership grant does not qualify a runtime connection. Per-operation
qualification verifies retained role OIDs/attributes, exact ACLs, owner,
physical schema fingerprint, reviewed function sources and security metadata,
and unexpected auxiliary schema behavior. Physical columns, constraints, and
indexes are also checked against the reviewed local definition rather than
trusting a mutable metadata fingerprint alone.

The private SQL canonical validators in
`bot_core.postgresql_issuer_history_canonical_sql` enforce the existing Python
canonical byte domain even for a direct sanctioned SQL append. Duplicate or
misordered keys, alternate spelling/whitespace/escapes, non-integer numbers,
and malformed envelopes cannot be admitted by recomputing their digests. The
validators retain decoded keys as bytes and scan integer digits without a SQL
numeric conversion. These helpers have no runtime/admin EXECUTE grant; their
source and security metadata are included in qualification.

Configure authenticated administrator and runtime login channels out of band
before provisioning the stream. Passwords or signing keys are not created or
stored by the installer. The isolated CI PostgreSQL `trust` authentication
setting is a test fixture, not a production installation procedure. Production
login policy must authenticate the intended principals and keep bootstrap,
administrator, and runtime connection material separate.

The following operator sequence uses three separately supplied
`PostgreSQLConnectionConfig` objects and a `trusted_stream` from reviewed
deployment configuration. It installs storage and authorizes that exact stream;
it performs no history append or issuance operation.

```python
from bot_core.postgresql_issuer_history_schema import (
    PostgreSQLIssuerHistoryProvisioning,
    provision_postgresql_issuer_history,
    provision_postgresql_issuer_history_stream,
)
from bot_core.postgresql_authenticated_issuer_history import (
    PostgreSQLAuthenticatedIssuerHistory,
)

installation = PostgreSQLIssuerHistoryProvisioning(
    schema="issuer_history",
    schema_owner_role="issuer_history_owner",
    runtime_role="issuer_history_runtime",
    admin_role="issuer_history_admin",
    trust_domain=trusted_stream.trust_domain,
)
provision_postgresql_issuer_history(bootstrap_connection, installation)
# Establish separate authenticated login policy before this administrator call.
provision_postgresql_issuer_history_stream(
    admin_connection, schema=installation.schema, stream=trusted_stream
)
history = PostgreSQLAuthenticatedIssuerHistory(
    runtime_connection, schema=installation.schema, stream=trusted_stream
)
history.verify()
```

There is no automatic or in-place migration path from an unknown schema. Future
version changes need a separately reviewed offline migration with preserved
canonical data and renewed qualification. Back up and recover the complete
authority state under the database operator's policy; never reconstruct a
missing retained event or substitute a newer local attempt from a cache.

Authoritative reads use a read-only repeatable-read transaction. Writes lock
the exact stream row and retain record/head changes in one database
transaction. Full-chain verification and complete snapshot materialization
take time and memory proportional to the stream's retained history. This
change supplies no throughput or large-history capacity qualification; it
does not replace complete verification with a cached digest or unchecked
pagination. Storage and performance qualification for an actual deployment
remain operator responsibilities.

## Checkpoint and reconciliation limits

`LocalCheckpointProvider` remains an in-memory reference authority. Its
checkpoint, UUID authority identity, and digest-linked historical acceptance
chain do not survive restart. `advance` explicitly accepts only the exact
`ReferenceAuthenticatedHistory` implementation. `reconcile_checkpoint` has the
same exact reference-authority restriction. The new PostgreSQL history
component does not bypass these restrictions or manufacture checkpoint
authority from `VerifiedHistoryHead`, `LocalCheckpoint`, or a caller-provided
acceptance record.

The adapter's `reconcile` only compares a verified durable history snapshot to
the exact existing reference checkpoint authority, using the existing outcome
rules. It performs no checkpoint advance. A valid nonempty history with no
committed checkpoint is `STALE`; this read-only classification does not claim
durable checkpoint qualification, positive issuance reconciliation evidence,
or recovery across a checkpoint restart.

A separate review must implement durable checkpoint state and the complete
V2 `HistoricalHeadAcceptanceEvidence` chain, preserve stable authority identity
across restart, and linearize verified history/lifecycle eligibility with
checkpoint CAS. In particular, a durable history database alone cannot qualify
historical REVOKED-key acceptance. Existing exact-head acceptance rules and
fail-closed unavailable behavior continue to apply.

`RootProofIssuanceReconciliationEvidenceV1` is frozen subordinate replacement
authorization evidence. It binds environment, trust domain, product scope,
issuer and registry identity, entitlement and generation, logical operation,
account, request fingerprint, old attempt, INITIAL_BINDING, authoritative
revision, authenticated evidence reference/digest, and verification profile.
It requires the positive `AUTHORITATIVELY_UNBOUND` outcome at an exact current
authority/history fence. Empty history, `NOT_FOUND`, a timeout, and a preflight
report do not prove authoritative unbound state. This change supplies no such
evidence and does not authorize attempt replacement or complete distributed
registry/history/checkpoint/CHA reconciliation.

The preferred eventual sequence remains registry decision, exact proof signing,
history append, checkpoint advance, and CHA finalization with retained exact
evidence. There is no distributed ACID transaction across those stores in this
change. Each later transition needs its own fresh authority checks and CAS.

PostgreSQL commit durability depends on the deployed server, synchronous commit,
storage, and backup configuration. The component can reject malformed records,
gaps, divergent predecessors, and unauthorized runtime SQL. It cannot detect a
privileged restoration of the entire database to an older internally valid
prefix. Production-local guarantees also exclude coordinated full-host restore,
owner/superuser mutation of all retained authority state, and host-admin key
extraction. SERVER_READY still needs independent rollback and administrative
domains and stronger key custody.

## Tests and CI measurement

The real-database integration suite uses isolated PostgreSQL 16.15 schemas and
generated test signing material. Its payloads explicitly represent test history
events, not issuance decisions. It covers recovery in a fresh process, competing
process CAS, identical/conflicting retry after response loss, predecessor/digest/
fork/gap corruption, direct-role ACL failures, all stream replay dimensions,
controlled process termination before append/before commit/after commit, oracle
agreement, and preservation of existing real entitlement and SQLite CHA data.

Completed local measurements:

| Scope | Result | pytest duration |
|---|---:|---:|
| Adapter canonical-envelope and error-boundary unit tests | 31 passed | 0.39 s |
| Existing history oracle and substrate tests | 230 passed | 1.16 s |
| Frozen-contract, oracle and substrate regression tests | 806 passed | 2.20 s |
| New real PostgreSQL history integration tests | 30 passed | 30.87 s |
| Independent real PostgreSQL security tests | 32 passed | 9.08 s |
| Complete added CI history gate, with branch coverage | 653 passed | 37.72 s |
| Existing real PostgreSQL authority regression tests | 369 passed | 2794.44 s |

The complete added gate produced 85% combined branch-aware coverage; its
changed-line coverage was 87%, passing the existing 70% ratchet. The adapter,
schema installer, and canonical SQL helper measured 84%, 75%, and 82%
branch-aware coverage respectively. These scopes overlap; the rows are separate
qualification runs, not an aggregate test count.

Repository Ruff, expanded E/F/B/I checks and formatting on all six new Python
files passed. Configured mypy passed on Linux and `win32` (282 files each).
Blocking custom Semgrep reported zero findings, and its five rule fixtures
passed. The offline Betterleaks working-tree gate passed with its existing
baseline and no new exceptions.

The longest new integration case was fresh-process history/head recovery at
4.75 s. These are measurements in the managed Linux test environment; they are
not measurements of a hosted GitHub Actions run or a production deployment.

The normal `Quality and Security` workflow adds a serial history step using its
existing PostgreSQL 16 service. It includes unit, reference-oracle, new
PostgreSQL integration/security, and relevant frozen-contract tests. It appends
branch coverage for the oracle, adapter, schema installer, and SQL canonical
validator to the existing report before the unchanged 70% changed-code gate.
Every existing test command and coverage scope remains enabled. Ruff and the
format gate include the new files, and all three new modules enter the existing
configured Linux and Windows mypy checks. Semgrep, secret scanning, and manual
hardware/destructive/Stage 10 gates remain unchanged.

The measured new integration component does not justify a separate long-running
job. The schema/process tests run serially with isolated authority fixtures;
no job timeout is increased. No Windows installer, custody filesystem, or
hardware implementation changes. Windows type checking is a portability check,
not live Windows PostgreSQL installation or hardware qualification; the local
Linux run does not establish those production-deployment claims.

## Status and remaining dependencies

Historical frozen snapshots and `deployment/stage9_current_status.json` are
unchanged. `ROOT_PROOF_ISSUER_IMPLEMENTED`,
`PRODUCTION_LOCAL_RUNTIME_AVAILABLE`, and `STAGE9_READY` are not promoted. The
Windows completion count remains `10/15 DONE`, Root Proof remains `NOT_ISSUED`,
and legal production enrollment remains `NOT_PERFORMED`.

Remaining dependencies include durable checkpoint and historical acceptance
review, genuine production pre-account credential provisioning, qualified
deployment trust and independent issuer/history signing composition, positive
reconciliation evidence, and reviewed crash-safe CAS/sign/history/checkpoint
orchestration. Entitlement BIND by Semantic RootProofIssuer, production Root
Proof signing/issuance, external issuer transport, legal enrollment, credential
provisioning, Stage 10, and public proof-issuance endpoints are outside this
change.
