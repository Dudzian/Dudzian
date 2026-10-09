# Stage 9 signed immutable issuance attempt

This implementation ends at **SIGNED_IMMUTABLE_DURABLE_NOT_SENT**. It prepares
and retains an issuance request, never sends it. Entitlement remains UNBOUND;
no issuer BOUND decision, root proof, admission, PREPARED, freshness, membership,
AccountGenesis COMMITTED or Stage 10 operation is added.

The subordinate contract and freeze are in
`cryptohunter_product_architecture/stage9_root_proof_signed_immutable_attempt_*.json`.
Historical parents remain byte-identical. Current global status is solely
`deployment/stage9_current_status.json`.

## Exact signed request

The closed `RootProofIssuanceRequestV1` payload contains:

`schema_version`, `environment`, `trust_domain`, `issuer_target_namespace`,
`entitlement_id`, `entitlement_generation`, `logical_operation_id`, `account_id`,
`canonical_genesis_request_fingerprint_sha256`, `initial_binding_reference`,
`initial_binding_digest_sha256`, `requester_id`, `requester_key_id`,
`requester_key_version`, `provisioning_principal_id`, `claimant_key_id`,
`claimant_key_version`, `issuance_attempt_id`.

Schema version is the string `1`. Repository JCS produces the exact UTF-8 bytes.
The signed payload digest is lowercase SHA-256 of these bytes, without a domain
prefix. The parent/source specifies no production issuer namespace value; this
child freezes `CryptoHunter.Stage9.RootProofIssuer.V1`. Environment and trust
domain remain explicit signed fields. No public caller chooses the namespace.

Both Ed25519 signatures cover the same retained bytes including reserved `rpa_`:

- Requester: ASCII `CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUEST_V1`
  followed by NUL and the canonical bytes.
- Claimant: ASCII `CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_ENTITLEMENT_CLAIM_V1`
  followed by NUL and the same canonical bytes.

Signatures use canonical unpadded base64url of exactly 64 bytes. Profiles append
`/JCS-SHA256-Ed25519-v1` to the respective domain literal. The existing
`AttemptIdentity.digest_sha256` remains SHA-256 of its existing identity domain
plus JCS of its payload. It differs from the request payload digest; no identity
digest appears in the signed request.

## Custody and offline ceremony

Requester owner is `CryptoHunterAccountAuthority`; its custody role is
`CHA_ROOT_PROOF_ISSUANCE_REQUESTER_CUSTODY`. The claimant belongs to the pre-account
provisioning/deployment security authority. CHA invokes only its narrow
`authorize_entitlement_claim` operation. Candidate account identity never becomes
the claimant. Neither port is a RootProofIssuer signing provider.

Dedicated directories, native keyring services, credential namespaces, lifecycle
namespaces and handles separate both owners. Service scope binds PRODUCTION,
trust domain, role and principal. Each private key is generated independently.
Runtime derives its public bytes from protected custody and checks exact
authoritative public material, key id/version, principal, role and ACTIVE state.
Both raw public keys must differ. Full public-registry role/non-alias qualification
continues before every operation.

Current main has no installed runtime factory exposing local issuer/freshness/
Catalog/storage signing identities to this CHA flow. Their distinct existing
roles are not reused, and this PR makes no material-alias claim for unavailable
providers. A future composition supplying those identities must qualify their
raw material against these two keys before enabling signing.

The offline administrator first writes a STAGED public draft, then creates the
private seed in its dedicated protected keyring service. It returns public facts
only. The administrator provisions those public bytes into the existing genuine
PostgreSQL registry, then activates custody after deriving the private public key
and verifying exact ACTIVE registry evidence. STAGED custody cannot sign.

This is a ceremony across three stores, not a distributed transaction. A crash
after draft publication but before keyring creation leaves an unusable draft;
retry fails closed without regenerating material. A missing or mismatched public
credential prevents activation. Missing/mismatched private custody prevents
runtime authorization even if the public registry is ACTIVE. An administrator
must reconcile or abandon an incomplete ceremony outside the runtime API.
Lifecycle can advance ACTIVE → VERIFY_ONLY → REVOKED, or ACTIVE → REVOKED.
Runtime cannot generate, rotate, delete, change lifecycle or export private keys.

Existing hardened publication and locking primitives are reused. Their current
PRODUCTION_LOCAL lock implementation requires POSIX. Windows native custody is
therefore blocked; Windows readiness remains NOT_READY. Tests mock the native OS
keyring layer; they do not provision installed production credentials or perform
legal enrollment.

## Schema and persistence

Schema v6 adds append-only `issuance_requests`, `signing_intents` and
`signature_checkpoints`, with exact foreign keys, unique identities and immutable
update/delete triggers. Request bytes are a BLOB with authority-generated
`immutable:req:sha256:<lowercase SHA256>` reference. Reads recompute its digest,
check canonical equality against the retained reservation authorization, and
verify retained signatures. A reference cannot resolve unequal bytes. Paths,
rowids and process addresses are never protocol identity.

Startup obtains BEGIN IMMEDIATE. A one-time v5→v6 migration verifies the existing
schema and history, adds the extension, advances metadata, restores its trigger,
and verifies the destination before one commit. Every unsigned reservation row,
authorization BLOB, `rpa_`, current state and fence is preserved. Migration accepts
only unsigned reservation history. Signed/recovery/replacement v5 history is
rejected because exact previously signed request bytes cannot be invented.
An injected migration failure rolls back DDL, metadata and trigger changes.

## Crash semantics and concurrency

The four states are RESERVED_AWAITING_SIGNATURES → REQUEST_SIGNED_BY_REQUESTER →
CLAIMANT_AUTHORIZED → SIGNED_IMMUTABLE_DURABLE_NOT_SENT. Each role has one immutable
invocation latch, bound to the request reference and complete signer identity.
Its latch is committed **before** the private operation. After the operation,
signature verification, signature append, transition append and current-state
CAS commit together. Finalization appends existing AttemptIdentity, its transition
and fenced current projection atomically, without changing the `rpa_`.

Local deterministic Ed25519 may return before its signature transaction commits.
If the process dies in that interval, the durable latch survives. An intent
without a checkpoint returns `SIGNATURE_OUTCOME_NOT_DURABLE` and permanently
blocks that role/attempt. It is never cleared, treated as evidence of no signing,
or retried. A crash after the latch but before the actual invocation also blocks.
This intentionally sacrifices availability to honor `resigning_allowed=false`.
There is no claim of lossless recovery or atomic signing across stores.

After a committed checkpoint, restart reuses the exact retained signature and
request bytes. Lost finalization response converges on the same identity and
digest with no signer call. Relevant credential changes fail closed even after
either signature; VERIFY_ONLY historical verification never authorizes completion
of a new UNBOUND request. No key substitution occurs.

A protected-host advisory lock serializes the complete signing sequence across
separate SQLite commits and processes. SQLite writer transactions, immutable unique
rows, exact byte/signature comparisons and fenced CAS remain the authority.
Two exact concurrent callers converge; unequal identity is corruption. The host
administrator, native keyring and working local filesystem locks remain trusted.
Independent live PostgreSQL checks and SQLite commits are not a distributed
transaction; final capability access requalifies current operation evidence.

Tests cover every write/CAS cut, abrupt `os._exit` without Python transaction
cleanup, restart, lost response, one invocation per role, concurrent exact callers,
canonical Unicode/control-character vectors, replay, tampering, migration rollback
and genuine PostgreSQL binding through the guarded InitialBinding lineage.

## API, status and CI

`sign_root_proof_issuance_attempt` accepts only the existing verified reservation.
`resume_root_proof_issuance_attempt` accepts existing verified InitialBinding and
issuance authorization; it never reserves or remints. The returned exact opaque
`VerifiedSignedImmutableRootProofIssuanceAttempt` cannot be publicly constructed,
subclassed or copied. Its consequential access rechecks upstream/custody and
exact durable current identity/digest/bytes. Raw DTOs, signatures and rows grant
no Stage 9 authority.

Implementation status advances to IMPLEMENTED for both custody roles and the
signed immutable boundary. Live availability remains
BLOCKED_UNTIL_PRODUCTION_CREDENTIAL_PROVISIONING. Production pre-account credentials
remain NOT_PROVISIONED, root proof NOT_ISSUED, admission/PREPARED NOT_STARTED,
production provisioning false and legal enrollment NOT_PERFORMED. The formal
Windows checklist count is unchanged.

Windows has a dedicated serial `cha-signed-attempt` shard, excluded from
`cha-root-proof`. Existing shards and timeout fixes remain. The Linux quality
workflow runs the new serial suite, genuine PostgreSQL checks and changed-code
coverage. No xdist or hardware qualification gate is added.
