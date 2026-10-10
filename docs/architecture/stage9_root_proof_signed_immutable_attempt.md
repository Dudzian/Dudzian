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

Existing hardened publication primitives are reused. POSIX custody keeps
`fcntl.flock` shared/exclusive ownership. Windows uses the repository's reviewed
`msvcrt.locking` pattern: one byte at offset zero, `LK_NBLCK` exclusive ownership
and explicit `LK_UNLCK` in `finally`. Readers also take the exclusive lock because
this primitive does not supply safe shared ownership. Contention retries for at
most five seconds and fails closed; there is no unlocked fallback or process-local
mutex authority. Native process tests cover concurrent exclusion, exceptions and
abrupt process termination. Unix mode-bit checks remain POSIX-specific.
Most tests mock the OS keyring layer beneath the actual custody adapters. The
Windows-only integration test instead creates isolated test-owned namespaces in
the real Windows Credential Manager, activates separate role keys, and signs and
finalizes through two spawned processes. It checks public/private binding,
cross-role rejection and lifecycle revocation, then deletes its own credentials.
Only its public registry activation is simulated; genuine PostgreSQL binding is
covered separately. Tests do not provision installed production credentials or
perform legal enrollment.

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
signing-operation identity binding the exact request reference/digest, `rpa_`,
key, principal, role/domain/profile and custody lifecycle identity. It commits
**before** the private operation. Signature verification, checkpoint append,
transition and current-state CAS commit together. Finalization appends the existing
AttemptIdentity and its fenced projection without changing the reserved `rpa_`.

The frozen parent's reservation crash recovery permits re-obtaining missing
signatures for the same `rpa_` under lifecycle rules. A permanently spent invocation
latch contradicted that rule, even before private invocation. Local Ed25519 is
deterministic for the exact same domain-separated bytes and private key, so an
identical retained intent without a checkpoint can resume after requalifying
unchanged authorization and ACTIVE original credentials. An unequal intent fails
closed. No intent is deleted or interpreted as proof that signing never occurred.

A durable requester checkpoint is verified and reused; requester custody is never
invoked again while completing claimant authorization. The same rule applies to
a claimant checkpoint. Finalized attempt retries reuse the retained identity and
digest with zero signer calls. The parent's post-send/BOUND prohibition on
re-signing remains phase-specific. Independent stores are not a distributed
transaction.

Changed credentials cannot be substituted inside an old `rpa_`. The explicit
`supersede_root_proof_issuance_attempt_pre_send(binding, fresh_authorization)` API
qualifies fresh genuine provider authorization and requires the exact existing
InitialBinding/operation/entitlement, a changed eligible credential tuple and the
current fence. Executable proofs require installed grants to be exactly local
RESERVE/SIGN/FINALIZE, no immutable attempt, no finalized immutable attempt and
no issuer recovery/BOUND record. Finalized or ambiguous history fails closed.
The conservative eligible states are RESERVED_AWAITING_SIGNATURES and
REQUEST_SIGNED_BY_REQUESTER and CLAIMANT_AUTHORIZED, including intent-only
interruptions and both checkpoints durable before finalization. Two checkpoints
alone grant no send authority: the authority-owned immutable finalization is
required before an externally-sendable attempt can exist.

One transaction appends a typed `LocalPreSendUnsendableV1` proof to the existing
immutable replacement relation, records SUPERSEDED_PRE_SEND_PROVEN_UNSENDABLE on
the old history and points to an authority-minted new `rpa_`. Original authorization,
request bytes, intents and checkpoints remain unchanged. The old digest can remain
NOT_YET_DEFINED. Exact decision retries converge; rollback leaves no hidden new
reservation. Private replacement custody must separately be provisioned and
qualified before any signature. This is distinct from authoritative UNBOUND
post-send reconciliation. Schema v6 and v5→v6 migration remain unchanged.

A protected-host advisory lock serializes the complete signing sequence across
separate SQLite commits and processes. SQLite writer transactions, immutable unique
rows, exact byte/signature comparisons and fenced CAS remain the authority.
Two exact concurrent callers converge; unequal identity is corruption. The host
administrator, native keyring and working local filesystem locks remain trusted.
Independent live PostgreSQL checks and SQLite commits are not a distributed
transaction; final capability access requalifies current operation evidence.

Tests cover every write/CAS cut, abrupt `os._exit` without Python transaction
cleanup, restart, lost response, exact pre-finalization recovery, checkpoint call counts,
concurrent exact callers, explicit safe supersession,
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
