# M0.5 authenticated issuer history and local checkpoint — executable contract

## Discovery and reuse

The current tree already defines `IssuerAuthenticatedHistory`,
`CheckpointAuthorityProvider`, `RootProofReconciliationEvidenceSource`, and the
separate `HistoryAttestationSigningProvider` port in the issuer substrate.  It
also freezes `RootProofIssuanceReconciliationEvidenceV1`, including
`product_scope`, and provides local signing custody with `ACTIVE`, `VERIFY_ONLY`
and `REVOKED` lifecycle states.  This stage reuses those names and the accepted
credential identities.  The unrelated E2E `RuntimeCheckpoint` is not an issuer
security authority.  No contradiction with a frozen contract was found.

`bot_core.authenticated_issuer_history` freezes the executable value model and
reference CAS semantics.  `ReferenceAuthenticatedHistory` is explicitly not a
production persistence implementation.  `LocalCheckpointProvider` is likewise
an executable local CAS oracle, not yet a durable provider qualification; the
semantic RootProofIssuer and runtime composition remain absent.

## Authenticated history

The canonical stream identity is the exact tuple `stream_id`,
`issuer_authority_identity`, `security_profile`, `environment`, `trust_domain`,
`product_scope`, and positive `security_epoch`.  The event identity reuses the
frozen `logical_operation_id`, `issuance_attempt_id`, and `root_proof_id` names.
Genesis is sequence 1 with literal `NO_PREDECESSOR`; every successor is exactly
the prior sequence plus one and names the prior authenticated record digest.

Canonical JSON accepts only exact null/string/bool/integer/list/object values,
recursively freezes containers, emits UTF-8 with sorted keys and unambiguous
separators, and is domain-separated for event, record, and head.  Floats,
non-string keys, mutable nested containers, bool-as-int, missing required
fields, and non-canonical attestation bytes fail closed.  The record digest
binds the stream, sequence, predecessor, exact event identity, payload, and
event digest.

The V2 immutable attested head binds the exact stream, sequence, record digest,
and complete `CredentialRoleIdentity`: semantic role, credential identity,
provider namespace, key handle/version, custody lifecycle namespace, and key
material identity.  It contains
canonical attestation bytes and signature, never a provider object.  New heads
are obtained only through the history signing port; the local custody provider
already permits signing only with `ACTIVE`.  `VERIFY_ONLY` remains suitable for
verification with its retained public key, while `REVOKED` cannot create a new
head.  Verification never accepts a caller-selected raw key: it resolves the
exact credential from `HistoryAttestationSigningProvider.credential_identities`,
requires the history provider/semantic roles and provider namespace, compares
the declared key version, obtains `public_key(credential_id)`, and checks its
canonical raw-key fingerprint against `key_material_identity` before Ed25519
verification.  Root-proof role use, credential/version mismatch, key mismatch,
and cross-role aliasing fail closed (the latter is also enforced by substrate
qualification).

The frozen production-substrate selection contract says a REVOKED-key signature
alone is insufficient: historical acceptance requires a currently trusted root,
the exact retained credential/generation/record, independently authenticated
history, and no compromised-key self-corroboration.  The executable reference
model uses the already-selected checkpoint authority, rather than inventing a
new key or role.  A REVOKED signature is checked cryptographically first, but is
accepted only when the trusted composition supplies the exact checkpoint
authority which retained a matching `HistoricalHeadAcceptanceEvidence` while
the credential was ACTIVE.  Missing evidence or the wrong authority is
`HistoricalRevokedSignatureVerificationUnavailable`.  ACTIVE and VERIFY_ONLY
remain valid verification states without historical acceptance evidence.

The accepted local signing custody lifecycle starts at generation 1/ACTIVE.
Each legal one-way transition (`ACTIVE -> VERIFY_ONLY`, `ACTIVE -> REVOKED`, or
`VERIFY_ONLY -> REVOKED`) atomically increments the persisted metadata
generation, while credential ID, key handle/version and public key remain
unchanged.  REVOKED is terminal.  Thus direct revocation has generation 2 and
planned retirement followed by revocation has generation 3.  Custody reloads
the current persisted generation after restart, but it retains no transition
journal and cannot independently reconstruct an earlier generation.  The
checkpoint-owned acceptance record therefore captures the ACTIVE generation at
the acceptance linearization point; it does not infer history from wall time or
from the later current custody value.

The existing composition gate returns qualification failures but does not issue
a transferable provider-qualification capability or public snapshot.  Therefore
the verification authority passed here has the explicit precondition that it
originates from the trusted composition root.  Satisfying the structural Python
protocol is not presented as proof of trusted provenance.

The reviewed domains are
`CryptoHunter/M0.5/IssuerAuthenticatedHistoryHead/v2` and
`CryptoHunter/M0.5/HistoricalHeadAcceptanceEvidence/v2`.  A V1 head or
acceptance omitted the complete namespaced signer identity and is deliberately
not promoted or interpreted as V2 evidence.

Exact event retry returns the original record deterministically, including
after a lost response.  The same event identity with a different payload is a
conflict.  Expected-predecessor CAS means append/append and retry/append races
cannot create two legal successors.  Gap, duplicate sequence with another
digest, fork, rewind, predecessor substitution, or cross-stream/trust/product
splice is corruption, never absence.  A production history adapter must use a
durable database CAS; an in-process lock alone cannot qualify.

The executable reference oracle linearizes its complete append authority
decision with a process-local reentrant lock: event lookup, predecessor read and
comparison, successor construction, record publication, and event-index
publication are one critical section.  Verification holds the same lock across
complete-chain verification and current-head capture.  This only makes the
in-memory semantic oracle coherent; it provides no durability, restart safety,
cross-process exclusion, transactional database authority, or crash semantics,
and therefore does not qualify PRODUCTION_LOCAL history persistence.
The reference authority snapshots its stream and each admitted event identity,
stores authority-owned records, and returns fresh detached `HistoryRecord`
values from append, replay, and `current_record`.  Public-object mutation cannot
alter the retained predecessor or event index.  Append also validates the
retained chain and event index before using a predecessor, so direct test-only
internal corruption cannot authorize a successor.

## Checkpoint and split brain

`LocalCheckpoint` binds checkpoint ID, full stream identity (and therefore
environment, trust domain and epoch), history sequence and authenticated head
digest, complete exact signing `CredentialRoleIdentity`, and checkpoint revision.  Advance uses
expected-revision CAS.  Exact same-head replay is idempotent; lower sequence is
rollback, and same sequence with a different digest or signer is split-brain.
A higher sequence is acceptable only when `LocalCheckpointProvider.advance`
itself invokes the exact history authority to verify the complete retained
chain, exact current-head relation, trusted credential, lifecycle and
attestation before CAS.  `VerifiedHistoryHead` is only an immutable reporting
snapshot and is never accepted as an authorization capability; Python object
privacy, seals and tokens are not authority boundaries.  Persistent
implementations must make equivalent validation and CAS one reviewed authority
operation, or bind them with a durable non-forgeable authority record.

Checkpoint verification completes before the checkpoint lock is acquired, so
history and checkpoint locks are never nested.  The verified head is copied to
a fresh exact snapshot; the checkpoint lock then linearizes current revision
read, expected-revision comparison, rewind/split-brain decision, and successor
publication.  A concurrent later history append can only leave that safely
committed checkpoint stale.  It cannot create two checkpoint commits at one
expected revision.
The provider snapshots its stream, retains an authority-owned checkpoint, and
returns a fresh detached `LocalCheckpoint` from both `advance` and
`current_checkpoint`.  Reconciliation accepts the exact checkpoint provider,
not a caller-created checkpoint value, and captures one detached authoritative
checkpoint snapshot before comparison.  With verified history and no committed
checkpoint the outcome is `STALE`; a fabricated matching value cannot produce
`EXACT_COMMITTED`.

Every non-replay checkpoint advance also atomically appends an authority-owned
V2 `HistoricalHeadAcceptanceEvidence`.  It binds the unguessable reference
checkpoint-authority instance identity, checkpoint revision, complete stream
(including product, trust domain and security epoch), exact sequence and head
digest, complete exact signing `CredentialRoleIdentity`, ACTIVE lifecycle generation and
state, and the preceding acceptance digest.  Acceptance records form a retained
digest-linked revision chain and are never replaced when the current checkpoint
advances, so a later checkpoint signed by credential B does not erase the exact
earlier acceptance of credential A.  Evidence is exact-head only: a record for
N cannot establish N-1 or N+1.  No ancestry shortcut is inferred.

The public evidence dataclass is only a detached report.  Historical
verification queries the exact provider instance and revalidates its retained
chain; it never consumes a caller-provided evidence value, seal or token.  The
provider's lock publishes checkpoint and evidence together under the same CAS,
so two commits at one expected revision have one winner.  History verification
and lifecycle sampling happen before the checkpoint lock; no history,
checkpoint and custody locks are nested.  The final observed ACTIVE state and
generation define the acceptance-before-revocation linearization point.  A
REVOKED observation cannot append evidence.

One central `_verify_authority_state_locked` definition protects every
checkpoint/evidence authority read and every decision inside `advance`.  It
validates exact evidence types, stream and authority identity, contiguous
checkpoint revisions, predecessor and canonical evidence digests, ACTIVE state,
positive lifecycle generation, and strictly increasing (not necessarily
adjacent) accepted history sequences.  It additionally requires `_current` to
be absent exactly when the evidence chain is empty, evidence cardinality to
equal the current checkpoint revision, and the last evidence to bind every
current checkpoint field exactly.  `advance` invokes this validator as its first
operation under the checkpoint lock, before revision/replay decisions or use of
the predecessor digest.  Consequently corrupt evidence, missing or trailing
evidence, and checkpoint-only rollback cannot authorize either a successor or
an exact-replay result.

`current_checkpoint`, `historical_acceptance`, and
`verify_historical_acceptance` use the same validator.  Reconciliation maps a
detected internal checkpoint/evidence inconsistency to `CORRUPT`; inability of
the signing/key/lifecycle authority remains `UNAVAILABLE`.  The SHA-256 chain is
an integrity mechanism inside the selected authority, not a trust root and not
a replacement for trusted composition provenance.

Checkpoint behind a valid history is `STALE` and may be advanced from verified
history.  Checkpoint ahead of history is `CORRUPT`, as is a checkpoint whose
head is absent.  Same-sequence disagreement is `CONFLICT`.  Two checkpoint
values, two heads at one sequence, stale replicas, different signers, or
conflicting retained prefixes are never arbitrated by choosing the numerically
larger value.

The local checkpoint can detect accidental database rewind, an uncoordinated
rollback of one store, and some local tamper/inconsistency.  It cannot protect
against privileged full-host snapshot restore, coordinated database and
checkpoint rollback, a host administrator changing both domains, or host-admin
key extraction.  A valid older history prefix is not intrinsically invalid;
it is detectable only against a checkpoint that was not itself rolled back.
SERVER_READY requires an independent checkpoint/rollback and administration
domain plus stronger key custody.

## Reconciliation and crash protocol

Outcomes are exact committed, not found, stale, conflict, corrupt, unavailable,
and indeterminate.  `CORRUPT`, `UNAVAILABLE`, and `INDETERMINATE` never become
`NOT_FOUND`.  Reconciliation evidence retains every frozen V1 binding:
environment, trust domain, product scope, issuer/registry identity, entitlement
and generation, logical operation, account, request fingerprint, old issuance
attempt, initial binding, authority state revision, authenticated history
reference/digest, verification profile, and signing credential/version where
the new head binds it.  Evidence is stale after history or registry advances,
attempt replacement/binding, or a credential epoch/version transition that
requires fresh evidence.  Cross-product, cross-trust, cross-environment,
cross-operation, cross-attempt, cross-account, or cross-proof replay is a
conflict.

The preferred non-distributed protocol is:

1. reserve/resolve the CHA attempt, then reserve/bind entitlement by its durable
   registry CAS;
2. generate and sign one proof for that exact binding (future issuer step);
3. append the exact issuance event to authenticated history by predecessor CAS;
4. advance the local checkpoint by checkpoint-revision CAS;
5. mark the CHA attempt `BOUND` with the exact proof/history evidence.

There is no distributed ACID across registry, attempt store, history, and
checkpoint.  Before registry commit, retry the reservation.  After registry
commit but before history, authoritative registry plus the reserved attempt is
the recovery source; absence or ambiguity stops with `INDETERMINATE`.  After a
history append whose response is lost, exact event lookup/replay returns the
committed record and never appends twice.  After head attestation response loss,
the authenticated history head and retained signer identity are the source.
After history but before checkpoint, a verified one-head-behind checkpoint is
advanced.  Checkpoint ahead/absent-history or any divergent digest is
`CORRUPT`/STOP.  After checkpoint but before CHA `BOUND`, exact registry,
history, checkpoint, and attempt evidence completes the local transition; any
disagreement is `CONFLICT`, `CORRUPT`, or `INDETERMINATE`, never blind issuance.

### Documented versus executable matrix

| Semantic | Executable in this stage | Ownership/status |
|---|---:|---|
| stale after authenticated-history advance | yes | history authority verifies chain/head/signer inside reconciliation before returning `STALE` |
| stale after entitlement-registry revision advance | no | later semantic RootProofIssuer/reconciliation-evidence composition |
| attempt replacement or binding | no | CHA plus later semantic RootProofIssuer composition |
| signing credential epoch/version transition | partial | head/checkpoint version is exact; freshness across transitions needs lifecycle evidence |
| cross-product/trust/environment replay | yes for history/checkpoint; no for issuance evidence | full stream identity is executable here; issuance evidence remains later-stage |
| cross-operation/attempt/account/proof replay | partial | operation/attempt/proof are history event identity; account and reconciliation authority remain later-stage |
| `UNAVAILABLE` | derived for missing signer/key/lifecycle evidence | broader provider-composition mapping remains later-stage |
| `INDETERMINATE` | representable, not derived | later crash/reconciliation state machine |

Thus the prose describing registry freshness, attempt replacement/binding,
cross-account reconciliation evidence, and provider-unavailability handling is
a requirement for the later semantic RootProofIssuer stage, not a claim that
those paths are already executable here.

Malformed/broken history, a current head that is not its exact current record,
and an
invalid attestation classify as `CORRUPT`.  Inability of the trusted signing or
lifecycle authority to supply required evidence classifies as `UNAVAILABLE`;
this is not asserted to prove corruption.  A historical REVOKED signature
without the exact retained checkpoint-authority evidence required by the frozen
substrate contract raises the explicit fail-closed unavailable error.  With
evidence, `verify_attested_historical_head` validates the complete retained
chain, exact record at the stated sequence, canonical signature and key
material, current terminal lifecycle state/generation, and exact acceptance.

The reference oracle remains in-memory.  A future PRODUCTION_LOCAL adapter must
durably and crash-atomically persist the checkpoint and complete acceptance
chain, preserve stable provider identity/capability binding across restart, and
return `UNAVAILABLE` if that durable evidence cannot be recovered.  Selecting
the canonical injected instance is a trusted-composition-root responsibility;
structural protocol compatibility is not provenance.  Same-host persistence
does not detect a coordinated privileged rollback of history, checkpoint /
evidence, and lifecycle state.  SERVER_READY still requires an independent
rollback and administration domain.

## Qualification and privileged mutation boundary

PostgreSQL V5 qualification is startup-time qualification, **not** continuous
catalog re-attestation.  Authenticated history can expose resulting payload,
binding, gap, digest, or reconciliation inconsistencies on a later operation,
but it does not prove that a privileged schema owner did not mutate live SQL or
ACLs between qualification and use.  Privileged schema-owner mutation,
coordinated full-host rollback, modification of both local authority stores,
and host-admin key extraction are outside the PRODUCTION_LOCAL component
guarantee.  They require requalification or a SERVER_READY independent domain.

Development/test reference providers cannot qualify as PRODUCTION_LOCAL by a
configuration override; qualification remains capability-based.

## Current-tree flags and provenance

* `ENTITLEMENT_REGISTRY_EXECUTABLE_SEMANTIC_CONTRACT_FROZEN = true`
* `ACCOUNT_GENESIS_CHA_ATTEMPT_STORE_PRODUCTION_LOCAL_IMPLEMENTED = true`
* `ROOT_PROOF_ISSUER_PRODUCTION_LOCAL_ENTITLEMENT_REGISTRY_IMPLEMENTED = true`
* `AUTHENTICATED_ISSUER_HISTORY_LOCAL_CHECKPOINT_EXECUTABLE_SEMANTIC_CONTRACT_FROZEN = false`
* `ROOT_PROOF_ISSUER_IMPLEMENTED = false`
* `PRODUCTION_LOCAL_RUNTIME_AVAILABLE = false`
* classification: `UNKNOWN`
* finding scope: `CURRENT_TREE_ONLY`
* formal project advancement: `WITHHELD`
