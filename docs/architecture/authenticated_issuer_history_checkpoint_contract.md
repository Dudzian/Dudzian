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

The immutable attested head binds the exact stream, sequence, record digest,
role `HISTORY_ATTESTATION_SIGNING`, credential ID, and key version.  It contains
canonical attestation bytes and signature, never a provider object.  New heads
are obtained only through the history signing port; the local custody provider
already permits signing only with `ACTIVE`.  `VERIFY_ONLY` remains suitable for
verification with its retained public key, while `REVOKED` cannot create a new
head.  Root-proof role use, credential/version mismatch, key mismatch, and
cross-role aliasing fail closed (the latter is also enforced by substrate
qualification).

Exact event retry returns the original record deterministically, including
after a lost response.  The same event identity with a different payload is a
conflict.  Expected-predecessor CAS means append/append and retry/append races
cannot create two legal successors.  Gap, duplicate sequence with another
digest, fork, rewind, predecessor substitution, or cross-stream/trust/product
splice is corruption, never absence.  A production history adapter must use a
durable database CAS; an in-process lock alone cannot qualify.

## Checkpoint and split brain

`LocalCheckpoint` binds checkpoint ID, full stream identity (and therefore
environment, trust domain and epoch), history sequence and authenticated head
digest, signing credential ID/version, and checkpoint revision.  Advance uses
expected-revision CAS.  Exact same-head replay is idempotent; lower sequence is
rollback, and same sequence with a different digest or signer is split-brain.
A higher sequence is acceptable only after the caller has verified the complete
history relation and attestation; persistent implementations must make that
validation and the CAS one authority operation.

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
* `AUTHENTICATED_ISSUER_HISTORY_LOCAL_CHECKPOINT_EXECUTABLE_SEMANTIC_CONTRACT_FROZEN = true`
* `ROOT_PROOF_ISSUER_IMPLEMENTED = false`
* `PRODUCTION_LOCAL_RUNTIME_AVAILABLE = false`
* classification: `UNKNOWN`
* finding scope: `CURRENT_TREE_ONLY`
* formal project advancement: `WITHHELD`
