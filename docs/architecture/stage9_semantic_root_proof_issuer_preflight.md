# Stage 9 semantic RootProofIssuer local preflight

`bot_core.root_proof_issuer_runtime.preflight_root_proof_issuance_attempt` is the
first production-code slice of semantic request processing. It accepts only a
genuine `VerifiedSignedImmutableRootProofIssuanceAttempt` from the installed local
CHA boundary. It returns an immutable reporting DTO with disposition
`VERIFIED_NOT_AUTHORIZED_TO_ISSUE`, request/attempt identity and observed public
authority revisions. A report is a historical observation, not an authorization;
neither constructing nor retaining it grants any operation.

The runtime resolves the genuine retained INITIAL_BINDING through the existing
CHA authority and rechecks the exact current SQLite attempt, fence, immutable
request and local custody. It independently compares the complete canonical
request and immutable attempt identity, then verifies both frozen,
domain-separated Ed25519 signatures against current PostgreSQL requester and
claimant public keys. It requires exact ACTIVE credentials, separated roles and
key material, and the exact ACTIVE/UNBOUND entitlement, generation, product,
environment, trust domain and provisioning provenance. A second full upstream
read after signature verification rejects changed or unavailable evidence.

All reads use existing authorities. There is no caller-selected provider, public
key, namespace, DSN, trust domain or transport parameter. The existing signed
capability's custody qualification remains mandatory; this local slice does not
establish an independent remote issuer endpoint. Independent reads across stores
are not a distributed transaction. Later mutation must perform fresh
authorization and the existing authority-owned CAS.

The inherited validator opens SQLite through its normal local-store adapter,
including WAL, permission and schema maintenance and locking. The read-only
scope concerns issuance operations and logical attempt/authority state;
filesystem bytes and database connection mode are outside that guarantee.

The frozen requirements come from:

- `cryptohunter_product_architecture/m05_account_genesis_independent_root_proof_issuer_contract.json`
  (requester, claimant, genuine INITIAL_BINDING and complete request semantics);
- `cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.json`
  (separate authorities, PostgreSQL serialization and durable recovery);
- `cryptohunter_product_architecture/stage9_root_proof_signed_immutable_attempt_contract.json`
  (immutable current attempts, profiles and the local no-send boundary).

These frozen artifacts retain their historical status snapshots and byte
identities. Current implementation and deployment status is authoritative only in
`deployment/stage9_current_status.json`, which this change leaves unchanged.

The runtime performs no signing, entitlement BIND, history append, checkpoint
advance, external send, proof admission, account commit or legal enrollment. It
does not install an additional issuance dispatch route. Successful preflight
leaves the request `SIGNED_IMMUTABLE_DURABLE_NOT_SENT` and entitlement UNBOUND.
Repeated evaluation returns the same report while all evidence remains exact.
Invalidated credentials, binding, attempt or authority configuration fail closed.

Full Semantic RootProofIssuer remains `NOT_IMPLEMENTED / BLOCKED`. Remaining
work includes genuine production pre-account credential provisioning, durable
authenticated issuer history/checkpoint and reconciliation providers, qualified
deployment trust and issuer-signing composition, authenticated transport and a
service boundary for genuine INITIAL_BINDING resolution, and crash-safe
CAS/sign/history/checkpoint orchestration with exact replay recovery. Root Proof
remains `NOT_ISSUED` and legal production enrollment `NOT_PERFORMED`.

Integration tests use isolated actual PostgreSQL authorities, generated test keys
and the existing native-custody test backend. They establish executable behavior
and security regressions only; they do not establish production provisioning or
live availability. The normal serial PostgreSQL CI gate includes the runtime
tests and coverage. No installer or physical-hardware qualification is needed.
