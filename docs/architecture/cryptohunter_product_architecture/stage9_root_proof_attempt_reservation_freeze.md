# Stage 9 exact root-proof attempt reservation

The subordinate [contract](stage9_root_proof_attempt_reservation_contract.json)
freezes the local boundary from a current `VerifiedAccountGenesisInitialBinding`
through provider-originated authorization to an authority-owned durable `rpa_`
in `RESERVED_AWAITING_SIGNATURES` and a verifier-issued reservation capability.
The [freeze artifact](stage9_root_proof_attempt_reservation_freeze.json) records
the contract hash and protected upstream and historical artifact hashes.

The local code boundary is implemented. The installed production resolver fails
closed before creating a reservation because genuine requester and claimant
production adapters are unavailable. Existing requester and claimant registry
ports define the required resolution surfaces; they do not supply authenticated
production identities. The existing PostgreSQL entitlement registry remains the
entitlement authority foundation. This change introduces neither a replacement
registry nor CHA entitlement provisioning or BIND authority.

The trusted aggregate adapter allowlist remains empty. A future genuine adapter
must perform live substrate, privilege, lifecycle and durability qualification,
then provide exact results through the existing entitlement, requester and
claimant registry ports. Verification checks active identities, an authoritative
active and unbound entitlement, exact claimant provenance, provider scope and
credential role separation. Test modules simulate this future provider output
without furnishing a production registration path.

`reservation_id = NONE` and recovery by `(pdsa_trust_domain,
logical_operation_id)` remain unchanged. The downstream `reservation_identity`
is `ibr_` plus a domain-separated SHA-256 of the closed JCS relation payload. It
describes the existing retained binding and includes the exact operation,
candidate account, canonical request fingerprint, environment, trust domain,
reservation relation and hashes of the retained binding. It is deterministic
and creates no authority identity or recovery root. The historical illustrative
TEST vector containing `res_` is retained byte-identically.

`initial_binding_reference` is `initial-binding-v1:` plus a separate
domain-separated reference digest. That digest binds the required root-proof
authenticity tuple, including the deterministic reservation identity and exact
binding bytes. Upstream `initial_binding_sha256` keeps its original record
integrity meaning. These hashes do not establish authentication, authority or
freshness; the verifier must requalify upstream provenance and durable state.

Semantic authorization retains `environment = PRODUCTION`; the provider and
store deployment/security profile remains `PRODUCTION_LOCAL`. These domains
are validated separately. A TEST provider cannot authorize PRODUCTION. Exact
live provider evidence, including entitlement and requester/claimant revisions,
is retained through `authorization_evidence_sha256` in the idempotency tuple.
Changing that evidence while retaining the same IDs and generation conflicts.

The existing SQLite CHA attempt store owns persistence. Schema version 5
explicitly rejects version 4; no migration is inferred. Reservation and current
fenced pointer commit atomically using `BEGIN IMMEDIATE`, WAL and FULL
synchronization. Exact retries, restart and lost responses resolve the retained
winner. Loading never mints. The reservation capability rechecks exact upstream,
provider evidence, authorization tuple, current `rpa_`, fence and state whenever
it is used. The child record name `RootProofIssuanceAttemptReservationV1` denotes
the frozen logical view over the existing `AttemptReservation` and reservation
row; it creates no second store or serializer.

The boundary stops before either signature, signed immutable finalization,
external issuer communication, entitlement BOUND, root proof, admission,
PREPARED, freshness CAS or final AccountGenesis COMMITTED. AccountGenesis
remains `INITIAL_BINDING_ONLY`; the candidate account is not genuine.
Membership, device, Secret Resource and Stage 10 remain outside this step.

The semantic RootProofIssuer remains `NOT_IMPLEMENTED / BLOCKED`. Historical
`PRODUCTION_LOCAL` profile values `implemented = false` and
`deployment_available_now = false` remain unchanged. Root proof is `NOT_ISSUED`,
admission and PREPARED are `NOT_STARTED`, and
`production_provisioning_ready = false`. Hosted tests do not establish physical
qualification or legal production enrollment. SQLite durability provides no
independent protection against a privileged coherent host rollback.
