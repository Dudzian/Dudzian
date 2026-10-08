# Stage 9 pre-account public credential registries

The subordinate contract is
`cryptohunter_product_architecture/stage9_root_proof_preaccount_credentials_contract.json`,
with its exact bytes and protected historical inputs retained by
`stage9_root_proof_preaccount_credentials_freeze.json`. Historical issuer,
implementation-readiness, production-selection and #3086 reservation artifacts
keep their original bytes and historical availability statements.

The selected `PRODUCTION_LOCAL` substrate implements requester and claimant
public credential authority in separate PostgreSQL schemas, authority APIs and
runtime/admin/owner roles. Protocol environment remains `PRODUCTION`. Each row
and lookup also binds the exact deployment trust domain. Requester ownership is
`CryptoHunterAccountAuthority` with role
`ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUESTER_V1`; claimant ownership is the
pre-account `provisioning_principal_id` provisioned by deployment/security
administration. The candidate account supplies neither authority.
Live composition rejects a claimant provisioning principal that exactly equals
the current initial binding's candidate account ID or logical operation ID.
This equality check does not establish the principal's temporal pre-account
history.

Both verifier-side registry ports add
`public_key(credential_identity: str) -> bytes`. The returned material is exactly
32 raw Ed25519 public-key bytes from retained authority history. The provider
recomputes `public_key_material_identity` from those bytes and rejects a stored
fingerprint mismatch. PEM/DER decoding and caller-owned public-key DTOs do not
replace this lookup. A registry supplies no private-key custody or signing
capability.

Append-only credential and lifecycle records retain old keys and all `ACTIVE`,
`VERIFY_ONLY` and `REVOKED` generations. Key rotation versions and lifecycle
history generations have separate meanings. Only an exact current `ACTIVE`
credential can authorize a new flow; historical lookup does not restore that
eligibility. Lifecycle-only transitions retain the key version. Rotation,
history append and current projection changes commit in PostgreSQL transactions,
with the expected authoritative revision fencing stale concurrent administration.
`REVOKED` is terminal. Database roles and reviewed APIs establish history
provenance; this layer does not supply a signed issuer checkpoint or independent
full-host rollback protection.

Lifecycle administration can target an exact retained credential ID. Revoking
a retired `VERIFY_ONLY` key preserves its replacement's current `ACTIVE`
pointer and retains the old public material. The revision fence refers to the
target credential's latest lifecycle record; the current projection can precede
the principal's history head after an old key is revoked.

Provisioning uses distinct registry admin connections and may create public
credentials, rotate keys and transition lifecycle. Runtime connections can only
resolve authoritative current/history/key/identity evidence. Qualification
checks actual schema identity/version, durability, role and ACL evidence.
Requester and claimant raw material must differ even when labels, credential
IDs and key IDs differ. The paired installer gives each schema owner only the
peer immutable public credential-table read needed for that constraint; it gives
no peer runtime/admin authority. Future complete composition must also check
issuer, freshness, Catalog and storage credential roles.

The runtime DDL prohibition covers persistent database schemas and credential
authority objects. Existing database `PUBLIC TEMP` grants remain unchanged and
may permit isolated temporary workspace. Such workspace grants no credential
authority: reviewed procedures pin `pg_catalog` as their search path and use
fully qualified authority tables, preventing temporary objects from shadowing
credential state. The forbidden `PUBLIC` privileges refer to the reviewed
authority schemas, tables and functions.

`PostgreSQLRootProofIssuanceAuthority` composes exactly these genuine providers
with the existing `PostgreSQLEntitlementRegistryProvider`. The internal installed
composition seam reads the six deployment configuration names frozen in the
child contract. Its entitlement lookup handle selects existing authority
evidence; claimant principal/key/version come from the authoritative entitlement
provenance, and requester is fixed. Configuration cannot assert capabilities or
manufacture credentials. Missing configuration, unavailable providers, wrong
roles/namespaces or changed live evidence fail closed before reservation writes.
The installed seam checks the exact frozen requester/claimant schema and role
triples against qualified live PostgreSQL metadata. The low-level installer and
aggregate also support explicit safe identifiers for isolated authority tests.
The public Stage 9 functions accept only the verified upstream capability and,
where required, verified authorization.

The completed boundary ends at live requalified
`VerifiedRootProofIssuanceAuthorization` and the existing durable
`RESERVED_AWAITING_SIGNATURES` reservation. Exact evidence and retry/restart
retain the same `rpa_`; changed credential revision, lifecycle, key identity,
material or provider evidence invalidates the capability. Entitlements remain
`ACTIVE + UNBOUND`.

`deployment/stage9_current_status.json` remains the global current-status
source. Public credential providers and composition are `IMPLEMENTED`.
Installed production pre-account credential provisioning is `NOT_PROVISIONED`,
and live authorization availability is `BLOCKED_UNTIL_PROVISIONING`. Real
PostgreSQL tests use temporary fixtures and do not provision production
credentials or establish legal enrollment. Signed immutable attempts remain
`NOT_STARTED`; the semantic issuer remains `NOT_IMPLEMENTED / BLOCKED`, root
proof remains `NOT_ISSUED`, and `PREPARED` remains `NOT_STARTED`. Windows stays
`10/15 DONE`, and production provisioning readiness stays false.
