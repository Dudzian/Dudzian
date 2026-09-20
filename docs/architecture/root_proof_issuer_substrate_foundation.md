# M0.5 Root-Proof Issuer — implementation foundation

This note records implementation integration points; the frozen architecture
contracts remain the design source of truth.

## Integration anchors discovered

* Dependencies are declared by `pyproject.toml`/`requirements.txt`; YAML/TOML
  is loaded through `bot_core.config`, while services use constructor/factory
  injection.
* The application composition root is `bot_core.runtime.core_host.CoreHost`,
  with frontend configuration assembled in `bot_core.runtime.frontend`. Its
  factory parameters are also the established test dependency-injection seam.
* Existing execution environments (`PAPER`, `TESTNET`, and `LIVE`) are trading
  scopes, not M0.5 security profiles, so they are not reused.
* Persistence uses the account-scoped SQLite `StateStore`, direct `sqlite3`
  adapters, and the async SQLAlchemy/aiosqlite `DatabaseManager`; none is
  promoted to issuer registry authority or PostgreSQL CAS.
* Schema evolution uses the persistence migration registry/engine and the
  database manager's migration phase. M0.5 adds no authority schema yet.
* Existing keychain `SecretStore`, fingerprint crypto, bootstrap/security
  authorities, Catalog authority, storage keys, and freshness credentials are
  integration references only and are explicitly not issuer/requester keys.
* Deployment is described by `deploy/Dockerfile` and
  `deploy/docker-compose.yml`; no server-ready vendor is configured.
* Future M0.5 provider assembly belongs at the scoped startup/composition seam
  before exposing that authority. The CoreHost process mutex must remain only a
  topology gate, never the entitlement registry's durable CAS.

## Implemented foundation

`bot_core.root_proof_issuer_substrate` defines the four exact profiles,
immutable profile/provider/credential identities, typed signing and checkpoint
capability evidence, eleven separate provider/policy contracts, and a central
fail-closed qualification gate. Configuration selects adapter implementations;
it cannot assert their security capabilities.

The gate validates that every declared provider role implements its exact
runtime port and supplies the role-specific minimum evidence. Credential-bearing
roles must expose current composition identities bound to the provider namespace
and expected semantic role. Credential IDs and key handles are compared as
namespaced identities, not as accidentally global strings.

For both signing roles, the single declared composition identity must exactly
equal the provider's active identity. Its key-material identity is a
domain-separated SHA-256 fingerprint of canonical public-key bytes returned by
the provider; private-key export is neither used nor required. Material identity
is global for alias detection, so copying or re-importing one signing key under
different provider namespaces or handles remains detectable.

Root-proof signing and history-attestation signing share only a read-only
signing-identity port. Their authority operations are disjoint:
`sign_root_proof` and `sign_history_head`; neither signing authority protocol is
a subtype of the other. The gate also rejects a concrete adapter that exposes
both authority operations, while allowing physical colocation behind separate
authority-specific wrappers, credentials, and keys.

Each qualification pass captures one immutable provider-evidence snapshot.
Provider identity, capabilities, and composition credential identities are read
once; a valid signing identity and its public key are also read once. All later
profile, credential, and alias checks consume only those snapshots, preventing
an adapter rotation from changing evidence between validation and alias checks.
This provides consistency for composition qualification only. Later runtime key
rotation still requires lifecycle, freshness, and requalification semantics and
is intentionally outside this foundation.

All security-boundary value objects enforce their annotations at runtime:
profiles and roles require the exact enum types, identity text requires exact
non-empty strings, and every capability flag requires the exact `bool` type.
Nested signing/checkpoint evidence also requires its exact frozen capability
type. Malformed direct construction is rejected rather than coerced, while
malformed provider or composition evidence is converted by the gate into a
structured fail-closed result.

Constructors, the qualification policy, and snapshot capture use the same small
set of deep validators. Exact outer classes alone are insufficient: nested
signing/checkpoint booleans and nested security identities are revalidated from
their current values, including objects fabricated outside normal constructors.
Credential snapshot evidence must be an exact built-in tuple containing exact
credential identity objects; tuple subclasses and other stateful containers are
rejected before active-key or alias validation.

`RootProofIssuerSubstrateConfig` follows the same no-coercion boundary: its
security identity, tuple container, pair entries, provider roles, and adapter
selection strings require exact runtime types. Configuration entries may remain
incomplete until composition, but every supplied entry is strictly typed.

Runtime port validation guarantees protocol-shaped callable presence, not exact
Python call-signature compatibility. Static typing and the trusted adapter
registration boundary remain responsible for signature compatibility; the gate
does not claim to be a general runtime introspection framework.

Requester and claimant registry ports do not yet expose canonical public-key
identity. Their credential IDs and handles therefore remain namespaced, while
cross-provider physical-key equality for those registry credentials is deferred
until concrete credential models expose a trustworthy canonical public identity.

Provider-originated capabilities are typed **adapter evidence**, not an
independent cryptographic attestation protocol. In particular, an arbitrary
untrusted Python object does not establish real hardware trust merely by
returning `True`. A production composition root must instantiate providers from
trusted application adapter registrations. Provider-specific trusted
attestation and a hardened provider-registration boundary remain future
SERVER_READY work; this foundation intentionally does not introduce plugin PKI.

The stateless composition gate calls one module-level frozen qualification
implementation directly and has no instance dictionary or policy attribute;
callers cannot inject, replace, or mutate its policy. The public policy helper is
a slot-only stateless wrapper over that same function and binds its target
profile to the provider identity before applying profile requirements.

Signing adapters must return the canonical Ed25519 public-key representation:
exactly 32 raw public-key bytes, never PEM, DER/SPKI, text encodings, or a
provider wrapper. The shared domain-separated material fingerprint hashes only
those public bytes and never requests private material. This makes the same key
identity comparable across root-proof and history-attestation adapters.

## Still not implemented

There is no semantic RootProofIssuer, NEW_BIND flow, issuance state machine,
PostgreSQL entitlement registry, production local key custody, durable local
checkpoint, HSM/KMS adapter, remote checkpoint, final CHA coordinator, or
LOCAL-to-SERVER_READY migration workflow. Composition into `CoreHost` is a
future scoped startup step after concrete providers exist.
