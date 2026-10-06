# Stage 9 production Windows pre-enrollment

`deployment/stage9_current_status.json` remains the canonical current-status
authority. Historical architecture/freeze snapshots remain unchanged. Stage 9
is `IN_PROGRESS`, production provisioning is false, and Windows production is
`NOT_READY`. This implementation does not perform legal enrollment or implement
Stage 10.

## Audit performed before implementation

The normative frozen architecture is
`docs/architecture/cryptohunter_product_architecture/stage9_external_provisioning_architecture_contract.json`.
Its SHA-256 still matches the freeze manifest:
`6538a75d8cf3a6e3cdab8f4a9022277a21d98e1a61923aa8d8eae9b80ae1d67d`.
The closure proposal and freeze JSON were inspected alongside the implementation.

`PreEnrollmentRequestV1` has exactly these canonical payload fields:

```text
schema_version
environment
product
product_profile
pdsa_trust_domain
pdsa_challenge_id
pdsa_challenge_digest_sha256
pdsa_challenge_nonce_digest_sha256
tpm_enrollment_request_digest_sha256
tpm_enrollment_challenge_digest_sha256
tpm_enrollment_response_digest_sha256
verified_tpm_exchange_reference
verified_tpm_public_projection_id
ek_public_digest
ak_public_digest
tpm_attestation_evidence_reference
pre_enrollment_public_key_algorithm_profile
pre_enrollment_public_key_canonical_bytes
pre_enrollment_public_key_fingerprint_sha256
release_policy_digest_sha256
release_policy_generation
request_nonce_hex
```

There are no timestamp fields in this request. The signed PDSA challenge and TPM
exchange have separate freshness requirements. Payload bytes use the repository
RFC 8785 JCS serializer. The request digest is SHA-256 of those exact bytes. The
request signature is ECDSA-P256-SHA256 over
`UTF8("CryptoHunter.Stage9.PreEnrollmentRequest.v1") || 0x00 || raw_digest32`,
encoded as strict minimal DER with low-S.

The key is provider-generated ECDSA NIST P-256 in Microsoft Platform Crypto
Provider, non-exportable. Public material is exactly SEC1 `04 || X32 || Y32`,
65 bytes represented by 130 lowercase hex characters. The fingerprint is
lowercase SHA-256 of those exact 65 bytes. CNG blobs, TPMT_PUBLIC, SPKI, PEM,
JSON and base64 are forbidden fingerprint inputs.

The later LPPI authority key is a distinct successor generated only after
package verification (`SUCCESSOR_KEY_AFTER_PACKAGE_VERIFICATION`). Its
TPMT_PUBLIC fingerprint contract is deliberately different. This block does
not implement `LPPIAuthorityKeyBindingV1` or successor signing.

Implementation conventions in underspecified portions are explicit: the
`schema_version` literal is the artifact name `PreEnrollmentRequestV1`, as with
existing Stage 9 schema-version artifacts; production product/profile/domain
follow existing `ProductionProvisioningPackageVerifier`; digest references use
existing SHA-256 identifiers; the evidence reference follows the existing
activation/projection envelope. These conventions do not redefine signed PDSA
challenge bytes or grant production authentication.

Reusable source inspected: `bot_core/licensing/{canonical,device_enrollment,
tpm_attestation,external_provisioning}.py`,
`deployment/windows_tpm_{activation_bridge,substrate_probe}.py`,
`deployment/windows_stage9_{production_trust,policy_material}.py`, and
`deployment/windows_installer/corehost_composition.py`. Canonical serialization,
public projection/retained exchange parsing, and verified Production Trust are
reused. The existing TBS activation bridge is a TEST_ONLY transient-key rehearsal;
its deterministic keys and cleanup lifecycle cannot supply production custody.
The substrate probe's software signing key cannot supply production custody.
Codebase Memory was unavailable in this session; findings were verified directly
against source.

## Subordinate frozen production contracts

The audit found three gaps: adoption of an existing unreserved CNG identity,
an unsigned historical `PDSAChallengeV1`, and the historical TPM verifier's
shared TEST_ONLY proof domains. The historical public projection also did not
establish manufacturer trust; a software-generated EK/AK could pass that verifier.
The expressly authorized subordinate contracts close these definitions without
modifying the parent freeze or promoting historical artifacts.

`stage9_pdsa_enrollment_challenge_contract.json` and its separate freeze define
`PDSAEnrollmentChallengeV1`: exact 13-field production payload, JCS envelope
`{payload, signatures}`, two or three distinct authorized Ed25519 signatures,
and the existing Production Trust threshold of two. The signing message is
`CryptoHunter.Stage9.PDSAEnrollmentChallenge.v1\0 || SHA256(JCS(payload))`.
The request references SHA-256 of the exact complete signed envelope and SHA-256
of the decoded issuer-generated 32-byte nonce. Issuance uses UTC seconds, exactly
604800 seconds of validity, and canonical issuer-generated `pchal_<UUIDv7>`.
Unknown fields and noncanonical bytes are rejected.

`PDSAChallengeStore` is durable **off-host issuer** state. SQLite
FULL durability retains exact `ISSUED` bytes before publication. Writer-locked
consumption binds exact authenticated request bytes/digest and a receipt.
`EXPIRED` and `CONSUMED` are terminal; an identical retry retrieves the prior
receipt, and another request returns `CHALLENGE_REPLAY_CONFLICT`. Production
authority additionally requires the factory provenance described below; the
SQLite schema, its records and an exact Python type do not establish it.

`stage9_production_tpm_custody_contract.json` and its separate freeze define
production-only exchange PoP and CertifyCreation domains, strict TPMT_PUBLIC,
TPMS_ATTEST and TPMT_SIGNATURE parsers, and
`ProductionPreEnrollmentKeyCustodyEvidenceV1`. Custody binds the exact SEC1 key,
subject Name/public area/creation hash, AK signature, target EK/AK/projection,
complete request digest, challenge and release. TPM attributes must establish
fixedTPM, fixedParent and sensitiveDataOrigin; software/imported/migratable
profiles are rejected. The request's evidence-reference field remains the
activation/projection reference, avoiding a request/evidence digest cycle.

The same contract defines `ProductionTPMEndorsementV1`: the genuine PDSA 2-of-3
quorum approves the exact hardware EK and certificate after independent issuer
manufacturer-chain validation and EK/public-key binding. This client verifies
the quorum approval and exact EK binding, then credential activation and AK
attestation. It does **not** independently validate an OEM chain from a signed
digest. A generic challenge, caller certificate flag, software fixture or
self-signed certificate cannot substitute for this endorsement. No actual
production endorsement or authority private key is generated by this PR.

All consequential operations require runtime-issued Production Trust and
reverify its package at current UTC; historical audit contexts cannot authorize
new requests. Verifier capabilities use issuance registries and immutable
snapshots; copied public fields do not transfer provenance.

## Composed local and issuer boundaries

`prepare_installed_production_request` selects installed paths and the fixed
CNG key, enforces local qualification (A), collects request-bound TPM creation
evidence (B), and signs exact request bytes (C). It returns public transport data,
not issuer authority. The client verifies challenge signatures/time; retained
issuer state is checked independently off host.

`authenticate_production_pre_enrollment` re-verifies raw challenge, endorsement,
retained production exchange, exact release/projection/key bindings, custody
attestation and request PoP. `accept_retained_production_request` commits its
sealed result through issuer-owned challenge consumption. It accepts no native
CNG handle or caller assertion of local qualification. The Windows producer
enforces provider/machine/export properties locally; the issuer's cryptographic
custody proof establishes target TPM origin and key identity. A provider-name
string alone does not prove Windows CNG properties remotely. Network service
deployment and complete provisioning-package publication remain separate work.

## Issuer retained-state provenance

Follow-up review confirmed that the initial implementation could issue a
`VerifiedIssuedPDSAChallenge` from a caller-selected database containing a valid
signed challenge and a reconstructed `ISSUED` row. The TPM pending store had the
same missing authority boundary. A copy made before canonical consumption could
therefore carry stale retained state into another store instance. The existing
subordinate contract already excludes caller-selected paths, database copies and
software fixtures as issuer authority; its freeze does not need to change.

`deployment.production_enrollment_issuer.open_installed_production_enrollment_issuer()`
is the production bootstrap. It accepts no arguments, environment overrides,
paths, connection, backend or caller-supplied factory. Its reviewed Linux
configuration is:

- OS service account `cryptohunter-pdsa-enrollment`, with a non-root UID and its
  exact primary GID as the running process identity.
- State directory `/var/lib/cryptohunter/pdsa-enrollment`, owned by that account
  with mode `0700`, under root-owned ancestors that deny group/world writes.
- Fixed files `pdsa-challenges.sqlite3` and `tpm-challenges.sqlite3`, owned by the
  service account with mode `0600` and one filesystem link.
- Current public Production Trust package at
  `/etc/cryptohunter/production-trust/<CEREMONY_ID>`, under protected root-owned
  directories, verified by the current-runtime loader.

Bootstrap fails on an unsupported platform, wrong principal, unsafe permissions,
symlinked/resolved-path mismatch or an unavailable current trust package. It does
not install the account, repair permissions, select another directory, or perform
the service deployment ceremony. The protected installed code and the OS service
boundary remain deployment prerequisites.

The factory issues an opaque `ProductionEnrollmentIssuerContext` and registers
its exact PDSA/TPM store pair in private weak registries. The service retains that
context for the bundle's lifetime; store references alone cannot prolong it.
Immutable snapshots bind
the configured trust object, store instances and directory/database source
identities: path, device, inode, owner/group and mode. Consequential store guards
recheck provenance and these identities; combining stores from distinct issuer
contexts is rejected, including two separately opened contexts for the same
canonical files. Constructors over arbitrary SQLite paths remain mechanics only
and cannot mint production challenge/exchange capabilities or consume an
authenticated request. Copying Python fields, rows or the entire database does
not register the copied instance as authority.

`close()` revokes the issuer context and both stores. After a legal service
restart, the same no-argument factory requalifies the protected canonical
configuration and reopens the durable files with fresh process-local provenance.
Previously consumed challenges and their exact receipts remain consumed;
process-local capabilities are neither serialized nor recovered from copied
tokens. A genuine reopen supports exact receipt retrieval and retained exchange
retry. Receipt retrieval grants no new authority or renewal of expired trust.
A stale `ISSUED` copy at another location cannot replay through production
verification. SQLite crash/concurrency consistency is preserved; independent
rollback protection against replacement of trusted canonical state is a separate
deployment/freshness responsibility.

Factory-provenance tests replace the private installed-configuration/current-trust
boundaries in explicit TEST_ONLY harnesses. That does not expose a production
registrar or persistence injection API. This PR supplies the factory and guards;
it does not deploy a live PDSA network service, perform legal enrollment, or change
Stage 9 readiness.

## Operator preparation and qualification

On a later physical Windows host with installed verified Production Trust:

```powershell
python -m deployment.windows_production_pre_enrollment --qualify-key
```

This operation creates or reuses persistent production key state. It selects
installed known-folder paths, the fixed provider and key name, and the canonical
Production Trust loader. It accepts no trust roots, PDSA keys, software private
key, backend, or arbitrary release-policy identity. Evidence contains public
material and qualification metadata only; it lists actual authentication
artifacts still required. Run it only as the separate operator live gate.

Local provider qualification, independent TPM attestation and cryptographic
proof of possession remain separate. Successful NCrypt property checks are a
local provider guarantee, not a verified manufacturer certificate, TPM quote or
legal enrollment. No CNG/TPM physical execution was performed in Codex:

```text
WINDOWS_NATIVE_PRODUCTION_KEY_QUALIFICATION=NOT_RUN
PHYSICAL_TPM_CREATION_CUSTODY_QUALIFICATION=NOT_RUN
LEGAL_PRODUCTION_ENROLLMENT=NOT_PERFORMED
```

## Key lifecycle and secret handling

The adapter opens the fixed machine key before considering creation. Only
`NTE_BAD_KEYSET` (key not found) permits first creation; other native errors stop.
A public `CREATION_RESERVED` record is durably written before the native effect.
Creation never requests overwrite. Finalized keys are reopened and qualified by
their actual provider handle, hardware implementation, ECDSA P-256 public blob,
machine scope, signing-only usage, zero export policy and the PCP export-allowed
BOOLEAN. That BOOLEAN establishes export permission, not imported-key origin;
independent CertifyCreation evidence establishes origin. Unsupported or
unavailable properties stop qualification.

An existing key with absent state is rejected with
`PREEXISTING_UNRESERVED_IDENTITY` before qualification, commitment or signing.
A committed public fingerprint must match
on every reopen; a disappeared committed key stops. An interrupted reservation
with an existing key can be reconciled to that exact qualified identity. A
reservation with no key stops for operator reconciliation instead of generating
another identity. Duplicate invocations serialize through the state lock. A
lost response after commitment returns the same key identity. No production key
is deleted, even on partial failure; only owned native handles and disposable
temporary state-write files are cleaned up.

Continuity depends on preserving the same protected machine-state directory.
The adapter rejects symlink, reparse and hard-linked state paths but does not
install Windows ACLs. The operator composition uses installer-owned State;
existing installation ACL qualification remains a prerequisite. These public
records provide local continuity, not authenticated TPM custody, rollback-proof
storage, or host-wide protection after administrative state removal.

The native boundary exposes public `ECCPUBLICBLOB` export only and converts it
to canonical SEC1. It has no private-export, key-import or deletion API.
`sign_request` requires the exact validated request, exact key identity and
verified Production Trust binding; it hashes the frozen signing bytes through
SHA-256 and uses NCrypt signing, normalizes low-S DER and verifies the result
locally. That signature establishes possession only. Evidence and state contain
public identifiers/digests and safe metadata, never private blobs, credentials or
TPM authorization values. CLI failures emit a fixed diagnostic without raw
exception data.

## Native custody bridge qualification

`windows_cng_custody_bridge.py` opens the already-retained machine AK only and
compares its exact TPM public area/Name to the target projection. It borrows the
PCP TBS context and transient TPM handles, performs ReadPublic/CertifyCreation,
and never closes/flushes borrowed resources or generates substitute keys.
Creation-hash inputs are only exact raw32 or TPM2B_DIGEST(size32); creation ticket
is only an exactly marshaled TPMT_TK_CREATION with creation tag, an approved
hierarchy and a 32/64-byte digest. Other encodings, unavailable properties,
password/policy requirements or TPM errors fail closed.

Microsoft headers expose creation-property names but do not establish property
availability or exact framing for reopened persistent ECDSA keys. The bridge's
supported input profile is explicit; only physical qualification can establish
its availability on the target Windows/PCP version. ABI/crypto simulations and
hosted Windows DLL smoke tests do not establish physical TPM custody.

## Remaining Stage 9 work

The old unavailable-definition blockers are resolved by the subordinate freezes
and executable verification. Earlier operational prerequisites remain: installed
current Production Trust, qualification of the canonical issuer account and
protected filesystem configuration, live issuer-service integration,
independently issued production EK approval, retained live challenge/exchange,
and native/physical custody qualification. These must pass before legal
production enrollment. LPPI successor-key and membership-signing closure,
including `LPPIAuthorityKeyBindingV1`, follow those prerequisites. The existing
`ProductionMembershipSignerUnavailable` and production handoff remain fail-closed.
Production freshness, secret resources, full handoff and legal enrollment are
still outstanding. Ceremony trust and provider qualification do not satisfy them.
