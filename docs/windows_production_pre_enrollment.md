# Stage 9 production Windows pre-enrollment foundations

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

## Conflicting authentication portions stopped

The frozen request requires a signed issuer-generated PDSA challenge retained as
`ISSUED`, bound by ID, exact challenge digest and nonce digest. The closure
proposal's offline challenge round trip names `PDSAEnrollmentChallengeV1`, with
production/product/profile/policy bindings, signature/key identity and a seven-day
window. The existing `PDSAChallengeV1` in `external_provisioning.py` is unsigned
and lacks those bindings. The frozen snapshot does not specify the signed
challenge envelope, canonical field encodings or signature domain bytes. A
production verifier cannot safely guess them.

The frozen request also requires raw-byte reverification by
`ProductionTPMAttestationVerifier` and TPM creation/custody evidence binding the
exact pre-enrollment key to the target projection. The current verifier shares
`POP_DOMAIN` and `CERTIFY_QUALIFYING_DOMAIN` containing `TEST_ONLY` with the
rehearsal verifier. `TPMPublicProjectionV1` checks Names/digests but does not
authenticate production profile labels. The convenience
`VerifiedTPMEnrollmentExchangeV1` is explicitly not authority. Neither object nor
a locally qualified CNG key establishes the required attestation.

Per the task's instruction to stop conflicting portions, production request
authentication fails closed with:

```text
SIGNED_PRODUCTION_PDSA_CHALLENGE_CONTRACT_UNAVAILABLE
PRODUCTION_TPM_CREATION_CUSTODY_ATTESTATION_UNAVAILABLE
```

The canonical request model, binding comparisons and signature verifier are
executable transport-neutral primitives. Parsing or proving possession does not
authenticate a PDSA challenge or TPM custody. This PR is an implementation of
those foundations, not closure of every acceptance criterion for an authenticated
production pre-enrollment flow. No TEST_ONLY artifact is promoted to authority.

## Operator preparation and qualification

On a later physical Windows host with installed verified Production Trust:

```powershell
python -m deployment.windows_production_pre_enrollment --qualify-key
```

This operation creates or reuses persistent production key state. It selects
installed known-folder paths, the fixed provider and key name, and the canonical
Production Trust loader. It accepts no trust roots, PDSA keys, software private
key, backend, or arbitrary release-policy identity. Evidence contains public
material and qualification metadata only; it explicitly retains the blocked
request-authentication status. Run it only as the separate operator live gate.

Local provider qualification, independent TPM attestation and cryptographic
proof of possession remain separate. Successful NCrypt property checks are a
local provider guarantee, not a verified manufacturer certificate, TPM quote or
legal enrollment. No CNG/TPM physical execution was performed in Codex:

```text
WINDOWS_NATIVE_QUALIFICATION = NOT_RUN
PHYSICAL_TPM_QUALIFICATION = NOT_RUN
```

## Key lifecycle and secret handling

The adapter opens the fixed machine key before considering creation. Only
`NTE_BAD_KEYSET` (key not found) permits first creation; other native errors stop.
A public `CREATION_RESERVED` record is durably written before the native effect.
Creation never requests overwrite. Finalized keys are reopened and qualified by
their actual provider handle, hardware implementation, ECDSA P-256 public blob,
machine scope, signing-only usage, zero export policy and PCP imported/export
origin flag. Unsupported or unavailable properties stop qualification.

An existing qualified key is reused. A committed public fingerprint must match
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

## Remaining Stage 9 work

Resolve/freeze the signed production challenge wire/signature contract and the
independent production key-attestation binding before emitting authenticated
requests. The next planned Stage 9 implementation block remains the production
LPPI successor-key and membership-signing closure, including
`LPPIAuthorityKeyBindingV1`, after those prerequisites. The existing
`ProductionMembershipSignerUnavailable` and production handoff remain fail-closed.
Production freshness, secret resources, full handoff and legal enrollment are
still outstanding. Ceremony trust and provider qualification do not satisfy them.
