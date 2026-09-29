# Stage 9 — production root-of-trust freeze

## Status and boundary

The canonical physical Windows 11 / TPM 2.0 cross-check completed with exit code `0`:
fixture generation `7` (`TRIAL_DIGEST_ONLY`), observed/runtime normal generation `13`, and final
generation `14`. TPM, canonical generator, and independent verifier agreed for both approved
policies, both branch digests, ordered root, both NV public areas/Names, post-write recovery, and
fixture/runtime normal vectors. Real `K_RECOVERY_TEST` and `K_PSA_TEST` used Owner-backed tickets;
all sessions, external keys, NV and TBS cleanup passed. This evidence is retained unchanged.

```text
Canonical vector ↔ TPM cross-check
[██████████] LIVE PASS

ROOT-OF-TRUST CLOSURE
[█████████░] POLICY TOPOLOGY + CANONICAL TPM SEMANTICS LIVE PROVEN / PRODUCTION FREEZE REMAINS

STAGE 9
[█████████░] IN PROGRESS

CAŁY BLOK WINDOWS 0–14
[██████░░░░] 60.0% — 9/15 DONE

WINDOWS_PRODUCTION_READY = NOT_READY
STAGE10 = NOT_STARTED
PRODUCTION_ROOT_MATERIAL = NOT_PROVISIONED
```

This layer is **IMPLEMENTATION READY / AWAITING PRODUCTION KEY CEREMONY**. It neither starts Stage
10 nor changes the updater, MSI, `WindowsExternalProvisioningHandoff`, or frozen Stages 0–8.

## Existing-authority review and acyclic dependency graph

The review preserves the existing ownership boundaries: PDSA establishes pre-account device
enrollment; LPPI transports and retains verified enrollment and never mints it; CHA remains the
only RootProofIssuer requester; RootProofIssuer never becomes a provisioning authority; Protected
State Authority and Secret Resource Authority retain separate service identities, stores, keys and
NV domains. Product Release Root only authorizes release policy; `K_RECOVERY` only authorizes the
exact recovery TPM policy; per-device `K_PSA` only authorizes the normal branch.

```text
offline Product Release Root (2-of-3)
  └─signs→ ReleasePolicyV1
       ├─pins→ PDSA verification key set/threshold
       │          └─signs→ PDSAEnrollmentPackageV1
       │                       └─binds→ device TPM + per-device K_PSA
       └─pins→ K_RECOVERY public TPMT_PUBLIC + allowed policyRef/scope

PDSA package → LPPI verified registry → CHA request → CHA-only RootProofIssuer
                                   ├→ Protected State Authority (separate key/NV)
                                   └→ Secret Resource Authority (separate TPM KEK/vault)
```

No edge returns to Product Release Root. PDSA cannot add/replace a root, alter its threshold, or
authorize a release policy. LPPI, CHA, RootProofIssuer, PSA and resource authority cannot mint
PDSA/root credentials. Therefore the authority graph has no cycle.

## Product Release Root and ReleasePolicyV1 ceremony

The production root is three independent Ed25519 keys, encoded as lowercase hex of the 32-byte
RFC 8032 public key, with stable ceremony-assigned key IDs and a 2-of-3 threshold. Private keys are
created and retained only in independently controlled offline HSM-equivalent custody; at least two
custodians approve a recorded signing ceremony. They never enter this repository, target hosts, CI,
or disposable tooling. The repository may retain only public keys/IDs, algorithms, signed metadata
and test vectors.

`ReleasePolicyV1` is the unsigned canonical payload. `SignedReleasePolicyV1` is a distinct envelope.
The payload bytes are `CRYPTOHUNTER_CANONICAL_JSON_V1`; `payload_digest = SHA-256(payload_bytes)`;
each signature is Ed25519 over
`"CryptoHunter.Stage9.ReleasePolicyV1\\0" || payload_digest`. Signatures are unique and ordered by
signer ID; unknown, duplicate, invalid, unordered or sub-threshold signers fail closed. Envelope and
payload version must be exactly understood. A valid old signature never permits downgrade.

Root rotation requires a higher release version, old-root quorum plus the documented offline
compromise path, append-only revocation/audit evidence, and monotonic state. Revoked keys never
count toward quorum. A total-root compromise invokes the recorded break-glass custody procedure;
it cannot silently replace pins or reset device/NV generation.

## PDSA verification authority

Release policy pins one lexically ordered `pdsa_verification_key_set` whose entries bind key ID,
algorithm, encoding and raw Ed25519 public key. The verifier rejects duplicate IDs/public keys,
parallel-projection mismatch, a threshold outside `1..key_count`, or a contract ID list different
from the key set. Rotation is authorized only by a newer Product Release Root quorum policy. Each
PDSA package is domain-separated and threshold-signed. Production package verification accepts a
`VerifiedReleasePolicyV1`, never caller-selected PDSA keys, threshold or release digest.

Release verification parses canonical UTC-second RFC3339 `valid_from`/`valid_until`, requires
`valid_from < valid_until` and evaluates an explicit timezone-aware `verification_time` against the
half-open interval. `VerifiedReleasePolicyV1` does not retain a mutable parsed envelope: it contains
only immutable projections (mapping proxies, tuples, frozen values and canonical bytes). Its
constructor is closed for API-misuse protection, but no Python object identity or module-private
token is treated as a security primitive. Every critical downstream entry point re-verifies the
canonical signed bytes, pinned-root identity and revocation chain, then compares the cached
projection. A forged look-alike without valid canonical authority bytes is rejected.

`PinnedProductReleaseRootV1` is loaded at the external trusted-configuration boundary and binds the
ordered IDs/public keys, Ed25519/raw encoding profile, 2-of-3 threshold, purpose/environment and a
cross-language canonical key-set digest. The digest is SHA-256 of
`CRYPTOHUNTER_CANONICAL_JSON_V1` bytes for `CryptoHunter.ProductReleaseRootKeySetV1`; it is never a
Python mapping representation. The same identity must match external pins, signed release payload
and the root context that verified revocation state, even when signer IDs are identical.

## Canonical revocation state

`CryptoHunter.Stage9RevocationStateV1` is a canonical, domain-separated envelope authorized by the
same externally pinned 2-of-3 Product Release Root. Its payload binds sequence, predecessor state
digest, lexically ordered revoked root/PDSA IDs, canonical UTC effective time and authority label.
Verification starts from `TrustedRetainedRevocationHead`, the explicit custody-owned monotonic-store
boundary, and requires the exact next sequence, exact predecessor digest and supersets of both
retained deny sets. Rollback, skipped sequence, removal/unrevocation, unknown root ID, bad signature,
noncanonical bytes and not-yet-effective state fail closed. Signers already revoked by the retained
head cannot authorize its successor.

There is one exact pinned `STAGE9_REVOCATION_GENESIS_V1`: sequence zero, zero state digest and empty
deny sets. Later production heads can only be obtained through a configured
`ProductionRevocationHeadCustodyReader`. Its `read_current_authenticated_record()` operation is the
Protected/Custody store boundary responsible for storage authentication and monotonic current-head
selection; business callers cannot submit record bytes plus a self-computed SHA-256. Downstream
revalidation asks the same reader for the current record again, so a cached older head stops being
valid after the protected store advances. The public head constructor cannot turn arbitrary caller
data into a trusted head. Unit tests use the separately named
`make_test_only_retained_revocation_head`, whose trust domain is rejected for production roots.

Only the opaque `VerifiedRevocationStateV1` result may enter release verification. A
cryptographically valid root or PDSA signature from its revoked set is rejected and never counts
toward quorum. This avoids a cycle: the externally pinned Product Release Root authorizes the
append-only revocation successor; PDSA never authorizes root or revocation changes.

## Production K_RECOVERY

Release policy pins the complete public `TPMT_PUBLIC`, its SHA-256 digest and derived TPM Name
`0x000b || SHA256(TPMT_PUBLIC)`. Profile: ECC NIST P-256, ECDSA/SHA-256, SHA-256 Name algorithm,
attributes `00040040`, empty `authPolicy`, null symmetric/KDF. The private authority remains in
offline custody outside both repository and target host.

Only `SHA256("CryptoHunter.Stage9.NV.BootstrapRecovery.v1")` is an allowed `policyRef`. The
authorized policy must include `PolicyCommandCode(TPM_CC_NV_Increment)` and an exact
`PolicyCpHash` binding both handles to the current pre- or post-write NV Name. It cannot authorize
arbitrary commands or indices. Replacement requires a newer root-signed policy, monotonic migration,
revocation of the predecessor and recorded ceremony; loss uses the offline compromise procedure,
never a generated substitute.

## Per-device K_PSA lifecycle and vector semantics

`K_PSA` is generated inside the target TPM and is expected non-exportable. Its fixed public profile
is ECC NIST P-256, ECDSA/SHA-256, SHA-256 Name, attributes `000400b2`, 32-byte `authPolicy`, null
symmetric/KDF. Production code consumes the public area/Name and TPM operations; it never requires
or exports a private scalar. EK certificate, AK public/quote, creation attestation, device identity,
public area and computed Name are verified and threshold-approved by PDSA.

Replacement is a monotonic PDSA recovery enrollment that revokes the former key. Migration to a
different device is a new enrollment, not key export. Device loss revokes the enrollment. TPM or
motherboard replacement requires reprovisioning and preserved audit linkage; it is never inferred as
an unseen device.

The global freeze covers algorithms, serialization, refs, branch order, NV template and equations.
There is intentionally no single global production root hex because `K_PSA.Name` is per-device.
Only after genuine enrollment may the same generator and independent verifier instantiate and check
the per-device vector.

## PDSAEnrollmentPackageV1

The byte-complete schema binds schema/version, canonical profile, release-policy digest,
device-identity digest, EK certificate digest, AK public/attestation evidence, `K_PSA` public
area/Name/creation attestation, generation, state-enrollment digest, signer IDs, threshold and
signatures. The signature input is
`"CryptoHunter.Stage9.PDSAEnrollmentPackageV1\\0" || SHA256(canonical payload)`.
Verification rejects wrong release, Name or target binding; wrong/unknown/duplicate keys;
insufficient threshold; wrong version; tampered signature; and noncanonical bytes.

## Migration and rollback matrix

Allowed transitions require `new_version == old_version + 1` and the exact enumerated authority,
monotonic evidence, audit evidence and recovery path below. Missing/extra fields, unknown strings,
larger jumps, ambiguous state and rollback are denied.

| Case | Required authority | Required monotonic evidence | Required audit evidence | Recovery path | Result |
|---|---|---|---|---|---|
| `RELEASE_POLICY_UPDATE` | `PRODUCT_RELEASE_ROOT_QUORUM` | `RELEASE_AND_NV_HEAD_ADVANCE` | `SIGNED_SUCCESSOR_RELEASE` | `OFFLINE_RELEASE_CEREMONY` | ALLOW |
| `PRODUCT_ROOT_ROTATION` | `CURRENT_PRODUCT_RELEASE_ROOT_QUORUM` | `RELEASE_AND_REVOCATION_HEAD_ADVANCE` | `DUAL_ROOT_CEREMONY_RECORD` | `OFFLINE_BREAK_GLASS_CEREMONY` | ALLOW |
| `PDSA_ROTATION` | `PRODUCT_RELEASE_ROOT_QUORUM` | `RELEASE_VERSION_ADVANCE` | `SIGNED_RELEASE_AND_REVOCATIONS` | `OFFLINE_RELEASE_CEREMONY` | ALLOW |
| `K_RECOVERY_ROTATION` | `PRODUCT_RELEASE_ROOT_QUORUM` | `RELEASE_VERSION_ADVANCE` | `TPMT_PUBLIC_NAMES_AND_CEREMONY` | `OFFLINE_RECOVERY_KEY_CEREMONY` | ALLOW |
| `K_PSA_REPLACEMENT` | `PDSA_THRESHOLD_RECOVERY` | `DEVICE_GENERATION_ADVANCE` | `EK_AK_CREATION_AND_PACKAGE` | `PDSA_RECOVERY_ENROLLMENT` | ALLOW |
| `TPM_REPLACEMENT` | `PDSA_THRESHOLD_AND_OPERATOR_RECOVERY` | `DEVICE_LINEAGE_ADVANCE` | `LOSS_REPLACEMENT_AND_PACKAGE` | `DEVICE_REPROVISION` | ALLOW |
| `NV_RECREATION` | `RECOVERY_AUTHORITY_AND_PDSA_POLICY` | `RETAINED_HEAD_ADVANCE` | `NV_NAMES_COUNTER_CPHASH_APPROVALS` | `OFFLINE_NV_RECOVERY` | ALLOW |
| `DEVICE_REPROVISION` | `PDSA_THRESHOLD` | `ENROLLMENT_LINEAGE_ADVANCE` | `SIGNED_RECOVERY_LINEAGE` | `PDSA_RECOVERY_ENROLLMENT` | ALLOW |
| `SERIALIZATION_PROFILE_CHANGE` | `PRODUCT_RELEASE_ROOT_QUORUM` | `SUPPORTED_PROFILE_SUCCESSOR` | `COMPATIBILITY_APPROVAL_AND_VECTORS` | `UPGRADE_VERIFIER_OUT_OF_BAND` | DENY in V1 |
| `ROLLBACK_ATTEMPT` | `NONE` | `NEVER` | `ROLLBACK_REJECTION_EVENT` | `RESTORE_CURRENT_EVIDENCE` | ALWAYS DENY |
| `UNKNOWN_FUTURE_VERSION` | `NONE` | `NEVER` | `UNKNOWN_VERSION_REJECTION_EVENT` | `UPGRADE_VERIFIER_OUT_OF_BAND` | ALWAYS DENY |

## FreezeManifestV1 and stop condition

`CryptoHunter.Stage9RootOfTrustFreezeManifestV1` has disjoint pending and frozen states and contains
only public/digest metadata: schemas,
release/envelope digests, root IDs/public keys, PDSA-set digest, recovery public digest/Name,
serialization version, refs, NV template, branch order, derivation, migration version, tool version
and actual Git `source_revision` (or `UNKNOWN` if unavailable). Its checker binds invariant fields to
the release template and rejects any partial material under the unprovisioned status. In
`PRODUCTION_ROOT_OF_TRUST_FROZEN` (or the explicitly isolated
`TEST_ONLY_ROOT_OF_TRUST_FROZEN`) every pending marker is forbidden. The checker recomputes the
payload/envelope digests, exact root set, PDSA-set digest and recovery public digest/Name from a
`VerifiedReleasePolicyV1`; it also binds the verified revocation state digest and sequence. Manifest
hex values are never treated as authority. `product_root_key_set_digest` additionally proves that
manifest root material, signed release root and revocation-verification root are the same canonical
authority rather than merely sharing signer IDs.

`source_revision` is provenance captured when the manifest is built. Verification validates its
canonical `UNKNOWN`/Git-hex representation and its inclusion in the canonical manifest, but does
not replace it with the checker's current checkout revision. Therefore a manifest produced at
revision A remains verifiable at revision B or in a packaged installation without `.git`.

No production anchor has been supplied, so the only valid current status is:

**PRODUCTION MATERIAL REQUIRED — DO NOT GENERATE SUBSTITUTE KEYS.**

MSI/handoff integration remains prohibited until ceremony-produced anchors, signed release,
complete manifest and per-device enrollment pass these verifiers. No further physical TPM run is
needed until a new TPM-dependent executable artifact exists.
