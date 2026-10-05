# Final acceptance audit: Marketplace production signing

Date: 2026-10-05
Audited change: `Harden marketplace signing trust boundary` (`26e4d1f`, merged by
`4798cf6`). The requested object `2f0898f` is not present in this shallow checkout;
the merge parents were therefore used to reconstruct and inspect the complete PR
diff.

## Verdict

**CHANGES REQUIRED**

The cryptographic verification primitives are fail-closed, but the production
runtime is not composed with the new trust policy. In addition, a normal push to
`main` selects the production-signing deploy job, so merging without provisioned
credentials causes the workflow to fail.

## Blocking findings

### 1. A normal `main` push requires production credentials

The repository-wide CI workflow runs on every pushed branch and tag. Its `deploy`
job depends on both `marketplace-catalog` and `ui-packaging` and runs for a push to
`main`, `stable`, `release/*`, or `v*`, and for every manual dispatch. The job has
no GitHub `environment` declaration or approval gate. It explicitly aborts when
any signing input is empty.

Consequently, absent credentials **will break CI after merge to `main`**. The
required values are:

* repository/environment secret `MARKETPLACE_PRODUCTION_ED25519_PRIVATE_KEY`;
* repository/environment secret `MARKETPLACE_PRODUCTION_ED25519_PUBLIC_KEY`;
* repository/environment secret `MARKETPLACE_PRODUCTION_HMAC_KEY`;
* repository/environment variable `MARKETPLACE_PRODUCTION_KEY_ID`;
* repository/environment variable `MARKETPLACE_PRODUCTION_ISSUER`.

Provisioning before merge is the smallest safe sequencing fix under the existing
release semantics. If an ordinary `main` push is not intended to be a production
release, instead change the job trigger before merge to the actual release event
(for example a protected version tag or a guarded manual dispatch). Do not make
the credentials optional: their absence during a selected production release
must continue to fail closed.

### 2. The production trust policy is not wired into the real runtime

`SignedPresetMarketplace` now treats its default environment as production and
raises when no `MarketplaceTrustPolicy` is supplied. However,
`StrategyCatalog.sync_signed_marketplace` still constructs it with only the old
`signing_keys` mapping. The runtime configuration model and loader likewise expose
only `signing_keys` and `allow_unsigned`; they do not construct or propagate a
policy containing environment, issuer authorization, and the independently
provisioned public key. Runtime initialization catches broad exceptions, so this
can degrade into Marketplace presets silently not being loaded rather than a
visible trust-boundary failure.

`MarketplacePresetInstaller` also still constructs `MarketplaceService` with only
the legacy key mapping. That path verifies possession of a configured public key,
but does not enforce the new issuer/environment authorization policy. Therefore
the audited end-to-end chain stops at bundle validation and is not demonstrated
through the actual installer/runtime registration composition.

Before approval, the central production composition must accept an explicit
`MarketplaceTrustPolicy`, pass it through every production import/load/sync path,
and have an integration regression test that signs with an ephemeral production
key, installs the resulting bundle, and reaches registration. Negative variants
must cover missing policy, development key/issuer, missing or unknown environment,
unknown key ID, wrong issuer, and a self-signed embedded key.

## Trust and cryptographic assessment

### Ed25519 trust anchor

Within the new verifier, `key_id`, issuer, and environment are authorized by
`MarketplaceTrustPolicy` before signature verification. Key material comes from
the policy; an artifact's embedded `public_key` is only checked for equality and
cannot become the trust anchor. Production policy construction rejects reserved
development identities and issuer, the known development key fingerprint, an
empty keyring, unknown keys, wrong issuers, wrong environments, and a missing
environment. Legacy missing-environment acceptance is explicitly unavailable in
production.

Storing the public key as a GitHub Secret is **acceptable but unnecessary**. It is
not confidential. A protected GitHub environment variable, or a deliberately
versioned public trust anchor reviewed under code-owner/branch protection, gives
better operational visibility. The important property is independent provisioning
and protected change control, not secrecy. The current Secret choice is not itself
an architectural blocker.

### HMAC

The production HMAC key creates and verifies the HMAC block on `catalog.json` and
`catalog.md` inside the CI/build bundle-validation boundary. It is read from a
GitHub secret, written to an ephemeral runner file, and consumed by catalog
generation and `build_release_bundle.py`. The runtime preset verifier supplied by
this PR rejects HMAC when a trust policy is active and relies on Ed25519.

Thus HMAC is currently a redundant build-time integrity/authentication check, not
the public distribution root of trust. This is cryptographically appropriate only
while the HMAC secret remains in trusted build infrastructure. It must not be
shipped to untrusted clients or treated as proof of artifact origin there;
Ed25519 and the independently provisioned public key must remain the production
authenticity boundary.

## Backward compatibility

At the policy layer, a signature without `environment` may be accepted only when
`allow_legacy_missing_environment=True` and the policy itself is non-production.
Production always rejects missing and unknown environments. Unit tests cover the
production rejection and the runtime rejection of a test-domain artifact.

The requested full integration proof is incomplete because the real runtime does
not construct the policy. There is also no end-to-end test of a legacy artifact
without `environment` being accepted in an explicitly development/test runtime
while rejected through the production installer/runtime path.

## Hardware-wallet follow-up (non-blocking for this PR)

This code is not connected to Marketplace signing. For both Ledger and Trezor,
`device_public_key` (or Ledger X/Y coordinates) is emitted from device/simulator
metadata into the signature document. Verification reconstructs its verification
key from that same signature document and verifies the canonical JSON payload.
The signature therefore proves possession of the corresponding private key, but
this module does not bind that key to an enrolled physical device, attestation
certificate, manufacturer root, account, or approved derivation-path identity.
The embedded key is presently an identity claim, not an independently established
trust anchor. A separate threat-model review should define enrollment/attestation
and the key-to-device binding before treating it as hardware identity.

## Test and validation record

* Targeted Marketplace, signing-policy, catalog, release-bundle, installer, and
  runtime-registration tests: **158 passed**.
* Ruff lint and formatting checks for all Python files changed by the PR: passed.
* YAML parsing of all 15 workflow files: passed.
* `actionlint` identified pre-existing repository workflow findings, including the
  unsupported `artifact-metadata` permission in `ci.yml`; blame places that line
  before this PR. It found no new error in `marketplace-catalog.yml`.
* A direct real-call-path probe of `StrategyCatalog.sync_signed_marketplace`
  produced `ValueError: production Marketplace runtime requires an external trust
  policy`, confirming the missing production composition described above.

## Required disposition

Do not merge the audited change as production-ready yet. Wire the external policy
through the real runtime/installer path and add its end-to-end test. Separately,
either provision all five production settings before merge under the existing
automatic-`main` release semantics, or narrow the production job to the real,
protected release event while retaining mandatory-credential failures.
