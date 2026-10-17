# PDSA-TR-01A: offline enrollment file exchange

The offline adapters implement the decision merged in PR #3147 (`e9cff17b`):
deliver the existing production protocol through files, with the same artifact
contract available to a future `ONLINE_API` adapter. They do not deploy a server,
create production PDSA keys, grant membership, or change Stage 9/10 readiness.

The shared contract is `bot_core/licensing/enrollment_transport.py`. Windows
composition is `deployment/windows_enrollment_offline.py`; the independent Linux
operator composition is `deployment/production_enrollment_offline.py`. Signed
inner artifacts retain their exact existing canonical bytes, signatures, trust
roots and domain verifiers. A transport request ID is a content digest, not an
authorization capability.

## Protocol order and authority

The production protocol requires distinct challenges and more than one outward
and return file:

1. The independent issuer retains an independently quorum-signed
   `PDSAEnrollmentChallengeV1`. It has seven days of validity and is not itself
   evidence of a particular device's custody. Issuance is an independent
   production prerequisite; the file adapter does not generate this signature.
2. Windows exports its public activation request, TPM enrollment request and
   retained AK identity as `DEVICE_REQUEST`. The independent issuer also needs
   the quorum-signed `ProductionTPMEndorsementV1`, which follows manufacturer
   certificate validation and approval of the exact EK.
3. The issuer checks its retained PDSA challenge, endorsement and public TPM
   bindings, then uses the existing TPM challenge store to retain the credential
   secret and issue a distinct TPM challenge. `TPM_CHALLENGE` returns that
   challenge with the signed PDSA challenge and endorsement.
4. Windows verifies the return artifact against its retained local request and
   identity. The target TPM must perform the qualified production credential
   activation, AK attestation and production-domain proof of possession. Its
   response is a separate artifact; the historical TEST_ONLY activation bridge
   cannot produce production evidence.
5. The existing fixed CNG pre-enrollment producer signs the exact
   `PreEnrollmentRequestV1` and collects request-bound custody evidence.
   `PRE_ENROLLMENT` transports these bytes and the complete challenge/exchange
   evidence back to the issuer.
6. The issuer uses `authenticate_production_pre_enrollment` and
   `issue_production_pdsa_enrollment_authorization`. It retains the issuance
   reservation, obtains the approved quorum signatures through the fixed signer
   boundary, verifies them, and commits the exact result. It does not consume the
   request through a separate receipt-only workflow before package issuance.
   `PDSA_PACKAGE` returns the existing signed authorization package.
7. Windows uses `accept_production_lppi_package` with its current installed trust,
   retained request, open-only local key, fresh possession proof and TPM custody.
   Import alone cannot establish legal membership. The explicit successor-key
   lifecycle in `windows_production_lppi_authority.py` is a later operation and
   is not called by the file importer.

The file sequence is therefore:

```text
Windows                         independent PDSA/Linux
DEVICE_REQUEST  ---------------------->
                <----------------------  TPM_CHALLENGE
qualified local TPM response + signed pre-enrollment request
PRE_ENROLLMENT  ----------------------->
                <----------------------  PDSA_PACKAGE
live local LPPI package acceptance
```

## Operator example

These commands describe the installed production path. They require the
prerequisites below and are not a recipe for a software-only production
enrollment. Directories must already exist with the required protection. Files
may be carried out of band; their names do not confer authority.

On Windows, export public artifacts from the qualified retained TPM identity.
`start` creates or reuses the fixed pre-enrollment CNG identity and checks the
retained AK; it is a consequential native operation, not a preview.

```powershell
python -m deployment.windows_enrollment_offline start --activation-request .\public\activation.json --tpm-request .\public\tpm-request.json --ak-key-name RETAINED_AK_NAME --output .\out\device.json
```

Retain the resulting 64-character device request ID. On the independent PDSA
station, inspect the untrusted file without opening issuer authority:

```sh
python -m deployment.production_enrollment_offline preview inbox/device.json
```

Under the installed issuer service principal, select an already issued, retained
PDSA challenge and the independently approved endorsement, and produce the TPM
challenge file:

```sh
python -m deployment.production_enrollment_offline export-challenge --challenge-id pchal_UUID --output out/pdsa-challenge.json
python -m deployment.production_enrollment_offline issue-tpm inbox/device.json --pdsa-challenge out/pdsa-challenge.json --endorsement approved/endorsement.json --output out/tpm-challenge.json
```

`export-challenge` exports the exact raw signed challenge already retained by the
issuer; it does not issue one or wrap it in a transport envelope. This file is an
operator input to `issue-tpm`. Only `tpm-challenge.json` needs to travel back to
Windows, since it embeds both challenges and the endorsement.

Back on Windows, import the challenge and obtain the qualified production TPM
response through the separately implemented and qualified native producer.
There is no production response producer supplied by this adapter.

```powershell
python -m deployment.windows_enrollment_offline import-challenge --input .\in\tpm-challenge.json
python -m deployment.windows_enrollment_offline prepare --device-request-id DEVICE_REQUEST_SHA256 --tpm-response .\public\tpm-response.json --output .\out\pre-enrollment.json
```

On PDSA/Linux, preview before authorizing. `authorize` invokes real production
verification, retained-state issuance and the fixed external quorum signer; it
is not a dry run.

```sh
python -m deployment.production_enrollment_offline preview inbox/pre-enrollment.json
python -m deployment.production_enrollment_offline authorize inbox/pre-enrollment.json --output out/package.json
```

Finally, on the original Windows machine:

```powershell
python -m deployment.windows_enrollment_offline import-package --device-request-id DEVICE_REQUEST_SHA256 --input .\in\package.json
```

Keep the exact request and response files for retries. Repeating an identical
request returns the retained exact response; it does not mint a fresh identity,
renew a challenge or extend a package's authority. The protected issuer journal
binds a device request to its selected PDSA challenge and endorsement. Changing
that selection on retry fails. Existing issuer TPM and authorization stores
retain pending/committed work across restarts; a committed package can be
retrieved after a lost response without creating a second issuance. Windows
retains its local identity and stage history and rejects conflicting imports.

## Security boundary

The envelope schema is `CryptoHunter.EnrollmentArtifactDelivery.v1`, with exactly
`schema_version`, `kind`, `parent_id` and `artifacts`. `kind` selects one of the
four fixed artifact sets above. Public bytes are embedded as lowercase hexadecimal
without rewriting them. The request ID is SHA-256 of the complete canonical
envelope; `parent_id` binds the preceding exact envelope, with `null` only for
`DEVICE_REQUEST`.

| Kind | Exact artifact names |
| --- | --- |
| `DEVICE_REQUEST` | `activation_request`, `tpm_request` |
| `TPM_CHALLENGE` | All `DEVICE_REQUEST` artifacts plus `pdsa_challenge`, `endorsement`, `tpm_challenge` |
| `PRE_ENROLLMENT` | All `TPM_CHALLENGE` artifacts plus `tpm_response`, `request`, `request_signature`, `custody_evidence` |
| `PDSA_PACKAGE` | `package` |

An envelope is at most 2,097,152 bytes. Embedded decoded limits are 32,768 bytes
for activation requests and packages; 16,384 for PDSA/TPM challenges and
pre-enrollment requests; 131,072 for TPM requests/responses, endorsements and
custody evidence; and 72 for the DER request signature. Unknown fields,
noncanonical JSON/hex, malformed or oversized data and invalid stage transitions
are rejected. The inner signed contracts are verified separately; an unsigned
transport hash cannot substitute for a signature, current trust, retained issuer
state or TPM custody.

The shared file layer accepts regular single-link files only, rejects unsafe
paths, bounds reads before parsing, pins directory identity and publishes exact
bytes atomically. On POSIX it uses directory-relative operations; Windows uses
pinned local NTFS directory/file handles and reparse checks. Existing destination
bytes must agree for an identical retry. Protected stage records preserve exact
artifact digests and parent links as audit evidence. These local records do not
provide independent rollback protection against a privileged administrator.
Linux publication requires `renameat2(RENAME_NOREPLACE)` and fails closed when
that primitive is unavailable. Windows operations hold the existing LPPI
lifecycle lock through verification and publication, so a concurrent successor
operation cannot invalidate the unenrolled-state check. Taking this lock may
create an empty authority directory and lock file; it creates no authority state.

The production adapters accept artifact paths, not selectable trust roots,
signers, production keys, issuer databases or network endpoints. Linux authority
comes only from the no-argument installed issuer factory, its service account,
protected canonical state and current installed trust. The signing boundary is
the existing fixed `/run/cryptohunter/pdsa-signing/quorum.sock` proxy to the
independent 2-of-3 PDSA quorum. Its absence remains a blocking error. Neither
adapter exports a CNG/TPM private key or carries issuer private keys. TEST_ONLY
fixtures and deterministic keys cannot bootstrap production authority.

The shared contract can later be delivered by `ONLINE_API`. An authenticated
network session would add delivery controls; it would still need the same
challenge states, signed bytes, current trust, TPM binding and live LPPI
acceptance. No online endpoint or always-on infrastructure is part of this work.

## Production prerequisites and qualification gaps

- An independently protected Linux PDSA station, the installed non-root issuer
  service account, canonical protected state directories and current Production
  Trust package must be deployed and qualified.
- Genuine production quorum keys and the independently operated signer/proxy
  must exist. Remote durable digest reservation, exact retry replies and peer
  authorization require deployment qualification. The client is not a signer
  implementation and does not generate or replace production keys.
- Signed PDSA challenge issuance and manufacturer-backed EK endorsement require
  independent production ceremonies/services. The available signing client signs
  authorization packages; it does not supply these missing issuance operations.
- A qualified production Windows TPM response producer remains required. The
  TEST_ONLY transient-key bridge is not an acceptable substitute.
- Installed Windows trust, protected state ACLs, retained physical TPM/AK
  identity, native CNG/PCP properties, credential activation and creation-custody
  evidence require native and physical qualification.
- Successor-key establishment and any later membership issuance remain explicit
  existing authority workflows. File import does not run them.

Software integration tests can exercise the complete exchange with genuine
cryptographic operations in isolated TEST_ONLY fixtures. They cannot establish
physical TPM custody, native Windows file-I/O qualification or live quorum
deployment. No legal production enrollment, MSI installation, physical TPM/NV
modification or Stage 9/10 status transition is authorized by those test results.
