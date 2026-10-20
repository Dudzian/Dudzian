# Stage 9 — S9-RPI-22: read-only cross-authority Linux principal inventory

**Status: `TEST_ONLY_DIAGNOSTIC`; `PRODUCTION_NO_GO`.** This step extends
[S9-RPI-21](stage9_native_linux_source_cca_host_preflight.md), not its trust
authority. Actual OS and PostgreSQL separation requires future installation
approval and physical evidence.

## Why RPI-21 was insufficient

RPI-21 checked internal UID/GID and supplementary group collisions only
between its **three proposed** Root Source, History Source and CCA accounts.
That cannot reveal a collision with an already prepared Root signer, History
signer or RootProofIssuer. A candidate source account which reuses a signer's
UID would otherwise have appeared internally distinct.

The same read-only NSS preflight now observes these **known named references**,
derived from existing reviewed service assets:

| Authority | Fixed reference NSS account |
| --- | --- |
| Root signer | `cryptohunter-root-proof-signer` |
| History signer | `cryptohunter-history-attestation-signer` |
| Issuer | `cryptohunter-root-proof-issuer` |

`_native_accounts()` performs lookups for both the three proposed and
three reference roles. It does not create accounts or supply fallback IDs.
It retains the same report structure and the same mandatory
`status=BLOCKED`, `qualification=NOT_QUALIFIED`,
`production_emission=BLOCKED` and CLI exit code `2`.

The extra report fields are `reference_principal_observations` and
`unobserved_reference_principals`. Missing fixed reference users
are reported explicitly, **not assumed to be isolated**. When identities
are visible, the tool rejects malformed/privileged reference UIDs/GIDs,
a root supplementary group, overlapping UIDs and shared effective groups
across the six observed roles. The three proposed source/CCA principals
retain their stricter **no supplementary group** rule.

## Scope and provenance limits

An NSS name/UID/group tuple is a read-only snapshot, **not** a protected
installation pin, proof of the current process `getresuid/getresgid`,
effective DAC/ACL result, systemd unit identity, source signature,
native keyring isolation or peer credential attestation. Root, History,
CCA, requester/claimant, lifecycle administrator, trust activation and
checkpoint authorities still need their **approved physical identities
and boundaries**. Names of further authorities are not invented to
make an artificial complete inventory.

Known references are looked up using the operating system's NSS
configuration; an administrator or compromised NSS backend can change
these facts. TOCTOU changes remain possible between calls. A missing
or malformed reference leaves current production decisions blocked,
and an apparently clean six-role snapshot does not authorize issuance.
The only trusted status source remains
[`stage9_current_status.json`](../../deployment/stage9_current_status.json)
with Stage 9 `IN_PROGRESS`.

No change to frozen contracts, SQL/grants, production signer/CCA
entrypoints, service units, file ownership, credential, enrollment,
`SIGN/BIND` or Stage 10. Existing Ruff and Linux Stage 9 CI jobs
already cover the changed files; no new workflow or physical CI job.

**Physical Linux/systemd host: REQUIRED for the subsequent actual
cross-role observation/qualification.** No second computer is needed to
test this software change with GitHub Actions. An independent rollback
domain remains a separate architecture and host/media decision.
