# Stage 9 — S9-RPI-21: native Linux source/CCA host preflight (READ-ONLY)

**Current authority:** `deployment/stage9_current_status.json`, Stage 9
`IN_PROGRESS`, `PRODUCTION_NO_GO`. The new preflight always returns
`BLOCKED`, `NOT_QUALIFIED` and exit code `2`, **even if all
proposed paths and account identities appear to match**. It is NOT a
qualification engine or permission to perform an installation.

## Transition from synthetic tests to a real host

S9-RPI-01/02 provide the existing independent lifecycle and diagnostic
signer service patterns. S9-RPI-06/07 define the future Root Source,
History Source and independent CCA boundaries. S9-RPI-15 through S9-RPI-20
prove original-byte retention, source+CCA Ed25519 signatures and
cooperative source/credential/log lock ordering with TEST-only principals.

Those checks still run as synthetic same-UID fixtures and cannot prove
a physical installation, production private-key control or a protected
trust anchor. This PR is the first **read-only host inventory** for
the *proposed*, not approved/installed, source/CCA topology.

The [preflight module](../../deployment/stage9_native_linux_source_cca_preflight.py)
resolves via host NSS the proposed accounts:

| Role | Proposed account | Candidate configuration | Candidate private state |
| --- | --- | --- | --- |
| Root Source | `ch_root_custody_source` | `/etc/cryptohunter/root-custody-source` | `/var/lib/cryptohunter/root-custody-source` |
| History Source | `ch_history_custody_source` | `/etc/cryptohunter/history-custody-source` | `/var/lib/cryptohunter/history-custody-source` |
| CCA | `ch_custody_completion` | `/etc/cryptohunter/custody-completion` | `/var/lib/cryptohunter/custody-completion` |

It records only account numeric UID/GID, supplementary group IDs and
`lstat` metadata of proposed directories and their ancestors.
Observed missing accounts/paths, shared UIDs/groups, any supplementary GID
(including GID `0` or a foreign signer group), wrong type, symlinks,
writable or non-root-owned ancestors, wrong expected directory ownership,
permissions and visible POSIX extended ACL are reported as failures.
S9-RPI-06 requires an exclusive primary service group with no supplementary
memberships; no exception has been approved by a separate contract.
The proposed mode is `0750 root:role_gid` for configuration and
`0700 role_uid:role_gid` for private state; ancestors must not have
group/world write permission. Checks include the base root itself: the live
`/` must be a root-owned directory without group/world write permissions or
extended POSIX ACL. A temporary TEST root may belong to the test runner,
with the same type, write-permission and ACL checks; it cannot qualify a
production installation. The tool does not open private data,
socket connections, keyring entries, PostgreSQL connections or installer
artifacts. It neither provisions nor repairs.

**This is only a point-in-time observational snapshot.**
NSS group membership may not describe runtime credentials; systemd
sandboxing, process UID/GID/groups, effective access, alternate paths,
mount/IPC/net namespaces, live peer credentials, POSIX ACL race,
native keyring cross-role denial, source/CCA owner-installed trust pins
and PostgreSQL catalog/close are **not** qualified by it. An observed
match is labeled `MATCHED_PROPOSED_METADATA_ONLY`, not `PASS`.
No caller-supplied approval, missing-account default, partial observation
or copied JSON can promote `status`. The CLI accepts no argument paths or
role selectors, and its exit code is deliberately nonzero.

## Execution once a trusted native Linux host exists

From an independently reviewed and operator-approved checkout on a
native Linux host, with appropriate authorization for read-only
diagnostics:

```bash
python -m deployment.stage9_native_linux_source_cca_preflight
```

Output is machine-readable JSON on standard output. The process ends
with exit code `2` by design; **do not pipe it into an automated
GO/installation decision**. Preserve the full original report alongside
independently observed, protected host identity. The command never writes
a result file, creates a user, installs units, mutates ACL or accesses
private keys. The GitHub Linux CI exercises **fixture-level refusal
behavior**, not the physical deployment and not an operator approval.

## Next production qualification (not implemented here)

1. Approve role names, UID/GID, trust pins, endpoints and physical
   installation, rather than treating the candidate list as an authority.
2. Install and independently observe native systemd services and
   effective principal/namespace/ACL/keyring separation with explicit
   cross-role denial tests. Require a qualified, protected
   installation and authenticated peer identity.
3. Qualify source execution provenance, custody lease, credential
   generation history, CCA independent durable log and trusted rollback
   floor. An original signed synthetic CCA envelope cannot attest these.
4. Qualify independent physical PostgreSQL 16 committed-close reader,
   catalog migration and privilege/lock order.
5. Integrate reviewed A+C transaction addendum and production
   Root Proof Issuer under its separately approved gates, then physical
   failure/COMMIT-unknown/recovery tests and legal enrollment.

Existing production `SIGN/BIND`, issuance, private-key enrollment,
Stage 10 and readiness remain unchanged and blocked. No frozen JSON,
SQL migration, ACL, systemd unit, principal or capability is modified.

**HARDWARE DEPENDENCY: physical native Linux host REQUIRED FOR THE NEXT
AUTHORITY QUALIFICATION PHASE** (but not for this PR's CI tests).
The existing Windows machine is not sufficient to prove Linux
systemd/UID/GID/DAC/namespace behavior. A dedicated Linux host or an
appropriately isolated approved Linux environment must be made
available; a separate rollback domain may require another
independent host/media by later architecture approval.
