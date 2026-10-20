# Stage 9 — S9-RPI-23: read-only native systemd declaration snapshot

**Status: OBSERVATIONAL_ONLY, NOT_QUALIFIED, PRODUCTION_NO_GO.**
This builds on the NSS/ACL snapshot in S9-RPI-21 and cross-authority
comparison in S9-RPI-22. No production services, users, credentials or
installation policy are provisioned by this change.

## Proposed unit candidates

The fixed *proposed* names below are not approval of the installation
topology or a new trust pin:

| Role | Candidate systemd unit | Candidate User and Group |
| --- | --- | --- |
| Root Source | `cryptohunter-root-custody-source.service` | `ch_root_custody_source` |
| History Source | `cryptohunter-history-custody-source.service` | `ch_history_custody_source` |
| CCA | `cryptohunter-custody-completion.service` | `ch_custody_completion` |

On a **native Linux** host, the existing diagnostic command

```bash
python -m deployment.stage9_native_linux_source_cca_preflight
```

uses fixed `/usr/bin/systemctl --system show --all --no-pager`
queries, requesting only preselected public unit properties. It does
not accept a PID, unit name, path, environment, principal, deployment
namespace or approval value from command-line input. It does not run
`start`, `enable`, `reload`, install packages, write unit files, open
private keys or communicate with the service endpoints.

The report adds `systemd_observations` with per-role candidate unit,
observation and, on a mismatch, machine-readable issues. It checks
declared `Id`, `LoadState`, `ActiveState`, `User`, `Group`,
`DynamicUser`, `MainPID`, `FragmentPath`, `DropInPaths`,
`SupplementaryGroups`, `NoNewPrivileges`, `PrivateNetwork`,
`PrivateIPC`, `PrivateMounts`, `PrivateTmp`, `ProtectSystem`,
`ProtectHome`, `RestrictAddressFamilies`, `CapabilityBoundingSet`
and `AmbientCapabilities`.

The parser rejects duplicate, missing, unexpected, oversized and
malformed property data. A missing service, inactive service, wrong
user/group, dynamic identity, additional group, network authority,
nonempty capabilities or unreviewed drop-in is not accepted. The
bounded subprocess always has a fixed unit name, a finite timeout
and no shell. Errors are deliberately summarized without exposing
arbitrary systemd stderr.

A matching declaration is labeled
`PROPOSED_DECLARATIONS_MATCHED_ONLY` — never `PASS` or
`QUALIFIED`. An offline, non-Linux or injected test snapshot is
labeled `NOT_OBSERVED` / `NOT_OBSERVED_TEST_FIXTURE`.
The existing overall report **always remains**
`BLOCKED / NOT_QUALIFIED / PROPOSED_NOT_APPROVED`,
`production_emission=BLOCKED`, CLI exit code `2`.
The permanent `RUNTIME_SERVICE_IDENTITY_AND_SYSTEMD_NOT_OBSERVED`
blocker is intentionally retained even when unit declarations match.

## Physical host qualification still missing

Even `systemctl show` reports systemd's **declared and cached**
properties, not proof of kernel-effective credentials or independent
authority. This snapshot does not:
- read `/proc/<PID>/status` UID/GID/groups, namespace inode identity
  or credentials of authenticated peers;
- independently check effective DAC, all symlinks/mount aliases,
  POSIX ACL/access-denial scenarios or private keyring isolation;
- pin installed unit bytes, executable digest, drop-ins, executable
  runtime attestation, install provenance or trusted host identity;
- establish rollback trust, real source execution provenance, CCA
  durable log, credential lifecycle lease, PostgreSQL 16 committed-close
  reader or production issuer/signing capability.

**Native Linux/systemd host required for the next genuine physical
qualification phase.** Hosted GitHub CI can exercise the synthetic
observation and refusal paths, but never attest an installed
source/CCA production topology. The controlled operator must separately
review and approve the exact unit/service policy before installing
anything. A distinct rollback trust domain may require independent
media or hardware by later architecture review.

No change to frozen contracts, current readiness status, SQL grants,
Root Proof SIGN/BIND, legal enrollment, Stage 10 or production release.
