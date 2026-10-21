# Stage 9 — S9-RPI-32: sanitized native Linux diagnostic handoff

**READ-ONLY · UNTRUSTED · NOT_QUALIFIED · PRODUCTION_NO_GO.**

This addition connects two existing, independent read-only observations without
combining them into an authorization decision:

- [Native host/NSS/systemd/path preflight](stage9_native_linux_source_cca_host_preflight.md),
  covering the three *proposed* Source/CCA identities and three known reference
  signer/issuer identities.
- [Two-pass Source/CCA drift observation](stage9_native_linux_two_pass_drift_observation.md),
  including the S9-RPI-31 credential/namespace comparisons.

## Running the handoff on a Linux host

From an independently reviewed checkout, in an appropriate read-only
operator session:

```bash
python -m deployment.stage9_native_linux_diagnostic_handoff
```

The command accepts **no CLI arguments** and never changes services, accounts,
permissions, keyrings, SQL schemas, signing state, systemd units, application
config or on-disk evidence. It invokes each existing diagnostic **once**,
sequentially, with no retry or new system operations. Existing native
observers may invoke bounded systemctl/procfs/NSS read-only queries.

The output has fixed schema
`CryptoHunter.Stage9NativeLinuxDiagnosticHandoffV1`.
It emits only fixed observation labels, fixed failure codes, role identifiers
and bounded counts. It does **not** forward subordinate paths, environment,
PIDs, numeric UID/GIDs, namespace inodes, digest bytes, unknown issue strings
or exception contents. An invalid/exception-producing backend is replaced by
`INVALID_*_REPORT` or `OBSERVER_EXCEPTION`.

A preflight label `UNTRUSTED_DECLARATIONS_ONLY` requires no reported issues,
all expected proposed roles, reference roles, systemd declaration hints and
proposed paths. A drift label `UNTRUSTED_TWO_PASS_METADATA_ONLY` requires
two attempted sweeps and complete, contradiction-free numeric hints across
three proposed roles. **These labels are not PASS, READY or trustworthy proof.**
Subordinate reports remain untrusted and may be affected by spoofed host
metadata, incomplete snapshots, races and coordinated attacks.

The output is **always**:
- `status=BLOCKED`;
- `qualification=NOT_QUALIFIED`;
- `production_emission=BLOCKED`;
- `topology=PROPOSED_NOT_APPROVED`;
- process exit code **2** (including two apparently consistent observations).

A Linux operator must treat the diagnostic as **triage only**, not as signed
evidence or a trusted installation attestation. Copying its JSON, rerunning it,
or forging matching values cannot satisfy source custody, effective cross-role
denial, authenticated socket `SO_PEERPIDFD` peer identity, keyring isolation,
independently pinned unit/executable manifests, PostgreSQL committed-close
durability, CCA log durability or lawful enrollment. `pidfd_open(MainPID)`
remains insufficient. No Stage 10 authorization or release gate changes.

## Validation and environment boundary

Synthetic tests verify closed input/report shapes, fixed output,
independent invocation order, private-data redaction, backend exceptions,
mixed role failures, incomplete observations, non-Linux refusal and exit 2.
Mandatory Ruff, Quality Ratchet and coverage scopes include this tool;
private-repository resource/permission limitations must not be confused with
test regressions or represented as successful validation.

**CURRENT HARDWARE REQUIREMENT:** S9-RPI-32 coding and synthetic testing need
no additional machine. To run the diagnostic against actual Linux systemd,
procfs and NSS you **need access to a Linux/systemd computer or appropriately
isolated test environment**. A newly set up Linux laptop can be used for
initial triage. Production qualification still requires a separately approved
physical trust/installation environment, controlled identities, verified
effective denial and protected evidence; a laptop's matching diagnostic output
alone cannot provide these.
