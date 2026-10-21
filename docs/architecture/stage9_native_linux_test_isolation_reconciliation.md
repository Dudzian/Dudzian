# Stage 9 — S9-RPI-37: offline TEST isolation reconciliation handoff

**WINDOWS-COMPATIBLE · PURE VALIDATION · UNTRUSTED · PRODUCTION NO-GO**

## Motivation and boundary

Merged Stage 9 components:

- S9-RPI-34: fixed 90 directed negative cases among six abstract roles
- S9-RPI-35: 90 claimed denials plus 18 owner-positive controls, distinguishing
  `EACCES`/`EPERM` from absent files, unknown failures and unattempted tests
- S9-RPI-36: six declared low-privilege TEST role layouts with 0700 protected
  directories, 0600 files and no declared group/UID collisions

S9-RPI-37 joins the **two later declarations** into a single closed, sanitized
operator handoff. A full 108-case outcome declaration alone is inadequate
if the corresponding six-role TEST fixture was missing, unsafe or for a
different *declared* session. This handoff is intentionally **offline** and
can be implemented/reviewed on Windows while Ubuntu is being configured.

**No kernel syscalls, process creation, account provisioning, access to
Linux filesystem state, installation, sudo, network traffic, secrets,
systemd changes, signing or PostgreSQL writes are involved.** This module
has no power to attest authentic processes, host identity or permission
boundaries. Equal declared session references can be trivially fabricated.

## Strict operator-declared interface

The Python function `inspect_declared_handoff(record)` accepts exactly:

- `schema`: `CryptoHunter.Stage9TestOnlyIsolationReconciliationV1`
- `fixture_session_ref`: 32 lowercase hexadecimal characters, not all zero
- `outcomes_session_ref`: same syntactic format
- `fixture`: a closed S9-RPI-36 `Stage9TestFixtureReadinessV1` declaration
- `outcomes`: a closed S9-RPI-35 `Stage9TestOnlySyscallOutcomeClassifierV1`
  declaration

The two references must be equal. They are only **caller-provided correlation
hints**; they are not host-bound, cryptographically authenticated, operator
approved, immutable, secret, or independently verified. Never rely on them
to attest a genuinely shared execution session. Both nested `snapshot`
declarations must also match, but that label is forgeable.

Nested inputs are validated by the original S9-RPI-35/36 closed contracts.
Wrong types, unknown fields, malformed schemas, duplicates and oversized
records are rejected without forwarding caller content. Invalid nested
inputs produce a fixed `INVALID_TEST_HANDOFF` reason, no partial counts
and no user-provided session references in the output.

## Reconciliation and refusal rules

All reports retain `BLOCKED`, `NOT_QUALIFIED`,
`production_emission=BLOCKED`, `topology=PROPOSED_NOT_APPROVED`, fixed
blockers and integer aggregate counts only. Results never include paths,
UID/GID values, role names, PIDs, private data or submitted session references.

- Invalid input/schema: `INVALID_TEST_HANDOFF`.
- Different declared session references: `SESSION_DECLARATIONS_DISAGREE`.
- Different nested snapshot declarations: `SNAPSHOT_DECLARATIONS_DISAGREE`.
- Claimed foreign-role success: `DECLARED_CROSS_ROLE_ACCESS`.
- Incomplete/unsafe/colliding six-role fixture: `TEST_FIXTURE_NOT_COMPLETE`.
- Missing/inconclusive negative cases or owner-positive controls:
  `SYSCALL_OUTCOMES_NOT_COMPLETE`.
- Otherwise, only the **untrusted** label
  `UNTRUSTED_RECONCILED_TEST_DECLARATIONS_ONLY` with
  `NO_INDEPENDENT_LINUX_EVIDENCE`.

The handoff cannot upgrade any component's independent `NOT_QUALIFIED`
status to PASS. It is not evidence of physical separation, verified errno,
privileged OS identity, approved installed services, socket-bound
`SO_PEERPIDFD`, keyring restrictions, PostgreSQL committed-close, CCA
durability, lawful production enrollment or Root Proof signing approval.

## Local no-input command

```bash
python -m deployment.stage9_native_linux_test_isolation_reconciliation
```

It prints only the fixed, no-observations plan and exits **2** on Windows
or Linux. This is intentional, not a failed integration test. In-process
synthetic test fixtures may submit complete closed dictionaries, but doing
so never runs physical denial checks.

## Native Linux handoff

All coding and regression-test authoring for S9-RPI-37 can happen while
the Dell is still being configured. **Actual negative access attempts
and owner-positive controls require native Linux.** Before any execution,
a separately reviewed TEST-only physical fixture must safely create
isolated temporary identities and resources, pin their provenance and
permissions, execute audited syscalls with bounded cleanup and collect
independent outcomes.

**This PR does not start that privileged/physical phase.** Do not add
SSH, Tailscale, production keys, real service names or a headless server
configuration. Existing CI inventories are extended without any security
gate relaxation; private-repository CI failures do not mean tests passed.
