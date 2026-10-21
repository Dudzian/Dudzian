# Stage 9 — S9-RPI-38: Windows-compatible offline TEST execution plan

**INERT · TEST-ONLY · WINDOWS-COMPATIBLE · NOT_QUALIFIED · PRODUCTION NO-GO**

## Purpose

The earlier pure Stage 9 declarations (S9-RPI-34 through S9-RPI-37) cover
six TEST roles, a 90-case cross-role access-denial matrix, 18 owner-positive
controls and consistency checks. S9-RPI-38 turns their fixed inventory into
a **deterministic execution order** suitable for reviewing a future native
Linux TEST harness *before* that harness exists or receives privileges.

This module contains **no executor**, shell command, target path, numeric
host identity, privilege setup, systemd action, database connection, key,
network call or probe. Building a plan on Windows **does not run a single
test operation** and cannot verify isolation on Linux.

## Deterministic 108-step order

The inert plan is generated from the existing S9-RPI-35 inventory:

- **18 owner-positive controls first**, identified by `OWN-001` through
  `OWN-018`. For each of six abstract resource owners, plan one
  `read`, `write`, and `traverse` test under that owner's TEST identity.
  These plan steps have `expected=SUCCESS` and
  `owner_control=SELF`.
- **90 directed cross-role negative checks second**, identified by
  `CROSS-001` through `CROSS-090`. Each fixed actor must try the
  same operation against each of the other five owners' protected TEST
  resources, with `expected=EACCES_OR_EPERM`. Its
  `owner_control` refers to the exact owner-positive control for
  that resource owner and operation.

Each plan row has only seven bounded, fixed-label string fields:
`id`, `phase`, `actor`, `owner`, `operation`,
`expected`, `owner_control`.

**The plan is not an authorization to run the listed actions.**
A later host-specific runner must refuse to attempt a negative test when
its owner control is unexecuted or failed. A control claimed successful
in an untrusted report is not genuine OS evidence. In particular, a write
positive control is potentially modifying: it must be restricted to
reviewed, disposable TEST resources and must not be executed against
installed Source/CCA state, real keys, or live data.

## Offline Python API and fail-closed handling

`build_execution_plan()` returns a fresh Python mapping:

- `schema=CryptoHunter.Stage9OfflineTestExecutionPlanV1`
- `snapshot=SYNTHETIC_TEST_PLAN`
- `steps`: the exact 108 closed operation descriptors above

`inspect_execution_plan(record)` accepts the exact mapping. It validates
the schema, closed keys, string types, strictly canonical ordering, and
the exact required owner-control dependency for every negative step.
A **canonical prefix** is reported only as
`INCOMPLETE_TEST_EXECUTION_PLAN`. A reordered row, duplicate, unknown
case, changed expected outcome or tampered owner dependency is
`INVALID_TEST_EXECUTION_PLAN` with no partial-case credit.

Only the full canonical plan obtains the non-authorizing
`UNTRUSTED_OFFLINE_TEST_PLAN_ONLY` label, accompanied by
`NO_NATIVE_EXECUTION_OR_INDEPENDENT_HOST_EVIDENCE`.
All reports are unconditionally
`BLOCKED / NOT_QUALIFIED / production_emission=BLOCKED /
topology=PROPOSED_NOT_APPROVED`. The report is sanitized to
aggregate counts and fixed status/reason codes.

### Read-only Windows command

From a reviewed repo checkout:

```powershell
python -m deployment.stage9_native_linux_test_execution_plan
```

This prints only the fixed blocked plan summary and returns **exit code 2**
by design, even with a complete plan. It does not probe host capabilities,
run permission operations or read files. Never change the exit code to 0
to manufacture PASS.

A separately reviewed developer fixture may use
`build_execution_plan()["steps"]` to inspect the fixed operations
offline. No host paths, processes or credentials appear in the plan.

## Stop conditions and native Linux boundary

The physical phase must wait for a separately reviewed disposable
TEST-only fixture on native Ubuntu. The executor must independently pin
process UID/GID and supplementary groups, trusted TEST target inodes and
path ancestry, owner-success controls, syscall/errno observations and
bounded cleanup. It must distinguish real permission denial from
`ENOENT`, missed syscalls, inherited privileges or wrong targets.
It must halt on failed owner controls and never write to production data.

**Windows remains sufficient to develop and review this PR.**
The Dell/native Linux is **required only when executing real per-role
syscalls and collecting physical isolation evidence**. No SSH/Tailscale
or server conversion is needed at this stage.

The existing strict Ruff, Stage 9 tests, format and coverage inventories
include this change with no new workflow or weakened security gate.
Private GitHub Actions failures do not establish either validation
success or code failure when the necessary repository tokens are absent.
