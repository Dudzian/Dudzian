# S9-RPI-35 — TEST-only syscall outcome classification with owner controls

**PURE PYTHON / DECLARED OUTCOMES / UNTRUSTED / PRODUCTION NO-GO**

## Purpose

S9-RPI-34 fixes the 90-case cross-role denial matrix for six Stage 9
Source/CCA and signer/issuer roles. A naive physical test might mistake an
absent target file, failed setup or an unexecuted syscall for successful Linux
access isolation. S9-RPI-35 adds a pure, bounded classification adapter that
**distinguishes those results** and insists on a successful positive control
performed by each resource-owning role, for each operation.

**This adapter does not execute syscalls, spawn processes, inspect real Linux
users, create resources, elevate privilege, provision systemd or query a live
host.** All input remains externally DECLARED and unverified. A forged record
can obtain an *untrusted classification label*, never a trusted PASS.

## The complete TEST case plan

The interface derives 90 negative cases from S9-RPI-34 and adds 18 owner
controls (six roles × three operations). Each fixed case has:

- `actor`: one of six reference role labels
- `owner`: one of the same fixed roles
- `operation`: `read`, `write`, or `traverse`
- `outcome`: exactly one of `SUCCESS`, `DENIED_EACCES`,
  `DENIED_EPERM`, `ENOENT`, `OTHER_FAILURE`, `NOT_ATTEMPTED`

Only `DENIED_EACCES` and `DENIED_EPERM` count as *claimed denials* for
cross-role checks. `SUCCESS` indicates *claimed cross-role access*.
`ENOENT`, unknown errors, and unexecuted operations are **inconclusive**,
not effective permission denials.

All 18 same-owner positive controls must be declared `SUCCESS`. A positive
control denied with EACCES or EPERM, missing a resource, not attempted or
reporting any error is a failed or inconclusive control. Positive-control
failure blocks even when every foreign actor is claimed denied.

These labels deliberately avoid overstating errno: even `EACCES` from a real
Linux syscall could arise at the wrong path component, in the wrong process,
or under an unrelated sandbox policy. Real fixture evidence needs additional
independent process/path/identity pinning.

## Closed declared-input contract

`inspect_declared_outcomes(record)` accepts exactly three top-level keys:
`schema` (fixed V1 identifier), `snapshot` (`SYNTHETIC_TEST_INPUT` or
`UNTRUSTED_LIVE_OBSERVATION`) and `cases` (at most 108 fixed rows).
Duplicates, unexpected roles, owner/actor pairs, operations, outcome values,
missing fields and unknown keys are rejected without echoing content.

Example of **synthetic-only** use:

```python
from deployment import stage9_native_linux_test_syscall_outcome_classifier as adapter

record = {
    "schema": adapter._SCHEMA,
    "snapshot": "SYNTHETIC_TEST_INPUT",
    "cases": [
        {
            "actor": actor,
            "owner": owner,
            "operation": operation,
            "outcome": "DENIED_EACCES" if actor != owner else "SUCCESS",
        }
        for actor, owner, operation in adapter.expected_cases()
    ],
}
report = adapter.inspect_declared_outcomes(record)
assert report["qualification"] == "NOT_QUALIFIED"
assert report["classification"]["reason"] == "NO_INDEPENDENT_KERNEL_EVIDENCE"
```

Running the default CLI with no input only shows fixed no-observation blockers
and exits **2**:

```bash
python -m deployment.stage9_native_linux_test_syscall_outcome_classifier
```

## Output and severity

Every report permanently carries `BLOCKED`, `NOT_QUALIFIED`,
`production_emission=BLOCKED` and `topology=PROPOSED_NOT_APPROVED`,
with fixed blockers. The report contains only aggregate counts and fixed
enums; there are **no** paths, user identities, PIDs, secret bytes, syscalls,
exception messages or free-text outcome fields.

- Any claimed foreign-role `SUCCESS`: `DECLARED_CROSS_ROLE_ACCESS`.
- Any failed or inconclusive positive-owner control:
  `OWNER_CONTROL_FAILED_OR_INCONCLUSIVE`.
- Missing positive controls: `OWNER_CONTROL_INCOMPLETE`.
- Missing or inconclusive foreign-role outcomes:
  `CROSS_ROLE_DENIAL_INCOMPLETE_OR_INCONCLUSIVE`.
- An invalid input shape: `INVALID_TEST_REPORT`.
- Only when **all 90 foreign-role claims are denied** and **all 18 owner
  controls claim success**:
  `UNTRUSTED_DECLARED_DENIAL_WITH_OWNER_CONTROLS_ONLY`
  with `NO_INDEPENDENT_KERNEL_EVIDENCE`.

The underlying S9-RPI-34 declared matrix is independently summarized with
its original closed `denial_matrix` observation and reason. Neither
summarization is authentic physical host evidence. Snapshot declarations
do not attest their own origin; `UNTRUSTED_LIVE_OBSERVATION` can be forged.

## Next physical Linux fixture — not in this PR

The eventual reviewed Ubuntu/Dell TEST fixture must identify and isolate
real test principals and resources before attempting operations. For each
claim it must pin effective UID/GID, supplementary groups, executed process,
target path and directory chain, actual attempted syscall, errno, and positive
owner-control evidence, with race/TOCTOU and cleanup considerations.
A passing test also needs evidence that denial was caused by the intended
permission boundary rather than a missing path, disabled service or
mistargeted operation.

**No extra machine is needed to author or review this classifier.**
Physical role-isolation tests will need the Dell running native Ubuntu,
and separately reviewed TEST-only fixture provisioning. No production
credentials, SSH/Tailscale, service creation, sudo or destructive changes
are part of this PR.

Existing Ruff and Stage 9 code/test inventory is extended without bypassing
security gates. Private-repository CI failures remain unverified, not PASS.
