# Stage 9 — S9-RPI-31: untrusted identity and namespace drift

**READ_ONLY / SYNTHETIC_TEST_INPUT or UNTRUSTED_LIVE_OBSERVATION /
NOT_QUALIFIED / PRODUCTION_NO_GO.**

S9-RPI-31 extends the existing
[two-pass read-only diagnostic](stage9_native_linux_two_pass_drift_observation.md)
without adding a new privileged collector or changing the five existing
observer functions. Two sequential observations of proposed Source/CCA
processes already compare numeric PID, starttime and unpinned unit/executable
object fingerprints. This addition examines additional **in-memory metadata**
which the existing strict S9-RPI-24 identity parser has already validated.

For each fixed role, compare the following between the two observations:
- observed numeric UID and GID;
- zero or one supplementary GID (same restricted shape as S9-RPI-29);
- mount, IPC and network namespace inode hints.

The shared report validator also rejects successful identity rows unless UID
and GID are builtin integers from 1 through `2**32 - 2` and groups are either
`[]` or `[gid]`, with the same integer restriction. Zero, booleans, foreign
groups and malformed types invalidate the identity source for that sweep;
they cannot become matching observations, even if repeated in both sweeps.
These checks validate only the numeric contract. They do not independently
confirm the report's UID/GID against NSS or promote the observations to trust.

For each pair of roles **within either sweep**, independently flag:
- a shared numeric UID, primary GID or supplementary group;
- a shared inode for the *same* namespace type (mnt, ipc or net).

The corresponding fixed issue codes are:
`identity: CREDENTIAL_HINT_CHANGED_BETWEEN_SWEEPS`,
`identity: NAMESPACE_HINT_CHANGED_BETWEEN_SWEEPS`,
`CROSS_ROLE_SHARED_CREDENTIAL_HINT`,
`CROSS_ROLE_SHARED_NAMESPACE_HINT`.

A refusal/malformed identity record cannot become a trusted observation:
it remains `identity: INCOMPLETE_TWO_PASS_OBSERVATION`.
Errors in one role do not automatically mislabel other roles, but collisions
affect each involved role. An observation can still be `NOT_QUALIFIED`
with a matching PID and starttime. The output contains only fixed role names,
refusal labels and counts, **not** UID/GID, namespace inode values, raw
subreports, paths, file hashes, private environment or exception messages.

The parent schema additionally records `sweeps_attempted`: 0 if the
environment/configuration refuses before sampling, 2 after both sweeps were
attempted. Existing `sweeps=2` continues to mean the *planned* sweeps.
This counter does not certify successful individual observations.

## Limits and authority

Same UID/GID/namespace numerical hints are not authenticated peer
credentials or proof of effective service access control. Sequential
`systemctl`, NSS, procfs and pidfd reads can race, and matching results
cannot exclude coordinated changes, PID recycling or ABA. In particular,
`pidfd_open(MainPID)` is not authenticated socket-bound `SO_PEERPIDFD`.
These observations cannot replace protected host/unit/executable pins,
effective cross-role denial, native keyring isolation, original source
custody provenance, PostgreSQL committed-close or CCA durable log proofs.

**Every report remains** `BLOCKED / NOT_QUALIFIED`,
`production_emission=BLOCKED`, `topology=PROPOSED_NOT_APPROVED`;
CLI always exits **2**, even when no change is detected. No new SQL
grants, service installations, elevation, credential mutations, signing
authority, release approvals, CI skips, changed timeouts or Stage 10
modifications are introduced.

## Validation and environment

Dedicated synthetic regressions exercise independent UID/GID/groups drift,
all three namespace types, shared credential/namespace hints in one
sweep, missing per-role observations, a changed object size with an
unchanged digest and a zero-count preflight refusal. Existing mandatory
Ruff, Quality Ratchet and coverage scopes for the extended module remain
unchanged; the current private-repository CI may be unavailable. None of
these synthetic tests substitute for real physical qualification.

**No additional computer is necessary for this code-only task.
A controlled native Linux/systemd host IS REQUIRED for genuine live-host
qualification and effective cross-role denial evidence.**
