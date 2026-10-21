# Stage 9 — S9-RPI-30: two-pass observational drift (NO-GO)

**OBSERVATIONAL_ONLY / NOT_QUALIFIED / PRODUCTION_NO_GO.**

This diagnostic follows
[S9-RPI-29](stage9_native_linux_cross_observer_consistency.md).
A single cross-source comparison can miss a coherent change of PID or
starttime between two moments. This additional probe makes **exactly two**
consecutive, deterministic sweeps over the same five existing observers,
without sleep, retry, privileged writes or new native syscalls.

Run:

```bash
python -m deployment.stage9_native_linux_two_pass_drift_observation
```

Each sweep invokes the fixed read-only identity, cgroup, unit-fragment,
executable and pidfd probes in that order; their existing bounded native
systemd/procfs checks remain unchanged. Every subreport must pass the strict
closed-shape S9-RPI-29 parser (including its fixed, unqualified production
status). Unparseable data, backend exceptions and incomplete rounds become
fixed refusal codes.

### What is compared

For the three proposed Source/CCA roles, the module compares only these
untrusted hints:

- numeric PID for all five observers, between sweep 1 and sweep 2;
- process starttime ticks for identity, cgroup, executable and pidfd;
- unit-fragment or executable digest **and size**, compared only in memory;
- cross-source PID/starttime contradictions within **each** sweep;
- cross-role reused PID hints within **either** sweep;
- per-role and source-wide anomalies already reported by subordinate probes.

A matching pair of sweeps yields only the literal diagnostic
`NO_TWO_PASS_CONTRADICTION_OBSERVED`. The output contains no PID values,
tick values, filesystem paths, hashes, raw error messages, procfs contents,
private metadata, or exception text. It contains only the fixed role names,
closed diagnostic code strings and the count of sources observed in both
sweeps.

### Explicit trust boundary

This cannot rule out process exit and restart after the final check,
PID-reuse/ABA changes, coordinated changes between sweeps, malicious systemd
metadata or host compromise. Unpinned digests are not signed manifests and
matching numeric IDs are not authenticated socket peer credentials.
`pidfd_open(MainPID)` is not an `SO_PEERPIDFD` acquired from an
authenticated IPC socket. The sweeps do not create an atomic snapshot and
cannot establish authority, genuine isolation, durable committed-close or
CCA persistence.

**Every output remains** `status=BLOCKED`,
`qualification=NOT_QUALIFIED`,
`production_emission=BLOCKED`,
`topology=PROPOSED_NOT_APPROVED` and CLI exit **2**, including when
both sweeps match. Nothing changes executable privileges, production signers,
keyrings, SIGN/BIND, enrollment, PostgreSQL SQL grants, service units,
Stage 10, CI timeouts, qualification gates or release authority.

### Validation and hardware

Synthetic tests cover changes in PIDs and starttime between apparently
internally consistent sweeps, unpinned digest changes, per-role error
isolation, malformed reports, observer exceptions, both-round cross-source
and cross-role contradictions, deterministic call counts, non-Linux refusal,
sanitized output and CLI exit 2.

Implementation and unit tests require no additional computer. An actual
host observation requires Linux/systemd and appropriate procfs visibility.
**Real process attestation / physical host qualification cannot be done
from these synthetic tests and must stay explicitly blocked.**
