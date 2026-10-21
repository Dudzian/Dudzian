# Stage 9 — S9-RPI-29: cross-observer consistency (diagnostic only)

**OBSERVATIONAL_ONLY · NOT_QUALIFIED · PRODUCTION_NO_GO.**

Follows [S9-RPI-28](stage9_native_linux_pidfd_lifetime_observation.md)
and its follow-up #3187. Prior steps observe individual, independently
untrusted surfaces. This step compares their **numeric PID hints** and
**procfs starttime ticks** for the same three fixed proposed Source/CCA roles.

## Run

`python -m deployment.stage9_native_linux_cross_observer_consistency`

On Linux the command invokes these five existing read-only observers
sequentially, once per source:

- process identity (S9-RPI-24)
- cgroup-v2 membership (S9-RPI-25)
- systemd unit fragment digest (S9-RPI-26)
- /proc/PID/exe digest (S9-RPI-27)
- pidfd lifetime metadata (S9-RPI-28)

The new module **does not reimplement any native syscall**. Each subordinate
report is checked against its exact expected schema, `BLOCKED` production
invariants, three expected role/unit names and a fixed success/refusal shape.
Unrecognized fields, untrusted backend exceptions or attempted "GO" promotion
are rejected. Test-injected observers produce `SYNTHETIC_TEST_INPUT` and
cannot produce a production qualification.

For each role, the comparison detects:
`CROSS_OBSERVER_MAINPID_MISMATCH` when numeric PID hints differ, and
`CROSS_OBSERVER_STARTTIME_MISMATCH` when available tick values differ.
The unit-fragment observation has no starttime, so it participates in PID
comparisons only. `CROSS_ROLE_SHARED_PID_HINT` additionally identifies
a numeric PID reused across proposed roles, even if none of the constituent
observers flags the collision. Missing/refused/unparseable observers and underlying
diagnostic anomalies are reported using **fixed reason codes** rather than
raw subreport error messages or stack traces.

Role-specific refusals and diagnostic anomalies affect only the named role.
Cross-role anomalies affect the named pair; global or unrecognized failure
labels affect all roles for that source. Numeric PID collision detection
remains independent of these labels. Validation follows the observers' bounds:
PID hints are integers from 2 to `2**31 - 1`, while starttime ticks and namespace
inodes are integers from 1 to `2**64 - 1`; booleans are rejected.

Output includes only fixed role names, a diagnostic comparison label,
bounded counts of observed sources, and fixed issue codes. **No file
digests, paths, environment, private metadata, observer error strings,
raw procfs contents, or potentially secret inputs are copied into output.**

## Not an attestation

The five observers run sequentially, including separate `systemctl` and
procfs operations. Even equal PID and starttime values are not a coherent,
atomic snapshot or proof that the same live process remained active. An
attacker can exploit PID recycling, ABA races, reused namespaces, dishonest
host metadata and the lack of socket-bound peer identity. Unexpected
collisions and subreport anomalies remain diagnostic refusals.

In particular, `pidfd_open(MainPID)` is **not** an authenticated
`SO_PEERPIDFD`, the unit and executable digests are not independently
pinned, and Python interpreter fingerprints do not authenticate imported
CryptoHunter source modules. No installation, service, permission, keyring,
signer, SQL grant, SIGN/BIND, enrollment or release gate is modified.
Production committed-close, durable CCA and cross-authority denial remain
unqualified.

**All results remain** `status=BLOCKED`,
`qualification=NOT_QUALIFIED`, `production_emission=BLOCKED`,
`topology=PROPOSED_NOT_APPROVED`; CLI exit **2** even when all hints
match. A positive comparison means only "no contradiction observed" and
must never authorize operational actions.

## Environment

The source-correlator and its injected-observer tests need no additional
hardware. A live host observation requires a Linux/systemd environment with
appropriate access to the already existing read-only probes. Genuine,
independently trusted physical Linux qualification remains a separate,
operator-approved task. **No additional computer is required to create or
merge this diagnostic-only change.**
