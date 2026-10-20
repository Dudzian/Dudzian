# Stage 9 — S9-RPI-27: read-only /proc/PID/exe object fingerprint

**OBSERVATIONAL_ONLY · NOT_QUALIFIED · PRODUCTION_NO_GO.**

Follows [S9-RPI-26](stage9_native_linux_unit_fragment_observation.md).
This step does not alter any service or installation. It observes the
kernel-reported executable object for the three fixed proposed Source/CCA
systemd MainPID hints and deliberately does not compare that object with
an independently approved executable digest.

## Invocation and data boundaries

Run `python -m deployment.stage9_native_linux_process_executable_observation`
on a native Linux host. The fixed non-shell systemd request uses the same
three-second, bounded-stdout runner as S9-RPI-25 and the existing exact-unit,
active-state, user/group and numeric MainPID parser from S9-RPI-24.

For each candidate PID, the observer opens `/proc/<pid>` with
`O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC` and reads bounded procfs `stat`
before and after executable hashing. Linux's `/proc/<pid>/exe` is a
**kernel-generated magic symlink**. It must be followed for reading executable
bytes; this is not a general permission to follow arbitrary filesystem links
or caller-supplied paths.

It checks the metadata reported by `stat("exe", dir_fd=proc_fd)`
against `fstat(exe_fd)`, hashes that one opened descriptor, then rechecks
`fstat(exe_fd)`, `stat("exe", dir_fd=proc_fd)` and process starttime.
The object must be a regular, root-owned, root-group, non-group/other-writable
file with at least one hard link. Reads are bounded to 64 MiB plus one
detection byte; empty and oversized executables are refused.

Only a lowercase SHA-256 digest, bounded file size, PID hint, start ticks,
fixed role/unit candidate and fixed refusal codes may appear in the report.
No file paths, executable contents, command lines, environment variables,
memory contents or exception strings are included. Synthetic backends
are labeled `SYNTHETIC_TEST_INPUT`. All opened fds are registered for
cleanup; failures are reported as fixed error codes.

## Explicit non-claims

- A MainPID from systemd is an **untrusted hint**, not an authenticated
  unit-owned process. The digest is not an independently pinned or signed
  executable and cannot approve installation or execution.
- The observed executable object is **not** proven to be the unit's
  configured `ExecStart`, systemd's loaded configuration, any child worker,
  an interpreter script, imported Python module, dynamic library, or
  mapped executable page. Inspecting `/proc/PID/exe` for a Python process
  normally identifies the Python interpreter, not the CryptoHunter source.
- Checks are sequential, not an atomic snapshot. PID reuse before opening,
  ABA changes, `execve` changes after observation, compromised procfs or
  kernel/host authority, and races outside the observation window remain.
- No pidfd lifetime pin, socket-bound peer credentials, independently
  approved unit/manifest digest, cross-role effective access proof,
  native keyring denial, genuine PostgreSQL committed close or durable CCA
  operation has been established.
- The 64 MiB limit is deliberately conservative. A legitimate executable
  exceeding it produces `NOT_QUALIFIED`, not a bypass or permission to
  increase the limit silently.

Every result has `status=BLOCKED`, `qualification=NOT_QUALIFIED`,
`production_emission=BLOCKED` and CLI exit **2**, even when all three
untrusted executable observations succeed. No changes to signers, enrollment,
SIGN/BIND, SQL grants, keyrings, production contracts or Stage 10.

## Required host work

This PR can be implemented and tested using existing code-only resources and
synthetic Linux procfs fixtures. Actual host observations require a genuine
native Linux/systemd installation. Even those observations are insufficient
for production qualification; independent pinning, pidfd/authenticated peer
binding and operator-controlled physical trust evidence remain necessary.
**No additional computer is required merely to develop or merge this step.**
