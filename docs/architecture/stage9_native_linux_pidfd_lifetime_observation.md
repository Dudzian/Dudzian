# Stage 9 — S9-RPI-28: read-only pidfd lifetime observation

**OBSERVATIONAL_ONLY · NOT_QUALIFIED · PRODUCTION_NO_GO.**

Follows [S9-RPI-27](stage9_native_linux_process_executable_observation.md).

## Observation

Run `python -m deployment.stage9_native_linux_pidfd_lifetime_observation`.
For the three fixed proposed Source/CCA unit names, the program obtains
`systemctl --system show` metadata with the bounded S9-RPI-25 runner and the
strict S9-RPI-24 `MainPID` parser. No service is started or changed.

On Linux hosts supporting `os.pidfd_open`, it obtains a pidfd for the
**untrusted numeric MainPID hint** and polls it with zero wait before and after
reading procfs status. Any event (exit/hangup/error) fails closed. It also reads
a bounded (4096-byte) `/proc/self/fdinfo/<pidfd>` record twice, requiring one
exact numeric `Pid:` field matching the hint. A directory descriptor for
`/proc/<pid>` is opened with `O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC`, and
the bounded `stat` file is inspected twice for unchanged starttime.
Descriptor cleanup is attempted on both success and failure; unsuccessful
cleanup is explicitly reported as a refusal.

No fallback to PID-only observation is allowed if pidfd opening or polling
is unsupported or denied. Both the process starttime and the pidfd
identity check can fail closed. Refusal codes distinguish
`PIDFD_OPEN_UNAVAILABLE` (opening the pidfd),
`PIDFD_FDINFO_UNAVAILABLE` (reading pidfd fdinfo),
`PIDFD_PROCFS_UNAVAILABLE` (opening or reading process procfs), and
`PIDFD_CLOSE_FAILED` (descriptor cleanup). They never include operating
system exception details. A failed close is a refusal, and cleanup still
attempts the other registered descriptors. The report retains fixed
malformed/missing codes and diagnoses duplicate `MainPID` hints across roles
even when a later pidfd read fails.

Synthetic regressions cover an exit event between the first and second
zero-timeout polls, a different `Pid` during the second fdinfo read,
starttime churn, and distinct I/O/close failures.

Only fixed role/unit names, bounded PID/starttime, a boolean indicating that
the pidfd was not signaled during both checks, and fixed failure codes appear
in output. Never emit fdinfo raw data, procfs contents, environment, executable
paths, private values or exception text. No changes to credentials or
native execution privileges.

## Strict trust boundary

- `pidfd_open(MainPID)` is **not** `SO_PEERPIDFD`. It did not come from an
  authenticated Unix socket peer, and it does not establish unit ownership.
- The numeric `/proc/<pid>` handle and the pidfd are **not atomically
  identity-bound**. PID reuse before `pidfd_open`, namespace effects, races
  between openings or after the final liveness poll, and ABA changes cannot
  be excluded by these observations.
- A successful zero-timeout poll means only that no exit event was reported
  in that instant. It is not a durable alive guarantee, peer attestation,
  service lifetime pin or trusted installation manifest.
- Real production qualification still requires authenticated socket-bound
  peer credentials, independently approved host, unit and executable pins,
  effective cross-role access denial, native keyring isolation, genuine
  PostgreSQL committed close and durable CCA.
- The source code must not use this report to grant production privileges.

**Every result** has `status=BLOCKED`, `qualification=NOT_QUALIFIED`,
`production_emission=BLOCKED`; CLI always exits **2** even when all
observations appear internally consistent. Source/CCA, signers, SIGN/BIND,
enrollment, SQL grants, service units, Stage 10 and release gates are unchanged.

## Hardware

Python unit tests and synthetic process-lifetime readers are feasible on
current CI and developer machines. An actual host observation requires Linux
with kernel support for pidfd, systemd, appropriate procfs visibility, and
operator approval. A physical production qualification is **not** part of
this PR. **No additional computer is needed merely to implement or merge
this diagnostic code.**
