# Stage 9 — S9-RPI-25: read-only cgroup-v2 membership observation

**OBSERVATIONAL_ONLY · NOT_QUALIFIED · PRODUCTION_NO_GO.**
This follows [S9-RPI-24](stage9_native_linux_process_identity_observation.md).

## Scope

Run `python -m deployment.stage9_native_linux_cgroup_membership_observation`
on a native Linux/systemd host. The probe reads only three fixed proposed
Source/CCA unit names, via a bounded, non-shell `systemctl --system show`
request, including untrusted `MainPID` and `ControlGroup` metadata.

The probe opens the candidate `/proc/<pid>` directory with
`O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC`; it reads bounded `stat` and
`cgroup` files through the directory descriptor. Both start-time and cgroup
data are read **twice**. A change causes an explicit diagnostic refusal.

The procfs parser accepts only a single unified cgroup-v2 entry in the
form `0::/absolute/path`. It rejects cgroup v1/hybrid, relative or malformed
paths and ambiguous prefix matches. A matching observation requires
equality with systemd's reported ControlGroup or child-directory
membership across a complete path component boundary. Cross-role
cgroup overlap or reuse of the same MainPID is reported as a failure.

Only fixed refusal codes, unit candidate names and bounded numeric
process metadata are reported; **raw cgroup paths and backend exception
messages are not emitted**. Injected synthetic readers cannot change
the production decision. No unit is installed, enabled, restarted or
mutated; no key material, environment or private service payload is read.

## Trust limitations

- `MainPID` and `ControlGroup` are sequential systemd observations, not
  independently pinned unit or executable provenance.
- The MainPID hint can race with PID reuse; double reads are not atomic.
- Cgroup namespaces may present namespace-relative paths, so a genuine
  service can yield `NOT_QUALIFIED` when paths differ.
- A membership match does not establish unit ownership, trustworthy
  pidfd-bound peer authentication, all descendant process identities,
  actual ACL/keyring isolation, or denial of alternate authority paths.
- It does not qualify genuine PostgreSQL committed close, durable Source
  or CCA work, rollback trust, SIGN/BIND, issuer or enrollment.

Report status is **always** `BLOCKED` / `NOT_QUALIFIED`,
`production_emission=BLOCKED`; the CLI returns exit code **2**.
Live data are labeled `UNTRUSTED_LIVE_OBSERVATION` and injected readers
are labeled `SYNTHETIC_TEST_INPUT`. No result is a production PASS.

## Hardware boundary

Hosted CI exercises synthetic parsers, refusal cases and local procfs
fixtures. A genuine host observation requires native Linux/systemd and
separate operator approval; **no additional machine is required merely
to implement or merge this observational step**. Physical isolation
qualification remains explicitly blocked. No changes to frozen
contracts, installed services, keyrings, SQL grants or Stage 10.
