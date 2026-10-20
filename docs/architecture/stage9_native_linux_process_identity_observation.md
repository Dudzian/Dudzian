# Stage 9 — S9-RPI-24: read-only Linux process identity inventory

**Observational only · PRODUCTION_NO_GO · NOT_QUALIFIED.**
This step follows [S9-RPI-23](stage9_native_linux_systemd_unit_observation.md)
systemd declaration checks. The new probe reads some **kernel-visible**
process facts, but it is not a deployment installer, peer credential
attestation or production qualification.

## What it reads

The closed executable entrypoint:

```bash
python -m deployment.stage9_native_linux_process_identity_observation
```

looks up only the three *proposed* NSS principals, then uses a fixed,
non-shell `systemctl --system show` query for their candidate units to
obtain an **untrusted numeric MainPID hint**. The unit query is bounded,
has a fixed locale and a 3-second timeout, and never invokes a systemd
mutator. The entrypoint accepts no user-chosen identity or unit.

For a candidate process the inspector opens `/proc/<pid>` with
`O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC`, reads bounded `stat` and
`status` via relative, `O_NOFOLLOW` opens, and observes mount, IPC
and network namespace inode numbers. It requires:

- the numeric PID hint to match the PID reported by `stat` and `status`;
- the same process start time before and after the inspection, no zombie;
- real, effective, saved and filesystem UID/GID to match the proposed
  service identity; supplementary groups to contain no foreign GID;
- `NoNewPrivs=1` and zero inherited, permitted, effective, bounding
  and ambient capabilities;
- observable namespace inode numbers, with no cross-role UID/GID or
  mount/IPC/network namespace reuse.

Missing NSS identity, missing/inactive unit, inaccessible procfs data,
permission denials, read failures, malformed procfs records, process
exit/restart or cross-role collisions produce explicit refusal reasons.
Only bounded numeric observations and fixed refusal codes are published;
no raw `/proc` content, executable/environment dump, private key or
backend error is emitted.

## Evidence boundaries (critical)

The `MainPID` lookup and `/proc` read are **not atomic**. A PID can
be reused between the systemctl query and opening procfs. Even when
the start time is stable during the two reads, this is **not proof**
that the observed process is the service intended by systemd, nor
an authenticated socket peer. It is intentionally weaker than
`SO_PEERPIDFD`-bound signer IPC, and must never replace that protocol.
The procfs mount, systemd daemon, NSS contents and host root are not
independently pinned by this tool. A process may also change credentials
after this snapshot. The tool does not attest all processes in a unit's
cgroup, other executable paths, namespace mappings, effective DAC/ACL,
socket peer identity or private keyring denial.

Report status is **always** `BLOCKED`, `NOT_QUALIFIED` and
`production_emission=BLOCKED`, CLI exit code **2**. An observed
snapshot is labeled `UNTRUSTED_LIVE_OBSERVATION`, and synthetic test
backends are explicitly labeled `SYNTHETIC_TEST_INPUT`. Even all
consistent process metadata remain observational-only. The
`MAINPID_IS_NOT_SOCKET_BOUND_PIDFD_OR_TRUSTED_PROCESS_PROVENANCE`
blocker is permanent.

Future genuine physical qualification must independently bind the
process to the installed, reviewed service unit, trusted executable,
protected installation/pin, pidfd lifetime and peer credentials;
prove source/CCA isolation from each other, both signers, the issuer,
all relevant privileged authorities, keys/sockets and alternate
pathways; and qualify PostgreSQL committed-close, durable source/CCA
logs, credential lifecycle and recovery.

**A native Linux/systemd host is REQUIRED for real observations.**
Hosted GitHub CI tests the parser and synthetic negative cases but
cannot provide trust qualification; no extra computer is necessary
merely to run those tests. Whether an approved isolation boundary needs
independent physical hardware is a separate deployment decision.

No changes to frozen contracts, canonical status,
`complete_close`/SQL grants, systemd services, native keyrings,
production Root Proof SIGN/BIND, enrollment or Stage 10.
