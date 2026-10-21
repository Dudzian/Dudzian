# Stage 9 — S9-RPI-33: rootless native Linux host capability triage

**READ_ONLY / UNTRUSTED / NOT_QUALIFIED / PRODUCTION_NO_GO**

## Purpose

S9-RPI-32 combines proposed Source/CCA service and identity observations.
A **fresh Linux test laptop** will normally have none of those proposed service
accounts, systemd units, configuration paths or protected source/CCA evidence.
That absence is expected; it cannot distinguish an unsuitable Linux test
environment from a host with the right kernel/user-space primitives but no
installation.

S9-RPI-33 adds a minimal **non-authorizing, host-self capability triage**.
It is independent of proposed service installation, PostgreSQL provisioning
and privileged setup. No capability result is a service qualification,
runtime isolation proof, attestation, or authorization to install anything.

From a reviewed repository checkout on an actual Linux computer:

\`\`\`bash
python -m deployment.stage9_native_linux_host_capability_triage
\`\`\`

The command requires no \`sudo\`, credentials, network, custom environment
variables or extra packages beyond the project's Python dependencies. It
accepts no selection paths, target PID, usernames or service names and
does not save or sign evidence. It always returns **exit status 2**,
including when all capabilities are observed. **Do not treat this exit code
as an ordinary test failure or change it to 0 for convenience.**

## Four independent fixed probes

| Capability | Read-only operation | What it does NOT establish |
| --- | --- | --- |
| \`procfs\` | Bounded read of the current process's \`/proc/<self-PID>/stat\` using directory-relative, no-follow access | Other process provenance, stable PID binding or protected host identity |
| \`namespaces\` | Read only the current process's mount, IPC and network namespace metadata | Isolation or cross-role access denial for Source/CCA |
| \`pidfd_self\` | Reuse the existing pidfd/procfs lifetime check on the **calling process** | Authenticated \`SO_PEERPIDFD\`, IPC peer credentials or another service's identity |
| \`systemd_manager\` | Bounded \`/usr/bin/systemctl --system show --property=Version\` query | Installed units, approved units, service ownership or effective sandboxing |

No target process beyond the calling process is opened. The probes do not
create test accounts, connect to databases, alter the service manager, install
software, touch keyrings, write system files, scan private directories, read
secrets or send network traffic.

The report exposes only four fixed capability names, fixed observation
labels and bounded reason codes. It never emits PIDs, kernel namespace
inodes, systemd versions, raw procfs data, paths returned from the host,
hostnames, personal accounts, environment variables or exception messages.
Probes are independent: one failure does not suppress other capability
observations. A malformed fixture or backend exception fails closed.

\`OBSERVED_UNTRUSTED\` means only that a **local check returned true at one
moment**; \`NOT_OBSERVED\` is an environment diagnostic, not proof that a
kernel feature does not exist. A restricted container, rootless session or
stopped systemd manager may legitimately return \`NOT_OBSERVED\`. Injected
testing backends use \`SYNTHETIC_TEST_INPUT\`, never a live label.

Regardless of observations, the envelope remains permanently
\`BLOCKED / NOT_QUALIFIED / production_emission=BLOCKED /
topology=PROPOSED_NOT_APPROVED\`. The host observation is not persisted as a
production trust pin and is never consumed by issuance, signing or enrollment.

## Initial Dell test-laptop sequence (after Linux installation)

1. Inspect the OS installation and run the capability command above **locally**.
2. Record its fixed-label output as *operator triage only*.
3. Run \`python -m deployment.stage9_native_linux_diagnostic_handoff\` to
   identify absent **proposed** accounts, units and state paths. Their absence
   on a clean laptop is expected, not a reason to weaken the preflight.
4. Only after separately reviewing a TEST-only isolation fixture design,
   deliberately provision temporary local test roles/services and perform
   physical denial tests. The capability probe does **not** perform that step.
5. Production host identity, independently approved install pins, effective
   access denial, native keyring isolation, PostgreSQL committed-close,
   CCA durability, rollback trust floor and legal enrollment remain separate
   qualification gates.

No SSH/Tailscale or headless/server conversion is part of this test-laptop
scope. Do not copy real secrets or live exchange trading credentials to the
machine for these diagnostics.

## CI and hardware boundary

Synthetic tests exercise all four fixed probe labels, independent failures,
malformed/exception results, command strictness, non-Linux refusal and CLI
exit 2. They do **not** establish a physical host capability. The tests
and coverage are registered in the existing Stage 9/Quality Ratchet scopes
with no new workflow, skip, runner, timeout or gate relaxation.

**A separate computer is NOT necessary to write/test this PR.** The user's
native Linux laptop is needed to run these probes on the actual intended
test host. Genuine effective isolation and production qualification require
additional reviewed installation/physical evidence, regardless of whether
these four capability hints are present.
