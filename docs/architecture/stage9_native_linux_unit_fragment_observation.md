# Stage 9 — S9-RPI-26: observational systemd unit-fragment fingerprints

**OBSERVATIONAL_ONLY · NOT_QUALIFIED · PRODUCTION_NO_GO.**

This follows [S9-RPI-25](stage9_native_linux_cgroup_membership_observation.md).
Unlike the earlier systemd-property, process-identity and cgroup inspections, this
module observes **bytes of the unit fragment** itself, without changing any
unit, installation or enrollment state.

## Read-only command

`python -m deployment.stage9_native_linux_unit_fragment_observation`

The program inspects only the three fixed candidate Source/CCA systemd units.
A bounded non-shell `systemctl --system show` request retrieves `MainPID`,
`FragmentPath` and `DropInPaths` alongside the existing fixed unit properties.
The only permissible fragment paths are the exact proposed unit filenames under
`/etc/systemd/system`, `/usr/lib/systemd/system` or `/lib/systemd/system`.
Unexpected drop-ins, unrecognized fragments, inactive/wrong-user unit properties
or malformed replies cause refusal.

The observer checks each path ancestor with `lstat`, refusing symlinks,
non-root ownership and group/other writability. The fragment is opened
`O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK`; it must be a
root-owned, regular, non-group-writable, single-linked file.
Reads are bounded to 131072 bytes. The opened file's before/after metadata,
actual byte count and path metadata are compared to detect some concurrent
changes. A SHA-256 digest and byte length are included in the output.
**File bytes, directives, environment variables, command arguments and raw
paths are never published.** No private keys or runtime process memory are read.

The systemd runner comes from S9-RPI-25 and enforces a three-second timeout,
a maximum 8192-byte stdout stream and child cleanup. Unit files are not
installed, modified, reloaded, started or enabled.

Injected readers and synthetic tests are marked `SYNTHETIC_TEST_INPUT`.
A matching live fingerprint is called `UNPINNED_FRAGMENT_DIGEST_ONLY`,
never `QUALIFIED`.

## Strict limitations

- SHA-256 alone does **not** authenticate installation provenance.
  The expected digest is neither independently approved nor stored in
  an operator-controlled trust anchor.
- `systemctl FragmentPath` is untrusted host metadata. Root/filesystem
  compromise, mount aliases, concurrent file replacement and ABA changes
  are not excluded. The sequential inspections are not atomic.
- Refusal of symlinked `/lib` or otherwise noncanonical host layouts
  is a conservative diagnostic decision; no host topology is auto-approved.
- This module does **not** attest the runtime `/proc/<pid>/exe`, command-line,
  executable hash, loaded unit bytes in systemd memory, worker cgroups,
  socket-bound pidfd credentials, installed service identity or effective
  access denial. Those require separate work and genuine host evidence.
- It cannot qualify independent PostgreSQL committed-close, durable CCA,
  source execution, signer/issuer trust, SIGN/BIND or enrollment.

Every result remains **`status=BLOCKED`, `qualification=NOT_QUALIFIED`,
`production_emission=BLOCKED`**; CLI exit code **2**. The tool cannot
approve a fragment fingerprint, accept a caller-supplied trust anchor
or promote an observation to production evidence.

## Environment

Pure tests and synthetic failures can be developed on existing GitHub-hosted
Linux CI. An installed native Linux/systemd host is required for genuine
observations, and independent physical qualification remains a separate,
operator-approved step. **No additional computer is required to implement
or merge this diagnostic PR.** No frozen contracts, installed units, credentials,
database grants, release gates or Stage 10 behavior change.
