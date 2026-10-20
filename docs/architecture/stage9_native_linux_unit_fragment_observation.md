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

The observer opens `/` and each path ancestor with
`O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK`, keeping every
descriptor until the observation ends. Each child is inspected with `lstat`
and opened relative to its retained parent descriptor. Symlinks, non-root
ownership and group/other writability cause refusal. The fragment is opened
relative to the final parent with `O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK`;
it must be a root-owned, regular, non-group/other-writable, single-linked file.

For every opened object, pre-open `lstat` metadata must match `fstat` metadata:
device, inode, full mode (including file type), owner, group, link count, size,
`ctime_ns` and `mtime_ns`. The fragment descriptor is checked before the first
read, so an A → B → A path substitution cannot substitute B's digest for A's.
Only that descriptor is read; it is never reopened or reread for comparison.
Reads remain bounded to 131072 content bytes, with at most one extra byte to
detect growth past the limit. After hashing, the observer checks byte count,
fragment metadata, all retained directory descriptors and their parent/name
bindings again. `atime` is excluded because the read itself can change it.

Detected replacement, metadata mismatch, symlink or filesystem I/O failure
(including failure to close) yields the fixed refusal code
`FRAGMENT_CHANGED_DURING_OBSERVATION`, without a digest. Initial unsafe
directory/file permissions and invalid sizes retain their separate fixed policy
codes. All opened descriptors are registered for cleanup immediately; cleanup
attempts every close even if one fails. Successful observations include only a
SHA-256 digest and byte length.
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
- `systemctl FragmentPath` is untrusted host metadata. Descriptor binding
  refuses observed file/directory substitution and metadata churn, including
  the tested A → B → A open race. It does not exclude root/filesystem compromise,
  mount aliases or changes outside the observation window. The sequential
  checks are not an atomic filesystem snapshot and cannot prove the absence
  of every possible ABA change on an untrusted host/filesystem.
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
