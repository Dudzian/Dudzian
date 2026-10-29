# Read-only real-DEB V2 diagnostic boundaries

This review covers `scripts/verify_stage9_real_deb_v2.py` against the
`_InputReader` implementation at PR #3227 input HEAD
`0ff0235a02dbf81f52544cc97c05aeaddee1373c`. It is a source review and synthetic
regression exercise, not approval of the real packages, interpreter or decoder.

The diagnostic now uses the same descriptor-based ancestor walk and complete
metadata snapshot as the main reader. Every ancestor is opened without following
symlinks. Archive files must be regular, have exactly one link and reside on the
pinned directory's device. Descriptor and pathname snapshots include device,
inode, mode, owner, group, link count, size, modification time and change time.
The directory descriptor and its original pathname are checked again before any
result is returned. Running as root refuses before inventory or archive I/O.

Native decoder failures use the adapter's existing controlled reason codes.
Successful frame measurement requires EOF and successful cleanup. A second
cleanup failure does not replace an earlier read or budget refusal. The command
prints a controlled reason code instead of filesystem paths, archive-controlled
text or raw native diagnostics.

Remaining differences and limitations:

- This command measures selected frozen package declarations. It does not run the
  main reader's closed image-tree inventory scan or validate target UID/GID/mode
  mapping. Extra files in the download directory do not acquire any authority.
- The frozen inventory digest establishes agreement with the declaration shipped
  in this checkout. It is not independent acceptance or APT authentication.
- Decoder version/backend checks select a supported API combination. An injected
  Python module can report those same values. They prove neither module origin
  nor authenticity of the native extension or its dependencies. Run from a
  separately trusted checkout, interpreter and import environment. There is no
  import sandbox or new root of trust in this diagnostic.
- Descriptor checks and final metadata comparisons bound the observed races;
  they do not establish a filesystem snapshot or protect against a hostile kernel,
  privileged host administrator or filesystem that misreports metadata.
- Deadlines and memory accounting are cooperative. A blocked kernel call or
  native decoder is not preempted by an OS-enforced sandbox.
- GNU/PAX TAR, symlinks and other unsupported archive records remain refused.

The focused synthetic suite checks root refusal, symlinked ancestors, initial
hardlinks, concurrent link/mode changes, file replacement, directory replacement,
descriptor release, native read/close failures, preservation of the first failure
and sanitized diagnostics. Its synthetic complete match still leaves every
qualification and authenticity block in place. These tests do not measure a real
Ubuntu archive or qualify a production installation.

```text
stage_9=IN_PROGRESS
qualification=BLOCKED
CONTRACT_APPROVED=NO
STAGE_3A_COMPLETE=NO
PRODUCTION_QUALIFIED=NO
production_status=PRODUCTION_NO_GO
runtime_attestation=MISSING
host_pid_namespace=UNVERIFIED
V2_COMPLETE=NO
```
