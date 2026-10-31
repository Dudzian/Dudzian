# Stage 9 / 3A: inert package inventory V3

Baseline: `4e5044ac1bff8f078cb27659348f0b2e41b3a4ac`.
This is a separate, inactive package-only verification profile. It compares a
complete DEB with an explicit declaration; it does not authorize a source,
construct an Observer image, approve a contract or attest execution.

## Separate declaration and acceptance boundary

The existing `UbuntuMember`, `UbuntuPackage`, `ObserverManifestV1` and
`verify_offline` contracts retain their previous meanings. In particular, the
legacy adapter still refuses GNU headers and symlinks. V3 does not add optional
link fields to a legacy declaration, fabricate an empty image manifest or turn
package agreement into an image `OfflineResult.MATCH`.

The V3 document has the explicit format
`stage9-ubuntu-package-inventory-v3`. Its package entries bind the package name,
version, architecture, archive filename, full SHA-256 and size to separate
`control` and `data` declarations. Each TAR declaration selects exactly one
profile, `gnu-inert-v3` or `ustar-inert-v3`, and supplies the full decoded TAR
size/hash and every member record. Unknown versions, profiles, keys, missing
fields, invalid types and duplicate records refuse; booleans are not integers.

`parse_package_inventory_v3(raw, limits=...)` validates a declaration and returns
the separate `UbuntuPackageInventoryV3` records. The public
`verify_ubuntu_package_inventory_v3(input_directory, inventory_raw=..., limits=...)`
performs the complete batch measurement. Its result has scope
`INERT_PACKAGE_INVENTORY_ONLY` and consistency `MATCH`, `MISMATCH` or
`INCOMPLETE`; it provides no image or production authorization capability.

Every record declares its raw name, typeflag, size, mode, UID/GID, mtime,
uname/gname, literal linkname, magic/version, raw device fields and raw bytes
345–511 (`tail_hex`). Every regular payload additionally has a SHA-256. A
directory or symlink has size zero and no regular-file digest. The parser also
checks the package identity in the `control` payload.

Both inventories are exact, including root `./`, explicit parent directories,
empty directories and maintainer scripts as inert bytes. Extra or missing
members refuse regardless of type. Directory records are not inferred from
regular-file ancestors or the host filesystem. Control and data remain separate
namespaces; package boundaries do not prohibit different packages from declaring
the same directory.

## TAR header profiles

The GNU marker is exactly `7573746172202000`; the POSIX USTAR marker is exactly
`7573746172003030`. A TAR declaration selects one profile for every header in
that stream, so mixed markers refuse. Checksums use the unsigned byte sum with
the checksum field treated as eight spaces.

The profiles have different layouts after byte 344. POSIX USTAR uses bytes
345–499 as a name prefix and bytes 500–511 as reserved padding. The old GNU
layout instead assigns:

| Byte offsets, zero based, end exclusive | GNU field |
| --- | --- |
| 345–357 | atime |
| 357–369 | ctime |
| 369–381 | offset |
| 381–385 | longnames |
| 385–386 | unused |
| 386–482 | four sparse entries |
| 482–483 | isextended |
| 483–495 | realsize |
| 495–512 | padding |

The layout is checked against GNU tar's `struct oldgnu_header` and its explicit
POSIX-prefix conflict comment in the
[pinned implementation source](https://github.com/gnu-mirror-unofficial/tar/blob/5c713540e34f51323b9f63b7459254bc845603ed/src/tar.h#L131).
This URL is an unofficial mirror of GNU implementation source, not an accepted
trust anchor. Acquisition details and source digests are retained in the
[separate V3 observation](evidence/stage9_observer_3a_real_deb_observation_v3.md).

GNU extension bytes must never be concatenated with `name` as a POSIX prefix.
V3 deliberately admits no GNU extension mechanism: all bytes 345–511 must be
NUL in the `gnu-inert-v3` profile. This is a restricted profile, not general GNU
TAR support. The complete raw tail remains in each declaration so a previously
unrecorded extension cannot be silently accepted as zero or ignored metadata.

Both profiles validate every header, numeric encoding, metadata field, text
termination, payload and padding. They admit only regular files, directories and
literal symlink records. Unsupported numeric encodings, GNU longname/longlink,
PAX local/global headers, sparse/continuation records, hardlinks, devices, FIFOs
and other types refuse. Text after a terminating NUL cannot contain hidden data.
The full decoded-stream size/hash, two zero end blocks, block alignment, zero
tail, decompressor EOF/checksum and exact ar range termination must all agree.
An early end marker does not excuse an unread or nonzero tail.

Numeric header fields use unsigned ASCII octal digits with the explicitly
accepted surrounding space/NUL padding; base-256 and signed encodings refuse.
Modes are limited to `0000`–`0777`, UID/GID to `0`–`2^31-1` and mtime to
`0`–`2^33-1`. Device fields must encode zero (an all-NUL field is also accepted).
The GNU name and literal target fields are each bounded to 100 UTF-8 bytes;
USTAR's combined prefix/name is bounded to 256 bytes. A regular typeflag may
be `0` or NUL, and its exact representation is compared with the declaration.
The JSON declaration is bounded to 2 MiB with conservative parser/record
allowances charged to the shared working-memory ceiling before construction.

## Symlinks, names and image closure

Member names are constrained relative archive names. The declared raw name is
retained for comparison, while the namespace check removes the permitted leading
`./` and directory trailing slash. Duplicate canonical names, traversal,
absolute member paths and nondirectory prefix conflicts such as symlink `a`
with `a/child` refuse in either order. Required directories must be explicit.

A symlink target follows a different rule: it is bounded literal data, never an
instruction to access the host. An absolute `/etc/python3.12/sitecustomize.py`,
`..` components, a dangling target, a cycle, a chain or a cross-package target
can be compared as declared text without resolving it. The verifier does not
create symlinks, open/stat their targets, call `resolve` or `readlink` on them,
extract members, run maintainer scripts or execute/import investigated code.
No target lookup is required to establish package inventory agreement.

Package-only records have no `file_id` or image projection. They cannot fill a
missing image directory, become a regular-file provenance record or establish
Python import closure. The existing image validator still refuses filesystem
symlinks and requires its closed image inventory. Archive UID/GID/mode are
compared as package metadata, not applied to a staging filesystem. Installation
transformations and archive-to-image metadata mapping remain separate V5 work.

## I/O, resource and tool boundaries

The public package path hashes the whole pinned archive before parsing it and
rechecks descriptor metadata and the named binding after reading. The selected
directory is opened through the existing descriptor ancestor walk; archive
opens do not follow symlinks, and nonregular, multilink or cross-device input
files refuse. Verification runs unprivileged and supports Linux descriptor I/O.

The DEB profile still requires exactly `debian-binary`, `control.tar[.gz/.zst]`
and `data.tar[.gz/.zst]`, with DEB version `2.0`. Unsupported ar extensions and
extra records refuse. V3 reuses the bounded transport decoder rather than
installing a package, invoking a tool or falling back to an alternate parser.

The supported Zstandard combination remains python-zstandard `0.25.0`, C
extension backend and embedded libzstd `1.5.7`. No dependency is fetched during
verification or added to the minimal Observer runtime. These compatibility
checks do not authenticate an imported module, native binary, interpreter,
loader or operating system; the [V2 toolchain limits](stage9_observer_offline_v2.md)
still apply.

Control, data and subsequent packages share the invocation's unpacked-byte and
member budgets. Whole-archive input bytes, retained declarations, names, literal
targets, lookup structures, read buffers and decoder allowance are included in
the resource limits. Regular payloads are hashed in bounded reads. A refusal,
truncated read, timeout, native error or unsuccessful cleanup cannot produce a
partial successful result, and cleanup cannot replace an earlier refusal.
Diagnostics expose controlled reason codes rather than raw native error text.

Memory accounting is conservative and cooperative, not a hard RSS limit.
Deadline checks cannot interrupt a blocked filesystem syscall or native call.
Descriptor snapshots detect some changes; they do not provide complete TOCTOU
protection or guarantee immutability after return. Trusted host/tool acceptance
and stronger isolation remain independent requirements.

## Evidence and reproduction

The historical V2 inventory and results remain unchanged. V3 uses a new
[frozen declaration](evidence/stage9_observer_3a_real_deb_inventory_v3.json) that
adds the raw header tails missing from the prior observational record. Its
SHA-256 is `aafaae5853b06b70f704a39bc6471cced92207b9ff64055b97a870f60dc10137`.
The separate observation found all 1,002 GNU header tails NUL and reproduced
the earlier member values and complete TAR hashes. This supports the narrow
all-NUL extension profile for these samples; it does not generalize to GNU TAR.
Expectations are retained separately before production-verifier
comparison; the reproduction tool has no acquisition or expectation-update
mode. The declaration remains
`SAME_ORIGIN_OBSERVED_COMPARISON_ONLY; NOT_INDEPENDENT_ACCEPTANCE` because the
same upstream artifacts supply the observations and comparison inputs.

From a trusted checkout and Python environment, after separately provisioning
the exact local DEBs, run:

```sh
python -m scripts.verify_stage9_real_deb_v3 --directory /path/to/exact-debs
```

The selected directory must contain exactly the declared archive inputs; there
is no single-package selection or inventory-update mode. Exit zero means only
complete package inventory agreement. A failed or incomplete comparison exits
two. Production qualification remains refused even when package consistency is
`MATCH`.

The intended real-package inventory contains 871 regular data members, 103 data
directories and seven symlinks, plus 17 control regular members and four control
directories: 1,002 records in total. Root directories are included in those
counts. Counts are a cross-check; acceptance compares every declared record.
The earlier eight-frame / 52,684,800-byte V2 observation is not a V3 parser
result. Actual V3 experiment outcomes must come from a new measured run.

The [separate V3 verification report](evidence/stage9_observer_3a_real_deb_verification_v3.md)
now records that new run: four whole-package inventory matches, 1,002 records
and 52,684,800 decoded bytes. It preserves the same-origin classification and
all source/image/production refusals.

Whole-archive hash/size agreement, exact control inventory, exact data inventory,
metadata, regular payloads, literal symlink records and decoder completion are
separate measured obligations. Even complete agreement does not authenticate
APT, independently approve expectations or establish image correspondence.

## Validation and unchanged refusals

The new modules and read-only script enter the existing configured mypy
Linux/Win32, explicit Ruff lint/format, branch-coverage and 70% changed-code
gates. The new negative-path tests join the existing quality test inventory;
all earlier V1/V2 tests remain enabled. CI uses synthetic fixtures without
downloading or installing Ubuntu packages. The all-files pre-commit format
gate is still mandatory before publishing or updating a PR. Thresholds,
timeouts, native platform and hardware opt-in gates are unchanged.

Local security/dependency validation used separate tool environments without
changing the repository test environment or dependency locks:

- Semgrep CE `1.179.0`: the existing five blocking rules passed with no findings;
  all five rule fixtures passed. The maintained community configuration remained
  **BLOCKED** because the enforced proxy refused `semgrep.dev` with CONNECT 403;
  it is not reported as a successful scan.
- Betterleaks `1.9.0`: its release archive matched both the workflow's pinned
  SHA-256 and the published release checksums. The unchanged, offline working-tree
  command passed with zero findings. An initial scan found only ignored generated
  private-key-fixture bytecode in `tests/__pycache__`; the disposable file was
  removed with authorization and the same scan rerun. No source finding, scanner
  rule, ignore or baseline was suppressed.
- pip-audit `2.10.1` under Python `3.11.17`: the canonical desktop lock's 104
  dependencies and the separately hash-pinned Zstandard dependency each had zero
  known vulnerabilities and zero skipped dependencies. No automatic dependency
  fix was used. `pip check` passed in the unchanged repository environment.

The blocking and fixture commands match the workflow:

```sh
semgrep scan --config .semgrep/cryptohunter.yml --metrics=off --disable-version-check --error bot_core core deployment scripts
semgrep scan --test --config .semgrep/cryptohunter.yml .semgrep/cryptohunter.py --metrics=off --disable-version-check --jobs 1
betterleaks dir --baseline-path .betterleaks-baseline.json --redact=100 --no-banner --no-color --report-format json --report-path /path/to/report.json .
python -m pip_audit --requirement deploy/packaging/requirements-desktop.lock --format json --output /path/to/desktop-audit.json
python -m pip_audit --requirement deployment/stage9-offline-verifier-cp311-manylinux-x86_64.lock --format json --output /path/to/zstd-audit.json
python -m pip check
```

Local writable cache/log paths were used because the execution environment's
home directory is read-only. Proxy configuration and TLS verification were
preserved. A vulnerability scan reports known advisories, not independent tool
acceptance or proof that no vulnerability exists. Final PR CI and its checked-out
commit remain separate evidence.

V4 must address APT provenance; V5 must address archive-to-image mapping.
Independent acceptance, runtime closure, trusted launch, scope authentication,
receiver/replay protection and physical qualification remain later gates.
V3 performs no Dell qualification or Stage 3B activation. Enrollment/macOS and
Windows remediation remain separate changes.

```text
SOURCE_AUTHENTICITY=UNVERIFIED
ARCHIVE_TO_TARGET_METADATA_MAPPING=NOT_YET_PROVEN
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
