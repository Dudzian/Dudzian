# Stage 9 / 3A: bounded offline artifact verifier V1

Historical V1 scope is recorded below. The
[V2 Zstandard extension and real-package limitations](stage9_observer_offline_v2.md)
supersede its compression-support statements; GNU headers and links still refuse.

Implementation baseline: `d88d601cb200ad5da08fd0de9ddf7064cbfbe76b`
(main including PRs #3224 and #3225). This is an advisory, unprivileged library,
not a Trusted Observer, launcher, Stage 3B collector or production trust backend.
Existing contracts and their standard refusing entrypoints are unchanged.

## API and evidence boundaries

`deployment.stage9_observer_offline.verify_offline(input_directory, *,
manifest_raw, policy_raw=None, launch_policy_raw=None, acceptance_raw=None,
trust_context=None, ubuntu_packages=(), limits=Limits())` reads an explicitly
selected absolute directory. A manifest image path `/observer/file.py` maps to
`input_directory/observer/file.py`; package/proof paths are explicitly relative
to the same directory. The caller supplies document bytes separately. Image
files and directories must occur explicitly in the manifest; image ancestors
are never inferred. Package/proof artifacts and their necessary transport
directories occupy a separate namespace outside the image. The complete input
tree must equal those two inventories. It must not contain a second copy of
metadata unless explicitly declared as ordinary data under the existing
contract rules.

The module reuses `parse_proposed_manifest`, `parse_proposed_acceptance_policy`,
`parse_launch_policy`, their existing cross-binding validator, the shared
canonicalizer and `parse_offline_result`. Proposed-profile parsing remains
`PROPOSED_UNAPPROVED`. Internal frozen dataclasses are invocation arguments and
diagnostics; they do not introduce another wire or acceptance schema.

| Output | Meaning |
| --- | --- |
| `structure=VALID` | Supplied documents passed the applicable existing parsers; absent documents are not invented. |
| `artifact_consistency=MATCH` | The complete staged tree and all supported source/member projections agree with caller declarations. |
| `MISMATCH` | Measured bytes, size, metadata, source ID or package version disagree. |
| `INCOMPLETE` | Missing records/files, drift, a resource ceiling, unsupported source/archive or failed read prevents a complete comparison. |
| `source_authenticity=UNVERIFIED` | Hashing opaque proof bytes does not authenticate APT signatures, keys, indexes or freshness. |
| `acceptance=UNVERIFIED` | No independently enrolled trust/approval backend exists. Any supplied trust context or acceptance envelope is refused; parsing a signature never verifies it. |
| `execution=MISSING` | No evidence that approved bytes executed, and no host PID namespace authentication. |

Every outcome has `refused=True`: this describes continued refusal of
authorization, including when the narrower byte comparison is `MATCH`.
Missing trust adds `TRUST_CONTEXT_MISSING`; unsupported variants use
`UNSUPPORTED_SOURCE`; missing evidence uses `PROVENANCE_INCOMPLETE`;
changes use `FILE_CHANGED`; ceilings use `LIMIT_EXCEEDED`. Diagnostics expose
only the existing fixed codes, never input paths or exception text.

Only when manifest, policy and launch documents were actually parsed and their
bindings checked is an `OfflineVerificationResultV1` produced. Its three digest
pins derive from those exact supplied bytes. It retains `synthetic=true` because
the existing advisory wire requires that field; this does not imply that file
I/O was simulated. Missing documents yield `report=None`, never placeholder pins.

## Closed directory inventory correction

The audit of PR #3226 HEAD
`65c844fb785034e04883aed038bb4349bc3c2f37` identified a P1 inventory gap:
the offline adapter automatically added staging ancestors to its expected tree
without requiring image directory records. A nested image path such as
`/image-parent/observer` could therefore admit an unrecorded `image-parent`
directory, including changed permissions. Package/proof paths placed inside
`observer/not-in-manifest/` could also introduce undeclared image entries.
Byte equality did not establish directory closure in those layouts.

The existing contract already supplies the relevant boundary:

- [Schema proposal, section 3.3](stage9_observer_3a_schema_proposal.md#33-files-provenance-and-image-ancestry)
  requires a non-`/` image root, explicit in-image directory ancestry and no
  unused image directories. It explicitly excludes host ancestry above the
  image root from invented image records.
- [`_validate_proposed_runtime`](../../deployment/stage9_observer_contract.py#L1715)
  requires every other manifest entry to lie strictly below the declared image
  root, every intermediate parent to be a declared directory, and the actual
  declared directory set to equal the required set. Thus an ordinary missing
  parent inside the image was already refused by structural parsing.
- [The base plan's metadata requirements, section 4](stage9_trusted_observer_contracts_and_implementation_plan.md#4-docelowy-start-closure-i-ochrona-przed-podmian%C4%85)
  distinguish metadata, provenance and closed inventory checks; they do not
  turn a locally generated parent into an expected image record.

The offline verifier now constructs image and transport inventories separately.
Every image parent up to the input-directory boundary must be an explicit
manifest directory; transport scaffolding cannot satisfy this condition.
Consequently this V1 staging layout supports an image root immediately below
the input directory. A nested image root with an undeclared staging ancestor
refuses with `CLOSURE_INCOMPLETE`; the verifier does not extend the unchanged
parser to accept extra records above the image root.

Only explicitly named archive/proof files may introduce transport ancestors.
Those files and directories must remain outside the image namespace and cannot
collide with image entries. Scans still require the complete input tree to
match the combined inventories. Every admitted image directory is checked
against its own manifest type, UID, GID and mode; identity and binding checks
continue to cover the input tree. The selected input directory and its host
ancestors are read boundaries, not newly authenticated image objects.

This correction does not add archive-to-installation metadata semantics.
USTAR type/name/size/checksum and source-member bytes are checked; its UID, GID
and mode fields are not currently interpreted or bound to target metadata.
Target UID/GID/mode are checked independently against manifest declarations.
The [existing transform catalogue](stage9_observer_3a_schema_proposal.md#34-transforms)
preserves bytes for `materialize_member` but has no metadata-mapping fields;
directories require `source_member` provenance and cannot be transform outputs.
Neither this absence nor byte equality authorizes an ownership or permission
change. All transforms remain unsupported in this offline implementation.
An accepted archive-to-target metadata mapping therefore remains a P2 evidence
gap requiring separate definition and review. `MATCH` does not establish that
such a transformation was permitted, performed or independently accepted.

## Reads, identity and resource limits

Linux-only V1 refuses execution as UID 0. Each directory component is opened
using directory descriptors and `O_NOFOLLOW`; regular files additionally use
`O_NONBLOCK`, then `fstat` type checks. Symlinks, hardlinks, cross-device entries,
FIFOs, devices and sockets refuse. Streaming scans before and after measurement
check the closed tree. Descriptor/name/root binding and device, inode, type,
mode, owner/group, link count, size, ctime and mtime are compared around reads.
Target metadata must agree with the manifest; directories have logical size 0.

These checks detect observed changes. They are **not complete TOCTOU
protection**, immutable snapshots, lifetime integrity or verified/executed
binding. No claim is made about a file after return or undetectable changes
between observations.

Local `Limits` are stricter implementation budgets, not a changed acceptance
policy schema: 4096 tree entries, 64 MiB per regular file, 256 MiB measured
bytes, 8192 TAR headers across packages, 256 MiB unpacked TAR bytes across
packages, 64 KiB read chunks, 32 path components, 60 seconds and a 32 MiB
conservative allocation allowance by default. Values may tighten and cannot
exceed the existing proposed ceilings; chunks smaller than 512 bytes refuse.
TAR zero padding counts against unpacked bytes, control and data share a budget,
and gzip output is capped at 100 times its compressed region. Fixed-size reads,
streaming enumeration and retained-record preallocation accounting bound the
work admitted by this implementation. A ceiling failure refuses the entire
measurement, rather than returning a partial match.

Memory accounting includes parser-input expansion allowances, tree snapshots,
archive record allowances and decoder buffers. It is not an OS RSS limit or a
qualification measurement of the verifier runtime. Deadlines use a monotonic
clock around bounded operations and cannot interrupt a blocked filesystem
syscall. A separately reviewed worker/resource-isolation design is still
required for hard termination and process-memory guarantees. No helper,
privilege change or launcher is introduced here.

## Restricted Ubuntu adapter

`deployment.stage9_observer_ubuntu` contains a standard-library DEB reader.
`UbuntuPackage` declares source ID, relative archive/proof paths, package name,
exact version/architecture and a complete regular-data-member inventory.
`UbuntuMember` binds member path, size and SHA-256 to a manifest file ID;
empty file IDs cover package members outside the staged image. Every regular
member must be declared, every projected file must bind once, and source-linked
manifest directory members must exist as TAR directories. The interpreter's
manifest version is bound to its package source.
Other data directories must be necessary ancestors of declared regular members;
unreferenced empty directories refuse as unexpected additions.

V1 permits only the enumerated CPython package family for
`3.12.3-1ubuntu0.17`, `amd64`. DEB must contain exactly `debian-binary` (`2.0`),
`control.tar[.gz]` and `data.tar[.gz]` in order. TAR must be USTAR with ordinary
regular files/directories, valid checksums, zero padding and complete EOF.
Control fields must match package/version/architecture. Maintainer scripts
are inert bytes which are read and discarded. Nothing is extracted or written.
Links, devices, duplicate/traversing names, GNU/PAX/sparse records, extra ar
members, XZ and Zstandard refuse the whole measurement.

Existing repository wheel/TAR helpers were reviewed: they buffer complete
archives/member tables or deliberately select partial metadata and do not
provide this closed, bounded DEB behavior. No external DEB tool was added to
the verifier's future TCB. Git/wheel/transform verification, APT signature/index
verification, cryptographic approval, rollback/time policy, ELF/import content
semantics and `.pyc` reconstruction remain unimplemented and refuse.

The earlier exact-build evidence remains a comparison source, not an accepted
root. A development check downloaded the actual `python3.12-minimal`
`3.12.3-1ubuntu0.17` amd64 DEB: 2,334,634 bytes,
SHA-256 `d452689b9660845345a4c3e05e4ad82c082d5474e04031b7aa47f1a6d5610a6e`,
matching the prior evidence. Its `control.tar.zst` and `data.tar.zst` correctly
produce `UNSUPPORTED_SOURCE`. This check did not validate a complete package
inventory, APT authenticity or acceptance. Raw/gzip success is currently
demonstrated on synthetic DEBs containing real staged fixture bytes.

## Security review and merge boundary

Tests exercise all 18 requested negative scenarios plus corrupt DEFLATE,
directory provenance, inode replacement, aggregate package budgets and malformed
invocation records. They assert explicit refusal results and unchanged blocks.
Real filesystem reads and hashes are exercised; fixtures simulate package
expectations, control metadata, mutations, clock expiry and resource ceilings.
No physical Dell tests, target imports, execution, installation or privileges run.

| Severity | Outstanding boundary |
| --- | --- |
| P0 | None established in the changed source review and local negative tests; this is not a repository-wide proof. |
| P1 | Independently accepted expectations/roots, APT authentication, complete runtime image and execution/host attestation remain absent. They block activation and production qualification. |
| P2 | Only raw/gzip USTAR is supported; actual Ubuntu Zstandard packages refuse. Archive-to-target UID/GID/mode mapping is not defined or verified. Cooperative time/allocation accounting does not supply hard worker isolation; metadata rechecks are not full TOCTOU protection. |
| P3 | The current offline wire requires the historical `synthetic` label even for actual byte reads; consumers must retain the documented advisory meaning. |

Conditional merge is appropriate only for this limited, inactive library after
the exact PR HEAD passes applicable CI and independent code review accepts the
documented limits. It does not approve contracts, complete Stage 3A, qualify
physical execution or authorize deployment. Codebase Memory MCP/executable was
unavailable; conclusions were checked against the actual named source files,
not a structural-index completeness assertion.

```text
stage_9=IN_PROGRESS
qualification=BLOCKED
CONTRACT_APPROVED=NO
STAGE_3A_COMPLETE=NO
PRODUCTION_QUALIFIED=NO
production_status=PRODUCTION_NO_GO
runtime_attestation=MISSING
host_pid_namespace=UNVERIFIED
```
