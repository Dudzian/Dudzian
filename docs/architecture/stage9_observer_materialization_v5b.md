# Stage 9 / Observer 3A — bounded offline member materialization V5b

Baseline: `6d019c97a8287db29013e0e2f68213f7f9e386bf`, main after PR #3272.

[Executed validation and preflight](stage9_observer_materialization_v5b_validation.md)
record the baseline repairs, tests, scanner results and remaining limitations.

V5b compares an explicitly selected archive symlink with a regular source
member and an independently declared regular image file. Its output is an
inert agreement result. It does not create an installation tree or write
materialized files.

```text
Stage 9 = IN_PROGRESS
Observer 3A qualification = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO

External authority acceptance = UNVERIFIED
Authenticated time = UNVERIFIED
Durable anti-rollback = BLOCKED
Runtime qualification = BLOCKED
```

## Contract boundary

The [V1 contract](../../deployment/stage9_observer_contract.py) defines
`materialize_member` with exactly one input and requires equal input/output
SHA-256 and size. An archive symlink and its regular target are two distinct
records. V5b therefore uses its own `symlink_to_regular_v5b` operation,
versioned declarations and `MaterializationResultV5b`. It does not reinterpret
V1's operation or claim to validate a V1 manifest. Standard Observer parsers
remain inactive.

The [V5a contract](stage9_observer_image_v5a.md) remains unchanged. Its complete
static mapping profile still refuses symlinks and transformations. V5b reuses
its bounded declaration, exact source-identity, full-record ID, direct mapping
and optional APT evidence checks. It adds its own closed record accounting,
symlink operation and verification of actual source bytes. It does not invent
a successful V5a result, discard symlinks to manufacture a complete V5a input,
or treat a partial static mapping as complete.

The original V3 package allowlist, version, architecture, filename, TAR record
and control identity restrictions continue to apply. V5b does not broaden the
package profile because an unhandled link occurs in an Ubuntu archive.

## Public operation and immutable input

The [materialization verifier](../../deployment/stage9_observer_materialization_v5b.py)
accepts these inputs:

```python
verify_member_materialization_v5b(
    *,
    inventory_raw: bytes,
    expected_image_raw: bytes,
    mapping_raw: bytes,
    expected_image_sha256: str,
    archives: tuple[FrozenPackageV5b, ...],
    package_result: PackageInventoryResultV3 | None = None,
    apt_evidence: AptEvidenceV5a | None = None,
    limits: Limits = Limits(),
) -> MaterializationResultV5b
```

`FrozenPackageV5b(filename, content)` contains an exact immutable `bytes`
object. Bytearrays, lazy readers and paths are not interchangeable with that
object. The [source adapter](../../deployment/stage9_observer_member_source_v5b.py)
checks the complete DEB's filename, size and SHA-256 against its V3 declaration
before interpreting its contents. Every declared archive is checked, including
archives without a selected symlink source. Extra and missing archives refuse.

The adapter validates the strict DEB/ar framing and both complete control/data
TAR streams, using the public V3 TAR verifier. It checks control identity and
retains selected regular payloads only after the archive verification succeeds.
Each retained payload must match the exact selected V3 record's size and
SHA-256. Record verification and payload capture use the same immutable DEB
contents, so replacing a filesystem path after capture cannot substitute bytes
between verification and use. Comparing two supplied hashes or supplying a
caller-created `MATCH` cannot replace this read.

The optional Linux snapshot helper reads a closed, explicitly selected input
tree through the existing no-follow descriptor reader. It checks file and
directory identity before/after reading, enforces the declared complete file
set and produces immutable archive copies. A change during this read refuses
the entire capture. These checks are cooperative filesystem observations;
they do not establish trusted custody or protection against every hostile
kernel/filesystem behavior. The subsequent verifier still checks the captured
bytes against the independently selected archive digest.

The initial adapter accepts uncompressed TAR and gzip TAR inside DEBs.
Zstandard archive capture is explicitly unsupported in this narrower adapter;
V3's existing Zstandard support is unchanged. Native snapshot handling is
separate from portable immutable-byte verification.

The `package_result` parameter defaults to `None` so that omission produces a
structured refusal. A successful V3 result is required for continuity with
V5a; its complete package measurements must equal the adapter's fresh measured
values. It remains an additional caller claim and cannot authorize skipping
fresh archive-byte checks. Optional V4 association keeps V5a's repository,
snapshot and complete trust-context checks; supplying stronger V4b/V4c claims
requires their explicit selected context. Neither an APT match nor a Python
result object grants authority acceptance.

## Independently selected expected image

The expected document has the following exact fields, with V5a's optional
`apt_context` extension:

```text
{
  format: "stage9-observer-expected-image-v5b",
  image_id,
  snapshot_sha256,
  sources: [SourceIdentity, ...],
  files: [ImageFile, ...],
  materializations: [ExpectedMaterialization, ...]
}

SourceIdentity = {
  source_id, package, version, architecture, filename, sha256, size
}

MemberReference = {source_id, record_id, member}

ExpectedMaterialization = {
  file_id,
  link: MemberReference,
  source: MemberReference,
  metadata_policy: "source-record-exact-v1"
}
```

`ImageFile` retains V5a's independently specified regular/directory fields,
including path, type, size, UID, GID, mode and regular-file SHA-256. Every parent
directory and `/` must be explicit. Each materialized output additionally
selects its exact link and regular source independently of the mapping.
The source table selects the exact package identity, and both documents select
one identical snapshot. Missing or competing origins refuse.

`expected_image_sha256` is a separate caller-supplied pin for the exact expected
document bytes. The mapping's embedded image pin and the measured document
digest must equal that pin. The transformation never generates an approved
expected hash from the payload it has just read. Caller control of all inputs
can still create self-consistent claims; independent acceptance of expectations
is explicitly unverified.

## Mapping declaration and record accounting

The mapping uses a separate format:

```text
{
  format: "stage9-observer-materialization-v5b",
  snapshot_sha256,
  inventory_sha256,
  expected_image_sha256,
  sources: [SourceIdentity, ...],
  directory_policy: "identical-explicit-v1",
  entries: [SourceMember | TransformOutput, ...],
  transforms: [SymlinkMaterialization, ...]
}

SourceMember = {
  kind: "source_member", source_id, record_id, member, file_id
}
TransformOutput = {kind: "transform_output", transform_id, file_id}

SymlinkMaterialization = {
  transform_id,
  kind: "symlink_to_regular_v5b",
  link: MemberReference,
  source: MemberReference,
  output_file_id,
  metadata_policy: "source-record-exact-v1"
}
```

The optional `apt_context` must appear identically in expected image and
mapping when selected, following V5a's unchanged requirements. Pins cover the
exact input JSON bytes rather than a reserialization.

Member references preserve V3's literal `name`. V5a's existing
`record_id_v5a(record)` identifies every field of the complete `TarMemberV3`,
including its literal spelling, type, metadata, link text and payload digest.
Lexical lookup uses a separate canonical name and never rewrites that evidence
identity. A similar filename or matching payload hash cannot supply a missing
reference.

Every `data.tar` record has exactly one accounting consumer: a direct mapping,
or the link input of exactly one materialization. A regular source remains
accounted through its own direct mapping and may also be a dependency of more
than one independently declared link. Reusing a source dependency does not
consume or hide its archive record. Control TAR members remain fully verified
in their package namespace and are never executed.

Every expected output has exactly one complete origin. Only V5a's existing
identical explicit directory-sharing policy permits multiple package directory
origins. Regular-file collisions, duplicate output use, missing records,
unmapped extra TAR members and extra image outputs refuse. Package ordering
never chooses a winner.

## Lexical symlink profile

Only symlinks in `data.tar` with a literal relative target are supported. The
explicit source reference must select a regular `data.tar` member in the same
exact package. Its canonical path must equal the target resolved relative to
the link's archive parent.

Resolution inspects archive data only. It never calls `Path.resolve()`, reads
host symlinks, follows local Python import paths, creates links, or examines
host `/usr`, `/lib` or `/etc`. Components are checked individually. Absolute
targets, empty components, backslashes and control characters refuse.
Lexical `.` and `..` handling remains inside the archive root, with each
traversed intermediate node required to be an explicitly declared directory.
In particular, `regular/../target` cannot bypass a non-directory component.
Escaping the root refuses; normalization cannot turn an unsupported traversal
into an authorized host lookup.

The final target must be the explicitly referenced regular member. Directories,
dangling links, links to links, cycles, cross-package references and ambiguous
aliases refuse. No recursive chain resolver or inferred package search exists.
The final image path equals the link's archive path. The output payload is
byte-for-byte the captured regular source payload; relocation and content
transformation are outside the profile.

`source-record-exact-v1` explicitly selects source regular-file UID, GID and
mode as the required output metadata. These must also equal the independently
declared image metadata. A symlink's mode does not silently become a regular
file mode. If the image requires different source metadata, the result records
an unsupported `metadata_change` requirement and refuses; it does not repair
permissions or ownership.

## Dependencies, ordering and result semantics

The verifier constructs explicit package, member, transformation and image-file
dependencies. The provenance binds the package, original archive link, regular
source and expected output. Graph nodes use these exact names and prerequisites:

```text
package:<source_id>                       no prerequisites
member:<source_id>:<record_id>            package:<source_id>
symlink:<transform_id>                    link member AND regular source member
expected:<file_id>                        symlink:<transform_id>
expected:<direct_file_id>                 every explicit contributing member
```

Dependency edges retain their exact source/record/output identities. The
returned `dependencies` tuple sorts node IDs and each node's dependencies.
`resolution_order` uses iterative topological ordering with a lexical min-heap
for ready nodes; changing declaration order cannot choose a different origin or
change the order. Both member prerequisites precede their symlink operation,
which precedes its expected output. Missing nodes or cycles refuse. Shared
regular sources are explicit shared dependencies. Names alone cannot introduce
provenance, and inputs from another snapshot cannot satisfy a graph edge.

`MaterializationResultV5b` is versioned separately from V5a. `MATCH` means only
complete agreement of the supported bounded materialization operations and the
declared image. Direct mapped files, materialized files and their graph/evidence
bindings are returned only after the entire operation succeeds. Every refusal,
interruption, exhausted budget or late failure discards all partial output and
successful evidence bindings. It does not leave earlier mapped files as a
partial positive result.

Successful bindings include exact inventory, expected-image, mapping and
snapshot digests, the validated V3 claim digest and optional APT claim digest.
`source_bytes_sha256` is domain-separated with
`b"stage9-v5b-source-bytes\x00"` and covers the canonical JSON list of actual
source-content bindings: transform ID, source ID, full record ID, literal
member, archive digest/size and measured payload digest/size. This digest
records an association with freshly checked bytes; it is not a signature or
authority attestation. Refusals clear all these successful binding fields and
return empty `materialized_files`, `mapped_files`, `dependencies` and
`resolution_order`.

`MATCH` does not mean installation, interpreter execution, V1 manifest approval,
import closure, authority acceptance, Stage 3B readiness or production
qualification. All statuses shown above remain blocked or unverified.

## Bounds and limitations

V5b uses the existing `Limits` model and V5a declaration/package ceilings. A
shared verifier invocation budget covers input/declaration bytes, retained immutable
copies and payloads, records, output files, transformations, paths, lexical
depth, graph comparisons and cooperative elapsed time. Capturing full immutable
archives and verified TAR data can refuse an otherwise V3-readable archive
when the unchanged memory budget is insufficient. The operation never raises a
limit to admit that archive.

Immutable archives may already be allocated by the caller; the verifier still
charges their retained bytes to its own budget. The optional filesystem
snapshot helper is a separate operation with its own budget by default and
accepts an explicit existing budget for capture accounting. There is no public
combined capture-and-verification wrapper that claims a single deadline across
both calls.

Accounting bounds retained buffers and operations conservatively. It is not an
OS RSS/CPU sandbox, cannot preempt a blocked kernel read, and does not establish
the trustworthiness of the Python process, native decompressors or host. An
archive digest pin detects substitution against the selected bytes; it does
not prove who selected that digest. These limits do not weaken the existing
production, native-platform or security gates.

## Unsupported transformation registry

The following remain refusal conditions or separate unresolved requirements:

| Requirement | V5b disposition |
| --- | --- |
| Absolute/escaping/dangling/cross-package link | Refuse |
| Link to directory, another link or cycle | Refuse |
| Hardlink, device, FIFO or other special member | Refuse through the existing V3 profile |
| Relocation, source metadata change or content rewriting | Refuse; no implicit correction |
| Zstandard member capture | Unsupported in the initial immutable adapter |
| Python bytecode generation or compilation | Separate future substage |
| Maintainer scripts or package installation | Unsupported; never execute |
| `dpkg` defaults, `ldconfig`, alternatives or diversions | Unsupported; no inferred transformations |
| Runtime/import closure, production image construction | Unimplemented; qualifications stay blocked |
| Production trust, durable checkpoints or authenticated time | Unimplemented; statuses stay blocked |

V5b does not mark the whole of V5 complete. A declared image requiring any
remaining transformation cannot receive `MATCH` from this profile.

## Future `.pyc` substage

Bytecode generation requires its own reviewed operation and independently
selected expected outputs. Before implementation that contract must specify:

1. The exact source record and verified source bytes, package and snapshot;
   source encoding, source size and any input timestamp must be explicit.
2. The exact CPython executable/tool bytes, build identity, version, bytecode
   magic and complete runtime/tool dependency closure. Host Python cannot be an
   implicit substitute for the selected compiler.
3. Optimization level, compiler flags, `co_filename`, output path/cache tag and
   path normalization, including their effect on reproducibility.
4. The precise invalidation mode and header values. Checked-hash,
   unchecked-hash and timestamp modes have different trust and reproducibility
   requirements; no mode can be silently substituted.
5. An independently declared expected `.pyc` size and SHA-256, explicit target
   UID/GID/mode and unique provenance. A generated hash cannot approve itself.
6. Deterministic execution isolation, input/output/memory/time limits,
   interruption behavior and disposal of every partial result on failure.
7. Explicit accounting for every generated output and unsupported operation,
   plus positive and negative integration tests on the selected platform.

This is a requirements register only. V5b does not compile source, generate
bytecode, execute the investigated CPython or authorize that future work.

## Validation boundary

Portable tests inspect real synthetic DEB bytes and independent image
expectations. Linux integration additionally uses the public V3 and V4a
operations on an inert synthetic archive chain. Such tests establish behavior
for those fixtures; they do not establish a native Windows filesystem result,
a real Ubuntu package's complete installation semantics or physical runtime
qualification. Skipped tests and queued CI are recorded separately from passes.
The separate validation report records executed commands and remaining gaps.
