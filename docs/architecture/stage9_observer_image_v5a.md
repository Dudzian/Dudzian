# Stage 9 / Observer 3A — offline archive-to-image mapping V5a

Baseline: `5d56a05192436b8772ab0fb73bbeac0d3ec91a00` (main after PR #3271).

[Local validation and limitations](stage9_observer_image_v5a_validation.md)
record the executed checks and unavailable native dependencies.

The separate [V5a module](../../deployment/stage9_observer_image_v5a.py)
compares complete frozen [V3 inventories](stage9_observer_offline_v3.md) and
associated measurements with a separately supplied expected static image. It
implements a bounded direct-mapping profile. It does not construct an image,
install a package, execute a transformation, or accept an expectation authority.

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

## Contract separation and compatibility

The archive inventory describes literal package contents. The expected-image
document describes a selected installation tree. The mapping document declares
the relationship between those two inputs. A package measurement describes a
previous local comparison. None of these roles authenticates another role or
demonstrates a host installation or executed runtime.

V5a has its own versioned input and result contracts. It does not change
`ObserverManifestV1`, `ProposedContract`, `OfflineResult`, V3 inventories, or
V4a/V4b/V4c results. A separate expected static tree is needed because the
[proposed V1 manifest](stage9_observer_3a_schema_proposal.md) also requires
runtime-role and import closure; even a one-file mapping fixture is meaningful
without inventing such closure. The static tree is not a complete V1 manifest,
an accepted projection of one, or an adapter that activates standard parsers.
Standard Observer parsers and their existing refusals remain unchanged.

The V5a-only `apt_context` extension below binds stronger supplied V4b/V4c
claims to a separately selected repository and full trust context. Existing
V4a-only inputs without that extension remain supported. No V1–V4 contract or
result schema changes.

V3's package allowlist, exact CPython package version, `amd64` architecture,
canonical DEB filename and complete control/data TAR profiles remain in force.
Unsupported package families need separate reviewed contract work. Historical
operator evidence and the [November reconciliation](stage9_observer_3a_reconciliation_2026-11-13.md)
are not rewritten to make them satisfy this new contract.

## Public operation

```python
verify_image_mapping_v5a(
    *,
    inventory_raw: bytes,
    expected_image_raw: bytes,
    mapping_raw: bytes,
    expected_image_sha256: str,
    package_result: PackageInventoryResultV3 | None = None,
    apt_evidence: AptEvidenceV5a | None = None,
    limits: Limits = Limits(),
) -> ImageMappingResultV5a
```

All three raw inputs are supplied before comparison. There is no discovery,
acquisition, manifest-generation, expectation-update or filesystem output mode.
The separate `expected_image_sha256` argument pins the exact expected document
bytes. Replacing that document and merely adjusting the mapping's embedded pin
must fail against the separate argument. Supplying both matching pins is still
a caller declaration; it does not prove independent acceptance or independent
administration of the documents.

A complete `MATCH` requires a matching V3 `package_result`. A missing or refused
measurement cannot be replaced by a self-consistent declaration. Optional V4
evidence is checked if supplied; its absence is recorded separately. Python
result objects, including objects whose fields claim success, do not prove
that a verifier was run or that its environment was trusted.

## Expected-image declaration

The top-level record has these mandatory fields and the optional extension below:

```text
{
  format: "stage9-observer-expected-image-v5a",
  image_id,
  snapshot_sha256,
  sources: [SourceIdentity, ...],
  files: [ImageFile, ...]
}
```

The optional additional field `apt_context` has exactly this shape:

```text
{repository_id, snapshot_id, context_sha256}
```

Repository and snapshot labels follow V4b's existing label syntax, and the
context digest is a nonzero lowercase SHA-256. This field must appear in both
the independent expected image and mapping, with identical values, or in neither.
It is required when V4b, V4c or a full trust context is supplied. The stronger
profile also requires the associated V4b signature evidence and full context;
an expectation pin alone cannot replace missing evidence.

Every source is identified explicitly by the exact record:

```text
SourceIdentity = {
  source_id, package, version, architecture, filename, sha256, size
}
```

`sha256` and `size` identify the whole DEB. The source table must correspond
exactly to the V3 package set and the mapping's source table. A source cannot
be omitted, reused under another ID, replaced with another package version,
or added without a matching V3 package. Names and identifiers are literal;
there is no guessing from a filename or similar package/member name.

Each image file is exactly:

```text
Directory = {file_id, path, type: "directory", size: 0, uid, gid, mode}
Regular   = {file_id, path, type: "regular", size, uid, gid, mode, sha256}
```

File IDs and absolute POSIX paths are unique. The root `/` and every parent
directory are explicit entries. Paths use the existing lexical rules: no
traversal, empty or dot components, repeated separators, backslashes,
trailing slash except `/`, or control characters. Files cannot contain children.
Symlinks and all other final-image record types are outside the profile.
Numeric metadata is declared independently of the executing host. Booleans
are not accepted as integers. Modes follow the proposed image's restricted
permission profile rather than granting permissions from local account state.

There is no implicit directory creation, inferred metadata, generated expected
hash, or hidden image member. In particular, constructing expectations only by
enumerating the same DEB and then comparing that enumeration to itself does not
constitute independent image verification. Tests construct expected paths,
metadata and content independently from literal fixture requirements.

## Mapping declaration and record identity

The mapping top-level record has these mandatory fields and the same optional extension:

```text
{
  format: "stage9-observer-image-mapping-v5a",
  snapshot_sha256,
  inventory_sha256,
  expected_image_sha256,
  sources: [SourceIdentity, ...],
  directory_policy: "identical-explicit-v1",
  entries: [SourceMember | TransformOutput, ...],
  transforms: [Transform, ...]
}
```

The mapping permits the same optional `apt_context` field defined above.
Its values must equal the independently pinned expected-image selection.

`inventory_sha256` covers the exact supplied V3 JSON bytes;
`expected_image_sha256` covers the exact supplied expected-image bytes. A
different serialization has a different byte identity. The independent image
pin, the mapping's image pin and the measured expected document must agree.
Both documents select one identical `snapshot_sha256`; the per-source
identities must also agree in full.

A direct entry is exactly:

```text
SourceMember = {kind: "source_member", source_id, record_id, member, file_id}
```

V3 has no member IDs. `record_id_v5a(record)` defines V5a's stable full-record
digest, including the literal member spelling, exact TAR typeflag, metadata,
payload digest, linkname, header marker and tail. References always include
the source ID and literal `member` spelling as well. Altering any of these
values cannot silently select another record. A record is not identified by
its content hash or a similar basename alone.

The exact record-ID preimage is:

```text
SHA256(b"stage9-v5a-record\x00" + canonical_json_bytes(all_TarMemberV3_fields)).hexdigest()
```

All fields of the frozen `TarMemberV3` dataclass participate, including `name`,
`typeflag`, `size`, `mode`, `uid`, `gid`, `mtime`, `linkname`,
`magic_version_hex`, `uname`, `gname`, `devmajor`, `devminor`, `sha256` and
`tail_hex`. `sha256` is `null` for directory and symlink records in this
internal record-ID preimage. The helper does not read or authenticate a TAR.

Every `data.tar` record must be referenced exactly once, including the literal
root directory. Every expected-image entry must be accounted for. Control TAR
records stay in the package measurement namespace; maintainer scripts are
never installation instructions. The mechanism does not discard additional
data members merely because their paths look irrelevant to the runtime.

For direct mapping, V3's canonical `.` root maps to `/`; every other canonical
member name maps to exactly `/` plus that name. The raw TAR spelling is still
checked in the reference. This limited lexical projection neither resolves
host paths nor authorizes relocation. Each direct record must agree with the
independent image's type, SHA-256 for regular files, size, UID, GID and mode.
A metadata change, relocation or content change is not applied automatically.

### Shared directories

`identical-explicit-v1` permits multiple packages to contribute a directory
only when each contributing record has its own complete direct reference,
all references select the same declared directory, and type, size, UID, GID
and mode agree exactly with each other and the expectation. This applies to
the root directory as well as paths such as `/usr`.

No regular-file merge, winner selection, package-order rule or unconditional
directory deduplication exists. Different directory metadata, a regular-file
collision, a directory/file conflict or an ambiguous source reference refuses.
The policy is a static equality profile; it does not predict how a particular
installer resolves real shared-directory metadata conflicts.

## Symlinks and transformation registry

V3 can observe literal symlinks; V5a cannot put them into the final image or
materialize their targets. An external, cross-package, dangling, chained or
cyclic target remains literal archive data. V5a does not call `readlink`,
`resolve`, create links, copy targets, traverse chains or infer a replacement
from filename similarity. Any unresolved input symlink prevents complete
`MATCH`, even if all regular-file hashes happen to agree.

A transformation output reference is exactly:

```text
TransformOutput = {kind: "transform_output", transform_id, file_id}
Transform = {
  transform_id,
  kind: "materialize_member" | "pyc",
  inputs: [{source_id, record_id, member}],
  output_file_id
}
```

Transform IDs, inputs and outputs must resolve unambiguously. A transform
output must select its declared regular file, and a reference cannot be used
to hide an omitted or multiply mapped record. Valid structure establishes
neither permission to execute nor agreement of a reconstructed output.
These are V5a's limited declarations, not substitutes for V1's richer tool,
parameters and reconstruction-proof contracts.

| Operation | V5a disposition | Required later work |
| --- | --- | --- |
| Direct regular file or directory with identical destination semantics | Compare in V5a | Independent expectation acceptance remains external |
| Symlink to regular-file materialization | Unsupported; no complete mapping | Explicit source/target semantics and separately approved V5b transform |
| `materialize_member` | Validate references, refuse execution | Reviewed tool/proof/metadata contract and independent output verification |
| `pyc` | Validate references, refuse execution | Pinned interpreter/tool, parameters, invalidation/filename semantics, independent expected hash |
| UID/GID/mode change | Unsupported transformation | Explicit reviewed metadata rule and verification |
| Relocation, generated configuration, maintainer-script effects, alternatives/diversions | Unsupported | Separate bounded semantics; no installer execution as a fallback |

A direct UID/GID/mode disagreement returns `MISMATCH` with
`ARTIFACT_MISMATCH` and `unsupported_transformations=("metadata_change",)`.
A direct destination change likewise returns `MISMATCH` with
`ARTIFACT_MISMATCH` and the `relocation` marker. These markers identify
required work; they do not authorize or execute a correction. Content hash,
size and record-type differences remain mismatches against the expectation.

## V3/V4 identity binding and trust

V3 measurements must match the complete package set, archive hashes and byte
counts, control/data TAR profiles, hashes, sizes and member-type counts. The
control identity must remain tied to the declared package identity. A positive
result for another package set, modified TAR inventory or a different archive
cannot satisfy the mapping.

Optional `AptEvidenceV5a` supplies these named fields:

```text
inventory_sha256, snapshot_sha256, declaration, result,
signature_result=None, acceptance_result=None, trust_context=None
```

`declaration` is an existing `AptDeclarationV4a`; `result` is an existing
`AptResultV4a`. Its inventory pin, Release hash and full V3 measurements must
agree with this invocation. Optional V4b and V4c results must bind to the same
chain, repository, snapshot and selected full context; they cannot introduce
measurements from another package set or snapshot or upgrade a refusal.
If supplied materials disagree, V5a refuses rather than
combining their individually positive fields.

The appended `trust_context` is an existing `TrustContextV4b`. In the stronger
profile, V5a computes its digest with the existing
`context_sha256_v4c(context)` contract after bounded inspection and compares
that digest with `apt_context.context_sha256` from both documents. The complete
context includes repository and snapshot IDs, Release and keyring hashes,
signer/algorithm rules, time rules, version, validity/revocation fields and
external-authority references. The context's repository ID, snapshot ID and
Release hash must also equal the selected labels and `snapshot_sha256`.
Matching Release bytes alone do not bind a different repository or context.

The existing V4b `_signer` policy checks are reapplied to the supplied GPG
evidence under this bound context, including signer pairs, algorithms,
revocation and time rules. This is a comparison against caller-supplied rules
and time, not a new signature verification or authenticated-time operation.

V4b's successful local signature/policy claims still carry their fixed external
authority, freshness, checkpoint, rollback, tool and isolation blockers. V4c's
successful technical validation similarly retains its external-authority,
authenticated-time, durable-rollback and runtime blockers. V5a expects these
existing blocker fields rather than treating their removal as stronger evidence.
It does not rerun `gpgv` or V4c signature verification.

The V4b claim must have the exact existing six-code blocker tuple and exact
`GpgvEvidenceV4b` and `AptResultV4a` types. V4c must have its exact four-code
blocker tuple and an exact `SnapshotStateV4c` that passes the existing state
validator, is not revoked and selects the same snapshot, repository and computed
context digest. `SnapshotStateV4c` has no snapshot-ID field: that ID is carried
by the full context, whose digest must match the state and independently pinned
selection. A label claiming success cannot bypass these checks.

V5a does not independently pin checkpoint IDs or checkpoint hashes here. Two
otherwise valid states for the same selected context and snapshot can have
different checkpoint identities. This binding establishes neither an exact
accepted checkpoint nor durable anti-rollback state.

With V4a evidence, `snapshot_sha256` denotes the selected Release bytes.
Without that evidence, it is only a caller-selected, pinned snapshot identity;
agreement cannot establish real repository membership or authenticated
same-snapshot provenance. Caller objects and these explicit associations
remain unverified observations. V5a performs no signature verification,
authority enrollment, authenticated-time operation or durable checkpoint
write, and never mutates the supplied V4 results.
`snapshot_membership` remains `UNVERIFIED` for every outcome, including a
static `MATCH` with a bound V4 claim.

## Results, atomicity and limits

`ImageMappingResultV5a` is separate from archive, APT and Observer-runtime
results. `MATCH` means complete equality in the supported static profile.
`MISMATCH` means compared identities or content disagree. `INCOMPLETE` means
the profile, evidence, structure, closure or available resources cannot support
a complete result. Failure reasons are controlled codes rather than raw input
or exception text.

`ARTIFACT_MISMATCH`, `SNAPSHOT_MISMATCH`, `PATH_COLLISION`,
`REPOSITORY_MISMATCH` and `CONTEXT_BINDING_REJECTED` produce
`MISMATCH`. Other controlled refusals produce `INCOMPLETE`, including
`FORMAT_REJECTED`, `LIMIT_EXCEEDED`, `UNSUPPORTED_SOURCE`,
`PROVENANCE_INCOMPLETE`, `CLOSURE_INCOMPLETE`, `FILE_POLICY_REJECTED` and
`UNSUPPORTED_TRANSFORMATION`. A full context rejected by the existing V4c
context-hash validator produces `CONTEXT_REJECTED` and `INCOMPLETE`.
A cooperative keyboard or I/O interruption
produces `OPERATION_INTERRUPTED`; allocation failure produces `LIMIT_EXCEEDED`.
These additional V5a diagnostics use a separate exception and leave V1's
closed reason-code enum unchanged.

Successful results retain exact SHA-256 identities for the raw inventory,
expected image and mapping, the selected snapshot, the supplied V3 result and,
when present, the supplied APT evidence. Measurement/evidence claim hashes
include the bounded complete dataclass fields; tuples become arrays and bytes
become hex text for hashing. They identify supplied claims, not authenticated
verifier transcripts. `v4_binding` is `NOT_PROVIDED` or `CALLER_CLAIM_BOUND`.
No result fields are added for the context extension: raw document hashes cover
the complete `apt_context` selections and `apt_evidence_sha256` covers the
complete supplied trust context. Every refusal still clears these bindings.
Each returned `MappedFileV5a` carries its file ID, path, type, content hash or
`None`, size, UID, GID, mode and every `MappedOriginV5a` source/package/version/
record/member reference.

Any refusal discards every staged mapped-file result and successful evidence
binding. A late failure,
unsupported transformation, limit breach or interrupted operation cannot return
a partially positive image. `mapped_files` is empty and successful input/evidence
pins are absent. Authorization gaps and unfinished requirements
remain separate from consistency, including after `MATCH`. Package-result
origin, expectation acceptance, host state, execution, runtime closure and
production qualification remain unverified or blocked.

For every outcome, `source_authenticity`, `evidence_origin`,
`external_authority_acceptance`, `acceptance`, `authenticated_time` and
`host_state` and `snapshot_membership` remain `UNVERIFIED`; `durable_anti_rollback`,
`runtime_qualification` and `qualification` remain `BLOCKED`. Execution remains
`MISSING`, `STAGE_3A_COMPLETE=NO`, `PRODUCTION_QUALIFIED=NO`,
`production_status=PRODUCTION_NO_GO` and `refused=True`. The result profile is
`stage9-observer-image-result-v5a` and its scope is
`STATIC_ARCHIVE_TO_IMAGE_ONLY`. `unsupported_transformations` records the
recognized unsupported operation kinds, and `unmet_requirements` retains
external authority, authenticated time, durable rollback, runtime qualification
and host-state blockers. A missing V4 association adds `V4_NOT_PROVIDED`; a
refusal adds `STATIC_MAPPING_INCOMPLETE`.

V5a reuses `Limits` and cooperative `_Budget` accounting. It bounds document
bytes and depth before decoding; package, member and image counts; total paths
and path lengths; retained input/metadata/index structures; comparison work;
and elapsed time. Dictionary-based source/member/path indexes and bounded
ancestor walks avoid unbounded all-to-all record comparisons. The accepted
range is the existing frozen V3 package profile; a limit breach refuses.

The fixed declaration cap is 2 MiB for each of the three raw documents, with
maximum lexical nesting depth 16 checked before decoding. The V5a package/source
cap is 128 and is narrowed by the V3 allowlist and `max_files`. Image paths are
at most 4096 UTF-8 bytes and have at most `max_path_depth` components; archive
names retain V3's narrower GNU/USTAR bounds. Both TAR namespaces count against
`max_archive_members`; the image counts against `max_files`; their sum is
bounded by those two ceilings together. The expected regular-file total cannot
exceed `max_unpacked_bytes`, and each file obeys `max_file_bytes`.

All raw input bytes share `max_total_bytes`. Preflight reserves 20 times the
raw byte length before decoding; V3's parser separately charges its existing
16-times raw allowance and retained record costs. Image rows, lookup records,
returned origins and bounded evidence serialization receive additional shared
reservations. Evidence strings/bytes are capped at 65,536 characters/bytes;
arrays are bounded by `max_archive_members`, dataclasses by 64 fields and
nesting by 16. Serialized dataclasses must have exact allowlisted types:
`AptEvidenceV5a`, `AptDeclarationV4a`, `AptResultV4a`, `AptResultV4b`,
`TrustContextV4b`, `SignerRuleV4b`, `TimeRulesV4b`,
`ExternalAuthorityReferenceV4b`, `AptResultV4c`, `SnapshotStateV4c`, `GpgvEvidenceV4b`,
`PackageInventoryResultV3`, `PackageMeasurementV3` and `TarResultV3`.
Arbitrary dataclasses and subclasses are rejected; their field descriptors
cannot introduce unbounded names or user-defined serialization behavior.
The known field names are also included in memory reservations.
Before calling the existing V4c context-hash helper, V5a serializes the full
context through this bounded known-type traversal, requires at most 65,536
canonical bytes and reserves a further 20 times that byte length for the
helper's dataclass, tuple/list and canonicalization copies. A larger or
resource-exhausting context refuses before the helper is called.
Comparison calls cannot exceed
`(max_archive_members + max_files + 128) * (max_path_depth + 32)`.

The existing `Limits()` defaults are 4096 image entries, 64 MiB per file,
256 MiB input bytes, 8192 archive records, 256 MiB unpacked bytes, 64 KiB chunks,
path depth 32, a 60-second cooperative deadline and 32 MiB accounted working
memory. Smaller caller ceilings can refuse inputs well below the fixed document
cap. The existing bounds validator prevents arbitrary enlargement.

The V5a JSON objects have closed field sets and reject duplicate keys, null,
floating-point values, non-finite values, invalid integers and malformed text.
The preflight is null-aware so the frozen V3 directory/symlink inventory can
retain its existing required null hashes. It does not broaden the V3 schema.
Expected-image and mapping bytes need not be rewritten into a canonical wire
serialization; byte pins cover the exact supplied representations.

Accounting is conservative and cooperative, not an OS-enforced RSS or CPU
sandbox. Time checks bound supported work but do not authenticate time. V5a
itself operates on in-memory data and performs no archive extraction,
filesystem-tree construction, investigated imports, package installation,
maintainer-script execution, subprocess or network access. The earlier V3/V4
measurement APIs retain their independent local I/O and platform boundaries.

## Unresolved mapping cases and exact V5b scope

Real-package symlinks, generated bytecode, installation-time metadata changes,
package relocation and conflicting shared-directory ownership remain unresolved
for this profile. An inventory with such members cannot be labelled a complete
image merely because its directly supported subset matches. The historical
V3 observation is not retroactively declared a V5a `MATCH`.

V5b must separately define and obtain approval for bounded transformation
semantics, exact tool/input/output/proof identities, explicit metadata changes,
link materialization rules, reproducible reconstruction and comparison with
independently pinned expected output hashes. It must preserve complete source
and destination accounting, refuse ambiguous links and sources, define failure
atomicity and resource limits, and retain the distinction between transformation
execution and independent image acceptance.

V5b is not implemented here. Production trust roots, authenticated time,
durable anti-rollback, installation, launch/lifetime enforcement, standard
Observer-parser activation, Stage 3B and production qualification remain
separate work requiring their existing authorization boundaries.
