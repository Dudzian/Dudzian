# Stage 9 / 3A — proposed completion of the closed V1 schemas

Status: **PROPOSAL FOR REVIEW; NOT AN APPROVED V1 CONTRACT.**

The approved design is commit `236fa623e1e88eaedf634d59d8bd3de8d38452de`.
Line references below refer to
`docs/architecture/stage9_trusted_observer_contracts_and_implementation_plan.md`
at that commit. Nothing here establishes acceptance, authorization, host scope,
runtime attestation, or permission to execute an artifact.

## Findings and safe behavior pending review

| Priority | Approved text | Missing decision / consequence |
| --- | --- | --- |
| P1 | 102–105, 159–165 | Exact nested manifest target/source/file/transform/runtime fields and source-kind literals are absent. A validator cannot distinguish missing/extra fields without inventing a wire format. |
| P1 | 231–234 | `AcceptancePolicyV1.allowed_sources`, `limits`, and the full policy target shape are not defined. Accepting an arbitrary object or interpreting source IDs as artifact authorization would widen policy. |
| P1 | 433–447 | `ObserverResultV1.counters` and `observations` have no exact shape, counter names, units, or field relationships. An accepting validator cannot substantiate consistent counts. |
| P2 | 105, 623–650 | Closed reason codes are required, but examples mix reason codes and status words; there is no normative exhaustive enum. |
| P2 | 337–342 | The ban on metadata back-references includes opaque profile/artifact bytes. A data-only contract module cannot inspect such bytes merely from their digests. Structural acyclicity must be distinguished from artifact-content verification in 3B. |

Until the relevant part is approved, decode and reject these underspecified
records with a fixed refusal code. Do not accept opaque nested dictionaries,
invent default values, construct a fake complete result, or claim that other
structurally valid records authorize anything. Exact already-specified schemas
(acceptance envelope, launch policy structure, embedded-module/import variants,
run context, namespace binding) can be implemented independently, but any check
requiring a valid manifest/policy remains unavailable.

No P0 contradiction was identified. The critical issue is missing wire schema,
not a disagreement about the mandated fail-closed trust model.

## Common notation

The following are proposed exact records, not Python constructors. `ID` means a
nonempty ASCII string of at most 128 characters; NUL remains forbidden globally.
`UINT` is an integer with `type(value) is int` and range 0..2^53−1. `SHA256`,
`GIT_OID`, `UTC`, `ABS_PATH`, and canonical JSON use the existing V1 rules.
Lists are unique where they represent IDs, names, sources, or edges; order is
preserved. No undeclared key is accepted.

Paths in the image are POSIX absolute paths without empty components, `.` or
`..`, repeated slashes, backslashes, or a trailing slash, except `/` itself.
Archive members use the same component rules but are relative and nonempty.
The 4096-byte UTF-8 ceiling applies before any path processing. This precise
path grammar is a proposal; the approved text only says canonical paths.

## Full target

Use exactly:

```text
{
  os: "Linux",
  distribution: "Ubuntu",
  distribution_version: "24.04",
  architecture: "x86_64",
  interpreter: {
    version: exact CPython "3.12.<patch>" string,
    package_version: nonempty exact package-version string,
    abi: nonempty exact ABI string,
    build_id: ID,
    profile: ID
  }
}
```

The manifest and acceptance policy carry this full target with exact equality.
Launch-policy target remains its already-approved four-field OS projection.
Launch-policy interpreter version/package_version/abi/build_id must equal the
full target interpreter fields. `profile` identifies the independently reviewed
build profile, not a wildcard or a runtime-discovered profile.

**Review decision:** the additional interpreter nesting and profile identifier
are proposed wire spelling. The document mandates their semantics but does not
choose their keys. Profile contents and independent build-proof verification
remain 3B work.

## Sources and source policy

Every source has exactly these common fields:

```text
{source_id: ID, kind: SOURCE_KIND, artifact_sha256: SHA256,
 artifact_size: UINT, proof_record_sha256: SHA256}
```

Proposed `SOURCE_KIND` literals are `git_object_proof`, `pinned_wheel`,
`ubuntu_apt_snapshot`, and `approved_transform`. Only `approved_transform`
adds the required field `transform_id: ID`; no other kind permits it.
`proof_record_sha256` identifies exact external proof bytes; it is never a
statement that the contract parser verified that proof. Source IDs are unique.
Artifact sizes are bounded by the approved general archive limit.

`AcceptancePolicyV1.allowed_sources` is proposed as the ordered list of complete
source records, with exact equality to `manifest.sources`. This repeats the
selected source inventory deliberately, avoiding an untyped allowlist or an
unresolved question about whether kind/ID alone authorizes changed bytes.

**Review decisions:** source-kind spelling, a digest-addressed proof-record
reference, and exact-inventory equality are not specified in the approved text.
The alternative of a broader allowed-source catalogue needs its own explicit
selection/matching rules; it must not be inferred in code. Proof record formats
and actual Git/APT/wheel verification are not defined by this proposal.

## Files and provenance

Proposed regular file:

```text
{file_id: ID, path: ABS_PATH, type: "regular", uid: UINT, gid: UINT,
 mode: UINT, size: UINT, sha256: SHA256, provenance: PROVENANCE}
```

Proposed directory has the same fields with `type="directory"` and without
`sha256`. `mode` contains permission bits only, at most 0777; set-ID/special
bits are forbidden. `size` remains an exact manifest value rather than a
platform-derived default. Source/transform references also apply to directories.
File IDs and paths are each unique. No link or special-file variant exists.

Provenance has exactly one of these shapes:

```text
{kind: "source_member", source_id: ID, member: canonical relative member}
{kind: "transform_output", transform_id: ID}
```

Referenced sources/transforms must exist. Directory ancestry that belongs to
the image must have directory entries; an image file cannot be its own ancestor.
Metadata documents remain outside the image.

**Review decisions:** `path`, `type`, nested provenance spelling, permission
bit representation, directory provenance, and which ancestor is the image root
need agreement. The approved document does not explicitly specify directory
size behavior or a separate `image_root`; do not silently choose a directory
size of zero or assume the host `/` is part of the image.

## Transforms

Proposed common record:

```text
{transform_id: ID, kind: TRANSFORM_KIND, inputs: [INPUT, ...],
 tool: TOOL, parameters: PARAMETERS, output: OUTPUT,
 reconstruction_proof_sha256: SHA256}
```

```text
INPUT = {kind: "source_member", source_id: ID, member: relative member,
         sha256: SHA256, size: UINT}
     or {kind: "file", file_id: ID, sha256: SHA256, size: UINT}
TOOL = {source_id: ID, member: relative member, sha256: SHA256,
        build_id: ID, runtime_source_ids: [ID, ...]}
OUTPUT = {file_id: ID, sha256: SHA256, size: UINT}
```

The toolchain references source artifacts, not executable files added to the
observer image merely because the transform used them. Every input digest/size
and output digest/size must match its referenced regular file where applicable.
Outputs point back to the same transform through file provenance. A dependency
graph uses source-transform, transform-input-file, file-provenance, and
tool/runtime-source edges; any cycle is refused. Distinguish a relationship
asserting an output from the dependency edge to avoid making every legitimate
output-provenance pair a trivial two-edge cycle.

Two narrow proposed kinds:

```text
kind="materialize_member"
PARAMETERS={input_index: UINT}

kind="pyc"
PARAMETERS={source_input_index: UINT, cpython_magic_hex: 8 lowercase hex,
            build_id: ID, optimization: 0|1|2,
            invalidation_mode: "checked_hash"|"unchecked_hash"|"timestamp",
            co_filename: ABS_PATH}
```

For timestamp mode only, parameters add exact required `source_timestamp` and
`source_size` UINT fields. Hash modes forbid those fields. `.pyc` output cannot
become a sourceless observer import. `materialize_member` materializes the exact
approved inert member; following host filesystem links is never allowed.

**Unresolved security decision:** the allowed transform-kind catalogue and its
complete parameter schema cannot be inferred. The two kinds above are a
candidate restricted subset, not an approved interpretation of every transform.
In particular, source-build transforms and link materialization require exact
independent provenance rules before acceptance. Do not add an arbitrary
`parameters` mapping or treat a local reconstruction as its expected output.

## Runtime

Proposed exact fields:

```text
{interpreter_file_id, entrypoint_file_id, libpython_file_ids,
 stdlib_file_ids, loader_file_id, library_file_ids, dependency_edges,
 loader_config_file_ids, absent_paths, hwcaps_file_ids, dlopen_file_ids,
 resolution_root_ids, rpath_runpath_policy, import_root_ids,
 data_file_ids, embedded_modules}
```

Singular file IDs are `ID`; plural file/root IDs are lists of IDs. All are
regular files except resolution/import roots, which are directories.
`absent_paths` is a unique list of canonical absolute paths that cannot also
occur in `files`. `dependency_edges` reuses the exact approved launch-policy
edge `{requester_file_id, needed, resolved_file_id}`. `rpath_runpath_policy`
is exactly `IN_IMAGE_ONLY`. `embedded_modules` reuses the exact approved
five-field variants from section 3.2.

The interpreter, entrypoint, libpython, loader/ELF lists, dependency edges,
absent paths, and roots equal their launch-policy projections. All referenced
file IDs exist with the required type; no ELF edge points outside its declared
binary set. Builtin/frozen containers are a nonempty subset of the interpreter
and libpython IDs, with names/kinds/build agreeing exactly with imports/target.
Runtime IDs plus import files and required directories cover the image inventory;
data/profile files are explicit `data_file_ids`, never implicit fallback.

**Review decisions:** full runtime field spelling and exact inventory projection
are not specified. Whether legitimate ELF cycles are forbidden independently of
provenance cycles needs an explicit choice; the task asks for cyclic dependency
rejection, but treating every kind of graph identically is not self-explanatory.
External metadata digests can be compared structurally when supplied. Detecting
a back-reference hidden in an opaque profile's actual bytes belongs to 3B.

## Acceptance policy limits

Propose an exact `limits` record with these literal V1 values. This uses the
existing ceilings and introduces no increased budget:

```text
manifest_bytes=4194304; acceptance_policy_bytes=65536;
launch_policy_bytes=65536; acceptance_bytes=65536; result_bytes=65536;
run_context_bytes=16384; depth=16; files=32768; sources=128;
reason_codes=64; identifier_ascii_chars=128; path_bytes=4096;
archive_bytes=536870912; unpacked_bytes=2147483648; archive_members=65536;
compression_ratio=100; artifact_timeout_s=60; bundle_timeout_s=300;
working_memory_bytes=268435456; wheel_archive_bytes=16777216;
wheel_unpacked_bytes=33554432; buffered_descriptor_bytes=16777216
```

Every value is an exact integer. No new tunable safety margin or permissive
fallback is introduced. Launch resources keep their independently approved
values, including the separately bounded `fd_limit`.

**Review decision:** field names and whether policies may tighten ceilings are
absent. Exact constants are the smallest initial closed profile. Supporting
smaller policy values requires an explicit `effective=min(policy,V1)` rule and
must not be silently inferred.

## Result counters and observations

Proposed exact `observations` object:

```text
{match_seen: bool, identity_stable: bool, bindings_stable: bool}
```

Proposed exact `counters` object, each field a UINT:

```text
{pids_seen, tids_seen, identities_checked, references_checked,
 matching_observations, permission_denied, identity_changes,
 binding_changes, incomplete_reads, limits_hit}
```

Counters are cumulative counts in the run window, not arbitrary PID values;
categories may overlap and are not summed. `matching_observations > 0` iff
`match_seen=true`; once positive, neither can be cleared by later errors.
`identity_changes > 0` requires `identity_stable=false`; `binding_changes > 0`
requires `bindings_stable=false`. Any instability/error/limit or missing runtime
attestation/host authentication requires `completeness=INCOMPLETE`. All Stage 3A
results remain synthetic, acceptance UNVERIFIED or REJECTED, runtime attestation
MISSING, host PID namespace UNVERIFIED, and retain the approved fixed blocks.
No accepted structure establishes a physical qualification.

**Review decisions:** the observations object shape, counter names/units and
relations are proposed because no normative set exists. UINT is already bounded
by the common exact-number ceiling; any narrower domain budget must be stated
with its unit. A stateless single-result validator cannot prove monotonic
preservation across multiple reports; synthetic sequence tests must supply the
previous observation state explicitly. There is one frame per actual run.

## Closed refusal codes

Propose the enum consisting of the refusal names already present in section 8:

```text
FORMAT_REJECTED, LIMIT_EXCEEDED, TRUST_CONTEXT_MISSING, ACCEPTANCE_REJECTED,
ROLLBACK_REJECTED, LAUNCH_POLICY_MISMATCH, RUNTIME_LAUNCH_UNVERIFIED,
HOST_SCOPE_UNVERIFIED, IMPORT_POLICY_REJECTED, CLOSURE_INCOMPLETE,
ARTIFACT_MISMATCH, UNSUPPORTED_SOURCE, PROVENANCE_INCOMPLETE,
FILE_POLICY_REJECTED, FILE_CHANGED, REPORT_REJECTED, CHANNEL_REJECTED
```

`INCOMPLETE`, `MISMATCH`, `UNVERIFIED`, and `PERMISSION_DENIED` are not silently
added as reason codes from their use as result/status words. A code specifically
for an unfinalized schema would itself require an explicit addition; until then
`FORMAT_REJECTED` is the existing safe refusal.

This proposal needs review before it becomes an accepting implementation of
the missing schemas. It does not change cryptography, `--evidence`, qualification
thresholds, host security, or the deliberately excluded Stage 3B mechanisms.
