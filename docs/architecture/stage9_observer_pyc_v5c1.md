# Stage 9 / Observer 3A — bounded offline CPython bytecode V5c1

V5c1 checks already supplied immutable `.pyc` bytes against a separately pinned
expected image. It does not generate bytecode, execute an interpreter, inspect
code objects or qualify an Observer runtime. V1–V5b contracts and their result
semantics remain unchanged. This is a separate, versioned result, not an image
approval or an extension of the V1 import allowlist.

## Trust boundary

Three properties remain separate:

| Property | Evidence available in V5c1 | Limit |
| --- | --- | --- |
| BYTE_IDENTITY | Full supplied artifact SHA-256 and size equal independently supplied expectations | Does not establish how the artifact was produced |
| SOURCE_BINDING | Exact declaration identifies source bytes remeasured through the existing V5b DEB member reader | Binds the declaration to bytes, not the bytecode body to source semantics |
| RECONSTRUCTION_PROOF | No approved proof is accepted or produced in this version | Always `RECONSTRUCTION_UNVERIFIED` |

Header structure is reported separately. The eight hash-mode header bytes are
compared with their explicit declaration. The CPython source-hash algorithm is
not implemented or inferred from the host: `SOURCE_HASH_UNVERIFIED` remains even
when those bytes agree. SHA-256 of the source is a separate inventory identity;
it is never substituted for CPython's PEP 552 hash.

The body is opaque. A header followed by arbitrary data can satisfy this bounded
profile if those exact bytes were independently expected. Such a result does
not prove a valid serialized code object, the asserted `co_filename`, use of the
declared compiler, or compilation of the declared source. The adversarial test
with an opaque body retains the reconstruction blocker after byte agreement.

## Inputs and supported profile

`deployment/stage9_observer_pyc_v5c1.py` exposes `verify_pyc_v5c1`. It receives
immutable V3 inventory JSON, independently pinned expected-image JSON, a
separate mapping declaration, frozen DEB inputs and frozen bytecode inputs.
The expected-image SHA-256 is mandatory and supplied out of band. This module
never constructs missing expectations from measured bytecode or from a result.

The source chain is:

```text
independent expected image -> exact package / archive / snapshot declaration
DEB bytes -> V5b complete archive remeasurement -> data.tar -> regular .py member
         -> exact V5a record ID -> source SHA-256 and size -> declared .pyc
```

The V5b adapter verifies the complete closed DEB set, control identity and TAR
inventories before returning any selected source bytes. V5c1 adds no alternate
DEB parser. Its source profile inherits V5b's raw/gzip TAR restriction, exact
Ubuntu CPython package version and `amd64` architecture. Other V3 compression
profiles remain available to V3, but are not silently admitted here.

The bytecode profile is explicit CPython 3.12.3, `amd64`, PEP 552 16-byte header,
magic `cb0d0d0a`, `checked_hash` only. The target magic is a profile constant,
not `importlib.util.MAGIC_NUMBER` of the verification host. Flags must equal 3.
Unknown bits, flag 2, `unchecked_hash` and `timestamp` are refused. Timestamp
invalidation needs its own exact timestamp/size contract before any later
support. All 16 header bytes, total length and complete artifact digest are
checked. This structural profile does not qualify a real CPython build.

Compiler identity includes the full build ID, interpreter SHA-256, CPython
version, architecture and magic. Per-output generation claims include
optimization 0–2, absolute `co_filename`, target `.pyc` path and explicit
generation/invalidation policies. Independent expected declarations and the
mapping must agree. These comparisons detect inconsistent declarations; they
do not establish that the asserted compiler or parameters produced the body.

### Closed declaration schema

JSON uses the existing bounded V5a decoder and strict exact-key checks. Unknown
keys, duplicate keys, floats, non-finite numbers and extra records refuse.
The expected image has exactly:

```text
{format: "stage9-observer-expected-image-v5c1", image_id,
 snapshot_sha256, sources, files, bytecode, import_roots: []}
```

`sources` uses the unchanged V5a exact package source records (`source_id`,
`package`, `version`, `architecture`, `filename`, `sha256`, `size`). `files`
uses V5a's strict regular/directory metadata records and explicit-parent policy.
Its regular-file SHA-256/size are the independent `.pyc` expectations.

The separate mapping has exactly:

```text
{format: "stage9-observer-pyc-mapping-v5c1", snapshot_sha256,
 inventory_sha256, expected_image_sha256, sources,
 directory_policy: "identical-explicit-v1", transforms}
```

`bytecode` and `transforms` contain identical operation declarations, matched by
unique `transform_id`. Each operation has exactly:

```text
{transform_id, kind: "prebuilt_pyc_v5c1", output_file_id,
 source: {source_id, package, version, architecture, snapshot_sha256,
          archive_sha256, archive_size, member, record_id, sha256, size},
 compiler: {build_id, interpreter_sha256, version, architecture, magic_hex,
            optimization, co_filename, target_path, generation_policy},
 bytecode_format: "cpython-pep552-v1", invalidation: "checked_hash",
 source_hash_hex, usage: "inert_artifact_only"}
```

`source_hash_hex` is exactly 16 lowercase hex characters, representing the
eight header bytes in file order. `generation_policy` is exactly
`prebuilt-pep552-checked-hash-v1`. The full build ID is 40 lowercase hex
characters for this Ubuntu GNU 20-byte build-ID profile. `co_filename` equals
the absolute canonical source-member path. The output basename must be
`<source-stem>.cpython-312.pyc`, or
`<source-stem>.cpython-312.opt-1.pyc` / `.opt-2.pyc` for the selected optimization.
The independently expected directory determines its inert placement.

Each candidate is `FrozenPycV5c1(file_id, path, content, uid, gid, mode)`;
`content` must be exact immutable `bytes`. `package_result` is a required
successful V3 continuity claim checked against fresh measurements from all
frozen DEBs; it cannot replace those bytes. Neither a V3 result nor a Python
dataclass authenticates its own producer.

`PycResultV5c1.consistency` is `BYTECODE_EVIDENCE_ONLY`, `NOT_DECLARED`,
`MISMATCH` or `INCOMPLETE`. Only the first has successful per-file byte/header
and declaration bindings. `NOT_DECLARED` expressly indicates that no bytecode
was declared. On refusal `files` is empty and all input/binding digest fields
are `None`. Successful records retain full source identity, measured artifact
metadata, parsed header and a digest of the complete compiler claim; the pinned
expected image retains the exact compiler fields. Aggregate input and binding
digests account for the declarations without upgrading authority.

## Closed inert image

Every output must be a declared regular `.pyc` file with exact path, size,
digest, UID, GID, mode, transformation and source identity. Frozen input metadata
is compared with those expectations as caller-supplied data; it is not a host
filesystem observation. Directories and every parent must be declared; the
verifier creates no files or directories. Additional input artifacts, outputs,
duplicate identities, ambiguous origins and missing bindings refuse the whole
operation.

This initial contract represents an inert bytecode subimage. It admits no
active import roots or runtime roles. It cannot grant a `.pyc` an interpreter,
entrypoint, stdlib, native-library or import role. Existing V1 checks still
require `.pyc` evidence to be inert data outside every allowed import root,
including `__pycache__` discovery paths. V5c1 is never a successful V1 manifest
or V5a/V5b mapping result.

## Resource and refusal model

Inputs must have exact immutable `bytes`/tuple types. The verifier shares the
existing validated `Limits` and cooperative budget with the V5b reader: JSON
size/depth, files, members, path depth, cumulative reads, retained memory and
elapsed monotonic time are bounded. Hashing checks the deadline between bounded
chunks. Memory accounting conservatively covers retained inputs, decoded
declarations, selected source buffers and result metadata; it is not an OS RSS
limit. Deadline checks do not claim kernel-level preemption.

No investigated code is imported, compiled, deserialized or executed. There
are no network, package installation, maintainer-script, privileged operation
or target-interpreter calls in this verifier. Dependencies are trusted verifier
code, not imports from the investigated image.

All per-file results and aggregate bindings are staged privately. Malformed
input, mismatch, cancellation, timeout, exhaustion or unexpected failure returns
no partial successful file list or successful binding digests. An empty
bytecode declaration reports that bytecode verification was not performed.
No aggregate `MATCH` suggests a reconstructed transformation or qualified image.

## Outstanding evidence and V5c2 requirements

The missing-proof register applies to every accepted V5c1 artifact, including
artifacts with matching complete-file SHA-256:

| Missing evidence | V5c1 status | Required work before V5c2 can discharge it |
| --- | --- | --- |
| Exact target source-hash algorithm | `SOURCE_HASH_UNVERIFIED` | Retain independently pinned, distribution-patched target source for the PEP 552 hash path, magic-derived key, byte order and exact byte-input rules; implement and cross-check fixed independent vectors without trusting the host algorithm |
| Real compiler invocation and input closure | `RECONSTRUCTION_UNVERIFIED` | Bind the full interpreter SHA-256/build ID, runtime/toolchain dependencies, exact source bytes, optimization, encoding, filename, generation policy and output path in an approved reconstruction evidence contract |
| Actual reproducible `.pyc` output | `RECONSTRUCTION_UNVERIFIED` | Perform an independently approved, bounded, isolated, network-free reconstruction using the exact qualified interpreter; retain invocation, all inputs, outputs and independent comparison evidence |
| Code-object validity and embedded `co_filename` | `RECONSTRUCTION_UNVERIFIED` | Supply approved reconstruction evidence or a separately reviewed bounded non-executing body parser; never infer body metadata from a matching header |
| Provenance and authority of expectations | `UNVERIFIED` | Qualify the external acceptance authority and evidence transport independently of this verifier and its supplied pins |
| Runtime non-execution and import policy enforcement | `BLOCKED` | Obtain separate runtime/Stage 3B authorization and qualification; V5c2 must not implicitly activate inert `.pyc` data |

V5c2 must version any proof-bearing extension, define its signer/authority and
failure policy, preserve atomic refusals and existing V1–V5b results, add
negative evidence-substitution/replay tests and keep unsupported invalidation
modes refused. A digest of a caller's claimed proof is not approved proof.
There is no proof-promotion parameter in V5c1.

```text
Stage 9 = IN_PROGRESS
Observer 3A qualification = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
External authority acceptance = UNVERIFIED
Authenticated time = UNVERIFIED
Durable anti-rollback = BLOCKED
Runtime qualification = BLOCKED
PYC_RECONSTRUCTION = UNVERIFIED
```

See [validation and preflight](stage9_observer_pyc_v5c1_validation.md),
[V5b source/materialization contract](stage9_observer_materialization_v5b.md),
[V1 schema](stage9_observer_3a_schema_proposal.md),
[runtime closure](stage9_observer_3a_cpython_runtime_closure_v1.md) and
[exact build materials](stage9_observer_3a_cpython_build_materials_v1.md).
