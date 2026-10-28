# Stage 9 / 3A — complete proposed normative V1 profile

Status: **PROPOSED_UNAPPROVED — normative within this review proposal only.**

This addendum resolves the missing wire definitions identified after PR #3214,
whose approved design is commit `236fa623e1e88eaedf634d59d8bd3de8d38452de`.
It is not independently approved, a trust anchor, production authorization,
physical qualification, or a statement that Stage 3A is complete. The complete
candidate is supplied before its separately labelled working implementation.
Review and independent acceptance of this contract remain required.

The existing [design](stage9_trusted_observer_contracts_and_implementation_plan.md)
continues to define the trust boundaries, known launch/scope/run records and
excluded work. This document gives exact additional rules; it does not import
Source/CCA or Windows release authority into the Observer domain.

## 1. Shared representation and closed diagnostic vocabulary

Every record below has exactly the listed fields; variants are disjoint. No
unknown key, default, coercion, wildcard, unknown schema/version, or implicit
source/transform is accepted. `version` is integer 1; `purpose` is exactly
`STAGE9_OBSERVER_TEST_ONLY` where present. Integers require `type(x) is int`.

| Notation | Exact rule |
| --- | --- |
| `UINT` | Integer 0..9007199254740991, with any additional field bound below. |
| `ID` | Full ASCII match `[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}`; no trimming, case folding or Unicode normalization. |
| `MODULE` | At most 128 ASCII bytes; full match `[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*`. |
| `SHA256` | Exactly 64 lowercase hex characters, no prefix; the all-zero placeholder is refused. |
| `GIT_OID` | Exactly 40 lowercase hex characters. This is syntax, not proof that a Git object exists. |
| `UTC` | Valid UTC calendar value `YYYY-MM-DDTHH:MM:SSZ`; no fractional seconds or timezone alternatives. |
| `ABS_PATH` | Nonempty absolute POSIX path, at most 4096 UTF-8 bytes; no backslash, empty/dot/dotdot component, repeated/trailing slash except `/`, or ASCII/C1 control character. |
| `MEMBER` | The same lexical component rules, but relative and nonempty. Never a path to open on the host. |
| `EXACT_TEXT` | Nonempty printable ASCII token of at most 128 bytes, with no whitespace or `*?[]`; used for exact version/package strings. |

CR, LF, ESC, DEL and all other control characters are outside the ID grammar.
Colon is deliberately allowed for Observer-domain IDs; this is distinct from
the narrower existing Source/CCA ID grammar. Existing synthetic Observer IDs
remain compatible. Previously accepted arbitrary ASCII is tightened explicitly;
there is no normalization or compatibility fallback for rejected IDs.

Global JSON, byte, depth, file/source/reason/path and archive ceilings from
section 3.1 of the design remain in force. NUL, surrogate, float, NaN, infinity,
null and duplicate keys are refused. No collection defined here exceeds 32768
entries; sources have the separate 128 bound. Strings also remain bounded by
the enclosing document limit; path/ID/module/token bounds above are narrower.
Lists are ordered; only explicitly stated sets may ignore order. ID/name/path
lists are unique. List equality in a projection or policy binding is ordered.

The normative closed reason-code enum for this proposed profile is exactly:

```text
FORMAT_REJECTED, LIMIT_EXCEEDED, TRUST_CONTEXT_MISSING, ACCEPTANCE_REJECTED,
ROLLBACK_REJECTED, LAUNCH_POLICY_MISMATCH, RUNTIME_LAUNCH_UNVERIFIED,
HOST_SCOPE_UNVERIFIED, IMPORT_POLICY_REJECTED, CLOSURE_INCOMPLETE,
ARTIFACT_MISMATCH, UNSUPPORTED_SOURCE, PROVENANCE_INCOMPLETE,
FILE_POLICY_REJECTED, FILE_CHANGED, REPORT_REJECTED, CHANNEL_REJECTED
```

`INCOMPLETE`, `MISMATCH`, `UNVERIFIED`, `BLOCKED` and `PERMISSION_DENIED`
are status/result words, not additional reason codes. Reason lists contain at
most 64 unique enum entries. A refusal exposes only a fixed code, never raw
input, a path, exception text or secret marker.

## 2. Canonical documents and the acceptance signing boundary

The proposed manifest, acceptance policy, acceptance envelope and ObserverResult
must already equal `deployment.canonical_json.canonical_json_bytes(decoded)`.
The existing launch policy, standalone scope binding and run context retain the
same raw-equals-canonical requirement. A parser refuses noncanonical bytes; it
must not normalize or rewrite a document after receiving a digest pin.

This is an explicit wire tightening relative to the earlier candidate that
accepted a noncanonical acceptance envelope. The private Windows
`_verify_envelope` takes a decoded dictionary and cannot check its original
serialization; its public revocation/release wrappers separately require
canonical envelope bytes. Those wrappers are the precedent, not authorization
for this Observer schema. Historical pinned bytes are not silently migrated.

The signing calculation is **unchanged**:

```text
canonical_payload = canonical_json_bytes(envelope["payload"])
digest_bytes = SHA256(canonical_payload).digest()          # 32 bytes
payload_digest = digest_bytes.hex()                       # JSON 64 lowercase hex
signed_message = b"CryptoHunter.Stage9ObserverAcceptanceV1\x00" + digest_bytes
```

No ASCII-hex signing, second hash, domain change or new cryptographic verifier
is introduced. Manifest/policy pins still hash their exact raw document bytes;
requiring those bytes to be canonical does not change the digest preimage.
File/artifact/proof hashes still denote exact artifact bytes, not canonical JSON
unless their own contract specifically says so. A structurally valid signature
never becomes a verified signature, independent reviewer or target acceptance.

## 3. Complete ObserverManifestV1

Top-level fields are exactly those in design section 3.2:

```text
{schema, version, purpose, manifest_id, sequence, source_revision, target,
 sources, files, transforms, runtime, imports, launch_policy_sha256}
```

`schema="CryptoHunter.Stage9ObserverManifestV1"`; `manifest_id` is ID,
`sequence` is UINT and `source_revision` is GIT_OID for the Observer's accepted
source. It must differ from both observed historical installation commits:
`782816540373d7b31f426dab645ad115320d1234` and
`26e17e7c23ea6cc1b265d57bfda1a1e6589d9e3f`.
`launch_policy_sha256` is SHA256 of the canonical launch policy.

### 3.1 Full target

```text
{os: "Linux", distribution: "Ubuntu", distribution_version: "24.04",
 architecture: "x86_64",
 interpreter: {version, package_version, abi, build_id, profile}}
```

`version` is EXACT_TEXT matching `3.12.(0|[1-9][0-9]*)`; `package_version`
is EXACT_TEXT. `abi`, `build_id` and `profile` are IDs. The profile identifies
accepted build materials; an ID does not replace proof of profile contents.
Acceptance policy carries this entire target with exact equality. Launch target
is its four-field OS projection; its interpreter version/package_version/ABI/
build_id equal the corresponding nested values. Actual CPython build provenance
remains Stage 3B verification.

### 3.2 Sources and source references

`sources` has 1..128 records and unique source IDs. Base source shape:

```text
{source_id: ID, kind: SOURCE_KIND, artifact_sha256: SHA256,
 artifact_size: UINT, proof_record_sha256: SHA256}
```

The only SOURCE_KIND values are `git_object_proof`, `pinned_wheel`,
`ubuntu_apt_snapshot`, and `approved_transform`. Only `approved_transform`
adds required `transform_id: ID`; all other variants forbid it. Artifact size
is 1..16777216 bytes for pinned_wheel and 1..536870912 bytes for the
other kinds. The wheel adapter additionally retains its 32 MiB expanded-data
budget; a source record does not authorize bypassing either worker limit.

A transformed source references exactly one declared transform, and its artifact
size/digest equal that transform's output. Every declared source is used by
file provenance, a transform input, or a tool/runtime-source reference. No
unreferenced extra source or undeclared source reference is allowed. Proof
records are external digest-addressed bytes; their contents/authenticity and
source-kind-specific Git/APT/wheel checks are not established by this parser.

### 3.3 Files, provenance and image ancestry

`files` has 1..32768 records; file IDs and absolute paths are independently
unique. A regular file has exactly:

```text
{file_id: ID, path: ABS_PATH, type: "regular", uid: UINT, gid: UINT,
 mode: UINT, size: UINT, sha256: SHA256, provenance: PROVENANCE}
```

A directory has the same fields with `type="directory"`, without `sha256`, and
`size=0` as a logical inventory value, not a claim about host `stat.st_size`.
UID/GID range is 0..4294967294. Mode is integer 0..0777 with group/world write
bits 0022 clear; set-ID/special bits cannot occur. Each regular size and their
sum are at most 2147483648 bytes. Symlink, hardlink and special-file variants
are absent, and therefore always refused.

Provenance is exactly one of:

```text
{kind: "source_member", source_id: ID, member: MEMBER}
{kind: "transform_output", transform_id: ID}
```

Directories require source_member provenance. A transform_output is regular
and points to the unique transform whose output asserts that same file ID.
Source_member references must exist; raw archive members are inert data.

`runtime.image_root_file_id` explicitly names the directory image root, whose
path must not be `/`. Every other entry lies strictly below it. All ancestry
inside that root has explicit directory entries; a regular file cannot serve as
an ancestor. Directories are exactly the root, declared import/resolution roots,
and ancestors required by inventory entries. Unused extra directories are
refused. The host ancestry above the image root is not invented as image files.

### 3.4 Transforms

`transforms` has 0..32768 records with unique transform IDs and unique output
file IDs. Exactly these fields exist:

```text
{transform_id: ID, kind: TRANSFORM_KIND, inputs: [INPUT], tool: TOOL,
 parameters: PARAMETERS, output: OUTPUT, reconstruction_proof_sha256: SHA256}

INPUT = {kind: "source_member", source_id: ID, member: MEMBER,
         sha256: SHA256, size: UINT}
     or {kind: "file", file_id: ID, sha256: SHA256, size: UINT}
TOOL = {source_id: ID, member: MEMBER, sha256: SHA256,
        build_id: ID, runtime_source_ids: [ID, ...]}
OUTPUT = {file_id: ID, sha256: SHA256, size: UINT}
```

There is exactly one input in this deliberately narrow initial catalogue. Input
and output sizes are 0..2147483648 bytes. A file input is a declared regular
file with equal size/digest; source and runtime-source IDs must exist. The
runtime-source list is unique and at most 128. A toolchain is represented by
source materials, not by silently adding an executable tool to the image.
The output is a declared regular file with matching size/digest and reciprocal
transform_output provenance. Every declared transform has this matching output.

Only two transform kinds exist:

| Kind | Exact parameters and relationships |
| --- | --- |
| `materialize_member` | `{input_index:0}`; input kind is source_member and output size/digest equal input size/digest. No arbitrary target, script or host link following. |
| `pyc` | `{source_input_index:0, cpython_magic_hex, build_id, optimization, invalidation_mode, co_filename}`. Magic is exactly 8 lowercase hex characters; build ID equals the full target build and tool build. Optimization is integer 0..2. Invalidation mode is `checked_hash`, `unchecked_hash` or `timestamp`; co_filename is ABS_PATH. Output path ends `.pyc`. |

Only timestamp invalidation adds required `source_timestamp` and `source_size`,
each integer 0..4294967295; source_size equals the input size. Hash modes forbid
those two fields. Unlisted transforms/build hooks/source-build/link-following
semantics are unsupported. This module neither performs a transform nor checks
reconstruction proof contents. Local output is never its own expected proof.
A `.pyc` output may occur only as inert data_file_ids evidence/data: it cannot
be the interpreter, entrypoint, loader, libpython, stdlib, library, loader
configuration, hwcaps, dlopen role, or a source/native/sourceless Observer import. Its path must be outside every import-root directory, including
__pycache__ subtrees, so the declared image cannot offer it to normal import
cache discovery. This record never authorizes cache discovery or execution.
Actual non-execution/resolution enforcement remains Stage 3B.

### 3.5 Runtime, ELF declarations and imports

Runtime has exactly:

```text
{image_root_file_id, interpreter_file_id, entrypoint_file_id,
 libpython_file_ids, stdlib_file_ids, loader_file_id, library_file_ids,
 dependency_edges, loader_config_file_ids, absent_paths, hwcaps_file_ids,
 dlopen_file_ids, resolution_root_ids, rpath_runpath_policy,
 import_root_ids, data_file_ids, embedded_modules}
```

Singular IDs name existing files; plural IDs are unique ordered lists of at
most 32768. Interpreter, entrypoint and loader IDs are three distinct regular
files; the entrypoint path ends `.py`. Every plural role list names regular
files except resolution/import roots, each a nonempty directory-ID list.
The interpreter cannot also be a libpython ID. The exact union of regular
role IDs and source/native import file IDs equals all regular inventory files;
there are no implicit data/config/profile files. Profiles used by launch sandbox
and integrity policy are regular files in data_file_ids with matching digests.
CWD names an inventoried directory. These are declared roles, not content proof.

`absent_paths` is an ordered unique list of ABS_PATH values, none present in
files. `rpath_runpath_policy` is exactly IN_IMAGE_ONLY. Each dependency edge
has exactly `{requester_file_id:ID, needed:string, resolved_file_id:ID}`.
Needed is nonempty, at most 4096 UTF-8 bytes, without ASCII/C1 controls.
Requester/resolved IDs belong to the declared binary set: interpreter, loader,
libpython, libraries, hwcaps, dlopen and native import files. The pair
(requester,needed) is unique. At least one edge binds interpreter to loader.
Edges are declarations of PT_INTERP/DT_NEEDED/dlopen resolution, not evidence
that all actual ELF dependencies were inspected.

`embedded_modules` is nonempty, with unique IDs and names, and exactly these
records from the approved design:

```text
{embedded_module_id:ID, name:MODULE, kind:"builtin"|"frozen",
 build_id:ID, container_file_ids:[ID,...]}
```

Build equals the full target build. Container IDs are nonempty, unique and
restricted to the actual interpreter/libpython file IDs. Their binary digests
come from files; no fictional source file or per-module digest is introduced.
The expected complete build inventory still needs independently accepted build
materials and content verification in Stage 3B.

`imports` is nonempty and unique by MODULE name. Exact disjoint variants:

```text
{name:MODULE, kind:"source"|"native", file_id:ID, sha256:SHA256}
{name:MODULE, kind:"builtin"|"frozen", embedded_module_id:ID}
```

Source/native entries reference a regular file with equal digest under an
explicit import root. Dotted MODULE a.b.c has lexical relative stem a/b/c.
A source maps exactly to <root>/a/b/c.py or <root>/a/b/c/__init__.py; a native
maps exactly to <root>/a/b/c.so or <root>/a/b/c.<target ABI>.so. No arbitrary
alias, name rewriting or custom hook is accepted. Across all ordered import
roots, all four allowed source/native suffix forms and the forbidden
<root>/<stem>.pyc shadow form, exactly one candidate path exists in the regular-file inventory and it is the declared imported file.
A competing candidate is refused even if labelled data instead of an import;
root order must not conceal ambiguous shadowing. Each dotted source/native
import's parent names are explicit source imports at the same selected root,
with exact <parent stem>/__init__.py paths. Native/builtin/frozen parents for
file-backed children are unsupported in this initial profile; a later build
package variant needs an explicit reviewed schema. This is a declared lexical
map, not proof that actual Python resolution or loaded bytes matched it.
Native file IDs participate in the binary set. Every embedded entry is referenced
exactly once by an import with equal name/kind; no missing/extra embedded entry
is accepted. File/member contents, normal Python import resolution, actual
loader behavior, and the independently expected set remain Stage 3B work.
Zipimport, namespace packages, plugins, .pth/custom hooks and sourceless pyc
have no accepting variant.

Launch interpreter/ELF/root lists equal their runtime projections, including
ordered edges/absent paths. Launch argv is exactly the interpreter path,
`-I`, `-S`, `-B`, and entrypoint path. Launch files/runtime/imports projections
hash the full ordered manifest arrays/records with the existing canonicalizer.
All three digest projections and both pinned launch-policy hashes must agree.

### 3.6 Structural graph versus external artifact verification

Build one bounded declared provenance graph with typed node identities F(file),
S(source) and T(transform), and precisely these dependency directions:

- F -> S for source_member provenance; F -> T for transform_output provenance.
- S -> T for an approved_transform source.
- T -> its input F/S, tool source S, and each runtime source S.

Every node reference exists. Reject a cycle with an iterative algorithm before
recursive graph traversal. The output assertion T.outputs=F is checked for
reciprocal equality but is **not** an additional T -> F dependency edge, which
would create an artificial cycle for every legitimate transform output.

Separately check the declared ELF requester -> resolved graph, including its
self/cyclic edges. This initial profile does not support cyclic ELF graphs.
Rejecting that unsupported profile does not claim ELF cycles are universally
invalid; supporting one needs a separately reviewed resolution model.

Manifest, launch policy, acceptance policy/envelope and contexts remain outside
the executable image. Direct known metadata back-references can be compared
when those exact bytes are supplied. Neither an acyclic declared graph nor an
equal SHA-256 proves that hidden references inside external profile, archive,
proof or binary bytes were examined. Opaque-content verification and complete
build/import/ELF discovery are Stage 3B work; they cannot be labelled complete
by the contract parser or by a synthetic trace.

## 4. Complete AcceptancePolicyV1 and cross-document bindings

Top-level is exactly the existing policy field set:

```text
{schema,version,purpose,root_set_id,reviewer_keys,threshold,allowed_sources,
 minimum_sequence,revocation_checkpoint,valid_until,target,limits,
 launch_policy_sha256}
```

Schema is CryptoHunter.Stage9ObserverAcceptancePolicyV1. Root set is ID;
minimum_sequence is UINT; valid_until is UTC. Revocation checkpoint is exactly
`{sequence:UINT,state_sha256:SHA256}`. Reviewer keys are at least two exact
`{signer_id:ID,reviewer_id:ID,public_key_hex:64-lowercase-hex}` records, sorted
by unique signer ID; reviewer IDs and raw key values are also unique. Threshold
is exactly integer 2. Syntax is not key approval, revocation verification or
proof of reviewer independence.

Allowed_sources is an exact ordered copy of all manifest source records,
including kind, artifact digest/size, proof digest and transform reference.
Kind/ID-only matching and a broader unspecified source catalogue are refused.
Target equals the entire manifest target. Launch hash equals the manifest and
canonical launch bytes. `limits` has exactly the following 22 integer constants;
this initial profile has no implicit tunable/tightened-limit variant:

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

A synthetically compared bundle additionally requires exact manifest/policy raw
hashes in acceptance, equal root_set_id/checkpoint, acceptance sequence equal
to manifest sequence and not below policy minimum, all supplied signer IDs in
the declared policy keys, and threshold agreement. Issued time precedes expiry;
expiry does not exceed policy validity. Caller-supplied synthetic_now is checked
against the declared window solely for fixture consistency, never represented
as an independent clock/checkpoint. All signature bytes remain unverified.
No JSON policy establishes its own trusted roots, revoked-state head or time.

## 5. Complete proposed ObserverResultV1

The top-level field set remains design section 3.5, with every fixed block and
mandatory host_pid_namespace=UNVERIFIED from the Stage 3A requirement. Schema is
ObserverResultV1; synthetic is exactly true. Installation commits retain the
fixed historical pair and order. Run ID/nonce, boot, manifest/policy/launch/scope
hashes bind exactly to the supplied synthetic run context and standalone scope.
Started/finished are UTC; `0 <= finished-started < 300` seconds. A bundle checks
the declared run window against acceptance/policy validity; no clock is read.

Counters has exactly the following 10 UINT fields:

| Field | Unit and interpretation |
| --- | --- |
| pids_seen | Cumulative count of distinct PID identities enumerated in this run window; no arbitrary PID number. |
| tids_seen | Cumulative count of enumerated thread identities; partial enumeration is possible. |
| identities_checked | Count of the enumerated PID identities checked at least once; at most pids_seen. |
| references_checked | Cumulative count of reference-check events; one identity can have multiple events. |
| matching_observations | Cumulative count of positive reference-check events; at most references_checked. |
| permission_denied | Cumulative count of denied-read events. |
| identity_changes | Cumulative count of detected identity-change events. |
| binding_changes | Cumulative count of detected binding-change events. |
| incomplete_reads | Cumulative count of incomplete-read events. |
| limits_hit | Cumulative count of resource/enumeration-limit events. |

Every field is bounded by UINT; no count is silently saturated or truncated.
Positive tids_seen or references_checked requires pids_seen>0. Other event
categories may overlap, and a PID may produce several events. They are not
summed to a PID count. Zero observed matches is not proof of no installation,
complete host scope, stable lifetime, past absence or future absence.

Observations is exactly:

```text
{match_seen:bool, identity_stable:bool, bindings_stable:bool}
```

Match_seen is true iff matching_observations>0. A positive identity_changes
requires identity_stable=false; positive binding_changes requires
bindings_stable=false. False stability need not have a positive change counter:
missing/incomplete observations may leave stability unverified. Limits_hit>0
requires LIMIT_EXCEEDED. Reason codes always include both
RUNTIME_LAUNCH_UNVERIFIED and HOST_SCOPE_UNVERIFIED in this proposed Stage 3A
result. A denied-read counter does not invent PERMISSION_DENIED as a reason enum.

Every result remains completeness=INCOMPLETE, runtime_attestation=MISSING,
host_pid_namespace=UNVERIFIED, qualification=BLOCKED, full_confirmation=false,
coexistence_authorized=false, preflight_exception=false,
physical_test_status=PHYSICAL_TEST_PENDING, status=IN_PROGRESS and
production_status=PRODUCTION_NO_GO. No counter/boolean/valid signature syntax
can upgrade those fields. Offline results retain acceptance=UNVERIFIED.

Synthetic sequence comparison is an explicit fixture-state operation before a
single final report, not permission to send two frames. Both snapshots bind the
same run ID/nonce/boot, ordered installation pair, document/scope hashes and
start time. Finish time, every counter, and the set of reason codes are
nondecreasing. Once match_seen is true it stays true; once either stability is
false it stays false. Later failure cannot erase an earlier positive match.

## 6. Approval gate and implementation status

Standard parse_manifest, parse_acceptance_policy, parse_observer_result and the
full parse_report_frame continue to refuse pending independent contract review.
The separately labelled working implementation may expose parse_proposed_* and
proposed synthetic bundle/sequence APIs. Their immutable result is permanently
marked authorization=UNVERIFIED, synthetic=true and
review_status=PROPOSED_UNAPPROVED. There is no production switch, environment
flag, implicit activation, evidence-file export or approval inferred from data.

A proposed bundle compares declarations only. It checks complete declared
references/projections, source matching, raw/canonical digests, sequence/window,
known signer syntax, nonce/boot/scope and result consistency. It does not verify
artifact/proof contents, Ed25519 signatures, reviewer independence, trust roots,
revocation freshness, actual import/ELF closure, actual launch enforcement,
producer/channel identity, runtime attestation, physical qualification or host
PID namespace. The trusted receiver's replay state/authenticated receipt remains
outside this parser. Missing evidence retains every required block.

Independent review must accept or revise this entire wire profile, then a
separate reviewed change can replace the standard parser refusals. That review
is still open; Stage 3A must not be reported complete and Stage 3B integration
must not be represented as approved merely because working tests pass.
