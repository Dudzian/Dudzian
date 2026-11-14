# Stage 9 / Observer 3A — local APT chain V4a

The separate [V4a module](../../deployment/stage9_observer_apt_v4a.py) compares a
closed, locally provisioned `Release → Packages → DEB` slice against frozen
caller declarations and the existing [V3 inventory](stage9_observer_offline_v3.md).
`MATCH` means only that all supplied identities, bytes and supported archive
records agree. The declarations and the materials can share the same untrusted
origin. Neither this comparison nor freezing a declaration authenticates them.

## Public input and result

`verify_apt_provenance_v4a(input_directory, *, declaration, inventory_raw,
limits=Limits())` accepts an absolute local directory, a frozen
`AptDeclarationV4a`, existing V3 JSON declaration bytes and existing `Limits`.
The declaration contains `release_path`, `release_sha256`, `release_size`,
`packages_path`, `packages_sha256`, `packages_size` and `packages_format`.
Sizes are byte counts; SHA-256 values are nonzero, lowercase, 64-digit hex.
The caller supplies these expectations before comparison; the verifier has no
discovery, download, expectation-update or command-line mode.

For example, with the named frozen values and V3 bytes supplied separately:

```python
from deployment.stage9_observer_apt_v4a import (
    AptDeclarationV4a,
    verify_apt_provenance_v4a,
)

selection = AptDeclarationV4a(
    release_path="dists/local/Release",
    release_sha256=frozen_release_sha256,
    release_size=frozen_release_size,
    packages_path="dists/local/main/binary-amd64/Packages.gz",
    packages_sha256=frozen_index_sha256,
    packages_size=frozen_index_size,
    packages_format="Packages.gz",
)
result = verify_apt_provenance_v4a(
    "/absolute/provisioned-slice",
    declaration=selection,
    inventory_raw=frozen_v3_inventory_bytes,
)
```

The corresponding Release checksum path is
`main/binary-amd64/Packages.gz`, relative to the Release directory. Selecting an
index outside that directory refuses. All selected DEBs remain directly below
the input root, using V3's canonical basenames. The directory must contain
exactly Release, the selected index, declared DEBs and their required ancestor
directories. Extra files, extra directories, missing inputs and path collisions
refuse. There is no implicit search for an alternative index or package.

`AptResultV4a` has profile `stage9-offline-apt-chain-v4a`, scope
`LOCAL_APT_CHAIN_ONLY`, `consistency`, controlled `reason_codes` and a tuple of
V3 package measurements. Only a complete `MATCH` returns measurements; any
failure discards every partial package result. These fields remain fixed for
all outcomes, including `MATCH`:

```text
signature_verification=NOT_PERFORMED
source_authenticity=UNVERIFIED
acceptance=UNVERIFIED
stage_9=IN_PROGRESS
qualification=BLOCKED
STAGE_3A_COMPLETE=NO
PRODUCTION_QUALIFIED=NO
production_status=PRODUCTION_NO_GO
refused=True
```

This result has no authorization capability and does not replace or reinterpret
historical V1–V3 results.

## Exact metadata profile

This initial profile is a bounded provisioned comparison slice. It does not
accept a general Ubuntu archive Release or an unchanged ordinary repository
Packages index: additional checksums, metadata fields and pool paths refuse.

| Material | Accepted form |
| --- | --- |
| Text | UTF-8 decoding followed by printable ASCII and LF only; final LF required; lines at most 4096 bytes. No BOM, CR, tab, NUL or non-ASCII text. |
| Release | Basename exactly `Release`; one stanza containing exactly one indented SHA256 checksum row. The row's digest, positive decimal size and relative path must equal the explicitly selected index. |
| Release fields | Required `SHA256`. Optional single-line `Origin`, `Label`, `Suite`, `Version`, `Codename`, `Date`, `Valid-Until`, `Architectures`, `Components`, `Description`, `Acquire-By-Hash`. Optional values are inert metadata, without time, freshness, architecture or authority semantics. |
| Packages | Basename exactly `Packages` or `Packages.gz`, matching `packages_format`. Each blank-line-separated stanza has exactly `Package`, `Version`, `Architecture`, `Filename`, `SHA256`, `Size`, in any order. No continuation lines or additional fields. |
| Field syntax | Names are case insensitive; duplicates therefore include capitalization aliases. A nonempty value follows colon plus one space and has no surrounding whitespace. Digests are lowercase; positive decimal sizes have no sign, leading zero or more than ten digits. |
| Input paths | Literal relative paths, validated by the existing path rules and an ASCII allowlist. No absolute paths, traversal, empty/dot components, backslashes, percent encodings or Unicode normalization aliases. No URL parsing or decoding. |
| Gzip index | One ordinary gzip/DEFLATE member, compression method 8 and header flags zero. Mtime, XFL and OS bytes are inert. CRC/length and full EOF must validate. No optional header fields, concatenated members, trailing bytes (including NULs), raw DEFLATE or zlib-wrapper fallback. |

Duplicate fields, repeated/conflicting records or checksum rows, and additional
algorithms or variants are refusals. All Packages stanzas must match the exact
V3 set once. Every stanza's name, version, architecture, filename, SHA-256 and
size must agree with its V3 declaration. `Filename` is the exact flat V3 DEB
basename; `pool/...` and alternate spellings are unsupported in this profile.
The complete selected compressed index bytes are compared before decoding;
the decoded content must independently pass the stanza and identity checks.

V3 retains its five-package CPython allowlist, version `3.12.3-1ubuntu0.17`,
architecture `amd64`, canonical filenames and complete inert TAR declarations.
V4a hashes each whole DEB before calling the existing archive verifier on the
same pinned descriptor and shared budget. Control identity, control/data TAR
bytes and records, payloads, metadata, literal symlinks and decoder completion
remain V3 obligations. Raw, gzip and the existing pinned Zstandard DEB transports
retain V3's exact profile; the index gzip restriction does not redefine them.
No archive links become filesystem operations or image records.

## Limits, I/O and refusals

V4a runs only on Linux as a non-root process. It reuses `_InputReader`, `_Budget`,
`Limits` and `ContractRefusal`; no simplified filesystem reader is introduced.
The input root and ancestors are opened without following symlinks. Files must
be regular, single-link and on the selected filesystem. Initial and final closed
inventories, descriptor metadata and named bindings are checked. A failed final
scan, descriptor check or close prevents `MATCH`.

The existing default limits remain 4096 tree entries, 64 MiB per file,
256 MiB aggregate input bytes, 8192 records, 256 MiB unpacked bytes, 64 KiB read
chunks, path depth 32, 60 seconds and 32 MiB accounted working memory. Caller
limits are validated by the existing maximum-bound checks. Additional V4a caps
are 1 MiB Release, 2 MiB encoded index and 2 MiB decoded index, also constrained
by applicable caller ceilings. Gzip expansion cannot exceed 100 times encoded
index size. Native output calls have explicit bounded output lengths.

Release, index and whole-DEB reads share the input-byte budget. The checksum row,
Packages stanzas and V3 TAR records share the record budget. Decoded index bytes
(plain or gzip) and every V3 control/data stream share the unpacked-byte budget.
Conservative reservations cover declarations, metadata bytes/text/containers,
lookup structures, snapshots, buffers and decoder allowance before allocation.
Index buffers are dropped before DEB parsing; their reservations remain charged
conservatively. Budget checks surround bounded work throughout the chain.

`ARTIFACT_MISMATCH` produces `MISMATCH`; other refusals produce `INCOMPLETE`.
Reasons include `FORMAT_REJECTED`, `UNSUPPORTED_SOURCE`, `FILE_POLICY_REJECTED`,
`PROVENANCE_INCOMPLETE`, `CLOSURE_INCOMPLETE`, `FILE_CHANGED` and
`LIMIT_EXCEEDED`. Expected I/O, decoding, allocation and parsing failures become
controlled refusals. An unsuccessful root close forces `INCOMPLETE` even after
all content comparisons succeeded. No raw native error text or partial successful
batch is returned.

These are cooperative accounting and deadline checks, not hard process RSS
limits or an operating-system sandbox. They cannot interrupt blocked filesystem
or native calls. Snapshots detect some changes without providing complete TOCTOU
protection or guaranteeing immutability after return. Interpreter, decoder,
kernel and filesystem acceptance remain separate requirements.

## Trust boundary and later work

The production flow performs no network access, keyserver access, signature-tool
invocation, key enrollment, DEB installation, maintainer-script execution,
investigated-code execution, privilege elevation or image materialization.
Even mutually agreeing materials are unauthenticated local data. This change
does not activate Stage 3B, approve a contract or qualify production.

V4b requires separate acceptance of external roots, signer/subkey and algorithm
policies, time/freshness, checkpoints/rollback, and the provenance, version and
isolation of `gpgv`. No test key is a default trust root. APT archive-key trust
remains distinct from Observer reviewer authority. V5 must separately define
archive-to-image transformations, metadata mapping, expected-image acceptance
and runtime closure. Neither stage is implemented by V4a.

Actual synthetic test and tool outcomes, unavailable checks and remaining risks
are recorded in the separate
[validation report](evidence/stage9_observer_apt_v4a_validation_2026-11-14.md).
This contract does not claim a real-repository, physical-host or production run.
