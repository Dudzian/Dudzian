# Stage 9 / 3A: bounded Zstandard transport, V2 incomplete

Baseline: `c6b0ca36c5b2b8c3506eb32c5fb5b5419c3a3b9a` (main, including
PR #3226). This extends the existing inert Ubuntu adapter. The proposed
contracts, closed image inventory correction, production refusals, coverage
thresholds and qualification gates remain unchanged.

**V2 is not complete.** The four exact Ubuntu DEBs use GNU TAR headers and
contain symlinks. Those structures remain unsupported. Fully decoding their
Zstandard regions is a narrower result than accepting the package inventory.
No real package can produce a complete byte-consistency MATCH with this scope.

## Evidence and reproducible real-package check

The [frozen observed inventory](evidence/stage9_observer_3a_real_deb_inventory_v2.json)
records all four packages, complete ar/TAR membership, regular-file digests,
link targets and archive metadata. It was persisted before invoking the
production decoder against it. Its classification is
`SAME_ORIGIN_OBSERVED_COMPARISON_ONLY; NOT_INDEPENDENT_ACCEPTANCE`: the same
package bytes supplied the inspection expectations in an earlier, separate
phase. This can detect implementation disagreement, but cannot independently
authenticate a source or approve an expectation. The historical
[archive-source evidence](evidence/stage9_observer_3a_cpython_archive_sources_v1.md)
supplies the separately retained whole-archive sizes and hashes.

The [real-package analysis](evidence/stage9_observer_3a_real_deb_zstd_v2.md)
records acquisition and exact outcomes. The reproduction tool accepts existing
local DEBs and the frozen inventory; it never downloads, installs, extracts,
executes maintainer scripts, imports the investigated Python or changes target
metadata. A fully decoded frame PASS must always be read with these results:

```text
REAL_DEB_ZSTD_PARSED=PASS/FAIL
REAL_DEB_MEMBER_INVENTORY=UNSUPPORTED
REAL_DEB_BYTE_CONSISTENCY=INCOMPLETE
SOURCE_AUTHENTICITY=UNVERIFIED
V2_COMPLETE=NO
```

The first result refers only to complete decompression, framing, checksum,
declared content size and comparison against the frozen TAR digest. The second
and third describe the unchanged adapter's refusal of unsupported TAR records.
It does not silently project a subset of package members into an accepted
inventory. Source acceptance, tool acceptance and execution remain separate.

## Decoder boundary

Only `control.tar.zst` and `data.tar.zst` are added to the existing DEB2 member
allowlist. Raw/gzip behavior remains supported. Zstandard requires precisely
python-zstandard **0.25.0**, its C extension backend and embedded libzstd
**1.5.7**; missing or differing implementations refuse. The existing broad
`compression` extra does not qualify a tool. A separate `offline-verifier`
extra pins the tested version; the test extra installs it for negative-path CI.
Neither extra adds it to the future minimal Observer runtime.

The [toolchain evidence](evidence/stage9_observer_zstandard_toolchain_v2.md)
records concrete binaries, artifact digests, native dependencies and remaining
provenance requirements. Version checks are compatibility restrictions, not
authentication of an installed library. A substituted module can imitate a
version string. Independently accepted tool binaries, the host interpreter,
native loader and operating-system dependencies are still required for trusted
verification. No dependency is fetched during verification, and neither
`bot_core` nor an external `zstd` executable is used by the adapter.

A bounded preflight scans the descriptor's compressed range before native
decoding. It accepts exactly one ordinary frame, validates header/block framing,
checks dictionary/content-size/window declarations, and requires the frame end
to equal the ar member boundary. Skippable/magicless frames, dictionaries,
concatenation and trailing bytes refuse. Payload and optional checksum validity
are then checked by the native streaming decoder; a structurally complete
frame alone cannot yield success. Actual decompressed bytes must agree with
any known frame content size.

The decoder receives a bounded input region and fixed read requests. It never
buffers the entire unpacked TAR. Its allowed window and conservative native
context/buffer allowance are checked against the remaining working-memory
budget before decoding; live native allocation is checked as well. In this
specific C backend `max_window_size` is passed to libzstd in **bytes**, despite
upstream Python API prose describing KiB. Source review and test evidence are
recorded in the toolchain document.

The existing unpacked-byte, 100:1 expansion, file/member, path-depth, read-buffer
and deadline ceilings still apply. Control/data share the remaining per-package
budget, and subsequent packages receive the remaining global budget. Every
ceiling failure refuses the measurement. Retained inventories and decoder
allocations share the invocation's conservative allowance; allocations are not
a hard operating-system RSS limit. Deadlines remain cooperative around bounded
operations and cannot interrupt blocked syscalls or a native call. Separate
worker isolation would require its own reviewed design.

## GNU TAR, links and metadata: separate proposal required

Actual package headers contain magic/version bytes `7573746172202000`
(`ustar` plus two spaces and NUL), while the existing reader accepts only
`7573746172003030` (POSIX `ustar\0` / `00`). Adding a decompressor cannot make
these encodings equivalent. The parser continues to refuse at the first GNU
header; symlink type `2` would be a separate later refusal.

The current contract permits only regular files and directories. A complete
link inventory cannot be represented by inventing a regular member, omitting
the link or following its target. A minimal future proposal must separately:

1. Define the admitted GNU header subset, validate all applicable numeric and
   extension fields, and reject GNU long-name, sparse and other unreviewed
   records. Test conflicting GNU/POSIX interpretations and malformed headers.
2. Decide whether inert package-only link records can be represented outside
   the target image. Define complete link declarations, duplicate/path rules,
   link-target bytes, cycles and traversal behavior. Never dereference a link.
   Any image-link support requires a distinct contract and threat review.
3. Define required explicit package directories, including empty directories,
   independently of implicit ancestors and transport paths. Preserve PR #3226's
   closed image inventory and all existing negative tests.
4. Obtain independently accepted expectations and metadata-mapping rules before
   claiming source authenticity, image closure or authorization.

Observed TAR UID/GID are `0/0`, directories use `0755`, regular files `0644`
or `0755`, and symlinks `0777`. These are **package-member metadata**. Manifest
UID/GID/mode are separate target-image declarations; `_metadata_matches` checks
the observed staged filesystem against those declarations. `materialize_member`
preserves bytes and defines no ownership/mode mapping; directories cannot be
transform outputs. Installation transformations and their independent acceptance
were not measured. No ownership normalization, equality rule or post-install
transformation is introduced. Archive-to-target metadata mapping remains
`NOT_YET_PROVEN`, even if bytes eventually agree.

## Review boundary

P0: no established changed-source finding is evidence of repository-wide safety.
P1: missing independently accepted source/expectations/tools and missing physical
execution/host attestation continue to block production. P2: real-package GNU
headers/links block complete V2; metadata mapping, cooperative resource limits
and incomplete TOCTOU protection remain explicit. P3: the existing report wire
retains `synthetic=true` for actual byte reads and requires its advisory meaning.

Only a draft, inactive extension is appropriate while required exact-HEAD CI or
review remains unfinished. Conditional merge requires applicable exact-HEAD
checks and independent review of this disclosed narrower scope; it cannot be
described as completion of V2 or Stage 3A. No automatic merge is permitted.

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
