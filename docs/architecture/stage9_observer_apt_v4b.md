# Stage 9 / Observer 3A — offline APT signature boundary V4b

The separate [V4b integration](../../deployment/stage9_observer_apt_v4b.py)
and [bounded adapter](../../deployment/stage9_observer_gpgv_v4b.py) verify a
detached OpenPGP signature over frozen local `Release` bytes using an explicitly
pinned local `gpgv` executable and keyring. It compares the same Release bytes with the complete bounded
`Release → Packages → DEB` slice supported by [V4a](stage9_observer_apt_v4a.md).
It does not create an APT client, discover keys, enroll roots, approve a trust
policy or qualify the Observer.

The implementation starts from `dcc71020c14ff1c09c49cd8be99f7740e490a818`,
the `main` commit containing merged [V4a PR #3269](https://github.com/Dudzian/Dudzian/pull/3269).
Neither this branch nor its draft PR authorizes a merge.

## Independent claims

Three claims have different evidence and cannot establish each other:

| Claim | Evidence | What it does not establish |
| --- | --- | --- |
| Cryptographic signature validity | Complete, coherent machine status from the selected `gpgv`, acceptable exit status and exact frozen input bytes | Publisher authorization, policy provenance, freshness or source authenticity |
| Signer-policy conformance | Exact signing and primary fingerprints, explicit subkey relation, algorithm and local validity rules from the supplied context | Independent acceptance of that context or its issuer |
| External acceptance of context | An independently authenticated approval by an authorized external authority, bound to the exact context version and contents | Cryptographic validity of an arbitrary signature or freshness of an arbitrary repository snapshot |

There is no accepted mechanism in this change to authenticate external authority
or approval records. The implementation does not invent an authority, bootstrap
trust from caller declarations or accept a caller-set approval flag. An authority
reference in a context is a claim to be independently checked. It never verifies
its own origin.

`V4a.source_authenticity=UNVERIFIED` and its historical signature semantics stay
unchanged. V4b has a separate result; it cannot upgrade or mutate a V4a result.
A cryptographically correct, policy-conforming synthetic signature can therefore
coexist with blocked acceptance and unverified source authenticity.

## Public integration

`verify_apt_signature_v4b(input_directory, *, declaration, signature,
inventory_raw, context, tool, limits=Limits(), gpgv_limits=GpgvLimitsV4b())`
combines a frozen V4a `AptDeclarationV4a` and V3 JSON inventory bytes with a
supplied `TrustContextV4b`, `SignatureDeclarationV4b` and `GpgvToolV4b`. The
signature declaration selects exact local signature/keyring paths, SHA-256
values and sizes plus repository/snapshot labels. The tool selects an absolute
ELF executable path and its SHA-256. It exposes V4b evidence separately from a
semantically unchanged `AptResultV4a` chain snapshot. Passing a previously
successful, caller-constructed V4a result cannot supply chain evidence.

`TrustContextV4b` uses explicit `SignerRuleV4b`, `TimeRulesV4b` and
`ExternalAuthorityReferenceV4b` records. There is no caller approval boolean.
Local success requires cryptography, signer-policy checks and the complete
frozen chain to pass together. After any local failure or cleanup/deadline
failure, no successful package batch or stale cryptographic evidence survives.
A complete local comparison returns `cryptographic_signature=VALID`,
`signer_policy=MATCH`, exact `GpgvEvidenceV4b`, the Release SHA-256 and the
separate V4a chain snapshot. A failed combined comparison returns both local
claims as `UNVERIFIED`, with no evidence, chain or Release digest. It does not
label a mathematically valid signature invalid merely because the policy or a
later chain link refused. Fixed `reason_codes` explain the refusal without
returning subprocess diagnostics.

Independent statuses remain conservative even after complete local success:

```text
external_authority_acceptance=UNVERIFIED
freshness=BLOCKED
checkpoint=BLOCKED
rollback=BLOCKED
tool_acceptance=UNVERIFIED
system_isolation=UNVERIFIED
source_authenticity=UNVERIFIED
acceptance=BLOCKED
```

## Versioned trust-context contract

`TrustContextV4b.schema` is exactly `stage9-apt-trust-context-v1`;
`context_id` and a positive integer `context_version` identify its frozen local
policy revision. It is separate from the signature result. The context contains
`allowed_signers`, `approved_algorithms`, `subkey_policy`, `repository_id`,
`snapshot_id`, `release_sha256`, `keyring_sha256`, `valid_from`, `valid_until`,
`revoked`, `time_rules` and `external_authority`. All identities and
constraints must be explicit; no key ID suffix, user ID text, keyserver result,
local default keyring or implicit subkey delegation supplies authorization.

| Required information | Boundary rule |
| --- | --- |
| Context identity and schema/revision | Explicit nonempty context identifier and supported version; changes produce a different frozen context |
| Exact allowed keys | Full primary and signing fingerprints; comparisons use complete identities |
| Algorithms | Explicit allowed public-key and digest algorithms within the adapter's supported subset; unknown and unsupported values refuse |
| Subkeys | Explicit permitted signing-subkey/primary relation; an otherwise valid subkey does not inherit permission implicitly |
| Repository or snapshot | Explicit expected repository/snapshot identity and exact frozen Release digest, with the selected chain declaration |
| Validity and revocation | Explicit validity interval and revoked/disallowed keys; cryptographic success does not override a local policy refusal |
| Time and freshness | Explicit time-policy information; the implementation cannot authenticate the caller's clock or bootstrap freshness from a signed date alone |
| Checkpoints and rollback | Explicit external checkpoint/rollback-policy references; no accepted monotonic state or antirollback authority is supplied here |
| External approving authority | Unambiguous reference to an independently verifiable approval mechanism; absent or unauthenticated acceptance stays UNVERIFIED/BLOCKED |

Allowed signer records contain full 40-digit uppercase signing/primary
fingerprints, `valid_from`, `valid_until` and a boolean `revoked`. At most 32
unique exact pairs are accepted. `subkey_policy` is exactly
`EXACT_SIGNING_PRIMARY_PAIRS`; signing directly with a primary key still needs
an explicit pair with both fingerprints equal. A different subkey under the
same primary is refused unless separately listed. Public-key/digest algorithm
pairs are selected from `(1, 8)`, `(1, 10)`, `(22, 8)`, `(22, 10)` (RSA or
EdDSA with SHA-256 or SHA-512). An allowed fingerprint does not imply algorithm
permission, and an allowed algorithm does not imply signer permission.

All local validity intervals are half open: `valid_from <= timestamp <
valid_until`. Context and signer revocation are explicit local policy refusals.
`TimeRulesV4b` contains `evaluation_time`, `maximum_signature_age_seconds`,
`maximum_future_skew_seconds`, and optional `freshness_authority_reference`,
`checkpoint_reference`, `rollback_policy_reference`. The clock and references
remain caller assertions. An `ExternalAuthorityReferenceV4b` contains
`authority_id`, `decision_reference`, `delivery_mechanism`, never a verified
approval boolean. Even nonempty well-formed references do not approve anything.

The contract does not define a production authority or root list. TEST keys and
policy records demonstrate local enforcement only. Repository names, snapshot
labels, clock values and approval references are caller-supplied assertions;
matching them does not authenticate their provenance.

## Frozen-byte chain binding

V4b must prove that the signature and full chain comparison use identical
Release bytes. Comparing a digest once, then reopening a mutable pathname for
the signature or chain, does not meet this condition. A path, matching filename
or successful `gpgv` process alone cannot establish the binding.

The boundary uses one existing `_InputReader` baseline for Release, selected
index, declared DEBs, detached signature and keyring. Release, index, signature
and keyring become immutable byte values. The exact Release value passed to the
adapter is also passed to V4a's `_release` parser; no pathname reopen supplies a
second version. Each DEB is copied from its pinned original descriptor into a
sealed memory file; whole-DEB hashing and V3 archive parsing consume that frozen
copy. A final original-tree inventory still detects observed input changes.

The complete index and DEB checks reuse V4a's `_selection`, `_read_metadata`,
`_release`, `_decode_index` and `_packages` and V3's `_parse` and
`_verify_archive`; they retain the narrow metadata profile and complete inert
archive checks. Any missing link, mutation, incomplete final inventory, cleanup
failure or inability to establish byte binding refuses the combined result and discards partial package
measurements. The source path is never used to replace already selected bytes
between signature verification and chain comparison.

V4a itself is unchanged. Its existing API, fixtures, refusal semantics and
`NOT_PERFORMED` signature field remain regression requirements. V1–V3 remain
separate historical contracts.

## Effective DEB size under the shared budget

`Limits()` sets `max_file_bytes=67,108,864` (64 MiB),
`working_memory_bytes=33,554,432` (32 MiB), and `chunk_bytes=65,536`.
The 64 MiB declaration ceiling is not an achievable default DEB copy size.
`_frozen_archive()` reserves the entire `size + chunk_bytes` before creating
its `memfd`. The copy already reads and hashes bounded chunks; it does not
construct a whole-DEB Python byte value. The full-size reservation accounts
conservatively for the kernel-backed immutable copy.

Let `M` be the working-memory ceiling, `C` the chunk size, and `R` the retained
budget immediately before the first DEB. For an accepted input tree, define
`S = sum(512 + 8 * len(relative_path))` over every selected file and ancestor
directory. Paths in this API are ASCII. The exact earlier reservation is:

```text
I = 16 * len(inventory_raw)
    + sum_over_each_control_and_data_inventory(
        4096 + sum_over_records(4096 + 8 * (UTF8_name_bytes + UTF8_linkname_bytes)))
A = 16 * (Release_size + signature_size + keyring_size + selected_index_size)
G = 0                                                   for Packages
G = 262144 + 4*C + 16*decoded_index_size                 for Packages.gz
R = I + (8*C + 1048576) + 2*S + A + G
```

The `2*S` comprises the expected-tree model and the initial directory scan.
For gzip indexes, `G` describes the retained value after successful decoding;
each temporary decompression allowance must also fit when reserved. Parsing,
metadata and decoder limits may refuse before a DEB is reached.

For DEB `i`, of size `D_i`, the copy-admission condition is exactly:

```text
8 <= D_i <= min(max_file_bytes, M - R - sum_previous(D_j + C) - C)
```

This is a necessary size condition, not evidence of valid archive structure.
The `_Budget` reservations are cumulative: deleting metadata values or closing
a sealed DEB does not refund them. Multiple DEBs therefore share the remaining
allowance; the limit is not independently available to every package.
Furthermore, the mandatory `reader.finish()` performs another directory scan
and reserves another `S`. With raw/gzip DEB members, the complete operation's
retained-budget condition is:

```text
sum(D_i) + number_of_DEBs*C + R + S <= M
single DEB: D <= min(max_file_bytes, M - R - C - S)
```

Every other check still applies, including total input reads, unpacked bytes,
record counts, TAR-inventory memory, complete hashes, format, deadline and
cleanup. For Zstandard members, each decoder additionally requires
`window_size + estimate_decompression_context_size() + 524288` bytes from
`M - budget.retained` at decoder creation. Its content/window/ratio and native
memory checks can narrow the admitted range further. There is consequently
no universal exact default maximum independent of the selected metadata,
inventory, paths, compression and number of DEBs.

Even ignoring all positive metadata/tree/inventory allowances, one default
DEB must be smaller than `31,916,032` bytes (30.4375 MiB), because the fixed
reservation alone consumes 1.5 MiB and the copy needs another 64 KiB. The
actual complete-operation limit is lower. The public regression fixture with
one gzip control member and one raw data member has `R=1,723,520` and
`S=3,272`: copying admits at most **31,765,376 bytes**, while full local success
admits at most **31,762,104 bytes** (about 30.291 MiB). Its complete valid DEB
at equality succeeds; the next valid ar size, two bytes larger, refuses during
the final scan. The same valid archive fails if the memory ceiling is tightened
by one byte. One additional legal JSON whitespace byte reserves sixteen more
bytes and likewise refuses at that boundary. At the separate copy-admission
boundary, one additional declared byte refuses before `memfd` allocation.
All refusals discard the cryptographic and package evidence. The two-package
boundary test also checks that closing the first copy does not restore budget.

### Possible future copy-accounting model

A separate model could retain chunked copying and track live sealed-archive
storage independently from Python buffers and decoder workspace. An explicit
hard cap on aggregate live `memfd` bytes, reservations before allocation, and
release only after confirmed descriptor closure could avoid charging closed
copies cumulatively. This could process larger declared inputs with bounded
Python buffers and describe peak resource use more accurately.

Charging only a chunk would instead omit kernel `memfd` storage. Those pages
consume memory or swap and can cause host/cgroup pressure or allocation
failure; streaming alone is not a hard RSS or system-isolation boundary.
Any replacement would need an independently enforced memory/storage quota,
decoder and concurrent-buffer accounting, bounded reads and writes (including
short-write handling), unchanged exact-size/hash and original-descriptor
mutation checks, mandatory seal/readback before parsing, deadlines and
fail-closed cleanup on every failure. Neither reopening a mutable pathname
nor parsing before sealing is a substitute. This remediation does not adopt
that model, increase any limit, refund reservations, or change V4a.

## Bounded local `gpgv` execution

`verify_detached_v4b(*, release, signature, keyring, tool,
limits=GpgvLimitsV4b())` returns `GpgvEvidenceV4b` or a controlled
`GpgvRefusalV4b`. It accepts explicitly provisioned frozen local materials. It uses an
absolute executable selection plus SHA-256 pin and exact keyring bytes selected
by the caller. Hash agreement establishes only local byte identity, not
independent approval of the tool or keyring.

The public adapter itself refuses effective UID zero with
`GPGV_INPUT_REJECTED` before opening the selected tool or starting a subprocess.
This check supplements the outer APT verifier's root refusal; it does not
establish independent tool acceptance or operating-system isolation.

The invocation uses an argument vector, never a shell, and a fresh private
working/home directory with a closed environment. It does not consult system
keyrings, import keys globally, use a keyserver, download keys or build trust.
Release, detached signature and keyring are individually bounded, with deadline
and aggregate output limits. `GpgvLimitsV4b` defaults to 5 seconds, a 16 MiB
aggregate input ceiling and a 64 KiB cumulative output ceiling shared by
status/stdout/stderr. Caller limits cannot exceed 30 seconds, 16 MiB input or
64 KiB output. Additional input caps are 1 MiB Release, 64 KiB signature and
8 MiB keyring; the combined integration narrows keyring input to 4 MiB and
applies the existing shared Observer budgets. The ELF tool itself is capped at
32 MiB. Tool and input bytes are copied into sealed memory files. Dynamic
loader and library bytes are not accepted by this pin.

A dedicated inherited machine-status descriptor is distinct from stdout and
human-readable stderr. Any stdout content refuses; stderr is counted but never
parsed as evidence. The environment contains only fixed `LC_ALL=C` and the
fresh private `GNUPGHOME`; the invocation uses explicit home/keyring paths and
`--` before material arguments.

Exit code zero is necessary but insufficient. The parser requires one each of
`NEWSIG`, `SIG_ID`, `GOODSIG`, `VALIDSIG`, with optional bounded
`KEY_CONSIDERED` records. Full fingerprints, key-ID suffix consistency,
algorithm identifiers, binary signature class, timestamp/date consistency,
primary relations and signature expiration shape must agree. Unknown status
names, including failure/expiry/revocation records, refuse. Only version 4
binary document signatures and 40-digit fingerprints are supported. The parser
requires complete, coherent evidence for exactly one signature; missing,
duplicated, ambiguous, contradictory, malformed, fabricated-channel or
unsupported status evidence refuses. Human diagnostic text never supplies a missing machine record.
Unknown signer identities, wrong subkeys, unsupported/disallowed algorithms,
revoked/disallowed policy keys and unverifiable local validity constraints
refuse. Timeout, output exhaustion, launch failure, process crash or input
integrity failure cannot produce a successful partial result.

These are application controls, not production isolation. `gpgv` itself lacks
key retrieval, but this implementation does not certify a kernel-enforced
network sandbox, syscall confinement, interpreter/runtime closure, dynamic
loader/library provenance, hard memory isolation or protection from a hostile
same-UID process. The operating system, chosen binary, shared libraries,
filesystem and execution environment need separate independent acceptance.

## Deliberately incomplete acceptance

Freshness, authenticated time, externally accepted checkpoints and durable
rollback prevention are not implemented by validating an OpenPGP timestamp,
Release `Date`, `Valid-Until`, caller clock or context reference. Full V4b remains
incomplete until independently approved rules and a verifiable delivery/state
mechanism exist. No successful synthetic test closes those requirements.

Every result preserves these project decisions:

```text
Stage 9 = IN_PROGRESS
Observer 3A qualification = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
```

V5 archive-to-image mapping, installation, services, execution of investigated
package code and production-policy changes are outside this implementation.
The adapter runs only the separately selected verification tool, never package
maintainer scripts or inspected binaries.

## Validation obligations

The [retained TEST fixtures](../../tests/fixtures/stage9_observer/apt_v4b/README.md)
contain public keyrings and detached signatures, never production roots or
retained private keys. Use exclusively synthetic TEST materials. Required negative coverage includes
bad signatures, altered Release, unknown signer, wrong subkey/algorithm,
revoked or forbidden key, missing keyring/context, ambiguous signatures, forged
status text, zero exit without proof, timeout, output cap, file mutation, process
crash and atomic refusal after any partial chain progress. Genuine local `gpgv`
verification must be distinguishable from fake-process protocol tests. Neither
fixture key generation nor retained TEST public keys enrolls production roots.

Both new modules and tests enter the existing configured mypy Linux/Win32,
Ruff/format, pytest and branch-coverage scopes. Existing thresholds, timeout
settings and manual hardware gates remain unchanged. Actual executed commands,
results and validation gaps belong in the separate validation report; this
contract alone does not claim successful CI, real repository authenticity,
physical qualification or production readiness.
