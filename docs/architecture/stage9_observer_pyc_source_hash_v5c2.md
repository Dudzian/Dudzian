# V5c2: independent source-hash comparison

This inactive, nonproduction Observer extension computes the exact eight-byte
PEP 552 hash from actual source bytes. It does not reconstruct a PYC, validate
its code object or attest compiler execution. V5c1 results, declarations and
production contracts retain their original meaning.

## Local algorithm profile

[`stage9_observer_source_hash_algorithm_v5c2.py`](../../deployment/stage9_observer_source_hash_algorithm_v5c2.py)
implements `independent_source_hash(source, algorithm_id, *, magic, check,
max_source_bytes)`. The supported identifier is
`cpython-3.12.3-ubuntu0.17-siphash13-le-v1`. It uses SipHash-1-3, with
`k0 = int.from_bytes(magic, "little")`, `k1 = 0`, little-endian eight-byte
message words, and explicit eight-byte little-endian output. All four magic
bytes participate, including CRLF; `cb0d0d0a` gives `k0 = 168627659`.
The source is immutable raw bytes, without decoding, newline conversion, BOM
removal or re-encoding. An empty source is supported.

The primitive is independent of host `importlib.util.source_hash`,
`_imp.source_hash`, hash randomization and host magic. Its static vectors come
from a retained, separately compiled C hash-only reference derived from the
exact patched distribution source. Neither implementation executes CPython or
compiles investigated Python. See the
[source-materials report](evidence/stage9_observer_pyc_source_hash_v5c2_sources.md)
and fixtures in
[`pyc_source_hash_v5c2`](../../tests/fixtures/stage9_observer/pyc_source_hash_v5c2/).
Material identifiers and measured SHA-256 values are fixed in the local profile;
the fixture manifest includes the exact archives and all 82 active patches.

Source inspection and independent vectors support the local computation.
Publisher authentication and external acceptance remain `UNVERIFIED`.
The exact Ubuntu interpreter was not run: reference execution is `NOT_TESTED`.
A matching computation therefore has `HASH_MATCH_LOCAL`, with the separate
`algorithm_evidence = ALGORITHM_EVIDENCE_INCOMPLETE` and
`algorithm_acceptance = UNVERIFIED`. No production `SOURCE_HASH_VERIFIED`
status is introduced.

## Inputs and verified byte chain

[`verify_pyc_source_hash_v5c2`](../../deployment/stage9_observer_pyc_source_hash_v5c2.py)
accepts the existing V5c1 expected-image and mapping formats, the independently
supplied expected-image SHA-256, actual immutable DEBs, actual inert PYC inputs,
a V3 measurement claim, the local algorithm identifier, and an explicit tuple
of `(material_identifier, sha256)` pairs. The material list must exactly match
the locally reviewed profile; missing, replaced, additional or ambiguous
materials refuse. Passing that list is a consistency declaration and cannot
authenticate a publisher or approve an algorithm.

The verifier reuses V5c1's strict declaration, full-file, header and source
validators and the **unchanged V5b adapter**. It does not accept a supplied V5c1
result as a substitute for byte verification and does not implement another
archive reader. The chain is:

```text
actual immutable DEB -> complete verified TAR -> exact selected .py bytes
  -> independent eight-byte source hash -> actual verified .pyc bytes 8..15
```

Every source preserves its package, exact version and architecture, DEB
SHA-256/size, source ID, V5a record ID, literal TAR member, source SHA-256/size
and snapshot pin. The PYC preserves full expected-image pin, file ID, path,
complete SHA-256/size, metadata, compiler descriptor digest and fixed header.
The target declaration is additionally restricted to the statically measured
Ubuntu `3.12.3-1ubuntu0.17` interpreter SHA-256
`e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f`
and complete build ID `337d65cf00021797985cc9f77c0cc334a9fbeb38`, as retained in
the [existing runtime closure](stage9_observer_3a_cpython_runtime_closure_v1.md).
Declaration agreement does not show that those interpreter bytes ran.

Only magic `cb0d0d0a`, flags `3` and `checked_hash` are supported. The computed
value must match bytes 8–15 of the remeasured PYC; agreement between a caller's
header declaration and header alone is insufficient. Full-file byte identity
remains independently required, including the opaque body.

## Separate result and atomic refusal

`PycSourceHashResultV5c2` has profile
`stage9-observer-pyc-source-hash-result-v5c2`. Successful file records explicitly
separate `SOURCE_BYTES_VERIFIED`, `SOURCE_HASH_MATCH` and
`BYTE_IDENTITY_VERIFIED` from `RECONSTRUCTION_UNVERIFIED`. Each records computed
and header hashes, source and compiler bindings, algorithm ID and material-list
digest. The aggregate retains inventory, expected image, mapping, snapshot,
V3-result claim and domain-separated source-binding digests, material identities
and unmet requirements. Digests of result claims are associations, not proof of
their origin or execution.

`HASH_MATCH_LOCAL` means every declared file passed the local byte comparison.
`HASH_MISMATCH` refuses inconsistent source/PYC/compiler/snapshot identities or
hashes. `ALGORITHM_EVIDENCE_INCOMPLETE` refuses an absent material declaration;
malformed, unsupported, exhausted or interrupted inputs return `INCOMPLETE`.
An empty bytecode set returns `NOT_DECLARED` without any successful bindings.
Every refusal clears **all** file records, algorithm/material evidence and
aggregate binding digests, including after a first file succeeded internally.

The result always keeps algorithm acceptance, evidence origin, external
authority and time unverified. The missing-proof register applies even to
matching headers with a deliberately invalid code-object body.

## Resource and execution boundary

One shared V5a/V5b budget covers declaration decoding, archive verification,
source capture, hashing, comparisons and private result staging. V5c2 permits
only the existing default limits or tighter limits; none may be enlarged.
The pure hash uses constant scratch space and eight-byte slices, with deadline
checks every 4096 source bytes and before/after hashing. Its own hard 64 MiB
source cap and 60-second monotonic deadline remain active without a callback.
Retained memory accounting is conservative and cooperative; it does not promise
an operating-system RSS cap or preempt a blocked system operation.

No target interpreter, investigated Python, imports from the image, bytecode,
`marshal` payload, package installation, maintainer scripts, full image, root
operation, authority mutation or durable checkpoint write is performed.

## Missing proof and next substage

| Requirement | Current status | Next required evidence |
| --- | --- | --- |
| Source-material publisher authentication | `UNVERIFIED` | Independently accepted exact source/patch provenance and authenticated transport |
| Target algorithm acceptance | `UNVERIFIED` | Review and acceptance under a separately approved authority contract |
| Exact target interpreter differential execution | `NOT_TESTED` | Explicitly authorized isolated reference procedure; identify full interpreter SHA-256/build/package and all dependencies |
| Compiler invocation, execution and isolation | `RECONSTRUCTION_UNVERIFIED` | Approved execution-evidence source and bounded isolation rules |
| Reconstructed full PYC output | `RECONSTRUCTION_UNVERIFIED` | Actual separately authorized run, complete input closure and independently pinned output comparison |
| Code-object structure and embedded filename | `UNVERIFIED` | Reconstruction evidence or separately reviewed bounded parser |
| Runtime qualification/import activation | `BLOCKED` | Separate runtime authorization and qualification |

The [future reconstruction contract](stage9_observer_pyc_reconstruction_v5c2.md)
validates declaration structure only. Even complete matching declarations do
not prove that a run occurred. The next substage must first approve isolation
and evidence origins, then perform a real bounded reconstruction and preserve
its input/output/interrupt evidence. It must not infer execution from this
contract's presence. Stage 3B remains outside that authorization.

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

The [validation report](stage9_observer_pyc_source_hash_v5c2_validation.md)
separates local results, hosted CI, reference execution and qualification.
