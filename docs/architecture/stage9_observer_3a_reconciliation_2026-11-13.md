# Stage 9 / Observer 3A — evidence and remaining scope, 2026-11-13

Review baseline: `d011240f3f5a81b41d1f81ef179efd763c641b30`.
This additive review preserves every historical report, inventory and digest
manifest. It records the operator's new account, reconciles older findings
against current source, and proposes work; it approves no contract or runtime.

```text
Stage 9 = IN_PROGRESS
Observer 3A qualification = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
CONTRACT_APPROVED = NO
STAGE_3A_COMPLETE = NO
runtime_attestation = MISSING
host_pid_namespace = UNVERIFIED
```

## Evidence origin and reconciliation

The [new operator ledger](evidence/stage9_observer_3a_operator_reconciliation_2026-11-13.json)
records seven supplied statements as `OPERATOR_REPORTED_AND_CHAT_REVIEWED`.
The original archive, DEBs, 419-row comparison and collection transcripts were
not supplied for this review. Codex has **not** independently replayed these
host checks. The date above is receipt/review date; the latest checks' execution
time was not supplied. The reported historical archive time is
`2026-10-29T11:04:23Z`, revision
`cad53867231a1a6e49ea770c07043b417af84449`.

The follow-up operator account supplies eight package names and per-package
counts, zero refused packages, control-field PASS for all eight, host repository
revision `d011240f3f5a81b41d1f81ef179efd763c641b30`, and the reported filename
and SHA-256 of `payload-host-observations.json`; the ledger retains these values.
The report itself was not supplied and its digest was not recomputed or
independently authenticated. Operator-reported data, original materials
delivered to Codex, independently reproduced results and externally accepted
evidence remain four distinct categories.

| New operator statement | Existing record and reconciliation | Consequence |
| --- | --- | --- |
| Eight DEBs audited, 8/8 PASS | Reported names are python3.12, python3.12-venv, libc6, libssl3t64, libpq5, libffi8, libnss3 and zlib1g. Exact versions, architectures, individual DEB hashes and raw audit materials remain unavailable. The [exact-build review](stage9_observer_3a_cpython_runtime_closure_v1.md) separately records eight downloaded/extracted packages; the [V3 comparison](evidence/stage9_observer_3a_real_deb_verification_v3.md) covers four CPython DEBs. | Retain a separate reported observation. Do not expand V3's measured four-package coverage or identify the sets by count. |
| 419 package objects compared to host, 419/419 MATCH | Reported per-package counts are 34, 13, 298, 20, 10, 10, 24 and 10 respectively; their sum is 419. V3 retains 1002 control/data records, including directories, links and analysis-only developer material. The 419 objects' types, paths and full comparison semantics remain unavailable. | Aggregate counts are internally consistent; row-level reconciliation and independent replay still require original materials. |
| Archive SHA-256 and manifest SHA-256 MATCH; 14/14 verified | [CPython findings CPY-P2-01](stage9_observer_3a_cpython_findings_v1.md) already retains the correction: 14 manifest entries, 15 regular files including the manifest, superseding an imprecise 13/13 account. Historical archive digest is `0a26027f9bf94149fe2b36d3fbba1fdb6e5393e2016c9bd9956463b154005b0f`; manifest digest is `f52859c04f703bfe6ba36190229e7b18ac1543df5c443d589a549835177c3fdb`. | New account agrees with the corrected count. Digests here are historical reference values, not new Codex measurements. Independent replay remains pending original bytes. |
| Historical SHA-256 of python3.12, _hashlib, _json agrees with inventory | The exact-build review retains separate downloaded-package hashes/Build IDs matching these historical identities. | Existing package-byte facts corroborate the report; they do not authenticate the host capture or prove active runtime selection. |
| Metadata of those three files identical before/after | Extends the reported capture context; raw fields/records are absent. | Narrows reported sampled drift only. Equal endpoints cannot establish absence of intervening changes, protected ancestry or lifetime integrity. |
| Archive time/revision above | Historical reports already pin the revision; the exact collection time is a newly supplied operator assertion. | Keep collection context separate from this review date, current CI and future runtime freshness. |

The [historical V3 observation](evidence/stage9_observer_3a_real_deb_observation_v3.md)
is explicitly `SAME_ORIGIN_OBSERVED_COMPARISON_ONLY; NOT_INDEPENDENT_ACCEPTANCE`.
The historical Ubuntu signature/index acquisition has its own root-bootstrap
and freshness caveats. Neither is upgraded by the operator ledger. Historical
`VERIFIED_FROM_EXACT_BUILD` / `VERIFIED_FROM_STATIC_ELF` labels describe those
retained reviews; this session did not reacquire their original artifacts.

## Contracts and implemented boundaries

The [base plan](stage9_trusted_observer_contracts_and_implementation_plan.md),
[schema proposal](stage9_observer_3a_schema_proposal.md),
[schema disposition](stage9_observer_3a_schema_gaps.md),
[eight-contract audit](stage9_observer_3a_contract_audit.md) and
[final review](stage9_observer_3a_final_review.md) govern an inactive,
`PROPOSED_UNAPPROVED` candidate. No independently approved complete Trusted
Observer operation set was found in this reviewed material. Thus an exact
minimal runtime for an already approved full Observer cannot yet be claimed.

There are eight contract kinds and **seven raw bundle inputs**. The advisory
offline result is not an eighth mutually authenticated input. The standard
`parse_manifest`, `parse_acceptance_policy`, `parse_observer_result` and
`parse_report_frame` still refuse. Proposed APIs remain synthetic and
unverified; syntactic signature membership is not Ed25519 verification.

Former schema findings P1-1/P1-2/P1-3 have implemented candidate validators:
`_validate_proposed_manifest`, `_validate_proposed_acceptance_policy` and
`_validate_proposed_observer_result` in
[the parser](../../deployment/stage9_observer_contract.py). Tests cover exact
records, bounds, canonicalization, cross-document bindings, import shadows,
provenance/ELF DAGs, counters and immutable blocks. Their **code/specification
subtasks are complete candidates**; independent approval and separately
reviewed standard-parser activation remain open.

The [older minimal-runtime report](stage9_observer_3a_minimal_cpython_runtime_scope_v1.md)
describes the offline verifier as absent/future at its historical baseline.
Current source contains [advisory verification](../../deployment/stage9_observer_offline.py),
the [Ubuntu adapter](../../deployment/stage9_observer_ubuntu.py),
[TAR V3](../../deployment/stage9_observer_tar_v3.py),
[package V3](../../deployment/stage9_observer_package_v3.py) and
[the frozen-declaration CLI](../../scripts/verify_stage9_real_deb_v3.py).
These implementations close narrow implementation gaps, including V3's inert
GNU/USTAR records and literal symlink comparison. Legacy adapters retain their
narrower refusals. V3 never resolves those links into an image, authenticates
APT origin, runs maintainer scripts or grants production authority.

No newly confirmed P0 defect was established in this bounded source/test
review. Historical CPY-P0-00 denotes that bounded observation, not a requirement
to mark PASS. No new code defect requiring a parser/policy change was identified.

## Minimal CPython scope: concrete lower bound and completion rule

The [exact import analysis](stage9_observer_3a_cpython_import_closure_v1.md)
and [build materials](stage9_observer_3a_cpython_build_materials_v1.md)
already distinguish module availability from required operations. If the future
Observer reuses the current parser/canonicalizer unchanged, its lower bound is:

1. Repository sources: `deployment/__init__.py`, the parser and shared
   `canonical_json.py`, plus the still-to-be-defined entrypoint/worker sources.
2. Direct imports: `__future__`, `hashlib`, `json`, `re`, `dataclasses`,
   `datetime`, `typing`, `math`. Eager transitive imports include
   `dataclasses -> inspect -> ast/dis/tokenize/linecache`, typing's support
   modules, JSON/re and hashlib initialization. The cited exact import map,
   including source package parents, is required; this direct list is not closure.
3. Startup/encoding: fixed path configuration, bootstrap/importer containers,
   codec registry, `encodings`, aliases, selected filesystem/stdio codecs,
   and `encodings.utf_16_be` for canonical key ordering. Pin actual source or
   frozen origins rather than guessing from a source filename.
4. Full available embedded inventory for the unchanged executable: the retained
   ELF/config evidence establishes **60 builtin registrations**, frozen
   **3 bootstrap / 15 stdlib / 11 test** records. In particular `math`,
   `_datetime`, `_typing`, `_opcode`, `_ast`, `_tokenize`, `_sre`, `_struct`,
   `_blake2`, `_sha1`, `_sha2`, `_md5`, `_sha3` are builtin for these exact
   bytes. They need container bindings, not invented separate `.so` files.
   Unused embedded entries remain available inventory; removing them requires
   a different reviewed build.
5. For the unchanged normal `_hashlib`/`_json` backend, a **nine-object static
   ELF candidate** follows the retained PT_INTERP/DT_NEEDED records:
   `python3.12`, `ld-linux-x86-64.so.2`, `libc.so.6`, `libm.so.6`,
   `libz.so.1`, `libexpat.so.1`, `_hashlib`, `_json`, `libcrypto.so.3`.
   This is not a complete image. `_json` still needs CPython symbol resolution;
   OpenSSL initialization, providers/configuration, actual loader search,
   relocations and later loads remain obligations.
6. Every supported initialization, success, refusal, fallback, exception,
   reporting and shutdown path must terminate inside the selected source,
   native, embedded and data inventory. Hashlib initializes more algorithms
   than SHA-256 and can import logging on constructor failure. Dataclasses
   generates methods inside trusted stdlib. Fixed-code application refusals
   alone do not eliminate these stdlib paths. Unsupported behavior must have
   a reviewed refusal boundary before it can escape the selected closure.

`_ssl`, `libssl`, `legacy.so`, all 46/47 installed extensions, account/DNS
lookup, timezone data and broad diagnostics are not justified as unconditional
parser requirements. This is not proof of safe removal: the chosen operations,
backend/configuration and error paths must settle their inclusion. Preserve
the current DAG restriction; neither SCC nor edge omission is justified by a
hypothetical provider cycle.

The offline toolchain is separate: its CPython **3.11** Zstandard wheel/C
backend lock does not describe the Ubuntu CPython **3.12** Observer ABI.
Offline package decoding adds its own dependencies and provenance obligations;
it must not silently enlarge the Observer's trusted runtime.

The minimal **complete** runtime is the least closed set for a pinned,
independently accepted operation/startup/error policy, including available
embedded containers, source/ELF/config/data dependencies and effective origins
throughout execution. Freeze it only after D1/D6 (entrypoint and OS operations),
D3/D8 (startup/search/lifetime), D4/D5 (backend/fallbacks) and D9
(errors/reporting/state) are resolved. New host hash counts cannot make those
architecture decisions. This report establishes a lower bound and explicit
completion criteria, not an approved final manifest.

## Next concrete implementation proposal

Proceed with **V4 offline APT provenance**, as already ordered by
[V3's remaining scope](stage9_observer_offline_v3.md), in a separate reviewed
change. First specify a versioned trust-context input whose roots, snapshot
identity, permitted signer/subkey/algorithm lifecycle, time/checkpoint and
rollback rules are supplied by an external accepted authority. Missing or
unapproved context refuses; verification must not enroll keys or trust the
machine clock by default. The base plan already specifies existing `gpgv`,
without fetch/keyservers; pin its path/build, keyring bytes and complete offline
toolchain. Its integration and trust/lifecycle semantics require explicit review.
APT archive keys are not Observer acceptance-reviewer keys.

The immediately implementable **V4a** sub-batch is a pure bounded local-chain
parser/comparator, with `signature_verification=NOT_PERFORMED` and
`source_authenticity=UNVERIFIED`, plus TEST-only signature adapter fixtures.
It can proceed without inventing accepted roots or historical-age policy.
Positive source authentication is a later V4b boundary after that external
context and adapter contract are accepted; no such acceptance is recorded here.

The full V4 bounded operation consumes provisioned local InRelease (or an explicitly
supported Release/signature variant), referenced compressed/uncompressed
indexes, exact DEBs and frozen V3 declarations. Verify signature/context and
the complete Release -> index -> package hash/size/name/version/architecture
chain. No acquisition, installation, extraction or target execution is part
of verification. Emit a separate provenance result; retain existing
`source_authenticity=UNVERIFIED` and blocked legacy results rather than
overloading their contracts. Any future positive source-authentication field
needs its own reviewed meaning and must confer no image/host/runtime authority.
Do not implement durable production state or a launcher in this batch.

| V4 regression group | Required positive and negative cases |
| --- | --- |
| Externally rooted signature/context | Frozen TEST-key fixture; missing/unknown/revoked root or subkey, invalid signature, disallowed algorithm, input-selected key, malformed/extra signature material and absent/unapproved trust context refuse. Forged `gpgv` status, exit 0 without required valid status, tool failure, timeout and excess output refuse. Multiplicity/subkey/lifecycle rules require pinned policy. |
| Time/checkpoint/rollback | Accepted fixture time/checkpoint; stale/future/rollback snapshot, unknown state, unauthenticated caller time and absent `Valid-Until` without explicit accepted policy refuse. Fixture state is not proof of durable production replay protection. |
| Complete content chain | Correct compressed and decoded index linkage; changed bytes, truncation, missing link, hash/size/name/version/arch mismatch, duplicate/conflicting fields/records/paths and unsupported index variants refuse. Rebind outer hashes to exercise deeper validation. |
| Resource and I/O failures | Aggregate limits, malformed compression, deadline, short read, path/descriptor substitution and cleanup failure yield no partial successful batch. Cooperative limits remain distinct from a hard sandbox. |
| Authority separation | Full fixture consistency still leaves 3A BLOCKED, 3B NOT AUTHORIZED, production NO-GO; no network/install/subprocess target execution or maintainer/link-target action. Preserve existing V1–V3 contracts and all refusal tests. |

**V5 archive-to-image mapping follows separately**. The current image V1 has
no links: original package links remain inert, and any materialization as a
regular file needs an accepted transform and a closed same-snapshot target.
Specify those transforms, directories, mode/UID/GID and independently selected
expected image without silently broadening V1. In parallel, prepare a
reviewable D1/D3/D4/D6/D8/D9 decision record. Do not activate standard parsers,
change security policy or implement a privileged launcher as part of V4.

Recover existing originals before any new host work: the eight package inputs
and audit, 419-row mapping, original archive/manifest and before/after records.
First replay static integrity and row-level reconciliation using those bytes.
Only a specifically identified residual fact could justify a later authorized
measurement; this review requests no fresh host execution.

## Current regression evidence and limits

The existing `.venv` uses CPython **3.11.17**, pytest **9.2.0**. Without any
installation, the five current Observer suites passed **1332 tests** in
**6.46 seconds**, exit 0, at the review baseline (only documentation is added
by this change):

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=work/observer3a-review/pytest tests/test_stage9_observer_contract.py tests/deployment/test_stage9_observer_offline.py tests/deployment/test_stage9_observer_zstd.py tests/deployment/test_stage9_observer_tar_v3.py tests/deployment/test_stage9_observer_package_v3.py
```

These are actual current Cloud regressions, not reproduction of the physical
host's 8/8, 419/419 or archive results, nor exact-target runtime qualification.
Codebase Memory tools were unavailable; targeted source inspection was used
without a structural completeness claim. An additional **55 tests** of the
V2/V3 reproduction tools passed in **0.51 seconds**, exit 0, using synthetic
fixtures only: **1387 tests in two invocations**. Exact commands, required PR
formatting/style status and limitations are in the
[validation report](evidence/stage9_observer_3a_reconciliation_validation_2026-11-13.md).

## Requirement disposition

The grouped rows retain historical IDs rather than counting overlapping
findings as independent defects. “Close” below always names its limited scope.

| Requirement | Current evidence | Missing element | Proposed solution | Closable from available data? |
| --- | --- | --- | --- | --- |
| P0 review | No new confirmed P0 in bounded inspected paths; current regressions | No repository-wide absence proof claimed | Preserve refusals; investigate concrete counterexamples if found | No artificial PASS or qualification closure |
| Former P1-1/2/3 schemas | Exact proposed validators and regression tests; historical complete-candidate disposition | Independent profile acceptance and later activation review | Record implementation complete candidate; keep standard parsers refusing | Yes for candidate implementation; no for approval |
| CPY-P2-01: original archive and manifest | Operator reports SHA MATCH, 14/14; historical correction and reference digests agree | Original retained bytes/transcript | Recover and statically replay existing material | No independent replay yet; no new host measurement needed |
| New eight-DEB / 419-object observations | Operator reports eight named packages, per-package counts summing to 419, 8/8 PASS, no refusals, control-field PASS and report digest | Exact versions/architectures/DEB hashes, row mapping, full audit semantics and original report/output | Reconcile recovered originals with existing ledgers; reported digest is not independently authenticated | No for independent confirmation or complete closure |
| MR-P1-02 registration subproblem | Retained exact ELF tables and config/frozen cross-checks: 60; 3/15/11 | No new symbol-count measurement needed for these exact bytes | Bind existing tables/containers into selected image | Yes for recorded static registration facts; not accepted image/startup |
| CA-P1-01 / CPY-P1-01 / DELL-P1-01: accepted build/image | Exact package/source/static evidence plus operator account | Independent expected-image selection, accepted provenance/reproducible linkage and complete closure | V4 provenance, later V5 mapping, external image/build acceptance | Partly narrowed; no full closure |
| MR-P1-01 / CPY-P1-03: entrypoint/operations | Current library sources and future plan | D1/D6 operation, success/refusal/error/reporting specification | Separate decision record and pinned source boundary | No; architecture decision, not host evidence |
| MR-P1-03: backend/provider/error paths | Exact hashlib/import/native placement and preliminary graph | D4/D5 selected configuration, native init, fallback/data closure | Analyze selected backend and all supported paths; explicit review | No; decision plus artifact analysis |
| MR-P1-04: ELF/DAG compatibility | Retained partial static edges; structural DAG rejection tests | Complete selected graph, search contexts, symbol scope and real required dynamic loads | Resolve exact objects/contexts; retain DAG until reviewed counterexample | No; a nine-object candidate is not completeness proof |
| CA-P1-02 / MR-P1-06: authority and freshness | Signing preimage, policy syntax and historical bounded APT evidence | External reviewers/roots, Ed25519 verification, lifecycle/time/checkpoint/durable rollback | Separate trust/receiver design; V4 only its explicit APT boundary | No; package agreement supplies no acceptance authority |
| CA-P1-03 / CPY-P1-02 / MR-P1-05 / DELL-P1-02: launch/scope/lifetime | Declarations and static observations | Authenticated pre-Python launch, executed=verified lifetime, host/run/channel/scope, enforced resources | Separately reviewed implementation and later authorized qualification | No; existing tests cannot establish host enforcement |
| Offline V1–V3 capability | Implemented staged checks and full inert package/TAR comparison; current regressions | V4 provenance, V5 image mapping and independent tool/expectation acceptance | Keep version-specific scope; build V4 next | Yes for implemented comparison capability; no authenticity/image/runtime closure |
| CA-P2-01–04 / CPY-P2-02–04 | Explicit offline-wire exception, synthetic precedence, cooperative budgets and static hardening tags | Reviewed wire/hardening choices; resource/replay/receipt/crash protocols and later enforcement evidence | Preserve classifications and handle separately; no policy change here | No full closure; do not inflate into newly confirmed P0/P1 defects |
