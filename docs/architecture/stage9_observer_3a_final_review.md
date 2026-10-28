# Stage 9 / 3A: final candidate review and future activation plan

Review date: 2026-10-28. PR: [#3215](https://github.com/Dudzian/Dudzian/pull/3215).
Base integrated: `52ea173221572364e21263433a2623c77af8ed3a` (platform policy V2).
Implementation reviewed through `4cc16d9cd7c4586a70822945a90dbf3e99b1d0b6`.
The PR description identifies the final documentation commit and CI snapshot.

This is a source review of an inactive candidate. It is not independent
approval of its security contract or evidence of an installed runtime.
The [proposed profile](stage9_observer_3a_schema_proposal.md) is normative
only within the proposal. Its status remains `PROPOSED_UNAPPROVED`.
The [earlier finding disposition](stage9_observer_3a_schema_gaps.md) continues
to distinguish implementation completeness from approval.

## Contract assessment

| Contract | Source assessment | Remaining authority boundary |
| --- | --- | --- |
| `ObserverManifestV1` | Exact nested target/source/file/provenance/transform/runtime/import/embedded records; typed references, unique paths/IDs, declared inventory closure, separate provenance and ELF DAGs, image/build/container bindings and launch projections. | No artifact contents, actual build inventory, import resolution or runtime measurement verified. Exact target ELF compatibility remains an approval blocker. |
| `AcceptancePolicyV1` | Exact target/source projection, 22 fixed limit fields, distinct reviewer keys/identities, threshold two, version/root/checkpoint/window/document bindings. | JSON cannot establish trusted reviewers, independent approval, revocation freshness or trusted time. |
| `ObserverResultV1` | Ten bounded counters with explicit overlap rules, three exact booleans, cumulative sequence consistency, fixed missing-proof reasons and qualification blocks. | Declared observations are synthetic; neither zero matches nor complete transport proves runtime enforcement. |
| `LaunchPolicyV1` | Closed launch/profile structures and exact manifest projections for interpreter, ELF/import inventory, image and profile references. | Structural policy validation does not install a launcher or prove any setting was enforced. |
| `ObserverRunContextV1` | Exact run/nonce/boot/scope/digest identities and bound result/context sequence; raw document digests are checked in the proposed bundle. | Caller bytes and fixture time are not an authenticated channel, producer, clock or fresh session. |
| `HostScopeBindingV1` | Closed scope and namespace identities, explicit missing host proof, exact digest/run binding. | Host PID namespace remains `UNVERIFIED`; no host scan or authenticated host measurement occurs. |
| `OfflineVerificationResultV1` | Closed fields and immutable blocked qualification status; accepts no self-authorizing positive result. | Offline result remains `UNVERIFIED`; it does not authenticate acceptance signatures or supply trust roots. |
| `AcceptanceV1` | Entire envelope must already be canonical; exact payload/checkpoint/reviewer membership, binding digests and signing preimage are preserved. | Signature syntax/membership is not cryptographic verification, signer independence or trusted revocation state. |

The three previous P1 schema gaps are technically specified and implemented
for review. Their independent acceptance remains open. Diagnostic vocabulary
is exactly 17 refusal codes, separate from status words. Identifier grammar
is `[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}` with no normalization. The shared
canonicalizer applies to whole documents, including the acceptance envelope;
the signing message remains its existing domain (one trailing NUL) followed
by 32 raw SHA-256 bytes.

The bundle reparses raw documents and checks source, target, build, launch,
policy, acceptance, run, result and scope relationships. Metadata digest
exclusion applies to declared fields; opaque bytes and hidden references
require later artifact verification. Standalone parsers cannot establish
cross-document relationships without those other documents.

## Import and ELF findings

Source/package names bind to exact paths; parent source packages must be
explicit in the same root. Builtin/frozen declarations bind to their declared
binary container and build identity. Native module paths bind to the declared
target ABI. Shadow candidates are checked across every declared import root.
Inert `.pyc` is allowed only as data outside import roots and executable/profile
roles. A declaration never proves the actual runtime imported those bytes.

This review found a concrete missing shadow check: a file labelled as data
could be named `helper.abi3.so`, `helper/__init__.so`,
`helper/__init__.cpython-312-x86_64-linux-gnu.so` or
`helper/__init__.abi3.so` beside a declared `helper.py`. CPython can resolve
these native competitors. The candidate now rejects them. It counts only
regular files as module candidates and deduplicates suffixes, avoiding a false
shadow from a data directory called `helper.so`. Native packages remain
unsupported; this change does not activate them. Negative tests cover all
four paths, with a positive directory case.

The work ceiling `(files + imports) * import_roots <= 65536` is still checked
before nested resolution comparisons. At most ten distinct fixed suffix
forms are examined per import/root pair. No limit field or quality threshold
was loosened.

Provenance dependencies and ELF dependencies are distinct directed graphs.
The ELF graph includes declared interpreter/native/library/dlopen/hwcaps
binary roles. New negative tests verify that optional binary roles cannot
hide a cycle and that loader configuration cannot serve as an ELF binary
node. These tests establish structural rejection properties.

**Open P1 contract-approval blocker:** there is no independently accepted,
complete dependency/build inventory for the exact CPython 3.12 x86_64 build on
Ubuntu 24.04 targeted by this profile. Synthetic acyclic/cyclic fixtures do
not establish that all required real dependencies satisfy this DAG model.
No target artifact was executed or qualified in this task. Consequently this
review cannot certify compatibility or justify weakening the DAG. Approval
must either accept the existing restriction against that exact inventory or
approve a bounded revised model with real counterexamples, threat analysis,
complete edge semantics and positive/negative regressions. Actual artifact
verification remains separate Stage 3B work and is not activated here.

## Future activation change: prerequisites and sequence

The following is a plan, not authorization to perform activation.

1. Obtain independently recorded acceptance of the entire versioned profile,
   including the three former P1 groups, diagnostic codes, identifier grammar,
   canonical envelope, all limits, import resolver restrictions and exact-target
   ELF decision above. Pin the accepted revision/digest and reviewers outside
   the submitted bundle. A policy or acceptance JSON must not approve itself.
2. Resolve incompatible wire decisions before activation. Preserve existing
   canonical vectors, payload digests and signing-domain vectors. Any approved
   incompatible profile gets an explicit version/migration decision. Do not
   normalize, re-sign or relabel historical evidence as newly accepted evidence.
3. Make a separate, reviewed parser-activation change. Replace unconditional
   refusal in `parse_manifest`, `parse_acceptance_policy`,
   `parse_observer_result` and `parse_report_frame` only for the independently
   accepted profile. Reuse the shared decoder, canonicalizer, bounded validators
   and binding logic; avoid duplicated schemas. Reconcile the current refusal
   field subsets with the full accepted records rather than layering them as
   conflicting checks. Wire the standard frame parser to the accepted result
   validator and exact expected run context. No environment flag or fallback
   may select an unapproved contract.
4. Keep explicit `parse_proposed_*` review APIs permanently labelled
   `PROPOSED_UNAPPROVED`, `synthetic=true`, `authorization=UNVERIFIED`.
   Standard structural parsing must also remain `authorization=UNVERIFIED`.
   Successfully decoding an approved schema must not create production trust,
   remove missing-proof reasons or upgrade qualification. Preserve immutable
   outputs and prohibit a proposed-record-to-authority conversion.
5. Run the negative matrix below and the unchanged architecture/security gates.
   Review all consumers before connecting any standard parser to an evidence
   exporter or production interface. No such consumer is added in this PR.
6. Keep cryptographic trust-context/signature verification, independent time and
   revocation state, actual artifact/build/import/ELF verification, authenticated
   run channel, runtime attestation and authenticated host PID scope in their
   separately authorized verification/qualification work. A future verifier
   must consume externally rooted trust, never keys merely declared by input.
   Parser activation alone authorizes none of these actions or Stage 10/LIVE.

Required activation regression matrix:

| Boundary | Required positive/negative evidence |
| --- | --- |
| Wire decoder | Accepted canonical vectors; malformed UTF-8/JSON, duplicate/unknown/missing fields, noncanonical envelope, float/bool-as-int, NaN/infinity, size/depth/count limits and malformed identifiers refuse. |
| Manifest | Exact accepted records; dangling/incorrectly typed references, path aliases/links/duplicates, incomplete inventory, provenance cycles, unsupported transforms and build/container mismatches refuse. |
| Runtime | Accepted exact-target ELF profile; missing/extra edges, forbidden cycles and non-binary nodes refuse; import parents, builtin/frozen/native binding, all shadow paths and forbidden `.pyc` refuse; resolution work limit remains bounded. |
| Policy and acceptance | Exact source/target/launch projections; all 22 constants, reviewer/key identity uniqueness, threshold, digests, version, root/checkpoint/sequence/window and metadata back-references refuse on mismatch. Signature syntax never implies validity. |
| Result and sequence | Every counter boundary/overlap and exact boolean; dropping positive observations, instability or diagnostics, changed run/boot/nonce/scope/digests, replay/reordering and cumulative rollback refuse. |
| Frame and channel | Exact header/body length and cap, truncation/trailing/duplicate frame, invalid result, wrong expected context and mismatched run refuse. Bytes alone never authenticate a sender. |
| Authorization | Proposed APIs retain all labels; standard parser cannot self-approve, trust caller time, clear blocked flags, create host proof or satisfy production trust. Existing forged-context and architecture-isolation tests remain mandatory. |
| Migration | Old unsupported/unapproved profiles refuse; no compatibility fallback, automatic normalization, retrospective re-signing or historical qualification upgrade. |

Independent approval and a separately reviewed activation commit are explicit
exit criteria. No production capability is enabled by this document.

## Current disposition

No remaining verified P0 implementation defect was identified in the reviewed
synthetic paths. The import shadow defect is fixed; the three schema P1 groups
are implemented but await approval. The exact-target ELF evidence decision is
an open P1 approval blocker. Earlier P2 vocabulary/identifier/canonicalization
findings are addressed within the proposal. No additional verified P2/P3 code
finding was established. This assessment is bounded by source inspection and
the reported tests, not a proof of complete security.

`parse_manifest`, `parse_acceptance_policy`, `parse_observer_result` and
`parse_report_frame` still refuse. No new production caller is introduced.
An inactive implementation may be reviewed for a conditional code merge;
that does not close 3A or authorize production. CI must separately satisfy
its required checks. The [CI remediation report](../audits/stage9_3a_ci_remediation_2026-10-28.md)
records test failures, timeout evidence and missing physical qualification.

```text
Stage 9 = IN_PROGRESS
qualification = BLOCKED
full_confirmation = false
coexistence_authorized = false
preflight_exception = false
physical_test_status = PHYSICAL_TEST_PENDING
production_status = PRODUCTION_NO_GO
runtime_attestation = MISSING
host_pid_namespace = UNVERIFIED
```

No Stage 41B, Root Proof, production enrollment, privileged helper/launcher,
Stage 10, LIVE or existing `--evidence` semantics are changed.
