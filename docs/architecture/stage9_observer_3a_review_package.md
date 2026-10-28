# Stage 9 / 3A — independent review package R1

Date: 2026-10-28. Status: **PROPOSED_UNAPPROVED**.

This package prepares the inactive Observer V1 proposal for independent review;
it does not record independent approval. Exact Ubuntu 24.04 x86_64 / CPython
3.12 build materials and a complete authenticated runtime inventory remain
missing. Final approval and Stage 3A completion are blocked.

## Exact revision and SHA-256

The normative candidate is main `ae712aea85e4cbc372bdf226d6a5bc5be02f4949`
(merged PR #3215): the [base design](stage9_trusted_observer_contracts_and_implementation_plan.md),
[V1 profile](stage9_observer_3a_schema_proposal.md),
[earlier findings](stage9_observer_3a_schema_gaps.md),
[previous final review](stage9_observer_3a_final_review.md),
`deployment/stage9_observer_contract.py`, its canonicalizer and contract tests.

R1 adds review evidence and proposed decisions. It changes no wire schema,
parser, test, trust root, workflow, threshold or activation state. The review
PR's full Git HEAD identifies the delivered revision. Exact document, contract
source and test-material hashes are in the [SHA-256 manifest](evidence/stage9_observer_3a_review_r1.sha256).
Verify from the repository root:

```bash
sha256sum --check docs/architecture/evidence/stage9_observer_3a_review_r1.sha256
```

The manifest excludes itself. Authenticate its selection through an independently
obtained Git revision/review record or external digest pin. A local hash or
archive cannot authenticate its own selection. No approval signatures or
attestations are generated.

## Review materials and evidence limits

| Material | Purpose and limit |
| --- | --- |
| [Post-merge CI](stage9_observer_3a_ci_review.md) | Actual merge-SHA push-run IDs and timestamped job evidence. Pending is not PASS; earlier PR failures are separate. |
| [Eight-contract audit](stage9_observer_3a_contract_audit.md) | Closed schemas, canonicalization, references, trust, signatures/revocation/replay, limits, imports, privileges and explicit assumptions. |
| [ELF qualification](stage9_observer_3a_elf_qualification.md) | Twelve requested resolution topics; primary-source and non-target static evidence; DAG disposition; unimplemented bounded cyclic alternative and threat/test matrices. |
| [Dell procedure](stage9_observer_3a_dell_readonly_diagnostics.md) | Separate voluntary read-only operator collection, not executed on the laptop and not an independent attestation. |
| [Previous CI remediation](../audits/stage9_3a_ci_remediation_2026-10-28.md) | Historical causes/fixes in #3215, not this merge's hosted outcome. |

Codebase Memory was unavailable in connected tools and the executable path.
No index coverage/freshness claim is made. Named sources/tests/workflows were
read directly. This bounded review is not an exhaustive security proof.

## Decisions submitted for independent acceptance

| ID | Recommendation, not an enacted normative change |
| --- | --- |
| D1 | Retain the inactive parser's DAG restriction. Qualify it only against complete authenticated exact-target evidence; otherwise commission a separately versioned typed/SCC revision against a required real counterexample. ELF validity and completeness do not follow from acyclicity. |
| D2 | Treat current untyped edges solely as synthetic declarations. Decide exact PT_INTERP/DT_NEEDED/native/dlopen semantics, request order/context, search scope, cache/config and CPU/hwcaps closure before claiming actual resolution. |
| D3 | Keep provenance DAG separate from runtime topology. A future ELF SCC cannot relax provenance, completeness, object identity or lifetime protection. |
| D4 | Accept independently expected image/build/embedded/native inventory and its authentication chain; supplied role/graph closure is only declared closure. |
| D5 | Explicitly decide the offline-result canonical exception: seven named evidence inputs require canonical raw bytes; offline parsing accepts noncanonical raw bytes and retains its raw SHA-256 separately from canonical bytes. Do not claim all eight are raw-canonical or silently tighten the wire. |
| D6 | Explicitly decide offline output/bundle scope: the proposed bundle binds seven inputs; standalone synthetic offline output is advisory. Any canonical/authenticated/bundle-bound receipt needs an approved revision. |
| D7 | Record precedence: proposal Stage 3A UNVERIFIED/INCOMPLETE restrictions override the base design's future COMPLETE/VERIFIED examples for this candidate. |
| D8 | Preserve signing preimage, 22 fixed limits, import restrictions and standard refusals. Approve version/migration, external roots/reviewer authority and revocation/time/replay protocols; demonstrate separately enforced resource budgets. |

Reviewers must resolve normative choices in the pinned revision before final
acceptance. The contract audit lists twelve explicit assumption groups; ELF
analysis adds exact resolver/build assumptions and proof obligations.

## Threat model

The adversary may supply canonical but false documents, archives, ELF files,
native modules, configuration, shadow candidates, forged signatures, stale
checkpoints and replayed results. It may hide dependencies or replace files
after inspection. The host producing inventory is not its own independent
trust root. Assets are image integrity/completeness, actual interpreter/build/
loader identity, evidence bindings, approval and the authorization boundary.

| Threat | Synthetic defense | Outstanding proof |
| --- | --- | --- |
| Malformed/ambiguous/resource-exhausting inputs | Closed records, strict byte/depth/count bounds, canonical evidence inputs, iterative graphs and import work budget. | Worst-case CPU/RSS and externally enforced limits on the accepted build. |
| Substitution, hidden native/dlopen dependency or fallback | Unique declared IDs/paths/digests/roles and in-image policy. | Authenticated expected contents, contextual resolution completeness and protected objects/ancestors/mounts throughout delayed loads. |
| Import/startup injection | Exact lexical maps, parents, shadow checks and isolated argv declaration. | Actual CPython embedded/native/suffix inventory and effective exclusion of hooks, .pth, executable bytecode and external roots. |
| Self-approved keys, stale/replayed results | Signing domain, digest/root/checkpoint/run consistency and UNVERIFIED labels. | External roots, valid signatures/distinct reviewers, fresh revocation/time, durable replay/rollback state and authenticated channel. |
| Incomplete host scope or privilege escalation | Fixed missing-proof/production blocks; standard manifest/policy/result/frame refusal. | Separately authorized launch/host/physical qualification; none enabled here. |

Compromised kernel, diagnostic tools, root distribution or reviewer process
remain explicit assumptions. Static metadata cannot prove which bytes ran;
hashing a declaration cannot prevent replacement or external fallback.

## Test matrix and available results

| Boundary | Existing evidence | Outstanding evidence |
| --- | --- | --- |
| Decoder/eight validators/seven-input bundle | Positive, malformed, limit and mismatch regressions. | Independent vector review and offline-wire decision. |
| Canonical signing/references | Envelope/preimage/projection/binding negative tests. | Externally rooted cryptographic/revocation/time verification in later authorized work. |
| Provenance/ELF | Declared DAG/shared-node positives; cycle/dangling/wrong-role negatives. | Exact-target resolution; typed/SCC matrix is planned, not implemented or executed. |
| Imports | Source/native/embedded/parent/shadow/work-budget tests. | Authenticated actual build inventory and effective runtime behavior. |
| Authority/incompleteness | Refusal, labels, counter retention and fixed blocks. | Preserve negative boundaries after any separately reviewed activation. |
| Resources | Byte/depth/count/comparison-budget refusal tests. | Maximum-shape CPU/RSS and hard timeout/memory enforcement. |
| Physical runtime | No Dell or production execution. | Separate operator decision and authenticated expected materials; diagnostics alone cannot qualify. |

Local checks used development **CPython 3.11.16**, not target CPython 3.12:

- Contract tests: **869 passed in 13.49 s**, explicit asyncio/pytest-cov plugins,
  plugin autoload disabled, existing hardware/destructive/network exclusions.
- Contract coverage: **994/997 lines (99.70%)**, **229/232 branches (98.71%)**,
  combined **99.51%**. Missing statements: unchanged source lines 93, 257, 295.
- Exact `python -m pre_commit run ruff-format --all-files`: PASS, no tracked
  reformatting; explicit deployment contract/test formatting: PASS.
- Configured mypy Linux and `--platform win32`: PASS, **359 files** each.
- Tier A Ruff scope plus deployment contract: PASS.
- Semgrep 1.179.0 custom blocking scan: 5 rules, 1085 targets, zero findings;
  fixtures 5/5 PASS. Standard size/ignore exclusions retained.
- Betterleaks 1.9.0 tracked/current-review snapshot: zero leaks; workflow-pinned
  and release checksums verified. Local tooling/Git history excluded. Final
  review-document edits receive a separate staged scan.

Detailed logs and raw coverage accompany the delivered report. Community
Semgrep and CodeQL remain hosted evidence; no full local CI or physical
procedure was run. Coverage measures synthetic paths, not artifact truth.
This documentation-only change has no executable diff coverage; the existing
70% CI ratchet and all quality/security thresholds remain unchanged.

## All open findings and readiness

| Severity | Findings requiring disposition |
| --- | --- |
| P0 | None newly confirmed in reviewed scope; not proof of absence. |
| P1 | CA-P1-01 exact-target inventory/ELF; untyped-edge resolution/completeness; CA-P1-02 external authorities/crypto/time/checkpoint; CA-P1-03 effective build/import/file/launch closure. |
| P2 | CA-P2-01 offline canonical/bundle scope; CA-P2-02 precedence; CA-P2-03 resource enforcement; CA-P2-04 receiver replay/receipt/crash protocol. |
| P3 | CA-P3-01 frozen dataclasses are not an in-process security boundary. |

See contract/ELF audits for complete detail and explicit assumptions. P1 items
block their stated acceptance/verification boundary. P2 decisions need explicit
disposition and future work needs an accepted interface and retained block.
Pending CI is a validation gap, not automatically a code regression. No new
synthetic code defect was demonstrated; no speculative parser fix was made.

| Status | Current disposition and exit condition |
| --- | --- |
| CODE_READY | **PENDING_CI** — local checks pass; required hosted checks of the submitted revision remain to be completed and assessed. |
| CONTRACT_REVIEW_READY | **BLOCKED for final approval** — packet assembled; exact-target materials and normative decisions missing. Independent issue review can start. |
| CONTRACT_APPROVED | **NO** — designated independent process must accept the pinned final revision and dispose blockers. |
| STAGE_3A_COMPLETE | **NO** — independent acceptance, validation and 3A gate closure outstanding. |
| PRODUCTION_QUALIFIED | **NO** — runtime/host/physical evidence absent. |

## Formal approval and subsequent activation

The designated independent authority must obtain this package through an
authenticated channel, verify revision/document pins, review all eight contracts
and matrices, receive authenticated exact-target expected materials and record
every decision/finding disposition. The approval must externally identify
reviewer authority/independence and pin profile/version, document digests,
root set, checkpoint and effective conditions.

The reviewed repository does not name a person/service for this R1 approval;
the project owner must designate the authority through real governance.
No reviewer, signature or attestation is invented. Threshold two in input does
not prove that process happened. Acceptance must verify against external keys
and freshness/revocation state, never roots supplied solely by the bundle.

Next: independent issue review and selection/authentication of one exact Ubuntu
build snapshot. The operator may separately choose Dell read-only collection;
an independent custodian's expected actual-image inventory must accompany it.
Resolve P1 ELF/edge completeness and offline/precedence decisions before final
acceptance. If required cycles exist, commission the separately versioned
typed/SCC revision and execute its positive/negative tests. Do not merely drop
the cycle check.

Only after formal acceptance and required 3A gate closure may a separately
reviewed change consider standard-parser activation using the exact accepted
profile and previous review's full negative matrix. No flag/fallback/automatic
normalization or retrospective upgrade may select an unapproved profile.
Structural acceptance must remain UNVERIFIED. Activation alone supplies no
runtime trust or permission for Stage 3B execution.

```text
Stage 9 = IN_PROGRESS
qualification = BLOCKED
production_status = PRODUCTION_NO_GO
runtime_attestation = MISSING
host_pid_namespace = UNVERIFIED
full_confirmation = false
coexistence_authorized = false
preflight_exception = false
physical_test_status = PHYSICAL_TEST_PENDING
```

No Stage 3B, privileged helper, trusted launcher, Root Proof change, legal
enrollment, Stage 10, LIVE, historical payload or physical test is executed or
authorized. No automatic merge is permitted.
