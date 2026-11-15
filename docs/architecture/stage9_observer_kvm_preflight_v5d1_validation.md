# V5d1 validation and execution readiness

Date: 2026-11-15, local unauthenticated UTC clock. Base `origin/main` is
`da6605a24c6cb29c2b2638ad6f2c2bb5eb5b361a`, the merge of PR #3276.
The Draft PR identifies the final reviewed revision. This is an implementation
and validation record; it is not a physical VM or execution receipt.

## Delivered mechanism

`IMPLEMENTED`: concrete fixed-URI read-only libvirt access; strict domain
identity/state/configuration comparison with drift detection; independent
requirements pinned by a protected signed external controller policy;
protected artifact/environment identity measurements; held-descriptor raw and
restricted-QCOW2 measurement with immutable-file checks and complete supported
backing-chain expectations; fixed-policy JSON CLI; and a read-only adapter
that refuses execution and whole-VM mutation.

The [operator procedure and trace design](stage9_observer_kvm_preflight_v5d1.md)
provide the exact external policy formats, local Ubuntu diagnostic launch,
supported image/domain profile, and missing independent enforcement and
complete runtime-observation requirements. The
[closure matrix](stage9_closure_matrix_v5d1.json) records each closure gate,
contract, evidence checksum, environment, verifier/expectations provenance,
missing prerequisites and qualification decision.

`SYNTHETIC_TESTED`: the retained automated mechanism results include 241
V5d1 cases. Tests with fake libvirt, substituted host observations or generated
image bytes cannot satisfy physical gates. Existing V5c3 runtime completeness,
collector-origin refusal and synthetic-result semantics remain unchanged.

## Base CI preflight

The [retained GitHub inspection](evidence/stage9_observer_kvm_preflight_v5d1_base_ci.md)
distinguishes exact merge-revision CI from branch/PR evidence. At its recorded
observation, broad native Windows and Ubuntu jobs succeeded on the PR branch
head while the full merge-revision suite still had queued/in-progress jobs.
Pending results were not counted as passes. The previously observed platform
qualification failure is an unchanged fail-closed missing-live-evidence gate,
confirmed against earlier V5c2 CI and unchanged readiness source. It does not
supply the missing physical qualification evidence.

## Local verification record

The [retained local execution summary](evidence/stage9_observer_kvm_preflight_v5d1_local_validation.json)
records exact commands, outcomes, source-artifact hashes and log hashes on
managed Cloud Linux with host CPython **3.11.17**. Its SHA-256 is
`4126e0d77665957a8ba126e9a287fb63e5f664b29102e76f14d7348833bb355e`. This is the trusted host test
interpreter, not the target Ubuntu CPython. Hosted checks on the new PR
revision remain required independently.

| Check | Actual local result |
| --- | --- |
| Observer V1–V5d1, including unchanged V5c3 regressions | **3167 passed, 30 skipped** |
| V5d1 synthetic cases within that run | **241 passed** |
| Workflow and explicit CI-inventory contracts | **177 passed** |
| New V5d1 combined branch-aware coverage | **93%**; line rate 94.75%, branch rate 86.23% |
| Changed-code coverage | **94%**, PASS against unchanged 70% threshold |
| Ruff repository and explicit new deployment scope | PASS |
| Explicit deployment formatting | PASS, 362 files already formatted |
| Exact `python -m pre_commit run ruff-format --all-files` | PASS |
| Configured mypy Linux / Win32 | PASS / PASS, 417 source files each |
| Full pre-commit | PASS, including staged offline secret scan |
| Full baselined offline Betterleaks 1.9.0 | PASS, zero findings |
| Semgrep 1.179.0 custom blocking scope / rule fixtures | PASS, zero findings / 5 of 5 cases |
| Final focused community advisory scan of all five new source files | PASS, zero findings and zero warnings |
| Real operator-host KVM/libvirt and exact target experiment | NOT_TESTED / EXECUTION_BLOCKED |

All 30 skips explicitly require a preprovisioned local `gpgv`; those
cryptographic integration checks were **NOT_TESTED**, not passed. The tests
never download a replacement tool. The 241 V5d1 cases exercise synthetic
libvirt and image/host boundaries, including unapproved policy and tool pins,
configuration drift, backing-chain mismatches, replaced files, unsafe paths,
incomplete observations and attempts to promote synthetic collector status.
They do not qualify a physical host.

The earlier full community advisory scan recorded 26 findings, eight rule
timeouts and one partial-parser warning. Its new XML-parser warning was
resolved by the pinned hardened parser; the final focused scan is clean.
The original full-scope output remains retained with its legacy findings and
coverage limitations. No full-repository advisory-clean claim is made.
No scanner exception, lowered coverage threshold, widened physical execution
permission or disabled gate is part of V5d1.

Codebase Memory MCP/CLI was unavailable in this session. Relevant sources,
contracts and workflow files were inspected directly. That fallback does not
prove structural-index coverage or physical qualification.

## Physical and execution results

| Scope | Result |
| --- | --- |
| Concrete controller and host procedure | `IMPLEMENTED` |
| Approved local Ubuntu/KVM host measurement | `NOT_TESTED`; `HOST_PREFLIGHT_VERIFIED` is not claimed |
| Actual immutable VM image and complete chain measurement on that host | `NOT_TESTED` |
| Independent controller launch/loaded-code identity | `COLLECTOR_IDENTITY_UNVERIFIED` |
| Effective complete host/VM isolation enforcement | `BLOCKED` |
| Complete independently accepted runtime-access/maps/output observation | `BLOCKED` |
| Separate concrete target-experiment authorization | `EXECUTION_NOT_AUTHORIZED` |
| Exact target CPython run | `EXECUTION_BLOCKED` / `NOT_TESTED` |
| PYC reconstruction | `UNVERIFIED` |

No target interpreter, investigated bytecode, maintainer script or privileged
VM operation was executed. Existing libvirt domains were not started,
reconfigured or taken over. Production roots of trust and durable anti-rollback
were not modified. No merge is authorized.

The absence of a suitable KVM host in Cloud is a missing physical qualification
prerequisite, not an implementation-test failure. The read-only CLI may gather
configuration and image diagnostics on a separately prepared host, but cannot
self-award trusted launch, effective isolation or execution authorization.

The first exact CPython experiment still needs independently approved
controller policy and launch evidence; a prepared disposable VM with stable
identities and complete effective isolation/resource enforcement; immutable
images and complete accepted runtime/package/source/output expectations; an
accepted external measurement and output channel with complete trace and
loss detection; reliable whole-VM timeout/stop/cleanup; separate explicit
experiment authorization; and a separately reviewed later real-observation
admission contract. Signature validity, a matching file digest, a synthetic
adapter or a report cannot supply these missing requirements.

```text
Stage 9 = IN_PROGRESS
Observer 3A = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
PYC_RECONSTRUCTION = UNVERIFIED
```
