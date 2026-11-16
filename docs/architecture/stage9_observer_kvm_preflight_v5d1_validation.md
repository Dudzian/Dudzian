# V5d1 validation and execution readiness

Date: 2026-11-16, local unauthenticated UTC clock. P2 remediation of reviewed
HEAD `05a41820c6f245acc6c891ce0f4a65441f3412f5`. Base `origin/main` is
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

`SYNTHETIC_TESTED`: both reviewed P2 issues are remediated. The XML boundary
requires explicit no-restart lifecycle and a documented fail-closed profile
for every recognized element/attribute. Unknown or unevaluable semantics,
duplicate singleton nodes and live/persistent semantic drift refuse
`config_matches`. Image records are provisional through complete-chain checks
and descriptor cleanup; the aggregate publishes all chains together only after
final domain, requirements, identity and connection-cleanup checks. Every
handled failure discards positive image records while retaining reason codes.
The [operator profile](stage9_observer_kvm_preflight_v5d1.md) lists values,
accepted defaults, missing-proof refusals and interruption/cleanup diagnostics.

## CI baseline and exact-head boundary

The [retained GitHub inspection](evidence/stage9_observer_kvm_preflight_v5d1_base_ci.md)
confirms that `windows-release-gate` and `linux-deployment-integration` failed
on reviewed HEAD `05a41820` because current physical qualification evidence is
missing. Their log categories also occur on PR #3276 HEAD `43447847`.
These existing blockers are unchanged, remain failures, and are separate from
the successful local software checks below. No workflow or dependency gate is
weakened. Exact new-HEAD Actions must be inspected after push and recorded in
PR #3277; this pre-push record does not count pending jobs as passes.

## Local verification record

The [retained local execution summary](evidence/stage9_observer_kvm_preflight_v5d1_local_validation.json)
records exact commands, outcomes, final source/configuration hashes, JUnit and
log hashes on managed Cloud Linux with host CPython **3.11.17**. Its SHA-256 is
`079dbb3b972e0d1d8560f75772e3654ce82a917436d947a81f6f2fe2d9c1e112`. This is the host test
interpreter, not the investigated target interpreter. The previous local
record is retained in Git history and identified by its digest in the new one.

| Check | Actual final local result |
| --- | --- |
| Full Observer V1–V5d1 regression | **3386 passed, 0 failed, 32 skipped** |
| V1–V5c2 within that run | **2727 passed, 31 skipped** |
| Unchanged V5c3 within that run | **201 passed** |
| V5d1 within that run | **458 passed, 1 skipped** (459 collected) |
| New P2 regressions within that run | **208 passed**: 179 XML, 29 image/aggregate |
| Workflow/dependency and dependency-authority contracts | **284 passed**, 0 failed/skipped |
| Distinct pytest total, without counting repeated focused runs | **3670 passed, 0 failed, 32 skipped** |
| Five-module branch-aware coverage | **94%**; line rate 95.37%, branch rate 89.44% |
| P2 changed-line coverage / whole-PR changed-line coverage | **100% / 94%**, unchanged **70%** threshold |
| Repository Ruff and explicit deployment Ruff | PASS |
| Exact workflow explicit format scope | PASS, **362 files** already formatted |
| Exact `python -m pre_commit run ruff-format --all-files` | PASS |
| Configured mypy Linux / Win32 | PASS / PASS, **417 source files** each |
| Full pre-commit | PASS, all hooks |
| Full baselined offline Betterleaks 1.9.0 | PASS, **zero findings** |
| Blocking Semgrep 1.179.0 / rule fixtures | PASS, **zero findings**, 5 rules on 1149 files / **5 of 5** fixtures |
| Canonical desktop-lock pip-audit | PASS, **104 dependencies**, zero vulnerabilities, no ignore/fix |
| Physical operator-host qualification and exact target experiment | **NOT_TESTED / EXECUTION_BLOCKED** |

The skips are **31** cryptographic integration cases requiring a preprovisioned
local `gpgv`, and **one** native unsupported-host refusal case in V5d1. They
were not passed or replaced by downloaded tools. All 208 new P2 regressions
passed. Inert host `/bin/sleep` children exercise kill/reap/descriptor cleanup;
they do not execute QEMU, the target interpreter or investigated code. Both
complete positive chains and all requested late-failure paths are covered.

Early checks against in-progress edits and an all-files pre-commit run during
documentation edits were superseded by the final successful checks. Initial
secret-scan findings were only generated ignored test caches and a downloaded
public libvirt reference; these intermediate files were preserved outside the
checkout. No scanner baseline, rule, ignore, threshold or required scope was
changed. The JSON records those resolved attempts separately.

The full community advisory Semgrep scan was **not rerun for P2**; historical
advisory findings/timeouts are not a clean-current-tree claim. Required
blocking rules and fixtures passed on final source bytes. Codebase Memory
MCP/CLI was unavailable; direct source and workflow review was used instead.
Independent review found no remaining observed P0/P1/P2 in the remediation
scope. Physical qualification and execution prerequisites remain open.

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
