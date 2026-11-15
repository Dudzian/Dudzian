# V5c3 validation and Windows V5c2 checkout repair

Date: 2026-11-15 (local unauthenticated UTC clock). Base `origin/main`:
`cca8e8370d0124bb651a4722942843b14b559330`, re-fetched before final validation.
Branch: `codex/stage9-v5c3-isolated-reconstruction`. The PR records the final
HEAD. This document is a validation summary, **not execution evidence**.

## V5c3 implementation and execution status

Implemented: the guarded fixed-path guest compiler; bounded ELF64 amd64
inspection and runtime inventory comparison; live outside-controller API;
V5b DEB/TAR/source capture and independent V5c2 SipHash; separately pinned
request plan and inventories; actual opaque PYC byte measurements; complete
versioned process records and canonical export; atomic invalidation on failure.
See the [contract and operator procedure](stage9_observer_pyc_execution_v5c3.md).

**Exact execution: EXECUTION_BLOCKED. Exact target test: NOT_TESTED.**
No target interpreter, investigated module or generated PYC was executed.
No VM, privileged infrastructure, host DEB installation or maintainer script
was invoked. No real local execution observation, independently bound record
or accepted reconstruction evidence is claimed.

The available Cloud workspace has enforced unrestricted network egress and a
forwarded GitHub credential. Read-only capability checks found no `virsh`,
`qemu-system-x86_64` or `/dev/kvm`; a host Docker socket exists. It is not a
qualified offline disposable VM. Merely running inside a managed Cloud
workspace cannot satisfy the requested isolation boundary. Existing Stage 9
VMs were not modified or started.

Required missing evidence:

| Requirement | Current state | Needed to unblock |
| --- | --- | --- |
| Qualified disposable offline VM | MISSING | Outside inspection of effective network/storage/shares/secrets/socket/production boundaries and resource enforcement |
| Approved collector identification | MISSING / BLOCKED | Separately approved implementation that verifies identity independently; labels, names, process declarations, nonces and report checksums are not authentication |
| Concrete outside-guest collector | MISSING | Reviewed integration for complete frozen image capture, bounded process execution, complete runtime accesses/maps, actual output capture and whole-VM stop |
| Complete exact runtime bytes and inventory | MISSING | Exact interpreter, loader, libraries, compiler imports/native modules/configurations and package/snapshot bindings |
| Independent real reconstruction plan | MISSING | Accepted local source/snapshot/target/optimization/dfile and independently expected complete PYC digest; synthetic fixture plans do not qualify |
| Level B provenance registry | UNVERIFIED | Independently maintained origin/run/integrity registry |
| Level C acceptance authority/policy | UNVERIFIED | Separate approved authority and acceptance decision |
| Authenticated time / durable anti-rollback | UNVERIFIED / BLOCKED | Separate qualified clock and durable state mechanism |

## Local tests and quality

Tests run with trusted host CPython **3.12.14**, not the target 3.12.3 Ubuntu
binary. Worker tests compile only synthetic sources; their independent
reference construction uses builtin `compile` plus `marshal.dumps`, never
`marshal.loads`, and never runs the resulting code. Orchestrator tests use a
real synthetic DEB and a clearly synthetic collector; its runtime validation
stub cannot establish target identity. Separate static runtime tests exercise
actual byte parsing, malformed ELF, aliases, library/import/config pins and
bounded manifests. No synthetic result is promoted to a target observation.

| Check | Result |
| --- | --- |
| Observer V1–V5c3 and explicit CI inventory regressions | 3021 passed, 30 skipped |
| V5c3 mechanism tests in that run | 201 passed (worker 40, orchestration 100, runtime identity 61) |
| New V5c3 branch-inclusive coverage | 97% combined; worker 95%, orchestration 96%, runtime identity 98% |
| Changed-code coverage gate | PASS: 97%, unchanged required minimum 70% |
| V5c2 focused integrity/regression invocation | 216 passed, zero skipped |
| Configured mypy Linux / Win32 | PASS / PASS, 412 source files each |
| Exact formatting hook `python -m pre_commit run ruff-format --all-files` | PASS; repeated after final edits |
| Exact workflow Ruff/explicit deployment format scopes | PASS |
| Full pre-commit / Betterleaks 1.9.0 | PASS: all hooks; full baselined offline working-tree scan has zero findings |
| Semgrep 1.179.0 custom blocking rules / rule fixtures | PASS: zero findings / 5 of 5 tests |
| Maintained Semgrep community baseline, advisory | 25 existing findings, four rule timeouts, one existing C partial-parse diagnostic; zero findings in V5c3 files |
| Qualified-VM integration and exact interpreter execution | NOT_TESTED / EXECUTION_BLOCKED |
| Native Windows, Ubuntu, Quality and CodeQL on final new HEAD | PENDING until the linked GitHub checks complete |

All 30 local skips require a preprovisioned `gpgv` for V4b/V4c cryptographic
integration tests. This environment lacks it; those tests were collected but
**NOT_TESTED**, not passed. No tool was acquired or installed to bypass that
prerequisite. Existing limits, coverage threshold (70%), timeouts, marker
exclusions and qualification gates are unchanged. Standard CI remains required;
the local scope does not replace the entire repository/OS test matrix.

The unbounded initial test-extra resolution selected newer numerical packages
whose stub syntax mypy 1.10 could not parse. The local validation environment
was aligned with the existing desktop lock (`numpy==1.26.4`, `pandas==2.2.2`,
`scipy==1.17.1`); `pip check` passed. No dependency lock or project constraint
was changed. Betterleaks is run without provider validation or added exceptions;
generated Python caches are removed before its working-tree scan.

Codebase Memory MCP/CLI was unavailable. Relevant source and CI configuration
were read directly, and the explicit quality-inventory contract was tested.

## P2: collector-origin classification repair

Review of HEAD `d60afdaf52071d1b96b552101d548f215af93ad3` found that changing
the synthetic collector's self-declared `origin` to `CONTROLLER_OBSERVED`
selected a real local-observation status. Four new regression cases failed on
that HEAD: two collector names, an origin change during a synthetic run, and
export of a manually declared local-observation result.

The API now refuses every controller-observation claim before invoking any
collector callback, with `EXECUTION_BLOCKED` / `COLLECTOR_IDENTITY_UNVERIFIED`.
Successful mechanism tests always produce `SYNTHETIC_MECHANISM_MATCH` and retain
the originally admitted synthetic label. The result and export have no admitted
real-observation status; `exact_target_test` stays `NOT_TESTED`. This is an
unconditional block until collector identification is separately approved and
implemented. No name allowlist, process declaration, nonce or report checksum
provides an override. Existing source, runtime, isolation and measured-output
checks and all qualification barriers remain unchanged.

After the repair, all four new cases pass and all 201 V5c3 cases pass within
the 3021-pass Observer regression run above. The 30 V4 cryptographic integration
skips still require preprovisioned `gpgv`; none is counted as PASS. Configured
mypy Linux/Win32, exact workflow Ruff/format scopes, the mandatory all-files
format hook, full pre-commit, offline Betterleaks and blocking Semgrep/rule
fixtures are required again for this update. Hosted checks on its new HEAD
remain PENDING until their own runs complete. No exact interpreter or VM run
was performed.

## Windows CI regression repair

All three baseline native Windows jobs failed the same original integrity test:

- [PR branch CI](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398560).
- [PR #3275 CI](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460555).
- [Merged main CI](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534621).

Decoded GitHub logs independently confirm **1 failed, 26320 passed, 4150
skipped, 1559 deselected** in each job. The failing test is
`test_reference_provenance_and_material_identities_are_pinned`; the observed
digest is exactly the CRLF conversion digest below. This is a real V5c2
checkout regression, separate from platform qualification refusals. The
[completed baseline CI report](evidence/stage9_observer_pyc_execution_v5c3_preflight.md)
supersedes the initial 09:46 UTC inspection, when these Windows jobs were
pending. Ubuntu, Quality, both CodeQL languages, Style and Windows Build passed
on the baseline; existing platform qualification refusals remain failures.

The sole `.gitattributes` addition is the requested file-specific `text eol=lf`
rule. `git check-attr text eol` reports `text: set`, `eol: lf`.

| Measurement | SHA-256 |
| --- | --- |
| Original Git LF blob and current checkout | `3b3686ea5c9187d5ee36ff4f69db159775b5e9f4462eecb95d1195d280e7c488` |
| Reproduced baseline checkout with `core.autocrlf=true` | `65512132c96579b8df11f8877f62fc1db3a9073fc205cc6dfb89907299c54d06` |
| Checkout with the explicit LF rule and `core.autocrlf=true` | `3b3686ea5c9187d5ee36ff4f69db159775b5e9f4462eecb95d1195d280e7c488` |

The [fixture audit](evidence/stage9_observer_pyc_execution_v5c3_windows_lf.md)
enumerated all 33 tracked fixture paths, identified 15 relevant byte-sensitive
files and performed actual Git checkout replay in disposable Linux repos.
Before the fix only the C reference changed; after the fix all 15 equal their
Git blobs. V5c2 JSON, all nine V4b signed/keyring pins and the exact review
patches already have adequate rules. No additional rule was justified. The C
source, expected SHA-256, golden vectors, SipHash algorithm and original
integrity tests are unchanged; no mass renormalization or verifier byte
normalization was performed. This checkout replay is not native Windows CI.

```text
Stage 9 = IN_PROGRESS
Observer 3A = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
External authority = UNVERIFIED
Authenticated time = UNVERIFIED
Durable anti-rollback = BLOCKED
Runtime qualification = BLOCKED
```
